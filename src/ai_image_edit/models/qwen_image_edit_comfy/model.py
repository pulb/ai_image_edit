# SPDX-License-Identifier: GPL-3.0-or-later
"""
ModelBackend implementation for Phr00t's Qwen-Image-Edit-Rapid-AIO,
served through a local ComfyUI instance:
https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO

All ComfyUI wire-protocol details live in comfy_client.py; pixel-level
color-matching/compositing lives in core/imaging.py, shared with every
other model that does external crop/mask/composite around inference. This
module owns workflow construction (building the ComfyUI graph for one
generation) and the top-level orchestration (generate()) that ties
everything together.
"""
import json
import math
import os
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models.qwen_image_edit_comfy import comfy_client

LORA_DIR = "models/loras"

MIN_STEPS, MAX_STEPS, DEFAULT_STEPS = 1, 10, 4
MIN_CFG, MAX_CFG, DEFAULT_CFG = 0.1, 10.0, 1.0
MIN_DENOISE, MAX_DENOISE, DEFAULT_DENOISE = 0.0, 1.0, 1.0
MIN_LORA_STRENGTH, MAX_LORA_STRENGTH, DEFAULT_LORA_STRENGTH = 0.1, 2.0, 0.7

DEFAULT_SAMPLER = "sa_solver"
DEFAULT_SCHEDULER = "beta"
DEFAULT_ASPECT_RATIO = "Original"

# Valid ComfyUI KSampler scheduler names (from comfy/samplers.py's SCHEDULER_HANDLERS).
SCHEDULER_CHOICES = ["normal", "karras", "exponential", "simple", "ddim_uniform", "beta", "sgm_uniform", "linear_quadratic", "kl_optimal"]

# Valid ComfyUI KSampler sampler names (comfy/samplers.py's KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]).
SAMPLER_CHOICES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2"
]

# Fixed negative prompt applied to every generation (node 4 in workflow_api.json).
NEGATIVE_PROMPT = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

# workflow_api.json lives next to this file (shipped as package data).
WORKFLOW_PATH = Path(__file__).parent / "workflow_api.json"

# Aspect ratio presets: label -> (w_ratio, h_ratio) fed into the target-area
# formula below. "Original" is a sentinel meaning "derive dimensions from
# the source image's own aspect ratio" rather than a fixed ratio.
_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "21:9": (21, 9),
    "9:21": (9, 21),
    "Original": (0, 0),
}

# Output resolution choices offered to the UI, in megapixels. The model
# (this file) is the only place that turns a chosen value into actual pixel
# dimensions — see _dimensions_for(). Phr00t's Qwen-Image-Edit-Rapid-AIO
# checkpoint is tuned for ~1MP outputs and doesn't hold up well above that,
# so only 1.0 is offered.
SUPPORTED_MEGAPIXELS = [1.0]
DEFAULT_MEGAPIXELS = 1.0


def _dimensions_for(aspect_w: int, aspect_h: int, target_area: float, multiple: int = 8) -> Tuple[int, int]:
    """
    Returns (width, height) for aspect_w:aspect_h that fits target_area
    pixels, rounded to a multiple of `multiple` (default 8, matching
    TextEncodeQwenImageEditPlus's own VAE-encoding grid — this avoids that
    node doing any internal rescaling). aspect_w/aspect_h must be nonzero;
    "Original" is resolved by the caller before reaching this function.
    """
    ratio = aspect_w / aspect_h

    # Formula: width = sqrt(area * ratio)
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    final_w = int(round(width / multiple) * multiple)
    final_h = int(round(height / multiple) * multiple)

    return (final_w, final_h)


class QwenImageEditComfyModel(ModelBackend):
    """Qwen-Image-Edit-Rapid-AIO, run through a local ComfyUI server."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None

    @property
    def model_name(self) -> str:
        return "Qwen-Image-Edit Rapid-AIO"

    @property
    def model_version(self) -> Optional[str]:
        """The "vNN" part of the workflow's checkpoint filename, e.g. "v23"."""
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            ckpt_name = json.load(f)["1"]["inputs"]["ckpt_name"]
        match = re.search(r"-(v\d+)(?=\.|-|_|$)", ckpt_name)
        return match.group(1) if match else None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_reference_images=3,  # primary source + image2 + image3
            supports_loras=True,
            supports_inpainting=True,
            supported_aspect_ratios=list(_ASPECT_RATIOS.keys()),
            default_aspect_ratio=DEFAULT_ASPECT_RATIO,
            supported_megapixels=SUPPORTED_MEGAPIXELS,
            default_megapixels=DEFAULT_MEGAPIXELS,
            sampler_choices=SAMPLER_CHOICES,
            default_sampler=DEFAULT_SAMPLER,
            scheduler_choices=SCHEDULER_CHOICES,
            default_scheduler=DEFAULT_SCHEDULER,
            supports_cfg=True,
            supports_denoise=True,
            supports_seed=True,
            step_range=RangeSpec(MIN_STEPS, MAX_STEPS, DEFAULT_STEPS, step=1),
            cfg_range=RangeSpec(MIN_CFG, MAX_CFG, DEFAULT_CFG, step=0.1),
            denoise_range=RangeSpec(MIN_DENOISE, MAX_DENOISE, DEFAULT_DENOISE, step=0.01),
            lora_strength_range=RangeSpec(MIN_LORA_STRENGTH, MAX_LORA_STRENGTH, DEFAULT_LORA_STRENGTH, step=0.05),
        )

    def start(self) -> None:
        self._process = comfy_client.launch_comfy_process()

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.terminate()

    def list_loras(self) -> Dict[str, List[str]]:
        """
        Lists available LoRAs in LORA_DIR, keyed by display name, each mapped to
        the list of weight file(s) — as paths relative to LORA_DIR, exactly what
        ComfyUI's LoraLoader node expects for its lora_name input — that belong
        to it. A LoRA is either a single .safetensors/.pt file directly in
        LORA_DIR (display name = filename), or a subdirectory containing one or
        more .safetensors/.pt files (display name = subdirectory name) — some
        LoRAs are distributed as multiple files (e.g. separate UNet/CLIP
        weights) that all need loading together.
        """
        if not os.path.exists(LORA_DIR):
            print(f"[WARNING] LoRA folder not found at expected path: {LORA_DIR}", flush=True)
            return {}

        loras: Dict[str, List[str]] = {}

        for entry in sorted(os.listdir(LORA_DIR)):
            entry_path = os.path.join(LORA_DIR, entry)
            if os.path.isdir(entry_path):
                files = sorted(
                    os.path.join(entry, f)
                    for f in os.listdir(entry_path)
                    if f.endswith((".safetensors", ".pt"))
                )
                if files:
                    loras[entry] = files
            elif entry.endswith((".safetensors", ".pt")):
                loras[entry] = [entry]

        if not loras:
            print(f"[WARNING] No LoRA files found in {LORA_DIR}", flush=True)

        return loras

    def _resolve_generation_dimensions(self, source_image_path: str, aspect_ratio: str, target_megapixels: float) -> Tuple[int, int]:
        """
        Determines the (gen_width, gen_height) to generate at.

        "Original" derives dimensions from the source image's own aspect
        ratio; any other choice uses the matching fixed ratio from
        _ASPECT_RATIOS. Either way, target_megapixels (chosen by the user
        from capabilities.supported_megapixels) sets the target pixel area.
        """
        target_area = target_megapixels * 1024 * 1024

        if aspect_ratio == "Original":
            with Image.open(source_image_path) as img:
                orig_w, orig_h = img.size
            return _dimensions_for(orig_w, orig_h, target_area)

        ratio_w, ratio_h = _ASPECT_RATIOS.get(aspect_ratio, _ASPECT_RATIOS["1:1"])
        return _dimensions_for(ratio_w, ratio_h, target_area)

    def _build_workflow(
        self, prompt: str, source_image_path: str, image2: Optional[str], image3: Optional[str],
        seed_value: int, steps: int, cfg: float, denoise: float, sampler_name: str, scheduler: str,
        lora_files: List[str], lora_strength: float, gen_width: int, gen_height: int,
    ) -> dict:
        """Loads workflow_api.json and fills in the Qwen-Image-Edit-specific fields for one generation."""
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        current_path = workflow["1"]["inputs"]["ckpt_name"]
        workflow["1"]["inputs"]["ckpt_name"] = current_path.replace("\\", "/")

        workflow["7"]["inputs"]["image"] = comfy_client.upload_image(source_image_path)

        if image2:
            workflow["8"]["inputs"]["image"] = comfy_client.upload_image(image2)
        else:
            if "8" in workflow:
                del workflow["8"]
            if "image2" in workflow["3"]["inputs"]:
                del workflow["3"]["inputs"]["image2"]

        if image3:
            workflow["10"] = {"inputs": {"image": comfy_client.upload_image(image3), "upload": "image"}, "class_type": "LoadImage"}
            workflow["3"]["inputs"]["image3"] = ["10", 0]
        else:
            if "10" in workflow:
                del workflow["10"]
            if "image3" in workflow["3"]["inputs"]:
                del workflow["3"]["inputs"]["image3"]

        workflow["2"]["inputs"]["seed"] = seed_value
        workflow["2"]["inputs"]["steps"] = int(steps)
        workflow["2"]["inputs"]["cfg"] = float(cfg)
        workflow["2"]["inputs"]["denoise"] = float(denoise)
        workflow["2"]["inputs"]["sampler_name"] = sampler_name
        workflow["2"]["inputs"]["scheduler"] = scheduler
        workflow["3"]["inputs"]["prompt"] = prompt
        workflow["4"]["inputs"]["prompt"] = NEGATIVE_PROMPT

        workflow["9"]["inputs"]["width"] = gen_width
        workflow["9"]["inputs"]["height"] = gen_height

        if lora_files:
            # Splice a LoraLoader per file between the checkpoint and everything
            # that currently consumes its model/clip outputs, chaining them so
            # each subsequent LoraLoader's model/clip inputs come from the
            # previous one's outputs, and redirecting the final references to
            # the last LoraLoader in the chain. VAE is untouched — LoraLoader
            # only modifies MODEL and CLIP.
            model_link = ["1", 0]
            clip_link = ["1", 1]
            for i, lora_file in enumerate(lora_files):
                node_id = str(100 + i)
                workflow[node_id] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": model_link,
                        "clip": clip_link,
                        "lora_name": lora_file,
                        "strength_model": float(lora_strength),
                        "strength_clip": float(lora_strength),
                    }
                }
                model_link = [node_id, 0]
                clip_link = [node_id, 1]
            workflow["2"]["inputs"]["model"] = model_link
            workflow["3"]["inputs"]["clip"] = clip_link
            workflow["4"]["inputs"]["clip"] = clip_link

        return workflow

    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Resolves target dimensions, then hands off to
        imaging.run_masked_generation for the crop -> infer -> composite
        sequence. This model's own contribution is _infer(): build/submit
        the Qwen-Image-Edit ComfyUI workflow and fetch the result. Extra
        reference images are never cropped/resized — only the primary
        source needs to align with the mask's coordinate space.
        """
        prompt = (params.prompt or "").strip()
        if not prompt:
            raise GenerationError("Please enter a prompt.")

        actual_seed = random.randint(1, 4294967295) if params.randomize_seed else int(params.seed)

        # A mask's coordinates are only meaningful relative to the source
        # image's own framing, so dimensions must come from the source
        # itself whenever a mask is present — the frontends already
        # force aspect_ratio to "Original" while a mask is drawn, but this
        # doesn't rely solely on that: reusing the "Original" branch here
        # too (rather than whatever aspect_ratio string was actually
        # passed) means a mask always gets correctly-aligned dimensions
        # even if that UI-level invariant is ever bypassed.
        effective_aspect_ratio = "Original" if params.mask_path else params.aspect_ratio
        gen_width, gen_height = self._resolve_generation_dimensions(
            params.source_image_path, effective_aspect_ratio, params.target_megapixels
        )
        print(
            f"[qwen_image_edit_comfy] mask={'yes' if params.mask_path else 'no'} gen_dims={gen_width}x{gen_height} "
            f"mp={params.target_megapixels} steps={params.steps} sampler={params.sampler_name} scheduler={params.scheduler}",
            flush=True,
        )

        image2 = params.reference_images[0] if len(params.reference_images) > 0 else None
        image3 = params.reference_images[1] if len(params.reference_images) > 1 else None

        def _infer(model_input_path: str) -> str:
            client_id = str(uuid.uuid4())
            workflow = self._build_workflow(
                prompt, model_input_path, image2, image3, actual_seed,
                params.steps, params.cfg, params.denoise, params.sampler_name, params.scheduler,
                params.lora_files, params.lora_strength, gen_width, gen_height,
            )

            t0 = time.time()
            prompt_id = comfy_client.submit_workflow_and_wait(workflow, client_id)
            print(f"[qwen_image_edit_comfy] ComfyUI generation took {time.time() - t0:.1f}s", flush=True)

            t1 = time.time()
            output_path = comfy_client.fetch_generated_image(prompt_id)
            print(f"[qwen_image_edit_comfy] fetch_generated_image took {time.time() - t1:.1f}s", flush=True)
            return output_path

        final_output_path = imaging.run_masked_generation(
            params.source_image_path,
            params.mask_path,
            gen_width,
            gen_height,
            params.feather_amount,
            params.apply_color_correction_enabled,
            _infer,
        )

        return GenerationResult(before_path=params.source_image_path, after_path=final_output_path, actual_seed=actual_seed)
