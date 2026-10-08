# SPDX-License-Identifier: GPL-3.0-or-later
"""
ModelBackend for Qwen-Image-2.1 with quantized GGUF weights, served through a
local ComfyUI instance with the ComfyUI-GGUF custom node
(https://github.com/leejet/ComfyUI-GGUF). The quantized weights fit on
consumer GPUs, unlike the full-precision qwen_image21 backend.

The graph is the official ComfyUI Qwen-Image-2.1 image-edit template with the
diffusion model loaded through UnetLoaderGGUF (see workflow_api.json). The
weights come from three environment variables set in the Dockerfile, each a
path relative to its folder under ComfyUI's models/:
MODEL_FILE (diffusion_models), TEXT_ENCODER_FILE (text_encoders), VAE_FILE (vae).

ComfyUI wire-protocol details live in models/_shared/comfyui/client.py; output sizes
are shared with the qwen_image21 backend (models/_shared/qwen21_size.py); crop/composite is
core/imaging.py.
"""
import json
import os
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.result_cache import cached_infer
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models._shared.comfyui import client as comfy_client
from ai_image_edit.models._shared.comfyui.common import MODEL_FILE_ENV, SAMPLER_CHOICES, SCHEDULER_CHOICES, configured_file
from ai_image_edit.models._shared.qwen21_size import (
    ASPECT_RATIOS,
    AUTO_ASPECT_RATIO,
    DEFAULT_CFG,
    DEFAULT_MEGAPIXELS,
    DEFAULT_STEPS,
    MAX_CFG,
    MAX_STEPS,
    MIN_CFG,
    MIN_STEPS,
    SUPPORTED_MEGAPIXELS,
    dimensions_from_source,
    megapixels_for_source as source_megapixels,
    resolution_for,
    resolve_size,
)

TEXT_ENCODER_FILE_ENV = "TEXT_ENCODER_FILE"
VAE_FILE_ENV = "VAE_FILE"
DIFFUSION_MODELS_DIR = Path("models/diffusion_models")
TEXT_ENCODERS_DIR = Path("models/text_encoders")
VAE_DIR = Path("models/vae")

# Source image plus references: the encode node takes image_1 .. image_10.
MAX_INPUT_IMAGES = 10
MAX_SEED = 2 ** 31 - 1

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# Node ids in workflow_api.json that the code below fills in.
UNET_NODE, CLIP_NODE, VAE_NODE, ENCODE_NODE, SAMPLER_NODE, SOURCE_IMAGE_NODE = "1", "2", "3", "5", "6", "10"
# Reference image nodes are added per generation, with ids from here on.
FIRST_REFERENCE_NODE_ID = 20

WORKFLOW_PATH = Path(__file__).parent / "workflow_api.json"

# Nodes the workflow needs beyond ComfyUI's built-ins, checked at startup.
REQUIRED_NODES = ["UnetLoaderGGUF", "QwenImage21Cache", "TextEncodeQwenImage21"]


class QwenImage21GGUFModel(ModelBackend):
    """Qwen-Image-2.1 (quantized GGUF), run through a local ComfyUI server."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._nodes_checked = False

    @property
    def model_name(self) -> str:
        return "Qwen-Image 2.1 GGUF"

    @property
    def model_version(self) -> Optional[str]:
        """The quantization of the GGUF file, e.g. "Q4_K_M" (and "UC" for the uncensored variant)."""
        name = Path(os.environ.get(MODEL_FILE_ENV, "")).name
        quant = re.search(r"(?<![A-Za-z0-9])(Q\d(?:_[A-Z0-9]+)*|BF16|F16)(?=\.gguf$)", name, re.IGNORECASE)
        if not quant:
            return None
        uncensored = re.search(r"(?<![A-Za-z0-9])UC(?![A-Za-z0-9])", name)
        return f"{'UC ' if uncensored else ''}{quant.group(1).upper()}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_reference_images=MAX_INPUT_IMAGES,
            supports_loras=False,
            supports_inpainting=True,
            supported_aspect_ratios=[AUTO_ASPECT_RATIO] + list(ASPECT_RATIOS.keys()),
            default_aspect_ratio=AUTO_ASPECT_RATIO,
            supported_megapixels=SUPPORTED_MEGAPIXELS,
            default_megapixels=DEFAULT_MEGAPIXELS,
            sampler_choices=SAMPLER_CHOICES,
            default_sampler=DEFAULT_SAMPLER,
            scheduler_choices=SCHEDULER_CHOICES,
            default_scheduler=DEFAULT_SCHEDULER,
            supports_cfg=True,
            supports_denoise=False,
            supports_seed=True,
            supports_negative_prompt=True,
            num_annotation_colors=3,
            step_range=RangeSpec(MIN_STEPS, MAX_STEPS, DEFAULT_STEPS, step=1),
            cfg_range=RangeSpec(MIN_CFG, MAX_CFG, DEFAULT_CFG, step=0.1),
        )

    def start(self) -> None:
        # Checked before ComfyUI starts, so a missing setting or file fails fast.
        configured_file(MODEL_FILE_ENV, DIFFUSION_MODELS_DIR)
        configured_file(TEXT_ENCODER_FILE_ENV, TEXT_ENCODERS_DIR)
        configured_file(VAE_FILE_ENV, VAE_DIR)
        self._process = comfy_client.launch_comfy_process()

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.terminate()

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        """The tier whose area is closest to the source image's own."""
        return source_megapixels(source_image_path)

    def _check_nodes(self) -> None:
        """
        Fails with a clear message if ComfyUI lacks a node the workflow needs
        (ComfyUI-GGUF not installed, or ComfyUI too old for Qwen-Image 2.1).
        Done on the first generation rather than in start(), because ComfyUI
        is still booting then; while it is, this raises the usual "still
        starting" error and is retried on the next generation.
        """
        if self._nodes_checked:
            return
        missing = comfy_client.missing_nodes(REQUIRED_NODES)
        if missing:
            raise GenerationError(
                f"ComfyUI is missing node(s) {', '.join(missing)}: install the ComfyUI-GGUF custom node "
                "and use a ComfyUI version that supports Qwen-Image 2.1."
            )
        self._nodes_checked = True

    def _build_workflow(self, params: GenerationParams, source_path: str, seed: int, resolution: int) -> dict:
        """Loads workflow_api.json and fills in the fields for one generation."""
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        workflow[UNET_NODE]["inputs"]["unet_name"] = configured_file(MODEL_FILE_ENV, DIFFUSION_MODELS_DIR)
        workflow[CLIP_NODE]["inputs"]["clip_name"] = configured_file(TEXT_ENCODER_FILE_ENV, TEXT_ENCODERS_DIR)
        workflow[VAE_NODE]["inputs"]["vae_name"] = configured_file(VAE_FILE_ENV, VAE_DIR)

        workflow[SOURCE_IMAGE_NODE]["inputs"]["image"] = comfy_client.upload_image(source_path)

        # Each reference image is a LoadImage node wired to the next free input of the encode node.
        for i, ref in enumerate(params.reference_images):
            node_id = str(FIRST_REFERENCE_NODE_ID + i)
            workflow[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": comfy_client.upload_image(ref), "upload": "image"},
            }
            workflow[ENCODE_NODE]["inputs"][f"images.image_{i + 2}"] = [node_id, 0]

        encode = workflow[ENCODE_NODE]["inputs"]
        encode["prompt"] = params.prompt.strip()
        encode["negative_prompt"] = params.negative_prompt or ""
        encode["resolution"] = resolution

        sampler = workflow[SAMPLER_NODE]["inputs"]
        sampler["seed"] = seed
        sampler["steps"] = int(params.steps)
        sampler["cfg"] = float(params.cfg)
        sampler["sampler_name"] = params.sampler_name
        sampler["scheduler"] = params.scheduler
        return workflow

    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Resolves target dimensions, then hands off to
        imaging.run_masked_generation for the crop -> infer -> composite
        sequence; _infer() builds and submits the ComfyUI workflow and
        fetches the result. The source is cropped/resized to the generation
        size first, and the encode node makes the output follow it, so the
        output matches. Reference images are passed as they are.
        """
        prompt = (params.prompt or "").strip()
        if not prompt:
            raise GenerationError("Please enter a prompt.")
        if 1 + len(params.reference_images) > MAX_INPUT_IMAGES:
            raise GenerationError(f"Up to {MAX_INPUT_IMAGES} input images are supported.")

        seed = random.randint(0, MAX_SEED) if params.randomize_seed else int(params.seed) % (MAX_SEED + 1)

        # As in the other backends: a mask only lines up with the source's
        # own framing and resolution, so both come from the source then.
        effective_aspect_ratio = AUTO_ASPECT_RATIO if params.mask_path else params.aspect_ratio
        target_megapixels = self.megapixels_for_source(params.source_image_path) if params.mask_path else params.target_megapixels
        resolution = resolution_for(target_megapixels)
        gen_width, gen_height = resolve_size(resolution, effective_aspect_ratio)
        if gen_width is None or gen_height is None:
            gen_width, gen_height = dimensions_from_source(params.source_image_path, resolution)
        print(
            f"[qwen_image21_gguf] mask={'yes' if params.mask_path else 'no'} images={1 + len(params.reference_images)} "
            f"resolution_tier={resolution} gen_dims={gen_width}x{gen_height} steps={params.steps} cfg={params.cfg}",
            flush=True,
        )

        def _infer(model_input_path: str) -> str:
            self._check_nodes()
            workflow = self._build_workflow(params, model_input_path, seed, resolution)
            t0 = time.time()
            prompt_id = comfy_client.submit_workflow_and_wait(workflow, str(uuid.uuid4()))
            print(f"[qwen_image21_gguf] ComfyUI generation took {time.time() - t0:.1f}s", flush=True)
            return comfy_client.fetch_generated_image(prompt_id)

        infer = cached_infer(_infer, self.display_name, params, seed=seed, resolution=resolution, width=gen_width, height=gen_height)

        final_output_path = imaging.run_masked_generation(
            params.source_image_path,
            params.mask_path,
            gen_width,
            gen_height,
            params.feather_amount,
            params.apply_color_correction_enabled,
            infer,
            params.annotated_image_path,
        )
        return GenerationResult(before_path=params.source_image_path, after_path=final_output_path, actual_seed=seed)
