# SPDX-License-Identifier: GPL-3.0-or-later
"""
ModelBackend implementation for Qwen-Image-2.1: a direct diffusers
pipeline (no ComfyUI), in-process on a ZeroGPU worker or any CUDA GPU.

All of the actual pipeline loading, AOTI-kernel loading, and the
@spaces.GPU-wrapped diffusion call live in the separately licensed
ai_image_edit_qwen package (its pipeline module). This module plays
the same role for this model that qwen_image_edit_comfy/model.py plays
for its own: capability declaration, dimension resolution, and
orchestrating one generate() call — nothing here touches CUDA directly.

Masking is handled externally, via core/imaging.py's
run_masked_generation() — the same crop/composite/color-correct sequence
qwen_image_edit_comfy uses. QwenImage21Pipeline itself has no mask_image
parameter or other inpainting-specific mechanism (confirmed against its
actual __call__ signature), so masking can only be done externally.
"""
import math
import os
import random
import uuid
from typing import Dict, Optional, Tuple

from ai_image_edit_qwen import pipeline
from PIL import Image

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.models.base import ModelBackend

MAX_INPUT_IMAGES = 10
MAX_SEED = 2 ** 31 - 1

DEFAULT_AOTI_REPO = "hugging-apps/qwen-image-2-1-aoti"

# The aspect ratios the model card lists, at its 2048-base resolution.
BASE_RESOLUTION = 2048
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# This project's sentinel for "derive dimensions from context rather than
# force a fixed ratio" is "Original" (e.g. the frontends lock the
# aspect-ratio dropdown to it while a mask is drawn). For this model it
# means the pipeline follows the input images' own aspect ratio, or squares
# up for pure text-to-image.
AUTO_ASPECT_RATIO = "Original"

# Resolution tiers, reusing the "megapixels" field every model's
# capabilities expose for its resolution choices. Here it's actually this
# model's own tier (1K/1.5K/2K, i.e. target side length 1024/1536/2048)
# rather than a literal computed pixel area. See _resolve_dimensions(),
# which turns the chosen value back into that side length before calling
# _resolve_size().
SUPPORTED_MEGAPIXELS = [1.0, 1.5, 2.0]
DEFAULT_MEGAPIXELS = 1.0

DEFAULT_STEPS, MIN_STEPS, MAX_STEPS = 40, 8, 60
DEFAULT_CFG, MIN_CFG, MAX_CFG = 1.0, 1.0, 10.0


def _round32(v: float) -> int:
    """Rounds to the nearest multiple of 32, minimum 32."""
    return max(32, int(round(v / 32)) * 32)


def _resolve_size(resolution: int, aspect_ratio: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Width/height for a resolution tier and a named aspect ratio: the model
    card's size for that ratio, scaled by resolution / BASE_RESOLUTION.
    Returns (None, None) for AUTO_ASPECT_RATIO — the pipeline then infers
    dimensions itself (from the input images, or a square default for pure
    text-to-image) rather than being told a fixed size. Only used when
    there's no mask — see _dimensions_from_source() for the masked case,
    which needs concrete numbers regardless of aspect_ratio.
    """
    if not aspect_ratio or aspect_ratio == AUTO_ASPECT_RATIO or aspect_ratio not in ASPECT_RATIOS:
        return None, None
    base_w, base_h = ASPECT_RATIOS[aspect_ratio]
    scale = resolution / BASE_RESOLUTION
    return _round32(base_w * scale), _round32(base_h * scale)


def _dimensions_from_source(source_image_path: str, resolution: int) -> Tuple[int, int]:
    """
    Derives concrete (width, height) from the source image's own aspect
    ratio at the given resolution tier. Used whenever _resolve_size left
    width/height unset (AUTO_ASPECT_RATIO, i.e. "Original") — not just while
    a mask is drawn: imaging.run_masked_generation needs real numbers up
    front to crop the source to before inference, and passing width=height=
    None straight through to the pipeline instead doesn't reliably keep the
    source's own aspect ratio either. Area scales with resolution the same
    way ASPECT_RATIOS' entries do (area ~ resolution^2), just computed
    directly from the source's own ratio instead of one of the model card's
    named presets — there's no precomputed table entry for an arbitrary
    uploaded image's ratio. A source that is already a valid size for the
    tier (multiples of 32, area within 10% of the tier's) keeps its own size.
    """
    with Image.open(source_image_path) as img:
        src_w, src_h = img.size
    target_area = float(resolution) ** 2
    if src_w % 32 == 0 and src_h % 32 == 0 and 0.9 <= src_w * src_h / target_area <= 1.1:
        return src_w, src_h
    ratio = src_w / src_h
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    return _round32(w), _round32(h)


class QwenImageModel(ModelBackend):
    """Qwen-Image-2.1, run as a direct diffusers pipeline."""

    @property
    def model_name(self) -> str:
        return "Qwen-Image"

    @property
    def model_version(self) -> str:
        return "2.1"

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
            sampler_choices=None,
            default_sampler=None,
            scheduler_choices=None,
            default_scheduler=None,
            supports_cfg=True,
            supports_denoise=False,
            supports_seed=True,
            supports_negative_prompt=True,  # only meaningful together with supports_cfg — see pipeline._diffuse
            num_annotation_colors=3,
            step_range=RangeSpec(MIN_STEPS, MAX_STEPS, DEFAULT_STEPS, step=1),
            cfg_range=RangeSpec(MIN_CFG, MAX_CFG, DEFAULT_CFG, step=0.1),
        )

    def start(self) -> None:
        aoti_repo = os.environ.get("QWEN21_AOTI_REPO", DEFAULT_AOTI_REPO)
        use_aoti = os.environ.get("QWEN21_AOTI", "1") != "0"
        pipeline.load(aoti_repo=aoti_repo, aoti_token=os.environ.get("HF_TOKEN"), use_aoti=use_aoti)

    def shutdown(self) -> None:
        # Nothing to explicitly tear down — the pipeline lives for the
        # lifetime of the process, and on ZeroGPU the platform owns the CUDA
        # allocation lifecycle around each @spaces.GPU call.
        pass

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        """The tier whose area is closest to the source image's own."""
        with Image.open(source_image_path) as img:
            area = img.width * img.height
        return min(SUPPORTED_MEGAPIXELS, key=lambda mp: abs(math.log(area / (mp * 1024) ** 2)))

    def _resolve_dimensions(self, aspect_ratio: str, target_megapixels: float) -> Tuple[int, Optional[int], Optional[int]]:
        """
        Returns (resolution_tier, width, height). resolution_tier is this
        model's own side-length tier (1024/1536/2048-ish, from
        target_megapixels — see SUPPORTED_MEGAPIXELS' comment); width/height
        are None, None for AUTO_ASPECT_RATIO.
        """
        resolution = int(round(target_megapixels * 1024))
        width, height = _resolve_size(resolution, aspect_ratio)
        return resolution, width, height

    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Runs one generation. Every image handed to the pipeline — the
        (possibly mask-cropped) source and any reference images — is a
        whole reference/condition image; the prompt describes the edit in
        words (model card convention: <image1>…<image10> in upload order).
        Masking is external, via imaging.run_masked_generation; this
        model's own contribution is _infer(), calling the diffusers
        pipeline for a given (possibly cropped) source path.

        params.source_image_path is always non-empty today, since both
        frontends require an image upload before Generate is enabled — a
        UI limitation, not one of this backend, which does support
        text-to-image.
        """
        prompt = (params.prompt or "").strip()
        if not prompt:
            raise GenerationError("Please enter a prompt.")

        if 1 + len(params.reference_images) > MAX_INPUT_IMAGES:
            raise GenerationError(f"Up to {MAX_INPUT_IMAGES} input images are supported.")

        resolved_seed = random.randint(0, MAX_SEED) if params.randomize_seed else int(params.seed) % (MAX_SEED + 1)

        # A mask's coordinates are only meaningful relative to the source
        # image's own framing, so dimensions must come from the source
        # itself whenever a mask is present — mirrors
        # qwen_image_edit_comfy/model.py's identical override (see the
        # comment in that file's generate()). Frontends
        # already lock the Aspect ratio control to "Original" while a mask
        # is drawn, but this doesn't rely solely on that: enforcing it here
        # too means a mask always gets correctly-aligned dimensions even if
        # that UI-level invariant is ever bypassed.
        # Likewise the tier: with a mask the output keeps the source's own
        # resolution, so the compositing doesn't resample the whole image.
        effective_aspect_ratio = AUTO_ASPECT_RATIO if params.mask_path else params.aspect_ratio
        target_megapixels = self.megapixels_for_source(params.source_image_path) if params.mask_path else params.target_megapixels
        resolution, gen_width, gen_height = self._resolve_dimensions(effective_aspect_ratio, target_megapixels)

        if gen_width is None or gen_height is None:
            # _resolve_size returns (None, None) for AUTO_ASPECT_RATIO
            # ("Original"), e.g. whenever a mask is being drawn — derive
            # concrete dimensions from the source image instead.
            gen_width, gen_height = _dimensions_from_source(params.source_image_path, resolution)

        def _infer(model_input_path: str) -> str:
            image_paths = [p for p in [model_input_path, *params.reference_images] if p]
            print(
                f"[qwen_image] images={len(image_paths)} resolution_tier={resolution} "
                f"size={(gen_width, gen_height)} steps={params.steps} cfg={params.cfg}",
                flush=True,
            )
            image = pipeline.generate(
                prompt=prompt,
                image_paths=image_paths,
                negative_prompt=params.negative_prompt,
                true_cfg_scale=params.cfg,
                num_inference_steps=params.steps,
                seed=resolved_seed,
                resolution=resolution,
                width=gen_width,
                height=gen_height,
            )
            out_path = WORK_DIR / f"qwen_image_{uuid.uuid4().hex}.png"
            image.save(out_path)
            return str(out_path)

        final_output_path = imaging.run_masked_generation(
            params.source_image_path,
            params.mask_path,
            gen_width,
            gen_height,
            params.feather_amount,
            params.apply_color_correction_enabled,
            _infer,
            params.annotated_image_path,
        )

        return GenerationResult(
            before_path=params.source_image_path,
            after_path=final_output_path,
            actual_seed=resolved_seed,
        )
