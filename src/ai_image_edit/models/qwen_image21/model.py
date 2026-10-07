# SPDX-License-Identifier: GPL-3.0-or-later
"""
ModelBackend implementation for Qwen-Image-2.1: a direct diffusers
pipeline (no ComfyUI), in-process on a ZeroGPU worker or any CUDA GPU.

All of the actual pipeline loading, AOTI-kernel loading, and the
@spaces.GPU-wrapped diffusion call live in the separately licensed
ai_image_edit_qwen package (its pipeline module). This module plays
the same role for this model that qwen_image_edit_2511_aio/model.py plays
for its own: capability declaration, dimension resolution, and
orchestrating one generate() call — nothing here touches CUDA directly.

Masking is handled externally, via core/imaging.py's
run_masked_generation() — the same crop/composite/color-correct sequence
qwen_image_edit_2511_aio uses. QwenImage21Pipeline itself has no mask_image
parameter or other inpainting-specific mechanism (confirmed against its
actual __call__ signature), so masking can only be done externally.
"""
import os
import random
import uuid
from typing import Optional, Tuple

from ai_image_edit_qwen import pipeline

from ai_image_edit.core import imaging
from ai_image_edit.core.result_cache import cached_infer
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models.qwen21_size import (
    ASPECT_RATIOS,
    AUTO_ASPECT_RATIO,
    DEFAULT_MEGAPIXELS,
    SUPPORTED_MEGAPIXELS,
    dimensions_from_source,
    megapixels_for_source as source_megapixels,
    resolution_for,
    resolve_size,
)

MAX_INPUT_IMAGES = 10
MAX_SEED = 2 ** 31 - 1

DEFAULT_AOTI_REPO = "hugging-apps/qwen-image-2-1-aoti"

class QwenImage21Model(ModelBackend):
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
        return source_megapixels(source_image_path)

    def _resolve_dimensions(self, aspect_ratio: str, target_megapixels: float) -> Tuple[int, Optional[int], Optional[int]]:
        """
        Returns (resolution_tier, width, height). resolution_tier is this
        model's own side-length tier (1024/1536/2048-ish, from
        target_megapixels — see qwen21_size.SUPPORTED_MEGAPIXELS' comment); width/height
        are None, None for AUTO_ASPECT_RATIO.
        """
        resolution = resolution_for(target_megapixels)
        width, height = resolve_size(resolution, aspect_ratio)
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
        # qwen_image_edit_2511_aio/model.py's identical override (see the
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
            # resolve_size returns (None, None) for AUTO_ASPECT_RATIO
            # ("Original"), e.g. whenever a mask is being drawn — derive
            # concrete dimensions from the source image instead.
            gen_width, gen_height = dimensions_from_source(params.source_image_path, resolution)

        def _infer(model_input_path: str) -> str:
            image_paths = [p for p in [model_input_path, *params.reference_images] if p]
            print(
                f"[qwen_image21] images={len(image_paths)} resolution_tier={resolution} "
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

        infer = cached_infer(
            _infer, self.display_name, params,
            seed=resolved_seed, resolution=resolution, width=gen_width, height=gen_height,
        )

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

        return GenerationResult(
            before_path=params.source_image_path,
            after_path=final_output_path,
            actual_seed=resolved_seed,
        )
