# SPDX-License-Identifier: GPL-3.0-or-later
"""Constants and helpers shared by the frontends."""
from typing import List, Optional, Sequence

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import GenerationParams, ModelCapabilities

# This app's one accent colour: the frontend reads its purple from here.
PRIMARY_COLOR = "#7c3aed"

# Newest work-dir files trim_work_dir() never deletes: the files of the
# generation that just finished (source, mask, annotated, result, ...).
RECENT_FILES_KEPT = 8

HOST = "0.0.0.0"
PORT = 7860

def default_choice(default: Optional[str], choices: Optional[Sequence[str]]) -> Optional[str]:
    """The model's declared default if it is one of `choices`, else the first choice (None if there are none)."""
    if not choices:
        return None
    return default if default in choices else choices[0]


def describe_error(exc: Exception) -> str:
    """The message to show the user for an exception raised by generate()."""
    return str(exc) if isinstance(exc, GenerationError) else f"Unexpected error: {exc}"


def params_from_ui(
    caps: ModelCapabilities,
    *,
    prompt: Optional[str],
    source_image_path: str,
    mask_path: Optional[str],
    annotated_image_path: Optional[str] = None,
    reference_images: List[str],
    seed: Optional[float],
    randomize_seed: bool,
    aspect_ratio: str,
    target_megapixels: float,
    steps: int,
    cfg: Optional[float] = None,
    denoise: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    lora_files: List[str],
    lora_strength: Optional[float] = None,
    apply_color_correction: bool,
    feather_amount: Optional[int] = None,
) -> GenerationParams:
    """
    Builds the GenerationParams for one generation. A control the model
    doesn't support may be passed as None (or hold any value): the
    capability decides, and the field gets a neutral value instead.

    cfg and lora_strength fall back to the capability's own declared
    default (not a bare 0.0) when unsupported: every model's cfg_range and
    lora_strength_range starts above 0, and no model currently
    guards its use of params.cfg/params.lora_strength behind
    supports_cfg/supports_loras, so a bare 0.0 would silently fall outside
    a future such model's valid range.
    """
    return GenerationParams(
        prompt=prompt or "",
        source_image_path=source_image_path,
        mask_path=mask_path if caps.supports_inpainting else None,
        steps=steps,
        reference_images=reference_images,
        annotated_image_path=annotated_image_path,
        seed=int(seed) if seed is not None else 0,
        randomize_seed=bool(randomize_seed),
        aspect_ratio=aspect_ratio,
        target_megapixels=target_megapixels,
        cfg=cfg if caps.supports_cfg and cfg is not None else caps.cfg_range.default,
        denoise=denoise if caps.supports_denoise and denoise is not None else 1.0,
        sampler_name=sampler_name if caps.sampler_choices else None,
        scheduler=scheduler if caps.scheduler_choices else None,
        negative_prompt=(negative_prompt or "") if caps.supports_negative_prompt else "",
        lora_files=lora_files if caps.supports_loras else [],
        lora_strength=lora_strength if caps.supports_loras and lora_strength is not None else caps.lora_strength_range.default,
        apply_color_correction_enabled=bool(apply_color_correction),
        feather_amount=int(feather_amount) if caps.supports_inpainting and feather_amount is not None else 0,
    )
