# SPDX-License-Identifier: GPL-3.0-or-later
"""Constants and helpers shared by the UI."""
import hashlib
import os
from typing import List, Optional, Sequence

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import (
    DEFAULT_RESCALE_MODE, GenerationParams, GenerationProgress, ModelCapabilities, ProgressStage, RescaleMode,
)

# This app's one accent colour: the UI reads its purple from here.
PRIMARY_COLOR = "#7c3aed"

# Newest work-dir files trim_work_dir() never deletes: the files of the
# generation that just finished (source, mask, annotated, result, ...). The
# files the page still uses are protected separately, however old they are.
RECENT_FILES_KEPT = 8

HOST = "0.0.0.0"
PORT = 7860

# The Masked edits dropdown: RescaleMode value -> label.
RESCALE_MODE_OPTIONS = {
    RescaleMode.ORIGINAL.value: "Rescale original to generated size",
    RescaleMode.EDIT.value: "Rescale edited area to original size",
}


# Fixed because the secret must be the same after a restart; APP_STORAGE_SECRET
# replaces the whole derivation with a random value.
_SECRET_SALT = b"ai-image-edit/session-secret/v1"
_SECRET_ITERATIONS = 200_000


def storage_secret(password: str) -> str:
    """The secret that signs the session cookie: APP_STORAGE_SECRET, else a salted PBKDF2 hash of the password."""
    configured = os.environ.get("APP_STORAGE_SECRET")
    if configured:
        return configured
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SECRET_SALT, _SECRET_ITERATIONS).hex()


def resolution_label(resolution: int) -> str:
    """A resolution (side length in pixels) as shown to the user: 1024 is "1K", 1536 is "1.5K"."""
    return f"{resolution / 1024:g}K"


def default_choice(default: Optional[str], choices: Optional[Sequence[str]]) -> Optional[str]:
    """The model's declared default if it is one of `choices`, else the first choice (None if there are none)."""
    if not choices:
        return None
    return default if default in choices else choices[0]


PROGRESS_TEXT = {
    ProgressStage.PREPARING: "Preparing",
    ProgressStage.QUEUED: "Waiting for another generation",
    ProgressStage.SAMPLING: "Generating",
    ProgressStage.FINISHING: "Finishing",
    ProgressStage.POSTPROCESSING: "Postprocessing",
}


def describe_progress(progress: GenerationProgress, elapsed_seconds: float, cancelling: bool = False) -> str:
    """The status line above the Generate button, e.g. "Generating: step 12 of 40 · 0:23"."""
    text = "Cancelling" if cancelling else PROGRESS_TEXT[progress.stage]
    if cancelling:
        text += "…"
    elif progress.stage is ProgressStage.SAMPLING and progress.step is not None and progress.total:
        text += f": step {progress.step} of {progress.total}"
    else:
        text += "…"
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    return f"{text} · {minutes}:{seconds:02d}"


def progress_fraction(progress: GenerationProgress) -> Optional[float]:
    """How full the bar is while sampling, 0 to 1; None (an indeterminate bar) otherwise."""
    if progress.stage is ProgressStage.SAMPLING and progress.step is not None and progress.total:
        return min(1.0, max(0.0, progress.step / progress.total))
    return None


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
    resolution: int,
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
    rescale_mode: Optional[str] = None,
    match_detail: bool = True,
) -> GenerationParams:
    """
    Builds the GenerationParams for one generation. A control the model
    doesn't support may be passed as None (or hold any value): the
    capability decides, and the field gets a neutral value instead.

    cfg and lora_strength fall back to the capability's own declared
    default (not a bare 0.0) when unsupported, since both ranges start above 0.
    rescale_mode is a RescaleMode or its value; anything else raises ValueError.
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
        resolution=resolution,
        cfg=cfg if caps.supports_cfg and cfg is not None else caps.cfg_range.default,
        denoise=denoise if caps.supports_denoise and denoise is not None else 1.0,
        sampler_name=sampler_name if caps.sampler_choices else None,
        scheduler=scheduler if caps.scheduler_choices else None,
        negative_prompt=(negative_prompt or "") if caps.supports_negative_prompt else "",
        lora_files=lora_files if caps.supports_loras else [],
        lora_strength=lora_strength if caps.supports_loras and lora_strength is not None else caps.lora_strength_range.default,
        apply_color_correction_enabled=bool(apply_color_correction),
        match_detail_enabled=bool(match_detail),
        feather_amount=int(feather_amount) if caps.supports_inpainting and feather_amount is not None else 0,
        rescale_mode=RescaleMode(rescale_mode) if caps.supports_inpainting and rescale_mode is not None else DEFAULT_RESCALE_MODE,
    )
