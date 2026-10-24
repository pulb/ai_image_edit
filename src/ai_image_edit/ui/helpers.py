# SPDX-License-Identifier: GPL-3.0-or-later
"""Constants and helpers shared by the UI."""
import hashlib
import os
from typing import Any, Mapping, Optional, Sequence

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import (
    DEFAULT_RESCALE_MODE, GenerationParams, GenerationProgress, Images, Inference, ModelCapabilities, Postprocessing,
    ProgressStage, RescaleMode, Size,
)

# This app's one accent colour: the UI reads its purple from here.
PRIMARY_COLOR = "#7c3aed"

# The grey frame of the image boxes and of the text and select fields, and the
# lighter grey a field's frame turns while the pointer is over it.
FRAME_BORDER = "#444"
FRAME_BORDER_HOVER = "#888"

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


def _option_value(caps: ModelCapabilities, name: str, value: Any) -> Any:
    """An option control's value as the model takes it, or None if the model has no such option."""
    if value is None:
        return None
    if name == "negative_prompt":
        return str(value) if caps.supports_negative_prompt else None
    supported = {"cfg": caps.supports_cfg, "denoise": caps.supports_denoise}
    return float(value) if supported.get(name) else None


def params_from_ui(
    caps: ModelCapabilities,
    *,
    prompt: Optional[str],
    source_image_path: str,
    mask_path: Optional[str],
    annotated_image_path: Optional[str] = None,
    reference_images: Sequence[str],
    seed: Optional[float],
    randomize_seed: bool,
    aspect_ratio: str,
    resolution: int,
    steps: int,
    sampler: Optional[str],
    scheduler: Optional[str],
    options: Mapping[str, Any],
    lora_files: Sequence[str],
    lora_strength: Optional[float] = None,
    color_correction: bool,
    feather_amount: Optional[int] = None,
    rescale_mode: Optional[str] = None,
    match_detail: bool = True,
) -> GenerationParams:
    """
    Builds the GenerationParams for one generation from the controls' values.
    `options` holds the optional workflow inputs' controls by name; one the
    model doesn't have, or that is None, is left out. Masks, LoRAs and the
    masked-edit settings are dropped for a model without them.

    lora_strength falls back to the capability's own default.
    rescale_mode is a RescaleMode or its value; anything else raises ValueError.
    """
    inpainting = caps.supports_inpainting
    chosen = {name: _option_value(caps, name, value) for name, value in options.items()}
    return GenerationParams(
        images=Images(
            source_path=source_image_path,
            mask_path=mask_path if inpainting else None,
            annotated_path=annotated_image_path,
            reference_paths=tuple(reference_images),
        ),
        size=Size(aspect_ratio=aspect_ratio, resolution=resolution),
        inference=Inference(
            prompt=prompt or "",
            steps=int(steps),
            sampler=default_choice(sampler, caps.sampler_choices),
            scheduler=default_choice(scheduler, caps.scheduler_choices),
            seed=None if randomize_seed or seed is None else int(seed),
            options={name: value for name, value in chosen.items() if value is not None},
            lora_files=tuple(lora_files) if caps.supports_loras else (),
            lora_strength=lora_strength if caps.supports_loras and lora_strength is not None else caps.lora_strength_range.default,
        ),
        postprocessing=Postprocessing(
            feather_amount=int(feather_amount) if inpainting and feather_amount is not None else 0,
            color_correction=bool(color_correction),
            match_detail=bool(match_detail),
            rescale_mode=RescaleMode(rescale_mode) if inpainting and rescale_mode is not None else DEFAULT_RESCALE_MODE,
        ),
    )
