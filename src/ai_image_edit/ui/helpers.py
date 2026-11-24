# SPDX-License-Identifier: GPL-3.0-or-later
"""Constants and helpers shared by the UI."""
import hashlib
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import (
    GenerationParams, GenerationProgress, Images, Inference, ModelCapabilities, Postprocessing,
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

# Aspect ratios of the original and the result closer than this (as a share)
# mean the original was fitted to the result's size by trimming its edges: a
# mask, the "Original" ratio (rounding to multiples of 32 px makes up to about
# 3.5% at 1K) or a ratio within sizes.QWEN21_FIT_ASPECT_TOLERANCE. Any two of
# the offered ratios differ by 10% or more; a result in another one was
# recomposed from the whole original.
SAME_FRAMING_TOLERANCE = 0.05


def before_image_fit(before_size: Tuple[int, int], after_size: Tuple[int, int]) -> str:
    """
    The CSS object-fit of the compare slider's "before" image over the result:
    "cover" trims the original as it was fitted to the result's framing, so
    the two line up; "contain" shows all of an original the model recomposed
    into a clearly different shape.
    """
    mismatch = abs((before_size[0] / before_size[1]) / (after_size[0] / after_size[1]) - 1)
    return "cover" if mismatch <= SAME_FRAMING_TOLERANCE else "contain"


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
    source_path: str,
    mask_path: Optional[str],
    annotated_path: Optional[str] = None,
    reference_paths: Sequence[str],
    seed: Optional[float],
    randomize_seed: bool,
    aspect_ratio: str,
    resolution: int,
    steps: int,
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
    `options` holds the values of the model's optional inputs' controls by
    name. Masks, LoRAs and the masked-edit settings are dropped for a model
    without them.

    lora_strength falls back to the capability's own default.
    rescale_mode is a RescaleMode or its value; anything else raises ValueError.
    """
    inpainting = caps.supports_inpainting
    specs = {spec.name: spec for spec in caps.options}
    return GenerationParams(
        images=Images(
            source_path=source_path,
            mask_path=mask_path if inpainting else None,
            annotated_path=annotated_path,
            reference_paths=tuple(reference_paths),
        ),
        size=Size(aspect_ratio=aspect_ratio, resolution=resolution),
        inference=Inference(
            prompt=prompt or "",
            steps=int(steps),
            seed=None if randomize_seed or seed is None else int(seed),
            options={name: specs[name].convert(value) for name, value in options.items()},
            lora_files=tuple(lora_files) if caps.supports_loras else (),
            lora_strength=lora_strength if caps.supports_loras and lora_strength is not None else caps.lora_strength_range.default,
        ),
        postprocessing=Postprocessing(
            feather_amount=int(feather_amount) if inpainting and feather_amount is not None else 0,
            color_correction=bool(color_correction),
            match_detail=bool(match_detail),
            rescale_mode=RescaleMode(rescale_mode) if inpainting and rescale_mode is not None else Postprocessing.rescale_mode,
        ),
    )
