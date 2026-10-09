# SPDX-License-Identifier: GPL-3.0-or-later
"""
Data contracts shared between the UI layer (frontends/nicegui.py) and every model backend under models/. Deliberately
dependency-free of both frontends/ and models/, so neither side ever needs
to import from the other.
"""
from dataclasses import dataclass, field
from typing import List, Optional


# The colours a user can annotate an image with, in the order a model uses
# them: a model with num_annotation_colors = N supports the first N.
ANNOTATION_COLORS = ("#ff0000", "#00ff00", "#0000ff")


@dataclass(frozen=True)
class RangeSpec:
    """A numeric control's min/max/default/step."""

    min: float
    max: float
    default: float
    step: float = 1.0


# The "derive the output size from the source image" aspect ratio label.
ORIGINAL_ASPECT_RATIO = "Original"

# The seed the frontend starts with.
DEFAULT_SEED = 65454653

# Inpainting mask edge softness (Gaussian blur radius), shared by the
# frontends' Feather slider and GenerationParams.feather_amount.
FEATHER_RANGE = RangeSpec(0, 16, 3, step=1)


@dataclass(frozen=True)
class ModelCapabilities:
    """
    What a model backend supports, declared by the model itself. The UI
    reads this once (via ModelBackend.capabilities) to decide which
    controls to render, instead of hardcoding assumptions about any one
    model — e.g. how many reference-image upload boxes to draw, whether to
    show the LoRA panel, or which sampler/scheduler/aspect-ratio/resolution
    choices to offer.
    """

    # Total number of reference/source images this model accepts, primary
    # source image included. E.g. 3 means "1 primary source + up to 2
    # extra reference images".
    max_reference_images: int

    supports_loras: bool
    supports_inpainting: bool

    # Aspect ratio labels this model understands — it interprets these
    # itself, the UI just offers them as-is in a dropdown. Include
    # "Original" only if the model supports deriving dimensions from the
    # source image's own aspect ratio. default_aspect_ratio must be one of
    # these labels.
    supported_aspect_ratios: List[str]
    default_aspect_ratio: str

    # Output resolution choices, in megapixels (e.g. [1.0, 1.5, 2.0] — or
    # just [1.0] for a model that only really supports one target area).
    # The model converts (aspect_ratio, megapixels) into actual pixel
    # dimensions internally — the UI never computes or sees real pixel
    # dimensions. default_megapixels must be one of supported_megapixels.
    supported_megapixels: List[float]
    default_megapixels: float

    # None hides the corresponding UI control entirely rather than showing
    # a dropdown with a meaningless single choice. default_sampler/
    # default_scheduler must be one of the respective choices list (or
    # None, when that list itself is None).
    sampler_choices: Optional[List[str]] = None
    default_sampler: Optional[str] = None
    scheduler_choices: Optional[List[str]] = None
    default_scheduler: Optional[str] = None

    supports_cfg: bool = True
    supports_denoise: bool = True
    supports_seed: bool = True
    # False by default: qwen_image_edit_2511_aio applies its own fixed negative
    # prompt internally and never asks the user for one. A model that wants
    # a user-facing negative-prompt box (only useful alongside supports_cfg,
    # since CFG is what actually uses it) sets this True.
    supports_negative_prompt: bool = False

    # How many annotation colours the model understands (the first N of
    # ANNOTATION_COLORS, clipped to the palette's length if N is larger);
    # 0 hides the annotate control entirely. Annotations
    # are strokes drawn on the source image that the model reads as part of
    # the picture.
    num_annotation_colors: int = 0

    step_range: RangeSpec = field(default_factory=lambda: RangeSpec(1, 50, 20, step=1))
    cfg_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.1, 10.0, 1.0, step=0.1))
    denoise_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.0, 1.0, 1.0, step=0.01))
    lora_strength_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.1, 2.0, 0.7, step=0.05))


@dataclass
class GenerationParams:
    """
    Bundles every input a model backend's generate() needs for one
    generation. Some fields are meaningless for a given model (e.g. cfg
    for a model whose capabilities.supports_cfg is False) — the model
    ignores what it doesn't use, and the UI only renders/sets a field when
    capabilities says it applies.
    """

    prompt: str
    source_image_path: str
    mask_path: Optional[str]

    # No default: the sensible step count is model-specific (see
    # capabilities.step_range), so the caller always states it.
    steps: int

    # Extra reference images beyond the primary source, in order. Length
    # is bounded by capabilities.max_reference_images - 1.
    reference_images: List[str] = field(default_factory=list)

    # The source image with the user's annotations drawn in, same size as
    # source_image_path. What the model sees instead of the source; masking,
    # compositing and colour correction still use the clean source.
    annotated_image_path: Optional[str] = None

    seed: int = 0
    randomize_seed: bool = True

    aspect_ratio: str = ORIGINAL_ASPECT_RATIO
    target_megapixels: float = 1.0

    cfg: float = 1.0
    denoise: float = 1.0
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    negative_prompt: str = ""

    lora_files: List[str] = field(default_factory=list)
    lora_strength: float = 0.7

    apply_color_correction_enabled: bool = False
    feather_amount: int = int(FEATHER_RANGE.default)


@dataclass
class GenerationResult:
    """What a model backend's generate() hands back to the UI."""

    before_path: str
    after_path: str
    actual_seed: int
