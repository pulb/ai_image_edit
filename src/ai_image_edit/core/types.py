# SPDX-License-Identifier: GPL-3.0-or-later
"""
Data contracts shared between the UI (ui/) and the model (workflow_model.py).
Deliberately dependency-free of both, so neither side needs to import the other.
"""
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple


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

# The seed the UI starts with.
DEFAULT_SEED = 65454653

# Inpainting mask edge softness (Gaussian blur radius), shared by the
# UI's Feather slider and GenerationParams.feather_amount.
FEATHER_RANGE = RangeSpec(0, 16, 3, step=1)


class RescaleMode(str, Enum):
    """
    What a masked edit rescales: ORIGINAL scales the source image to the
    generated size, and the result has that size; EDIT scales the edited
    area to the source size, and the result keeps the source's size and
    every pixel outside that area.
    """

    ORIGINAL = "original"
    EDIT = "edit"


DEFAULT_RESCALE_MODE = RescaleMode.EDIT


@dataclass(frozen=True)
class ModelCapabilities:
    """
    What a model supports, derived from its workflow file. The UI
    reads this once (via WorkflowModel.capabilities) to decide which
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

    # Output resolution choices: the side length in pixels of a square with
    # the area to generate (e.g. [1024, 1536, 2048], shown as 1K, 1.5K and
    # 2K — or just [1024] for a model that only supports one). The model
    # converts (aspect_ratio, resolution) into actual pixel dimensions
    # internally — the UI never computes or sees real pixel dimensions.
    # default_resolution must be one of supported_resolutions.
    supported_resolutions: List[int]
    default_resolution: int

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
    # False by default: a model whose workflow fixes its own negative prompt
    # has nothing to ask for. A model that wants a user-facing negative-prompt
    # box (only useful alongside supports_cfg, since CFG is what actually
    # uses it) sets this True.
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


@dataclass(frozen=True)
class Images:
    """The images of one request, as local paths."""

    source_path: str
    mask_path: Optional[str] = None
    # The source with the user's annotations drawn in, same size as the source:
    # what the model sees instead of it. Masking, compositing and colour
    # correction still use the clean source.
    annotated_path: Optional[str] = None
    # Extra reference images beyond the source, in order; at most
    # capabilities.max_reference_images - 1.
    reference_paths: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Size:
    """The size the user chose: an aspect ratio label and a resolution (see sizes.py)."""

    aspect_ratio: str
    resolution: int


@dataclass(frozen=True)
class Inference:
    """
    What the model's run gets besides the images and the size. seed None
    means a random one. options are the model's optional workflow inputs by
    name (see capabilities.options); one the model does not have is absent.
    """

    prompt: str
    steps: int
    sampler: str
    scheduler: str
    seed: Optional[int]
    options: Mapping[str, Any] = field(default_factory=dict)
    lora_files: Tuple[str, ...] = ()
    lora_strength: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True)
class Postprocessing:
    """What happens to the model's output in the app, the same for every model."""

    feather_amount: int = int(FEATHER_RANGE.default)  # masked edits
    color_correction: bool = False  # unmasked edits
    match_detail: bool = True  # masked edits
    rescale_mode: RescaleMode = DEFAULT_RESCALE_MODE  # masked edits


@dataclass(frozen=True)
class GenerationParams:
    """Every input of one generation, in the groups the steps of a generation use."""

    images: Images
    size: Size
    inference: Inference
    postprocessing: Postprocessing = Postprocessing()


class ProgressStage(Enum):
    """Where one generation is; see GenerationProgress."""

    PREPARING = "preparing"  # before the sampler runs: uploads, loading models, encoding the prompt
    QUEUED = "queued"  # waiting in ComfyUI's queue behind another generation
    SAMPLING = "sampling"  # the sampler's steps
    FINISHING = "finishing"  # after the sampler: decoding, saving
    POSTPROCESSING = "postprocessing"  # compositing and color correction in the app


@dataclass(frozen=True)
class GenerationProgress:
    """How far one generation is. step and total are set while sampling."""

    stage: ProgressStage
    step: Optional[int] = None
    total: Optional[int] = None


@dataclass
class GenerationResult:
    """What a model's generate() hands back to the UI."""

    before_path: str
    after_path: str
    actual_seed: int
