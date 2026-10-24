# SPDX-License-Identifier: GPL-3.0-or-later
"""
How a model turns the UI's size choices (aspect ratio, resolution)
into the numbers its workflow needs. A manifest names one policy under
"size"; adding a model with a new sizing scheme means adding a class here and
registering it in POLICIES.

A resolution is the side length, in pixels, of a square with the area to
generate (1024 is "1K"); the aspect ratio then shapes that area.

A policy answers three questions: what the UI may offer (aspect ratios,
resolutions), which size to generate at for one request, and which named
values (e.g. "width"/"height", or "resolution") the manifest should write
into the workflow.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from PIL import Image

from ai_image_edit.core.types import ORIGINAL_ASPECT_RATIO, Size


@dataclass
class ResolvedSize:
    width: int
    height: int
    # Named values the manifest's size.bind maps to workflow inputs.
    values: Dict[str, Any]
    # Extra identity of the request for the result cache (besides width/height).
    cache_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # Set when the app, not the model, scales the input images: the source is
    # fitted to input_size (width, height) before inference, and the reference
    # images are scaled to reference_resolution's area, keeping their own
    # aspect ratio. Unset, the images are sent as they are and the model's
    # workflow scales them. A masked edit needs input_size to be width x height,
    # or unset (then it is fitted to width x height all the same).
    input_size: Optional[Tuple[int, int]] = None
    reference_resolution: Optional[int] = None


class SizePolicy:
    """Built from the manifest's "size" section."""

    aspect_ratios: List[str]
    default_aspect_ratio: str
    resolutions: List[int]
    default_resolution: int
    # The names resolve() puts in ResolvedSize.values. A manifest must bind
    # exactly these, so a choice offered in the UI cannot end up unused.
    value_names: FrozenSet[str]

    def __init__(self, config: dict) -> None:
        pass

    def resolution_for_source(self, source_image_path: str) -> Optional[int]:
        """The resolution that fits the source image, or None if the policy has no such notion."""
        return None

    def resolve(self, size: Size, source_path: str, masked: bool) -> ResolvedSize:
        """The size to generate at for the chosen size, the source image and whether the edit has a mask."""
        raise NotImplementedError


# --- Qwen-Image-2.1 -------------------------------------------------------

# The aspect ratios the model card lists, at its 2048-base resolution.
QWEN21_BASE_RESOLUTION = 2048
QWEN21_ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# The model's resolution tiers: 1K, 1.5K and 2K.
QWEN21_RESOLUTIONS = [1024, 1536, 2048]
QWEN21_DEFAULT_RESOLUTION = 1024

# How far (as a share) a source's aspect ratio may be from the chosen ratio for
# the source to be fitted to the generation size, which trims at most about
# this much off one edge. Beyond it the ratios differ on purpose: the source
# keeps its own ratio and the model recomposes it into the chosen one.
QWEN21_FIT_ASPECT_TOLERANCE = 0.02


def _round32(v: float) -> int:
    """Rounds to the nearest multiple of 32, minimum 32."""
    return max(32, int(round(v / 32)) * 32)


def _image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _split_area(ratio: float, area: float) -> Tuple[float, float]:
    """The (width, height) of the given area at the given aspect ratio (width / height)."""
    width = math.sqrt(area * ratio)
    return width, width / ratio


def _tier_size(resolution: int, aspect_ratio: str) -> Optional[Tuple[int, int]]:
    """
    Width/height for a resolution and a named aspect ratio: the model
    card's size for that ratio, scaled by resolution / QWEN21_BASE_RESOLUTION.
    None for ORIGINAL_ASPECT_RATIO and for unknown names; use
    dimensions_from_source() then.
    """
    if aspect_ratio not in QWEN21_ASPECT_RATIOS:
        return None
    base_w, base_h = QWEN21_ASPECT_RATIOS[aspect_ratio]
    scale = resolution / QWEN21_BASE_RESOLUTION
    return _round32(base_w * scale), _round32(base_h * scale)


def _dimensions_for(source_size: Tuple[int, int], resolution: int) -> Tuple[int, int]:
    """dimensions_from_source() for a source of the given (width, height)."""
    src_w, src_h = source_size
    target_area = float(resolution) ** 2
    if src_w % 32 == 0 and src_h % 32 == 0 and 0.9 <= src_w * src_h / target_area <= 1.1:
        return src_w, src_h
    w, h = _split_area(src_w / src_h, target_area)
    return _round32(w), _round32(h)


def dimensions_from_source(source_image_path: str, resolution: int) -> Tuple[int, int]:
    """
    Derives concrete (width, height) from the source image's own aspect
    ratio at the given resolution tier, for when _tier_size has no size (the
    "Original" ratio): the generation needs concrete numbers up front, to crop
    the source to. Area scales with resolution the same way
    QWEN21_ASPECT_RATIOS' entries do (area ~ resolution^2). A source that is
    already a valid size for the tier (multiples of 32, area within 10% of the
    tier's) keeps its own size.
    """
    return _dimensions_for(_image_size(source_image_path), resolution)


def _nearest_resolution(source_size: Tuple[int, int]) -> int:
    """The resolution closest to a source of the given size (the side of a square of its area)."""
    side = math.sqrt(source_size[0] * source_size[1])
    return min(QWEN21_RESOLUTIONS, key=lambda r: abs(math.log(side / r)))


def _ratio_matches(source_size: Tuple[int, int], width: int, height: int) -> bool:
    """Whether the source's aspect ratio is within QWEN21_FIT_ASPECT_TOLERANCE of width:height."""
    return abs(source_size[0] / source_size[1] / (width / height) - 1) <= QWEN21_FIT_ASPECT_TOLERANCE


class Qwen21Tiers(SizePolicy):
    """
    Qwen-Image-2.1: resolution tiers (1K/1.5K/2K) and the model card's aspect ratios.
    The workflow samples an empty latent of width x height, so the output size is
    independent of the source's own shape. The app scales every input image
    and the encode node gets resolution 0, so it keeps their sizes: where the
    source keeps its framing (a mask, the "Original" ratio, or a chosen ratio
    close to its own) it is fitted to exactly width x height, so that it and
    the latent have the same size; otherwise it keeps its own ratio at the
    tier, as the node would have scaled it, and the model recomposes it.
    """

    aspect_ratios = [ORIGINAL_ASPECT_RATIO, *QWEN21_ASPECT_RATIOS]
    default_aspect_ratio = ORIGINAL_ASPECT_RATIO
    resolutions = QWEN21_RESOLUTIONS
    default_resolution = QWEN21_DEFAULT_RESOLUTION
    value_names = frozenset({"resolution", "width", "height"})

    def resolution_for_source(self, source_image_path: str) -> Optional[int]:
        return _nearest_resolution(_image_size(source_image_path))

    def resolve(self, size: Size, source_path: str, masked: bool) -> ResolvedSize:
        source_size = _image_size(source_path)
        # A mask only lines up with the source's own framing and resolution,
        # so both come from the source then.
        aspect = ORIGINAL_ASPECT_RATIO if masked else size.aspect_ratio
        resolution = _nearest_resolution(source_size) if masked else size.resolution
        tier = _tier_size(resolution, aspect)
        width, height = tier or _dimensions_for(source_size, resolution)
        # A mask always takes the source's size (its aspect ratio is forced
        # to "Original" above), so it needs no check of its own here.
        if tier is None or _ratio_matches(source_size, width, height):
            input_size = (width, height)
        else:
            # A clearly different ratio: the whole image, at its own ratio.
            input_size = _dimensions_for(source_size, resolution)
        # The encode node would rescale the images by a rule of its own, which
        # can land a few pixels off the size the latent is sampled at, and
        # sampling at another size than the scaled input shifts the edit
        # against the original. Resolution 0 makes it keep each image's size
        # (rounded to a multiple of 32, which ours already are).
        values = {"resolution": 0, "width": width, "height": height}
        return ResolvedSize(
            width, height, values, {"resolution": resolution},
            f"resolution={resolution} input={input_size[0]}x{input_size[1]}",
            input_size=input_size, reference_resolution=resolution,
        )


class FixedArea(SizePolicy):
    """A target pixel area (the square of the resolution) split by an aspect ratio, rounded to a multiple of `multiple`."""

    RATIOS: Dict[str, Tuple[int, int]] = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "9:16": (9, 16),
        "4:3": (4, 3),
        "3:4": (3, 4),
        "3:2": (3, 2),
        "2:3": (2, 3),
        "21:9": (21, 9),
        "9:21": (9, 21),
    }

    value_names = frozenset({"width", "height"})

    def __init__(self, config: dict) -> None:
        self.aspect_ratios = [*self.RATIOS, ORIGINAL_ASPECT_RATIO]
        self.default_aspect_ratio = ORIGINAL_ASPECT_RATIO
        self.resolutions = list(config["resolutions"])
        self.default_resolution = config.get("default_resolution", self.resolutions[0])
        self.multiple = int(config.get("multiple", 8))

    def _dimensions_for(self, aspect_w: int, aspect_h: int, target_area: float) -> Tuple[int, int]:
        width, height = _split_area(aspect_w / aspect_h, target_area)
        return (int(round(width / self.multiple) * self.multiple), int(round(height / self.multiple) * self.multiple))

    def resolve(self, size: Size, source_path: str, masked: bool) -> ResolvedSize:
        # A mask's coordinates only mean something relative to the source's
        # own framing, so a mask always gets the source's aspect ratio.
        aspect = ORIGINAL_ASPECT_RATIO if masked else size.aspect_ratio
        target_area = float(size.resolution) ** 2
        if aspect == ORIGINAL_ASPECT_RATIO:
            ratio_w, ratio_h = _image_size(source_path)
        else:
            ratio_w, ratio_h = self.RATIOS[aspect]
        width, height = self._dimensions_for(ratio_w, ratio_h, target_area)
        return ResolvedSize(width, height, {"width": width, "height": height}, {}, f"resolution={size.resolution}")


POLICIES = {
    "qwen21_tiers": Qwen21Tiers,
    "fixed_area": FixedArea,
}
