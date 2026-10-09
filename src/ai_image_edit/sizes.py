# SPDX-License-Identifier: GPL-3.0-or-later
"""
How a model turns the UI's size choices (aspect ratio, "megapixels")
into the numbers its workflow needs. A manifest names one policy under
"size"; adding a model with a new sizing scheme means adding a class here and
registering it in POLICIES.

A policy answers three questions: what the UI may offer (aspect ratios,
megapixel choices), which resolution to generate at for one request, and which
named values (e.g. "width"/"height", or "resolution") the manifest should write
into the workflow.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ai_image_edit.core.types import ORIGINAL_ASPECT_RATIO, GenerationParams

ORIGINAL = ORIGINAL_ASPECT_RATIO


@dataclass
class ResolvedSize:
    width: int
    height: int
    # Named values the manifest's size.bind maps to workflow inputs.
    values: Dict[str, Any]
    # Extra identity of the request for the result cache (besides width/height).
    cache_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class SizePolicy:
    aspect_ratios: List[str]
    default_aspect_ratio: str
    megapixels: List[float]
    default_megapixels: float

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        """The megapixels value that fits the source image, or None if the policy has no such notion."""
        return None

    def resolve(self, params: GenerationParams) -> ResolvedSize:
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

# Resolution tiers, reusing the "megapixels" field every model's
# capabilities expose for its resolution choices. Here it's actually this
# model's own tier (1K/1.5K/2K, i.e. target side length 1024/1536/2048)
# rather than a literal computed pixel area. _resolution_for() turns the chosen
# value back into that side length.
QWEN21_TIERS = [1.0, 1.5, 2.0]
QWEN21_DEFAULT_TIER = 1.0


def _round32(v: float) -> int:
    """Rounds to the nearest multiple of 32, minimum 32."""
    return max(32, int(round(v / 32)) * 32)


def _tier_size(resolution: int, aspect_ratio: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Width/height for a resolution tier and a named aspect ratio: the model
    card's size for that ratio, scaled by resolution / QWEN21_BASE_RESOLUTION.
    Returns (None, None) for ORIGINAL and for unknown names; use
    _dimensions_from_source() then.
    """
    if not aspect_ratio or aspect_ratio == ORIGINAL or aspect_ratio not in QWEN21_ASPECT_RATIOS:
        return None, None
    base_w, base_h = QWEN21_ASPECT_RATIOS[aspect_ratio]
    scale = resolution / QWEN21_BASE_RESOLUTION
    return _round32(base_w * scale), _round32(base_h * scale)


def _dimensions_from_source(source_image_path: str, resolution: int) -> Tuple[int, int]:
    """
    Derives concrete (width, height) from the source image's own aspect
    ratio at the given resolution tier. Used whenever _tier_size left
    width/height unset (ORIGINAL, i.e. "Original"): the generation
    needs concrete numbers up front, to crop the source to. Area scales with resolution the same
    way QWEN21_ASPECT_RATIOS' entries do (area ~ resolution^2), just computed
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


def _resolution_for(target_megapixels: float) -> int:
    """The side-length tier (1024/1536/2048) a QWEN21_TIERS value stands for."""
    return int(round(target_megapixels * 1024))


def _nearest_tier(source_image_path: str) -> float:
    """The tier whose area is closest to the source image's own."""
    with Image.open(source_image_path) as img:
        area = img.width * img.height
    return min(QWEN21_TIERS, key=lambda mp: abs(math.log(area / (mp * 1024) ** 2)))


class Qwen21Tiers(SizePolicy):
    """Qwen-Image-2.1: resolution tiers (1K/1.5K/2K) and the model card's aspect ratios."""

    def __init__(self, config: dict) -> None:  # takes no settings; the signature is the policies' common one
        self.aspect_ratios = [ORIGINAL] + list(QWEN21_ASPECT_RATIOS)
        self.default_aspect_ratio = ORIGINAL
        self.megapixels = list(QWEN21_TIERS)
        self.default_megapixels = QWEN21_DEFAULT_TIER

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        return _nearest_tier(source_image_path)

    def resolve(self, params: GenerationParams) -> ResolvedSize:
        # A mask only lines up with the source's own framing and resolution,
        # so both come from the source then.
        aspect = ORIGINAL if params.mask_path else params.aspect_ratio
        megapixels = self.megapixels_for_source(params.source_image_path) if params.mask_path else params.target_megapixels
        resolution = _resolution_for(megapixels)
        width, height = _tier_size(resolution, aspect)
        if width is None or height is None:
            width, height = _dimensions_from_source(params.source_image_path, resolution)
        return ResolvedSize(width, height, {"resolution": resolution}, {"resolution": resolution}, f"resolution_tier={resolution}")


class FixedArea(SizePolicy):
    """A target pixel area (megapixels) split by an aspect ratio, rounded to a multiple of `multiple`."""

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
        ORIGINAL: (0, 0),
    }

    def __init__(self, config: dict) -> None:
        self.aspect_ratios = list(self.RATIOS.keys())
        self.default_aspect_ratio = ORIGINAL
        self.megapixels = list(config["megapixels"])
        self.default_megapixels = config.get("default_megapixels", self.megapixels[0])
        self.multiple = int(config.get("multiple", 8))

    def _dimensions_for(self, aspect_w: int, aspect_h: int, target_area: float) -> Tuple[int, int]:
        ratio = aspect_w / aspect_h
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        return (int(round(width / self.multiple) * self.multiple), int(round(height / self.multiple) * self.multiple))

    def resolve(self, params: GenerationParams) -> ResolvedSize:
        # A mask's coordinates only mean something relative to the source's
        # own framing, so a mask always gets the source's aspect ratio.
        aspect = ORIGINAL if params.mask_path else params.aspect_ratio
        target_area = params.target_megapixels * 1024 * 1024
        if aspect == ORIGINAL:
            with Image.open(params.source_image_path) as img:
                ratio_w, ratio_h = img.size
        else:
            ratio_w, ratio_h = self.RATIOS[aspect]
        width, height = self._dimensions_for(ratio_w, ratio_h, target_area)
        return ResolvedSize(width, height, {"width": width, "height": height}, {}, f"mp={params.target_megapixels}")


POLICIES = {
    "qwen21_tiers": Qwen21Tiers,
    "fixed_area": FixedArea,
}
