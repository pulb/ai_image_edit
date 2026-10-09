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
from ai_image_edit.models._shared import qwen21_size

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


class Qwen21Tiers(SizePolicy):
    """Qwen-Image-2.1: resolution tiers (1K/1.5K/2K) and the model card's aspect ratios."""

    def __init__(self, config: dict) -> None:  # takes no settings; the signature is the policies' common one
        self.aspect_ratios = [qwen21_size.AUTO_ASPECT_RATIO] + list(qwen21_size.ASPECT_RATIOS.keys())
        self.default_aspect_ratio = qwen21_size.AUTO_ASPECT_RATIO
        self.megapixels = list(qwen21_size.SUPPORTED_MEGAPIXELS)
        self.default_megapixels = qwen21_size.DEFAULT_MEGAPIXELS

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        return qwen21_size.megapixels_for_source(source_image_path)

    def resolve(self, params: GenerationParams) -> ResolvedSize:
        # A mask only lines up with the source's own framing and resolution,
        # so both come from the source then.
        aspect = qwen21_size.AUTO_ASPECT_RATIO if params.mask_path else params.aspect_ratio
        megapixels = self.megapixels_for_source(params.source_image_path) if params.mask_path else params.target_megapixels
        resolution = qwen21_size.resolution_for(megapixels)
        width, height = qwen21_size.resolve_size(resolution, aspect)
        if width is None or height is None:
            width, height = qwen21_size.dimensions_from_source(params.source_image_path, resolution)
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
