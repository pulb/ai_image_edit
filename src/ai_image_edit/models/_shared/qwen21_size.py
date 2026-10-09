# SPDX-License-Identifier: GPL-3.0-or-later
"""
Output sizes for Qwen-Image-2.1 models: the aspect ratios and resolution
tiers from the model card, and the helpers that turn a choice of either into
pixel dimensions. Used by the Qwen-Image-2.1 workflow files (qwen_image21,
qwen_image21_gguf); pure functions, no model dependencies.
"""
import math
from typing import Dict, Optional, Tuple

from PIL import Image

# The aspect ratios the model card lists, at its 2048-base resolution.
BASE_RESOLUTION = 2048
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# This project's sentinel for "derive dimensions from context rather than
# force a fixed ratio" is "Original" (e.g. the frontends lock the
# aspect-ratio dropdown to it while a mask is drawn). For this model it
# means the pipeline follows the input images' own aspect ratio, or squares
# up for pure text-to-image.
AUTO_ASPECT_RATIO = "Original"

# Resolution tiers, reusing the "megapixels" field every model's
# capabilities expose for its resolution choices. Here it's actually this
# model's own tier (1K/1.5K/2K, i.e. target side length 1024/1536/2048)
# rather than a literal computed pixel area. See _resolve_dimensions(),
# which turns the chosen value back into that side length before calling
# resolve_size().
SUPPORTED_MEGAPIXELS = [1.0, 1.5, 2.0]
DEFAULT_MEGAPIXELS = 1.0

DEFAULT_STEPS, MIN_STEPS, MAX_STEPS = 40, 8, 60
DEFAULT_CFG, MIN_CFG, MAX_CFG = 1.0, 1.0, 10.0


def round32(v: float) -> int:
    """Rounds to the nearest multiple of 32, minimum 32."""
    return max(32, int(round(v / 32)) * 32)


def resolve_size(resolution: int, aspect_ratio: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Width/height for a resolution tier and a named aspect ratio: the model
    card's size for that ratio, scaled by resolution / BASE_RESOLUTION.
    Returns (None, None) for AUTO_ASPECT_RATIO — the pipeline then infers
    dimensions itself (from the input images, or a square default for pure
    text-to-image) rather than being told a fixed size. Only used when
    there's no mask — see dimensions_from_source() for the masked case,
    which needs concrete numbers regardless of aspect_ratio.
    """
    if not aspect_ratio or aspect_ratio == AUTO_ASPECT_RATIO or aspect_ratio not in ASPECT_RATIOS:
        return None, None
    base_w, base_h = ASPECT_RATIOS[aspect_ratio]
    scale = resolution / BASE_RESOLUTION
    return round32(base_w * scale), round32(base_h * scale)


def dimensions_from_source(source_image_path: str, resolution: int) -> Tuple[int, int]:
    """
    Derives concrete (width, height) from the source image's own aspect
    ratio at the given resolution tier. Used whenever resolve_size left
    width/height unset (AUTO_ASPECT_RATIO, i.e. "Original") — not just while
    a mask is drawn: imaging.run_masked_generation needs real numbers up
    front to crop the source to before inference, and passing width=height=
    None straight through to the pipeline instead doesn't reliably keep the
    source's own aspect ratio either. Area scales with resolution the same
    way ASPECT_RATIOS' entries do (area ~ resolution^2), just computed
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
    return round32(w), round32(h)


def resolution_for(target_megapixels: float) -> int:
    """The side-length tier (1024/1536/2048) a SUPPORTED_MEGAPIXELS value stands for."""
    return int(round(target_megapixels * 1024))


def megapixels_for_source(source_image_path: str) -> float:
    """The tier whose area is closest to the source image's own."""
    with Image.open(source_image_path) as img:
        area = img.width * img.height
    return min(SUPPORTED_MEGAPIXELS, key=lambda mp: abs(math.log(area / (mp * 1024) ** 2)))
