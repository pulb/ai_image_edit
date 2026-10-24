# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting images to a size without distorting them: scaled to cover it, then centre-cropped."""
import math
from typing import Tuple

from PIL import Image

from ai_image_edit.core.paths import new_work_file


def cover_geometry(size: Tuple[int, int], target_w: int, target_h: int) -> Tuple[int, int, int, int]:
    """
    How resize_to_cover_and_crop() fits an image of `size` to the target: the
    (width, height) it is scaled to and the (left, top) of the crop in that.
    ceil (not round) guarantees the scaled image covers the target on both
    axes even after float rounding; round() could land a pixel short.
    """
    src_w, src_h = size
    scale = max(target_w / src_w, target_h / src_h)
    covered_w = max(target_w, math.ceil(src_w * scale))
    covered_h = max(target_h, math.ceil(src_h * scale))
    return covered_w, covered_h, (covered_w - target_w) // 2, (covered_h - target_h) // 2


def resize_to_cover_and_crop(image: Image.Image, target_w: int, target_h: int, resample: int) -> Image.Image:
    """
    Fits image to exactly (target_w, target_h) without distorting it: scales
    uniformly (same factor on both axes) until it covers the target box,
    then center-crops the overhang — CSS object-fit: cover. No-op (returns
    image unchanged) if already that exact size.
    """
    if image.size == (target_w, target_h):
        return image

    covered_w, covered_h, left, top = cover_geometry(image.size, target_w, target_h)
    resized = image.resize((covered_w, covered_h), resample)
    return resized.crop((left, top, left + target_w, top + target_h))


def fit_image_to_size(image_path: str, gen_width: int, gen_height: int, resample: int) -> str:
    """
    Resizes image_path (via resize_to_cover_and_crop) to exactly
    (gen_width, gen_height) and returns the new file's path, or image_path
    itself unchanged if it's already that size: the file counterpart of
    resize_to_cover_and_crop(). Used to align inpaint mode's source image
    with the model's generation canvas before inference;
    composite_with_soft_transition() applies the same transform to the
    original and mask at blend time, keeping all three in the same
    coordinate space. Also scales a source or reference image to a size of its
    own aspect ratio (rounded to multiples of 32, so at most a few pixels are
    cropped) when the app rather than the model scales the inputs (see
    sizes.ResolvedSize.input_size).
    """
    with Image.open(image_path) as img:
        if img.size == (gen_width, gen_height):
            return image_path
        print(f"[fit_image_to_size] Resizing {img.size} to {(gen_width, gen_height)} (uniform scale + center crop, no distortion).", flush=True)
        fitted = resize_to_cover_and_crop(img.convert("RGB"), gen_width, gen_height, resample)

    out_path = new_work_file("fitted")
    fitted.save(out_path)
    return str(out_path)
