# SPDX-License-Identifier: GPL-3.0-or-later
"""
The mask-feathered composite that merges a generated result back into the
original (composite_with_soft_transition()), with its detail matching, and the
whole-image colour correction of an unmasked edit.
"""
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ai_image_edit.core.imaging.color import apply_color_match, masked_color_match, ramp, whole_image_color_match
from ai_image_edit.core.imaging.files import read_mask
from ai_image_edit.core.imaging.geometry import cover_geometry, resize_to_cover_and_crop
from ai_image_edit.core.paths import new_work_file
from ai_image_edit.core.types import RescaleMode

# Detail matching for an enlarged patch (RescaleMode.EDIT with an original
# larger than the generated image). The enlarged patch is smoother than the
# photo around it, so it is sharpened a little and given grain at the level
# of the original's. Below this enlargement the patch is left alone; the
# effects reach full strength at the second value.
DETAIL_ENLARGEMENT_RAMP = (1.1, 1.5)
# Unsharp mask amount at full strength, and its radius per unit of enlargement.
DETAIL_SHARPEN_AMOUNT = 0.4
DETAIL_SHARPEN_RADIUS = 0.7
# The noise level is measured in square blocks of this size, on the quietest
# quarter of them, so texture and edges don't count as noise.
DETAIL_NOISE_BLOCK = 16


def _feather_padding(feather_amount: int) -> int:
    """
    The margin, in generated-image pixels, added around the mask's bounding
    box before cropping. It gives the blurred mask room to fade to 0 before
    the crop boundary; a still partly opaque mask there would show as a seam
    where the pasted crop meets the untouched original. The margin costs
    nothing visually: its pixels have mask value 0 and are pasted back
    unchanged. A hard mask is already 0 at the bounding box's own edge, so it
    needs none. 8x the blur radius matches the 64 px a radius-8 blur needed.
    """
    return feather_amount * 8 if feather_amount > 0 else 0


def _padded_box(bbox: Tuple[int, int, int, int], pad: int, bounds: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """bbox grown by pad on every side, clamped to bounds (all as left, top, right, bottom)."""
    return (
        max(bounds[0], bbox[0] - pad),
        max(bounds[1], bbox[1] - pad),
        min(bounds[2], bbox[2] + pad),
        min(bounds[3], bbox[3] + pad),
    )


def _blend(gen_crop: Image.Image, orig_crop: Image.Image, mask_crop: Image.Image, blur_radius: float) -> Image.Image:
    """
    Alpha-blends gen_crop over orig_crop through mask_crop, Gaussian-blurred
    if blur_radius > 0. Without blur the mask is used as it is: the canvas
    draws a hard 0/255 cutout.
    """
    if blur_radius > 0:
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(gen_crop, orig_crop, mask_crop)


def _composite_at_generated_size(orig: Image.Image, gen: Image.Image, mask: Image.Image, feather_amount: int) -> Optional[Image.Image]:
    """The result at the generated image's size, or None if nothing is masked."""
    gen_size = gen.size
    if orig.size != gen_size:
        print(f"[composite_with_soft_transition] Fitting original {orig.size} to {gen_size} for blending (uniform scale + center crop).", flush=True)
        orig = resize_to_cover_and_crop(orig, gen_size[0], gen_size[1], Image.Resampling.LANCZOS)

    if mask.size != gen_size:
        print(f"[composite_with_soft_transition] Fitting mask {mask.size} to {gen_size} for feathering (uniform scale + center crop).", flush=True)
        mask = resize_to_cover_and_crop(mask, gen_size[0], gen_size[1], Image.Resampling.NEAREST)

    bbox = mask.getbbox()
    if not bbox:
        return None

    match = masked_color_match(gen, orig, mask)

    crop_box = _padded_box(bbox, _feather_padding(feather_amount), (0, 0, gen_size[0], gen_size[1]))
    gen_crop = gen.crop(crop_box)
    if match:
        gen_crop = apply_color_match(gen_crop, match, crop_box)
    blended = _blend(
        gen_crop,
        orig.crop(crop_box),
        mask.crop(crop_box),
        feather_amount,
    )
    orig.paste(blended, crop_box[:2])
    return orig


def _luma(img: Image.Image) -> np.ndarray:
    """The luma of an RGB image as a float array."""
    return np.asarray(img, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _noise_level(img: Image.Image) -> Optional[float]:
    """
    The level of an image's fine detail (noise and grain): its difference from
    a slightly blurred copy, as RMS per block, in the quietest quarter of the
    blocks (see DETAIL_NOISE_BLOCK). None if the image is smaller than a block.
    """
    detail = _luma(img) - _luma(img.filter(ImageFilter.GaussianBlur(1.5)))
    block = DETAIL_NOISE_BLOCK
    rows, cols = detail.shape[0] // block, detail.shape[1] // block
    if not rows or not cols:
        return None
    blocks = detail[:rows * block, :cols * block].reshape(rows, block, cols, block)
    return float(np.percentile(np.sqrt((blocks ** 2).mean(axis=(1, 3))), 25))


def _match_detail(patch: Image.Image, orig: Image.Image, box: Tuple[int, int, int, int], enlargement: float) -> Image.Image:
    """
    Sharpens a patch that was enlarged by `enlargement` a little, and adds grain
    until its fine detail is as strong as the original's at `box` (the patch's
    place in orig; its pixels there are still the untouched photo). Both scale
    with DETAIL_ENLARGEMENT_RAMP; grain is only added if the patch is smoother.
    """
    strength = ramp(enlargement, DETAIL_ENLARGEMENT_RAMP)
    if strength == 0:
        return patch

    amount = DETAIL_SHARPEN_AMOUNT * strength
    patch = patch.filter(ImageFilter.UnsharpMask(radius=DETAIL_SHARPEN_RADIUS * enlargement, percent=round(amount * 100), threshold=0))

    original_level, patch_level = _noise_level(orig.crop(box)), _noise_level(patch)
    if original_level is None or patch_level is None or original_level <= patch_level:
        print(f"[composite_with_soft_transition] Enlarged {enlargement:.2f}x: sharpened at {amount:.2f}, no grain needed.", flush=True)
        return patch

    # Grain with a camera sensor's fine correlation (a 1-2-1 blur), scaled by
    # its own measured level, as the measure sees correlated noise as weaker
    # than its standard deviation. Fixed seed: the same image gets the same grain.
    noise = np.random.default_rng(0).normal(0, 1, (patch.height + 2, patch.width + 2)).astype(np.float32)
    noise = (noise[:-2] + 2 * noise[1:-1] + noise[2:]) / 4
    noise = (noise[:, :-2] + 2 * noise[:, 1:-1] + noise[:, 2:]) / 4
    probe = Image.fromarray(np.clip(np.rint(128 + 10 * noise), 0, 255).astype(np.uint8)).convert("RGB")
    noise *= math.sqrt(original_level ** 2 - patch_level ** 2) / (_noise_level(probe) / 10)
    print(
        f"[composite_with_soft_transition] Enlarged {enlargement:.2f}x: sharpened at {amount:.2f}, "
        f"adding grain (fine detail: original {original_level:.1f}, generated {patch_level:.1f}).",
        flush=True,
    )
    grainy = np.asarray(patch, dtype=np.float32) + noise[:, :, None]
    return Image.fromarray(np.clip(np.rint(grainy), 0, 255).astype(np.uint8), mode="RGB")


def _composite_at_original_size(
    orig: Image.Image, gen: Image.Image, mask: Image.Image, feather_amount: int, match_detail: bool = True
) -> Optional[Image.Image]:
    """
    The original with the blended region resampled from the generated image;
    None if nothing is masked. The original's other pixels are not touched.
    """
    orig_w, orig_h = orig.size
    gen_w, gen_h = gen.size

    if mask.size != orig.size:
        print(f"[composite_with_soft_transition] Fitting mask {mask.size} to {orig.size} for feathering (uniform scale + center crop).", flush=True)
        mask = resize_to_cover_and_crop(mask, orig_w, orig_h, Image.Resampling.NEAREST)

    bbox = mask.getbbox()
    if not bbox:
        return None

    # Where the generated image sits on the original: fit_image_to_size
    # scaled the original to cover the generated size, then center-cropped
    # the overhang (see resize_to_cover_and_crop). Integer arithmetic keeps
    # the covered region exact.
    covered_w, covered_h, left, top = cover_geometry(orig.size, gen_w, gen_h)
    scale_x, scale_y = covered_w / orig_w, covered_h / orig_h
    covered = (
        -(-left * orig_w // covered_w),
        -(-top * orig_h // covered_h),
        (left + gen_w) * orig_w // covered_w,
        (top + gen_h) * orig_h // covered_h,
    )

    pad = math.ceil(_feather_padding(feather_amount) / scale_x)
    box = _padded_box(bbox, pad, covered)
    if box[0] >= box[2] or box[1] >= box[3]:
        return None

    # The box in generated-image pixels (fractional), clamped against rounding.
    src = (
        max(0.0, box[0] * scale_x - left),
        max(0.0, box[1] * scale_y - top),
        min(float(gen_w), box[2] * scale_x - left),
        min(float(gen_h), box[3] * scale_y - top),
    )
    if src[0] >= src[2] or src[1] >= src[3]:
        return None

    # Colour matching works at the generated size, like the generated-size
    # composite, and is applied before resampling.
    match = masked_color_match(
        gen,
        resize_to_cover_and_crop(orig, gen_w, gen_h, Image.Resampling.LANCZOS),
        resize_to_cover_and_crop(mask, gen_w, gen_h, Image.Resampling.BOX),
    )
    x0, y0, x1, y1 = math.floor(src[0]), math.floor(src[1]), math.ceil(src[2]), math.ceil(src[3])
    gen_region = gen.crop((x0, y0, x1, y1))
    if match:
        gen_region = apply_color_match(gen_region, match, (x0, y0, x1, y1))
    patch_size = (box[2] - box[0], box[3] - box[1])
    if src == (x0, y0, x1, y1) and patch_size == gen_region.size:
        gen_crop = gen_region  # same pixels at the same scale: nothing to resample
    else:
        print(f"[composite_with_soft_transition] Resampling generated area {(x1 - x0, y1 - y0)} to {patch_size} for the original-size result.", flush=True)
        gen_crop = gen_region.resize(
            patch_size,
            Image.Resampling.LANCZOS,
            box=(src[0] - x0, src[1] - y0, src[2] - x0, src[3] - y0),
        )
    if match_detail:
        gen_crop = _match_detail(gen_crop, orig, box, patch_size[0] / (src[2] - src[0]))

    blended = _blend(gen_crop, orig.crop(box), mask.crop(box), feather_amount / scale_x)
    orig.paste(blended, box[:2])
    return orig


def composite_with_soft_transition(
    orig_img_path: str,
    gen_img_path: str,
    mask_img_path: Optional[str],
    feather_amount: int,
    rescale_mode: RescaleMode,
    match_detail: bool = True,
) -> str:
    """
    Color-matches the generated image to the original where its colours
    drifted (see masked_color_match()), then alpha-blends just the region
    around the mask's bounding box back into the original — Gaussian-feathered
    if feather_amount > 0, else a hard 0/255 cutout. Returns gen_img_path if
    there is no mask or nothing is masked.

    RescaleMode.ORIGINAL: if resolutions differ, the original (and mask) are
    resized to match the generated image via resize_to_cover_and_crop, never
    the other way around, since upscaling the generated image would soften
    its detail. The result has the generated image's size.

    RescaleMode.EDIT: the result has the original's size and keeps all its
    pixels outside the blended region; the generated image is resampled into
    that region, with the feather scaled to match. If the area had to be
    enlarged, match_detail sharpens it and adds grain to fit the original.
    """
    if not mask_img_path:
        return gen_img_path

    with Image.open(gen_img_path) as gen_file, Image.open(orig_img_path) as orig_file:
        gen = gen_file.convert("RGB")
        orig = orig_file.convert("RGB")
    mask = read_mask(mask_img_path)

    if rescale_mode == RescaleMode.EDIT:
        result = _composite_at_original_size(orig, gen, mask, feather_amount, match_detail)
    else:
        result = _composite_at_generated_size(orig, gen, mask, feather_amount)
    if result is None:
        return gen_img_path

    output_path = new_work_file("composited")
    result.save(output_path)
    return str(output_path)


def apply_whole_image_color_correction(orig_img_path: str, gen_img_path: str) -> str:
    """
    Color-matches the whole generated image to the original (see
    whole_image_color_match(); gen must show orig's framing) and returns the
    corrected file's path, or gen_img_path if nothing was corrected. The no-mask
    counterpart to composite_with_soft_transition()'s masked blend, used when
    the "Apply color corrections" toggle is on. orig and gen may have
    different sizes.
    """
    with Image.open(gen_img_path) as gen_file, Image.open(orig_img_path) as orig_file:
        gen = gen_file.convert("RGB")
        orig = orig_file.convert("RGB")

    match = whole_image_color_match(gen, orig)
    if match is None:
        return gen_img_path
    corrected = apply_color_match(gen, match, (0, 0, gen.width, gen.height))

    output_path = new_work_file("color_corrected")
    corrected.save(output_path)
    return str(output_path)
