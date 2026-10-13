# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pixel-level helpers for the crop/mask/composite steps around
inference: LAB color-space color matching,
uniform-scale cover-crop resizing, the mask-feathered composite that
merges a generated result back into the original image, and
run_masked_generation(), which orchestrates the crop -> infer -> composite
sequence around an inference callback.
"""
import math
import uuid
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import RescaleMode

# Minimum standard deviation (0-255 scale, matching PIL's LAB representation)
# a channel needs before we trust it enough to rescale by a std-dev ratio.
# Below this, we only shift the mean — rescaling by ratio when src_std is
# tiny can blow up small noise into huge swings (e.g. tgt_std/src_std =
# 45/0.8 =~ 56x), which clips at 0/255 and shows up as blown-out / wrong
# colors in the pasted region. Applies to L, a, and b alike; a/b
# (chrominance) channels are typically lower-variance than L in natural
# photos, so they'll fall back to a mean-only shift more often — that's
# expected and conservative, not a sign anything's wrong.
COLOR_MATCH_STD_EPS = 4.0
# Clamp on how aggressively we're allowed to rescale variance, for the same reason.
COLOR_MATCH_RATIO_CLAMP = (0.5, 2.0)

# sRGB (D65) <-> CIE XYZ matrices, used by rgb_to_lab()/lab_to_rgb() for the
# LAB color-space conversion compute_color_correction()/apply_color_correction()
# do their work in.
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

# D65 reference white.
_LAB_XN, _LAB_YN, _LAB_ZN = 0.95047, 1.00000, 1.08883
_LAB_DELTA = 6 / 29
# Scales L from its standard 0-100 range to 0-255, matching how PIL's "LAB"
# image mode scales L — so COLOR_MATCH_STD_EPS stays valid without retuning.
_LAB_L_SCALE = 2.55


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Inverse sRGB gamma (companding). c is 0-1 normalized sRGB; returns linear-light RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Forward sRGB gamma (companding). c is 0-1 linear-light RGB; returns 0-1 sRGB."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """
    Converts an (H, W, 3) array of 0-255 sRGB values to CIE L*a*b*, in float
    (no 8-bit quantization round-trip, unlike PIL's built-in "LAB" mode).
    a/b keep their real signed values, unlike PIL's mode which stores
    negative a/b via byte wraparound. L is scaled to 0-255 to match PIL's
    convention, so COLOR_MATCH_STD_EPS stays meaningful for L.
    """
    rgb = arr.astype(np.float64) / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _RGB_TO_XYZ.T

    x = xyz[:, :, 0] / _LAB_XN
    y = xyz[:, :, 1] / _LAB_YN
    z = xyz[:, :, 2] / _LAB_ZN

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > _LAB_DELTA ** 3, np.cbrt(t), t / (3 * _LAB_DELTA ** 2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)

    L = (116 * fy - 16) * _LAB_L_SCALE
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab() — converts back to 0-255 sRGB, in float. See its docstring for the scale convention."""
    L = arr[:, :, 0].astype(np.float64) / _LAB_L_SCALE
    a = arr[:, :, 1].astype(np.float64)
    b = arr[:, :, 2].astype(np.float64)

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > _LAB_DELTA, t ** 3, 3 * _LAB_DELTA ** 2 * (t - 4 / 29))

    x = _LAB_XN * finv(fx)
    y = _LAB_YN * finv(fy)
    z = _LAB_ZN * finv(fz)

    xyz = np.stack([x, y, z], axis=-1)
    linear = xyz @ _XYZ_TO_RGB.T
    srgb = _linear_to_srgb(linear)

    return np.clip(srgb * 255.0, 0, 255).astype(np.float32)


def compute_color_correction(source_img: Image.Image, target_img: Image.Image) -> List[Tuple[float, float]]:
    """
    Computes a per-channel (shift, scale) correction that maps source_img's
    color statistics onto target_img's, in LAB space (separates luminance
    from color, so a brightness fix can't bleed into a color-cast fix).
    Returns one (shift, scale) tuple per L/a/b channel; apply with
    apply_color_correction().

    Always call with the FULL images, not a masked-down crop — a small
    region gives an unstable statistical estimate.
    """
    src_arr = rgb_to_lab(np.array(source_img, dtype=np.float32))
    tgt_arr = rgb_to_lab(np.array(target_img, dtype=np.float32))

    ratio_lo, ratio_hi = COLOR_MATCH_RATIO_CLAMP
    corrections = []

    for i in range(3):
        src_ch = src_arr[:, :, i]
        tgt_ch = tgt_arr[:, :, i]

        src_mean, src_std = src_ch.mean(), src_ch.std()
        tgt_mean, tgt_std = tgt_ch.mean(), tgt_ch.std()

        if src_std < COLOR_MATCH_STD_EPS:
            # Not enough variance to safely rescale by ratio — just shift the mean
            # so we still correct color cast without amplifying noise.
            corrections.append((tgt_mean - src_mean, 1.0))
        else:
            ratio = np.clip(tgt_std / src_std, ratio_lo, ratio_hi)
            corrections.append((tgt_mean - src_mean * ratio, ratio))

    return corrections


def apply_color_correction(img: Image.Image, corrections: List[Tuple[float, float]]) -> Image.Image:
    """
    Applies the per-channel (shift, scale) corrections from
    compute_color_correction() — converts img to the same LAB representation
    those corrections were computed in (see rgb_to_lab()), applies them, and
    converts back to RGB.
    """
    arr = rgb_to_lab(np.array(img, dtype=np.float32))
    for i, (shift, scale) in enumerate(corrections):
        arr[:, :, i] = arr[:, :, i] * scale + shift
    rgb = lab_to_rgb(arr)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def apply_exif_orientation(image_path: str) -> None:
    """
    Rotates/flips the image file in place if its EXIF orientation says so, so
    the stored pixels are upright. Browsers show such images upright while
    PIL (and a model fed raw pixels) sees them sideways, which misaligns
    anything drawn in the browser, such as the mask. Files without an
    orientation, or an upright one, are left untouched.
    """
    with Image.open(image_path) as img:
        if img.getexif().get(0x0112, 1) == 1:
            return
        fmt = img.format
        upright = ImageOps.exif_transpose(img)
    options = {"quality": 95} if fmt in ("JPEG", "WEBP") else {}
    upright.save(image_path, format=fmt, **options)


def compose_annotations(source_image_path: str, layer_path: str) -> str:
    """
    Draws the transparent annotation layer onto the source image and saves
    the result as a new PNG in WORK_DIR, returning its path. The layer is
    scaled to the source's size if it differs.
    """
    with Image.open(source_image_path) as source, Image.open(layer_path) as layer:
        base = source.convert("RGBA")
        layer = layer.convert("RGBA")
        if layer.size != base.size:
            print(f"[compose_annotations] Scaling annotation layer {layer.size} to {base.size}.", flush=True)
            layer = layer.resize(base.size, Image.Resampling.BILINEAR)
        composed = Image.alpha_composite(base, layer)
        if "A" not in source.getbands():
            composed = composed.convert("RGB")
    out_path = WORK_DIR / f"annotated_{uuid.uuid4().hex}.png"
    composed.save(out_path)
    return str(out_path)


def resize_to_cover_and_crop(image: Image.Image, target_w: int, target_h: int, resample: int) -> Image.Image:
    """
    Fits image to exactly (target_w, target_h) without distorting it: scales
    uniformly (same factor on both axes) until it covers the target box,
    then center-crops the overhang — CSS object-fit: cover. No-op (returns
    image unchanged) if already that exact size.
    """
    if image.size == (target_w, target_h):
        return image

    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    # ceil (not round) guarantees the scaled image covers the target box on
    # both axes even after float rounding — round() could occasionally land
    # a pixel short, which crop() below would then fail on.
    covered_w = max(target_w, math.ceil(src_w * scale))
    covered_h = max(target_h, math.ceil(src_h * scale))

    resized = image.resize((covered_w, covered_h), resample)
    left = (covered_w - target_w) // 2
    top = (covered_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def prepare_inpaint_image(image_path: str, gen_width: int, gen_height: int, resample: int) -> str:
    """
    Resizes image_path (via resize_to_cover_and_crop) to exactly
    (gen_width, gen_height) and returns the new file's path, or image_path
    itself unchanged if it's already that size. Used to align inpaint
    mode's source image with the model's generation canvas before
    inference; composite_with_soft_transition() applies the same transform
    to the original and mask at blend time, keeping all three in the same
    coordinate space.
    """
    with Image.open(image_path) as img:
        if img.size == (gen_width, gen_height):
            return image_path
        print(f"[prepare_inpaint_image] Resizing {img.size} to {(gen_width, gen_height)} (uniform scale + center crop, no distortion).", flush=True)
        prepared = resize_to_cover_and_crop(img.convert("RGB"), gen_width, gen_height, resample)

    out_path = WORK_DIR / f"prepared_{uuid.uuid4()}.png"
    prepared.save(out_path)
    return str(out_path)


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

    # Calibrate the color correction from the whole of both (now
    # same-resolution) images — see compute_color_correction() docstring.
    # Unaffected by feathering: color matching and edge softening are
    # separate concerns, so this still runs either way.
    corrections = compute_color_correction(gen, orig)

    crop_box = _padded_box(bbox, _feather_padding(feather_amount), (0, 0, gen_size[0], gen_size[1]))
    blended = _blend(
        apply_color_correction(gen.crop(crop_box), corrections),
        orig.crop(crop_box),
        mask.crop(crop_box),
        feather_amount,
    )
    orig.paste(blended, crop_box[:2])
    return orig


def _composite_at_original_size(orig: Image.Image, gen: Image.Image, mask: Image.Image, feather_amount: int) -> Optional[Image.Image]:
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

    # Where the generated image sits on the original: prepare_inpaint_image
    # scaled the original to cover the generated size, then center-cropped
    # the overhang (see resize_to_cover_and_crop). Integer arithmetic keeps
    # the covered region exact.
    scale = max(gen_w / orig_w, gen_h / orig_h)
    covered_w = max(gen_w, math.ceil(orig_w * scale))
    covered_h = max(gen_h, math.ceil(orig_h * scale))
    left, top = (covered_w - gen_w) // 2, (covered_h - gen_h) // 2
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

    # Colour matching works at the generated size, on the same statistics as
    # the generated-size composite, and is applied before resampling.
    orig_at_gen_size = orig
    if orig.size != gen.size:
        print(f"[composite_with_soft_transition] Fitting original {orig.size} to {gen.size} for color matching (uniform scale + center crop).", flush=True)
        orig_at_gen_size = resize_to_cover_and_crop(orig, gen_w, gen_h, Image.Resampling.LANCZOS)
    corrections = compute_color_correction(gen, orig_at_gen_size)
    x0, y0, x1, y1 = math.floor(src[0]), math.floor(src[1]), math.ceil(src[2]), math.ceil(src[3])
    gen_region = apply_color_correction(gen.crop((x0, y0, x1, y1)), corrections)
    patch_size = (box[2] - box[0], box[3] - box[1])
    print(f"[composite_with_soft_transition] Resampling generated area {(x1 - x0, y1 - y0)} to {patch_size} for the original-size result.", flush=True)
    gen_crop = gen_region.resize(
        patch_size,
        Image.Resampling.LANCZOS,
        box=(src[0] - x0, src[1] - y0, src[2] - x0, src[3] - y0),
    )

    blended = _blend(gen_crop, orig.crop(box), mask.crop(box), feather_amount / scale_x)
    orig.paste(blended, box[:2])
    return orig


def composite_with_soft_transition(
    orig_img_path: str,
    gen_img_path: str,
    mask_img_path: Optional[str],
    feather_amount: int,
    rescale_mode: RescaleMode = RescaleMode.ORIGINAL,
) -> str:
    """
    Color-matches the generated image to the original (full frame), then
    alpha-blends just the region around the mask's bounding box back into
    the original — Gaussian-feathered if feather_amount > 0, else a hard
    0/255 cutout. Returns gen_img_path if there is no mask or nothing is
    masked.

    RescaleMode.ORIGINAL: if resolutions differ, the original (and mask) are
    resized to match the generated image via resize_to_cover_and_crop, never
    the other way around, since upscaling the generated image would soften
    its detail. The result has the generated image's size.

    RescaleMode.EDIT: the result has the original's size and keeps all its
    pixels outside the blended region; the generated image is resampled into
    that region, with the feather scaled to match.
    """
    if not mask_img_path:
        return gen_img_path

    with Image.open(gen_img_path) as gen_file, Image.open(orig_img_path) as orig_file, Image.open(mask_img_path) as mask_file:
        gen = gen_file.convert("RGB")
        orig = orig_file.convert("RGB")
        mask = mask_file.getchannel('A') if mask_file.mode == 'RGBA' else mask_file.convert("L")

    composite = _composite_at_original_size if rescale_mode == RescaleMode.EDIT else _composite_at_generated_size
    result = composite(orig, gen, mask, feather_amount)
    if result is None:
        return gen_img_path

    output_path = WORK_DIR / f"composited_{uuid.uuid4()}.png"
    result.save(output_path)
    return str(output_path)


def apply_whole_image_color_correction(orig_img_path: str, gen_img_path: str) -> str:
    """
    Color-matches the whole generated image to the whole original — every
    pixel gets the same per-channel correction, no mask or crop. The
    no-mask counterpart to composite_with_soft_transition()'s masked blend,
    used when the "Apply color corrections" toggle is on. orig and gen
    don't need matching resolutions here, since correction only uses each
    image's aggregate per-channel mean/std.
    """
    with Image.open(gen_img_path) as gen_file, Image.open(orig_img_path) as orig_file:
        gen = gen_file.convert("RGB")
        orig = orig_file.convert("RGB")

    corrections = compute_color_correction(gen, orig)
    corrected = apply_color_correction(gen, corrections)

    output_path = WORK_DIR / f"color_corrected_{uuid.uuid4()}.png"
    corrected.save(output_path)
    return str(output_path)


def run_masked_generation(
    source_image_path: str,
    mask_path: Optional[str],
    gen_width: Optional[int],
    gen_height: Optional[int],
    feather_amount: int,
    apply_color_correction_enabled: bool,
    infer: Callable[[str], str],
    annotated_image_path: Optional[str] = None,
    rescale_mode: RescaleMode = RescaleMode.ORIGINAL,
) -> str:
    """
    Orchestrates the crop -> infer -> composite/color-correct sequence
    around inference. `infer` is the only model-specific piece: a callback
    taking the (possibly cropped) source path and returning the path to the
    raw generated image.

    - If mask_path is set: the model's input image is cropped/resized to
      exactly (gen_width, gen_height) via prepare_inpaint_image before
      infer(), then the result is composited back into the untouched
      original via composite_with_soft_transition, feathered by
      feather_amount, as rescale_mode selects. gen_width/gen_height
      must be real dimensions in this case.
    - Otherwise: the model's input image is passed to infer() unprepared,
      and the raw result is returned as-is, or color-corrected as a whole if
      apply_color_correction_enabled is True. gen_width/gen_height and
      rescale_mode are ignored — infer() handles its own dimension needs.

    The model's input image is annotated_image_path if given, else
    source_image_path. Compositing and color correction always use the clean
    source_image_path, so annotations never end up in the result.
    """
    model_source_path = annotated_image_path or source_image_path
    model_input_path = (
        prepare_inpaint_image(model_source_path, gen_width, gen_height, Image.Resampling.LANCZOS)
        if mask_path
        else model_source_path
    )

    output_path = infer(model_input_path)

    if mask_path:
        return composite_with_soft_transition(source_image_path, output_path, mask_path, feather_amount, rescale_mode)
    if apply_color_correction_enabled:
        return apply_whole_image_color_correction(source_image_path, output_path)
    return output_path
