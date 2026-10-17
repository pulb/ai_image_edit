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
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import DEFAULT_RESCALE_MODE, RescaleMode

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

# Masked edits are colour-corrected only when the model shifted the colours
# outside the mask, where nothing should change. The drift is the average
# CIE76 delta E the correction would apply there: below the first value
# nothing is corrected, from the second on the correction applies in full, and
# in between it is blended in linearly, so near-identical results don't jump.
#
# Delta E (the "E" is from German "Empfindung", sensation) is the perceived
# difference between two colours. CIE76 is its simplest form: the straight-line
# distance between them in L*a*b* space, with L on its standard 0-100 scale.
# Rough reading: up to about 1 is invisible, 2-3 is noticeable only side by
# side, 5 and more is clearly visible. Hence the ramp.
COLOR_DRIFT_RAMP = (2.0, 5.0)
# The longest side the drift is measured at: fast, and less sensitive to noise
# and to content the model shifted by a pixel or two.
COLOR_DRIFT_SAMPLE_SIDE = 512
# Pixels (at the sample size) around the mask left out of the measurement:
# models often alter a halo around the edit.
COLOR_DRIFT_MASK_MARGIN = 8
# The smallest share of the image outside the mask (and its margin) that
# gives a reliable measurement; with less, nothing is corrected.
COLOR_DRIFT_MIN_REFERENCE = 0.1

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

# Local colour matching at the mask's border. The model may tint the area
# around its edit (a wall turning grey next to the face) more than the rest of
# the image, which shows as a seam where the edit is blended into the
# original, though the global drift is small. So the generated image is
# compared with the original at the same pixels, in a band just inside the
# border: there the edit should still look like the original, and there the
# blend meets it. (Just outside, the generated image is barely used, so a tint
# only inside the mask would be missed.) All sizes are in pixels at the drift
# measurement's sample size.
#
# How far the band reaches into the mask. The comparison is made per square
# cell of the same size.
COLOR_LOCAL_BAND = 6
# The share of a cell that must be band pixels for the cell to count.
COLOR_LOCAL_MIN_COVERAGE = 0.2
# The ramp (see COLOR_DRIFT_RAMP) for a cell's difference. Far lower than the
# global one: where two colours meet at an edge, on a smooth surface like a
# wall, even a difference of about 1 shows, so a partly corrected border
# would still leave a visible seam. Correcting towards the original at the
# border is safe, so anything above that is corrected in full.
COLOR_LOCAL_RAMP = (0.5, 1.0)
# At least this many cells must have been compared.
COLOR_LOCAL_MIN_CELLS = 4
# A pixel that differs by more than this delta E holds edited content, not a
# tint, and is ignored.
COLOR_LOCAL_OUTLIER_DE = 12.0
# The offsets are spread over the edit by a Gaussian weighted by the compared
# cells; this is its sigma as a share of the mask's longer side. Small enough
# that each stretch of the border follows its own offset (the wall beside a
# face is tinted differently from the face), large enough to stay smooth.
COLOR_LOCAL_SIGMA = 0.1
# The most (in delta E, per L*, a* and b*) the local correction may change a colour.
COLOR_LOCAL_LIMIT_DE = 15.0

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
    src_arr = rgb_to_lab(np.array(source_img, dtype=np.float32)).reshape(-1, 3)
    tgt_arr = rgb_to_lab(np.array(target_img, dtype=np.float32)).reshape(-1, 3)
    return _correction_from_lab(src_arr, tgt_arr)


def _correction_from_lab(src_arr: np.ndarray, tgt_arr: np.ndarray) -> List[Tuple[float, float]]:
    """compute_color_correction() on (N, 3) arrays of LAB pixels."""
    ratio_lo, ratio_hi = COLOR_MATCH_RATIO_CLAMP
    corrections = []

    for i in range(3):
        src_ch = src_arr[:, i]
        tgt_ch = tgt_arr[:, i]

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


@dataclass(frozen=True)
class ColorMatch:
    """
    How to colour-match the generated image of a masked edit: the global
    per-channel (shift, scale) corrections, and a smooth field of L*a*b*
    offsets on a coarse grid over the whole generated image. Either may be None.
    """

    size: Tuple[int, int]
    corrections: Optional[List[Tuple[float, float]]] = None
    field: Optional[np.ndarray] = None


def _dilate(binary: np.ndarray, steps: int) -> np.ndarray:
    """binary (a boolean array) grown by `steps` pixels (square neighbourhood)."""
    img = Image.fromarray(binary.astype(np.uint8) * 255)
    for _ in range(steps):
        img = img.filter(ImageFilter.MaxFilter(3))
    return np.array(img) > 0


def _delta_e(lab_diff: np.ndarray) -> np.ndarray:
    """CIE76 delta E of (..., 3) L*a*b* differences, with L on the 0-255 scale of rgb_to_lab()."""
    scaled = np.array(lab_diff, dtype=np.float32)
    scaled[..., 0] /= _LAB_L_SCALE
    return np.linalg.norm(scaled, axis=-1)


def _samples(gen: Image.Image, orig: Image.Image, mask: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gen and orig as L*a*b* arrays and the mask as a boolean array, all scaled down to COLOR_DRIFT_SAMPLE_SIDE."""
    factor = min(1.0, COLOR_DRIFT_SAMPLE_SIDE / max(gen.size))
    size = (max(1, round(gen.width * factor)), max(1, round(gen.height * factor)))
    return (
        rgb_to_lab(np.array(gen.resize(size, Image.Resampling.BOX), dtype=np.float32)),
        rgb_to_lab(np.array(orig.resize(size, Image.Resampling.BOX), dtype=np.float32)),
        np.array(mask.resize(size, Image.Resampling.BOX)) > 0,
    )


def _global_correction(gen_lab: np.ndarray, orig_lab: np.ndarray, masked: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """The (shift, scale) per L*a*b* channel that undoes the drift outside the mask, scaled by its size, or None if none is needed."""
    reference = ~_dilate(masked, COLOR_DRIFT_MASK_MARGIN)
    share = float(reference.mean())
    if share < COLOR_DRIFT_MIN_REFERENCE:
        print(f"[color_correction] Only {share:.0%} of the image is outside the mask, too little to measure color drift; not correcting.", flush=True)
        return None

    gen_ref, orig_ref = gen_lab[reference], orig_lab[reference]
    corrections = _correction_from_lab(gen_ref, orig_ref)

    shift = np.array([c[0] for c in corrections], dtype=np.float32)
    scale = np.array([c[1] for c in corrections], dtype=np.float32)
    drift = float(_delta_e(gen_ref * (scale - 1) + shift).mean())

    low, high = COLOR_DRIFT_RAMP
    strength = min(1.0, max(0.0, (drift - low) / (high - low)))
    print(f"[color_correction] Color drift outside the mask: delta E {drift:.1f} (measured on {share:.0%} of the image), correcting at {strength:.0%}.", flush=True)
    if strength == 0:
        return None
    return [(s * strength, 1 + (r - 1) * strength) for s, r in corrections]


def _cell_average(values: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Averages a 2-D float array over the cells of a (width, height) grid."""
    return np.array(Image.fromarray(values.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX))


def _gaussian_matrix(n: int, sigma: float) -> np.ndarray:
    """An n x n matrix that blurs a length-n vector with a Gaussian."""
    idx = np.arange(n)
    return np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * sigma ** 2))


def _local_field(gen_lab: np.ndarray, orig_lab: np.ndarray, masked: np.ndarray) -> Optional[np.ndarray]:
    """The (grid height, grid width, 3) offsets that match the edit to the original along the mask's border, or None."""
    height, width = masked.shape
    diff = orig_lab - gen_lab
    # Pixels that changed a lot hold edited content (the glasses' frame), not a tint.
    band = masked & _dilate(~masked, COLOR_LOCAL_BAND) & (_delta_e(diff) <= COLOR_LOCAL_OUTLIER_DE)
    if not band.any():
        return None

    grid = (math.ceil(width / COLOR_LOCAL_BAND), math.ceil(height / COLOR_LOCAL_BAND))
    share = _cell_average(band, grid)
    offsets = np.stack([
        _cell_average(np.where(band, diff[:, :, i], 0), grid) / np.maximum(share, 1e-6) for i in range(3)
    ], axis=-1)

    cell_de = _delta_e(offsets)
    weight = np.where(share >= COLOR_LOCAL_MIN_COVERAGE, share, 0)
    cells = weight > 0
    if np.count_nonzero(cells) < COLOR_LOCAL_MIN_CELLS:
        print("[color_correction] Too little comparable border around the mask to measure a local color difference; not correcting locally.", flush=True)
        return None

    # The ramp applies per cell, so a tint along one stretch of the border is
    # corrected in full, not diluted by the stretches that are fine.
    low, high = COLOR_LOCAL_RAMP
    strength = np.clip((cell_de - low) / (high - low), 0, 1)
    active = int(np.count_nonzero(cells & (strength > 0)))
    print(
        f"[color_correction] Color difference along the mask border: delta E {float(cell_de[cells].mean()):.1f} on average, "
        f"{float(cell_de[cells].max()):.1f} at most; {active} of {np.count_nonzero(cells)} cells need correcting.",
        flush=True,
    )
    if active == 0:
        return None

    ys, xs = np.nonzero(masked)
    long_side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / COLOR_LOCAL_BAND
    sigma = max(1.5, COLOR_LOCAL_SIGMA * long_side)
    blur_y, blur_x = _gaussian_matrix(grid[1], sigma), _gaussian_matrix(grid[0], sigma)

    def blur(arr: np.ndarray) -> np.ndarray:
        return blur_y @ arr.astype(np.float64) @ blur_x.T

    norm = np.maximum(blur(weight), 1e-300)
    field = np.stack([blur(offsets[:, :, i] * strength * weight) / norm for i in range(3)], axis=-1)
    limit = np.array([COLOR_LOCAL_LIMIT_DE * _LAB_L_SCALE, COLOR_LOCAL_LIMIT_DE, COLOR_LOCAL_LIMIT_DE])
    return np.clip(field, -limit, limit).astype(np.float32)


def masked_color_match(gen: Image.Image, orig: Image.Image, mask: Image.Image) -> Optional[ColorMatch]:
    """
    How to colour-match the generated image of a masked edit, or None if it
    needs none; gen, orig and mask must have the same size. Two stages, each
    scaled by how far the colours drifted: a global correction computed only
    from the pixels outside the mask (and a margin around it), where the
    generated image should match the original, so the edit's own colours
    don't pull it off (see COLOR_DRIFT_RAMP); then a local one for what is
    left along the mask's border (see COLOR_LOCAL_BAND), ramped per stretch
    of the border.
    """
    gen_lab, orig_lab, masked = _samples(gen, orig, mask)
    corrections = _global_correction(gen_lab, orig_lab, masked)
    if corrections:
        for i, (shift, scale) in enumerate(corrections):
            gen_lab[:, :, i] = gen_lab[:, :, i] * scale + shift
    field = _local_field(gen_lab, orig_lab, masked)
    if corrections is None and field is None:
        return None
    return ColorMatch(gen.size, corrections, field)


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


def apply_color_match(img: Image.Image, match: ColorMatch, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Applies a ColorMatch to img, the part of the generated image at `box`
    (left, top, right, bottom, in generated-image pixels): the global
    corrections as in apply_color_correction(), then the local offsets,
    interpolated to img's pixels.
    """
    arr = rgb_to_lab(np.array(img, dtype=np.float32))
    for i, (shift, scale) in enumerate(match.corrections or []):
        arr[:, :, i] = arr[:, :, i] * scale + shift
    if match.field is not None:
        grid_h, grid_w = match.field.shape[:2]
        sx, sy = grid_w / match.size[0], grid_h / match.size[1]
        source = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        for i in range(3):
            channel = Image.fromarray(match.field[:, :, i], mode="F")
            arr[:, :, i] += np.array(channel.resize(img.size, Image.Resampling.BILINEAR, box=source))
    return Image.fromarray(lab_to_rgb(arr).astype(np.uint8), mode="RGB")


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
    low, high = DETAIL_ENLARGEMENT_RAMP
    strength = min(1.0, max(0.0, (enlargement - low) / (high - low)))
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
    gen_crop = _match_detail(gen_crop, orig, box, patch_size[0] / (src[2] - src[0]))

    blended = _blend(gen_crop, orig.crop(box), mask.crop(box), feather_amount / scale_x)
    orig.paste(blended, box[:2])
    return orig


def composite_with_soft_transition(
    orig_img_path: str,
    gen_img_path: str,
    mask_img_path: Optional[str],
    feather_amount: int,
    rescale_mode: RescaleMode = DEFAULT_RESCALE_MODE,
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
    rescale_mode: RescaleMode = DEFAULT_RESCALE_MODE,
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
      apply_color_correction_enabled is True (with a mask that flag has no
      effect: composite_with_soft_transition() matches colours by itself). gen_width/gen_height and
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
