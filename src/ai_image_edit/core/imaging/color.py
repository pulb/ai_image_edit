# SPDX-License-Identifier: GPL-3.0-or-later
"""
Colour matching in CIE L*a*b*: the masked edit's global and local correction
(masked_color_match()), the unmasked edit's (whole_image_color_match()), and
applying either (apply_color_match()).
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ai_image_edit.core.imaging.geometry import resize_to_cover_and_crop

# Minimum standard deviation per channel (L*, a*, b*; for L* the equivalent of
# 4 on the 0-255 scale of PIL's LAB mode, which it was tuned on)
# a channel needs before we trust it enough to rescale by a std-dev ratio.
# Below this, we only shift the mean — rescaling by ratio when src_std is
# tiny can blow up small noise into huge swings (e.g. tgt_std/src_std =
# 45/0.8 =~ 56x), which clips at 0/255 and shows up as blown-out / wrong
# colors in the corrected image. Applies to L, a, and b alike; a/b
# (chrominance) channels are typically lower-variance than L in natural
# photos, so they'll fall back to a mean-only shift more often — that's
# expected and conservative, not a sign anything's wrong.
COLOR_MATCH_STD_EPS = (4.0 / 2.55, 4.0, 4.0)
# Clamp on how aggressively we're allowed to rescale variance, for the same reason.
COLOR_MATCH_RATIO_CLAMP = (0.5, 2.0)

# Masked edits are colour-corrected only when the model shifted the colours
# outside the mask, where nothing should change (unmasked ones likewise
# outside what the edit changed, see whole_image_color_match()). The drift is the average
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
# The smallest share of the image outside the mask (and its margin), or
# outside what an unmasked edit changed, that gives a reliable measurement;
# with less, nothing is corrected.
COLOR_DRIFT_MIN_REFERENCE = 0.1

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

# Colour matching of an unmasked edit (whole_image_color_match()), when the
# "Apply color corrections" switch is on: the global correction made outside a
# mask, with its ramp (COLOR_DRIFT_RAMP), measured on every pixel the edit did
# not change. Pixels beyond the outlier limit (COLOR_LOCAL_OUTLIER_DE) hold
# what the edit changed and are left out, as a mask's area is, so a recoloured
# object doesn't tint the rest of the image; if that leaves too little
# (COLOR_DRIFT_MIN_REFERENCE), the edit changed nearly all colours (a night
# scene, a restyle) and nothing is corrected.
#
# It is a safety net for a model that clearly tints or dulls the whole image.
# In edits of a selfie and a sandwich (October 2026) the model's own drift was
# a delta E of 1.2 to 1.3 over the whole image, 1.3 to 1.8 on average in
# 32-pixel cells (at COLOR_DRIFT_SAMPLE_SIDE) and 3 to 4 at most: below the
# ramp, and not visible without a seam to compare it against, unlike along a
# mask's border. A local correction as there, made across the whole image,
# made no visible difference to the selfie, while correcting such small
# differences pulls back colour changes asked for.
#
# It tells changed pixels from the rest pixel by pixel, so it needs the output
# to show the input's framing (the same aspect ratio; see run_generation()).
# A slight misalignment only drops edge pixels from the measurement. A
# recomposed output (another aspect ratio) is not corrected: its colours differ
# because its content does (more wall, less sky), which its statistics can't
# tell from a drift.

# sRGB (D65) <-> CIE XYZ matrices, used by rgb_to_lab()/lab_to_rgb() for the
# LAB color-space conversion the colour matching does its work in.
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

# D65 reference white.
_LAB_XN, _LAB_YN, _LAB_ZN = 0.95047, 1.00000, 1.08883
_LAB_DELTA = 6 / 29


def ramp(value, bounds: Tuple[float, float]):
    """0 up to bounds[0], 1 from bounds[1] on, linear in between; works on numbers and arrays."""
    low, high = bounds
    if np.ndim(value) == 0:
        return min(1.0, max(0.0, (float(value) - low) / (high - low)))
    return np.clip((value - low) / (high - low), 0, 1)


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
    L is on its standard 0-100 scale and a/b keep their real signed values,
    unlike PIL's mode, which scales L to 0-255 and stores negative a/b via
    byte wraparound.
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

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab() — converts back to 0-255 sRGB, in float."""
    L = arr[:, :, 0].astype(np.float64)
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


def _correction_from_lab(src_arr: np.ndarray, tgt_arr: np.ndarray) -> List[Tuple[float, float]]:
    """
    A per-channel (shift, scale) correction that maps the colour statistics
    (mean and spread) of src_arr's L*a*b* pixels, an (N, 3) array, onto
    tgt_arr's, one (shift, scale) per channel. In L*a*b*, a brightness fix
    can't bleed into a colour-cast fix. Needs many pixels: a small sample
    gives an unstable estimate.
    """
    ratio_lo, ratio_hi = COLOR_MATCH_RATIO_CLAMP
    corrections = []

    for i in range(3):
        src_ch = src_arr[:, i]
        tgt_ch = tgt_arr[:, i]

        src_mean, src_std = src_ch.mean(), src_ch.std()
        tgt_mean, tgt_std = tgt_ch.mean(), tgt_ch.std()

        if src_std < COLOR_MATCH_STD_EPS[i]:
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
    How to colour-match the generated image of an edit (masked_color_match(),
    whole_image_color_match()): the global per-channel (shift, scale)
    corrections, and, for a masked edit, a smooth field of L*a*b* offsets on a
    coarse grid over the whole generated image. Either may be None.
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
    """CIE76 delta E of (..., 3) L*a*b* differences."""
    return np.linalg.norm(np.asarray(lab_diff, dtype=np.float32), axis=-1)


def _samples(gen: Image.Image, orig: Image.Image, mask: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gen and orig as L*a*b* arrays and the mask as a boolean array, all scaled down to COLOR_DRIFT_SAMPLE_SIDE."""
    factor = min(1.0, COLOR_DRIFT_SAMPLE_SIDE / max(gen.size))
    size = (max(1, round(gen.width * factor)), max(1, round(gen.height * factor)))
    return (
        rgb_to_lab(np.array(gen.resize(size, Image.Resampling.BOX), dtype=np.float32)),
        rgb_to_lab(np.array(orig.resize(size, Image.Resampling.BOX), dtype=np.float32)),
        np.array(mask.resize(size, Image.Resampling.BOX)) > 0,
    )


def _global_correction(
    gen_lab: np.ndarray, orig_lab: np.ndarray, masked: np.ndarray, where: str = "outside the mask",
    margin: int = COLOR_DRIFT_MASK_MARGIN,
) -> Optional[List[Tuple[float, float]]]:
    """
    The (shift, scale) per L*a*b* channel that undoes the drift outside
    `masked` (and `margin` pixels around it), scaled by its size (see
    COLOR_DRIFT_RAMP), or None if none is needed. `where` names the measured
    area in the log.
    """
    reference = ~_dilate(masked, margin)
    share = float(reference.mean())
    if share < COLOR_DRIFT_MIN_REFERENCE:
        print(f"[color_correction] Only {share:.0%} of the image could be compared ({where}), too little to measure color drift; not correcting.", flush=True)
        return None

    gen_ref, orig_ref = gen_lab[reference], orig_lab[reference]
    corrections = _correction_from_lab(gen_ref, orig_ref)

    shift = np.array([c[0] for c in corrections], dtype=np.float32)
    scale = np.array([c[1] for c in corrections], dtype=np.float32)
    drift = float(_delta_e(gen_ref * (scale - 1) + shift).mean())

    strength = ramp(drift, COLOR_DRIFT_RAMP)
    print(f"[color_correction] Color drift {where}: delta E {drift:.1f} (measured on {share:.0%} of the image), correcting at {strength:.0%}.", flush=True)
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
    strength = ramp(cell_de, COLOR_LOCAL_RAMP)
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
    limit = COLOR_LOCAL_LIMIT_DE
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


def whole_image_color_match(gen: Image.Image, orig: Image.Image) -> Optional[ColorMatch]:
    """
    How to colour-match the generated image of an unmasked edit to the
    original, or None if it needs nothing or can't be matched (see the notes
    below COLOR_LOCAL_LIMIT_DE). gen must show orig's framing, cropped to its
    aspect ratio.
    """
    orig = resize_to_cover_and_crop(orig, gen.width, gen.height, Image.Resampling.LANCZOS)
    gen_lab, orig_lab, _ = _samples(gen, orig, Image.new("L", gen.size, 0))
    # Pixels that changed a lot hold edited content, not a tint. They are left
    # out as a mask's area is, without its margin, as they are scattered.
    changed = _delta_e(orig_lab - gen_lab) > COLOR_LOCAL_OUTLIER_DE
    corrections = _global_correction(gen_lab, orig_lab, changed, "outside what the edit changed", margin=0)
    return ColorMatch(gen.size, corrections, None) if corrections else None


def apply_color_match(img: Image.Image, match: ColorMatch, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Applies a ColorMatch to img, the part of the generated image at `box`
    (left, top, right, bottom, in generated-image pixels): the global
    per-channel corrections in L*a*b* (see rgb_to_lab()), then the local
    offsets, interpolated to img's pixels.
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
    return Image.fromarray(np.rint(lab_to_rgb(arr)).astype(np.uint8), mode="RGB")
