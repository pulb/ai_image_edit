# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pixel-level helpers around inference: preparing uploads (prepare_upload()),
LAB color-space color matching of masked and unmasked edits,
uniform-scale cover-crop resizing, the mask-feathered composite that
merges a generated result back into the original image, the alignment
that undoes the model's own drift before that (align_to_input()), and
run_generation(), which orchestrates the crop -> infer -> align -> composite
sequence around an inference callback.
"""
import io
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import DEFAULT_RESCALE_MODE, RescaleMode

# Minimum standard deviation (0-255 scale, matching PIL's LAB representation)
# a channel needs before we trust it enough to rescale by a std-dev ratio.
# Below this, we only shift the mean — rescaling by ratio when src_std is
# tiny can blow up small noise into huge swings (e.g. tgt_std/src_std =
# 45/0.8 =~ 56x), which clips at 0/255 and shows up as blown-out / wrong
# colors in the corrected image. Applies to L, a, and b alike; a/b
# (chrominance) channels are typically lower-variance than L in natural
# photos, so they'll fall back to a mean-only shift more often — that's
# expected and conservative, not a sign anything's wrong.
COLOR_MATCH_STD_EPS = 4.0
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
# It compares pixel by pixel, so it needs the output to show the input's
# framing (the source was fitted to the output's size). A recomposed output
# (another aspect ratio) is not corrected: its colours differ because its
# content does (more wall, less sky), which its statistics can't tell from a
# drift.

# Alignment of the model's output with its input (align_to_input()).
#
# Why: a masked edit is pasted back onto the original, so any displacement of
# the model's output against its input shows as a visible offset at the mask's
# border. Edit models regenerate the whole picture and do not keep it exactly
# in place. What was found (October 2026):
# - ComfyUI's Qwen-Image 2.1 model (comfy/ldm/qwen_image21/model.py) gives
#   every latent token a position id centred on its image, so a reference
#   image of the output's size is aligned token for token. A size mismatch
#   between reference and output squeezes or stretches the content instead: a
#   Diffusers-based measurement of Qwen-Image 2.1 (lilting.ch, "Qwen-Image 2.1
#   edit: pixel perfect output resolution") found block shifts of up to 8 px
#   from a 2.6% mismatch at about 1 MP, and 0.1 px or less once the sizes
#   matched. Sizes now match (sizes.ResolvedSize.input_size).
# - The rest is the model's own drift: the authors of a Qwen-Image 2.1
#   "Consistency LoRA" (comfyui-wiki.com, 2026-10-01) describe each edit as a
#   fresh generation that recomposes the picture, with a median offset of 0.7 px
#   for local edits; their LoRA was trained at 1408 px and not tested at 2 MP.
# - Measured here on masked edits with matched sizes: 0.5 px or less at 1 MP,
#   but 1.5 to 3 px (in generated pixels) for 12 MP photos generated at the 2K
#   tier, all to the left and slightly down, with a slight uneven stretch of
#   0.1 to 0.2% between the quarters. 3 px at 2K is about 5 px in the original.
# - The drift is not even across the image: in one 12 MP photo (a selfie, mask
#   over the eyes) three quarters moved 2.3 to 4.9 px left, but the bottom left
#   one, mostly a plain dark T-shirt, only 0.2 px. Plain areas give weak or
#   unreliable measurements, and only the movement around the edit shows as a
#   seam, as the original is kept everywhere else.
# - The same drift shows without our sizing changes: with the encode node
#   scaling the input itself (before the empty latent) and with the earlier
#   Diffusers pipeline, the same selfie moved 1 to 4 px left at the region's
#   centre, more on its right than on its left (stretch x -0.1 to -0.5%), in all
#   five runs.
#
# What is done: the displacement is measured by phase correlation between the
# model's input and its raw output, in a grid of blocks over the region around
# the mask (the mask's bounding box and a margin), leaving the mask itself out.
# Blocks with too little unmasked area or too weak a correlation (plain areas,
# redrawn content) are skipped. A shift plus a stretch per axis is fitted to the
# rest and undone before compositing. A model that redraws the area around the
# edit (a relight, a restyle) can make that measurement wrong, so the
# correction is only applied when it is plausible (the limits below); otherwise
# the output is used as it is. An unmasked edit has nothing to line up with and
# is left as it is.
# Uneven drift within the region cannot be undone this way.
#
# The measurement runs at this longest side (shifts are scaled back to full
# size), with frequencies above SHIFT_LOWPASS (cycles per pixel) faded out, as
# fine detail is what the model redraws most freely. Around a mask,
# SHIFT_MASK_MARGIN pixels are left out too, with soft edges: a hard edge in the
# weighting pulls the result towards zero.
SHIFT_SAMPLE_SIDE = 1024
SHIFT_LOWPASS = 0.15
SHIFT_MASK_MARGIN = 16
# The region around the mask reaches this share of the mask's bounding box
# beyond it on every side, and is split into at most ALIGN_GRID x ALIGN_GRID
# blocks of at least ALIGN_MIN_BLOCK pixels (at the sample size) per side.
ALIGN_REGION_MARGIN = 0.5
ALIGN_GRID = 3
ALIGN_MIN_BLOCK = 96
# A block needs this share of unmasked area, and this correlation strength
# (1 for a perfect match; about 0.1 at most for unrelated content in blocks of
# ALIGN_MIN_BLOCK, while matching content redrawn with heavy noise still
# reached about 0.2 in tests), to count. At least ALIGN_MIN_BLOCKS must count.
ALIGN_MIN_COVERAGE = 0.3
ALIGN_MIN_CONFIDENCE = 0.15
ALIGN_MIN_BLOCKS = 3
# Below this largest displacement (in generated pixels, over the blocks'
# centres) nothing is corrected: the measurement itself is good to about 0.1 px,
# and the model's run-to-run noise at 1 MP was up to 0.5 px.
ALIGN_MIN_SHIFT = 0.5
# The correction must not move any counting block more than this many pixels
# further from where it belongs than it was (the measurement noise, as for
# ALIGN_MIN_SHIFT); otherwise the blocks disagree or the measurement is
# unreliable, and it is left alone. The drift is not rigid, so some blocks stay off: in the six
# selfie runs with blocks (the five above and one with matched sizes), blocks
# that barely moved (plain wall and T-shirt, weak correlation) ended 0.0 to
# 0.3 px further off, while the farthest block went from 1.9 to 7.5 px off to
# 0.4 to 1.9 px. Requiring every block to end closer refused three of those
# runs, and an earlier limit on how far blocks may stay off the fit (0.5 px,
# then 1.5 px) refused corrections that helped every block.
ALIGN_MAX_WORSE = 0.5
# Implausibly large movements (a recomposed picture, not a drift) are left alone:
# a shift beyond this share of the image's longer side, or a stretch beyond
# this share per axis.
ALIGN_MAX_SHIFT = 0.02
ALIGN_MAX_STRETCH = 0.01

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

    low, high = COLOR_DRIFT_RAMP
    strength = min(1.0, max(0.0, (drift - low) / (high - low)))
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
    return Image.fromarray(lab_to_rgb(arr).astype(np.uint8), mode="RGB")


# Preparing uploads (prepare_upload()).
#
# Why: browsers apply an image's EXIF orientation and embedded colour profile,
# while PIL, the model and the colour matching read the pixels as stored.
# Phone photos carry both (iPhones tag every photo Display P3). What was found
# (October 2026): a selfie in Display P3 showed full colour in the browser,
# but the model got its stored numbers, read as sRGB, so a duller face. An
# unmasked result has no profile, so the browser showed it as sRGB, about 15
# to 20% less saturated than the input, most of all on skin, although its
# numbers matched the input's within a delta E of 1.7. A masked result looked
# right only because it is built on the original and keeps its profile.
#
# So uploads are converted to sRGB rather than their profile carried along:
# every step from the model to the colour matching to the browser then reads
# the same numbers the same way. Colours beyond sRGB (rare in photos) are
# clipped (relative colorimetric). Whether a profile is sRGB is tested by its
# effect, not its name, as there are many sRGB profiles. A changed file is
# saved as PNG, so it isn't compressed again.
_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
# The colours that tell whether a profile describes sRGB (_profile_to_convert()).
_PROFILE_TEST_COLORS = np.array(
    [(r, g, b) for r in range(0, 256, 51) for g in range(0, 256, 51) for b in range(0, 256, 51)], dtype=np.uint8
).reshape(6, 36, 3)


def _profile_to_convert(img: Image.Image) -> Optional[ImageCms.ImageCmsProfile]:
    """
    img's embedded colour profile if its colours need converting to sRGB, or
    None: without a profile (browsers show such images as sRGB too), with one
    that describes sRGB (no test colour moves by more than 1 of 255), or in a
    mode without a profile conversion here (only RGB, RGBA and CMYK have one).
    """
    icc = img.info.get("icc_profile")
    if not icc or img.mode not in ("RGB", "RGBA", "CMYK"):
        return None
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if img.mode == "CMYK":
            return profile
        test = Image.fromarray(_PROFILE_TEST_COLORS, mode="RGB")
        converted = ImageCms.profileToProfile(
            test, profile, _SRGB, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB"
        )
    except (OSError, ImageCms.PyCMSError) as exc:
        print(f"[upload] Colour profile unreadable ({exc}); colours kept as stored.", flush=True)
        return None
    moved = np.abs(np.array(converted, dtype=np.int16) - _PROFILE_TEST_COLORS.astype(np.int16)).max()
    return profile if moved > 1 else None


def _to_srgb(img: Image.Image, profile: ImageCms.ImageCmsProfile) -> Image.Image:
    """img converted from `profile` to sRGB (relative colorimetric: colours sRGB can't show are clipped), alpha kept."""
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    colors = img.convert("RGB") if alpha is not None else img
    converted = ImageCms.profileToProfile(
        colors, profile, _SRGB, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB"
    )
    if alpha is not None:
        converted.putalpha(alpha)
    return converted


# The modes PNG stores.
_PNG_MODES = ("1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA")


def prepare_upload(image_path: str) -> str:
    """
    Makes an uploaded image's stored pixels what the browser shows, which
    applies an EXIF orientation and an embedded colour profile while PIL, the
    model and the colour matching read the pixels as stored: turned upright,
    and converted to sRGB if its profile isn't sRGB (see
    _profile_to_convert()). A file that needs neither is left untouched and
    its path returned; otherwise it is replaced by a PNG (lossless, without a
    profile) next to it, whose path is returned. Both steps are logged.
    """
    with Image.open(image_path) as img:
        orientation = img.getexif().get(0x0112, 1)
        rotated = orientation in range(2, 9)  # 1 is upright, anything else not an orientation
        profile = _profile_to_convert(img)
        if not rotated and profile is None:
            return image_path
        prepared = ImageOps.exif_transpose(img)
    name = Path(image_path).name
    if rotated:
        print(f"[upload] {name}: turned upright (EXIF orientation {orientation}).", flush=True)
    if profile is not None:
        description = ImageCms.getProfileDescription(profile).strip() or "its colour profile"
        prepared = _to_srgb(prepared, profile)
        print(f"[upload] {name}: converted from {description} to sRGB.", flush=True)
    if prepared.mode not in _PNG_MODES:  # an untagged CMYK photo, say: as PIL reads it everywhere else
        prepared = prepared.convert("RGBA" if "A" in prepared.getbands() else "RGB")
    prepared.info.pop("icc_profile", None)
    png_path = Path(image_path).with_suffix(".png")
    prepared.save(png_path, format="PNG")
    if png_path != Path(image_path):
        Path(image_path).unlink(missing_ok=True)
        print(f"[upload] {name}: replaced by {png_path.name}.", flush=True)
    else:
        print(f"[upload] {name}: saved again.", flush=True)
    return str(png_path)


JPEG_QUALITY = 95


def save_as_jpeg(image_path: str) -> str:
    """
    A JPEG copy of the image next to it (`<stem>.jpg`), made once and reused.
    Full-resolution colour (no chroma subsampling), as that would blur small
    edited areas; transparency is flattened onto white.
    """
    jpeg_path = Path(image_path).with_suffix(".jpg")
    if jpeg_path.exists():
        return str(jpeg_path)
    with Image.open(image_path) as img:
        if "A" in img.getbands() or img.mode == "P":
            rgba = img.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = img.convert("RGB")
    partial = jpeg_path.with_name(jpeg_path.name + ".part")
    rgb.save(partial, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    partial.replace(jpeg_path)
    return str(jpeg_path)


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

    out_path = WORK_DIR / f"fitted_{uuid.uuid4()}.png"
    fitted.save(out_path)
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
    rescale_mode: RescaleMode = DEFAULT_RESCALE_MODE,
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

    with Image.open(gen_img_path) as gen_file, Image.open(orig_img_path) as orig_file, Image.open(mask_img_path) as mask_file:
        gen = gen_file.convert("RGB")
        orig = orig_file.convert("RGB")
        mask = mask_file.getchannel('A') if mask_file.mode == 'RGBA' else mask_file.convert("L")

    if rescale_mode == RescaleMode.EDIT:
        result = _composite_at_original_size(orig, gen, mask, feather_amount, match_detail)
    else:
        result = _composite_at_generated_size(orig, gen, mask, feather_amount)
    if result is None:
        return gen_img_path

    output_path = WORK_DIR / f"composited_{uuid.uuid4()}.png"
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

    output_path = WORK_DIR / f"color_corrected_{uuid.uuid4()}.png"
    corrected.save(output_path)
    return str(output_path)


def _phase_shift(ref: np.ndarray, moved: np.ndarray, weight: np.ndarray) -> Tuple[float, float, float]:
    """
    The (dx, dy) by which `moved` shows ref's content displaced, to 1/20 pixel,
    and how strongly they correlate (1 for a perfect match, near 0 for
    unrelated content): windowed, weighted phase correlation, refined on a fine
    grid around the peak (an upsampled DFT, after Guizar-Sicairos).
    """
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w)) * weight
    total = max(float(win.sum()), 1e-9)

    def spectrum(img: np.ndarray) -> np.ndarray:
        return np.fft.fft2((img - (img * win).sum() / total) * win)

    cross = np.conj(spectrum(ref)) * spectrum(moved)
    cross /= np.abs(cross) + 1e-9
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    lowpass = np.exp(-(fx ** 2 + fy ** 2) / (2 * SHIFT_LOWPASS ** 2))
    cross *= lowpass
    corr = np.fft.ifft2(cross).real
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    py, px = (py - h if py > h // 2 else py), (px - w if px > w // 2 else px)
    steps = np.arange(-20, 21) / 20
    rows = np.exp(2j * np.pi * np.outer(py + steps, np.fft.fftfreq(h) * h) / h)
    cols = np.exp(2j * np.pi * np.outer(px + steps, np.fft.fftfreq(w) * w) / w)
    fine = (rows @ cross @ cols.T).real
    iy, ix = np.unravel_index(np.argmax(fine), fine.shape)
    return float(px + steps[ix]), float(py + steps[iy]), float(fine.max() / lowpass.sum())


def _splits(start: int, stop: int) -> List[Tuple[int, int]]:
    """[start, stop) cut into at most ALIGN_GRID equal parts of at least ALIGN_MIN_BLOCK."""
    count = max(1, min(ALIGN_GRID, (stop - start) // ALIGN_MIN_BLOCK))
    edges = np.linspace(start, stop, count + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def measure_shift(ref: Image.Image, moved: Image.Image, mask: Optional[Image.Image] = None) -> Optional[dict]:
    """
    How far `moved` (the same size as ref) shows ref's content displaced
    around the masked area (mask: same size, non-zero = edited), or over the
    whole image without one: the region measured (left, top, right, bottom)
    and, for each block of it, row by row, its centre, shift (None if
    skipped), correlation strength and unmasked share, all in ref's pixels.
    None if the mask leaves nothing to compare. _fit_displacement() turns
    the blocks into a shift and stretch.
    """
    factor = min(1.0, SHIFT_SAMPLE_SIDE / max(ref.size))
    size = (max(1, round(ref.width * factor)), max(1, round(ref.height * factor)))
    a = np.asarray(ref.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float64)
    b = np.asarray(moved.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float64)
    h, w = a.shape
    weight = np.ones_like(a)
    region = (0, 0, w, h)
    if mask is not None:
        small = mask.convert("L").resize(size, Image.Resampling.BOX)
        box = small.getbbox()
        if box is not None:
            mx = max(ALIGN_MIN_BLOCK, round((box[2] - box[0]) * ALIGN_REGION_MARGIN))
            my = max(ALIGN_MIN_BLOCK, round((box[3] - box[1]) * ALIGN_REGION_MARGIN))
            region = (max(0, box[0] - mx), max(0, box[1] - my), min(w, box[2] + mx), min(h, box[3] + my))
        grown = small.filter(ImageFilter.BoxBlur(SHIFT_MASK_MARGIN))
        outside = Image.fromarray(np.where(np.asarray(grown) > 0, 0, 255).astype(np.uint8))
        weight = np.asarray(outside.filter(ImageFilter.GaussianBlur(SHIFT_MASK_MARGIN / 2)), dtype=np.float64) / 255
    if weight[region[1]:region[3], region[0]:region[2]].mean() < 0.01:
        return None

    blocks = []
    for y0, y1 in _splits(region[1], region[3]):
        for x0, x1 in _splits(region[0], region[2]):
            wb = weight[y0:y1, x0:x1]
            coverage = float(wb.mean())
            shift, confidence = None, 0.0
            if coverage >= ALIGN_MIN_COVERAGE:
                dx, dy, confidence = _phase_shift(a[y0:y1, x0:x1], b[y0:y1, x0:x1], wb)
                if confidence >= ALIGN_MIN_CONFIDENCE:
                    shift = (dx / factor, dy / factor)
            blocks.append({
                "centre": ((x0 + x1) / 2 / factor, (y0 + y1) / 2 / factor),
                "shift": shift, "confidence": confidence, "coverage": coverage,
            })
    return {"region": tuple(v / factor for v in region), "blocks": blocks}


def _fit_displacement(result: dict) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    The shift (tx, ty) at the region's centre and the stretch (kx, ky) per axis
    (0 for none, or where the counting blocks share one position) that best
    explain the counting blocks' shifts of a measure_shift() result, as
    displacement = t + k * (point - region centre), and the largest
    distance of a block from it in pixels. None with fewer than
    ALIGN_MIN_BLOCKS counting blocks.
    """
    used = [blk for blk in result["blocks"] if blk["shift"] is not None]
    if len(used) < ALIGN_MIN_BLOCKS:
        return None
    left, top, right, bottom = result["region"]
    centre = ((left + right) / 2, (top + bottom) / 2)
    t, k = [], []
    for axis in (0, 1):
        pos = np.array([blk["centre"][axis] - centre[axis] for blk in used])
        shift = np.array([blk["shift"][axis] for blk in used])
        if np.ptp(pos) < 1:
            slope, offset = 0.0, float(shift.mean())
        else:
            slope, offset = np.polyfit(pos, shift, 1)
        t.append(float(offset))
        k.append(float(slope))
    residual = max(float(np.hypot(*_left_over(blk, centre, (t[0], t[1]), (k[0], k[1])))) for blk in used)
    return (t[0], t[1]), (k[0], k[1]), residual


def _left_over(
    blk: dict, centre: Tuple[float, float], shift: Tuple[float, float], stretch: Tuple[float, float]
) -> Tuple[float, float]:
    """A measured block's displacement that the fitted shift and stretch do not explain."""
    return tuple(blk["shift"][a] - shift[a] - stretch[a] * (blk["centre"][a] - centre[a]) for a in (0, 1))


def _undo_displacement(
    img: Image.Image, shift: Tuple[float, float], stretch: Tuple[float, float], centre: Tuple[float, float]
) -> Image.Image:
    """
    img with the displacement shift + stretch * (point - centre) undone: each
    pixel is taken from where the model put it. Edges uncovered by the move
    repeat the image's outermost pixels.
    """
    reach = max(abs(stretch[0]) * img.width, abs(stretch[1]) * img.height)
    pad = int(math.ceil(max(abs(shift[0]), abs(shift[1])) + reach)) + 2
    padded = Image.fromarray(np.pad(np.asarray(img.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="edge"))
    # Output pixel (x, y) samples the padded image at (x + d_x(x) + pad, y + d_y(y) + pad).
    matrix = (
        1 + stretch[0], 0, shift[0] - stretch[0] * centre[0] + pad,
        0, 1 + stretch[1], shift[1] - stretch[1] * centre[1] + pad,
    )
    return padded.transform(img.size, Image.Transform.AFFINE, matrix, Image.Resampling.BICUBIC)


def align_to_input(model_input_path: str, output_path: str, mask_path: str) -> str:
    """
    Measures how far the model moved its input's content around the mask (see
    measure_shift and the notes above SHIFT_SAMPLE_SIDE), logs it, and, if the
    movement is plausible, undoes it: returns the path of the aligned output,
    else output_path itself. Logs what it did and why. Only measures when the
    output has the input's size; never raises.
    """
    try:
        with Image.open(model_input_path) as model_input, Image.open(output_path) as output:
            if model_input.size != output.size:
                print(f"[model_shift] Not measured: the output {output.size} has another size than the model's input {model_input.size}.", flush=True)
                return output_path
            size = model_input.size
            with Image.open(mask_path) as mask_file:
                alpha = mask_file.getchannel("A") if mask_file.mode == "RGBA" else mask_file.convert("L")
                mask = resize_to_cover_and_crop(alpha, *size, Image.Resampling.NEAREST)
            result = measure_shift(model_input, output, mask)
        if result is None:
            print("[model_shift] Not measured: too little unchanged image to compare.", flush=True)
            return output_path

        def fmt(shift: Tuple[float, float]) -> str:
            return f"({shift[0]:+.1f}, {shift[1]:+.1f})"

        def block(blk: dict) -> str:
            if blk["coverage"] < ALIGN_MIN_COVERAGE:
                return f"masked ({blk['coverage']:.0%} free)"
            if blk["shift"] is None:
                return f"weak {blk['confidence']:.2f}"
            return f"{fmt(blk['shift'])} {blk['confidence']:.2f}"

        left, top, right, bottom = (round(v) for v in result["region"])
        blocks = ", ".join(block(blk) for blk in result["blocks"])
        fit = _fit_displacement(result)
        described = ""
        if fit is not None:
            shift, stretch, residual = fit
            described = f"shift {fmt(shift)} px at the region's centre, stretch x {stretch[0] * 100:+.2f}%, y {stretch[1] * 100:+.2f}%"
        print(
            f"[model_shift] The model's output against its input, around the mask (px, x right, y down), "
            f"region ({left}, {top})-({right}, {bottom}): {described + '; ' if described else ''}"
            f"blocks (shift, correlation; row by row): {blocks}.",
            flush=True,
        )
        if fit is None:
            print(f"[model_shift] Not corrected: fewer than {ALIGN_MIN_BLOCKS} blocks could be measured reliably.", flush=True)
            return output_path
        centre = ((result["region"][0] + result["region"][2]) / 2, (result["region"][1] + result["region"][3]) / 2)
        counted = [blk for blk in result["blocks"] if blk["shift"] is not None]
        largest = max(
            max(abs(shift[0] + stretch[0] * (cx - centre[0])), abs(shift[1] + stretch[1] * (cy - centre[1])))
            for cx, cy in (blk["centre"] for blk in counted)
        )
        before = [float(np.hypot(*blk["shift"])) for blk in counted]
        worsened = [
            float(np.hypot(*_left_over(blk, centre, shift, stretch))) - b for blk, b in zip(counted, before)
        ]
        worse = sum(w > ALIGN_MAX_WORSE for w in worsened)
        if worse:
            reason = (
                f"undoing {described} would move {worse} of {len(counted)} blocks further from where they belong, "
                f"by up to {max(worsened):.1f} px, more than {ALIGN_MAX_WORSE} px"
            )
        elif largest < ALIGN_MIN_SHIFT:
            reason = f"at most {largest:.1f} px, below {ALIGN_MIN_SHIFT} px"
        elif max(abs(shift[0]), abs(shift[1])) > ALIGN_MAX_SHIFT * max(size):
            reason = f"{described} is more than {ALIGN_MAX_SHIFT:.0%} of the image, a recomposition rather than a drift"
        elif max(abs(stretch[0]), abs(stretch[1])) > ALIGN_MAX_STRETCH:
            reason = f"{described} stretches by more than {ALIGN_MAX_STRETCH:.0%}"
        else:
            reason = None
        if reason:
            print(f"[model_shift] Not corrected: {reason}.", flush=True)
            return output_path

        with Image.open(output_path) as output:
            aligned = _undo_displacement(output, shift, stretch, centre)
        aligned_path = WORK_DIR / f"aligned_{uuid.uuid4()}.png"
        aligned.save(aligned_path, compress_level=1)  # an intermediate file: speed over size
        print(f"[model_shift] Corrected: undid {described} ({len(counted)} blocks, none more than {ALIGN_MAX_WORSE} px further off, the farthest {residual:.1f} px off, against {max(before):.1f} px before).", flush=True)
        return str(aligned_path)
    except Exception as e:  # noqa: BLE001 — the alignment must never break a generation
        print(f"[model_shift] Not corrected: the measurement failed ({e}).", flush=True)
        return output_path


def run_generation(
    source_image_path: str,
    mask_path: Optional[str],
    gen_width: Optional[int],
    gen_height: Optional[int],
    feather_amount: int,
    apply_color_correction_enabled: bool,
    infer: Callable[[str], str],
    annotated_image_path: Optional[str] = None,
    rescale_mode: RescaleMode = DEFAULT_RESCALE_MODE,
    input_size: Optional[Tuple[int, int]] = None,
    match_detail: bool = True,
) -> str:
    """
    Orchestrates the crop -> infer -> align -> composite/color-correct sequence
    around inference. `infer` is the only model-specific piece: a callback
    taking the (possibly cropped) source path and returning the path to the
    raw generated image.

    - The model's input image is cropped/resized to exactly input_size via
      fit_image_to_size before infer() if that is set, and passed to infer()
      unprepared otherwise. With a mask it is always fitted to
      (gen_width, gen_height), which must be real dimensions then, as the
      composite maps the mask onto that image; an input_size of another size
      is a ValueError.
    - If mask_path is set, the result is lined up with the model's input
      (align_to_input) and composited back into the untouched original via
      composite_with_soft_transition, feathered by feather_amount, as
      rescale_mode selects, and sharpened and grained if match_detail.
    - Otherwise the raw result is returned as-is, or color-corrected as a
      whole if apply_color_correction_enabled is True and it shows the
      input's framing (with a mask that flag has no effect:
      composite_with_soft_transition() matches colours by itself).
      rescale_mode is ignored.

    The model's input image is annotated_image_path if given, else
    source_image_path. Compositing and color correction always use the clean
    source_image_path, so annotations never end up in the result.
    """
    if mask_path:
        if input_size not in (None, (gen_width, gen_height)):
            raise ValueError(f"A masked edit needs its input at the generation size {(gen_width, gen_height)}, not {input_size}.")
        input_size = (gen_width, gen_height)
    model_source_path = annotated_image_path or source_image_path
    model_input_path = (
        fit_image_to_size(model_source_path, *input_size, Image.Resampling.LANCZOS) if input_size else model_source_path
    )

    output_path = infer(model_input_path)

    if mask_path:
        # Pasted back onto the original, so lined up with the model's input first.
        output_path = align_to_input(model_input_path, output_path, mask_path)
        return composite_with_soft_transition(
            source_image_path, output_path, mask_path, feather_amount, rescale_mode, match_detail
        )
    if apply_color_correction_enabled:
        # The output shows the input's framing when it has the input's size:
        # the source fitted to it, or a source already that size. Otherwise it
        # was recomposed, and its colours differ because its content does.
        with Image.open(model_input_path) as model_input, Image.open(output_path) as output:
            framed = model_input.size == output.size
        if not framed:
            print("[color_correction] Not corrected: the output has another framing than the input, so its colours can't be compared.", flush=True)
            return output_path
        return apply_whole_image_color_correction(source_image_path, output_path)
    return output_path
