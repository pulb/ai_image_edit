# SPDX-License-Identifier: GPL-3.0-or-later
"""
Lining up the model's output with its input before a masked edit is
composited (align_to_input()), measured by phase correlation (measure_shift()).
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ai_image_edit.core.imaging.files import read_mask
from ai_image_edit.core.imaging.geometry import resize_to_cover_and_crop
from ai_image_edit.core.paths import new_work_file

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


@dataclass(frozen=True)
class Block:
    """One block of a measure_shift() region, in the reference image's pixels."""

    centre: Tuple[float, float]
    shift: Optional[Tuple[float, float]]  # None if the block was skipped
    confidence: float  # correlation strength, see _phase_shift()
    coverage: float  # its unmasked share


@dataclass(frozen=True)
class ShiftMeasurement:
    """What measure_shift() found: the region measured (left, top, right, bottom) and its blocks, row by row."""

    region: Tuple[float, float, float, float]
    blocks: List[Block]

    @property
    def centre(self) -> Tuple[float, float]:
        left, top, right, bottom = self.region
        return (left + right) / 2, (top + bottom) / 2

    @property
    def counted(self) -> List[Block]:
        """The blocks that could be measured."""
        return [blk for blk in self.blocks if blk.shift is not None]


@dataclass(frozen=True)
class Displacement:
    """
    A shift (at `centre`) and a stretch per axis (0 for none): content at
    point p was moved by shift + stretch * (p - centre). residual is the
    largest distance (px) of a measured block from it.
    """

    shift: Tuple[float, float]
    stretch: Tuple[float, float]
    centre: Tuple[float, float]
    residual: float = 0.0

    def at(self, point: Tuple[float, float]) -> Tuple[float, float]:
        return tuple(self.shift[a] + self.stretch[a] * (point[a] - self.centre[a]) for a in (0, 1))

    def left_over(self, blk: Block) -> Tuple[float, float]:
        """A measured block's displacement that this one does not explain."""
        moved = self.at(blk.centre)
        return blk.shift[0] - moved[0], blk.shift[1] - moved[1]

    def describe(self) -> str:
        return (
            f"shift {_fmt(self.shift)} px at the region's centre, "
            f"stretch x {self.stretch[0] * 100:+.2f}%, y {self.stretch[1] * 100:+.2f}%"
        )


def _fmt(shift: Tuple[float, float]) -> str:
    return f"({shift[0]:+.1f}, {shift[1]:+.1f})"


def measure_shift(ref: Image.Image, moved: Image.Image, mask: Optional[Image.Image] = None) -> Optional[ShiftMeasurement]:
    """
    How far `moved` (the same size as ref) shows ref's content displaced
    around the masked area (mask: same size, non-zero = edited), or over the
    whole image without one, block by block. None if the mask leaves nothing
    to compare. _fit_displacement() turns the blocks into a shift and stretch.
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
            blocks.append(Block(((x0 + x1) / 2 / factor, (y0 + y1) / 2 / factor), shift, confidence, coverage))
    return ShiftMeasurement(tuple(v / factor for v in region), blocks)


def _fit_displacement(result: ShiftMeasurement) -> Optional[Displacement]:
    """
    The shift and stretch that best explain the counting blocks' shifts (a
    stretch of 0 where they share one position), or None with fewer than
    ALIGN_MIN_BLOCKS counting blocks.
    """
    used = result.counted
    if len(used) < ALIGN_MIN_BLOCKS:
        return None
    centre = result.centre
    t, k = [], []
    for axis in (0, 1):
        pos = np.array([blk.centre[axis] - centre[axis] for blk in used])
        shift = np.array([blk.shift[axis] for blk in used])
        if np.ptp(pos) < 1:
            slope, offset = 0.0, float(shift.mean())
        else:
            slope, offset = np.polyfit(pos, shift, 1)
        t.append(float(offset))
        k.append(float(slope))
    fit = Displacement((t[0], t[1]), (k[0], k[1]), centre)
    residual = max(float(np.hypot(*fit.left_over(blk))) for blk in used)
    return Displacement(fit.shift, fit.stretch, centre, residual)


def _undo_displacement(img: Image.Image, displacement: Displacement) -> Image.Image:
    """
    img with the displacement undone: each pixel is taken from where the model
    put it. Edges uncovered by the move repeat the image's outermost pixels.
    """
    shift, stretch = displacement.shift, displacement.stretch
    reach = max(abs(stretch[0]) * img.width, abs(stretch[1]) * img.height)
    pad = int(math.ceil(max(abs(shift[0]), abs(shift[1])) + reach)) + 2
    padded = Image.fromarray(np.pad(np.asarray(img.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="edge"))
    # Output pixel (x, y) samples the padded image at (x, y) + its displacement + pad.
    origin = displacement.at((0.0, 0.0))
    matrix = (1 + stretch[0], 0, origin[0] + pad, 0, 1 + stretch[1], origin[1] + pad)
    return padded.transform(img.size, Image.Transform.AFFINE, matrix, Image.Resampling.BICUBIC)


def _rejection_reason(fit: Displacement, counted: List[Block], size: Tuple[int, int]) -> Optional[str]:
    """Why undoing `fit` would not be plausible (see ALIGN_MAX_WORSE and the limits after it), or None."""
    largest = max(max(abs(v) for v in fit.at(blk.centre)) for blk in counted)
    worsened = [float(np.hypot(*fit.left_over(blk))) - float(np.hypot(*blk.shift)) for blk in counted]
    worse = sum(w > ALIGN_MAX_WORSE for w in worsened)
    if worse:
        return (
            f"undoing {fit.describe()} would move {worse} of {len(counted)} blocks further from where they belong, "
            f"by up to {max(worsened):.1f} px, more than {ALIGN_MAX_WORSE} px"
        )
    if largest < ALIGN_MIN_SHIFT:
        return f"at most {largest:.1f} px, below {ALIGN_MIN_SHIFT} px"
    if max(abs(fit.shift[0]), abs(fit.shift[1])) > ALIGN_MAX_SHIFT * max(size):
        return f"{fit.describe()} is more than {ALIGN_MAX_SHIFT:.0%} of the image, a recomposition rather than a drift"
    if max(abs(fit.stretch[0]), abs(fit.stretch[1])) > ALIGN_MAX_STRETCH:
        return f"{fit.describe()} stretches by more than {ALIGN_MAX_STRETCH:.0%}"
    return None


def _describe_block(blk: Block) -> str:
    if blk.coverage < ALIGN_MIN_COVERAGE:
        return f"masked ({blk.coverage:.0%} free)"
    if blk.shift is None:
        return f"weak {blk.confidence:.2f}"
    return f"{_fmt(blk.shift)} {blk.confidence:.2f}"


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
            mask = resize_to_cover_and_crop(read_mask(mask_path), *size, Image.Resampling.NEAREST)
            result = measure_shift(model_input, output, mask)
        if result is None:
            print("[model_shift] Not measured: too little unchanged image to compare.", flush=True)
            return output_path

        left, top, right, bottom = (round(v) for v in result.region)
        blocks = ", ".join(_describe_block(blk) for blk in result.blocks)
        fit = _fit_displacement(result)
        described = fit.describe() if fit is not None else ""
        print(
            f"[model_shift] The model's output against its input, around the mask (px, x right, y down), "
            f"region ({left}, {top})-({right}, {bottom}): {described + '; ' if described else ''}"
            f"blocks (shift, correlation; row by row): {blocks}.",
            flush=True,
        )
        if fit is None:
            print(f"[model_shift] Not corrected: fewer than {ALIGN_MIN_BLOCKS} blocks could be measured reliably.", flush=True)
            return output_path
        counted = result.counted
        reason = _rejection_reason(fit, counted, size)
        if reason:
            print(f"[model_shift] Not corrected: {reason}.", flush=True)
            return output_path

        with Image.open(output_path) as output:
            aligned = _undo_displacement(output, fit)
        aligned_path = new_work_file("aligned")
        aligned.save(aligned_path, compress_level=1)  # an intermediate file: speed over size
        before = max(float(np.hypot(*blk.shift)) for blk in counted)
        print(f"[model_shift] Corrected: undid {described} ({len(counted)} blocks, none more than {ALIGN_MAX_WORSE} px further off, the farthest {fit.residual:.1f} px off, against {before:.1f} px before).", flush=True)
        return str(aligned_path)
    except Exception as e:  # noqa: BLE001 — the alignment must never break a generation
        print(f"[model_shift] Not corrected: the measurement failed ({e}).", flush=True)
        return output_path
