# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pixel-level helpers around inference: preparing uploads (prepare_upload()),
LAB color-space color matching of masked and unmasked edits,
uniform-scale cover-crop resizing, the mask-feathered composite that
merges a generated result back into the original image, the alignment
that undoes the model's own drift before that (align_to_input()), and
run_generation(), which orchestrates the crop -> infer -> align -> composite
sequence around an inference callback.

The parts live in the submodules (color, files, geometry, compositing,
alignment); what the rest of the app uses is available from this package.
"""
from typing import Callable, Optional, Tuple

from PIL import Image

from ai_image_edit.core.imaging.alignment import align_to_input, measure_shift
from ai_image_edit.core.imaging.color import (
    ColorMatch, apply_color_match, lab_to_rgb, masked_color_match, rgb_to_lab, whole_image_color_match,
)
from ai_image_edit.core.imaging.compositing import apply_whole_image_color_correction, composite_with_soft_transition
from ai_image_edit.core.imaging.files import JPEG_QUALITY, compose_annotations, prepare_upload, read_mask, save_as_jpeg
from ai_image_edit.core.imaging.geometry import fit_image_to_size, resize_to_cover_and_crop
from ai_image_edit.core.types import Postprocessing


def run_generation(
    source_image_path: str,
    mask_path: Optional[str],
    annotated_image_path: Optional[str],
    gen_size: Optional[Tuple[int, int]],
    infer: Callable[[str], str],
    postprocessing: Postprocessing,
    input_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Orchestrates the crop -> infer -> align -> composite/color-correct sequence
    around inference. `infer` is the only model-specific piece: a callback
    taking the (possibly cropped) source path and returning the path to the
    raw generated image.

    - The model's input image is cropped/resized to exactly input_size via
      fit_image_to_size before infer() if that is set, and passed to infer()
      unprepared otherwise. With a mask it is always fitted to gen_size (the
      generation's (width, height), which must be given then), as the
      composite maps the mask onto that image; an input_size of another size
      is a ValueError.
    - If mask_path is set, the result is lined up with the model's input
      (align_to_input) and composited back into the untouched original via
      composite_with_soft_transition, feathered by postprocessing.feather_amount,
      as its rescale_mode selects, and sharpened and grained if its match_detail.
    - Otherwise the raw result is returned as-is, or color-corrected as a
      whole if postprocessing.color_correction is set and it shows the
      input's framing (with a mask that setting has no effect:
      composite_with_soft_transition() matches colours by itself).

    The model's input image is annotated_image_path if given, else
    source_image_path. Compositing and color correction always use the clean
    source_image_path, so annotations never end up in the result.
    """
    if mask_path:
        if input_size not in (None, gen_size):
            raise ValueError(f"A masked edit needs its input at the generation size {gen_size}, not {input_size}.")
        input_size = gen_size
    model_source_path = annotated_image_path or source_image_path
    model_input_path = (
        fit_image_to_size(model_source_path, *input_size, Image.Resampling.LANCZOS) if input_size else model_source_path
    )

    output_path = infer(model_input_path)

    if mask_path:
        # Pasted back onto the original, so lined up with the model's input first.
        output_path = align_to_input(model_input_path, output_path, mask_path)
        return composite_with_soft_transition(
            source_image_path, output_path, mask_path, postprocessing.feather_amount,
            postprocessing.rescale_mode, postprocessing.match_detail,
        )
    if postprocessing.color_correction:
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
