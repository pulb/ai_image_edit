# SPDX-License-Identifier: GPL-3.0-or-later
"""
Image files coming in and going out: uploads made to show what the browser
shows (prepare_upload()), JPEG copies of results (save_as_jpeg()) and the
annotations drawn onto the source (compose_annotations()).
"""
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageCms, ImageOps

from ai_image_edit.core.paths import new_work_file

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
    out_path = new_work_file("annotated")
    composed.save(out_path)
    return str(out_path)


def read_mask(path: str) -> Image.Image:
    """A mask file as a greyscale image: its alpha channel if it has one (the editor exports white on transparent), else its grey levels."""
    with Image.open(path) as img:
        return img.getchannel("A") if img.mode == "RGBA" else img.convert("L")
