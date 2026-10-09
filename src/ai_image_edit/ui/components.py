# SPDX-License-Identifier: GPL-3.0-or-later
"""Small UI building blocks: uploads, image frames, sliders, browser-side helpers."""
import base64
import json
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Tuple

from nicegui import app, ui
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.paths import WORK_DIR, to_url
from ai_image_edit.core.types import RangeSpec
from ai_image_edit.ui.helpers import PRIMARY_COLOR


async def save_uploaded_file(file) -> str:
    """
    Saves a NiceGUI FileUpload — the `.file` attribute of the
    UploadEventArguments a ui.upload widget's on_upload event delivers
    (nicegui.elements.upload_files.FileUpload) — to a new file under
    WORK_DIR, returning its local path. Uses FileUpload.save(), the
    documented method for this (nicegui.elements.upload_files.FileUpload.save).
    Always writes to a fresh, server-generated name — file.name (already
    sanitized by NiceGUI itself) is only used for its extension, read via
    pathlib so it can't smuggle in a path. The saved file is made upright
    if it carries an EXIF orientation (see imaging.apply_exif_orientation).
    """
    suffix = Path(file.name).suffix or ".png"
    stem = Path(file.name).stem or "upload"
    path = WORK_DIR / f"{stem}_{uuid.uuid4().hex}{suffix}"
    await file.save(path)
    await nicegui_run.io_bound(imaging.apply_exif_orientation, str(path))
    return str(path)


def save_data_url(data_url: str, filename_hint: str = "mask.png") -> str:
    """
    Decodes a `data:image/png;base64,...` URL — as produced by
    canvas.toDataURL() for the drawn mask — into a WORK_DIR file, returning
    its local path. The mask is the one piece of image data that's
    inherently synthesized client-side (drawn, not uploaded), so it has no
    original file to read bytes from — this is its only source.
    """
    _header, b64data = data_url.split(",", 1)
    raw = base64.b64decode(b64data)
    suffix = Path(filename_hint).suffix or ".png"
    stem = Path(filename_hint).stem or "mask"
    path = WORK_DIR / f"{stem}_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(raw)
    return str(path)


def call_js(ns: str, fn: str, *args, timeout: float = 5.0):
    """Await `ns.fn(*args)` in the browser; arguments are JSON-encoded."""
    arglist = ", ".join(json.dumps(a) for a in args)
    return ui.run_javascript(f"{ns}.{fn}({arglist})", timeout=timeout)


async def init_widget(ns: str, widget_id: str, label: str) -> None:
    """Run `ns.init(widget_id)`; a timeout is logged, not fatal."""
    try:
        await call_js(ns, "init", widget_id, timeout=10.0)
    except TimeoutError:
        print(f"[{label}] {ns}.init('{widget_id}') timed out — the widget may not respond until the page is reloaded.", flush=True)


def apply_dark_theme() -> None:
    """
    Dark mode in this app's colours. dark_page is Quasar's page/body
    background variable, dark is the surface color for dark-mode components
    like ui.card — the documented way to set these
    (nicegui.io/documentation/colors) rather than a manual `!important` CSS
    override fighting Quasar's own theme layer.
    """
    ui.dark_mode().enable()
    ui.colors(primary=PRIMARY_COLOR, dark="#111111", dark_page="#000000")


def create_hidden_uploader() -> ui.upload:
    """
    A hidden single-file ui.upload whose file picker a visible box opens via
    run_method("pickFiles").

    accept="*/*" rather than "image/*": on Android, restricting to images
    forces a limited gallery-style picker and hides the full file browser
    (folder navigation, cloud storage sources, etc.), which is often the
    only practical way to reach a downloaded or shared image file. Nothing
    downstream validates file type either way; it is only a hint to the OS
    picker.
    """
    # Hidden with a CSS class, not set_visibility(False): since NiceGUI 3.18
    # the upload route rejects (403) uploads to elements that are not visible.
    uploader = ui.upload(auto_upload=True, max_files=1).props('accept="*/*"').classes("hidden")
    return uploader


def create_clear_badge(on_click: Callable) -> ui.button:
    """A round X button overlapping the top-right corner of its .aie-corner-wrap; hidden until shown."""
    badge = ui.button(icon="close", on_click=on_click).props("round dense unelevated size=sm").classes("aie-clear")
    badge.set_visibility(False)
    return badge


PLACEHOLDER_TEXT = "Click to upload an image"


def create_image_frame(box_class: str, on_clear: Callable) -> Tuple[ui.element, ui.button]:
    """
    The wrapper shared by the editor and the reference-image slots: a
    .aie-corner-wrap holding the framed box (`box_class`) and its clear
    badge. Returns (box, badge); the box's content goes in `with box:`.
    """
    with ui.element("div").classes("aie-corner-wrap"):
        with ui.element("div").classes(box_class) as box:
            pass
        badge = create_clear_badge(on_clear)
    return box, badge


def attach_file_picker(
    box: ui.element,
    on_file: Callable[[str], Awaitable[None]],
    enabled: Callable[[], bool] = lambda: True,
) -> None:
    """
    Clicking `box` while enabled() is true opens the file picker of a hidden
    uploader; the chosen file is saved and its local path handed to on_file.
    """
    uploader = create_hidden_uploader()

    async def open_picker() -> None:
        if enabled():
            await uploader.run_method("pickFiles", timeout=5.0)

    box.on("click", open_picker)

    async def handle_upload(e) -> None:
        await on_file(await save_uploaded_file(e.file))
        uploader.reset()

    uploader.on_upload(handle_upload)


def create_simple_image_upload(label: str) -> dict:
    """
    A click-to-upload image box with no editing — used for the reference
    image slots. Clicking anywhere in the box calls pickFiles()
    on a hidden ui.upload (QUploader's own file-picker method, invoked via
    NiceGUI's documented run_method — the same mechanism NiceGUI's own
    Upload.reset() uses internally for QUploader's reset()); once a file is
    chosen, its on_upload event hands it to Python, which saves it and
    swaps in the preview.

    Styling (.aie-upload-box etc.) comes from WIDGET_CSS, a small stylesheet
    this app defines itself and injects via ui.add_head_html — see the
    comment above WIDGET_CSS for why that's preferred here over NiceGUI's
    bundled Tailwind classes. The box itself is a real nicegui `ui.element`
    rather than raw HTML so a NiceGUI click handler can be attached to it
    directly (see attach_file_picker).

    Returns holder — holder['path'] always reflects the currently selected
    file's local path (or None). holder['container'] is the slot's element
    (to show or hide it) and holder['clear'] removes the selected image.
    """
    holder: dict = {"path": None}

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("text-sm text-gray-400")

        def clear_image() -> None:
            holder["path"] = None
            preview.set_visibility(False)
            placeholder.set_visibility(True)
            badge.set_visibility(False)

        box, badge = create_image_frame("aie-upload-box", clear_image)
        with box:
            placeholder = ui.label(PLACEHOLDER_TEXT).classes("aie-placeholder")
            preview = ui.image().classes("aie-preview")
            preview.set_visibility(False)

        async def handle_file(path: str) -> None:
            holder["path"] = path
            preview.set_source(to_url(path))
            preview.set_visibility(True)
            placeholder.set_visibility(False)
            badge.set_visibility(True)

        attach_file_picker(box, handle_file)

    holder["container"] = container
    holder["clear"] = clear_image

    return holder


def create_value_slider(spec: RangeSpec) -> ui.slider:
    """
    A slider with its current value as text to its right. The value is kept
    in sync with the server, which reads it at Generate time.
    """
    with ui.row().classes("w-full items-center gap-1 no-wrap"):
        slider = ui.slider(min=spec.min, max=spec.max, step=spec.step, value=spec.default).props("dense").classes("flex-1")
        ui.label().classes("text-xs text-gray-400").style("min-width: 1.5em").bind_text_from(
            slider, "value", backward=lambda v: f"{v:g}"
        )
    return slider
