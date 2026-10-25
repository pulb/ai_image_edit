# SPDX-License-Identifier: GPL-3.0-or-later
"""Small UI building blocks: uploads, image frames, sliders, browser-side helpers."""
import base64
import json
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional, Tuple

from nicegui import ui
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.paths import new_work_file, to_url
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
    pathlib so it can't smuggle in a path. The saved file is then prepared
    (upright, sRGB; see imaging.prepare_upload()), which may replace it with a
    PNG: the returned path is the one to use.
    """
    suffix = Path(file.name).suffix or ".png"
    stem = Path(file.name).stem or "upload"
    path = new_work_file(stem, suffix)
    await file.save(path)
    return await nicegui_run.io_bound(imaging.prepare_upload, str(path))


def save_data_url(data_url: str, stem: str) -> str:
    """
    Decodes a `data:image/png;base64,...` URL — as produced by
    canvas.toDataURL() for the drawn mask and annotations — into a WORK_DIR file, returning
    its local path, `<stem>_<random hex>.png`.
    """
    _header, b64data = data_url.split(",", 1)
    raw = base64.b64decode(b64data)
    path = new_work_file(stem)
    path.write_bytes(raw)
    return str(path)


def call_js(ns: str, fn: str, *args, timeout: float = 5.0):
    """Await `ns.fn(*args)` in the browser; arguments are JSON-encoded."""
    arglist = ", ".join(json.dumps(a) for a in args)
    return ui.run_javascript(f"{ns}.{fn}({arglist})", timeout=timeout)


async def init_widget(ns: str, widget_id: str, label: str, *args) -> None:
    """Run `ns.init(widget_id, *args)`; a timeout is logged, not fatal."""
    try:
        await call_js(ns, "init", widget_id, *args, timeout=10.0)
    except TimeoutError:
        print(f"[{label}] {ns}.init('{widget_id}') timed out — the widget may not respond until the page is reloaded.", flush=True)


class StatusLine:
    """
    A small grey status text over a thin progress bar, the same wherever the
    page reports progress. Hidden until show() is called. With the class
    aie-progress-overlay (and created inside the result image's frame) it is
    the progress of a generation, on a dark panel centered over the image.
    """

    def __init__(self, classes: str = "") -> None:
        with ui.column().classes(f"w-full gap-1 items-center {classes}") as self.box:
            self.label = ui.label().classes("text-xs w-full text-center text-gray-400")
            # Two bars, one moving and one filling, rather than one switched
            # between the two: Quasar draws a moving bar's fill at full width,
            # so a switched bar would shrink back from there to its value.
            self.moving_bar = ui.linear_progress(show_value=False, size="4px").props("rounded indeterminate").classes("w-full")
            self.bar = ui.linear_progress(value=0, show_value=False, size="4px").props("rounded").classes("w-full")
        self.box.set_visibility(False)

    def show(self, text: str, fraction: Optional[float] = None, *, bar: bool = True, error: bool = False) -> None:
        """Show text, in red if error, over a bar that is fraction (0 to 1) full, or moving if None."""
        self.box.set_visibility(True)
        self.label.set_text(text)
        self.label.classes(add="text-negative" if error else "text-gray-400", remove="text-gray-400" if error else "text-negative")
        self.moving_bar.set_visibility(bar and fraction is None)
        self.bar.set_visibility(bar and fraction is not None)
        # Emptied while hidden, so the next time it shows it fills from 0.
        self.bar.set_value(0 if fraction is None else fraction)

    def hide(self) -> None:
        self.box.set_visibility(False)
        self.bar.set_value(0)


GROUP_BG = "#1f1f1f"
"""The background of grouped controls (GROUP_CLASSES) and of filled buttons on a card, a little brighter than the card."""

GROUP_CLASSES = f"w-full gap-3 bg-[{GROUP_BG}] rounded-lg"
"""A box that groups controls on a card, a little brighter than the card. Padding is added by the user: p-4 around controls that have inner padding of their own (radios), p-6 around plain fields."""


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


PLACEHOLDER_TEXT = "Click to upload an image"


def create_image_frame(box_class: str, on_clear: Callable) -> Tuple[ui.element, ui.button]:
    """
    The wrapper shared by the editor and the reference-image slots: a
    .aie-corner-wrap holding the framed box (`box_class`) and its clear
    badge, a round X over the top-right corner, hidden until shown. Returns
    (box, badge); the box's content goes in `with box:`.
    """
    with ui.element("div").classes("aie-corner-wrap"):
        box = ui.element("div").classes(box_class)
        badge = ui.button(icon="close", on_click=on_clear).props("round dense unelevated size=sm").classes("aie-clear")
        badge.set_visibility(False)
    return box, badge


def attach_file_picker(
    box: ui.element,
    on_file: Callable[[str], Awaitable[None]],
    enabled: Callable[[], bool] = lambda: True,
) -> None:
    """
    Clicking `box` while enabled() is true opens the file picker of a hidden
    uploader; the chosen file is saved and its local path handed to on_file.

    accept="*/*" rather than "image/*": on Android, images only means a
    gallery picker without the file browser, often the only way to reach a
    downloaded file. A file that isn't an image fails in prepare_upload().
    """
    # Hidden with a CSS class, not set_visibility(False): since NiceGUI 3.18
    # the upload route rejects (403) uploads to elements that are not visible.
    uploader = ui.upload(auto_upload=True, max_files=1).props('accept="*/*"').classes("hidden")

    async def open_picker() -> None:
        if enabled():
            await uploader.run_method("pickFiles", timeout=5.0)

    box.on("click", open_picker)

    async def handle_upload(e) -> None:
        await on_file(await save_uploaded_file(e.file))
        uploader.reset()

    uploader.on_upload(handle_upload)


class ImageSlot:
    """
    A click-to-upload image box with no editing — used for the reference
    image slots. Clicking anywhere in the box calls pickFiles()
    on a hidden ui.upload (QUploader's own file-picker method, invoked via
    NiceGUI's documented run_method — the same mechanism NiceGUI's own
    Upload.reset() uses internally for QUploader's reset()); once a file is
    chosen, its on_upload event hands it to Python, which saves it and
    swaps in the preview.

    Styling (.aie-upload-box etc.) comes from WIDGET_CSS, a small stylesheet
    this app defines itself and injects via ui.add_head_html (see the
    module docstring of assets.py). The box itself is a real nicegui `ui.element`
    rather than raw HTML so a NiceGUI click handler can be attached to it
    directly (see attach_file_picker).

    `path` always reflects the currently selected file's local path (or
    None); clear() removes the selected image and set_visible() shows or
    hides the whole slot.
    """

    def __init__(self, label: str) -> None:
        self.path: Optional[str] = None
        with ui.column().classes("w-full gap-1") as self._container:
            ui.label(label).classes("text-sm text-gray-400")
            box, self._badge = create_image_frame("aie-upload-box", self.clear)
            with box:
                self._placeholder = ui.label(PLACEHOLDER_TEXT).classes("aie-placeholder")
                self._preview = ui.image().classes("aie-preview")
                self._preview.set_visibility(False)
            attach_file_picker(box, self._show_file)

    def clear(self) -> None:
        self.path = None
        self._show(False)

    def set_visible(self, visible: bool) -> None:
        self._container.set_visibility(visible)

    async def _show_file(self, path: str) -> None:
        self.path = path
        self._preview.set_source(to_url(path))
        self._show(True)

    def _show(self, has_image: bool) -> None:
        self._preview.set_visibility(has_image)
        self._placeholder.set_visibility(not has_image)
        self._badge.set_visibility(has_image)


class SliderPopup:
    """
    A compact slider: a small button with an icon and the current value, which
    opens a menu with a labelled slider when tapped. For rows too tight for a
    slider of their own.

    The value shown in the button is updated in the browser while dragging, and
    js_on_value (JavaScript statements that see the new value as `value`) runs
    in the same handler, so a value the page only uses in the browser needs no
    server round-trip. Further events can be added to .slider.
    """

    def __init__(self, icon: str, label: str, spec: RangeSpec, *, js_on_value: str = "") -> None:
        value_id = f"aie_slider_value_{uuid.uuid4().hex}"
        self.button = ui.button(icon=icon).props("flat dense no-caps no-wrap size=md")
        with self.button:
            ui.html(f'<span id="{value_id}" class="q-ml-xs">{spec.default:g}</span>', sanitize=False)
            with ui.menu().props("anchor='bottom middle' self='top middle'"):
                with ui.column().classes("q-pa-md gap-1").style("min-width: 240px"):
                    ui.label(label).classes("text-xs text-gray-400")
                    self.slider = ui.slider(min=spec.min, max=spec.max, step=spec.step, value=spec.default).props("dense").classes("w-full")
        self.slider.on(
            "update:model-value",
            js_handler=f"(value) => {{ document.getElementById('{value_id}').textContent = value; {js_on_value} }}",
        )


def create_value_slider(spec: RangeSpec) -> ui.slider:
    """
    A slider with its current value in Quasar's bubble above the thumb. The
    value is kept in sync with the server, which reads it at Generate time.
    """
    # The bubble is drawn above the slider without taking space of its own,
    # so the margin keeps it clear of the label above.
    return ui.slider(min=spec.min, max=spec.max, step=spec.step, value=spec.default).props("label-always").classes(
        "w-full q-mt-lg"
    )
