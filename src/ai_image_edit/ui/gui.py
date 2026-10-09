# SPDX-License-Identifier: GPL-3.0-or-later
"""
NiceGUI UI. Entry point is run(model), called through ui/__init__.py
— importing this module does nothing by itself.

UI-only: renders controls, collects a GenerationParams, and calls
model.generate(). Controls (mask editor, before/after slider, reference
slots, LoRA panel, sampler/scheduler/cfg/denoise, aspect-ratio/resolution)
are shown or hidden based on the active model's declared capabilities.
"""

import asyncio
import base64
import hmac
import json
import os
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.paths import WORK_DIR, to_url, from_url, trim_work_dir
from ai_image_edit.core.types import (
    ANNOTATION_COLORS, DEFAULT_SEED, FEATHER_RANGE, ORIGINAL_ASPECT_RATIO, RangeSpec,
)
from ai_image_edit.ui.assets import CLIENT_JS, DEFAULT_BRUSH_SIZE, DEFAULT_FEATHER_AMOUNT, WIDGET_CSS
from ai_image_edit.ui.helpers import (
    HOST,
    PORT,
    PRIMARY_COLOR,
    RECENT_FILES_KEPT,
    default_choice,
    describe_error,
    params_from_ui,
    storage_secret,
)
from ai_image_edit.workflow_model import WorkflowModel


# --- Small file helpers shared by the UI layer ---
# (model-independent — these don't need `model`, so
# they stay at module scope rather than inside run(), below.)

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


def install_password_login(password: str) -> str:
    """
    Puts every route behind a password login page and returns the
    storage_secret ui.run() needs for the session cookie.

    Only /login, /favicon.ico and NiceGUI's own /_nicegui assets are open;
    /files (uploaded and generated images) is covered like any other route.
    The secret is derived from the password unless APP_STORAGE_SECRET is
    set, so logins survive restarts and a changed password logs everyone out.
    """
    open_routes = {"/favicon.ico", "/login"}

    @app.add_middleware
    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if app.storage.user.get("authenticated") or path in open_routes or path.startswith("/_nicegui"):
                return await call_next(request)
            return RedirectResponse(f"/login?redirect_to={quote(path)}")

    @ui.page("/login")
    def login_page(redirect_to: str = "/") -> Optional[RedirectResponse]:
        if app.storage.user.get("authenticated"):
            return RedirectResponse("/")

        # Only same-site paths: "//host" and "/\host" would redirect off-site.
        target = redirect_to if redirect_to.startswith("/") and not redirect_to.startswith("//") and "\\" not in redirect_to else "/"

        async def try_login() -> None:
            if hmac.compare_digest(field.value.encode(), password.encode()):
                app.storage.user["authenticated"] = True
                ui.navigate.to(target)
            else:
                await asyncio.sleep(1.0)
                ui.notify("Wrong password", color="negative")

        apply_dark_theme()
        with ui.card().classes("absolute-center items-stretch"):
            field = ui.input("Password", password=True, password_toggle_button=True).props("autofocus")
            field.on("keydown.enter", try_login)
            ui.button("Log in", on_click=try_login)
        return None

    return storage_secret(password)


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


# --- UI component factories ---
#
# Every ui.html(...) call below passes sanitize=False. ui.html() defaults to
# client-side DOMPurify sanitization (nicegui.io/documentation/html), which
# strips inline event-handler attributes (oninput, onchange, ...) as a
# standard XSS defense — exactly the attributes the brush-size and
# comparison sliders depend on, so without this they silently do nothing.
# This is safe here because none of this HTML is ever built from user input
# — only server-generated uuids and fixed markup are interpolated into it.

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


class EditorInputs(NamedTuple):
    """What the editor holds at Generate time: saved layer files (None if not drawn) and the Feather amount."""

    mask_path: Optional[str]
    annotation_layer_path: Optional[str]
    feather_amount: int


async def create_mask_editor(
    on_mask_change: Optional[Callable[[bool], None]] = None,
    num_annotation_colors: int = 0,
) -> Tuple[
    dict,
    Callable[[], Awaitable[EditorInputs]],
    Callable[[str], Awaitable[None]],
]:
    """
    The "Input Image" widget: click-to-upload image + a <canvas> for
    drawing an optional inpainting mask, with a resizable brush. A radio
    group switches between pinch-to-zoom (default), mask drawing and,
    when num_annotation_colors > 0, annotating in one of that many colours
    (ANNOTATION_COLORS, thin fixed stroke). Drawing a stroke in mask mode
    makes this an Inpaint Edit, "Remove mask" reverts to a plain Image Edit. The Feather slider (0-16,
    default 3) is the Gaussian blur radius composite_with_soft_transition()
    applies to the mask edges; 0 falls back to a hard cutout.

    If given, on_mask_change(has_mask) fires whenever the mask goes from
    empty to non-empty or back (via the 'aie_mask_state' custom event).

    Returns (holder, get_inputs, set_image):
      * holder['path'] is the current source image's local path (or None).
      * get_inputs() is async: one browser call that reads the mask, the
        annotation layer and the Feather amount, and saves the two layers to
        disk. Called once, on Generate.
      * set_image(path) loads a new image and clears the mask and annotations.
    """
    editor_id = f"edit_{uuid.uuid4().hex}"
    holder: dict = {"path": None}

    if on_mask_change:
        def handle_mask_event(e) -> None:
            args = list(e.args) if isinstance(e.args, (list, tuple)) else [e.args]
            if len(args) < 2 or args[0] != editor_id:
                return
            on_mask_change(bool(args[1]))

        ui.on("aie_mask_state", handle_mask_event)

    async def clear_image() -> None:
        holder["path"] = None
        badge.set_visibility(False)
        await call_js("AIE", "clearImage", editor_id)

    editor_box, badge = create_image_frame("aie-editor-box", clear_image)
    with editor_box:
        ui.html(
            f'<div id="{editor_id}_zoomwrap" class="aie-zoomwrap">'
            f'<canvas id="{editor_id}_canvas" class="aie-canvas"></canvas>'
            f'<canvas id="{editor_id}_annot" class="aie-annot"></canvas>'
            f'<canvas id="{editor_id}_overlay" class="aie-overlay"></canvas>'
            f'<div id="{editor_id}_placeholder" class="aie-placeholder">{PLACEHOLDER_TEXT}</div>'
            f'</div>'
            f'<div id="{editor_id}_brushpreview" class="aie-brush-preview"></div>',
            sanitize=False,
        )

    async def handle_file(path: str) -> None:
        holder["path"] = path
        await call_js("AIE", "loadImage", editor_id, to_url(path), timeout=10.0)
        badge.set_visibility(True)

    # Once an image is loaded, a click on the box draws on the canvas instead
    # of reopening the picker — holder['path'] already tells us which state
    # we're in, no need to ask the browser.
    attach_file_picker(editor_box, handle_file, enabled=lambda: not holder["path"])

    async def remove_mask() -> None:
        await call_js("AIE", "clearMask", editor_id)

    async def remove_annotations() -> None:
        await call_js("AIE", "clearAnnotations", editor_id)

    swatches: Dict[str, ui.button] = {}

    with ui.column().classes("w-full gap-1 p-2 bg-neutral-900 rounded-lg"):
        # One q-radio per row (ui.radio is a single vertical option list and
        # cannot hold the sliders), all showing the same current mode.
        zoom_radio = ui.element("q-radio").props('val=zoom model-value=zoom label="Pinch to zoom"')

        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            mask_radio = ui.element("q-radio").props('val=mask model-value=zoom label="Mask"')

            with ui.column().classes("flex-1 gap-1"):
                ui.label("Brush size").classes("text-xs text-gray-400")
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    brush_slider = ui.slider(min=5, max=80, step=1, value=DEFAULT_BRUSH_SIZE).props("dense").classes("flex-1")
                    # Updated by the slider's js_handler below, in the browser
                    # (no server round-trip while dragging).
                    ui.html(f'<span id="{editor_id}_brushvalue">{DEFAULT_BRUSH_SIZE}</span>', sanitize=False).classes(
                        "text-xs text-gray-400"
                    ).style("min-width: 1.5em")
                brush_slider.disable()
                # js_handler runs entirely in the browser, no server round-trip
                # (see CLIENT_JS's AIE comment for why that matters for a
                # dragged brush size) — 'update:model-value' fires continuously
                # while dragging; 'change' fires once on release.
                brush_slider.on(
                    "update:model-value",
                    js_handler=(
                        f"(value) => {{ AIE.setBrush('{editor_id}', value); AIE.showBrushPreview('{editor_id}', value); "
                        f"document.getElementById('{editor_id}_brushvalue').textContent = value; }}"
                    ),
                )
                brush_slider.on("change", js_handler=f"() => AIE.hideBrushPreview('{editor_id}')")

            with ui.column().classes("flex-1 gap-1"):
                ui.label("Feather").classes("text-xs text-gray-400")
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    feather_slider = ui.slider(min=FEATHER_RANGE.min, max=FEATHER_RANGE.max, step=FEATHER_RANGE.step, value=DEFAULT_FEATHER_AMOUNT).props("dense").classes("flex-1")
                    ui.html(f'<span id="{editor_id}_feathervalue">{DEFAULT_FEATHER_AMOUNT}</span>', sanitize=False).classes(
                        "text-xs text-gray-400"
                    ).style("min-width: 1.5em")
                feather_slider.disable()
                feather_slider.on(
                    "update:model-value",
                    js_handler=(
                        f"(value) => {{ AIE.setFeather('{editor_id}', value); "
                        f"document.getElementById('{editor_id}_feathervalue').textContent = value; }}"
                    ),
                )

            ui.button(icon="layers_clear", on_click=remove_mask).props("flat dense size=md").classes("text-xs").tooltip("Remove mask")

        annotate_radio = None
        if num_annotation_colors > 0:
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                annotate_radio = ui.element("q-radio").props('val=annotate model-value=zoom label="Annotate"')

                async def select_color(color: str) -> None:
                    for c, btn in swatches.items():
                        btn.classes(add="aie-swatch-selected" if c == color else "", remove="" if c == color else "aie-swatch-selected")
                    await call_js("AIE", "setAnnotationColor", editor_id, color)
                    await set_mode("annotate")

                for color in ANNOTATION_COLORS[:num_annotation_colors]:
                    swatches[color] = (
                        ui.button(on_click=lambda color=color: select_color(color))
                        .props("unelevated dense")
                        .classes("aie-swatch" + (" aie-swatch-selected" if color == ANNOTATION_COLORS[0] else ""))
                        .style(f"background: {color} !important")
                    )

                ui.space()
                ui.button(icon="layers_clear", on_click=remove_annotations).props("flat dense size=md").classes("text-xs").tooltip("Remove annotations")

        async def set_mode(mode: str) -> None:
            for radio in (zoom_radio, mask_radio, annotate_radio):
                if radio is not None:
                    radio.props(f"model-value={mode}")
            for slider in (brush_slider, feather_slider):
                if mode == "mask":
                    slider.enable()
                else:
                    slider.disable()
            await call_js("AIE", "setMode", editor_id, mode)

        async def select_zoom() -> None:
            await set_mode("zoom")

        async def select_mask() -> None:
            await set_mode("mask")

        async def select_annotate() -> None:
            await set_mode("annotate")

        zoom_radio.on("update:model-value", select_zoom)
        mask_radio.on("update:model-value", select_mask)
        if annotate_radio is not None:
            annotate_radio.on("update:model-value", select_annotate)

    await init_widget("AIE", editor_id, "create_mask_editor")

    async def get_inputs() -> EditorInputs:
        data = await call_js("AIE", "getInputs", editor_id, timeout=15.0) if holder["path"] else None
        if not data:
            return EditorInputs(None, None, DEFAULT_FEATHER_AMOUNT)
        return EditorInputs(
            save_data_url(data["mask"], "mask.png") if data["mask"] else None,
            save_data_url(data["annotations"], "annotations.png") if data["annotations"] else None,
            int(data["feather"]),
        )

    async def set_image(path: str) -> None:
        holder["path"] = path
        await call_js("AIE", "loadImage", editor_id, to_url(path), timeout=10.0)
        badge.set_visibility(True)

    return holder, get_inputs, set_image


async def create_compare_slider() -> Tuple[
    Callable[[str, str], Awaitable[None]],
    Callable[[bool], Awaitable[None]],
    Callable[[], Awaitable[None]],
    Callable[[], Awaitable[Optional[str]]],
]:
    """
    The result panel: a before/after image pair with a CSS clip-path
    divider dragged via a slider, kept fully client-side via js_handler.
    When comparison is off, only the after image shows. Two-finger pinch
    zooms both images together (clamped 1x-8x); once zoomed, a single
    finger pans. Both reset whenever a new result loads.

    The two <img> elements are the source of truth for "what's the current
    result" — get_after_path() reads the browser's current after-image URL
    back and resolves it to its local WORK_DIR path.

    Returns (set_images, set_compare, reset, get_after_path), all async.
    """
    slider_id = f"cmp_{uuid.uuid4().hex}"

    with ui.element("div").classes("ais-container"):
        ui.html(
            f'<div id="{slider_id}_zoomwrap" class="ais-zoomwrap">'
            f'<img id="{slider_id}_after" class="ais-after">'
            f'<img id="{slider_id}_before" class="ais-before">'
            f'<div id="{slider_id}_handle" class="ais-handle"></div>'
            f'</div>',
            sanitize=False,
        )
        # A ui.slider rather than a raw <input type="range"> — a bare <input>
        # (even type="range") is enough to pop the Android software keyboard
        # on some WebViews/browsers just from being focused. No label (label-always) here: this
        # overlays directly on the image, where a value bubble would be
        # visual noise rather than useful information. js_handler keeps
        # dragging fully client-side (see CLIENT_JS's AIE comment for why
        # that matters) — 'update:model-value' fires continuously while
        # dragging, calling AIS.applyClip directly with no server
        # round-trip.
        range_slider = ui.slider(min=0, max=100, step=1, value=50).props("dense").classes("ais-range")
        range_slider.props(f"id={slider_id}_range")
        range_slider.on(
            "update:model-value",
            js_handler=f"(value) => AIS.applyClip('{slider_id}', value)",
        )

    await init_widget("AIS", slider_id, "create_compare_slider")

    async def set_images(before_path: str, after_path: str) -> None:
        # The slider position itself is set from Python (NiceGUI's own
        # reliable value-setting API) rather than from JS — programmatic
        # value pushes don't fire the same DOM events a real drag does, so
        # there's no update:model-value for a js_handler to catch here.
        # AIS.setImages (below) still re-applies the clip/handle position
        # to match via AIS.applyClip(id, 50), since setting the Python
        # value alone doesn't touch the clip-path.
        range_slider.set_value(50)
        await call_js("AIS", "setImages", slider_id, to_url(before_path), to_url(after_path), timeout=10.0)

    async def set_compare(enabled: bool) -> None:
        await call_js("AIS", "setCompare", slider_id, bool(enabled))

    async def reset() -> None:
        await call_js("AIS", "reset", slider_id)

    async def get_after_path() -> Optional[str]:
        url = await call_js("AIS", "getAfterUrl", slider_id)
        return from_url(url) if url else None

    return set_images, set_compare, reset, get_after_path

# --- Page ---

# Every top-level section on the page (Prompt/Seed, Aspect ratio + editor,
# the settings accordions, the Result panel) is wrapped in a ui.card using
# this same class string, so their content all sits flush-aligned with each
# other and none of them indents by a different amount than its neighbors.
# q-pa-none strips QCard's default padding (Quasar's own utility class, not
# a NiceGUI quirk) — genuinely needed, not just a style choice: the editor
# canvas (.aie-editor-box) and the result comparison image (.ais-container)
# are both width: 100% against their containing card, so the card's default
# padding would inset them and shrink the actual touch-drawing/viewing area
# on a small screen. Applied to all cards rather than just those two so
# nothing looks oddly inset next to the ones that need it.
CARD_CLASSES = "w-full q-pa-none"


def run(model: WorkflowModel) -> None:
    # Everything below is model-dependent, so it lives inside run()
    # rather than at module scope: app.py builds the model once and
    # calls this through ui.run()
    # (see app.py and ui/__init__.py).

    @ui.page("/")
    async def main_page() -> None:
        # add_head_html must run while the page is still being assembled — it
        # only affects the initial HTML document, so it has to happen before
        # the client connects below (once connected, the document this browser
        # already has is fixed; adding to <head> after that point has no
        # effect, which would silently drop the CLIENT_JS script and WIDGET_CSS
        # stylesheet entirely).
        ui.add_head_html(f"<style>{WIDGET_CSS}</style><script>{CLIENT_JS}</script>")
        apply_dark_theme()

        # ui.run_javascript() already awaits the client connection internally
        # before executing (since NiceGUI 3.0), so this isn't needed to make
        # those calls work. It's kept because of how NiceGUI routes exceptions
        # (see the "Error handling" section of the docs): anything raised
        # before this point is page-blocking and produces a full 500 error
        # page; after it, exceptions go to ui.on_exception instead, scoped to
        # just this client.
        await ui.context.client.connected()

        # Read once per page load. Everything below renders or hides controls
        # based on this — nothing in this function assumes any one model.
        caps = model.capabilities

        ui.label("AI Image Edit").classes("text-2xl font-bold text-white w-full text-center")
        with ui.column().classes("w-full gap-0 items-center q-mb-md"):
            ui.label(f"Model: {model.display_name}").classes("text-sm w-full text-center").style(f"color: {PRIMARY_COLOR}")
            setup_label = ui.label().classes("text-xs w-full text-center")
            setup_label.set_visibility(False)
            setup_bar = ui.linear_progress(value=0, show_value=False, size="4px").props("rounded").classes("w-full q-mt-xs")
            setup_bar.set_visibility(False)

        with ui.column().classes("w-full gap-3 aie-page"):
            with ui.card().classes(CARD_CLASSES):
                prompt = ui.textarea(label="Prompt").props("rows=6 outlined dark").classes("w-full")

                with ui.row().classes("w-full items-center gap-4"):
                    seed_input = ui.number(label="Seed", value=DEFAULT_SEED, format="%d").props("outlined dark").classes("flex-1")
                    randomize_seed = ui.switch("Randomize seed", value=True)
                    if not caps.supports_seed:
                        seed_input.disable()
                        randomize_seed.value = False
                        randomize_seed.disable()

            # Aspect ratio + resolution + the unified image/mask editor. As soon
            # as a mask is drawn on a model that supports inpainting, dimensions
            # must come from the source image's own aspect ratio (a mask's
            # coordinates are only meaningful relative to the source's own
            # framing), so both Aspect ratio and Resolution are disabled while a
            # mask exists — Aspect ratio additionally force-set to "Original",
            # Resolution set to the tier the model will use for the image
            # (model.megapixels_for_source) — only when the model actually
            # offers "Original" as a choice, and restored the moment the mask
            # is removed.
            with ui.card().classes(CARD_CLASSES):
                with ui.row().classes("w-full items-center gap-4"):
                    aspect_ratio = ui.select(
                        caps.supported_aspect_ratios, value=caps.default_aspect_ratio, label="Aspect ratio"
                    ).props("outlined dark").classes("flex-1")

                    megapixel_options = {mp: f"{mp:g} MP" for mp in caps.supported_megapixels}
                    megapixels = ui.select(
                        megapixel_options, value=caps.default_megapixels, label="Resolution"
                    ).props("outlined dark").classes("flex-1")
                    if len(caps.supported_megapixels) == 1:
                        # Nothing to choose — showing a disabled single-option
                        # dropdown says "this model only supports 1MP" more
                        # clearly than a dropdown that looks pickable but isn't.
                        megapixels.disable()

                previous_aspect_ratio = aspect_ratio.value
                previous_megapixels = megapixels.value

                def handle_mask_change(has_mask: bool) -> None:
                    nonlocal previous_aspect_ratio, previous_megapixels
                    if ORIGINAL_ASPECT_RATIO not in caps.supported_aspect_ratios:
                        return
                    if has_mask:
                        if aspect_ratio.value != ORIGINAL_ASPECT_RATIO:
                            previous_aspect_ratio = aspect_ratio.value
                        aspect_ratio.value = ORIGINAL_ASPECT_RATIO
                        aspect_ratio.disable()
                        # Shows the tier the model will use: it follows the image.
                        source_mp = model.megapixels_for_source(editor_holder["path"]) if editor_holder["path"] else None
                        if source_mp is not None:
                            previous_megapixels = megapixels.value
                            megapixels.value = source_mp
                        megapixels.disable()
                    else:
                        aspect_ratio.enable()
                        aspect_ratio.value = previous_aspect_ratio
                        megapixels.value = previous_megapixels
                        if len(caps.supported_megapixels) > 1:
                            # Otherwise it was already permanently disabled above
                            # (nothing to choose), independent of any mask.
                            megapixels.enable()

                editor_holder, get_editor_inputs, set_editor_image = await create_mask_editor(
                    on_mask_change=handle_mask_change if caps.supports_inpainting else None,
                    num_annotation_colors=caps.num_annotation_colors,
                )

            reference_holders: List[dict] = []
            if caps.max_reference_images > 1 or caps.supports_loras:
                with ui.card().classes(CARD_CLASSES):
                    if caps.max_reference_images > 1:
                        with ui.expansion("Reference images (optional)").props("dense").classes("w-full"):
                            for i in range(caps.max_reference_images - 1):
                                reference_holders.append(create_simple_image_upload(f"Input image {i + 2}"))

                            # Only the first slot shows; + / - reveal or hide
                            # the rest. Hiding a slot also clears its image, so
                            # nothing invisible is sent with the request.
                            ref_state = {"visible": 1}
                            for holder in reference_holders[1:]:
                                holder["container"].set_visibility(False)

                            with ui.row().classes("w-full items-center gap-2"):
                                ref_remove_btn = ui.button(icon="remove").props("flat dense round size=sm")
                                ref_add_btn = ui.button(icon="add").props("flat dense round size=sm")
                                ref_count_label = ui.label().classes("text-xs text-gray-400")

                            def update_reference_slots() -> None:
                                total = len(reference_holders)
                                ref_count_label.set_text(f"{ref_state['visible']} of {total}")
                                ref_remove_btn.set_enabled(ref_state["visible"] > 1)
                                ref_add_btn.set_enabled(ref_state["visible"] < total)

                            def add_reference_slot() -> None:
                                if ref_state["visible"] < len(reference_holders):
                                    reference_holders[ref_state["visible"]]["container"].set_visibility(True)
                                    ref_state["visible"] += 1
                                    update_reference_slots()

                            def remove_reference_slot() -> None:
                                if ref_state["visible"] > 1:
                                    ref_state["visible"] -= 1
                                    holder = reference_holders[ref_state["visible"]]
                                    holder["clear"]()
                                    holder["container"].set_visibility(False)
                                    update_reference_slots()

                            ref_add_btn.on_click(add_reference_slot)
                            ref_remove_btn.on_click(remove_reference_slot)
                            update_reference_slots()

                    available_loras: Dict[str, List[str]] = {}
                    lora_name = None
                    lora_strength = None
                    if caps.supports_loras:
                        with ui.expansion("LoRAs").props("dense").classes("w-full"):
                            available_loras = model.list_loras()
                            lora_name = ui.select(["None"] + list(available_loras.keys()), value="None", label="Name").props("outlined dark").classes("w-full")
                            ui.label("Strength").classes("text-xs text-gray-400 q-mt-sm")
                            ls = caps.lora_strength_range
                            lora_strength = create_value_slider(ls)

            with ui.card().classes(CARD_CLASSES):
                with ui.expansion("Advanced settings").props("dense").classes("w-full"):
                    sr = caps.step_range
                    ui.label("Inference steps").classes("text-xs text-gray-400")
                    steps = create_value_slider(sr)

                    cfg = None
                    if caps.supports_cfg:
                        cr = caps.cfg_range
                        ui.label("CFG scale").classes("text-xs text-gray-400 q-mt-sm")
                        cfg = create_value_slider(cr)

                    negative_prompt = None
                    if caps.supports_negative_prompt:
                        negative_prompt = ui.textarea(label="Negative prompt").props("rows=2 outlined dark").classes("w-full q-mt-sm")

                    denoise = None
                    if caps.supports_denoise:
                        dr = caps.denoise_range
                        ui.label("Denoise").classes("text-xs text-gray-400 q-mt-sm")
                        denoise = create_value_slider(dr)

                    sampler_name = None
                    if caps.sampler_choices:
                        sampler_name = ui.select(caps.sampler_choices, value=default_choice(caps.default_sampler, caps.sampler_choices), label="Sampler name").props("outlined dark").classes("w-full q-mt-sm")

                    scheduler = None
                    if caps.scheduler_choices:
                        scheduler = ui.select(caps.scheduler_choices, value=default_choice(caps.default_scheduler, caps.scheduler_choices), label="Scheduler").props("outlined dark").classes("w-full")

                    apply_color_correction_switch = ui.switch("Apply color corrections", value=False).classes("q-mt-sm")

            # --- Result panel ---
            with ui.card().classes(CARD_CLASSES):
                set_result_images, set_compare, reset_result, get_result_after_path = await create_compare_slider()

                compare_switch = ui.switch("Compare input / output", value=False)
                compare_switch.disable()

                async def on_compare_change(e) -> None:
                    await set_compare(e.value)

                compare_switch.on_value_change(on_compare_change)

                with ui.row().classes("w-full items-center gap-3"):
                    use_as_input_btn = ui.button("Use as input").props("outline").classes("flex-1")
                    use_as_input_btn.disable()

                    download_btn = ui.button("Download result").props("outline").classes("flex-1")
                    download_btn.disable()

            generate_btn = ui.button("Generate").props("color=primary unelevated").classes("w-full")

        async def do_generate() -> None:
            generate_btn.props("loading")
            generate_btn.disable()
            compare_switch.disable()
            use_as_input_btn.disable()
            download_btn.disable()
            await reset_result()
            await set_compare(compare_switch.value)
            try:
                if not editor_holder["path"]:
                    ui.notify("Please upload an image to edit!", type="negative")
                    return

                source_image_path = editor_holder["path"]
                editor_inputs = await get_editor_inputs()
                annotated_image_path = None
                if caps.num_annotation_colors > 0 and editor_inputs.annotation_layer_path:
                    annotated_image_path = await nicegui_run.io_bound(
                        imaging.compose_annotations, source_image_path, editor_inputs.annotation_layer_path
                    )
                lora_files = available_loras.get(lora_name.value, []) if lora_name is not None else []

                params = params_from_ui(
                    caps,
                    prompt=prompt.value,
                    source_image_path=source_image_path,
                    mask_path=editor_inputs.mask_path,
                    annotated_image_path=annotated_image_path,
                    reference_images=[h["path"] for h in reference_holders if h["path"]],
                    seed=seed_input.value,
                    randomize_seed=randomize_seed.value,
                    aspect_ratio=aspect_ratio.value,
                    target_megapixels=megapixels.value,
                    steps=steps.value,
                    cfg=cfg.value if cfg is not None else None,
                    denoise=denoise.value if denoise is not None else None,
                    sampler_name=sampler_name.value if sampler_name is not None else None,
                    scheduler=scheduler.value if scheduler is not None else None,
                    negative_prompt=negative_prompt.value if negative_prompt is not None else None,
                    lora_files=lora_files,
                    lora_strength=lora_strength.value if lora_strength is not None else None,
                    apply_color_correction=apply_color_correction_switch.value,
                    feather_amount=editor_inputs.feather_amount,
                )
                result = await nicegui_run.io_bound(model.generate, params)
                await nicegui_run.io_bound(trim_work_dir, RECENT_FILES_KEPT)

                seed_input.value = result.actual_seed
                await set_result_images(result.before_path, result.after_path)
                compare_switch.enable()
                use_as_input_btn.enable()
                download_btn.enable()

            except Exception as e:  # noqa: BLE001 — surface unexpected errors instead of hanging silently
                ui.notify(describe_error(e), type="negative")
            finally:
                generate_btn.props(remove="loading")
                generate_btn.enable()

        generate_btn.on_click(do_generate)

        async def use_as_input() -> None:
            after_path = await get_result_after_path()
            if not after_path:
                return

            await set_editor_image(after_path)

            # The previous before/after pair no longer corresponds to the input
            # that's now in place, so clear it and re-disable the controls that
            # depend on a fresh result. The Compare toggle's own checked/
            # unchecked state is left alone (not the visual divider, which
            # reset() unconditionally hides) — reapplying it right after
            # restores the divider immediately if it was checked, matching the
            # same reset-then-reapply approach as do_generate.
            await reset_result()
            await set_compare(compare_switch.value)
            compare_switch.disable()
            use_as_input_btn.disable()
            download_btn.disable()

        use_as_input_btn.on_click(use_as_input)

        # While the model sets itself up in the background (weights and LoRAs
        # downloading, custom nodes installing) show how far it is, and add the
        # LoRAs to the list as their files arrive.
        def refresh_setup() -> None:
            progress = model.setup_progress()
            text = progress.error or progress.message
            setup_label.set_text(text)
            setup_label.set_visibility(bool(text))
            setup_label.classes(add="text-negative" if progress.error else "text-gray-400",
                                remove="text-gray-400" if progress.error else "text-negative")
            running = not progress.done
            setup_bar.set_visibility(running)
            if running:
                if progress.fraction is None:
                    setup_bar.props("indeterminate")
                else:
                    setup_bar.props(remove="indeterminate")
                    setup_bar.set_value(progress.fraction)
            if lora_name is not None:
                found = model.list_loras()
                if found != available_loras:
                    available_loras.clear()
                    available_loras.update(found)
                    lora_name.set_options(["None"] + list(found), value=lora_name.value if lora_name.value in found else "None")
            if progress.done:
                setup_timer.deactivate()

        setup_timer = ui.timer(1.0, refresh_setup)  # stops itself once the setup is done
        refresh_setup()

        async def download_result() -> None:
            after_path = await get_result_after_path()
            if not after_path:
                ui.notify("No generated image to download yet.", type="negative")
                return
            ui.download.file(after_path, filename=Path(after_path).name)

        download_btn.on_click(download_result)


    # Serve everything under WORK_DIR at /files/<n> so <img>/<canvas> loads in
    # the browser can reference uploaded/generated files directly by URL.
    app.add_static_files("/files", str(WORK_DIR))

    # Make sure whatever the active model started (a ComfyUI subprocess, ...)
    # gets torn down on shutdown instead of leaking.
    app.on_shutdown(model.shutdown)

    # APP_PASSWORD turns the login on; without it the app is open. app.py
    # refuses to start when REQUIRE_PASSWORD is set but APP_PASSWORD is not.
    password = os.environ.get("APP_PASSWORD", "")
    storage_secret = install_password_login(password) if password else None

    # reload=False is important here: app.main() already started whatever the
    # active model needs (e.g. ComfyUI) once, before this — NiceGUI's
    # auto-reloader re-executes the app in a subprocess, which would
    # launch a second instance on the same port if left on.
    #
    # reconnect_timeout is optional — NiceGUI works fine without it,
    # defaulting to 3 seconds. It governs the one websocket connection that
    # this whole app runs over (not something specific to one feature): every
    # click, slider/checkbox change, ui.run_javascript() call, and file upload
    # flows over it in both directions, since NiceGUI keeps all UI state
    # server-side and the browser just renders it. reconnect_timeout is how
    # long the server keeps a client's session alive after that connection
    # drops before discarding it and forcing a full reload on return. Mobile
    # browsers (Android and iOS) kill a backgrounded tab's websocket outright,
    # so returning to the tab after even a short time away needs a reconnect,
    # and the 3s default is often too short for that. 300s (5 minutes) trades a
    # bit of server-side cleanup delay for genuinely abandoned sessions
    # (negligible at this app's likely traffic) for a lot more headroom on
    # "switched apps and came back later" — an untested, reversible tuning
    # choice, not a required setting. It also can't fully fix the reload: some
    # backgrounding will always outlast any timeout, and a NiceGUI bug already
    # tracks a related over-eager reload-on-reconnect case
    # (github.com/zauberzeug/nicegui/issues/6018) — worth an upgrade if this
    # keeps happening even for quick tab switches.
    ui.run(
        host=HOST,
        port=PORT,
        title="AI Image Edit",
        dark=True,
        reload=False,
        reconnect_timeout=300,
        storage_secret=storage_secret,
    )
