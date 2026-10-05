# SPDX-License-Identifier: GPL-3.0-or-later
"""
NiceGUI frontend. Entry point is run(model), called by
app.py — importing this module does nothing by itself.

UI-only: renders controls, collects a GenerationParams, and calls
model.generate(). Controls (mask editor, before/after slider, reference
slots, LoRA panel, sampler/scheduler/cfg/denoise, aspect-ratio/resolution)
are shown or hidden based on the active model's declared capabilities.
"""

import asyncio
import base64
import hashlib
import hmac
import os
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR, to_url, from_url
from ai_image_edit.core.types import ANNOTATION_COLORS, GenerationParams
from ai_image_edit.models.base import ModelBackend


# --- Configuration Constants ---

# Brush min/max stay hardcoded on the slider itself (5-80) since nothing
# else in the app references them — only the *default* needs a shared
# constant, since it's duplicated into CLIENT_JS's initial JS-side state
# (see the __DEFAULT_BRUSH_SIZE__ substitution below CLIENT_JS) as well as
# the Python-side ui.slider.
DEFAULT_BRUSH_SIZE = 48

# Same idea, but feather has no JS-side counterpart to keep in sync (it's
# only ever read server-side via get_feather_amount() at Generate time) —
# named for consistency/discoverability rather than to keep two literals
# from drifting apart.
DEFAULT_FEATHER_AMOUNT = 6

# Annotation stroke width in on-screen CSS pixels (converted to image pixels
# at draw time, like the brush). Thin and fixed: there is no slider.
ANNOTATION_STROKE = 3

# This app's one accent color — the "Model: ..." subtitle's text color and
# ui.colors()'s primary (buttons, the active-tab/slider color, etc.) both
# read from here so they can't drift apart.
PRIMARY_COLOR = "#7c3aed"


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

        # Only same-site paths: "//host" would redirect off-site.
        target = redirect_to if redirect_to.startswith("/") and not redirect_to.startswith("//") else "/"

        async def try_login() -> None:
            if hmac.compare_digest(field.value.encode(), password.encode()):
                app.storage.user["authenticated"] = True
                ui.navigate.to(target)
            else:
                await asyncio.sleep(1.0)
                ui.notify("Wrong password", color="negative")

        ui.dark_mode().enable()
        ui.colors(primary=PRIMARY_COLOR, dark="#111111", dark_page="#000000")
        with ui.card().classes("absolute-center items-stretch"):
            field = ui.input("Password", password=True, password_toggle_button=True).props("autofocus")
            field.on("keydown.enter", try_login)
            ui.button("Log in", on_click=try_login)
        return None

    return os.environ.get("APP_STORAGE_SECRET") or hashlib.sha256(f"ai-image-edit-session:{password}".encode()).hexdigest()


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
    uploader = ui.upload(auto_upload=True, max_files=1).props('accept="*/*"')
    uploader.set_visibility(False)
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
    rather than raw HTML so a NiceGUI click handler (box.on("click", ...))
    can be attached to it directly.

    Returns holder — holder['path'] always reflects the currently selected
    file's local path (or None). holder['container'] is the slot's element
    (to show or hide it) and holder['clear'] removes the selected image.
    """
    holder: dict = {"path": None}

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("text-sm text-gray-400")
        with ui.element("div").classes("aie-upload-box") as box:
            placeholder = ui.label("Click to upload an image").classes("aie-placeholder")
            preview = ui.image().classes("aie-preview")
            preview.set_visibility(False)

        uploader = create_hidden_uploader()

        async def open_picker() -> None:
            await uploader.run_method("pickFiles", timeout=5.0)

        box.on("click", open_picker)

        def clear_image() -> None:
            holder["path"] = None
            preview.set_visibility(False)
            placeholder.set_visibility(True)

        ui.button("Clear", on_click=clear_image).props("flat dense size=sm").classes("text-xs")

    holder["container"] = container
    holder["clear"] = clear_image

    async def handle_upload(e) -> None:
        path = await save_uploaded_file(e.file)
        holder["path"] = path
        preview.set_source(to_url(path))
        preview.set_visibility(True)
        placeholder.set_visibility(False)
        uploader.reset()

    uploader.on_upload(handle_upload)

    return holder

async def create_mask_editor(
    on_mask_change: Optional[Callable[[bool], None]] = None,
    num_annotation_colors: int = 0,
) -> Tuple[
    dict,
    Callable[[], Awaitable[Optional[str]]],
    Callable[[str], Awaitable[None]],
    Callable[[], int],
    Callable[[], Awaitable[Optional[str]]],
]:
    """
    The "Input Image" widget: click-to-upload image + a <canvas> for
    drawing an optional inpainting mask, with a resizable brush. A radio
    group switches between pinch-to-zoom (default), mask drawing and,
    when num_annotation_colors > 0, annotating in one of that many colours
    (ANNOTATION_COLORS, thin fixed stroke). Drawing a stroke in mask mode
    makes this an Inpaint Edit, "Remove mask" reverts to a plain Image Edit. The Feather slider (0-16,
    default 6) is the Gaussian blur radius composite_with_soft_transition()
    applies to the mask edges; 0 falls back to a hard cutout.

    If given, on_mask_change(has_mask) fires whenever the mask goes from
    empty to non-empty or back (via the 'aie_mask_state' custom event).

    Returns (holder, get_mask_path, set_image, get_feather_amount,
    get_annotation_layer_path):
      * holder['path'] is the current source image's local path (or None).
      * get_mask_path() is async, reads the mask canvas and saves it to
        disk, or returns None if nothing was drawn. Called once, on Generate.
      * set_image(path) loads a new image and clears the mask.
      * get_feather_amount() reads the Feather slider's current value.
      * get_annotation_layer_path() is async, saves the annotation layer
        (transparent PNG, annotations only) or returns None if none were drawn.
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

    with ui.element("div").classes("aie-editor-box") as editor_box:
        ui.html(
            f'<div id="{editor_id}_zoomwrap" class="aie-zoomwrap">'
            f'<canvas id="{editor_id}_canvas" class="aie-canvas"></canvas>'
            f'<canvas id="{editor_id}_annot" class="aie-annot"></canvas>'
            f'<canvas id="{editor_id}_overlay" class="aie-overlay"></canvas>'
            f'<div id="{editor_id}_placeholder" class="aie-placeholder">Click to upload an image</div>'
            f'</div>'
            f'<div id="{editor_id}_brushpreview" class="aie-brush-preview"></div>',
            sanitize=False,
        )

    uploader = create_hidden_uploader()

    async def open_picker_if_empty() -> None:
        # Once an image is loaded, a click on the box should draw on the
        # canvas instead of reopening the picker — holder['path'] already
        # tells us which state we're in, no need to ask the browser.
        if not holder["path"]:
            await uploader.run_method("pickFiles", timeout=5.0)

    editor_box.on("click", open_picker_if_empty)

    async def handle_upload(e) -> None:
        path = await save_uploaded_file(e.file)
        holder["path"] = path
        await ui.run_javascript(f"AIE.loadImage('{editor_id}', '{to_url(path)}')", timeout=10.0)
        uploader.reset()

    uploader.on_upload(handle_upload)

    async def clear_image() -> None:
        holder["path"] = None
        await ui.run_javascript(f"AIE.clearImage('{editor_id}')", timeout=5.0)

    ui.button("Clear", on_click=clear_image).props("flat dense size=sm").classes("text-xs")

    async def remove_mask() -> None:
        await ui.run_javascript(f"AIE.clearMask('{editor_id}')", timeout=5.0)

    async def remove_annotations() -> None:
        await ui.run_javascript(f"AIE.clearAnnotations('{editor_id}')", timeout=5.0)

    swatches: Dict[str, ui.button] = {}

    with ui.column().classes("w-full gap-1 p-2 bg-neutral-900 rounded-lg"):
        # One q-radio per row (ui.radio is a single vertical option list and
        # cannot hold the sliders), all showing the same current mode.
        zoom_radio = ui.element("q-radio").props('val=zoom model-value=zoom label="Pinch to zoom"')

        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            mask_radio = ui.element("q-radio").props('val=mask model-value=zoom label="Mask"')

            with ui.column().classes("flex-1 gap-3"):
                ui.label("Brush size").classes("text-xs text-gray-400")
                brush_slider = ui.slider(min=5, max=80, step=1, value=DEFAULT_BRUSH_SIZE).props("label-always dense")
                brush_slider.disable()
                # js_handler runs entirely in the browser, no server round-trip
                # (see CLIENT_JS's AIE comment for why that matters for a
                # dragged brush size) — 'update:model-value' fires continuously
                # while dragging; 'change' fires once on release.
                brush_slider.on(
                    "update:model-value",
                    js_handler=f"(value) => {{ AIE.setBrush('{editor_id}', value); AIE.showBrushPreview('{editor_id}', value); }}",
                )
                brush_slider.on("change", js_handler=f"() => AIE.hideBrushPreview('{editor_id}')")

            with ui.column().classes("flex-1 gap-3"):
                ui.label("Feather").classes("text-xs text-gray-400")
                feather_slider = ui.slider(min=0, max=16, step=1, value=DEFAULT_FEATHER_AMOUNT).props("label-always dense")
                feather_slider.disable()

            ui.button(icon="layers_clear", on_click=remove_mask).props("flat dense size=md").classes("text-xs").tooltip("Remove mask")

        annotate_radio = None
        if num_annotation_colors > 0:
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                annotate_radio = ui.element("q-radio").props('val=annotate model-value=zoom label="Annotate"')

                async def select_color(color: str) -> None:
                    for c, btn in swatches.items():
                        btn.classes(add="aie-swatch-selected" if c == color else "", remove="" if c == color else "aie-swatch-selected")
                    await ui.run_javascript(f"AIE.setAnnotationColor('{editor_id}', '{color}')", timeout=5.0)
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
            await ui.run_javascript(f"AIE.setMode('{editor_id}', '{mode}')", timeout=5.0)

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

    try:
        await ui.run_javascript(f"AIE.init('{editor_id}')", timeout=10.0)
    except TimeoutError:
        print(f"[create_mask_editor] AIE.init('{editor_id}') timed out — the editor may not respond until the page is reloaded.", flush=True)

    def get_feather_amount() -> int:
        return int(feather_slider.value)

    async def get_mask_path() -> Optional[str]:
        if not holder["path"]:
            return None
        data_url = await ui.run_javascript(f"AIE.getMaskDataUrl('{editor_id}')", timeout=15.0)
        if not data_url:
            return None
        return save_data_url(data_url, "mask.png")

    async def get_annotation_layer_path() -> Optional[str]:
        if not holder["path"]:
            return None
        data_url = await ui.run_javascript(f"AIE.getAnnotationLayerDataUrl('{editor_id}')", timeout=15.0)
        if not data_url:
            return None
        return save_data_url(data_url, "annotations.png")

    async def set_image(path: str) -> None:
        holder["path"] = path
        await ui.run_javascript(
            f"AIE.loadImage('{editor_id}', '{to_url(path)}')",
            timeout=10.0,
        )

    return holder, get_mask_path, set_image, get_feather_amount, get_annotation_layer_path

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
        # on some WebViews/browsers just from being focused. No label (label-always) here: unlike Brush/Feather, this
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

    try:
        await ui.run_javascript(f"AIS.init('{slider_id}')", timeout=10.0)
    except TimeoutError:
        print(f"[create_compare_slider] AIS.init('{slider_id}') timed out — the compare slider may not respond until the page is reloaded.", flush=True)

    async def set_images(before_path: str, after_path: str) -> None:
        # The slider position itself is set from Python (NiceGUI's own
        # reliable value-setting API) rather than from JS — programmatic
        # value pushes don't fire the same DOM events a real drag does, so
        # there's no update:model-value for a js_handler to catch here.
        # AIS.setImages (below) still re-applies the clip/handle position
        # to match via AIS.applyClip(id, 50), since setting the Python
        # value alone doesn't touch the clip-path.
        range_slider.set_value(50)
        await ui.run_javascript(
            f"AIS.setImages('{slider_id}', '{to_url(before_path)}', '{to_url(after_path)}')",
            timeout=10.0,
        )

    async def set_compare(enabled: bool) -> None:
        await ui.run_javascript(f"AIS.setCompare('{slider_id}', {str(bool(enabled)).lower()})", timeout=5.0)

    async def reset() -> None:
        await ui.run_javascript(f"AIS.reset('{slider_id}')", timeout=5.0)

    async def get_after_path() -> Optional[str]:
        url = await ui.run_javascript(f"AIS.getAfterUrl('{slider_id}')", timeout=5.0)
        return from_url(url) if url else None

    return set_images, set_compare, reset, get_after_path

# --- Custom stylesheet ---
# Named classes for this app's bespoke widget markup (the mask editor,
# upload boxes, and comparison slider) — injected once via ui.add_head_html,
# the same mechanism already used for CLIENT_JS. These are ordinary classes
# this app defines itself, so — unlike NiceGUI's bundled Tailwind utility
# classes, which only cover whatever subset NiceGUI happens to ship — their
# existence is never in question, and there's exactly one place to look if
# something needs to change. Values that get overridden dynamically at
# runtime (AIS's clip-path/left/transform, AIE's brush-preview size) are
# still set here as sensible defaults; a plain inline style set directly via
# JS (element.style.foo = ...) always takes precedence over a class
# regardless, so the two coexist without conflict. These rules also don't
# need !important to beat Quasar's own defaults: NiceGUI 3.0 moved Quasar's
# base styles into a CSS @layer, and per the cascade spec, any rule outside
# a layer — including this plain <style> block — already outranks layered
# rules regardless of specificity.
WIDGET_CSS = """
.aie-upload-box {
    position: relative;
    width: 100%;
    min-height: 160px;
    cursor: pointer;
    border: 1px solid #444;
    border-radius: 8px;
    background: #000;
    overflow: hidden;
}

.aie-preview {
    display: block;
    width: 100%;
    height: auto;
}

.aie-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #888;
    font-family: sans-serif;
    font-size: 14px;
    pointer-events: none;
}

.aie-editor-box {
    position: relative;
    width: 100%;
    min-height: 120px;
    cursor: pointer;
    overflow: hidden;
}

.aie-zoomwrap {
    position: relative;
    width: 100%;
    touch-action: none;
    transform-origin: 50% 50%;
}

.aie-canvas {
    display: block;
    width: 100%;
    height: auto;
    touch-action: none;
    background: #000;
    border: 1px solid #333;
    border-radius: 8px;
}

.aie-swatch {
    width: 28px;
    height: 28px;
    min-width: 28px;
    min-height: 28px;
    border-radius: 6px;
    border: 2px solid transparent;
    padding: 0;
}

.aie-swatch-selected {
    border-color: #ffffff;
    box-shadow: 0 0 0 2px #7c3aed;
}

.aie-annot {
    position: absolute;
    inset: 0;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    border: 1px solid transparent;
    pointer-events: none;
}

.aie-overlay {
    position: absolute;
    inset: 0;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    border: 1px solid transparent;
    opacity: 0.5;
    pointer-events: none;
}

.aie-brush-preview {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border: 2px solid rgba(255, 255, 255, 0.9);
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.15);
    pointer-events: none;
    display: none;
}

.ais-container {
    position: relative;
    width: 100%;
    min-height: 200px;
    background: #000;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
}

.ais-zoomwrap {
    position: relative;
    width: 100%;
    touch-action: none;
    transform-origin: 50% 50%;
}

.ais-after {
    display: none;
    width: 100%;
    height: auto;
    min-height: 200px;
}

.ais-before {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    clip-path: inset(0 100% 0 0);
    display: none;
}

.ais-handle {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 100%;
    width: 2px;
    background: #fff;
    pointer-events: none;
    display: none;
}

.ais-range {
    position: absolute;
    left: 8px;
    right: 8px;
    bottom: 8px;
    width: calc(100% - 16px);
    display: none;
}

.aie-page {
    max-width: 720px;
    margin: 0 auto;
}
"""

# --- Client-side JS ---
# Three small namespaces, defined once as generic functions keyed by an
# element-id prefix so each widget instance just calls e.g. AIE.init('some_id')
# after creating its own DOM elements with matching ids. Everything that
# NiceGUI already has a documented element/API for — file pickers (ui.upload
# + element.run_method('pickFiles'), the same mechanism NiceGUI's own
# Upload.reset() uses internally), image previews (ui.image.set_source) —
# uses that instead of custom JS. What's left needs custom JS because
# there's no NiceGUI equivalent:
#
#   AIE — the mask/image editor: a <canvas> drawn on with mouse/touch
#         handlers. NiceGUI's ui.interactive_image can report mouse
#         coordinates, but it round-trips every single mouse-move event to
#         the server to do so, which would make drawing feel laggy. A raw
#         <canvas> keeps every brush stroke fully client-side; the Brush
#         size slider is a regular ui.slider, but wired up with js_handler
#         (see create_mask_editor) so it still calls AIE.setBrush directly
#         in the browser, with no server round-trip. When a mask goes from
#         empty to non-empty or back, AIE pushes that as an 'aie_mask_state'
#         event via emitEvent(...) — the documented pattern for a
#         browser-side state change to reach Python without being tied to
#         one specific DOM event (see the "Custom events" section of the
#         Generic Events docs) — which create_mask_editor listens for with
#         ui.on(...) and passes on to its on_mask_change callback, used to
#         keep Aspect ratio and Resolution in sync.
#
#   AIS — the before/after comparison: a CSS clip-path dragged by a
#         ui.slider (via js_handler), entirely client-side.
#
#   AIU — helpers shared by the two (pan/zoom, resetting the mask and annotations).
CLIENT_JS = r"""
window.AIU = window.AIU || {};
window.AIE = window.AIE || {};
window.AIS = window.AIS || {};

// Shared helpers.
AIU.applyPanZoom = function (wrap, state) {
    wrap.style.transform = 'translate(' + state.panX + 'px, ' + state.panY + 'px) scale(' + state.scale + ')';
};

AIU.resetPanZoom = function (wrap, state) {
    state.scale = 1;
    state.panX = 0;
    state.panY = 0;
    if (wrap) AIU.applyPanZoom(wrap, state);
};

// Pan/zoom shared by the editor and the comparison slider: two-finger pinch
// or Ctrl/Cmd + mouse wheel (which is also what a trackpad pinch sends)
// zooms wrap (clamped 1x-8x), and once zoomed in a single finger or a mouse
// drag pans. A plain wheel is left alone so the page still scrolls.
// state holds scale/panX/panY; enabled() gates the gestures.
AIU.attachPanZoom = function (wrap, state, enabled) {
    function distance(touches) {
        return Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
    }
    function clampPan() {
        // offsetWidth/offsetHeight are the element's own layout size, which
        // a transform on that same element never affects — using
        // getBoundingClientRect() here instead would read back (possibly
        // stale) post-transform geometry, since this runs before the new
        // scale has actually been applied to the DOM.
        const maxX = wrap.offsetWidth * (state.scale - 1) / 2;
        const maxY = wrap.offsetHeight * (state.scale - 1) / 2;
        state.panX = Math.min(maxX, Math.max(-maxX, state.panX));
        state.panY = Math.min(maxY, Math.max(-maxY, state.panY));
    }

    let pinchStartDist = null;
    let pinchStartScale = 1;
    let panStart = null;

    wrap.addEventListener('touchstart', function (e) {
        if (!enabled()) return;
        if (e.touches.length === 2) {
            pinchStartDist = distance(e.touches);
            pinchStartScale = state.scale;
            panStart = null;
        } else if (e.touches.length === 1 && state.scale > 1) {
            panStart = { x: state.panX, y: state.panY, tx: e.touches[0].clientX, ty: e.touches[0].clientY };
        }
    }, { passive: true });

    wrap.addEventListener('touchmove', function (e) {
        if (!enabled()) return;
        if (e.touches.length === 2 && pinchStartDist) {
            state.scale = Math.min(Math.max(pinchStartScale * distance(e.touches) / pinchStartDist, 1), 8);
        } else if (e.touches.length === 1 && panStart) {
            state.panX = panStart.x + (e.touches[0].clientX - panStart.tx);
            state.panY = panStart.y + (e.touches[0].clientY - panStart.ty);
        } else {
            return;
        }
        e.preventDefault();
        clampPan();
        AIU.applyPanZoom(wrap, state);
    }, { passive: false });

    wrap.addEventListener('touchend', function (e) {
        if (e.touches.length < 2) pinchStartDist = null;
        if (e.touches.length < 1) panStart = null;
    });

    wrap.addEventListener('wheel', function (e) {
        if (!enabled() || !(e.ctrlKey || e.metaKey)) return;
        e.preventDefault();
        // Zoom about the cursor: scaling happens about the element's centre,
        // which the translation moves by pan, so the cursor is measured from
        // the untransformed centre.
        const rect = wrap.getBoundingClientRect();
        const cx = e.clientX - (rect.left + rect.width / 2) + state.panX;
        const cy = e.clientY - (rect.top + rect.height / 2) + state.panY;
        const next = Math.min(Math.max(state.scale * Math.exp(-e.deltaY * 0.002), 1), 8);
        const ratio = next / state.scale;
        state.panX = cx - (cx - state.panX) * ratio;
        state.panY = cy - (cy - state.panY) * ratio;
        state.scale = next;
        clampPan();
        AIU.applyPanZoom(wrap, state);
    }, { passive: false });

    let dragStart = null;
    wrap.addEventListener('mousedown', function (e) {
        if (!enabled() || e.button !== 0 || state.scale <= 1) return;
        e.preventDefault();
        dragStart = { x: state.panX, y: state.panY, mx: e.clientX, my: e.clientY };
    });
    window.addEventListener('mousemove', function (e) {
        if (!dragStart) return;
        state.panX = dragStart.x + (e.clientX - dragStart.mx);
        state.panY = dragStart.y + (e.clientY - dragStart.my);
        clampPan();
        AIU.applyPanZoom(wrap, state);
    });
    window.addEventListener('mouseup', function () { dragStart = null; });
};

AIE.init = function (id) {
    const canvas = document.getElementById(id + '_canvas');
    const maskCanvas = document.createElement('canvas');
    const overlayCanvas = document.getElementById(id + '_overlay');
    const annotCanvas = document.getElementById(id + '_annot');

    // brush is stored in on-screen CSS pixels (what the slider and preview
    // circle actually show) — getScale() converts it to canvas coordinate
    // space at draw time, since the canvas's internal resolution (the
    // image's native size) is usually far larger than its rendered size.
    // mode is 'zoom' (the default), 'mask' or 'annotate'; the radio group
    // in create_mask_editor switches it. They are mutually exclusive, so a
    // single finger can safely mean "pan" in zoom mode, since drawing is
    // only ever active in the other two. Annotations keep a fixed thin
    // stroke (annotStroke, CSS pixels) in the selected colour.
    const state = {
        canvas: canvas, ctx: canvas.getContext('2d'),
        maskCanvas: maskCanvas, overlayCanvas: overlayCanvas, annotCanvas: annotCanvas,
        zoomwrap: document.getElementById(id + '_zoomwrap'),
        img: null, drawing: false, brush: __DEFAULT_BRUSH_SIZE__, lastX: 0, lastY: 0, hasMask: false,
        hasAnnotation: false, annotColor: '__ANNOTATION_COLOR__', annotStroke: __ANNOTATION_STROKE__,
        mode: 'zoom', scale: 1, panX: 0, panY: 0,
    };
    // A mask stroke is painted onto two layers: the exported mask, opaque
    // white, and the on-screen overlay in the tint colour, opaque on the
    // canvas and shown at 50% by CSS (.aie-overlay), so repeated strokes
    // don't build up. An annotation stroke goes onto the single opaque
    // annotation layer, which is exported on its own (transparent
    // background) and composed onto the source image by the server.
    const maskLayers = [
        [maskCanvas.getContext('2d'), '#ffffff'],
        [overlayCanvas.getContext('2d'), '__MASK_TINT__'],
    ];
    const annotCtx = annotCanvas.getContext('2d');

    // The layers and width (canvas pixels) the current mode draws with.
    function brushSpec() {
        if (state.mode === 'annotate') {
            return { layers: [[annotCtx, state.annotColor]], width: state.annotStroke * getScale() };
        }
        return { layers: maskLayers, width: state.brush * getScale() };
    }
    AIE[id] = state;

    function getScale() {
        const rect = canvas.getBoundingClientRect();
        return rect.width ? canvas.width / rect.width : 1;
    }

    // canvas.getBoundingClientRect() already reflects the zoomwrap's
    // current CSS transform (scale/translate), so drawing coordinates stay
    // correct at any zoom/pan level without any changes here.
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const t = e.touches && e.touches.length ? e.touches[0] : e;
        return {
            x: (t.clientX - rect.left) * canvas.width / rect.width,
            y: (t.clientY - rect.top) * canvas.height / rect.height,
        };
    }

    function dot(x, y) {
        const spec = brushSpec();
        const r = spec.width / 2;
        for (const [c, color] of spec.layers) {
            c.beginPath(); c.fillStyle = color;
            c.arc(x, y, r, 0, Math.PI * 2); c.fill();
        }
    }

    function lineTo(x0, y0, x1, y1) {
        const spec = brushSpec();
        for (const [c, color] of spec.layers) {
            c.strokeStyle = color; c.lineWidth = spec.width;
            c.lineCap = 'round'; c.lineJoin = 'round';
            c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
        }
    }

    function down(e) {
        if (state.mode === 'zoom' || !state.img) return;
        if (e.touches && e.touches.length !== 1) return;
        e.preventDefault();
        state.drawing = true;
        const p = getPos(e);
        state.lastX = p.x; state.lastY = p.y;
        dot(p.x, p.y);
        if (state.mode === 'annotate') {
            state.hasAnnotation = true;
        } else if (!state.hasMask) {
            state.hasMask = true;
            emitEvent('aie_mask_state', id, true);
        }
    }
    function move(e) {
        if (state.mode === 'zoom' || !state.drawing) return;
        if (e.touches && e.touches.length !== 1) return;
        e.preventDefault();
        const p = getPos(e);
        lineTo(state.lastX, state.lastY, p.x, p.y);
        state.lastX = p.x; state.lastY = p.y;
    }
    function up() { state.drawing = false; }

    canvas.addEventListener('mousedown', down);
    canvas.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    canvas.addEventListener('touchstart', down, { passive: false });
    canvas.addEventListener('touchmove', move, { passive: false });
    canvas.addEventListener('touchend', up);

    if (state.zoomwrap) AIU.attachPanZoom(state.zoomwrap, state, function () { return state.mode === 'zoom'; });
};

// Empties both mask layers and reports the mask as gone if it wasn't already.
AIU.resetMask = function (id, state) {
    for (const c of [state.maskCanvas, state.overlayCanvas]) {
        c.getContext('2d').clearRect(0, 0, c.width, c.height);
    }
    if (state.hasMask) {
        state.hasMask = false;
        emitEvent('aie_mask_state', id, false);
    }
};

AIU.resetAnnotations = function (state) {
    state.annotCanvas.getContext('2d').clearRect(0, 0, state.annotCanvas.width, state.annotCanvas.height);
    state.hasAnnotation = false;
};

AIU.showPlaceholder = function (id, visible) {
    const ph = document.getElementById(id + '_placeholder');
    if (ph) ph.style.display = visible ? 'flex' : 'none';
};

AIE.setMode = function (id, mode) {
    // Brush/Feather sliders are NiceGUI ui.slider widgets, so their enabled
    // state is toggled from Python (see set_mode in create_mask_editor)
    // rather than by reaching into the DOM here.
    const state = AIE[id];
    if (!state) return;
    state.mode = mode;
    state.drawing = false;
};

AIE.resetZoom = function (id) {
    const state = AIE[id];
    if (state) AIU.resetPanZoom(state.zoomwrap, state);
};

AIE.loadImage = function (id, url) {
    const state = AIE[id];
    if (!state) return;
    const img = new Image();
    img.onload = function () {
        state.img = img;
        for (const c of [state.canvas, state.maskCanvas, state.overlayCanvas, state.annotCanvas]) {
            c.width = img.naturalWidth;
            c.height = img.naturalHeight;
        }
        state.ctx.drawImage(img, 0, 0);
        AIU.resetMask(id, state);
        AIU.resetAnnotations(state);
        AIE.resetZoom(id);
        AIU.showPlaceholder(id, false);
    };
    img.src = url;
};

AIE.clearMask = function (id) {
    const state = AIE[id];
    if (state && state.img) AIU.resetMask(id, state);
};

AIE.clearAnnotations = function (id) {
    const state = AIE[id];
    if (state && state.img) AIU.resetAnnotations(state);
};

AIE.clearImage = function (id) {
    const state = AIE[id];
    if (!state) return;
    state.img = null;
    state.drawing = false;
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    AIU.resetMask(id, state);
    AIU.resetAnnotations(state);
    AIE.resetZoom(id);
    AIU.showPlaceholder(id, true);
};

AIE.setBrush = function (id, size) {
    if (AIE[id]) AIE[id].brush = Number(size);
};

AIE.setAnnotationColor = function (id, color) {
    if (AIE[id]) AIE[id].annotColor = color;
};

AIE.showBrushPreview = function (id, size) {
    const el = document.getElementById(id + '_brushpreview');
    if (!el) return;
    el.style.width = size + 'px';
    el.style.height = size + 'px';
    el.style.display = 'block';
};

AIE.hideBrushPreview = function (id) {
    const el = document.getElementById(id + '_brushpreview');
    if (el) el.style.display = 'none';
};

AIE.getMaskDataUrl = function (id) {
    const state = AIE[id];
    if (!state || !state.img || !state.hasMask) return null;
    return state.maskCanvas.toDataURL('image/png');
};

// The annotation layer alone, transparent where nothing was drawn.
AIE.getAnnotationLayerDataUrl = function (id) {
    const state = AIE[id];
    if (!state || !state.img || !state.hasAnnotation) return null;
    return state.annotCanvas.toDataURL('image/png');
};

AIS.init = function (id) {
    // sliderValue is tracked here (rather than read back from the slider's
    // own DOM) since range_slider is a Quasar-rendered ui.slider, not a raw
    // <input> with a plain .value property.
    const state = {
        wrap: document.getElementById(id + '_zoomwrap'),
        compare: false, sliderValue: 50, scale: 1, panX: 0, panY: 0,
    };
    AIS[id] = state;
    if (state.wrap) AIU.attachPanZoom(state.wrap, state, function () { return true; });
};

AIS.applyClip = function (id, val) {
    const state = AIS[id];
    if (!state) return;
    state.sliderValue = val;
    const before = document.getElementById(id + '_before');
    const handle = document.getElementById(id + '_handle');
    if (before) before.style.clipPath = 'inset(0 ' + (100 - val) + '% 0 0)';
    if (handle) handle.style.left = val + '%';
};

AIS.setImages = function (id, beforeUrl, afterUrl) {
    const state = AIS[id];
    if (!state) return;
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    if (after) { after.src = afterUrl; after.style.display = 'block'; }
    if (before) { before.src = beforeUrl; before.style.display = 'block'; }
    // The slider's own value is reset from Python (see set_images in
    // create_compare_slider) — this just re-applies the clip/handle
    // position to match, since a Python-side value push alone doesn't
    // touch the clip-path.
    AIS.applyClip(id, 50);
    AIU.resetPanZoom(state.wrap, state);
    AIS.setCompare(id, state.compare);
};

AIS.setCompare = function (id, enabled) {
    const state = AIS[id];
    if (!state) return;
    state.compare = enabled;
    const range = document.getElementById(id + '_range');
    const handle = document.getElementById(id + '_handle');
    const before = document.getElementById(id + '_before');
    if (!handle || !before) return;
    // Hiding "before" with display:none rather than a clip at 100%: its box
    // is computed independently from "after"'s (object-fit: cover vs. the
    // zoomwrap's auto height), so a clip boundary at exactly 100% can leave
    // a sub-pixel sliver of it visible at the bottom/right edge.
    const display = enabled ? 'block' : 'none';
    before.style.display = display;
    handle.style.display = display;
    if (range) range.style.display = display;
    if (enabled) AIS.applyClip(id, state.sliderValue);
};

AIS.getAfterUrl = function (id) {
    const el = document.getElementById(id + '_after');
    return (el && el.getAttribute('src')) ? el.src : null;
};

AIS.reset = function (id) {
    const state = AIS[id];
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    // removeAttribute('src') alone can leave a visible broken-image icon in
    // some browsers once an <img> has previously held a real src — hiding
    // both elements outright avoids that regardless of browser quirks.
    if (after) { after.removeAttribute('src'); after.style.display = 'none'; }
    if (before) { before.removeAttribute('src'); before.style.display = 'none'; }
    if (state) AIU.resetPanZoom(state.wrap, state);
    AIS.setCompare(id, false);
};
"""
CLIENT_JS = CLIENT_JS.replace("__DEFAULT_BRUSH_SIZE__", str(DEFAULT_BRUSH_SIZE))
CLIENT_JS = CLIENT_JS.replace("__MASK_TINT__", PRIMARY_COLOR)
CLIENT_JS = CLIENT_JS.replace("__ANNOTATION_COLOR__", ANNOTATION_COLORS[0])
CLIENT_JS = CLIENT_JS.replace("__ANNOTATION_STROKE__", str(ANNOTATION_STROKE))

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


def run(model: ModelBackend) -> None:
    # Everything below is model-dependent, so it lives inside run()
    # rather than at module scope: app.py builds the model once and
    # calls this after picking this module via the FRONTEND env var
    # (see app.py and frontends/__init__.py).

    @ui.page("/")
    async def main_page() -> None:
        # add_head_html must run while the page is still being assembled — it
        # only affects the initial HTML document, so it has to happen before
        # the client connects below (once connected, the document this browser
        # already has is fixed; adding to <head> after that point has no
        # effect, which would silently drop the CLIENT_JS script and WIDGET_CSS
        # stylesheet entirely).
        ui.add_head_html(f"<style>{WIDGET_CSS}</style><script>{CLIENT_JS}</script>")
        ui.dark_mode().enable()
        # dark_page is Quasar's page/body background variable, dark is the
        # surface color for dark-mode components like ui.card — the documented
        # way to set these (nicegui.io/documentation/colors) rather than a
        # manual `!important` CSS override fighting Quasar's own theme layer.
        ui.colors(primary=PRIMARY_COLOR, dark="#111111", dark_page="#000000")

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
        ui.label(f"Model: {model.display_name}").classes("text-sm q-mb-md w-full text-center").style(f"color: {PRIMARY_COLOR}")

        with ui.column().classes("w-full gap-3 aie-page"):
            with ui.card().classes(CARD_CLASSES):
                prompt = ui.textarea(label="Prompt").props("rows=6 outlined dark").classes("w-full")

                with ui.row().classes("w-full items-center gap-4"):
                    seed_input = ui.number(label="Seed", value=65454653, format="%d").props("outlined dark").classes("flex-1")
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
                    if "Original" not in caps.supported_aspect_ratios:
                        return
                    if has_mask:
                        if aspect_ratio.value != "Original":
                            previous_aspect_ratio = aspect_ratio.value
                        aspect_ratio.value = "Original"
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

                editor_holder, get_mask_path, set_editor_image, get_feather_amount, get_annotation_layer_path = await create_mask_editor(
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
                            lora_strength = ui.slider(min=ls.min, max=ls.max, step=ls.step, value=ls.default).props("label-always")

            with ui.card().classes(CARD_CLASSES):
                with ui.expansion("Advanced settings").props("dense").classes("w-full"):
                    sr = caps.step_range
                    ui.label("Inference steps").classes("text-xs text-gray-400")
                    steps = ui.slider(min=sr.min, max=sr.max, step=sr.step, value=sr.default).props("label-always")

                    cfg = None
                    if caps.supports_cfg:
                        cr = caps.cfg_range
                        ui.label("CFG scale").classes("text-xs text-gray-400 q-mt-sm")
                        cfg = ui.slider(min=cr.min, max=cr.max, step=cr.step, value=cr.default).props("label-always")

                    negative_prompt = None
                    if caps.supports_negative_prompt:
                        negative_prompt = ui.textarea(label="Negative prompt").props("rows=2 outlined dark").classes("w-full q-mt-sm")

                    denoise = None
                    if caps.supports_denoise:
                        dr = caps.denoise_range
                        ui.label("Denoise").classes("text-xs text-gray-400 q-mt-sm")
                        denoise = ui.slider(min=dr.min, max=dr.max, step=dr.step, value=dr.default).props("label-always")

                    sampler_name = None
                    if caps.sampler_choices:
                        sampler_default = caps.default_sampler if caps.default_sampler in caps.sampler_choices else caps.sampler_choices[0]
                        sampler_name = ui.select(caps.sampler_choices, value=sampler_default, label="Sampler name").props("outlined dark").classes("w-full q-mt-sm")

                    scheduler = None
                    if caps.scheduler_choices:
                        scheduler_default = caps.default_scheduler if caps.default_scheduler in caps.scheduler_choices else caps.scheduler_choices[0]
                        scheduler = ui.select(caps.scheduler_choices, value=scheduler_default, label="Scheduler").props("outlined dark").classes("w-full")

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
                mask_path = await get_mask_path() if caps.supports_inpainting else None
                annotated_image_path = None
                if caps.num_annotation_colors > 0:
                    layer_path = await get_annotation_layer_path()
                    if layer_path:
                        annotated_image_path = await nicegui_run.io_bound(
                            imaging.compose_annotations, source_image_path, layer_path
                        )
                lora_files = available_loras.get(lora_name.value, []) if lora_name is not None else []

                params = GenerationParams(
                    prompt=prompt.value or "",
                    source_image_path=source_image_path,
                    mask_path=mask_path,
                    annotated_image_path=annotated_image_path,
                    reference_images=[h["path"] for h in reference_holders if h["path"]],
                    seed=seed_input.value,
                    randomize_seed=randomize_seed.value,
                    aspect_ratio=aspect_ratio.value,
                    target_megapixels=megapixels.value,
                    steps=steps.value,
                    # cfg/lora_strength fall back to the capability's own
                    # declared default (not a bare 0.0) when the control
                    # wasn't created — every model's own cfg_range/
                    # lora_strength_range starts above 0, and neither model
                    # backend currently guards its use of params.cfg/
                    # params.lora_strength behind supports_cfg/supports_loras,
                    # so a bare 0.0 here would silently fall outside a
                    # future such model's valid range.
                    cfg=cfg.value if cfg is not None else caps.cfg_range.default,
                    denoise=denoise.value if denoise is not None else 1.0,
                    sampler_name=sampler_name.value if sampler_name is not None else None,
                    scheduler=scheduler.value if scheduler is not None else None,
                    negative_prompt=negative_prompt.value if negative_prompt is not None else "",
                    lora_files=lora_files,
                    lora_strength=lora_strength.value if lora_strength is not None else caps.lora_strength_range.default,
                    apply_color_correction_enabled=apply_color_correction_switch.value,
                    feather_amount=get_feather_amount() if caps.supports_inpainting else 0,
                )
                result = await nicegui_run.io_bound(model.generate, params)

                seed_input.value = result.actual_seed
                await set_result_images(result.before_path, result.after_path)
                compare_switch.enable()
                use_as_input_btn.enable()
                download_btn.enable()

            except GenerationError as e:
                ui.notify(str(e), type="negative")
            except Exception as e:  # noqa: BLE001 — surface unexpected errors instead of hanging silently
                ui.notify(f"Unexpected error: {e}", type="negative")
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

    # Make sure whatever the active model started (a ComfyUI subprocess, a
    # loaded pipeline, ...) gets torn down on shutdown instead of leaking.
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
        host="0.0.0.0",
        port=7860,
        title="AI Image Edit",
        dark=True,
        reload=False,
        reconnect_timeout=300,
        storage_secret=storage_secret,
    )
