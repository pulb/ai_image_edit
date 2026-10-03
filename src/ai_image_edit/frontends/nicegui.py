# SPDX-License-Identifier: GPL-3.0-or-later
"""
NiceGUI frontend. Entry point is run(model, model_backend), called by
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

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR, to_url, from_url
from ai_image_edit.core.types import GenerationParams
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

# This app's one accent color — the "Model: ..." subtitle's text color and
# ui.colors()'s primary (buttons, the active-tab/slider color, etc.) both
# read from here so they can't drift apart.
PRIMARY_COLOR = "#7c3aed"


# --- Small file helpers shared by the UI layer ---
# (model-independent — these don't need `model` or `model_backend`, so
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
    pathlib so it can't smuggle in a path.
    """
    suffix = Path(file.name).suffix or ".png"
    stem = Path(file.name).stem or "upload"
    path = WORK_DIR / f"{stem}_{uuid.uuid4().hex}{suffix}"
    await file.save(path)
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
    Image 2 / Image 3 slots. Clicking anywhere in the box calls pickFiles()
    on a hidden ui.upload (QUploader's own file-picker method, invoked via
    NiceGUI's documented run_method — the same mechanism NiceGUI's own
    Upload.reset() uses internally for QUploader's reset()); once a file is
    chosen, its on_upload event hands it to Python, which saves it and
    swaps in the preview.

    Styling (.qie-upload-box etc.) comes from WIDGET_CSS, a small stylesheet
    this app defines itself and injects via ui.add_head_html — see the
    comment above WIDGET_CSS for why that's preferred here over NiceGUI's
    bundled Tailwind classes. The box itself is a real nicegui `ui.element`
    rather than raw HTML so a NiceGUI click handler (box.on("click", ...))
    can be attached to it directly.

    Returns holder — holder['path'] always reflects the currently selected
    file's local path (or None).
    """
    holder: dict = {"path": None}

    with ui.column().classes("w-full gap-1"):
        ui.label(label).classes("text-sm text-gray-400")
        with ui.element("div").classes("qie-upload-box") as box:
            placeholder = ui.label("Click to upload an image").classes("qie-placeholder")
            preview = ui.image().classes("qie-preview")
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
) -> Tuple[
    dict,
    Callable[[], Awaitable[Optional[str]]],
    Callable[[str], Awaitable[None]],
    Callable[[], int],
]:
    """
    The "Input Image" widget: click-to-upload image + a <canvas> for
    drawing an optional inpainting mask, with a resizable brush. The lock
    button toggles between pan/zoom (locked, default) and mask drawing
    (unlocked); drawing a stroke while unlocked makes this an Inpaint Edit,
    "Remove mask" reverts to a plain Image Edit. The Feather slider (0-16,
    default 6) is the Gaussian blur radius composite_with_soft_transition()
    applies to the mask edges; 0 falls back to a hard cutout.

    If given, on_mask_change(has_mask) fires whenever the mask goes from
    empty to non-empty or back (via the 'qie_mask_state' custom event).

    Returns (holder, get_mask_path, set_image, get_feather_amount):
      * holder['path'] is the current source image's local path (or None).
      * get_mask_path() is async, reads the mask canvas and saves it to
        disk, or returns None if nothing was drawn. Called once, on Generate.
      * set_image(path) loads a new image and clears the mask.
      * get_feather_amount() reads the Feather slider's current value.
    """
    editor_id = f"edit_{uuid.uuid4().hex}"
    holder: dict = {"path": None}

    if on_mask_change:
        def handle_mask_event(e) -> None:
            args = list(e.args) if isinstance(e.args, (list, tuple)) else [e.args]
            if len(args) < 2 or args[0] != editor_id:
                return
            on_mask_change(bool(args[1]))

        ui.on("qie_mask_state", handle_mask_event)

    with ui.element("div").classes("qie-editor-box") as editor_box:
        ui.html(
            f'<div id="{editor_id}_zoomwrap" class="qie-zoomwrap">'
            f'<canvas id="{editor_id}_canvas" class="qie-canvas"></canvas>'
            f'<canvas id="{editor_id}_overlay" class="qie-overlay"></canvas>'
            f'<div id="{editor_id}_placeholder" class="qie-placeholder">Click to upload an image</div>'
            f'</div>'
            f'<div id="{editor_id}_brushpreview" class="qie-brush-preview"></div>',
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
        await ui.run_javascript(f"QIE.loadImage('{editor_id}', '{to_url(path)}')", timeout=10.0)
        uploader.reset()

    uploader.on_upload(handle_upload)

    with ui.row().classes("w-full items-center gap-3"):
        async def remove_mask() -> None:
            await ui.run_javascript(f"QIE.clearMask('{editor_id}')", timeout=5.0)

        ui.button("Remove mask", on_click=remove_mask).props("flat dense size=sm").classes("text-xs")

        async def clear_image() -> None:
            holder["path"] = None
            await ui.run_javascript(f"QIE.clearImage('{editor_id}')", timeout=5.0)

        ui.button("Clear", on_click=clear_image).props("flat dense size=sm").classes("text-xs")

    with ui.row().classes("w-full items-center gap-3 p-2 bg-neutral-900 rounded-lg"):
        with ui.column().classes("flex-1 gap-3"):
            ui.label("Brush size").classes("text-xs text-gray-400")
            brush_slider = ui.slider(min=5, max=80, step=1, value=DEFAULT_BRUSH_SIZE).props("label-always dense")
            brush_slider.disable()
            # js_handler runs entirely in the browser, no server round-trip
            # (see CLIENT_JS's QIE comment for why that matters for a
            # dragged brush size) — 'update:model-value' fires continuously
            # while dragging; 'change' fires once on release.
            brush_slider.on(
                "update:model-value",
                js_handler=f"(value) => {{ QIE.setBrush('{editor_id}', value); QIE.showBrushPreview('{editor_id}', value); }}",
            )
            brush_slider.on("change", js_handler=f"() => QIE.hideBrushPreview('{editor_id}')")

        with ui.column().classes("flex-1 gap-3"):
            ui.label("Feather").classes("text-xs text-gray-400")
            feather_slider = ui.slider(min=0, max=16, step=1, value=DEFAULT_FEATHER_AMOUNT).props("label-always dense")
            feather_slider.disable()

        lock_state = {"locked": True}

        async def toggle_lock() -> None:
            lock_state["locked"] = not lock_state["locked"]
            lock_btn.props(f"icon={'lock' if lock_state['locked'] else 'lock_open'}")
            if lock_state["locked"]:
                brush_slider.disable()
                feather_slider.disable()
            else:
                brush_slider.enable()
                feather_slider.enable()
            await ui.run_javascript(
                f"QIE.setLocked('{editor_id}', {str(lock_state['locked']).lower()})", timeout=5.0
            )

        lock_btn = ui.button(icon="lock", on_click=toggle_lock).props("flat dense size=md")

    try:
        await ui.run_javascript(f"QIE.init('{editor_id}')", timeout=10.0)
    except TimeoutError:
        print(f"[create_mask_editor] QIE.init('{editor_id}') timed out — the editor may not respond until the page is reloaded.", flush=True)

    def get_feather_amount() -> int:
        return int(feather_slider.value)

    async def get_mask_path() -> Optional[str]:
        if not holder["path"]:
            return None
        data_url = await ui.run_javascript(f"QIE.getMaskDataUrl('{editor_id}')", timeout=15.0)
        if not data_url:
            return None
        return save_data_url(data_url, "mask.png")

    async def set_image(path: str) -> None:
        holder["path"] = path
        await ui.run_javascript(
            f"QIE.loadImage('{editor_id}', '{to_url(path)}'); QIE.clearMask('{editor_id}');",
            timeout=10.0,
        )

    return holder, get_mask_path, set_image, get_feather_amount

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

    with ui.element("div").classes("qis-container"):
        ui.html(
            f'<div id="{slider_id}_zoomwrap" class="qis-zoomwrap">'
            f'<img id="{slider_id}_after" class="qis-after">'
            f'<img id="{slider_id}_before" class="qis-before">'
            f'<div id="{slider_id}_handle" class="qis-handle"></div>'
            f'</div>',
            sanitize=False,
        )
        # A ui.slider rather than a raw <input type="range"> — a bare <input>
        # (even type="range") is enough to pop the Android software keyboard
        # on some WebViews/browsers just from being focused. No label (label-always) here: unlike Brush/Feather, this
        # overlays directly on the image, where a value bubble would be
        # visual noise rather than useful information. js_handler keeps
        # dragging fully client-side (see CLIENT_JS's QIE comment for why
        # that matters) — 'update:model-value' fires continuously while
        # dragging, calling QIS.applyClip directly with no server
        # round-trip.
        range_slider = ui.slider(min=0, max=100, step=1, value=50).props("dense").classes("qis-range")
        range_slider.props(f"id={slider_id}_range")
        range_slider.on(
            "update:model-value",
            js_handler=f"(value) => QIS.applyClip('{slider_id}', value)",
        )

    try:
        await ui.run_javascript(f"QIS.init('{slider_id}')", timeout=10.0)
    except TimeoutError:
        print(f"[create_compare_slider] QIS.init('{slider_id}') timed out — the compare slider may not respond until the page is reloaded.", flush=True)

    async def set_images(before_path: str, after_path: str) -> None:
        # The slider position itself is set from Python (NiceGUI's own
        # reliable value-setting API) rather than from JS — programmatic
        # value pushes don't fire the same DOM events a real drag does, so
        # there's no update:model-value for a js_handler to catch here.
        # QIS.setImages (below) still re-applies the clip/handle position
        # to match via QIS.applyClip(id, 50), since setting the Python
        # value alone doesn't touch the clip-path.
        range_slider.set_value(50)
        await ui.run_javascript(
            f"QIS.setImages('{slider_id}', '{to_url(before_path)}', '{to_url(after_path)}')",
            timeout=10.0,
        )

    async def set_compare(enabled: bool) -> None:
        await ui.run_javascript(f"QIS.setCompare('{slider_id}', {str(bool(enabled)).lower()})", timeout=5.0)

    async def reset() -> None:
        await ui.run_javascript(f"QIS.reset('{slider_id}')", timeout=5.0)

    async def get_after_path() -> Optional[str]:
        url = await ui.run_javascript(f"QIS.getAfterUrl('{slider_id}')", timeout=5.0)
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
# runtime (QIS's clip-path/left/transform, QIE's brush-preview size) are
# still set here as sensible defaults; a plain inline style set directly via
# JS (element.style.foo = ...) always takes precedence over a class
# regardless, so the two coexist without conflict. These rules also don't
# need !important to beat Quasar's own defaults: NiceGUI 3.0 moved Quasar's
# base styles into a CSS @layer, and per the cascade spec, any rule outside
# a layer — including this plain <style> block — already outranks layered
# rules regardless of specificity.
WIDGET_CSS = """
.qie-upload-box {
    position: relative;
    width: 100%;
    min-height: 160px;
    cursor: pointer;
    border: 1px solid #444;
    border-radius: 8px;
    background: #000;
    overflow: hidden;
}

.qie-preview {
    display: block;
    width: 100%;
    height: auto;
}

.qie-placeholder {
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

.qie-editor-box {
    position: relative;
    width: 100%;
    min-height: 120px;
    cursor: pointer;
    overflow: hidden;
}

.qie-zoomwrap {
    position: relative;
    width: 100%;
    touch-action: none;
    transform-origin: 50% 50%;
}

.qie-canvas {
    display: block;
    width: 100%;
    height: auto;
    touch-action: none;
    background: #000;
    border: 1px solid #333;
    border-radius: 8px;
}

.qie-overlay {
    position: absolute;
    inset: 0;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    border: 1px solid transparent;
    opacity: 0.5;
    pointer-events: none;
}

.qie-brush-preview {
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

.qis-container {
    position: relative;
    width: 100%;
    min-height: 200px;
    background: #000;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
}

.qis-zoomwrap {
    position: relative;
    width: 100%;
    touch-action: none;
    transform-origin: 50% 50%;
}

.qis-after {
    display: none;
    width: 100%;
    height: auto;
    min-height: 200px;
}

.qis-before {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    clip-path: inset(0 100% 0 0);
    display: none;
}

.qis-handle {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 100%;
    width: 2px;
    background: #fff;
    pointer-events: none;
    display: none;
}

.qis-range {
    position: absolute;
    left: 8px;
    right: 8px;
    bottom: 8px;
    width: calc(100% - 16px);
    display: none;
}

.qie-page {
    max-width: 720px;
    margin: 0 auto;
}
"""

# --- Client-side JS ---
# Two small namespaces, defined once as generic functions keyed by an
# element-id prefix so each widget instance just calls e.g. QIE.init('some_id')
# after creating its own DOM elements with matching ids. Everything that
# NiceGUI already has a documented element/API for — file pickers (ui.upload
# + element.run_method('pickFiles'), the same mechanism NiceGUI's own
# Upload.reset() uses internally), image previews (ui.image.set_source) —
# uses that instead of custom JS. What's left needs custom JS because
# there's no NiceGUI equivalent:
#
#   QIE — the mask/image editor: a <canvas> drawn on with mouse/touch
#         handlers. NiceGUI's ui.interactive_image can report mouse
#         coordinates, but it round-trips every single mouse-move event to
#         the server to do so, which would make drawing feel laggy. A raw
#         <canvas> keeps every brush stroke fully client-side; the Brush
#         size slider is a regular ui.slider, but wired up with js_handler
#         (see create_mask_editor) so it still calls QIE.setBrush directly
#         in the browser, with no server round-trip. When a mask goes from
#         empty to non-empty or back, QIE pushes that as a 'qie_mask_state'
#         event via emitEvent(...) — the documented pattern for a
#         browser-side state change to reach Python without being tied to
#         one specific DOM event (see the "Custom events" section of the
#         Generic Events docs) — which main_page listens for with
#         ui.on(...) to keep Aspect ratio in sync (see create_mask_editor's
#         on_mask_change).
#
#   QIS — the before/after comparison: a CSS clip-path dragged by a native
#         range input, entirely client-side.
CLIENT_JS = r"""
window.QIE = window.QIE || {};

QIE.init = function (id) {
    const canvas = document.getElementById(id + '_canvas');
    const ctx = canvas.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d');
    // On-screen overlay: the same strokes as the mask, in the tint colour and
    // opaque on the canvas, shown at 50% by CSS (.qie-overlay), so repeated
    // strokes don't build up. The exported mask stays opaque white.
    const overlayCanvas = document.getElementById(id + '_overlay');
    const overlayCtx = overlayCanvas.getContext('2d');
    const zoomwrap = document.getElementById(id + '_zoomwrap');

    // brush is stored in on-screen CSS pixels (what the slider and preview
    // circle actually show) — getScale() converts it to canvas coordinate
    // space at draw time, since the canvas's internal resolution (the
    // image's native size) is usually far larger than its rendered size.
    // locked starts true: pan/zoom (mirroring QIS) is the default mode;
    // the lock button switches to mask drawing. The two are mutually
    // exclusive, so there's no ambiguity between a stroke and a pan to
    // resolve — unlike QIS, a single finger can safely mean "pan" here
    // too, since drawing is only ever active while unlocked.
    const state = {
        img: null, drawing: false, brush: __DEFAULT_BRUSH_SIZE__, lastX: 0, lastY: 0, hasMask: false,
        locked: true, zoomScale: 1, panX: 0, panY: 0,
    };
    state.canvas = canvas; state.ctx = ctx;
    state.maskCanvas = maskCanvas; state.maskCtx = maskCtx;
    state.overlayCanvas = overlayCanvas; state.overlayCtx = overlayCtx;
    QIE[id] = state;

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
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return { x: (t.clientX - rect.left) * scaleX, y: (t.clientY - rect.top) * scaleY };
    }

    function dot(x, y) {
        const r = (state.brush * getScale()) / 2;
        maskCtx.beginPath(); maskCtx.fillStyle = '#ffffff';
        maskCtx.arc(x, y, r, 0, Math.PI * 2); maskCtx.fill();
        overlayCtx.beginPath(); overlayCtx.fillStyle = '__MASK_TINT__';
        overlayCtx.arc(x, y, r, 0, Math.PI * 2); overlayCtx.fill();
    }

    function lineTo(x0, y0, x1, y1) {
        const w = state.brush * getScale();
        maskCtx.strokeStyle = '#ffffff'; maskCtx.lineWidth = w;
        maskCtx.lineCap = 'round'; maskCtx.lineJoin = 'round';
        maskCtx.beginPath(); maskCtx.moveTo(x0, y0); maskCtx.lineTo(x1, y1); maskCtx.stroke();
        overlayCtx.strokeStyle = '__MASK_TINT__'; overlayCtx.lineWidth = w;
        overlayCtx.lineCap = 'round'; overlayCtx.lineJoin = 'round';
        overlayCtx.beginPath(); overlayCtx.moveTo(x0, y0); overlayCtx.lineTo(x1, y1); overlayCtx.stroke();
    }

    function markMask() {
        if (!state.hasMask) {
            state.hasMask = true;
            emitEvent('qie_mask_state', id, true);
        }
    }

    function down(e) {
        if (state.locked) return;
        if (!state.img) return;
        if (e.touches && e.touches.length !== 1) return;
        e.preventDefault();
        state.drawing = true;
        const p = getPos(e);
        state.lastX = p.x; state.lastY = p.y;
        dot(p.x, p.y);
        markMask();
    }
    function move(e) {
        if (state.locked) return;
        if (!state.drawing) return;
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

    // Pan/zoom — mirrors QIS's implementation exactly (single-finger pan
    // once zoomed in, two-finger pinch to zoom), active only while locked.
    if (zoomwrap) {
        function distance(touches) {
            const dx = touches[0].clientX - touches[1].clientX;
            const dy = touches[0].clientY - touches[1].clientY;
            return Math.sqrt(dx * dx + dy * dy);
        }
        function clampPan() {
            if (state.zoomScale <= 1) {
                state.panX = 0;
                state.panY = 0;
                return;
            }
            // offsetWidth/offsetHeight are the element's own layout size,
            // which a transform on that same element never affects — using
            // getBoundingClientRect() here instead would read back
            // (possibly stale) post-transform geometry, since this runs
            // before the new scale has actually been applied to the DOM.
            const maxX = zoomwrap.offsetWidth * (state.zoomScale - 1) / 2;
            const maxY = zoomwrap.offsetHeight * (state.zoomScale - 1) / 2;
            state.panX = Math.min(maxX, Math.max(-maxX, state.panX));
            state.panY = Math.min(maxY, Math.max(-maxY, state.panY));
        }
        function applyZoomTransform() {
            zoomwrap.style.transform = 'translate(' + state.panX + 'px, ' + state.panY + 'px) scale(' + state.zoomScale + ')';
        }

        let pinchStartDist = null;
        let pinchStartScale = 1;
        let panning = false;
        let panStartX = 0;
        let panStartY = 0;
        let panStartTouchX = 0;
        let panStartTouchY = 0;

        zoomwrap.addEventListener('touchstart', function (e) {
            if (!state.locked) return;
            if (e.touches.length === 2) {
                pinchStartDist = distance(e.touches);
                pinchStartScale = state.zoomScale;
                panning = false;
            } else if (e.touches.length === 1 && state.zoomScale > 1) {
                panning = true;
                panStartX = state.panX;
                panStartY = state.panY;
                panStartTouchX = e.touches[0].clientX;
                panStartTouchY = e.touches[0].clientY;
            }
        }, { passive: true });

        zoomwrap.addEventListener('touchmove', function (e) {
            if (!state.locked) return;
            if (e.touches.length === 2 && pinchStartDist) {
                e.preventDefault();
                const factor = distance(e.touches) / pinchStartDist;
                state.zoomScale = Math.min(Math.max(pinchStartScale * factor, 1), 8);
                clampPan();
                applyZoomTransform();
            } else if (e.touches.length === 1 && panning) {
                e.preventDefault();
                state.panX = panStartX + (e.touches[0].clientX - panStartTouchX);
                state.panY = panStartY + (e.touches[0].clientY - panStartTouchY);
                clampPan();
                applyZoomTransform();
            }
        }, { passive: false });

        zoomwrap.addEventListener('touchend', function (e) {
            if (e.touches.length < 2) {
                pinchStartDist = null;
            }
            if (e.touches.length < 1) {
                panning = false;
            }
        });
    }
};

QIE.setLocked = function (id, locked) {
    // Brush/Feather sliders are real NiceGUI ui.slider widgets now, so
    // their enabled state is toggled from Python (see toggle_lock in
    // create_mask_editor) instead of by reaching into the DOM here.
    const state = QIE[id];
    if (state) state.locked = locked;
};

QIE.resetZoom = function (id) {
    const state = QIE[id];
    if (!state) return;
    state.zoomScale = 1;
    state.panX = 0;
    state.panY = 0;
    const zoomwrap = document.getElementById(id + '_zoomwrap');
    if (zoomwrap) zoomwrap.style.transform = 'translate(0px, 0px) scale(1)';
};

QIE.loadImage = function (id, url) {
    const state = QIE[id];
    if (!state) return;
    const img = new Image();
    img.onload = function () {
        state.img = img;
        state.canvas.width = img.naturalWidth;
        state.canvas.height = img.naturalHeight;
        state.maskCanvas.width = img.naturalWidth;
        state.maskCanvas.height = img.naturalHeight;
        state.overlayCanvas.width = img.naturalWidth;
        state.overlayCanvas.height = img.naturalHeight;
        state.ctx.drawImage(img, 0, 0);
        state.maskCtx.clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
        state.overlayCtx.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
        QIE.resetZoom(id);
        if (state.hasMask) {
            state.hasMask = false;
            emitEvent('qie_mask_state', id, false);
        }
        const ph = document.getElementById(id + '_placeholder');
        if (ph) ph.style.display = 'none';
    };
    img.src = url;
};

QIE.clearMask = function (id) {
    const state = QIE[id];
    if (!state || !state.img) return;
    state.maskCtx.clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
    state.overlayCtx.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
    if (state.hasMask) {
        state.hasMask = false;
        emitEvent('qie_mask_state', id, false);
    }
};

QIE.clearImage = function (id) {
    const state = QIE[id];
    if (!state) return;
    state.img = null;
    state.drawing = false;
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.maskCtx.clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
    state.overlayCtx.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
    QIE.resetZoom(id);
    if (state.hasMask) {
        state.hasMask = false;
        emitEvent('qie_mask_state', id, false);
    }
    const ph = document.getElementById(id + '_placeholder');
    if (ph) ph.style.display = 'flex';
};

QIE.setBrush = function (id, size) {
    if (QIE[id]) QIE[id].brush = Number(size);
};

QIE.showBrushPreview = function (id, size) {
    const el = document.getElementById(id + '_brushpreview');
    if (!el) return;
    el.style.width = size + 'px';
    el.style.height = size + 'px';
    el.style.display = 'block';
};

QIE.hideBrushPreview = function (id) {
    const el = document.getElementById(id + '_brushpreview');
    if (el) el.style.display = 'none';
};

QIE.getMaskDataUrl = function (id) {
    const state = QIE[id];
    if (!state || !state.img || !state.hasMask) return null;
    return state.maskCanvas.toDataURL('image/png');
};

window.QIS = window.QIS || {};

QIS.init = function (id) {
    QIS[id] = { compare: false, scale: 1, panX: 0, panY: 0 };

    const wrap = document.getElementById(id + '_zoomwrap');
    if (!wrap) return;

    function distance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function clampPan() {
        const s = QIS[id];
        if (s.scale <= 1) {
            s.panX = 0;
            s.panY = 0;
            return;
        }
        // offsetWidth/offsetHeight are the element's own layout size, which
        // a transform on that same element never affects — using
        // getBoundingClientRect() here instead would read back (possibly
        // stale) post-transform geometry, since this runs before the new
        // scale has actually been applied to the DOM.
        const maxX = wrap.offsetWidth * (s.scale - 1) / 2;
        const maxY = wrap.offsetHeight * (s.scale - 1) / 2;
        s.panX = Math.min(maxX, Math.max(-maxX, s.panX));
        s.panY = Math.min(maxY, Math.max(-maxY, s.panY));
    }

    function applyTransform() {
        const s = QIS[id];
        wrap.style.transform = 'translate(' + s.panX + 'px, ' + s.panY + 'px) scale(' + s.scale + ')';
    }

    let pinchStartDist = null;
    let pinchStartScale = 1;
    let panning = false;
    let panStartX = 0;
    let panStartY = 0;
    let panStartTouchX = 0;
    let panStartTouchY = 0;

    wrap.addEventListener('touchstart', function (e) {
        if (e.touches.length === 2) {
            pinchStartDist = distance(e.touches);
            pinchStartScale = QIS[id].scale;
            panning = false;
        } else if (e.touches.length === 1 && QIS[id].scale > 1) {
            panning = true;
            panStartX = QIS[id].panX;
            panStartY = QIS[id].panY;
            panStartTouchX = e.touches[0].clientX;
            panStartTouchY = e.touches[0].clientY;
        }
    }, { passive: true });

    wrap.addEventListener('touchmove', function (e) {
        if (e.touches.length === 2 && pinchStartDist) {
            e.preventDefault();
            const factor = distance(e.touches) / pinchStartDist;
            QIS[id].scale = Math.min(Math.max(pinchStartScale * factor, 1), 8);
            clampPan();
            applyTransform();
        } else if (e.touches.length === 1 && panning) {
            e.preventDefault();
            QIS[id].panX = panStartX + (e.touches[0].clientX - panStartTouchX);
            QIS[id].panY = panStartY + (e.touches[0].clientY - panStartTouchY);
            clampPan();
            applyTransform();
        }
    }, { passive: false });

    wrap.addEventListener('touchend', function (e) {
        if (e.touches.length < 2) {
            pinchStartDist = null;
        }
        if (e.touches.length < 1) {
            panning = false;
        }
    });
};

QIS.applyClip = function (id, val) {
    const before = document.getElementById(id + '_before');
    const handle = document.getElementById(id + '_handle');
    // Tracked here (rather than read back from the slider's own DOM) since
    // range_slider is a Quasar-rendered ui.slider now, not a raw <input>
    // with a plain .value property — QIS.setCompare reads this instead of
    // reaching into the slider element directly.
    if (!QIS[id]) QIS[id] = {};
    QIS[id].sliderValue = val;
    if (!before) return;
    before.style.clipPath = 'inset(0 ' + (100 - val) + '% 0 0)';
    if (handle) handle.style.left = val + '%';
};

QIS.setImages = function (id, beforeUrl, afterUrl) {
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    const wrap = document.getElementById(id + '_zoomwrap');
    if (after) { after.src = afterUrl; after.style.display = 'block'; }
    if (before) { before.src = beforeUrl; before.style.display = 'block'; }
    // The slider's own value is reset from Python (see set_images in
    // create_compare_slider) — this just re-applies the clip/handle
    // position to match, since a Python-side value push alone doesn't
    // touch the clip-path.
    QIS.applyClip(id, 50);
    if (wrap) wrap.style.transform = 'translate(0px, 0px) scale(1)';
    if (!QIS[id]) QIS[id] = { compare: false };
    QIS[id].scale = 1;
    QIS[id].panX = 0;
    QIS[id].panY = 0;
    QIS.setCompare(id, QIS[id].compare);
};

QIS.setCompare = function (id, enabled) {
    if (!QIS[id]) QIS[id] = {};
    QIS[id].compare = enabled;
    const range = document.getElementById(id + '_range');
    const handle = document.getElementById(id + '_handle');
    const before = document.getElementById(id + '_before');
    if (!handle || !before) return;
    if (enabled) {
        before.style.display = 'block';
        if (range) range.style.display = 'block';
        handle.style.display = 'block';
        QIS.applyClip(id, QIS[id].sliderValue !== undefined ? QIS[id].sliderValue : 50);
    } else {
        // clip-path alone can leave a sub-pixel sliver of "before" visible
        // at the bottom/right edge — its box is computed independently
        // from "after"'s (different aspect ratio, one from object-fit:
        // cover, one from the zoomwrap's auto height), so a clip boundary
        // at exactly 100% can miss by a fraction of a CSS pixel depending
        // on the image's exact dimensions. display:none removes it from
        // rendering entirely, immune to that rounding.
        before.style.display = 'none';
        if (range) range.style.display = 'none';
        handle.style.display = 'none';
        before.style.clipPath = 'inset(0 100% 0 0)';
    }
};

QIS.getAfterUrl = function (id) {
    const el = document.getElementById(id + '_after');
    return (el && el.getAttribute('src')) ? el.src : null;
};

QIS.reset = function (id) {
    const after = document.getElementById(id + '_after');
    const before = document.getElementById(id + '_before');
    const wrap = document.getElementById(id + '_zoomwrap');
    // removeAttribute('src') alone can leave a visible broken-image icon in
    // some browsers once an <img> has previously held a real src — hiding
    // both elements outright avoids that regardless of browser quirks.
    if (after) { after.removeAttribute('src'); after.style.display = 'none'; }
    if (before) { before.removeAttribute('src'); before.style.display = 'none'; }
    if (wrap) wrap.style.transform = 'translate(0px, 0px) scale(1)';
    if (QIS[id]) {
        QIS[id].scale = 1;
        QIS[id].panX = 0;
        QIS[id].panY = 0;
    }
    QIS.setCompare(id, false);
};
"""
CLIENT_JS = CLIENT_JS.replace("__DEFAULT_BRUSH_SIZE__", str(DEFAULT_BRUSH_SIZE))
CLIENT_JS = CLIENT_JS.replace("__MASK_TINT__", PRIMARY_COLOR)

# --- Page ---

# Every top-level section on the page (Prompt/Seed, Aspect ratio + editor,
# the settings accordions, the Result panel) is wrapped in a ui.card using
# this same class string, so their content all sits flush-aligned with each
# other and none of them indents by a different amount than its neighbors.
# q-pa-none strips QCard's default padding (Quasar's own utility class, not
# a NiceGUI quirk) — genuinely needed, not just a style choice: the editor
# canvas (.qie-editor-box) and the result comparison image (.qis-container)
# are both width: 100% against their containing card, so the card's default
# padding would inset them and shrink the actual touch-drawing/viewing area
# on a small screen. Applied to all cards rather than just those two so
# nothing looks oddly inset next to the ones that need it.
CARD_CLASSES = "w-full q-pa-none"


def run(model: ModelBackend, model_backend: str) -> None:
    # Everything below is model-dependent, so it lives inside run()
    # rather than at module scope: app.py builds the model once and
    # calls this after picking this module via the FRONTEND env var
    # (see app.py and frontends/__init__.py). model_backend is just
    # the MODEL_BACKEND name, used below for the UI's own subtitle.

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
        ui.label(f"Model: {model_backend}").classes("text-sm q-mb-md w-full text-center").style(f"color: {PRIMARY_COLOR}")

        with ui.column().classes("w-full gap-3 qie-page"):
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
            # mask exists — Aspect ratio additionally force-set to "Original" —
            # only when the model actually offers "Original" as a choice, and
            # restored the moment the mask is removed.
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

                def handle_mask_change(has_mask: bool) -> None:
                    nonlocal previous_aspect_ratio
                    if "Original" not in caps.supported_aspect_ratios:
                        return
                    if has_mask:
                        if aspect_ratio.value != "Original":
                            previous_aspect_ratio = aspect_ratio.value
                        aspect_ratio.value = "Original"
                        aspect_ratio.disable()
                        megapixels.disable()
                    else:
                        aspect_ratio.enable()
                        aspect_ratio.value = previous_aspect_ratio
                        if len(caps.supported_megapixels) > 1:
                            # Otherwise it was already permanently disabled above
                            # (nothing to choose), independent of any mask.
                            megapixels.enable()

                editor_holder, get_mask_path, set_editor_image, get_feather_amount = await create_mask_editor(
                    on_mask_change=handle_mask_change if caps.supports_inpainting else None
                )

            reference_holders: List[dict] = []
            if caps.max_reference_images > 1 or caps.supports_loras:
                with ui.card().classes(CARD_CLASSES):
                    if caps.max_reference_images > 1:
                        with ui.expansion("Reference images (optional)").props("dense").classes("w-full"):
                            for i in range(caps.max_reference_images - 1):
                                reference_holders.append(create_simple_image_upload(f"Input image {i + 2}"))

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
                lora_files = available_loras.get(lora_name.value, []) if lora_name is not None else []

                params = GenerationParams(
                    prompt=prompt.value or "",
                    source_image_path=source_image_path,
                    mask_path=mask_path,
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

    # Called unconditionally (no "if __name__ == '__main__':" guard) — this
    # module is only ever reached via app.py's FRONTEND-driven
    # dispatch (see frontends/__init__.py), which imports it as
    # frontends.nicegui rather than running it as a script, so __name__ here is
    # never "__main__" to begin with. Same reasoning as gradio_ui.py's own
    # unconditional demo.queue().launch().
    #
    # reload=False is important here: model.start() already launched whatever
    # the active model needs (e.g. ComfyUI) once, above — NiceGUI's
    # auto-reloader re-executes this module in a subprocess, which would
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
