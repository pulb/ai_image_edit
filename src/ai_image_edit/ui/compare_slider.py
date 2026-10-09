# SPDX-License-Identifier: GPL-3.0-or-later
"""The before/after comparison slider."""
import uuid
from typing import Awaitable, Callable, Optional, Tuple

from nicegui import app, ui

from ai_image_edit.core.paths import from_url, to_url
from ai_image_edit.ui.components import call_js, init_widget


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
        # dragging fully client-side (see the module docstring of
        # assets.py for why that matters) — 'update:model-value' fires continuously while
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
