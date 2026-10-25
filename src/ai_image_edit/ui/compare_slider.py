# SPDX-License-Identifier: GPL-3.0-or-later
"""The before/after comparison slider."""
import uuid
from typing import Optional

from nicegui import ui

from ai_image_edit.core.paths import to_url
from ai_image_edit.ui.components import call_js, init_widget


class CompareSlider:
    """
    The result panel: a before/after image pair with a CSS clip-path
    divider dragged via a slider, kept fully client-side via js_handler.
    When comparison is off, only the after image shows. Two-finger pinch
    zooms both images together (clamped 1x-8x); once zoomed, a single
    finger pans. Both reset whenever a new result loads.

    after_path is the local path of the result on show (None while there is
    none). frame is the image frame, for overlays. Create it with create().
    """

    def __init__(self) -> None:
        self.after_path: Optional[str] = None
        self._id = slider_id = f"cmp_{uuid.uuid4().hex}"
        with ui.element("div").classes("ais-container") as self.frame:
            ui.html(
                f'<div id="{slider_id}_zoomwrap" class="ais-zoomwrap">'
                f'<img id="{slider_id}_after" class="ais-after">'
                f'<img id="{slider_id}_before" class="ais-before">'
                f'</div>'
                # Outside the zoomed layer, so it stays under the slider's thumb.
                f'<div id="{slider_id}_handle" class="ais-handle"></div>',
                sanitize=False,
            )
            # A ui.slider, not an <input type="range">: a focused <input> can pop
            # up the Android keyboard. The js_handler moves the clip while
            # dragging without a server round trip.
            self._range = ui.slider(min=0, max=100, step=1, value=50).props("dense").classes("ais-range")
            self._range.props(f"id={slider_id}_range")
            self._range.on(
                "update:model-value",
                js_handler=f"(value) => AIS.applyClip('{slider_id}', value)",
            )

    @classmethod
    async def create(cls) -> "CompareSlider":
        slider = cls()
        await init_widget("AIS", slider._id, "CompareSlider")
        return slider

    async def set_images(self, before_path: str, after_path: str) -> None:
        # Setting the value from Python fires no update:model-value in the
        # browser, so AIS.setImages moves the clip to match.
        self._range.set_value(50)
        await call_js("AIS", "setImages", self._id, to_url(before_path), to_url(after_path), timeout=10.0)
        self.after_path = after_path

    async def set_compare(self, enabled: bool) -> None:
        await call_js("AIS", "setCompare", self._id, bool(enabled))

    async def reset(self) -> None:
        self.after_path = None
        await call_js("AIS", "reset", self._id)
