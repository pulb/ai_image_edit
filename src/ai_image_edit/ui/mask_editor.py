# SPDX-License-Identifier: GPL-3.0-or-later
"""The mask and annotation editor (an HTML canvas drawn on in the browser)."""
import uuid
from typing import Callable, Dict, NamedTuple, Optional

from nicegui import ui

from ai_image_edit.core.paths import to_url
from ai_image_edit.core.types import ANNOTATION_COLORS, FEATHER_RANGE, RangeSpec
from ai_image_edit.ui.building_blocks import GROUP_CLASSES, PLACEHOLDER_TEXT, SliderPopup, attach_file_picker, call_js, create_image_frame, init_widget, save_data_url
from ai_image_edit.ui.helpers import PRIMARY_COLOR

# The starting values of the editor's sliders, in on-screen CSS pixels (turned
# into image pixels at draw time), also passed to AIE.init in client.js. The
# maximum is shared by the mask brush and the annotation stroke.
DEFAULT_BRUSH_SIZE = 48
MAX_BRUSH_SIZE = 80
ANNOTATION_STROKE = 3


class EditorInputs(NamedTuple):
    """What the editor holds at Generate time: saved layer files (None if not drawn) and the Feather amount."""

    mask_path: Optional[str]
    annotation_layer_path: Optional[str]
    feather_amount: int


class MaskEditor:
    """
    The "Input Image" widget: click-to-upload image + a <canvas> for
    drawing an optional inpainting mask, with a resizable brush. A radio
    group switches between pinch-to-zoom (default), mask drawing and,
    when num_annotation_colors > 0, annotating in one of that many colours
    (ANNOTATION_COLORS, with an adjustable stroke width). A stroke in mask
    mode makes the edit a masked one until "Remove mask"; without
    supports_mask there is no mask mode. The Feather slider
    (FEATHER_RANGE) is the Gaussian blur radius composite_with_soft_transition()
    applies to the mask edges; 0 falls back to a hard cutout.

    If given, on_mask_change(has_mask) fires whenever the mask goes from
    empty to non-empty or back (via the 'aie_mask_state' custom event).

    `path` is the current source image's local path (or None). get_inputs()
    is async: one browser call that reads the mask, the annotation layer and
    the Feather amount, and saves the two layers to disk; called once, on
    Generate. set_image(path) loads a new image and clears the mask and
    annotations. Create it with create().
    """

    def __init__(
        self,
        on_mask_change: Optional[Callable[[bool], None]] = None,
        num_annotation_colors: int = 0,
        supports_mask: bool = True,
    ) -> None:
        self.path: Optional[str] = None
        self._id = editor_id = f"edit_{uuid.uuid4().hex}"

        if on_mask_change:
            def handle_mask_event(e) -> None:
                args = list(e.args) if isinstance(e.args, (list, tuple)) else [e.args]
                if len(args) < 2 or args[0] != editor_id:
                    return
                on_mask_change(bool(args[1]))

            ui.on("aie_mask_state", handle_mask_event)

        editor_box, self._badge = create_image_frame("aie-editor-box", self._clear_image)
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

        # Once an image is loaded, a click on the box draws on the canvas instead
        # of reopening the picker — self.path already tells us which state
        # we're in, no need to ask the browser.
        attach_file_picker(editor_box, self.set_image, enabled=lambda: not self.path)

        self._radios: Dict[str, ui.element] = {}
        self._swatches: Dict[str, ui.button] = {}
        if supports_mask or num_annotation_colors > 0:
            self._build_controls(supports_mask, num_annotation_colors)

    @classmethod
    async def create(cls, **options) -> "MaskEditor":
        """A MaskEditor (options are __init__'s), with its browser side set up."""
        editor = cls(**options)
        await init_widget("AIE", editor._id, "MaskEditor", {
            "brush": DEFAULT_BRUSH_SIZE,
            "feather": int(FEATHER_RANGE.default),
            "annotationColor": ANNOTATION_COLORS[0],
            "annotationStroke": ANNOTATION_STROKE,
            "maskTint": PRIMARY_COLOR,
        })
        return editor

    def _build_controls(self, supports_mask: bool, num_annotation_colors: int) -> None:
        editor_id = self._id
        with ui.column().classes(GROUP_CLASSES + " p-4"):
            # One q-radio per row (ui.radio is a single vertical option list and
            # cannot hold the sliders), all showing the same current mode.
            self._radios["zoom"] = ui.element("q-radio").props('val=zoom model-value=zoom label="Pinch to zoom"')
            row_controls = {}

            if supports_mask:
                with ui.row().classes("w-full items-center gap-3 no-wrap"):
                    self._radios["mask"] = ui.element("q-radio").props('val=mask model-value=zoom label="Mask"')

                    # The brush size and feather are only used in the browser, so their
                    # js_on_value handlers pass them on there, without a server round-trip
                    # (see the module docstring of assets.py for why that matters for a
                    # dragged brush size). 'update:model-value' fires continuously while
                    # dragging, 'change' once on release.
                    brush_popup = SliderPopup(
                        "brush", "Brush size", RangeSpec(5, MAX_BRUSH_SIZE, DEFAULT_BRUSH_SIZE),
                        js_on_value=f"AIE.setBrush('{editor_id}', value); AIE.showBrushPreview('{editor_id}', value);",
                    )
                    brush_popup.slider.on("change", js_handler=f"() => AIE.hideBrushPreview('{editor_id}')")
                    feather_popup = SliderPopup(
                        "blur_circular", "Feather", FEATHER_RANGE, js_on_value=f"AIE.setFeather('{editor_id}', value);",
                    )
                    row_controls["mask"] = [brush_popup.button, feather_popup.button]

                    ui.space()
                    ui.button(icon="layers_clear", on_click=self._remove_mask).props("flat dense size=md").classes("text-xs").tooltip("Remove mask")

            if num_annotation_colors > 0:
                with ui.row().classes("w-full items-center gap-3 no-wrap"):
                    self._radios["annotate"] = ui.element("q-radio").props('val=annotate model-value=zoom label="Annotate"')

                    for color in ANNOTATION_COLORS[:num_annotation_colors]:
                        self._swatches[color] = (
                            ui.button(on_click=lambda color=color: self._select_color(color))
                            .props("unelevated dense")
                            .classes("aie-swatch" + (" aie-swatch-selected" if color == ANNOTATION_COLORS[0] else ""))
                            .style(f"background: {color} !important")
                        )

                    stroke_popup = SliderPopup(
                        "brush", "Stroke width", RangeSpec(1, MAX_BRUSH_SIZE, ANNOTATION_STROKE),
                        js_on_value=f"AIE.setAnnotationStroke('{editor_id}', value); AIE.showBrushPreview('{editor_id}', value);",
                    )
                    stroke_popup.slider.on("change", js_handler=f"() => AIE.hideBrushPreview('{editor_id}')")
                    row_controls["annotate"] = [stroke_popup.button]

                    ui.space()
                    ui.button(icon="layers_clear", on_click=self._remove_annotations).props("flat dense size=md").classes("text-xs").tooltip(
                        "Remove annotations"
                    )

        for mode, radio in self._radios.items():
            radio.on("update:model-value", lambda mode=mode: self._set_mode(mode))
        # Like the swatches, every control of a row selects that row's mode.
        for mode, controls in row_controls.items():
            for control in controls:
                control.on("click", lambda mode=mode: self._set_mode(mode))

    async def _set_mode(self, mode: str) -> None:
        for radio in self._radios.values():
            radio.props(f"model-value={mode}")
        await call_js("AIE", "setMode", self._id, mode)

    async def _select_color(self, color: str) -> None:
        for c, btn in self._swatches.items():
            if c == color:
                btn.classes(add="aie-swatch-selected")
            else:
                btn.classes(remove="aie-swatch-selected")
        await call_js("AIE", "setAnnotationColor", self._id, color)
        await self._set_mode("annotate")

    async def _remove_mask(self) -> None:
        await call_js("AIE", "clearMask", self._id)

    async def _remove_annotations(self) -> None:
        await call_js("AIE", "clearAnnotations", self._id)

    async def _clear_image(self) -> None:
        self.path = None
        self._badge.set_visibility(False)
        await call_js("AIE", "clearImage", self._id)

    async def set_image(self, path: str) -> None:
        self.path = path
        await call_js("AIE", "loadImage", self._id, to_url(path), timeout=10.0)
        self._badge.set_visibility(True)

    async def get_inputs(self) -> EditorInputs:
        data = await call_js("AIE", "getInputs", self._id, timeout=15.0) if self.path else None
        if not data:
            return EditorInputs(None, None, int(FEATHER_RANGE.default))
        return EditorInputs(
            save_data_url(data["mask"], "mask") if data["mask"] else None,
            save_data_url(data["annotations"], "annotations") if data["annotations"] else None,
            int(data["feather"]),
        )
