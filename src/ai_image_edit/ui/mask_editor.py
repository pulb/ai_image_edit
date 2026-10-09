# SPDX-License-Identifier: GPL-3.0-or-later
"""The mask and annotation editor (an HTML canvas drawn on in the browser)."""
import uuid
from typing import Awaitable, Callable, Dict, NamedTuple, Optional, Tuple

from nicegui import app, ui

from ai_image_edit.core.paths import to_url
from ai_image_edit.core.types import ANNOTATION_COLORS, FEATHER_RANGE
from ai_image_edit.ui.assets import DEFAULT_BRUSH_SIZE, DEFAULT_FEATHER_AMOUNT
from ai_image_edit.ui.components import PLACEHOLDER_TEXT, attach_file_picker, call_js, create_image_frame, init_widget, save_data_url


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
    makes this an Inpaint Edit, "Remove mask" reverts to a plain Image Edit.
    The Feather slider (FEATHER_RANGE) is the Gaussian blur radius
    composite_with_soft_transition() applies to the mask edges; 0 falls back
    to a hard cutout.

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
                # (see the module docstring of assets.py for why that matters
                # for a dragged brush size) — 'update:model-value' fires continuously
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
