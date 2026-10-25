# SPDX-License-Identifier: GPL-3.0-or-later
"""
NiceGUI UI. Entry point is run(model), called through ui/__init__.py
— importing this module does nothing by itself.

UI-only: renders controls, collects a GenerationParams, and calls
model.generate(). Controls (mask editor, before/after slider, reference
slots, LoRA panel, the model's optional inputs, aspect-ratio/resolution)
are shown or hidden based on the active model's declared capabilities.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from nicegui import app, ui
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationCancelled
from ai_image_edit.core.paths import WORK_DIR, to_url, trim_work_dir
from ai_image_edit.core.types import (
    DEFAULT_SEED, ORIGINAL_ASPECT_RATIO, GenerationProgress, ModelCapabilities, Postprocessing, ProgressStage, RescaleMode,
)
from ai_image_edit.ui.assets import CLIENT_JS, WIDGET_CSS
from ai_image_edit.ui.compare_slider import CompareSlider
from ai_image_edit.ui.components import GROUP_BG, GROUP_CLASSES, ImageSlot, StatusLine, apply_dark_theme, create_value_slider
from ai_image_edit.ui.helpers import (
    HOST,
    RESCALE_MODE_OPTIONS,
    PORT,
    PRIMARY_COLOR,
    RECENT_FILES_KEPT,
    describe_error,
    describe_progress,
    params_from_ui,
    progress_fraction,
    resolution_label,
)
from ai_image_edit.ui.login import install_password_login
from ai_image_edit.ui.mask_editor import MaskEditor
from ai_image_edit.version import VERSION
from ai_image_edit.workflow_model import WorkflowModel


COLOR_CORRECTION_INFO = (
    "Applies to edits without a mask. If the model shifts the colors of the whole picture, so the result "
    "looks duller, darker or tinted next to your photo, this evens the shift out, while what you asked "
    "to change keeps its new colors. Small shifts that don't show are left alone. A color change you "
    "asked for across the whole picture may be toned down, and when nearly all colors changed, nothing "
    "is corrected. It only works when the result keeps your photo's aspect ratio; otherwise the result "
    "is left as it is. "
    "With a mask, this switch has no effect: the edited area's colors are always matched to your photo."
)

MATCH_DETAIL_INFO = (
    "Applies to edits with a mask, when the edited area is kept at your original's size and had to be "
    "enlarged. The enlarged area looks softer and smoother than the rest of your photo; this sharpens it "
    "a little and adds fine grain to match your photo's own. The more it was enlarged, the stronger the "
    "effect. Turn it off to keep the area exactly as the model made it."
)

RESCALE_MODE_INFO_INTRO = "Applies to edits with a mask."
# Per RescaleMode value; the heading is the option's label in RESCALE_MODE_OPTIONS.
RESCALE_MODE_INFO = {
    RescaleMode.ORIGINAL.value: "Your original is scaled to the size the model generated at, and the result has that size.",
    RescaleMode.EDIT.value: (
        "The result keeps your original's size and every pixel outside the edited area; "
        "only the edited area is taken from the generated image, scaled to fit."
    ),
}

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


# --- Page sections ---
# Each builder creates one section of the page at the current position and
# returns the widgets other code needs.


@dataclass
class Header:
    setup_status: StatusLine


def build_header(model: WorkflowModel) -> Header:
    ui.label("AI Image Edit").classes("text-2xl font-bold text-white w-full text-center")
    with ui.column().classes("w-full gap-0 items-center q-mb-md"):
        for text in (f"Version {VERSION}", f"Model: {model.display_name}"):
            ui.label(text).classes("text-sm w-full text-center").style(f"color: {PRIMARY_COLOR}")
        # The same width as the cards below (aie-page).
        with ui.column().classes("w-full gap-0 aie-page"):
            setup_status = StatusLine("mt-3")
    return Header(setup_status)


@dataclass
class PromptCard:
    prompt: ui.textarea
    seed_input: ui.number
    randomize_seed: ui.switch


def build_prompt_card() -> PromptCard:
    with ui.card().classes(CARD_CLASSES):
        prompt = ui.textarea(label="Prompt").props("rows=6 outlined dark").classes("w-full")

        with ui.row().classes("w-full items-center gap-4"):
            seed_input = ui.number(label="Seed", value=DEFAULT_SEED, format="%d").props("outlined dark").classes("flex-1")
            randomize_seed = ui.switch("Randomize seed", value=True)
    return PromptCard(prompt, seed_input, randomize_seed)


@dataclass
class AdvancedCard:
    steps: ui.slider
    options: Dict[str, ui.element]  # the model's optional inputs' controls, by name
    rescale_mode: ui.select
    color_correction: ui.switch
    match_detail: ui.switch


def build_advanced_card(caps: ModelCapabilities) -> AdvancedCard:
    with ui.card().classes(CARD_CLASSES):
        with ui.expansion("Advanced settings").props("dense").classes("w-full"):
            with ui.column().classes(GROUP_CLASSES + " p-6"):
                ui.label("Generation").classes("text-sm text-weight-bold q-mb-sm").style(f"color: {PRIMARY_COLOR}")
                with ui.column().classes("w-full gap-0"):
                    ui.label("Inference steps").classes("text-xs text-gray-400")
                    steps = create_value_slider(caps.step_range)

                options: Dict[str, ui.element] = {}
                for spec in caps.options:
                    if spec.range is not None:
                        with ui.column().classes("w-full gap-0"):
                            ui.label(spec.label).classes("text-xs text-gray-400")
                            options[spec.name] = create_value_slider(spec.range)
                    elif spec.choices:
                        options[spec.name] = (
                            ui.select(list(spec.choices), value=spec.default, label=spec.label).props("outlined dark").classes("w-full")
                        )
                    else:
                        options[spec.name] = (
                            ui.textarea(label=spec.label, value=spec.default).props("rows=2 outlined dark").classes("w-full")
                        )

            with ui.column().classes(GROUP_CLASSES + " p-6"):
                ui.label("Postprocessing").classes("text-sm text-weight-bold q-mb-sm").style(f"color: {PRIMARY_COLOR}")
                defaults = Postprocessing()
                with ui.row().classes("w-full items-center no-wrap"):
                    rescale_mode = ui.select(
                        RESCALE_MODE_OPTIONS, value=defaults.rescale_mode.value, label="Masked edits"
                    ).props("outlined dark").classes("flex-1 min-w-0")
                    with ui.button(icon="info").props("flat round dense size=sm").classes("text-gray-400"):
                        with ui.menu():
                            with ui.column().classes("q-pa-sm gap-2").style("max-width: 280px"):
                                ui.label(RESCALE_MODE_INFO_INTRO).classes("text-xs")
                                for mode, description in RESCALE_MODE_INFO.items():
                                    with ui.column().classes("gap-0"):
                                        ui.label(RESCALE_MODE_OPTIONS[mode]).classes("text-xs text-weight-bold")
                                        ui.label(description).classes("text-xs")

                with ui.row().classes("w-full items-center no-wrap"):
                    match_detail = ui.switch("Match sharpness and grain", value=defaults.match_detail)
                    ui.space()
                    with ui.button(icon="info").props("flat round dense size=sm").classes("text-gray-400"):
                        with ui.menu():
                            ui.label(MATCH_DETAIL_INFO).classes("text-xs q-pa-sm").style("max-width: 280px")

                with ui.row().classes("w-full items-center no-wrap"):
                    color_correction = ui.switch("Apply color corrections", value=defaults.color_correction)
                    ui.space()
                    with ui.button(icon="info").props("flat round dense size=sm").classes("text-gray-400"):
                        with ui.menu():
                            ui.label(COLOR_CORRECTION_INFO).classes("text-xs q-pa-sm").style("max-width: 280px")
    return AdvancedCard(steps, options, rescale_mode, color_correction, match_detail)


FORMAT_LABELS = {"png": "PNG", "jpeg": "JPEG"}


@dataclass
class ResultCard:
    slider: CompareSlider
    compare_switch: ui.switch
    use_as_input_btn: ui.button
    download_btn: ui.dropdown_button
    progress: StatusLine

    def set_actions_enabled(self, enabled: bool) -> None:
        """Compare, Use as input and Download, which need a result to work on."""
        for control in (self.compare_switch, self.use_as_input_btn, self.download_btn):
            control.set_enabled(enabled)

    async def show(self, before_path: str, after_path: str) -> None:
        await self.slider.set_images(before_path, after_path)
        self.set_actions_enabled(True)

    async def clear(self) -> None:
        """Removes the result on show; the Compare toggle keeps its state for the next one."""
        self.set_actions_enabled(False)
        await self.slider.reset()


async def build_result_card() -> ResultCard:
    with ui.card().classes(CARD_CLASSES):
        slider = await CompareSlider.create()
        with slider.frame:
            progress = StatusLine("aie-progress-overlay")

        with ui.row().classes("w-full items-center no-wrap gap-0"):
            compare_switch = ui.switch("Compare", value=False).tooltip("Compare input / output")

            ui.space()

            use_as_input_btn = (
                ui.button(icon="move_up", color=GROUP_BG)
                .props('unelevated dense size=md text-color=primary aria-label="Use as input"')
                .classes("text-xs q-px-xs").tooltip("Use as input")
            )

            # Split: the icon downloads in the chosen format, the arrow opens the choice.
            download_format = "png"
            download_btn = (
                ui.dropdown_button("PNG", icon="download", color=GROUP_BG, split=True, auto_close=True)
                # A title, not a tooltip: the dropdown's children make up its menu.
                .props('unelevated dense no-caps size=md text-color=primary aria-label="Download result" '
                       'toggle-aria-label="Choose the download format" title="Download result"')
                .classes("text-xs q-ml-sm aie-download")
            )
            with download_btn, ui.list().props("dense").style("min-width: 150px"):
                format_items = {}
                for fmt, icon, label in (("png", "image", "PNG (lossless)"), ("jpeg", "photo", "JPEG")):
                    with ui.item(on_click=lambda fmt=fmt: select_format(fmt)) as item:
                        with ui.item_section().props("avatar").style("min-width: 32px"):
                            ui.icon(icon)
                        ui.item_section(label)
                    format_items[fmt] = item

        def select_format(choice: str) -> None:
            nonlocal download_format
            download_format = choice
            download_btn.set_text(FORMAT_LABELS[choice])
            for fmt, item in format_items.items():
                if fmt == choice:
                    item.props("active")
                else:
                    item.props(remove="active")

        async def download() -> None:
            after_path = slider.after_path
            if not after_path:
                ui.notify("No generated image to download yet.", type="negative")
                return
            if download_format == "jpeg":
                after_path = await nicegui_run.io_bound(imaging.save_as_jpeg, after_path)
            # The file's regular address, not a single-use link: phone browsers
            # may fetch a download twice, and the second fetch of such a link fails.
            ui.download.from_url(to_url(after_path), filename=Path(after_path).name)

        select_format(download_format)
        download_btn.on_click(download)

        async def on_compare_change(e) -> None:
            await slider.set_compare(e.value)

        compare_switch.on_value_change(on_compare_change)
    card = ResultCard(slider, compare_switch, use_as_input_btn, download_btn, progress)
    card.set_actions_enabled(False)
    return card


@dataclass
class EditorCard:
    aspect_ratio: ui.select
    resolution: ui.select
    editor: MaskEditor


@dataclass
class ReferenceCard:
    slots: List[ImageSlot] = field(default_factory=list)
    available_loras: Dict[str, List[str]] = field(default_factory=dict)
    lora_name: Optional[ui.select] = None
    lora_strength: Optional[ui.slider] = None


def build_reference_card(model: WorkflowModel, caps: ModelCapabilities) -> ReferenceCard:
    card = ReferenceCard()
    slots = card.slots
    if caps.max_reference_images > 1 or caps.supports_loras:
        with ui.card().classes(CARD_CLASSES):
            if caps.max_reference_images > 1:
                with ui.expansion("Reference images (optional)").props("dense").classes("w-full"):
                    for i in range(caps.max_reference_images - 1):
                        slots.append(ImageSlot(f"Input image {i + 2}"))

                    # Only the first slot shows; + / - reveal or hide
                    # the rest. Hiding a slot also clears its image, so
                    # nothing invisible is sent with the request.
                    visible = 1
                    for slot in slots[1:]:
                        slot.set_visible(False)

                    with ui.row().classes("w-full items-center gap-2"):
                        ref_remove_btn = ui.button(icon="remove").props("flat dense round size=md")
                        ref_add_btn = ui.button(icon="add").props("flat dense round size=md")
                        ref_count_label = ui.label().classes("text-xs text-gray-400")

                    def update_reference_slots() -> None:
                        ref_count_label.set_text(f"{visible} of {len(slots)}")
                        ref_remove_btn.set_enabled(visible > 1)
                        ref_add_btn.set_enabled(visible < len(slots))

                    def add_reference_slot() -> None:
                        nonlocal visible
                        if visible < len(slots):
                            slots[visible].set_visible(True)
                            visible += 1
                            update_reference_slots()

                    def remove_reference_slot() -> None:
                        nonlocal visible
                        if visible > 1:
                            visible -= 1
                            slots[visible].clear()
                            slots[visible].set_visible(False)
                            update_reference_slots()

                    ref_add_btn.on_click(add_reference_slot)
                    ref_remove_btn.on_click(remove_reference_slot)
                    update_reference_slots()

            if caps.supports_loras:
                with ui.expansion("LoRAs").props("dense").classes("w-full"):
                    card.available_loras.update(model.list_loras())
                    card.lora_name = ui.select(["None"] + list(card.available_loras), value="None", label="Name").props("outlined dark").classes("w-full")
                    with ui.column().classes("w-full gap-0 q-mt-sm"):
                        ui.label("Strength").classes("text-xs text-gray-400")
                        card.lora_strength = create_value_slider(caps.lora_strength_range)
    return card


# Aspect ratio + resolution + the unified image/mask editor. As soon
# as a mask is drawn on a model that supports inpainting, dimensions
# must come from the source image's own aspect ratio (a mask's
# coordinates are only meaningful relative to the source's own
# framing), so both Aspect ratio and Resolution are disabled while a
# mask exists — Aspect ratio additionally force-set to "Original",
# Resolution set to the tier the model will use for the image
# (model.resolution_for_source) — only when the model actually
# offers "Original" as a choice, and restored the moment the mask
# is removed.
async def build_editor_card(model: WorkflowModel, caps: ModelCapabilities) -> EditorCard:
    with ui.card().classes(CARD_CLASSES):
        with ui.row().classes("w-full items-center gap-4"):
            aspect_ratio = ui.select(
                caps.supported_aspect_ratios, value=caps.default_aspect_ratio, label="Aspect ratio"
            ).props("outlined dark").classes("flex-1")

            resolution_options = {r: resolution_label(r) for r in caps.supported_resolutions}
            resolution = ui.select(
                resolution_options, value=caps.default_resolution, label="Resolution"
            ).props("outlined dark").classes("flex-1")
            if len(caps.supported_resolutions) == 1:
                # Nothing to choose — showing a disabled single-option
                # dropdown says "this model only supports 1K" more
                # clearly than a dropdown that looks pickable but isn't.
                resolution.disable()

        previous_aspect_ratio = aspect_ratio.value
        previous_resolution = resolution.value

        # editor is assigned below; this only runs after that.
        def handle_mask_change(has_mask: bool) -> None:
            nonlocal previous_aspect_ratio, previous_resolution
            if ORIGINAL_ASPECT_RATIO not in caps.supported_aspect_ratios:
                return
            if has_mask:
                previous_aspect_ratio = aspect_ratio.value
                previous_resolution = resolution.value
                aspect_ratio.value = ORIGINAL_ASPECT_RATIO
                aspect_ratio.disable()
                # Shows the tier the model will use: it follows the image.
                source_resolution = model.resolution_for_source(editor.path) if editor.path else None
                if source_resolution is not None:
                    resolution.value = source_resolution
                resolution.disable()
            else:
                aspect_ratio.enable()
                aspect_ratio.value = previous_aspect_ratio
                resolution.value = previous_resolution
                if len(caps.supported_resolutions) > 1:
                    # Otherwise it was already permanently disabled above
                    # (nothing to choose), independent of any mask.
                    resolution.enable()

        editor = await MaskEditor.create(
            on_mask_change=handle_mask_change if caps.supports_inpainting else None,
            num_annotation_colors=caps.num_annotation_colors,
            supports_mask=caps.supports_inpainting,
        )
    return EditorCard(aspect_ratio, resolution, editor)


def run(model: WorkflowModel) -> None:
    # Everything below is model-dependent, so it lives inside run()
    # rather than at module scope: app.py builds the model once and
    # passes it in (see app.py and ui/__init__.py).

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

        header = build_header(model)

        with ui.column().classes("w-full gap-3 aie-page"):
            prompt_card = build_prompt_card()

            editor_card = await build_editor_card(model, caps)
            editor = editor_card.editor

            references = build_reference_card(model, caps)

            advanced = build_advanced_card(caps)
            result_card = await build_result_card()

            generate_btn = ui.button("Generate").props("color=primary unelevated").classes("w-full")

        # The running generation's latest progress. generate() reports it on a
        # worker thread, which only replaces the snapshot here; the timer below
        # shows it, since NiceGUI elements must not be changed from that thread.
        latest_progress = GenerationProgress(ProgressStage.PREPARING)
        started = 0.0
        # Set while a generation runs; Generate is its Cancel button then.
        cancel: Optional[threading.Event] = None

        def set_progress(progress: GenerationProgress) -> None:
            nonlocal latest_progress
            latest_progress = progress

        def refresh_progress() -> None:
            cancelling = cancel is not None and cancel.is_set()
            result_card.progress.show(
                describe_progress(latest_progress, time.monotonic() - started, cancelling),
                None if cancelling else progress_fraction(latest_progress),
            )

        progress_timer = ui.timer(0.25, refresh_progress, active=False)

        def cancel_generation() -> None:
            # Stops only this page's generation; see comfy_client.cancel_prompt()
            # for how far ComfyUI's interrupt reaches.
            cancel.set()
            generate_btn.disable()
            refresh_progress()

        def value_of(control: Optional[Any]) -> Any:
            """The value of a control the model may not have (None then)."""
            return control.value if control is not None else None

        async def do_generate() -> None:
            nonlocal cancel, latest_progress, started
            if not editor.path:
                ui.notify("Please upload an image to edit!", type="negative")
                return
            cancel = threading.Event()
            generate_btn.set_text("Cancel")
            await result_card.clear()
            try:
                source_path = editor.path
                editor_inputs = await editor.get_inputs()
                annotated_path = None
                if caps.num_annotation_colors > 0 and editor_inputs.annotation_layer_path:
                    annotated_path = await nicegui_run.io_bound(
                        imaging.compose_annotations, source_path, editor_inputs.annotation_layer_path
                    )
                lora_files = references.available_loras.get(value_of(references.lora_name), [])
                reference_paths = [slot.path for slot in references.slots if slot.path]

                params = params_from_ui(
                    caps,
                    prompt=prompt_card.prompt.value,
                    source_path=source_path,
                    mask_path=editor_inputs.mask_path,
                    annotated_path=annotated_path,
                    reference_paths=reference_paths,
                    seed=prompt_card.seed_input.value,
                    randomize_seed=prompt_card.randomize_seed.value,
                    aspect_ratio=editor_card.aspect_ratio.value,
                    resolution=editor_card.resolution.value,
                    steps=advanced.steps.value,
                    options={name: control.value for name, control in advanced.options.items()},
                    lora_files=lora_files,
                    lora_strength=value_of(references.lora_strength),
                    color_correction=advanced.color_correction.value,
                    feather_amount=editor_inputs.feather_amount,
                    rescale_mode=advanced.rescale_mode.value,
                    match_detail=advanced.match_detail.value,
                )
                latest_progress, started = GenerationProgress(ProgressStage.PREPARING), time.monotonic()
                refresh_progress()
                progress_timer.activate()
                result = await nicegui_run.io_bound(model.generate, params, set_progress, cancel)
                # The input image (which may be an older result that was used as
                # the input), the references and the result on show must outlive
                # the trim, however many files later generations add.
                in_use = [source_path, result.after_path, *reference_paths]
                await nicegui_run.io_bound(trim_work_dir, RECENT_FILES_KEPT, in_use)

                prompt_card.seed_input.value = result.actual_seed
                await result_card.show(result.before_path, result.after_path)

            except GenerationCancelled:
                pass  # the user asked for it, so there is nothing to report
            except Exception as e:  # noqa: BLE001 — surface unexpected errors instead of hanging silently
                ui.notify(describe_error(e), type="negative")
            finally:
                cancel = None
                progress_timer.deactivate()
                result_card.progress.hide()
                generate_btn.set_text("Generate")
                setup = model.setup_progress()
                generate_btn.set_enabled(setup.done and not setup.error)

        async def on_generate_click() -> None:
            if cancel is not None:
                cancel_generation()
            else:
                await do_generate()

        generate_btn.on_click(on_generate_click)

        async def use_as_input() -> None:
            after_path = result_card.slider.after_path
            if not after_path:
                return

            await editor.set_image(after_path)

            # The previous before/after pair no longer corresponds to the input
            # that's now in place, so clear it and re-disable the controls that
            # depend on a fresh result. The Compare toggle keeps its state;
            # the next result shows the divider again if it is on.
            await result_card.clear()

        result_card.use_as_input_btn.on_click(use_as_input)

        # While the model sets itself up in the background (weights and LoRAs
        # downloading, custom nodes installing, ComfyUI starting) show how far
        # it is, and add the LoRAs to the list as their files arrive.
        def refresh_setup() -> None:
            progress = model.setup_progress()
            text = progress.error or progress.message
            if text:
                header.setup_status.show(text, progress.fraction, bar=not progress.done, error=bool(progress.error))
            else:
                header.setup_status.hide()
            # A generation needs the finished setup, so Generate waits for it (and stays off after an error).
            generate_btn.set_enabled(progress.done and not progress.error)
            if references.lora_name is not None:
                found = model.list_loras()
                if found != references.available_loras:
                    references.available_loras.clear()
                    references.available_loras.update(found)
                    references.lora_name.set_options(
                        ["None"] + list(found), value=references.lora_name.value if references.lora_name.value in found else "None"
                    )
            if progress.done:
                setup_timer.deactivate()

        setup_timer = ui.timer(1.0, refresh_setup)  # stops itself once the setup is done
        refresh_setup()

    # Serve everything under WORK_DIR at /files/<n> so <img>/<canvas> loads in
    # the browser can reference uploaded/generated files directly by URL.
    app.add_static_files("/files", str(WORK_DIR))

    # Make sure whatever the active model started (a ComfyUI subprocess, ...)
    # gets torn down on shutdown instead of leaking.
    app.on_shutdown(model.shutdown)

    # APP_PASSWORD turns the login on; without it the app is open. When
    # REQUIRE_PASSWORD is set but APP_PASSWORD is not, app.py serves the error
    # page instead of this one.
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
