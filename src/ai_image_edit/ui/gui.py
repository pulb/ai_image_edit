# SPDX-License-Identifier: GPL-3.0-or-later
"""
NiceGUI UI. Entry point is run(model), called through ui/__init__.py
— importing this module does nothing by itself.

UI-only: renders controls, collects a GenerationParams, and calls
model.generate(). Controls (mask editor, before/after slider, reference
slots, LoRA panel, sampler/scheduler/cfg/denoise, aspect-ratio/resolution)
are shown or hidden based on the active model's declared capabilities.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nicegui import app, ui
from nicegui import run as nicegui_run

from ai_image_edit.core import imaging
from ai_image_edit.core.paths import WORK_DIR, trim_work_dir
from ai_image_edit.core.types import DEFAULT_SEED, ORIGINAL_ASPECT_RATIO, ModelCapabilities
from ai_image_edit.ui.assets import CLIENT_JS, WIDGET_CSS
from ai_image_edit.ui.compare_slider import create_compare_slider
from ai_image_edit.ui.components import apply_dark_theme, create_simple_image_upload, create_value_slider
from ai_image_edit.ui.helpers import (
    HOST,
    PORT,
    PRIMARY_COLOR,
    RECENT_FILES_KEPT,
    default_choice,
    describe_error,
    params_from_ui,
)
from ai_image_edit.ui.login import install_password_login
from ai_image_edit.ui.mask_editor import create_mask_editor
from ai_image_edit.workflow_model import WorkflowModel


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
    setup_label: ui.label
    setup_bar: ui.linear_progress


def build_header(model: WorkflowModel) -> Header:
    ui.label("AI Image Edit").classes("text-2xl font-bold text-white w-full text-center")
    with ui.column().classes("w-full gap-0 items-center q-mb-md"):
        ui.label(f"Model: {model.display_name}").classes("text-sm w-full text-center").style(f"color: {PRIMARY_COLOR}")
        setup_label = ui.label().classes("text-xs w-full text-center")
        setup_label.set_visibility(False)
        setup_bar = ui.linear_progress(value=0, show_value=False, size="4px").props("rounded").classes("w-full q-mt-xs")
        setup_bar.set_visibility(False)
    return Header(setup_label, setup_bar)


@dataclass
class PromptCard:
    prompt: ui.textarea
    seed_input: ui.number
    randomize_seed: ui.switch


def build_prompt_card(caps: ModelCapabilities) -> PromptCard:
    with ui.card().classes(CARD_CLASSES):
        prompt = ui.textarea(label="Prompt").props("rows=6 outlined dark").classes("w-full")

        with ui.row().classes("w-full items-center gap-4"):
            seed_input = ui.number(label="Seed", value=DEFAULT_SEED, format="%d").props("outlined dark").classes("flex-1")
            randomize_seed = ui.switch("Randomize seed", value=True)
    return PromptCard(prompt, seed_input, randomize_seed)


@dataclass
class AdvancedCard:
    steps: ui.slider
    cfg: Optional[ui.slider]
    negative_prompt: Optional[ui.textarea]
    denoise: Optional[ui.slider]
    sampler_name: Optional[ui.select]
    scheduler: Optional[ui.select]
    color_correction: ui.switch


def build_advanced_card(caps: ModelCapabilities) -> AdvancedCard:
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

            color_correction = ui.switch("Apply color corrections (unmasked only)", value=False).classes("q-mt-sm")
    return AdvancedCard(steps, cfg, negative_prompt, denoise, sampler_name, scheduler, color_correction)


@dataclass
class ResultCard:
    set_result_images: Callable
    set_compare: Callable
    reset_result: Callable
    get_result_after_path: Callable
    compare_switch: ui.switch
    use_as_input_btn: ui.button
    download_btn: ui.button


async def build_result_card() -> ResultCard:
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
    return ResultCard(
        set_result_images, set_compare, reset_result, get_result_after_path,
        compare_switch, use_as_input_btn, download_btn,
    )


@dataclass
class EditorCard:
    aspect_ratio: ui.select
    megapixels: ui.select
    editor_holder: dict
    get_editor_inputs: Callable
    set_editor_image: Callable


@dataclass
class ReferenceCard:
    reference_holders: List[dict] = field(default_factory=list)
    available_loras: Dict[str, List[str]] = field(default_factory=dict)
    lora_name: Optional[ui.select] = None
    lora_strength: Optional[ui.slider] = None


def build_reference_card(model: WorkflowModel, caps: ModelCapabilities) -> ReferenceCard:
    card = ReferenceCard()
    reference_holders = card.reference_holders
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

            if caps.supports_loras:
                with ui.expansion("LoRAs").props("dense").classes("w-full"):
                    card.available_loras.update(model.list_loras())
                    card.lora_name = ui.select(["None"] + list(card.available_loras), value="None", label="Name").props("outlined dark").classes("w-full")
                    ui.label("Strength").classes("text-xs text-gray-400 q-mt-sm")
                    card.lora_strength = create_value_slider(caps.lora_strength_range)
    return card



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
async def build_editor_card(model: WorkflowModel, caps: ModelCapabilities) -> EditorCard:
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

        # editor_holder is assigned below, by create_mask_editor; this only runs after that.
        def handle_mask_change(has_mask: bool) -> None:
            nonlocal previous_aspect_ratio, previous_megapixels
            if ORIGINAL_ASPECT_RATIO not in caps.supported_aspect_ratios:
                return
            if has_mask:
                previous_aspect_ratio = aspect_ratio.value
                previous_megapixels = megapixels.value
                aspect_ratio.value = ORIGINAL_ASPECT_RATIO
                aspect_ratio.disable()
                # Shows the tier the model will use: it follows the image.
                source_mp = model.megapixels_for_source(editor_holder["path"]) if editor_holder["path"] else None
                if source_mp is not None:
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
            supports_mask=caps.supports_inpainting,
        )
    return EditorCard(aspect_ratio, megapixels, editor_holder, get_editor_inputs, set_editor_image)


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

        header = build_header(model)

        with ui.column().classes("w-full gap-3 aie-page"):
            prompt_card = build_prompt_card(caps)

            editor = await build_editor_card(model, caps)

            references = build_reference_card(model, caps)

            advanced = build_advanced_card(caps)
            result_card = await build_result_card()

            generate_btn = ui.button("Generate").props("color=primary unelevated").classes("w-full")

        async def do_generate() -> None:
            generate_btn.props("loading")
            generate_btn.disable()
            result_card.compare_switch.disable()
            result_card.use_as_input_btn.disable()
            result_card.download_btn.disable()
            await result_card.reset_result()
            await result_card.set_compare(result_card.compare_switch.value)
            try:
                if not editor.editor_holder["path"]:
                    ui.notify("Please upload an image to edit!", type="negative")
                    return

                source_image_path = editor.editor_holder["path"]
                editor_inputs = await editor.get_editor_inputs()
                annotated_image_path = None
                if caps.num_annotation_colors > 0 and editor_inputs.annotation_layer_path:
                    annotated_image_path = await nicegui_run.io_bound(
                        imaging.compose_annotations, source_image_path, editor_inputs.annotation_layer_path
                    )
                lora_files = references.available_loras.get(references.lora_name.value, []) if references.lora_name is not None else []

                params = params_from_ui(
                    caps,
                    prompt=prompt_card.prompt.value,
                    source_image_path=source_image_path,
                    mask_path=editor_inputs.mask_path,
                    annotated_image_path=annotated_image_path,
                    reference_images=[h["path"] for h in references.reference_holders if h["path"]],
                    seed=prompt_card.seed_input.value,
                    randomize_seed=prompt_card.randomize_seed.value,
                    aspect_ratio=editor.aspect_ratio.value,
                    target_megapixels=editor.megapixels.value,
                    steps=advanced.steps.value,
                    cfg=advanced.cfg.value if advanced.cfg is not None else None,
                    denoise=advanced.denoise.value if advanced.denoise is not None else None,
                    sampler_name=advanced.sampler_name.value if advanced.sampler_name is not None else None,
                    scheduler=advanced.scheduler.value if advanced.scheduler is not None else None,
                    negative_prompt=advanced.negative_prompt.value if advanced.negative_prompt is not None else None,
                    lora_files=lora_files,
                    lora_strength=references.lora_strength.value if references.lora_strength is not None else None,
                    apply_color_correction=advanced.color_correction.value,
                    feather_amount=editor_inputs.feather_amount,
                )
                result = await nicegui_run.io_bound(model.generate, params)
                await nicegui_run.io_bound(trim_work_dir, RECENT_FILES_KEPT)

                prompt_card.seed_input.value = result.actual_seed
                await result_card.set_result_images(result.before_path, result.after_path)
                result_card.compare_switch.enable()
                result_card.use_as_input_btn.enable()
                result_card.download_btn.enable()

            except Exception as e:  # noqa: BLE001 — surface unexpected errors instead of hanging silently
                ui.notify(describe_error(e), type="negative")
            finally:
                generate_btn.props(remove="loading")
                generate_btn.enable()

        generate_btn.on_click(do_generate)

        async def use_as_input() -> None:
            after_path = await result_card.get_result_after_path()
            if not after_path:
                return

            await editor.set_editor_image(after_path)

            # The previous before/after pair no longer corresponds to the input
            # that's now in place, so clear it and re-disable the controls that
            # depend on a fresh result. The Compare toggle's own checked/
            # unchecked state is left alone (not the visual divider, which
            # reset() unconditionally hides) — reapplying it right after
            # restores the divider immediately if it was checked, matching the
            # same reset-then-reapply approach as do_generate.
            await result_card.reset_result()
            await result_card.set_compare(result_card.compare_switch.value)
            result_card.compare_switch.disable()
            result_card.use_as_input_btn.disable()
            result_card.download_btn.disable()

        result_card.use_as_input_btn.on_click(use_as_input)

        # While the model sets itself up in the background (weights and LoRAs
        # downloading, custom nodes installing) show how far it is, and add the
        # LoRAs to the list as their files arrive.
        def refresh_setup() -> None:
            progress = model.setup_progress()
            text = progress.error or progress.message
            header.setup_label.set_text(text)
            header.setup_label.set_visibility(bool(text))
            header.setup_label.classes(add="text-negative" if progress.error else "text-gray-400",
                                remove="text-gray-400" if progress.error else "text-negative")
            running = not progress.done
            header.setup_bar.set_visibility(running)
            if running:
                if progress.fraction is None:
                    header.setup_bar.props("indeterminate")
                else:
                    header.setup_bar.props(remove="indeterminate")
                    header.setup_bar.set_value(progress.fraction)
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

        async def download_result() -> None:
            after_path = await result_card.get_result_after_path()
            if not after_path:
                ui.notify("No generated image to download yet.", type="negative")
                return
            ui.download.file(after_path, filename=Path(after_path).name)

        result_card.download_btn.on_click(download_result)


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
