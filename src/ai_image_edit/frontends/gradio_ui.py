# SPDX-License-Identifier: GPL-3.0-or-later
"""
Gradio frontend — needed for Hugging Face ZeroGPU Spaces, which require the
Gradio SDK. Entry point is run(model), called by app.py —
importing this module does nothing by itself.

UI-only, same contract as the sibling frontends/nicegui.py: renders
controls, builds a GenerationParams, calls model.generate(). Every control
that depends on model.capabilities is built with visible=False rather than
never-created, so this file works unmodified against any ModelBackend.
Every image component uses type="filepath", since model.generate() only
wants plain local paths.
"""

import atexit
import hmac
import os
import uuid
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from ai_image_edit.core.paths import WORK_DIR, trim_work_dir
from ai_image_edit.core.types import DEFAULT_SEED, FEATHER_RANGE, ORIGINAL_ASPECT_RATIO
from ai_image_edit.frontends._shared.common import (
    HOST,
    PORT,
    PRIMARY_COLOR,
    RECENT_FILES_KEPT,
    default_choice,
    describe_error,
    params_from_ui,
)
from ai_image_edit.models.base import ModelBackend

# --- Configuration constants ---

# A stroke has to move at least this much total RGB distance from the
# untouched background before _extract_mask treats a pixel as painted — see
# that function's docstring for why a small tolerance is needed here at all.
_MASK_DIFF_THRESHOLD = 24

# --- Entry point ---
# Everything below is model-dependent, so it lives inside run() rather
# than at module scope: app.py builds the model once and calls this
# after picking this module via the FRONTEND env var (see app.py and
# frontends/__init__.py).

def run(model: ModelBackend) -> None:
    # Read once, at process start — every control below is shown/hidden/ranged
    # from this, exactly like main_page() reads it once per page load in the
    # NiceGUI app. Nothing below assumes any one model.
    caps = model.capabilities
    available_loras: Dict[str, List[str]] = model.list_loras() if caps.supports_loras else {}

    HAS_EDITOR_MASK = caps.supports_inpainting


    # --- Mask extraction ---

    def _extract_mask(editor_value: Optional[dict]) -> Optional[str]:
        """
        gr.ImageEditor (type="filepath") hands back {"background": path,
        "layers": [...], "composite": path}, all flattened to opaque RGB
        (image_mode="RGB"), so there's no alpha channel to read directly.
        Instead, diffs composite against background: pixels that differ by
        more than _MASK_DIFF_THRESHOLD (tolerance for PNG encoding noise)
        become the mask (white where painted, black elsewhere). Returns
        None when nothing has been drawn.
        """
        if not editor_value:
            return None
        bg_path = editor_value.get("background")
        composite_path = editor_value.get("composite")
        if not bg_path or not composite_path:
            return None

        bg = np.asarray(Image.open(bg_path).convert("RGB"), dtype=np.int16)
        composite = np.asarray(Image.open(composite_path).convert("RGB"), dtype=np.int16)
        if bg.shape != composite.shape:
            return None

        painted = np.abs(composite - bg).sum(axis=-1) > _MASK_DIFF_THRESHOLD
        if not np.any(painted):
            return None

        mask_path = WORK_DIR / f"mask_{uuid.uuid4().hex}.png"
        Image.fromarray((painted * 255).astype(np.uint8), mode="L").save(mask_path)
        return str(mask_path)

    def _quick_has_edit(editor_value: Optional[dict]) -> bool:
        """
        Cheap stand-in for "_extract_mask(editor_value) is not None", used
        to live-lock the Aspect ratio/Resolution controls on every
        edit_image.change event (fires on every brush stroke). Compares
        file sizes instead of decoding and diffing pixels, so it's cheap
        enough to run per-stroke. A heuristic, not exact — a stroke that
        happens to re-encode to the same byte size would be missed — but
        this only drives the live lock/unlock UI; the real mask still
        comes from _extract_mask(), called once in do_generate().
        """
        if not editor_value:
            return False
        bg_path = editor_value.get("background")
        composite_path = editor_value.get("composite")
        if not bg_path or not composite_path:
            return False
        try:
            return os.path.getsize(bg_path) != os.path.getsize(composite_path)
        except OSError:
            return False


    # --- Custom CSS ---
    # Named classes for this app's bespoke layout tweaks — kept minimal since
    # Gradio components already carry most of their own styling. The dark
    # purple/black palette mirrors the NiceGUI app's ui.colors(primary=PRIMARY_COLOR,
    # dark="#111111", dark_page="#000000").
    #
    # Border color/dash source (confirmed against Gradio's frontend source,
    # js/atoms/src/Block.svelte, which every component — including
    # gr.Image/gr.ImageEditor/gr.ImageSlider — wraps its content in):
    #   * border-color comes from the CSS custom property --block-border-color,
    #     border-width from --block-border-width, both read by the shared
    #     ".block" class. Setting these two on the component's own wrapper (via
    #     elem_classes) reaches every state, solid or dashed alike.
    #   * border-style is NOT class-based at all — Block.svelte sets it as an
    #     inline style directly on the element (`style:border-style={variant}`,
    #     Svelte's shorthand for a plain DOM style attribute), toggled between
    #     "solid"/"dashed" by each component's own "variant" prop (e.g.
    #     ImageEditor/ImageSlider switch to "dashed" for their empty
    #     "drop a file here" state). A plain CSS rule — however correctly
    #     targeted — can never beat an inline style; only `!important` can. This
    #     which is why the rule below puts `!important` on border-style itself,
    #     not only on border-color.
    # Component-specific class names (upload-container, image-frame, .empty, ...)
    # carry no border/dashed styling of their own in this version, so
    # overriding them has no effect.
    CUSTOM_CSS = f"""
    .aie-page {{ max-width: 720px; margin: 0 auto; }}

    /* Explicit sizes rather than relying on gr.Markdown's own "prose" heading/
       paragraph scale: that scale renders noticeably smaller than the NiceGUI
       app's "text-2xl font-bold" title / "text-sm" subtitle, which is why the
       title looked "shrunk" even with no CSS bug involved (the subtitle sitting
       right underneath it is a size/color contrast, not the actual cause), and
       is also why the subtitle read as too small on its own. Targeting the
       heading/paragraph elements directly (not the zero-specificity :where(...)
       selectors gr.Markdown's own "prose" styling uses) is enough to win the
       cascade without needing !important here. */
    .aie-title, .aie-title h1, .aie-title h2, .aie-title h3 {{
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.25rem;
    }}
    .aie-subtitle, .aie-subtitle p {{
        text-align: center;
        color: {PRIMARY_COLOR} !important;
        font-size: 1rem;
        margin-top: 0;
    }}
    footer {{ display: none !important; }}

    /* !important on the custom properties themselves (not just on the final
       border-* properties) is what makes this survive dark mode: this app's
       dark palette (toggled on below via document.body.classList.add('dark'))
       redefines --block-border-color/--block-border-width itself for the dark
       variant, and since that theme rule and this one are equal-specificity
       single-class selectors, whichever comes later in the stylesheet would
       normally win — which, for gr.ImageEditor/gr.ImageSlider specifically
       (unlike plain gr.Image), turned out to be the theme's rule, leaving the
       border grey there while the plain reference-image borders went purple.
       !important side-steps that ordering question entirely, on both the
       variables and the literal border-* properties, so it can't lose to a
       same-specificity rule regardless of source order. */
    .aie-image-border {{
        --block-border-color: {PRIMARY_COLOR} !important;
        --block-border-width: 2px !important;
        border-color: {PRIMARY_COLOR} !important;
        border-width: 2px !important;
        border-style: solid !important;
    }}
    """

    THEME = gr.themes.Default(primary_hue="purple", neutral_hue="gray")

    # Runs once, on page load. The dark-mode toggle is Gradio's own escape
    # hatch for doing this (there's no Python-level "start in dark mode"
    # option); the accept-relaxing half mirrors the NiceGUI app's
    # ui.upload(...).props('accept="*/*"') and exists for the same reason —
    # see relaxAccept's own comment below for why.
    LOAD_JS = """
    () => {
        document.body.classList.add('dark');

        // Broadens gr.Image/gr.ImageEditor's upload picker beyond
        // images-only, same rationale as the NiceGUI app's accept="*/*":
        // on Android, an image-only accept collapses the OS file picker to
        // the gallery view and hides the full file browser (folder
        // navigation, cloud storage sources, ...), which is often the only
        // practical way to reach a downloaded or shared image. This is
        // purely a picker-UX relaxation, not a validation bypass: Gradio's
        // own backend (image_utils.preprocess_image) still runs every
        // upload through PIL.Image.open() regardless of this attribute, so
        // a genuinely non-image file still gets rejected server-side with
        // an error message — this only ever changes which files the OS
        // *lets you pick*, not what the app actually accepts.
        const relaxAccept = (el) => {
            if (el.tagName === "INPUT" && el.type === "file" && el.accept) {
                el.removeAttribute("accept");
            }
        };
        document.querySelectorAll('input[type="file"]').forEach(relaxAccept);

        // Gradio's Svelte components can recreate their <input> on
        // re-render (e.g. after a value is cleared and re-uploaded), so a
        // one-time querySelectorAll at load isn't enough on its own — this
        // keeps relaxing new file inputs as they appear for the page's
        // whole lifetime, the same way the CSS !important rules elsewhere
        // in this file have to survive Gradio's own re-renders.
        new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const node of m.addedNodes) {
                    if (node.nodeType !== 1) continue;
                    if (node.matches?.('input[type="file"]')) relaxAccept(node);
                    node.querySelectorAll?.('input[type="file"]').forEach(relaxAccept);
                }
            }
        }).observe(document.body, { childList: true, subtree: true });
    }
    """

    def build_app() -> gr.Blocks:
        with gr.Blocks(
            title="AI Image Edit",
            theme=THEME,
            css=CUSTOM_CSS,
            js=LOAD_JS,
        ) as demo:
            with gr.Column(elem_classes="aie-page"):
                gr.Markdown("## AI Image Edit", elem_classes="aie-title")
                gr.Markdown(f"Model: {model.display_name}", elem_classes="aie-subtitle")

                # --- Prompt / Seed ---
                with gr.Group():
                    prompt = gr.Textbox(label="Prompt", lines=6)
                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed", value=DEFAULT_SEED, precision=0,
                            interactive=caps.supports_seed,
                        )
                        randomize_seed = gr.Checkbox(
                            label="Randomize seed", value=caps.supports_seed,
                            interactive=caps.supports_seed,
                        )

                # --- Aspect ratio / resolution / input image + mask editor ---
                with gr.Group():
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            choices=caps.supported_aspect_ratios,
                            value=caps.default_aspect_ratio,
                            label="Aspect ratio",
                        )
                        megapixels = gr.Dropdown(
                            choices=[(f"{mp:g} MP", mp) for mp in caps.supported_megapixels],
                            value=caps.default_megapixels,
                            label="Resolution",
                            interactive=len(caps.supported_megapixels) > 1,
                        )

                    if HAS_EDITOR_MASK:
                        edit_image = gr.ImageEditor(
                            label="Upload and draw mask for inpainting",
                            type="filepath",
                            format="png",
                            sources=["upload"],
                            image_mode="RGB",
                            layers=False,
                            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                            height=600,
                            elem_classes="aie-image-border",
                        )
                        feather_slider = gr.Slider(
                            minimum=FEATHER_RANGE.min, maximum=FEATHER_RANGE.max, step=FEATHER_RANGE.step,
                            value=FEATHER_RANGE.default, label="Feather",
                        )
                    else:
                        edit_image = gr.Image(
                            label="Input Image", type="filepath", format="png", sources=["upload"], height=600,
                            elem_classes="aie-image-border",
                        )
                        feather_slider = gr.Slider(
                            minimum=FEATHER_RANGE.min, maximum=FEATHER_RANGE.max, step=FEATHER_RANGE.step,
                            value=FEATHER_RANGE.default, label="Feather", visible=False,
                        )

                # --- Reference images + LoRAs ---
                reference_editors: List[gr.Image] = []
                with gr.Group(visible=(caps.max_reference_images > 1 or caps.supports_loras)):
                    if caps.max_reference_images > 1:
                        with gr.Accordion("Reference images (optional)", open=False):
                            for i in range(caps.max_reference_images - 1):
                                reference_editors.append(
                                    gr.Image(
                                        label=f"Input image {i + 2}", type="filepath",
                                        format="png", sources=["upload"],
                                        elem_classes="aie-image-border",
                                    )
                                )

                    if caps.supports_loras:
                        with gr.Accordion("LoRAs", open=False):
                            lora_name = gr.Dropdown(
                                choices=["None"] + list(available_loras.keys()), value="None", label="Name",
                            )
                            ls = caps.lora_strength_range
                            lora_strength = gr.Slider(
                                minimum=ls.min, maximum=ls.max, step=ls.step, value=ls.default, label="Strength",
                            )
                    else:
                        # Created hidden (rather than omitted) so do_generate's
                        # fixed input signature stays the same regardless of
                        # whether this model supports LoRAs.
                        lora_name = gr.Dropdown(choices=["None"], value="None", label="Name", visible=False)
                        lora_strength = gr.Slider(minimum=0, maximum=1, step=0.05, value=0, label="Strength", visible=False)

                # --- Advanced settings ---
                with gr.Group():
                    with gr.Accordion("Advanced settings", open=False):
                        sr = caps.step_range
                        steps = gr.Slider(minimum=sr.min, maximum=sr.max, step=sr.step, value=sr.default, label="Inference steps")

                        cfg = gr.Slider(
                            minimum=caps.cfg_range.min, maximum=caps.cfg_range.max,
                            step=caps.cfg_range.step, value=caps.cfg_range.default,
                            label="CFG scale", visible=caps.supports_cfg,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt", lines=2, visible=caps.supports_negative_prompt,
                        )
                        denoise = gr.Slider(
                            minimum=caps.denoise_range.min, maximum=caps.denoise_range.max,
                            step=caps.denoise_range.step, value=caps.denoise_range.default,
                            label="Denoise", visible=caps.supports_denoise,
                        )
                        sampler_name = gr.Dropdown(
                            choices=caps.sampler_choices or [], value=default_choice(caps.default_sampler, caps.sampler_choices),
                            label="Sampler name", visible=bool(caps.sampler_choices),
                        )
                        scheduler = gr.Dropdown(
                            choices=caps.scheduler_choices or [], value=default_choice(caps.default_scheduler, caps.scheduler_choices),
                            label="Scheduler", visible=bool(caps.scheduler_choices),
                        )
                        apply_color_correction = gr.Checkbox(label="Apply color corrections", value=False)

                # --- Result panel ---
                with gr.Group():
                    # gr.ImageSlider is Gradio's own before/after comparison
                    # widget (a draggable divider between two images) — a
                    # closer match to the NiceGUI app's comparison slider than
                    # two plain gr.Image components toggled by a switch, so
                    # there's no separate "Compare input/output" control here.
                    result_slider = gr.ImageSlider(
                        format="png", label="Result", show_label=False, interactive=False, type="filepath",
                        elem_classes="aie-image-border",
                    )

                    with gr.Row():
                        use_as_input_btn = gr.Button("Use as input", interactive=False)
                        download_btn = gr.DownloadButton("Download result", interactive=False)

                generate_btn = gr.Button("Generate", variant="primary")

            # --- Aspect ratio + resolution lock while a mask exists ---
            # As soon as a mask is drawn on a model that supports inpainting, the
            # output must derive its dimensions from the source image's own
            # framing, so both Aspect ratio and Resolution are disabled while a
            # mask exists — Aspect ratio additionally forced to ORIGINAL_ASPECT_RATIO,
            # Resolution set to the tier the model will use for the image
            # (model.megapixels_for_source) — only
            # when "Original" is actually one of the model's
            # supported_aspect_ratios — mirroring handle_mask_change() in the
            # NiceGUI app (simplified: it doesn't restore the exact prior
            # selection, just re-enables the dropdowns once the mask is cleared).
            if HAS_EDITOR_MASK and ORIGINAL_ASPECT_RATIO in caps.supported_aspect_ratios:
                def on_editor_change(editor_value):
                    # _quick_has_edit(), not _extract_mask(): this fires on
                    # every edit_image.change event (every brush stroke
                    # while drawing, not just once), so it uses the cheap
                    # file-size heuristic rather than decoding and diffing
                    # full images each time. The precise mask (via
                    # _extract_mask) is still computed exactly once, in
                    # do_generate() — this function only ever decides
                    # whether to lock the dropdowns, never what mask is
                    # actually sent to the model.
                    has_mask = _quick_has_edit(editor_value)
                    if has_mask:
                        # Shows the tier the model will use: it follows the image.
                        source_mp = model.megapixels_for_source(editor_value["background"])
                        mp_update = gr.update(interactive=False) if source_mp is None else gr.update(value=source_mp, interactive=False)
                        return gr.update(value=ORIGINAL_ASPECT_RATIO, interactive=False), mp_update
                    return gr.update(interactive=True), gr.update(interactive=len(caps.supported_megapixels) > 1)

                edit_image.change(fn=on_editor_change, inputs=edit_image, outputs=[aspect_ratio, megapixels])

            # --- Generate ---
            def do_generate(
                prompt_val, editor_val, seed_val, randomize_val, ar_val, mp_val,
                steps_val, cfg_val, negprompt_val, denoise_val, sampler_val, scheduler_val,
                lora_name_val, lora_strength_val, color_corr_val, feather_val,
                *ref_vals,
            ):
                if HAS_EDITOR_MASK:
                    source_path = editor_val.get("background") if editor_val else None
                else:
                    source_path = editor_val
                if not source_path:
                    raise gr.Error("Please upload an image to edit!")

                mask_path = _extract_mask(editor_val) if HAS_EDITOR_MASK else None
                reference_paths = [p for p in ref_vals if p]
                lora_files = available_loras.get(lora_name_val, []) if caps.supports_loras else []

                params = params_from_ui(
                    caps,
                    prompt=prompt_val,
                    source_image_path=source_path,
                    mask_path=mask_path,
                    reference_images=reference_paths,
                    seed=seed_val,
                    randomize_seed=randomize_val,
                    aspect_ratio=ar_val,
                    target_megapixels=mp_val,
                    steps=steps_val,
                    cfg=cfg_val,
                    denoise=denoise_val,
                    sampler_name=sampler_val,
                    scheduler=scheduler_val,
                    negative_prompt=negprompt_val,
                    lora_files=lora_files,
                    lora_strength=lora_strength_val,
                    apply_color_correction=color_corr_val,
                    feather_amount=feather_val,
                )

                try:
                    result = model.generate(params)
                except Exception as e:  # noqa: BLE001 — surface unexpected errors instead of hanging silently
                    raise gr.Error(describe_error(e))
                trim_work_dir(RECENT_FILES_KEPT)

                return (
                    result.actual_seed,
                    (result.before_path, result.after_path),
                    gr.update(interactive=True),
                    gr.update(value=result.after_path, interactive=True),
                )

            generate_btn.click(
                fn=do_generate,
                inputs=[
                    prompt, edit_image, seed_input, randomize_seed, aspect_ratio, megapixels,
                    steps, cfg, negative_prompt, denoise, sampler_name, scheduler,
                    lora_name, lora_strength, apply_color_correction, feather_slider,
                    *reference_editors,
                ],
                outputs=[seed_input, result_slider, use_as_input_btn, download_btn],
            )

            # --- Use as input ---
            def use_as_input(slider_value: Optional[Tuple[str, str]]):
                if not slider_value or not slider_value[1]:
                    return gr.update(), gr.update(), gr.update(), gr.update()

                after_path = slider_value[1]
                new_editor_value = (
                    {"background": after_path, "layers": [], "composite": after_path}
                    if HAS_EDITOR_MASK else after_path
                )

                return (
                    new_editor_value,
                    gr.update(value=None),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

            use_as_input_btn.click(
                fn=use_as_input,
                inputs=result_slider,
                outputs=[edit_image, result_slider, use_as_input_btn, download_btn],
            )

        return demo


    demo = build_app()

    # Make sure whatever the active model started (a ComfyUI subprocess, a loaded
    # pipeline, ...) gets torn down when the process exits instead of leaking.
    # Blocks.unload() is deliberately NOT used for this — it fires per browser
    # tab disconnect, not on process shutdown, which would tear the model down
    # under one user just because a different tab closed. atexit mirrors the
    # NiceGUI app's app.on_shutdown(model.shutdown) (a once-per-process hook).
    atexit.register(model.shutdown)


    # APP_PASSWORD turns the login on; without it the app is open. Any
    # username is accepted, only the password is checked. app.py refuses to
    # start when REQUIRE_PASSWORD is set but APP_PASSWORD is not.
    password = os.environ.get("APP_PASSWORD", "")
    auth = None
    if password:
        def auth(username: str, entered: str) -> bool:
            return hmac.compare_digest(entered.encode(), password.encode())

    demo.queue().launch(
        server_name=HOST,
        server_port=PORT,
        allowed_paths=[str(WORK_DIR)],
        auth=auth,
    )
