# SPDX-License-Identifier: GPL-3.0-or-later
"""
The UI's stylesheet (static/widgets.css) and client-side JavaScript
(static/client.js), with the values they share with the Python side filled in.

Stylesheet:
    Named classes for this app's bespoke widget markup (the mask editor,
    upload boxes, and comparison slider) — injected once via ui.add_head_html,
    the same mechanism already used for CLIENT_JS. These are ordinary classes
    this app defines itself, so — unlike NiceGUI's bundled Tailwind utility
    classes, which only cover whatever subset NiceGUI happens to ship — their
    existence is never in question, and there's exactly one place to look if
    something needs to change. Values that get overridden dynamically at
    runtime (AIS's clip-path/left/transform, AIE's brush-preview size) are
    still set here as sensible defaults; a plain inline style set directly via
    JS (element.style.foo = ...) always takes precedence over a class
    regardless, so the two coexist without conflict. These rules also don't
    need !important to beat Quasar's own defaults: NiceGUI 3.0 moved Quasar's
    base styles into a CSS @layer, and per the cascade spec, any rule outside
    a layer — including this plain <style> block — already outranks layered
    rules regardless of specificity.

Client-side JS:
    Three small namespaces, defined once as generic functions keyed by an
    element-id prefix so each widget instance just calls e.g. AIE.init('some_id')
    after creating its own DOM elements with matching ids. Everything that
    NiceGUI already has a documented element/API for — file pickers (ui.upload
    + element.run_method('pickFiles'), the same mechanism NiceGUI's own
    Upload.reset() uses internally), image previews (ui.image.set_source) —
    uses that instead of custom JS. What's left needs custom JS because
    there's no NiceGUI equivalent:

      AIE — the mask/image editor: a <canvas> drawn on with mouse/touch
            handlers. NiceGUI's ui.interactive_image can report mouse
            coordinates, but it round-trips every single mouse-move event to
            the server to do so, which would make drawing feel laggy. A raw
            <canvas> keeps every brush stroke fully client-side; the Brush
            size slider is a regular ui.slider, but wired up with js_handler
            (see create_mask_editor) so it still calls AIE.setBrush directly
            in the browser, with no server round-trip. When a mask goes from
            empty to non-empty or back, AIE pushes that as an 'aie_mask_state'
            event via emitEvent(...) — the documented pattern for a
            browser-side state change to reach Python without being tied to
            one specific DOM event (see the "Custom events" section of the
            Generic Events docs) — which create_mask_editor listens for with
            ui.on(...) and passes on to its on_mask_change callback, used to
            keep Aspect ratio and Resolution in sync.

      AIS — the before/after comparison: a CSS clip-path dragged by a
            ui.slider (via js_handler), entirely client-side.

      AIU — pan/zoom helpers shared by the two.
"""
from pathlib import Path

from ai_image_edit.core.types import ANNOTATION_COLORS, FEATHER_RANGE
from ai_image_edit.ui.helpers import PRIMARY_COLOR

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Brush min/max stay hardcoded on the slider itself (5-80) since nothing
# else in the app references them — only the *default* needs a shared
# constant, since it's duplicated into the JS-side initial state
# (__DEFAULT_BRUSH_SIZE__ in client.js) as well as the Python-side ui.slider.
DEFAULT_BRUSH_SIZE = 48

# Same idea for feather: the default is duplicated into the JS initial state
# (__DEFAULT_FEATHER_AMOUNT__) and the slider.
DEFAULT_FEATHER_AMOUNT = int(FEATHER_RANGE.default)

# Annotation stroke width in on-screen CSS pixels (converted to image pixels
# at draw time, like the brush). Thin and fixed: there is no slider.
ANNOTATION_STROKE = 3

WIDGET_CSS = (STATIC_DIR / "widgets.css").read_text(encoding="utf-8").replace("__PRIMARY_COLOR__", PRIMARY_COLOR)

CLIENT_JS = (
    (STATIC_DIR / "client.js").read_text(encoding="utf-8")
    .replace("__DEFAULT_BRUSH_SIZE__", str(DEFAULT_BRUSH_SIZE))
    .replace("__DEFAULT_FEATHER_AMOUNT__", str(DEFAULT_FEATHER_AMOUNT))
    .replace("__MASK_TINT__", PRIMARY_COLOR)
    .replace("__ANNOTATION_COLOR__", ANNOTATION_COLORS[0])
    .replace("__ANNOTATION_STROKE__", str(ANNOTATION_STROKE))
)
