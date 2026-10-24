# SPDX-License-Identifier: GPL-3.0-or-later
"""
The UI's stylesheet (static/widgets.css, with the accent and frame colours filled in)
and client-side JavaScript (static/client.js), and the editor's starting values.

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
            (see MaskEditor) so it still calls AIE.setBrush directly
            in the browser, with no server round-trip. When a mask goes from
            empty to non-empty or back, AIE pushes that as an 'aie_mask_state'
            event via emitEvent(...) — the documented pattern for a
            browser-side state change to reach Python without being tied to
            one specific DOM event (see the "Custom events" section of the
            Generic Events docs) — which MaskEditor listens for with
            ui.on(...) and passes on to its on_mask_change callback, used to
            keep Aspect ratio and Resolution in sync.

      AIS — the before/after comparison: a CSS clip-path dragged by a
            ui.slider (via js_handler), entirely client-side.

      AIU — pan/zoom helpers shared by the two.
"""
from pathlib import Path

from ai_image_edit.core.types import FEATHER_RANGE
from ai_image_edit.ui.helpers import FRAME_BORDER, FRAME_BORDER_HOVER, PRIMARY_COLOR

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The starting values of the editor's sliders, which MaskEditor also passes
# to AIE.init in client.js. The maximum is shared by the mask brush and the
# annotation stroke.
DEFAULT_BRUSH_SIZE = 48
MAX_BRUSH_SIZE = 80

DEFAULT_FEATHER_AMOUNT = int(FEATHER_RANGE.default)

# Annotation stroke width in on-screen CSS pixels (converted to image pixels
# at draw time, like the brush). This is the default of the stroke slider.
ANNOTATION_STROKE = 3

WIDGET_CSS = (
    (STATIC_DIR / "widgets.css").read_text(encoding="utf-8")
    .replace("__PRIMARY_COLOR__", PRIMARY_COLOR)
    .replace("__FRAME_BORDER_HOVER__", FRAME_BORDER_HOVER)
    .replace("__FRAME_BORDER__", FRAME_BORDER)
)

CLIENT_JS = (STATIC_DIR / "client.js").read_text(encoding="utf-8")
