# SPDX-License-Identifier: GPL-3.0-or-later
"""
The UI's stylesheet (static/widgets.css, with the accent and frame colours filled in)
and client-side JavaScript (static/client.js).

The stylesheet styles this app's own widget markup (mask editor, upload boxes,
comparison slider). It needs no !important against Quasar: NiceGUI puts
Quasar's styles in a CSS @layer, and unlayered rules outrank layered ones.

The JavaScript is what needs to stay in the browser, in three namespaces keyed
by an element id prefix (e.g. AIE.init(id)):

    AIE  the mask/image editor: drawing on a <canvas> without a server round
         trip per mouse move (ui.interactive_image sends every one). It
         reports a mask appearing or disappearing as the 'aie_mask_state'
         event (emitEvent), which MaskEditor passes on.
    AIS  the before/after comparison: a clip-path dragged by a ui.slider
         through a js_handler.
    AIU  pan and zoom, shared by the two.
"""
from pathlib import Path

from ai_image_edit.ui.helpers import FRAME_BORDER, FRAME_BORDER_HOVER, PRIMARY_COLOR

STATIC_DIR = Path(__file__).resolve().parent / "static"

WIDGET_CSS = (
    (STATIC_DIR / "widgets.css").read_text(encoding="utf-8")
    .replace("__PRIMARY_COLOR__", PRIMARY_COLOR)
    .replace("__FRAME_BORDER_HOVER__", FRAME_BORDER_HOVER)
    .replace("__FRAME_BORDER__", FRAME_BORDER)
)

CLIENT_JS = (STATIC_DIR / "client.js").read_text(encoding="utf-8")
