# SPDX-License-Identifier: GPL-3.0-or-later
"""
The web UI.

frontends/nicegui.py exposes a single function, run(model), which builds and
launches the UI (ui.run()) against the given model. Importing it does nothing
by itself; run_frontend() below imports it lazily and calls run() with the
already-built model, so importing this package stays cheap (and the tests don't
need NiceGUI installed).

Model selection happens once, in app.py's main(): the built model is passed
straight to run() as a plain function argument, not pulled from somewhere else
at import time.
"""
from ai_image_edit.models.base import Model


def run_frontend(model: Model) -> None:
    import ai_image_edit.frontends.nicegui as _frontend

    _frontend.run(model)
