# SPDX-License-Identifier: GPL-3.0-or-later
"""
Registry + factory for UI frontends.

Mirrors models/__init__.py's pattern: each frontend module is imported
lazily, inside its loader function, rather than at the top of this file.
That matters concretely here too — nicegui and gradio have disjoint
dependency sets, and a given deployment only ever runs one of them, so a
gradio-only deployment shouldn't need nicegui installed, and vice versa.

Unlike a model backend, a frontend has no class to instantiate. Instead,
each frontend module (frontends/nicegui.py, frontends/gradio_ui.py)
exposes a single function, run(model), which builds and
launches its UI (ui.run() / demo.queue().launch()) against the given
model. Importing a frontend module does nothing by itself; run_frontend()
below imports the chosen module and calls its run() with the already-built
model, so app.py doesn't need to know any of that — it just
picks a name by FRONTEND, the same way models/__init__.py's get_model()
picks one by MODEL_WORKFLOW.

Model selection also happens once, in app.py's main(), rather than each frontend
module building its own model — run_frontend() takes the already-built
model and passes it straight to the chosen module's run() as a plain
function argument, so both frontends stay ordinary, explicit Python:
model is just a parameter of run(), not a value pulled from somewhere else
at import time.

Add a new frontend by writing its module in this package (with its own
run(model) function) and adding one loader function + one
line in FRONTEND_LOADERS — nothing else needs to change.
"""
from typing import Callable, Dict

from ai_image_edit.models.base import ModelBackend


def _load_nicegui(model: ModelBackend) -> None:
    import ai_image_edit.frontends.nicegui as _frontend

    _frontend.run(model)


def _load_gradio_ui(model: ModelBackend) -> None:
    import ai_image_edit.frontends.gradio_ui as _frontend

    _frontend.run(model)


FRONTEND_LOADERS: Dict[str, Callable[[ModelBackend], None]] = {
    "nicegui": _load_nicegui,
    "gradio": _load_gradio_ui,
}


def run_frontend(name: str, model: ModelBackend) -> None:
    try:
        loader = FRONTEND_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown frontend '{name}'. Available: {', '.join(sorted(FRONTEND_LOADERS))}"
        )
    loader(model)
