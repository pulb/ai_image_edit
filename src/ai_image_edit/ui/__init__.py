# SPDX-License-Identifier: GPL-3.0-or-later
"""
The web UI (NiceGUI, in gui.py).

run(model) builds and launches it against the given model; run_error_page()
serves the page shown instead when the app cannot start. gui.py is imported
only when run() is called, so importing this package stays cheap and the tests
do not need NiceGUI installed.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_image_edit.workflow_model import WorkflowModel


def run(model: "WorkflowModel") -> None:
    from ai_image_edit.ui import gui

    gui.run(model)


def run_error_page(area: str, detail: str = "") -> None:
    """Serves the page saying that the app could not start, until the app is stopped."""
    from ai_image_edit.ui import error_page

    error_page.run(area, detail)
