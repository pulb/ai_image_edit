# SPDX-License-Identifier: GPL-3.0-or-later
"""
The web UI (NiceGUI, in gui.py).

run(model) builds and launches it against the given model. gui.py is imported
only when run() is called, so importing this package stays cheap and the tests
do not need NiceGUI installed.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_image_edit.workflow_model import WorkflowModel


def run(model: "WorkflowModel") -> None:
    from ai_image_edit.ui import gui

    gui.run(model)
