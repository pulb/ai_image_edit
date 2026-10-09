# SPDX-License-Identifier: GPL-3.0-or-later
"""
Finding and building models.

Every model is a workflow file (see doc/contribution/workflow_files.md). The
ones in ai_image_edit/data/workflows/ ship with the app and are picked up
automatically, under the file's name; any other file can be run by its path.
"""
from pathlib import Path
from typing import Optional

from ai_image_edit.models.base import Model
from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows


def get_model(workflow: str, variant: Optional[str] = None) -> Model:
    """The model of a bundled workflow (by name) or of a workflow file (by path).

    `variant` selects one of the workflow's variants.
    """
    bundled = available_workflows()
    if workflow in bundled:
        path = bundled[workflow]
    elif Path(workflow).is_file():
        path = Path(workflow)
    else:
        raise ValueError(
            f"Unknown workflow '{workflow}'. Available: {', '.join(sorted(bundled))}; "
            "or give the path of a workflow file."
        )
    from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

    return ComfyWorkflowModel.from_file(path, variant)
