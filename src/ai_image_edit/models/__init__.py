# SPDX-License-Identifier: GPL-3.0-or-later
"""
Registry + factory for model backends.

Every ComfyUI model is a workflow file in ai_image_edit/data/workflows/ (see
doc/contribution/workflow_files.md); it is picked up automatically, under the
file's name. Any other kind of model is a class that implements ModelBackend
(models/base.py), added with one small loader function and one line in
MODEL_LOADERS. Import the model's module inside its loader function, not at the
top of this file: backends can have almost disjoint dependency sets, and a
deployment should only need the ones of the model it runs. Nothing else in the
app needs to change.
"""
from pathlib import Path
from typing import Callable, Dict, Optional

from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows


def _workflow_file_loader(path: Path) -> Callable[..., ModelBackend]:
    def load(variant: Optional[str] = None) -> ModelBackend:
        from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

        return ComfyWorkflowModel.from_file(path, variant)

    return load


# Models that are not workflow files: name -> loader.
MODEL_LOADERS: Dict[str, Callable[..., ModelBackend]] = {}
# Every workflow file is a ComfyUI model, named after the file.
for _name, _path in available_workflows().items():
    MODEL_LOADERS[_name] = _workflow_file_loader(_path)


def get_model(name: str, variant: Optional[str] = None) -> ModelBackend:
    """The backend registered as `name`; `variant` selects one of a workflow file's variants."""
    try:
        loader = MODEL_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model backend '{name}'. Available: {', '.join(sorted(MODEL_LOADERS))}. "
            "ComfyUI models are the files in ai_image_edit/data/workflows/."
        )
    if variant is None:
        return loader()
    if name not in available_workflows():
        raise ValueError(f"Model '{name}' has no variants.")
    return loader(variant)


def get_model_from_file(path: Path, variant: Optional[str] = None) -> ModelBackend:
    """A ComfyUI model from any workflow file (see doc/contribution/workflow_files.md)."""
    return _workflow_file_loader(Path(path))(variant)
