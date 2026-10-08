# SPDX-License-Identifier: GPL-3.0-or-later
"""
Registry + factory for model backends.

Each model's module is imported lazily, inside its loader function, rather
than at the top of this file. That matters concretely here:
the ComfyUI models (qwen_image_edit_2511_aio, qwen_image21_gguf) and
qwen_image21 have almost disjoint dependency sets (ComfyUI's own stack vs. the
separately installed ai-image-edit-qwen package with torch/diffusers/spaces),
and a given deployment only ever runs one of them. Importing all eagerly at
package load would mean a comfy-only deployment breaks at startup unless it
*also* installs diffusers/spaces/torch, and vice versa — with lazy imports,
each deployment only needs the one model it actually selected to be installed.

Add a ComfyUI model by adding a workflow file to ai_image_edit/data/workflows/ (see
doc/contribution/workflow_files.md) — it is picked up automatically, under the
file's name. Add any other kind of model by writing a class that implements
ModelBackend (models/base.py) and adding one small loader function + one line
in MODEL_LOADERS. Nothing else in the app needs to change.
"""
from pathlib import Path
from typing import Callable, Dict

from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows


def _workflow_file_loader(path: Path) -> Callable[[], ModelBackend]:
    def load() -> ModelBackend:
        from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

        return ComfyWorkflowModel.from_file(path)

    return load


def _load_qwen_image21() -> ModelBackend:
    try:
        from ai_image_edit.models.qwen_image21 import QwenImage21Model
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "ai_image_edit_qwen":
            raise
        raise RuntimeError(
            "Backend 'qwen_image21' needs the ai-image-edit-qwen package: "
            "pip install git+https://github.com/pulb/ai_image_edit_qwen"
        ) from exc

    return QwenImage21Model()


MODEL_LOADERS: Dict[str, Callable[[], ModelBackend]] = {
    "qwen_image21": _load_qwen_image21,
}
# Every workflow file is a ComfyUI model, named after the file.
for _name, _path in available_workflows().items():
    MODEL_LOADERS[_name] = _workflow_file_loader(_path)


def get_model(name: str) -> ModelBackend:
    try:
        loader = MODEL_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model backend '{name}'. Available: {', '.join(sorted(MODEL_LOADERS))}. "
            "ComfyUI models are the files in ai_image_edit/data/workflows/."
        )
    return loader()
