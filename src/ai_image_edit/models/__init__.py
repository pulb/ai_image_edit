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

Add a ComfyUI model by creating a folder here with a workflow_api.json and a
manifest.json (see models/_shared/comfyui/MANIFEST.md) — it is picked up
automatically. Add any other kind of model by writing a class that implements
ModelBackend (models/base.py) and adding one small loader function + one line
in MODEL_LOADERS. Nothing else in the app needs to change.
"""
from pathlib import Path
from typing import Callable, Dict

from ai_image_edit.models.base import ModelBackend


def _manifest_loader(manifest_path: Path) -> Callable[[], ModelBackend]:
    def load() -> ModelBackend:
        from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

        return ComfyWorkflowModel.from_manifest(manifest_path)

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
# Every folder with a manifest.json is a ComfyUI model, named after its folder.
for _manifest in sorted(Path(__file__).parent.glob("*/manifest.json")):
    MODEL_LOADERS[_manifest.parent.name] = _manifest_loader(_manifest)


def get_model(name: str) -> ModelBackend:
    try:
        loader = MODEL_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model backend '{name}'. Available: {', '.join(sorted(MODEL_LOADERS))}"
        )
    return loader()
