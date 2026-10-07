# SPDX-License-Identifier: GPL-3.0-or-later
"""
Registry + factory for model backends.

Each model's module is imported lazily, inside its loader function, rather
than at the top of this file. That matters concretely here:
the ComfyUI backends (qwen_image_edit_2511_aio, qwen_image21_gguf) and
qwen_image have almost disjoint dependency sets (ComfyUI's own stack vs. the
separately installed ai-image-edit-qwen package with torch/diffusers/spaces),
and a given deployment only ever runs one of them. Importing all eagerly at
package load would mean a comfy-only deployment breaks at startup unless it
*also* installs diffusers/spaces/torch, and vice versa — with lazy imports,
each deployment only needs the one model it actually selected to be installed.

Add a new model by writing a class that implements ModelBackend
(models/base.py) and adding one small loader function + one line in
MODEL_LOADERS — nothing else in the app needs to change.
"""
from typing import Callable, Dict

from ai_image_edit.models.base import ModelBackend


def _load_qwen_image_edit_2511_aio() -> ModelBackend:
    from ai_image_edit.models.qwen_image_edit_2511_aio import QwenImageEdit2511AIOModel

    return QwenImageEdit2511AIOModel()


def _load_qwen_image21_gguf() -> ModelBackend:
    from ai_image_edit.models.qwen_image21_gguf import QwenImage21GGUFModel

    return QwenImage21GGUFModel()


def _load_qwen_image() -> ModelBackend:
    try:
        from ai_image_edit.models.qwen_image import QwenImageModel
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "ai_image_edit_qwen":
            raise
        raise RuntimeError(
            "Backend 'qwen_image' needs the ai-image-edit-qwen package: "
            "pip install git+https://github.com/pulb/ai_image_edit_qwen"
        ) from exc

    return QwenImageModel()


MODEL_LOADERS: Dict[str, Callable[[], ModelBackend]] = {
    "qwen_image_edit_2511_aio": _load_qwen_image_edit_2511_aio,
    "qwen_image21_gguf": _load_qwen_image21_gguf,
    "qwen_image": _load_qwen_image,
}


def get_model(name: str) -> ModelBackend:
    try:
        loader = MODEL_LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model backend '{name}'. Available: {', '.join(sorted(MODEL_LOADERS))}"
        )
    return loader()
