# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reuses the last raw model output when a generation would repeat the same
inference, so changing only the mask, feather or colour correction
re-composites without running the model again.

Currently single-user only: there is one cache for the whole process. With
several simultaneous users, each one's generation replaces the other's entry
and identical inputs would share a result file, so proper session handling
(one cache per session, passed in by the UI) is needed first.
"""
import hashlib
import json
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from ai_image_edit.core.types import Inference


def image_fingerprint(path: str) -> str:
    """Hash of an image's pixels, so equal images match whatever their file names."""
    with Image.open(path) as img:
        digest = hashlib.sha256()
        digest.update(f"{img.mode}{img.size}".encode())
        digest.update(img.tobytes())
    return digest.hexdigest()


class _LastResult:
    """Holds the most recent (key, output path) pair."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[str] = None
        self._path: Optional[str] = None

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key == self._key and self._path and Path(self._path).exists():
                return self._path
        return None

    def put(self, key: str, path: str) -> None:
        with self._lock:
            self._key, self._path = key, path


_cache = _LastResult()


def cached_infer(
    infer: Callable[[str], str],
    model: str,
    inference: Inference,
    reference_paths: Sequence[str],
    **derived: Any,
) -> Callable[[str], str]:
    """
    Wraps a model's infer callback (model input path -> raw output path).
    The key covers the model, the pixels of the model input image and of the
    reference images, every field of `inference` (so a new one is covered
    automatically; its seed must be the resolved one), and `derived`: values
    the model computed for the request, such as the final width and height.
    The mask and the postprocessing are not part of it: they only affect what
    happens after inference.
    """
    settings = {f.name: getattr(inference, f.name) for f in fields(inference)}
    settings["options"] = dict(inference.options)

    def wrapped(model_input_path: str) -> str:
        key = hashlib.sha256(json.dumps({
            "model": model,
            "input": image_fingerprint(model_input_path),
            "references": [image_fingerprint(p) for p in reference_paths],
            "inference": settings,
            "derived": derived,
        }, sort_keys=True, default=str).encode()).hexdigest()
        cached = _cache.get(key)
        if cached:
            print("[result_cache] reusing the previous generation", flush=True)
            return cached
        output_path = infer(model_input_path)
        _cache.put(key, output_path)
        return output_path

    return wrapped
