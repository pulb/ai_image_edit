# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reuses the last raw model output when a generation would repeat the same
inference, so changing only the mask, feather or colour correction
re-composites without running the model again.

Currently single-user only: there is one cache for the whole process. With
several simultaneous users, each one's generation replaces the other's entry
and identical inputs would share a result file, so proper session handling
(one cache per session, passed in by the frontend) is needed first.
"""
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from ai_image_edit.core.types import GenerationParams


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


# GenerationParams fields that never reach the model's inference: the source
# and annotation paths (their pixels are fingerprinted as the model input),
# the mask and what happens after inference, and the raw seed settings (the
# resolved seed is passed in instead).
_NOT_KEYED = frozenset({
    "source_image_path", "annotated_image_path", "mask_path", "reference_images",
    "feather_amount", "apply_color_correction_enabled", "seed", "randomize_seed",
})


def cached_infer(
    infer: Callable[[str], str],
    model: str,
    params: GenerationParams,
    **derived: Any,
) -> Callable[[str], str]:
    """
    Wraps a backend's infer callback (model input path -> raw output path).
    The key covers the model, the pixels of the model input image and of the
    reference images, every other GenerationParams field except _NOT_KEYED
    (so a new parameter is covered automatically; a field the backend does
    not use at worst causes an extra miss), and `derived`: values the
    backend computed from params, such as the resolved seed and the final
    width and height. The mask is not part of the key: it only affects
    compositing, after inference.
    """
    settings = {k: v for k, v in asdict(params).items() if k not in _NOT_KEYED}

    def wrapped(model_input_path: str) -> str:
        key = hashlib.sha256(json.dumps({
            "model": model,
            "input": image_fingerprint(model_input_path),
            "references": [image_fingerprint(p) for p in params.reference_images],
            "settings": settings,
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
