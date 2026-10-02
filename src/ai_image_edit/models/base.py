# SPDX-License-Identifier: GPL-3.0-or-later
"""
The interface every model backend implements.

Kept intentionally small: the UI needs exactly three things from a model —
what it can do (capabilities), how to start/stop it, and how to run one
generation. How a model actually gets there (ComfyUI over a websocket, an
in-process torch pipeline, or anything else) is the model's own
business and never leaks into this contract — that's what lets a completely
different execution model (e.g. qwen_image's direct torch pipeline, with no
ComfyUI involved at all) implement the same interface as
qwen_image_edit_comfy without either one needing to know about the other.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities


class ModelBackend(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Static description of what this model supports. Read once by the UI at startup."""

    @abstractmethod
    def start(self) -> None:
        """
        One-time setup before this model can serve requests: launch a
        subprocess, load weights onto the GPU, warm up a pipeline —
        whatever this particular model needs. Called once, at app startup.
        """

    def shutdown(self) -> None:
        """Optional cleanup (e.g. terminate a subprocess, free GPU memory). No-op by default."""

    def list_loras(self) -> Dict[str, List[str]]:
        """
        Available LoRAs, keyed by display name, each mapped to the weight
        file(s) that belong to it. Only override this if
        capabilities.supports_loras is True — the default empty dict is
        correct for any model that doesn't support LoRAs, and the UI never
        calls this when supports_loras is False.
        """
        return {}

    @abstractmethod
    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Runs one full generation: inference, plus — internally, whenever
        params.mask_path is set — whatever inpaint/crop/color-merge this
        model needs to produce its final output. Blocking; the caller is
        expected to run this off the UI's event loop (e.g. via
        nicegui.run.io_bound).
        """
