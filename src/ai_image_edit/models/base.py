# SPDX-License-Identifier: GPL-3.0-or-later
"""
The interface every model backend implements.

Kept intentionally small: the UI needs exactly three things from a model —
what it can do (capabilities), how to start/stop it, and how to run one
generation — plus a readable name and version to show. How a model actually gets there (ComfyUI over a websocket, an
in-process torch pipeline, or anything else) is the model's own
business and never leaks into this contract — that's what lets a completely
different execution model (e.g. qwen_image's direct torch pipeline, with no
ComfyUI involved at all) implement the same interface as
qwen_image_edit_comfy without either one needing to know about the other.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities


class ModelBackend(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Static description of what this model supports. Read once by the UI at startup."""

    @property
    def model_name(self) -> str:
        """Human-readable model name shown in the UI. Defaults to the class name."""
        return type(self).__name__

    @property
    def model_version(self) -> Optional[str]:
        """Model version as it should be displayed (e.g. "2.1", "v23"), or None if there is none."""
        return None

    @property
    def display_name(self) -> str:
        """model_name followed by model_version, if any. Not meant to be overridden."""
        return f"{self.model_name} {self.model_version}" if self.model_version else self.model_name

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

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        """
        The supported_megapixels value that fits the given image, or None if
        the model has no such notion. Used while a mask is drawn, where the
        output keeps the image's own resolution instead of the selected one.
        """
        return None

    @abstractmethod
    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Runs one full generation: inference, plus — internally, whenever
        params.mask_path is set — whatever inpaint/crop/color-merge this
        model needs to produce its final output. Blocking; the caller is
        expected to run this off the UI's event loop (e.g. via
        nicegui.run.io_bound).
        """
