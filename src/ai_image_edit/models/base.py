# SPDX-License-Identifier: GPL-3.0-or-later
"""
The interface every model implements.

Kept intentionally small: the UI needs exactly three things from a model —
what it can do (capabilities), how to start/stop it, and how to run one
generation — plus a readable name and version to show. How a model actually gets there
(today: a ComfyUI process driven over a websocket) is the model's own business
and never leaks into this contract.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities


@dataclass(frozen=True)
class SetupProgress:
    """How far the model's background setup (downloads, installs) is."""

    done: bool = True
    message: str = ""  # what is being done right now, e.g. "Downloading x.safetensors (2/5): 42%"
    error: Optional[str] = None


class Model(ABC):
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
        subprocess, load weights onto the GPU —
        whatever this particular model needs. Called once, at app startup.
        """

    def shutdown(self) -> None:
        """Optional cleanup (e.g. terminate a subprocess, free GPU memory). No-op by default."""

    def setup_progress(self) -> SetupProgress:
        """Progress of the setup that start() continues in the background. Done by default."""
        return SetupProgress()

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
