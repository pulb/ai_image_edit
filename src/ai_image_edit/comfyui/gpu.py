# SPDX-License-Identifier: GPL-3.0-or-later
"""
Chooses the command line arguments ComfyUI is started with.

If the free GPU memory covers the model's weights plus some headroom for the
generation itself, ComfyUI keeps the weights on the GPU (FAST_ARGS), which saves
10 to 15 s per generation. Otherwise it gets no extra arguments. COMFY_EXTRA_ARGS
replaces the decision: if the variable is set, even to an empty value, its
arguments are used as they are.

ComfyUI is started with CUDA_DEVICE_ORDER=PCI_BUS_ID (unless the variable is
set), so GPU numbers, CUDA_VISIBLE_DEVICES included, are the ones nvidia-smi shows.
"""
import os
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple

# Working memory a generation needs on top of the weights, unless the workflow
# file says otherwise (`vram_headroom_gb`).
DEFAULT_HEADROOM_GB = 6.0
FAST_ARGS = ["--highvram", "--disable-dynamic-vram"]
GIB = 1024 ** 3


def device_order() -> str:
    return os.environ.get("CUDA_DEVICE_ORDER") or "PCI_BUS_ID"


def environment() -> Dict[str, str]:
    """The environment ComfyUI is started with."""
    return {**os.environ, "CUDA_DEVICE_ORDER": device_order()}


def free_vram_bytes() -> Optional[int]:
    """
    Free memory of the first GPU ComfyUI can use (see CUDA_VISIBLE_DEVICES), or
    None if nvidia-smi cannot tell, which includes a CUDA_DEVICE_ORDER whose
    numbering differs from nvidia-smi's.
    """
    if device_order() != "PCI_BUS_ID":
        return None
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        first = visible.split(",")[0].strip()
        if not first or first.startswith("-"):  # no GPU is visible
            return None
        command += ["-i", first]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.splitlines()[0].strip()) * 1024 * 1024  # MiB
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def decide(weights_bytes: int, headroom_gb: float, free_bytes: Optional[int]) -> Tuple[List[str], str]:
    """The extra arguments for the given numbers, and the reason to log."""
    if free_bytes is None:
        return [], "free GPU memory unknown, no extra arguments"
    needed = weights_bytes + headroom_gb * GIB
    summary = f"{free_bytes / GIB:.1f} GB free, {needed / GIB:.1f} GB needed ({weights_bytes / GIB:.1f} GB weights + {headroom_gb:g} GB headroom)"
    if free_bytes >= needed:
        return list(FAST_ARGS), f"{summary}: keeping the weights on the GPU"
    return [], f"{summary}: no extra arguments"


def comfy_extra_args(weights_bytes: int, headroom_gb: Optional[float] = None) -> Tuple[List[str], str]:
    """The extra ComfyUI arguments: COMFY_EXTRA_ARGS if set, otherwise chosen from the free GPU memory."""
    configured = os.environ.get("COMFY_EXTRA_ARGS")
    if configured is not None:
        return shlex.split(configured), "COMFY_EXTRA_ARGS is set"
    return decide(weights_bytes, DEFAULT_HEADROOM_GB if headroom_gb is None else headroom_gb, free_vram_bytes())
