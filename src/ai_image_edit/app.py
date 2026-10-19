# SPDX-License-Identifier: GPL-3.0-or-later
"""
Entry point: builds the model and hands it to the web UI.

    ai-image-edit [--workflow NAME_OR_FILE] [--variant ID] [--no-lora-downloads]
    ai-image-edit --list-workflows
    ai-image-edit --version

Each option falls back to an environment variable, which is how the container
images are configured: MODEL_WORKFLOW (a bundled name or a workflow file;
default qwen_image21), MODEL_VARIANT and DOWNLOAD_LORAS (on unless set to 0).
APP_PASSWORD adds a password login
to the UI. The UI code lives in ui/, the model in workflow_model.py; the built
model is passed straight to the UI's run(model).
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional

from ai_image_edit import ui
from ai_image_edit.comfy import client, downloads, gpu
from ai_image_edit.core import paths
from ai_image_edit.version import VERSION
from ai_image_edit.workflow_model import WorkflowModel, get_model
from ai_image_edit.workflows import available_workflows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai-image-edit", description="AI image editing with a web UI.")
    parser.add_argument("--workflow", metavar="NAME_OR_FILE", help="model workflow to run: a bundled name (see --list-workflows) or the path of a workflow file; env MODEL_WORKFLOW, default qwen_image21")
    parser.add_argument("--variant", help="variant of the workflow's model, e.g. a quantization; env MODEL_VARIANT")
    parser.add_argument(
        "--lora-downloads", action=argparse.BooleanOptionalAction, default=None,
        help="download the workflow's own LoRA files; env DOWNLOAD_LORAS, default on",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--list-workflows", action="store_true", help="print the bundled workflows and exit")
    return parser.parse_args(argv)


_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


def option_flag(value: Optional[bool], env_name: str, default: bool = True) -> bool:
    """A command line switch (None if not given), else the environment variable, else default."""
    if value is not None:
        return value
    raw = os.environ.get(env_name, "").strip().lower()
    if not raw:
        return default
    if raw not in _ON | _OFF:
        raise SystemExit(f"{env_name}={os.environ[env_name]!r} is not a switch: use 1 or 0.")
    return raw in _ON


def describe_config(model: WorkflowModel, workflow: str, variant: Optional[str], lora_downloads: bool = True) -> str:
    """The settings the app starts with, one per line. Secrets are only reported as set or not set."""
    def flag(name: str) -> str:
        return "set" if os.environ.get(name) else "(not set)"

    rows = [
        ("version", VERSION),
        ("workflow", workflow),
        ("variant", variant or "(default)"),
        ("model", model.display_name),
        ("LoRA downloads", "on" if lora_downloads else "off"),
        ("work directory", f"{paths.WORK_DIR} (cap {paths.format_size(paths.WORK_MAX_BYTES)})"),
        ("ComfyUI directory", Path.cwd()),
        ("generation timeout", f"{client.STALL_TIMEOUT_SECONDS} s without progress, {client.MAX_GENERATION_SECONDS} s in all"),
        ("download connections", downloads.connections()),
        ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")),
        ("CUDA_DEVICE_ORDER", gpu.device_order()),
        ("accepted licenses", os.environ.get("ACCEPT_LICENSES", "").strip() or "(none)"),
        ("password login", "on" if os.environ.get("APP_PASSWORD") else "off"),
        ("HF_TOKEN", flag("HF_TOKEN")),
    ]
    return "Configuration:\n" + "\n".join(f"  {name + ':':<22}{value}" for name, value in rows)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.list_workflows:
        print("\n".join(sorted(available_workflows())))
        return

    variant = args.variant or os.environ.get("MODEL_VARIANT") or None
    lora_downloads = option_flag(args.lora_downloads, "DOWNLOAD_LORAS")

    # Checked before the model starts, which can take minutes.
    password = os.environ.get("APP_PASSWORD", "")
    if os.environ.get("REQUIRE_PASSWORD") and not password:
        raise SystemExit("REQUIRE_PASSWORD is set but APP_PASSWORD is empty: set APP_PASSWORD to start the app.")

    workflow = args.workflow or os.environ.get("MODEL_WORKFLOW") or "qwen_image21"
    try:
        model = get_model(workflow, variant, lora_downloads=lora_downloads)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    print(describe_config(model, workflow, variant, lora_downloads), flush=True)
    print(f"Starting model '{workflow}' in the background...", flush=True)
    try:
        model.start()
    except (RuntimeError, OSError) as exc:  # a missing file, an unaccepted license, ComfyUI not launchable: a message is enough
        raise SystemExit(str(exc))

    ui.run(model)
