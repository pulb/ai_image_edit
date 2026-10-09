# SPDX-License-Identifier: GPL-3.0-or-later
"""
Entry point: builds the model and hands it to the web UI.

    python -m ai_image_edit [--workflow NAME_OR_FILE] [--variant ID]
    python -m ai_image_edit --list-workflows

Each option falls back to an environment variable, which is how the container
images are configured: MODEL_WORKFLOW (a bundled name or a workflow file;
default qwen_image21) and MODEL_VARIANT. APP_PASSWORD adds a password login
to the UI. The UI code lives in ui/, the model in workflow_model.py; the built
model is passed straight to the UI's run(model).
"""
import argparse
import os
from typing import List, Optional

from ai_image_edit import ui
from ai_image_edit.workflow_model import get_model
from ai_image_edit.workflows import available_workflows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai_image_edit", description="AI image editing with a web UI.")
    parser.add_argument("--workflow", metavar="NAME_OR_FILE", help="model workflow to run: a bundled name (see --list-workflows) or the path of a workflow file; env MODEL_WORKFLOW, default qwen_image21")
    parser.add_argument("--variant", help="variant of the workflow's model, e.g. a quantization; env MODEL_VARIANT")
    parser.add_argument("--list-workflows", action="store_true", help="print the bundled workflows and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.list_workflows:
        print("\n".join(sorted(available_workflows())))
        return

    variant = args.variant or os.environ.get("MODEL_VARIANT") or None

    # Checked before the model starts, which can take minutes.
    password = os.environ.get("APP_PASSWORD", "")
    if os.environ.get("REQUIRE_PASSWORD") and not password:
        raise SystemExit("REQUIRE_PASSWORD is set but APP_PASSWORD is empty: set APP_PASSWORD to start the app.")

    workflow = args.workflow or os.environ.get("MODEL_WORKFLOW") or "qwen_image21"
    try:
        model = get_model(workflow, variant)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    print(f"Starting model '{workflow}' in the background...", flush=True)
    try:
        model.start()
    except RuntimeError as exc:  # a missing file, an unaccepted license: a message is enough
        raise SystemExit(str(exc))

    ui.run(model)
