# SPDX-License-Identifier: GPL-3.0-or-later
"""
Entry point: builds the model and hands it to the web UI.

    python -m ai_image_edit [--model NAME | --workflow FILE] [--variant ID]
    python -m ai_image_edit --list-models

Each option falls back to an environment variable, which is how the container
images are configured: MODEL_WORKFLOW (default qwen_image21), WORKFLOW_FILE,
and MODEL_VARIANT. APP_PASSWORD adds a password login
to the UI. The UI code lives under frontends/, the model code under
models/; run_frontend() passes the built model straight to the UI's run(model).
"""
import argparse
import os
from typing import List, Optional

from ai_image_edit.frontends import run_frontend
from ai_image_edit.models import MODEL_LOADERS, get_model, get_model_from_file


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai_image_edit", description="AI image editing with a web UI.")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--model", help="model to run (see --list-models); env MODEL_WORKFLOW, default qwen_image21")
    which.add_argument("--workflow", metavar="FILE", help="run the ComfyUI model in this workflow file; env WORKFLOW_FILE")
    parser.add_argument("--variant", help="variant of a workflow file's model, e.g. a quantization; env MODEL_VARIANT")
    parser.add_argument("--list-models", action="store_true", help="print the available models and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.list_models:
        print("\n".join(sorted(MODEL_LOADERS)))
        return

    variant = args.variant or os.environ.get("MODEL_VARIANT") or None

    # Checked before the model starts, which can take minutes.
    password = os.environ.get("APP_PASSWORD", "")
    if os.environ.get("REQUIRE_PASSWORD") and not password:
        raise SystemExit("REQUIRE_PASSWORD is set but APP_PASSWORD is empty: set APP_PASSWORD to start the app.")

    # An option on the command line beats the environment, whichever of the two it is.
    workflow = args.workflow or (None if args.model else os.environ.get("WORKFLOW_FILE") or None)
    try:
        if workflow:
            model_name = workflow
            model = get_model_from_file(workflow, variant)
        else:
            model_name = args.model or os.environ.get("MODEL_WORKFLOW", "qwen_image21")
            model = get_model(model_name, variant)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    print(f"Starting model '{model_name}' in the background...", flush=True)
    try:
        model.start()
    except RuntimeError as exc:  # a missing file, an unaccepted license: a message is enough
        raise SystemExit(str(exc))

    run_frontend(model)
