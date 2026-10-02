# SPDX-License-Identifier: GPL-3.0-or-later
"""
Entry point: builds the model backend and hands it to a UI frontend.

MODEL_BACKEND (default qwen_image_edit_comfy) and FRONTEND (default
nicegui) are read when main() is called, not at import time, so a launcher
can set them first. APP_PASSWORD adds a password login to either frontend.
The UI code lives under frontends/, the model code under models/;
run_frontend() passes the built model straight to the chosen
frontend's run(model, model_backend).
"""
import os

from ai_image_edit.frontends import run_frontend
from ai_image_edit.models import get_model


def main() -> None:
    frontend = os.environ.get("FRONTEND", "nicegui")

    # Checked before the model starts, which can take minutes.
    password = os.environ.get("APP_PASSWORD", "")
    if os.environ.get("REQUIRE_PASSWORD") and not password:
        raise SystemExit("REQUIRE_PASSWORD is set but APP_PASSWORD is empty: set APP_PASSWORD to start the app.")

    model_backend = os.environ.get("MODEL_BACKEND", "qwen_image_edit_comfy")
    model = get_model(model_backend)
    print(f"Starting model backend '{model_backend}' in the background...", flush=True)
    model.start()

    run_frontend(frontend, model, model_backend)
