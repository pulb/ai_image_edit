# SPDX-License-Identifier: GPL-3.0-or-later
"""
================================================================================
Entry point
================================================================================

Builds the model backend (MODEL_BACKEND, defaulting to
qwen_image_edit_comfy) and picks a UI frontend (FRONTEND, defaulting to
nicegui) — the two choices this app makes, in the one place that makes
both of them.

All the actual UI code lives under frontends/ (frontends/nicegui.py,
frontends/gradio_ui.py); all the actual model code lives under models/.
This file just builds the model once and hands it to whichever frontend
FRONTEND names, via frontends.run_frontend() — see frontends/__init__.py
for how that passes the already-built model straight into the chosen
frontend module's own run(model, model_backend) function.
================================================================================
"""
import os

from frontends import run_frontend
from models import get_model

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "qwen_image_edit_comfy")
model = get_model(MODEL_BACKEND)
print(f"Starting model backend '{MODEL_BACKEND}' in the background...", flush=True)
model.start()

FRONTEND = os.environ.get("FRONTEND", "nicegui")
run_frontend(FRONTEND, model, MODEL_BACKEND)
