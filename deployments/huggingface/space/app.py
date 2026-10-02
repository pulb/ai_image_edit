# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point for a Hugging Face Space (copy to the Space root as app.py).

The defaults below are overridden by MODEL_BACKEND / FRONTEND variables set
in the Space settings. ZeroGPU needs the Gradio frontend; on a paid GPU
change "gradio" to "nicegui" to use NiceGUI.
"""
import os

os.environ.setdefault("MODEL_BACKEND", "qwen_image")
os.environ.setdefault("FRONTEND", "gradio")

from ai_image_edit.app import main

main()
