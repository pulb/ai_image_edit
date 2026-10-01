# Hugging Face Space

Runs the `qwen_image` backend (diffusers pipeline for Qwen-Image-2.1) in a
Gradio-SDK Space. The app is started as `python app.py` and serves on port
`7860`.

## Space files

Copy these into the **root** of the Space repo:

| From this repo | To the Space |
|---|---|
| `app.py`, `core/`, `frontends/`, `models/` | same paths |
| `deployments/huggingface/requirements.txt` | `requirements.txt` |
| `deployments/huggingface/SPACE_README.md` | `README.md` (the Space card) |

A Space only reads `requirements.txt` and `README.md` from its own root.

The `qwen_image` backend needs the separately licensed
[`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen) package,
which `requirements.txt` installs. Keep the Space private (see the
[License](../../README.md#license) section).

## Backend and frontend

Set these under **Settings → Variables and secrets** in the Space:

| Variable | Value | Default if unset |
|---|---|---|
| `MODEL_BACKEND` | `qwen_image` | `qwen_image_edit_comfy` |
| `FRONTEND` | `gradio` or `nicegui` | `nicegui` |

`MODEL_BACKEND=qwen_image` is required here; the default backend needs
ComfyUI and won't start on a Space. Optional: `HF_TOKEN` (gated weights or a
private AOTI repo), `QWEN21_AOTI` / `QWEN21_AOTI_REPO` (see
`models/qwen_image/model.py`).

## Hardware

### ZeroGPU

- **Settings → Space hardware → ZeroGPU** (needs a PRO account).
- `FRONTEND=gradio` is required. ZeroGPU (`spaces.GPU`) only works with the
  Gradio SDK Space type, which `SPACE_README.md` already declares.

### Paid GPU

- Pick a GPU under **Settings → Space hardware**.
- `FRONTEND` can be `gradio` or `nicegui`. `nicegui` allows arbitrary
  reference file uploads and also runs under the Gradio SDK, because the
  Space just runs `app.py` on port `7860`. This combination is untested.
- To run it as a Docker Space instead, see
  [`../runpod/DEPLOY.md`](../runpod/DEPLOY.md): the same image works, with
  `sdk: docker` in the card and the Dockerfile copied to the Space root as
  `Dockerfile`.
