# AI Image Edit

A small, model-agnostic web app for AI-assisted image editing and
generation. It provides a browser UI — prompt, reference images, an
optional inpainting mask, and generation controls (aspect ratio,
resolution, steps, CFG, sampler/scheduler, seed, LoRAs) — in front of a
pluggable image-generation backend.

Backends, selected with `MODEL_BACKEND`:

- **`qwen_image_edit_comfy`** (default) — Qwen-Image-Edit plus a curated set
  of LoRAs, driven through [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- **`qwen_image`** — a direct [diffusers](https://github.com/huggingface/diffusers)
  pipeline for Qwen-Image-2.1. Needs the separately licensed
  [`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen)
  package (see [License](#license)).

Frontends, selected with `FRONTEND`:

- **[NiceGUI](https://nicegui.io/)** (default) — supports arbitrary
  reference file uploads.
- **[Gradio](https://www.gradio.app/)** — required for Hugging Face ZeroGPU.

The UI is served on port `7860`.

## Deployment

| Target | Backend | Guide |
|---|---|---|
| Docker (self-hosted) | `qwen_image_edit_comfy` | [`deployments/docker`](deployments/docker/DEPLOY.md) |
| Hugging Face Space (ZeroGPU or paid GPU) | `qwen_image` | [`deployments/huggingface`](deployments/huggingface/DEPLOY.md) |
| RunPod GPU Pod | `qwen_image` | [`deployments/runpod`](deployments/runpod/DEPLOY.md) |

## Layout

```
app.py                  # entry point: builds a model backend, hands it to a frontend
core/                   # shared data contracts and path/masking helpers
frontends/              # nicegui.py, gradio_ui.py — each exposes run(model, model_backend)
models/
  base.py               # ModelBackend interface every backend implements
  qwen_image/           # adapter for the ai-image-edit-qwen package
  qwen_image_edit_comfy/# ComfyUI-driven backend
deployments/            # docker/, huggingface/, runpod/ — Dockerfiles, requirements, guides
```

Adding a model means implementing `ModelBackend` (see `models/base.py`) and
registering one loader in `models/__init__.py`. Adding a frontend means
writing a module that exposes `run(model, model_backend)` and registering it
in `frontends/__init__.py`.

## License

Licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See [`LICENSE`](LICENSE) for the full text.

The `qwen_image` backend depends on a separate package,
[`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen), which
is under the Qwen RESEARCH LICENSE AGREEMENT rather than the GPL, as are
the Qwen-Image-2.1 weights it loads. That agreement allows non-commercial
use (research or evaluation) only. This repository contains no code under
that license.

Copyright (C) 2026 AI Image Edit authors
