# AI Image Edit

A small, model-agnostic web app for AI-assisted image editing and
generation. It provides a browser UI — prompt, reference images, an
optional inpainting mask, and generation controls (aspect ratio,
resolution, steps, CFG, sampler/scheduler, seed, LoRAs) — in front of a
pluggable image-generation backend.

Two backends currently ship:

- **`qwen_image_edit_comfy`** (default) — the UI driven through a
  [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend (Qwen-Image-Edit
  plus a curated set of LoRAs), for a self-hosted/Docker deployment.
- **`qwen_image`** — a direct [diffusers](https://github.com/huggingface/diffusers)
  pipeline for Qwen-Image-2.1, meant to run in-process on a Hugging Face
  ZeroGPU worker or a GPU pod. It needs the separately licensed
  [`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen)
  package (see [License](#license)).

Two frontends currently ship as well, selectable independently of the
backend:

- **[NiceGUI](https://nicegui.io/)** (default) — supports arbitrary
  reference file uploads.
- **[Gradio](https://www.gradio.app/)** — required if you want to run on
  Hugging Face ZeroGPU, since ZeroGPU (`spaces.GPU`) is tied to the Gradio
  SDK Space type.

## Architecture

```
app.py                 # entry point: builds a model backend, hands it to a frontend
core/                   # shared data contracts (GenerationParams/Result, ModelCapabilities)
                        # and path/masking helpers used by every backend
docker/
  Dockerfile.qwen_image_edit_comfy # ComfyUI backend image
  Dockerfile.qwen_image            # qwen_image + NiceGUI image
frontends/
  nicegui.py            # NiceGUI UI — exposes run(model, model_backend)
  gradio_ui.py          # Gradio UI — exposes run(model, model_backend)
models/
  base.py               # ModelBackend interface every backend implements
  qwen_image/           # backend adapter for the ai-image-edit-qwen package (ZeroGPU)
  qwen_image_edit_comfy/# ComfyUI-driven backend
```

Adding a new model means implementing `ModelBackend` (see
`models/base.py`) and registering one loader function in
`models/__init__.py` — nothing else in the app needs to change. Adding a
new frontend means writing a module that exposes
`run(model: ModelBackend, model_backend: str) -> None:` and registering
it in `frontends/__init__.py`.

## Running it

Select the model backend and frontend via environment variables:

```bash
pip install -r requirements.txt

# Defaults: MODEL_BACKEND=qwen_image_edit_comfy, FRONTEND=nicegui
python app.py

# Or explicitly (qwen_image needs the ai-image-edit-qwen package,
# which requirements.txt installs):
MODEL_BACKEND=qwen_image FRONTEND=gradio python app.py
```

The UI is served on port `7860`.

### Docker (ComfyUI backend)

`docker/Dockerfile.qwen_image_edit_comfy` builds a self-contained image
for the `qwen_image_edit_comfy` backend: it clones ComfyUI, installs
PyTorch, downloads the model checkpoint and a set of LoRAs, and starts the
app with `MODEL_BACKEND=qwen_image_edit_comfy`.

```bash
docker build -f docker/Dockerfile.qwen_image_edit_comfy -t ai-image-edit .
docker run -p 7860:7860 --gpus all ai-image-edit
```

### Hugging Face ZeroGPU Space (diffusers backend)

Deploy with the top-level `requirements.txt` and set `MODEL_BACKEND=qwen_image`
and `FRONTEND=gradio` in the Space's own settings; `MODEL_BACKEND`
otherwise defaults to `qwen_image_edit_comfy`.

### Docker (qwen_image + NiceGUI, RunPod / Hugging Face Docker Space)

`docker/Dockerfile.qwen_image` builds a minimal image for the
`qwen_image` + NiceGUI combination only — no ComfyUI, no gradio; weights
are pulled from the Hugging Face Hub at startup rather than baked into
the image. Suited to a RunPod GPU pod or a Dockerfile-type Hugging Face
Space, both of which expect the app on port `7860`. The image installs
the `ai-image-edit-qwen` package from GitHub, so it combines GPL and
Qwen-licensed code: keep it private (see [License](#license)).

```bash
docker build -f docker/Dockerfile.qwen_image -t ai-image-edit-qwen .
docker run -p 7860:7860 --gpus all ai-image-edit-qwen
```

Pass `--build-arg QWEN_LIB_REF=<branch|tag|commit>` to pin the package
version (default `main`).

Baking the weights into the image would speed up container start, but
Qwen-Image-2.1's weights are tens of GB — that trades a slow first start
for a much larger image to build and push on every change. On a RunPod
pod that persists across restarts, mount a network volume and set
`HF_HUB_CACHE` to a path on it (e.g. `-e HF_HUB_CACHE=/runpod-volume/hf-cache`)
so the download only happens once, without growing the image itself.

### CI: building the RunPod image

`.github/workflows/docker-qwen-image.yml` builds `docker/Dockerfile.qwen_image`
on every push to `main` (when relevant files change) and pushes it to
GHCR as `ghcr.io/<owner>/<repo>-qwen-image`, tagged `latest`, the git tag
(on a `v*` tag push), and the short commit SHA. Each build installs the
newest commit of the `ai-image-edit-qwen` package, resolved when the
build starts. Pushes to the package's own repo don't start a build:
run the workflow manually (or push here) to pick them up. It only builds and
pushes — RunPod has no API to swap a running Pod's image in place, so
rolling out a new build means terminating and recreating the Pod on the
new tag yourself (console or `runpodctl`), once you're ready.

If the GHCR package is private (the default), give the RunPod Pod a
container registry credential (Settings → Container Registry Auth in the
RunPod console) using a GitHub PAT with `read:packages` scope — or make
the package public from its GitHub package settings to skip that step.

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
