# AI Image Edit

A comfortable, responsive, model-agnostic app for AI-assisted image editing
and generation. It provides a web based UI optimized for mobile devices in
front of a pluggable image-generation backend.

<p align="center">
  <img src="doc/annotation-workflow.webp" alt="Annotation workflow: type a prompt, draw colored circles, generate" width="480">
</p>

| Feature | Description |
|---|---|
| Privacy | Runs on your own hardware or in a disposable container, so your images and prompts stay under your control and vanish with it |
| Inpainting | Advanced inpainting and mask functionality with seamless soft blending |
| Annotations | Draw thin colored strokes directly on the image to point the model at what to change |
| Incremental edits | Feed any result back in as the next input to refine an image step by step |
| Before/after comparison | Interactive slider to compare generated outputs with the source |
| Aspect ratios | Presets for common, widely used aspect ratios |
| LoRAs | Optional LoRA support, auto-detected from a local loras folder |

Backends, selected with `MODEL_BACKEND`:

- **`qwen_image_edit_comfy`** (default) — Qwen-Image-Edit
  ([Phr00t's AIO merge](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO))
  plus a curated set of LoRAs, driven through [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- **`qwen_image`** — a direct [diffusers](https://github.com/huggingface/diffusers)
  pipeline for Qwen-Image-2.1. Needs the separately licensed
  [`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen)
  package (see [License](#license)).

Frontends, selected with `FRONTEND`:

- **[NiceGUI](https://nicegui.io/)** (default) — the snappier frontend.
- **[Gradio](https://www.gradio.app/)** — fallback frontend, required for
  Hugging Face ZeroGPU.

The UI is served on port `7860`.

## Deployment

| Target | Backend | Frontend | Guide |
|---|---|---|---|
| ⭐ **RunPod GPU Pod** (recommended) | `qwen_image_edit_comfy`, `qwen_image` (one image each) | NiceGUI only | [`deployments/runpod`](deployments/runpod/DEPLOY.md) |
| Hugging Face Space | `qwen_image` | Gradio (ZeroGPU), Gradio or NiceGUI (paid GPU) | [`deployments/huggingface`](deployments/huggingface/DEPLOY.md) |
| Docker (self-hosted) | `qwen_image_edit_comfy`, `qwen_image` (one image each) | NiceGUI only | [`deployments/docker`](deployments/docker/DEPLOY.md) |

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

## Examples

<a href="doc/screenshots/inpainting.png"><img src="doc/screenshots/inpainting.png" alt="Replacing an animal in a masked photo, using a reference image" width="22%" align="top"></a>
<a href="doc/screenshots/annotations.png"><img src="doc/screenshots/annotations.png" alt="Placing things in a photo with colored annotations" width="22%" align="top"></a>
<a href="doc/screenshots/sketch_to_figure.png"><img src="doc/screenshots/sketch_to_figure.png" alt="Turning a drawing into an action figure" width="22%" align="top"></a>
<a href="doc/screenshots/perspective.png"><img src="doc/screenshots/perspective.png" alt="Changing the perspective of a photo" width="22%" align="top"></a>
