# AI Image Edit

A comfortable, responsive, model-agnostic app for AI-assisted image editing
and generation. It provides a web based UI optimized for mobile devices in
front of interchangeable image-generation models.

<p align="center">
  <br>
  <img src="doc/annotation-workflow.webp" alt="Annotation workflow: type a prompt, draw colored circles, generate" width="300">
  <br>
  <br>
</p>

## Features

| Feature | Description |
|---|---|
| Privacy | Runs on your own hardware or in a disposable container, so your images and prompts stay under your control and vanish with it |
| Inpainting | Advanced inpainting and mask functionality with seamless soft blending |
| Annotations | Draw thin colored strokes directly on the image to point the model at what to change |
| Incremental edits | Feed any result back in as the next input to refine an image step by step |
| Before/after comparison | Interactive slider to compare generated outputs with the source |
| Aspect ratios | Presets for common, widely used aspect ratios |
| LoRAs | Optional LoRA support, auto-detected from a local loras folder |

## Model workflows

Each model is a [workflow file](doc/contribution/workflow_files.md): a ComfyUI
workflow plus a manifest that lists its weights, custom nodes and variants. They
run in a ComfyUI process that the app starts, and the weights are downloaded
when the model starts. Select one with `--model NAME` or the `MODEL_WORKFLOW`
variable (`--list-models` prints the names). `--workflow FILE` (or
`WORKFLOW_FILE`) runs any workflow file of your own instead, and `--variant ID`
(`MODEL_VARIANT`) picks one of its variants, e.g. another GGUF quantization.
The [deployments](#deployment) are already preconfigured.

- **`qwen_image21`** (default) — Qwen-Image-2.1 with the
  [official ComfyUI weights](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
  (bf16, or `--variant int8`). Meant for GPUs with plenty of memory. The
  weights are under the [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE).
- **`qwen_image21_gguf`** — Qwen-Image-2.1 with quantized
  [GGUF weights](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF).
  Meant for low-VRAM GPUs: it needs far less GPU memory than `qwen_image21`, at
  some cost in quality. The weights are under the
  [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE).
- **`qwen_image_edit_2511_aio`** — Qwen-Image-Edit 2511
  ([Phr00t's AIO merge](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO))
  plus a curated set of LoRAs.

The UI is served on port `7860`.

## Deployment

| Target | Model workflows | Guide |
|---|---|---|
| ⭐ **RunPod GPU Pod** (recommended) | all (one ComfyUI image) | [`deployments/runpod`](deployments/runpod/DEPLOY.md) |
| Hugging Face Space | the ComfyUI image as a Docker Space on a paid GPU | [`deployments/huggingface`](deployments/huggingface/DEPLOY.md) |
| Docker (self-hosted) | all (one ComfyUI image) | [`deployments/docker`](deployments/docker/DEPLOY.md) |

### Generation times

Approximate generation times for the `qwen_image21` and `qwen_image21_gguf`
backends (Qwen-Image-2.1) on common GPUs, by output resolution in megapixels
(MP), at the default of 40 steps. The GGUF figures are for the default
`UC Q4_K_M` variant; the RTX 4090 figures were measured with
`--highvram --disable-dynamic-vram` (see [Docker](deployments/docker/DEPLOY.md#comfyui-arguments-comfy_extra_args)),
the RTX 2000 Ada figures without. They are rough figures from single runs, not benchmarks,
and will vary with the step count and reference images. A dash means not
measured. The `qwen_image21` figures were measured with the former diffusers
backend; the ComfyUI workflow that replaced it performed on par in our tests on
an A100, but it was not re-measured on the other GPUs.

| GPU | `qwen_image21` 1 MP | `qwen_image21` 2 MP | `qwen_image21_gguf` 1 MP | `qwen_image21_gguf` 2 MP |
|---|---|---|---|---|
| NVIDIA H200 | ~10 s | ~1 min | – | – |
| NVIDIA A100 | ~20 s | ~2 min | – | – |
| NVIDIA L40S | ~23 s | insufficient mem | – | – |
| NVIDIA RTX 4090 (24 GB) | insufficient mem | insufficient mem | ~31 s | ~2 min 40 s |
| NVIDIA RTX 2000 Ada (16 GB) | insufficient mem | insufficient mem | ~2 min 10 s | ~12 min |

## License

The code in this repository is licensed under the GNU General Public License
v3.0 or later (GPL-3.0-or-later). See [`LICENSE`](LICENSE) for the full text.

The third-party model and LoRA weights are not part of this repository and are
not under the GPL. They are downloaded from their publishers when a model
starts, and each has its own license. Check the licenses of the linked
repositories before use.

A model whose files carry a license that must be accepted is not downloaded
until you accept it with `ACCEPT_LICENSES`; the error names the license and its
URL. Custom nodes installed for a workflow keep their own licenses.

Copyright (C) 2026 AI Image Edit authors

## Examples

<a href="doc/screenshots/inpainting.png"><img src="doc/screenshots/inpainting.png" alt="Replacing an animal in a masked photo, using a reference image" width="22%" align="top"></a>
<a href="doc/screenshots/annotations.png"><img src="doc/screenshots/annotations.png" alt="Placing things in a photo with colored annotations" width="22%" align="top"></a>
<a href="doc/screenshots/sketch_to_figure.png"><img src="doc/screenshots/sketch_to_figure.png" alt="Turning a drawing into an action figure" width="22%" align="top"></a>
<a href="doc/screenshots/perspective.png"><img src="doc/screenshots/perspective.png" alt="Changing the perspective of a photo" width="22%" align="top"></a>
