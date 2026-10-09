# Model workflows

A model in AI Image Edit is a *model workflow*: one JSON file that holds a
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflow and a manifest
describing everything the app needs to run it (weights to download, custom
nodes to install, variants, which workflow inputs the prompt, seed and images go
to). The app starts a ComfyUI process, loads the workflow into it and downloads
what is missing when the model starts. Adding or changing a model needs no
Python; see [workflow files](contribution/workflow_files.md) for the format and
[adding a model](contribution/extending.md).

## Choosing a workflow

| Setting | Meaning |
|---|---|
| `--workflow NAME_OR_FILE` / `MODEL_WORKFLOW` | The bundled workflow to run (default `qwen_image21`), or the path of a workflow file of your own. `--list-workflows` prints the bundled names. |
| `--variant ID` / `MODEL_VARIANT` | One of the workflow's variants, e.g. another quantization. |
| `ACCEPT_LICENSES` | Licenses you accept, comma-separated (`all` for every one). Files with a license are not downloaded without it. |

The deployment guides ([Docker](../deployments/docker/DEPLOY.md),
[RunPod](../deployments/runpod/DEPLOY.md),
[Hugging Face](../deployments/huggingface/DEPLOY.md)) list all other settings,
such as the ComfyUI arguments for each GPU size and the download options.

## Bundled workflows

### `qwen_image21` (default)

Qwen-Image-2.1 with the
[official ComfyUI weights](https://huggingface.co/Comfy-Org/Qwen-Image-2.1):
the bf16 diffusion model, text encoder and VAE.

- **Use it for:** the best quality, on GPUs with plenty of memory. It was tested
  on an A100; the 48 GB class is the practical minimum for bf16.
- **Variants:** `int8`, which uses less memory.
- **Steps:** 8 to 60, 40 by default. The official pipeline uses 40 to 50 steps
  with the euler sampler; ComfyUI's own template defaults to 25.
- **License:** the weights are under the
  [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE)
  (accept it with `ACCEPT_LICENSES=qwen-research`).

### `qwen_image21_gguf`

Qwen-Image-2.1 as quantized
[GGUF weights](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF),
loaded with the [ComfyUI-GGUF](https://github.com/pulb/ComfyUI-GGUF) custom node
(installed automatically at a pinned commit).

- **Use it for:** GPUs with little memory. A 16 GB GPU is enough (tested); the
  download is about 15 GB. Quality is somewhat below the bf16 weights.
- **Variants:** the quantization, from `Q4_0` over `Q4_K_M` (default), `Q5_K_M`
  and `Q6_K` to `Q8_0` and `BF16`. Larger is better and needs more memory. These
  are the third-party *uncensored* version of the model; the `standard-Q4_0` to
  `standard-Q8_0` variants are the unmodified model.
- **Steps:** 8 to 60, 40 by default.
- **License:** the GGUF files are third-party conversions of the Qwen-Image-2.1
  weights and are under the same
  [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE)
  (`ACCEPT_LICENSES=qwen-research`).

### `qwen_image_edit_2511_aio`

Qwen-Image-Edit 2511 in
[Phr00t's Rapid-AIO merge](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO),
combined with a curated set of LoRAs (camera angle changes, style transfer,
photo-realism, polaroid look, unblurring and upscaling, comic and noir styles,
pose transfer). The LoRAs are downloaded with the model and can be applied from
the UI.

- **Use it for:** fast edits: it needs only 1 to 10 steps (4 by default) and
  supports a LoRA strength setting.
- **License:** the weights and LoRAs come from third parties; check the licenses
  of the linked repositories before use.

## Downloads and custom nodes

When a model starts, the app lists the weight files, LoRAs and custom nodes its
workflow needs and fetches only what is missing, with parallel ranged downloads,
resume support and a checksum check. Until the last file is there, generating
answers that the model is still being set up and shows the progress. Mount a
volume at the models folder (see the Docker guide) so the weights are downloaded
only once.

Custom nodes are installed from git at an exact, pinned commit, so a workflow
keeps working when the node's repository changes. Their requirements are
installed with pip.

## Versions

The version shown in the UI is the manifest's `model_version`, or the variant's
if it has one. Set `MODEL_VERSION` to show another text.
