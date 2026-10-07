# Docker

Three images, all built from the repo root. The app serves on port `7860`.

| Dockerfile | Backend | Frontend | Use |
|---|---|---|---|
| `Dockerfile.qwen_image_edit_2511_aio` | `qwen_image_edit_2511_aio` | `nicegui` | self-hosted ComfyUI backend |
| `Dockerfile.qwen_image21_gguf` | `qwen_image21_gguf` | `nicegui` | quantized Qwen-Image-2.1 for consumer GPUs |
| `Dockerfile.qwen_image` | `qwen_image` | `nicegui` | see [RunPod](../runpod/DEPLOY.md) |

## Image storage

Uploads and results are kept in RAM, in a folder under `/dev/shm`, so they
never reach the container's disk, and they disappear when the container stops.
The app deletes the oldest files once the folder exceeds
`AI_IMAGE_EDIT_WORK_MAX_SIZE` (default `512m`, set in the Dockerfile; `0` means as
much as the filesystem allows).

The actual cap may be lower than that setting. It is limited to 90% of the
size of `/dev/shm`, and Docker gives `/dev/shm` only 64 MB by default: pass
`--shm-size=640m` (or more) to `docker run`. 640 MB is the smallest round
size that fits the default 512 MB cap: 512 MB needs a `/dev/shm` of at least
about 570 MB (512 / 0.9), and 640 MB leaves some headroom. If you raise
`AI_IMAGE_EDIT_WORK_MAX_SIZE`, raise `--shm-size` to at least that value
divided by 0.9. If the setting had to be lowered,
the app prints a warning at startup. `/dev/shm` is shared with anything else
in the container that uses it. `AI_IMAGE_EDIT_WORK_DIR` moves the folder
elsewhere.

RAM disks keep images off the disk only: the host can still read memory, and
the traffic to the app is not covered. Memory used here counts against the
container's RAM limit.

## Password login

The app sits behind a password login when `APP_PASSWORD` is
set (for example `-e APP_PASSWORD=...`). All images set
`REQUIRE_PASSWORD=1` and refuse to start without it. Details are in
[`../runpod/DEPLOY.md`](../runpod/DEPLOY.md).

## ComfyUI image

Clones ComfyUI, installs PyTorch, downloads the model checkpoint and a set of
LoRAs, then starts the app. The checkpoint is set by two variables near the top
of the Dockerfile: `MODEL_FILE_URL` (where it is downloaded from) and
`MODEL_FILE` (its path under ComfyUI's `models/checkpoints` folder).
The app loads the file `MODEL_FILE` names and refuses to start if it
is empty or the file does not exist.

```bash
docker build -f deployments/docker/Dockerfile.qwen_image_edit_2511_aio -t ai-image-edit .
docker run -p 7860:7860 --gpus all --shm-size=640m -e APP_PASSWORD=... ai-image-edit
```

## GGUF image

Qwen-Image-2.1 with quantized GGUF weights, run through ComfyUI and the
[ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF) node, which needs far
less GPU memory than the `qwen_image` image. The weights are baked into the
image; choose them with build arguments:

| Build argument | Default | Meaning |
|---|---|---|
| `QWEN_GGUF_REPO` | `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` | Hugging Face repo with the GGUF, text encoder and VAE files |
| `QWEN_GGUF_UNCENSORED` | `1` | non-empty: third-party uncensored version; empty: the unmodified model |
| `QWEN_GGUF_QUANT` | `Q4_K_M` | `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, or `BF16` (uncensored only); larger is better and needs more memory |
| `QWEN_TEXT_ENCODER_QUANT` | `int8_convrot` | `int8_convrot` (9.4 GB) or `bf16` (17.5 GB) |

```bash
docker build -f deployments/docker/Dockerfile.qwen_image21_gguf -t ai-image-edit-gguf .
# unmodified model, Q8_0:
docker build -f deployments/docker/Dockerfile.qwen_image21_gguf \
    --build-arg QWEN_GGUF_UNCENSORED= --build-arg QWEN_GGUF_QUANT=Q8_0 -t ai-image-edit-gguf .
docker run -p 7860:7860 --gpus all --shm-size=640m -e APP_PASSWORD=... ai-image-edit-gguf
```

The image sets `MODEL_FILE`, `TEXT_ENCODER_FILE` and `VAE_FILE` (paths under
ComfyUI's `models/diffusion_models`, `models/text_encoders` and `models/vae`)
to the files it downloaded; override them to use other files that exist in
the container. Set `COMFY_EXTRA_ARGS` at run time to pass arguments to ComfyUI,
for example `-e COMFY_EXTRA_ARGS=--lowvram` on GPUs with little memory, and
`COMFY_GENERATION_TIMEOUT` (seconds, default 600 here) for slow GPUs.
The weights are under the Qwen RESEARCH LICENSE AGREEMENT
(non-commercial use only).

## qwen_image image

Build and run steps, the Pod setup and the CI build are in
[`../runpod/DEPLOY.md`](../runpod/DEPLOY.md).
