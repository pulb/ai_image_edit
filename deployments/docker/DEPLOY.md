# Docker

Two images, both built from the repo root. The app serves on port `7860`.

| Dockerfile | Backend | Frontend | Use |
|---|---|---|---|
| `Dockerfile.qwen_image_edit_comfy` | `qwen_image_edit_comfy` | `nicegui` | self-hosted ComfyUI backend |
| `Dockerfile.qwen_image` | `qwen_image` | `nicegui` | see [RunPod](../runpod/DEPLOY.md) |

## Password login

The app sits behind a password login when `APP_PASSWORD` is
set (for example `-e APP_PASSWORD=...`). Both images set
`REQUIRE_PASSWORD=1` and refuse to start without it. Details are in
[`../runpod/DEPLOY.md`](../runpod/DEPLOY.md).

## ComfyUI image

Clones ComfyUI, installs PyTorch, downloads the model checkpoint and a set of
LoRAs, then starts the app. The checkpoint is set by two variables near the top
of the Dockerfile: `COMFY_CHECKPOINT_URL` (where it is downloaded from) and
`COMFY_CHECKPOINT_PATH` (its path under ComfyUI's `models/checkpoints` folder).
The app loads the file `COMFY_CHECKPOINT_PATH` names and refuses to start if it
is empty or the file does not exist.

```bash
docker build -f deployments/docker/Dockerfile.qwen_image_edit_comfy -t ai-image-edit .
docker run -p 7860:7860 --gpus all -e APP_PASSWORD=... ai-image-edit
```

## qwen_image image

Build and run steps, the Pod setup and the CI build are in
[`../runpod/DEPLOY.md`](../runpod/DEPLOY.md).
