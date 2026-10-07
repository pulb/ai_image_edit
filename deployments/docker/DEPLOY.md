# Docker

Two images, both built from the repo root. The app serves on port `7860`.

| Dockerfile | Backend | Frontend | Use |
|---|---|---|---|
| `Dockerfile.qwen_image_edit_comfy` | `qwen_image_edit_comfy` | `nicegui` | self-hosted ComfyUI backend |
| `Dockerfile.qwen_image` | `qwen_image` | `nicegui` | see [RunPod](../runpod/DEPLOY.md) |

## Image storage

Uploads and results are kept in RAM, in a folder under `/dev/shm`, so they
never reach the container's disk, and they disappear when the container stops.
The app deletes the oldest files once the folder exceeds
`AI_IMAGE_EDIT_WORK_MAX_SIZE` (default `512m`, set in the Dockerfile; `0` means as
much as the filesystem allows).

The actual cap may be lower than that setting. It is limited to 90% of the
size of `/dev/shm`, and Docker gives `/dev/shm` only 64 MB by default: pass
`--shm-size=1g` (or more) to `docker run`. If the setting had to be lowered,
the app prints a warning at startup. `/dev/shm` is shared with anything else
in the container that uses it. `AI_IMAGE_EDIT_WORK_DIR` moves the folder
elsewhere.

RAM disks keep images off the disk only: the host can still read memory, and
the traffic to the app is not covered. Memory used here counts against the
container's RAM limit.

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
docker run -p 7860:7860 --gpus all --shm-size=1g -e APP_PASSWORD=... ai-image-edit
```

## qwen_image image

Build and run steps, the Pod setup and the CI build are in
[`../runpod/DEPLOY.md`](../runpod/DEPLOY.md).
