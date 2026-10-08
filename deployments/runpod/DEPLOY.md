# RunPod

Runs any backend with the NiceGUI frontend in a GPU Pod. Each backend has
its own image, built from the repo root; how to build, configure and run them
is described in [Docker](../docker/DEPLOY.md). This page covers only what is
specific to RunPod.

A public [RunPod template](https://console.runpod.io/hub/template/ayt3pyrp7w?ref=e1nr94ls)
works with any of the images. You only need to replace its container image with
your own and, if your registry is private, add registry credentials (see
[Pod setup](#pod-setup)).

| Backend | Image | Weights |
|---|---|---|
| `qwen_image21_gguf`, `qwen_image_edit_2511_aio` (`MODEL_BACKEND`) | [ComfyUI image](../docker/DEPLOY.md#comfyui-image-dockerfile) | downloaded on first start (see [Pod setup](#pod-setup)) |
| `qwen_image21` | [`qwen_image21`](../docker/DEPLOY.md#qwen_image21-image) | downloaded from the Hugging Face Hub at startup |

Environment variables, including the ComfyUI arguments for each GPU size, are
set as Pod environment variables and are described per image in
[Docker](../docker/DEPLOY.md), together with the GitHub workflows that build
the images.

## Pod setup

1. Create a Pod from the image, exposing HTTP port `7860`. For example
   `ghcr.io/<owner>/<repo>:latest` for the ComfyUI models (select the model
   with `MODEL_BACKEND` and, for `qwen_image21_gguf`, `ACCEPT_LICENSES=qwen-research`),
   or `ghcr.io/<owner>/<repo>-qwen-image21:<tag>` for `qwen_image21`. Choose one of the GPUs the template recommends
   that has enough memory: at least 48 GB of VRAM for `qwen_image21`, at least
   16 GB for `qwen_image21_gguf`.
2. If the registry package is private (GHCR packages are by default), add a
   registry credential under **Settings → Container Registry Auth** in the
   RunPod console. For GHCR that is your GitHub username and a PAT with
   `read:packages`; making the package public skips this.
3. Optional: attach a network volume (a Pod mounts it at `/workspace`) so the
   multi-GB weights download once instead of on every fresh container. For
   the ComfyUI image, mount it at `/home/user/app/models` if the Pod
   lets you choose the mount path (otherwise the weights are stored on the
   Pod's disk); for
   `qwen_image21`, set `HF_HUB_CACHE=/workspace/hf-cache`.

## Image storage

Uploads and results are kept in RAM under `/dev/shm`, not on the Pod's disk
(see [Docker](../docker/DEPLOY.md#image-storage) for the details and the
`AI_IMAGE_EDIT_WORK_MAX_SIZE` / `AI_IMAGE_EDIT_WORK_DIR` variables). The
size of `/dev/shm` is set by RunPod, not by the image, and the actual cap may
be lower than `AI_IMAGE_EDIT_WORK_MAX_SIZE` if `/dev/shm` is small (the app
prints a warning at startup). Check it with `df -h /dev/shm` in the Pod.

## Access protection

RunPod's HTTP proxy is public and has no login of its own, so anyone with
the Pod URL can use the app. Set the Pod environment variable `APP_PASSWORD`
to put the whole app behind a password login page; all images refuse to start
without it. Details: [Docker](../docker/DEPLOY.md#password-login).

## Updating

RunPod can't swap a running Pod's image. Terminate and recreate the Pod on
the new tag (console or `runpodctl`).
