# RunPod

Runs any model workflow in a GPU Pod. One image serves them all; how to build, configure and run them
is described in [Docker](../docker/DEPLOY.md). This page covers only what is
specific to RunPod.

A public [RunPod template](https://console.runpod.io/hub/template/ayt3pyrp7w?ref=e1nr94ls)
is preconfigured and ready to go, and works with any model workflow.

| Model workflow (`MODEL_WORKFLOW`) | Weights |
|---|---|
| `qwen_image21` (default) | downloaded on first start |
| `qwen_image21_turbo` | downloaded on first start |
| `qwen_image21_gguf` | downloaded on first start |
| `qwen_image21_turbo_gguf` | downloaded on first start |
| `qwen_image_edit_2511_aio` | downloaded on first start |

Environment variables, including the ComfyUI arguments for each GPU size, are
set as Pod environment variables and are described in
[Docker](../docker/DEPLOY.md), together with the GitHub workflows that build
the image.

## Pod setup

1. Create a Pod from the image, exposing HTTP port `7860`, e.g.
   `ghcr.io/pulb/ai_image_edit:latest`. The default is `qwen_image21`; select another with
   `MODEL_WORKFLOW`. The Qwen-Image-2.1 weights are for non-commercial use only
   ([license](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE)). Choose one of the GPUs the template recommends
   that has enough memory: at least 48 GB of VRAM for `qwen_image21` and `qwen_image21_turbo`, at least
   16 GB for the GGUF workflows.
2. Set the Pod's volume mount path to `/home/user/app/models`
   (`--volume-mount-path` with `runpodctl`), so the multi-GB weights are
   downloaded once. RunPod mounts the volume at `/workspace` by default, which
   leaves the weights on the container disk, and that is wiped whenever the Pod
   stops or restarts. The Pod's own volume disk keeps them until the Pod is
   terminated. A network volume keeps them after that too, but it can only be
   attached when the Pod is created.

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
to put the whole app behind a password login page; without it the app does not
start, and its page says that `APP_PASSWORD` is not set. Details: [Docker](../docker/DEPLOY.md#password-login).

## When the app cannot start

The page says which part failed (for example a missing `APP_PASSWORD` or an
invalid setting); the details are in the Pod's logs. See
[Docker](../docker/DEPLOY.md#when-the-app-cannot-start).

## Updating

Change the Pod's image to the new tag, for example
`runpodctl pod update <pod-id> --image ghcr.io/pulb/ai_image_edit:latest`. The
Pod restarts on it; weights on its volume (see [Pod setup](#pod-setup)) are kept.

`latest` is the newest build of `main`, and every build is also tagged
`sha-<short commit>`. To try a branch, run the *Build Docker image* workflow on
it (GitHub: Actions → Build Docker image → Run workflow) and use the `sha-` tag
it pushes.
