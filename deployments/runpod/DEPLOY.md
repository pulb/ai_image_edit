# RunPod

Runs any backend with the NiceGUI frontend in a GPU Pod. Each backend has
its own image, built from the repo root. The app serves on port `7860`.

A public [RunPod template](https://console.runpod.io/hub/template/ayt3pyrp7w?ref=e1nr94ls)
works with any of the images. You only need to replace its container image with
your own and, if your registry is private, add registry credentials. Both
steps are described under [Build](#build) and [Pod setup](#pod-setup) below.

| Backend | Dockerfile | Weights |
|---|---|---|
| `qwen_image21` | [`Dockerfile.qwen_image21`](../docker/Dockerfile.qwen_image21) | downloaded from the Hugging Face Hub at startup |
| `qwen_image_edit_2511_aio` | [`Dockerfile.qwen_image_edit_2511_aio`](../docker/Dockerfile.qwen_image_edit_2511_aio) | checkpoint and LoRAs baked into the image |
| `qwen_image21_gguf` | [`Dockerfile.qwen_image21_gguf`](../docker/Dockerfile.qwen_image21_gguf) | quantized GGUF weights baked into the image, chosen with build arguments (see [Docker](../docker/DEPLOY.md#gguf-image)) |

The `qwen_image21` image installs the separately licensed
[`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen) package,
so it combines GPL and Qwen-licensed code: keep it private (see the
[License](../../README.md#license) section).

## Backend and frontend

Each image sets its own `MODEL_BACKEND` and uses `FRONTEND=nicegui`, so
no backend or frontend configuration is needed. Pick the backend by
choosing the image.

Optional Pod environment variables for `qwen_image21`: `HF_TOKEN` (gated
weights or a private AOTI repo), `QWEN21_AOTI` / `QWEN21_AOTI_REPO` (see
`src/ai_image_edit/models/qwen_image21/model.py`), `HF_HUB_CACHE`.

For `qwen_image_edit_2511_aio`, `MODEL_FILE` selects the checkpoint
(a path under ComfyUI's `models/checkpoints` folder). For `qwen_image21_gguf`,
`MODEL_FILE`, `TEXT_ENCODER_FILE` and `VAE_FILE` select the weights, and
`COMFY_EXTRA_ARGS` (e.g. `--lowvram`) passes arguments to ComfyUI. The image sets it to the
weights it ships with; override it to use another file that exists in the Pod.

## Image storage

Uploads and results are kept in RAM under `/dev/shm`, not on the Pod's disk
(see [Docker](../docker/DEPLOY.md#image-storage) for the details and the
`AI_IMAGE_EDIT_WORK_MAX_SIZE` / `AI_IMAGE_EDIT_WORK_DIR` variables). The
size of `/dev/shm` is set by RunPod, not by the image, and the actual cap may
be lower than `AI_IMAGE_EDIT_WORK_MAX_SIZE` if `/dev/shm` is small (the app
prints a warning at startup). Check it with `df -h /dev/shm` in the Pod.

## Access protection

RunPod's HTTP proxy is public and has no login of its own, so anyone with
the Pod URL can use the app. Set the Pod environment variable
`APP_PASSWORD` to put the whole app, including the image files it serves,
behind a password login page. All images set `REQUIRE_PASSWORD=1` and
refuse to start without `APP_PASSWORD`.

With NiceGUI (the images' frontend), logins survive restarts as long as the
password stays the same; changing it logs everyone out. Set
`APP_STORAGE_SECRET` to a long random value to keep sessions independent of
the password. A wrong password is delayed by one second. The Gradio frontend
also honors `APP_PASSWORD` (any username, shared password), but every restart
logs everyone out. Either way this is a single shared password without real rate
limiting. For stronger protection, don't expose the port and use an
SSH tunnel or Tailscale.

## Build

From the repo root:

```bash
# qwen_image21
docker build -f deployments/docker/Dockerfile.qwen_image21 -t ai-image-edit-qwen .

# qwen_image_edit_2511_aio
docker build -f deployments/docker/Dockerfile.qwen_image_edit_2511_aio -t ai-image-edit .

# qwen_image21_gguf
docker build -f deployments/docker/Dockerfile.qwen_image21_gguf -t ai-image-edit-gguf .
```

`--build-arg QWEN_LIB_REF=<branch|tag|commit>` pins the package version of
the `qwen_image21` image (default `main`).

## CI build

`.github/workflows/docker-qwen-image21.yml` builds the `qwen_image21` image on
every push to `main` that touches the relevant files, on `v*` tags, and on
manual runs. It pushes `ghcr.io/<owner>/<repo>-qwen-image21` tagged `latest`,
the git tag and the short commit SHA. Each build uses the newest commit of
`ai-image-edit-qwen`, resolved when the build starts. Pushes to that
package's repo don't trigger a build: run the workflow manually to pick them
up.

There is no CI build for the ComfyUI image. Build it yourself and push it to
a registry RunPod can pull from. It has not been tried on RunPod.

## Pod setup

1. Create a Pod from the image (for `qwen_image21`:
   `ghcr.io/<owner>/<repo>-qwen-image21:<tag>`), exposing HTTP port `7860`.
   Recommended GPUs: A100 or L40S.
2. If the registry package is private (GHCR packages are by default), add a
   registry credential under **Settings → Container Registry Auth** in the
   RunPod console. For GHCR that is your GitHub username and a PAT with
   `read:packages`; making the package public skips this.
3. `qwen_image21` only: attach a network volume (a Pod mounts it at
   `/workspace`) and set `HF_HUB_CACHE=/workspace/hf-cache`, so the multi-GB
   weights download once instead of on every fresh container.

## Updating

RunPod can't swap a running Pod's image. Terminate and recreate the Pod on
the new tag (console or `runpodctl`).

`.github/workflows/docker-qwen-image21-gguf.yml` builds the `qwen_image21_gguf`
image (about 15 GB with its weights) on manual runs, where quantization,
uncensored or unmodified model and text encoder precision are inputs, and on
`v*` tags. It pushes `ghcr.io/<owner>/<repo>-qwen-image21-gguf` tagged with the
variant, for example `uc-q4_k_m`. Keep the package private: the weights are
under the Qwen RESEARCH LICENSE AGREEMENT (non-commercial use only).
