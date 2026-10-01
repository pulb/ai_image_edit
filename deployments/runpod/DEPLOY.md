# RunPod

Runs the `qwen_image` backend with the NiceGUI frontend in a GPU Pod, using
the image built from
[`deployments/docker/Dockerfile.qwen_image`](../docker/Dockerfile.qwen_image).
Weights are downloaded from the Hugging Face Hub at startup, not baked into
the image. The app serves on port `7860`.

The image installs the separately licensed
[`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen) package,
so it combines GPL and Qwen-licensed code: keep it private (see the
[License](../../README.md#license) section).

## Backend and frontend

The image sets `MODEL_BACKEND=qwen_image` and `FRONTEND=nicegui`. No
configuration is needed.

Optional Pod environment variables: `HF_TOKEN`, `QWEN21_AOTI`,
`QWEN21_AOTI_REPO`, `HF_HUB_CACHE`.

## Build

From the repo root:

```bash
docker build -f deployments/docker/Dockerfile.qwen_image -t ai-image-edit-qwen .
docker run -p 7860:7860 --gpus all ai-image-edit-qwen
```

`--build-arg QWEN_LIB_REF=<branch|tag|commit>` pins the package version
(default `main`).

## CI build

`.github/workflows/docker-qwen-image.yml` builds the image on every push to
`main` that touches the relevant files, on `v*` tags, and on manual runs. It
pushes `ghcr.io/<owner>/<repo>-qwen-image` tagged `latest`, the git tag and
the short commit SHA. Each build uses the newest commit of
`ai-image-edit-qwen`, resolved when the build starts. Pushes to that
package's repo don't trigger a build: run the workflow manually to pick them
up.

## Pod setup

1. Create a Pod from `ghcr.io/<owner>/<repo>-qwen-image:<tag>`, exposing
   HTTP port `7860`.
2. If the GHCR package is private (the default), add a registry credential
   under **Settings → Container Registry Auth** in the RunPod console: your
   GitHub username and a PAT with `read:packages`. Making the package public
   skips this.
3. Mount a network volume and set
   `HF_HUB_CACHE=/runpod-volume/hf-cache`, so the multi-GB weights download
   once instead of on every fresh container.

## Updating

RunPod can't swap a running Pod's image. Terminate and recreate the Pod on
the new tag (console or `runpodctl`).
