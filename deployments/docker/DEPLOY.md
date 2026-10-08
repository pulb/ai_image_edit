# Docker

Three images, all built from the repo root. The app serves on port `7860`.
Each image sets its own `MODEL_BACKEND` and uses `FRONTEND=nicegui`, so no
backend or frontend configuration is needed: pick the backend by choosing the
image.

| Dockerfile | Backend | Frontend | Use |
|---|---|---|---|
| `Dockerfile.qwen_image21_gguf` | `qwen_image21_gguf` | `nicegui` | quantized Qwen-Image-2.1 run through ComfyUI, for low-VRAM GPUs |
| `Dockerfile.qwen_image_edit_2511_aio` | `qwen_image_edit_2511_aio` | `nicegui` | all-in-one checkpoint run through ComfyUI |
| `Dockerfile.qwen_image21` | `qwen_image21` | `nicegui` | diffusers pipeline, weights downloaded at startup |

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

The app sits behind a password login when `APP_PASSWORD` is set (for example
`-e APP_PASSWORD=...`), which protects the whole app, including the image
files it serves. All images set `REQUIRE_PASSWORD=1` and refuse to start
without it.

With NiceGUI (the images' frontend), logins survive restarts as long as the
password stays the same; changing it logs everyone out. Set
`APP_STORAGE_SECRET` to a long random value to keep sessions independent of
the password. A wrong password is delayed by one second. The Gradio frontend
also honors `APP_PASSWORD` (any username, shared password), but every restart
logs everyone out. Either way this is a single shared password without real rate
limiting. For stronger protection, don't expose the port and use an
SSH tunnel or Tailscale.

## qwen_image_edit_2511_aio image

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

`.github/workflows/docker-qwen-image-edit-2511-aio.yml` builds the image on
manual runs only and pushes `ghcr.io/<owner>/<repo>-qwen-image-edit-2511-aio`
tagged `latest` and the short commit SHA.

## qwen_image21_gguf image

Qwen-Image-2.1 with quantized GGUF weights, run through ComfyUI and the
[ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF) node, which needs far
less GPU memory than the `qwen_image21` image. This backend and image are
meant for low-VRAM GPUs, on RunPod or in self-hosted environments; the GPU
needs at least 16 GB of VRAM. The weights are under the Qwen RESEARCH LICENSE
AGREEMENT (non-commercial use only): keep the image and its registry package
private. They are baked into the image; choose them with build arguments:

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
the container. `COMFY_GENERATION_TIMEOUT` (seconds, default 600 here) is
for slow GPUs.

### ComfyUI arguments (`COMFY_EXTRA_ARGS`)

Set `COMFY_EXTRA_ARGS` at run time to pass arguments to ComfyUI, for example
`-e COMFY_EXTRA_ARGS="--highvram --disable-dynamic-vram"`. The variable also
applies to the `qwen_image_edit_2511_aio` image, but the advice below was
tested only with this one:

| Arguments | Use when |
|---|---|
| `--lowvram` | The GPU has little memory (about 8 GB or less). |
| `--highvram --disable-dynamic-vram` | The GPU has about 24 GB or more (estimate; see below). By default ComfyUI re-stages the text encoder for every new prompt, which added roughly 10-15 s per generation in our tests; with these flags the weights stay on the GPU and generation time matches the Diffusers-based backend. `--disable-dynamic-vram` is deprecated in ComfyUI and will be removed. |
| `--disable-comfy-compiler` | Sampling appears to hang on the first step. |

The weights that `--highvram` keeps resident add up to about 14 GB with the
default `Q4_K_M` model (4.6 GB model, 8.9 GB text encoder, under 1 GB VAE).
Sampling needs a few GB of activations on top (a rough guess: 3-6 GB), so
expect roughly 17-20 GB in total, and `Q8_0` (7.6 GB model) about 3 GB more.
These figures are estimates: a 24 GB GPU should be enough, while a 16 GB GPU is
probably too small. The image runs without the flags on a 16 GB GPU (tested).
If you want to try the flags on a smaller GPU, check the peak with
`nvidia-smi` during a generation.

### Pinned versions

ComfyUI (`COMFYUI_REF`) and ComfyUI-GGUF (`COMFYUI_GGUF_REF`) are pinned to
tested commits. ComfyUI-GGUF is also patched at build time
(`patches/comfyui_gguf_input_act.py`): ComfyUI 0.39 passes new keyword
arguments such as `input_act` to every Linear layer, which upstream does not
accept yet. The build fails if the patch no longer applies.

### CI build

`.github/workflows/docker-qwen-image21-gguf.yml` builds the image (about
15 GB with its weights) on manual runs, where quantization, uncensored or
unmodified model and text encoder precision are inputs, and on `v*` tags. It
pushes `ghcr.io/<owner>/<repo>-qwen-image21-gguf` tagged with the variant, for
example `uc-q4_k_m`.

## qwen_image21 image

Diffusers pipeline with the weights downloaded from the Hugging Face Hub at
startup (no ComfyUI). It installs the separately licensed
[`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen) package, so
it combines GPL and Qwen-licensed code: keep it private (see the
[License](../../README.md#license) section).

The GPU needs at least 48 GB of VRAM.

```bash
docker build -f deployments/docker/Dockerfile.qwen_image21 -t ai-image-edit-qwen .
docker run -p 7860:7860 --gpus all --shm-size=640m -e APP_PASSWORD=... ai-image-edit-qwen
```

`--build-arg QWEN_LIB_REF=<branch|tag|commit>` pins the package version
(default `main`).

Optional environment variables: `HF_TOKEN` (gated weights or a private AOTI
repo), `QWEN21_AOTI` / `QWEN21_AOTI_REPO` (see
`src/ai_image_edit/models/qwen_image21/model.py`), `HF_HUB_CACHE`. Point
`HF_HUB_CACHE` at a volume so the multi-GB weights download once instead of on
every fresh container.

`.github/workflows/docker-qwen-image21.yml` builds the image on every push to
`main` that touches the relevant files, on `v*` tags, and on manual runs. It
pushes `ghcr.io/<owner>/<repo>-qwen-image21` tagged `latest`, the git tag and
the short commit SHA. Each build uses the newest commit of
`ai-image-edit-qwen`, resolved when the build starts. Pushes to that package's
repo don't trigger a build: run the workflow manually to pick them up.
