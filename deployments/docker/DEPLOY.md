# Docker

One image for every model, built from `deployments/docker/Dockerfile`. The app
serves the NiceGUI UI on port `7860`. The image holds ComfyUI,
PyTorch and the app, but no model weights: which model runs is chosen when the
container starts, and any [workflow file](../../doc/contributing/workflow_files.md)
of your own works too.

## Image storage

Uploads and results are kept in RAM, in a folder under `/dev/shm`, so they
never reach the container's disk, and they disappear when the container stops.
The app deletes the oldest files once the folder exceeds
`AI_IMAGE_EDIT_WORK_MAX_SIZE` (default `256m`, set in the Dockerfile; `0` means as
much as the filesystem allows). It keeps the image you are editing, the
reference images and the result on screen, so the folder can stay above the
limit while they alone are larger than it.

The actual cap may be lower than that setting. It is limited to 90% of the
size of `/dev/shm`, and Docker gives `/dev/shm` only 64 MB by default: pass
`--shm-size=320m` (or more) to `docker run`. 320 MB is the smallest round
size that fits the default 256 MB cap: 256 MB needs a `/dev/shm` of at least
about 285 MB (256 / 0.9), and 320 MB leaves some headroom. If you raise
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
files it serves. The image sets `REQUIRE_PASSWORD=1`, so without a password the
app does not start and its page says that `APP_PASSWORD` is not set; set
`REQUIRE_PASSWORD` to an empty value to run without a password.

Logins survive restarts as long as the
password stays the same; changing it logs everyone out. The session secret is
a salted PBKDF2 hash of the password. Set `APP_STORAGE_SECRET` to a long random
value to keep sessions independent of the password. A wrong password is delayed by one second. This is a single shared password without real rate
limiting. For stronger protection, don't expose the port and use an
SSH tunnel or Tailscale.

The password does not make the app ready for several users: it is made for
one person at a time (see the [README](../../README.md#deployments)).

## When the app cannot start

If the app cannot start (a missing password, an invalid setting or workflow,
ComfyUI not launching), it serves a page that says which part failed, with the
HTTP status 503, until it is stopped. The log has the details.

## Image (`Dockerfile`)

ComfyUI, PyTorch and the app, without model weights. Which model runs is
chosen when the container starts, and its weights, LoRAs and custom nodes
are fetched on first start as its workflow file lists them (see
[Downloads](../../doc/contributing/workflow_files.md#downloads)). While that
runs the app is already up: the page shows the progress, and Generate stays off
until everything is there and ComfyUI has started. Nothing is baked in, so the
image is small and, as it contains no weights, can be public.

The published image (`ghcr.io/pulb/ai_image_edit`) has zstd-compressed layers,
which unpack faster than gzip; pulling it needs Docker 23 or newer (or a
containerd of 1.5 or newer).

```bash
docker build -f deployments/docker/Dockerfile -t ai-image-edit deployments/docker
docker run -p 7860:7860 --gpus all --shm-size=320m \
    -v ai-image-edit-models:/home/user/app/models \
    -e APP_PASSWORD=... ai-image-edit
```

| Variable | Meaning |
|---|---|
| `MODEL_WORKFLOW` | Model workflow: `qwen_image21` (default; the official bf16 weights, or the smaller `int8` ones with `MODEL_VARIANT=int8`; needs a large GPU), `qwen_image21_gguf` (quantized, for smaller GPUs), `qwen_image21_turbo` / `qwen_image21_turbo_gguf` (the 8-step Turbo version, full precision or quantized) or `qwen_image_edit_2511_aio`. A path to a workflow file in the container (mount it) runs that file instead. |
| `MODEL_VARIANT` | Variant of the model, e.g. `Q8_0` for `qwen_image21_gguf` (`Q4_0`, `Q4_K_M` (default), `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`: the uncensored third-party version; `standard-Q4_0` to `standard-Q8_0`: the unmodified model; `qwen_image21_turbo_gguf` has `Q2_K` to `F16`, default `Q4_K_M`). Larger is better and needs more memory. |
| `DOWNLOAD_LORAS` | `0` skips the LoRAs a workflow brings along (the Turbo LoRA of `qwen_image21`). LoRAs already in the folder are still offered. On by default. |
| `DOWNLOAD_CONNECTIONS` | Parts of a file downloaded at once (default 16, at most 32). |
| `HF_TOKEN` | Hugging Face token, if a file needs a login. Sent to Hugging Face only. |

The same can be given as arguments to the image's command, `ai-image-edit`:
`docker run ... ai-image-edit ai-image-edit --workflow qwen_image21_gguf --variant Q8_0`
(`--no-lora-downloads` for `DOWNLOAD_LORAS=0`).

The model folders are below `/home/user/app/models`. **Mount a volume there**
(as above), or the weights (about 15 GB for `qwen_image21_gguf`) are
downloaded again by every new container. Any other location works the same
way: it is just the volume you mount.

A generation is abandoned if ComfyUI reports no progress (a node starting or a
sampler step) for `COMFY_GENERATION_TIMEOUT` seconds (default 300; a single
node such as loading the weights from a slow disk must finish within it), or
if it takes more than `COMFY_GENERATION_MAX_TIME` seconds in all (default
1800, including the wait behind other generations).

The GPU needs at least 16 GB of VRAM for `qwen_image21_gguf`. `qwen_image21` with the bf16 weights needs far more; it was tested on an A100, and `MODEL_VARIANT=int8` uses less.

The Qwen-Image-2.1 weights are under the
[Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE):
**non-commercial use only.** They are not in the image but downloaded when the
model starts, and using them means accepting that license.

Build arguments: `APP_REPO` and `APP_REF` (where the app is installed from:
`pulb/ai_image_edit` at `main` by default; the image is not built from the
build context, so to try a change, push it to a branch and pass its name),
`COMFYUI_REF`, and the PyTorch versions.

### ComfyUI arguments

At start-up the app reads the free GPU memory (`nvidia-smi`) and compares it
with the size of the model files on disk plus a headroom of 6 GB for the
generation itself (a workflow can change it with `vram_headroom_gb`). If the
memory is sufficient, ComfyUI is started with `--highvram --disable-dynamic-vram`
so the weights stay on the GPU, which saves roughly 10-15 s per generation.
Otherwise it gets no extra arguments. The decision and its reason are printed
to the log (`ComfyUI arguments: ...`). The 6 GB is an estimate, not a
measurement.

ComfyUI is started with `CUDA_DEVICE_ORDER=PCI_BUS_ID` (unless you set the
variable), so GPU numbers, `CUDA_VISIBLE_DEVICES` included, are the ones
`nvidia-smi` shows. With another `CUDA_DEVICE_ORDER` the app cannot tell which
GPU ComfyUI uses and starts it without extra arguments.

Set `COMFY_EXTRA_ARGS` at run time to override this, for example
`-e COMFY_EXTRA_ARGS="--lowvram"`. If the variable is set, even to an empty
value, its arguments are used as they are and nothing is chosen automatically.
The advice below was tested only with `qwen_image21_gguf`:

| Arguments | Use when |
|---|---|
| `--lowvram` | The GPU has little memory (about 8 GB or less). |
| `--highvram --disable-dynamic-vram` | The GPU has about 24 GB or more (estimate; see below). By default ComfyUI re-stages the text encoder for every new prompt, which added roughly 10-15 s per generation in our tests; with these flags the weights stay on the GPU. `--disable-dynamic-vram` is deprecated in ComfyUI and will be removed. |
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

ComfyUI (`COMFYUI_REF`) is pinned to a tested commit. Custom nodes are pinned
in the workflow files: `qwen_image21_gguf` installs the
[ComfyUI-GGUF](https://github.com/pulb/ComfyUI-GGUF) fork at an exact commit,
which fixes an incompatibility with ComfyUI 0.39 (it passes new keyword
arguments such as `input_act` to every Linear layer, which upstream does not
accept yet).

### CI build

`.github/workflows/docker.yml` builds the image on pushes to `main` that touch
the Dockerfile, `pyproject.toml` or `src/`, on `v*` tags and on manual runs. It
installs the app from the commit it was started for and pushes
`ghcr.io/<owner>/<repo>` tagged `sha-<short commit>`, plus `latest` for builds
of `main` and the git tag for tag builds. The app's version (shown in the page header and the log, printed by
`ai-image-edit --version`) is derived from the git tags, so a `v*` tag marks a release.
