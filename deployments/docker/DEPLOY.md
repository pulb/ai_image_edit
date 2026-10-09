# Docker

One image for every model, built from `deployments/docker/Dockerfile`. The app
serves the NiceGUI UI on port `7860`. The image holds ComfyUI,
PyTorch and the app, but no model weights: which model runs is chosen when the
container starts, and any [workflow file](../../doc/contribution/workflow_files.md)
of your own works too.

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

Logins survive restarts as long as the
password stays the same; changing it logs everyone out. Set
`APP_STORAGE_SECRET` to a long random value to keep sessions independent of
the password. A wrong password is delayed by one second. This is a single shared password without real rate
limiting. For stronger protection, don't expose the port and use an
SSH tunnel or Tailscale.

## ComfyUI image (`Dockerfile`)

ComfyUI, PyTorch and the app, without model weights. Which model runs is
chosen when the container starts, and its weights, LoRAs and custom nodes
are fetched on first start as its workflow file lists them (see
[Downloads](../../doc/contribution/workflow_files.md#downloads)). While that
runs the app is already up and says what is being downloaded when you try to
generate; ComfyUI starts when everything is there. Nothing is baked in, so the
image is small and, as it contains no weights, can be public.

```bash
docker build -f deployments/docker/Dockerfile -t ai-image-edit deployments/docker
docker run -p 7860:7860 --gpus all --shm-size=640m \
    -v ai-image-edit-models:/home/user/app/models \
    -e APP_PASSWORD=... -e ACCEPT_LICENSES=qwen-research ai-image-edit
```

| Variable | Meaning |
|---|---|
| `MODEL_WORKFLOW` | Model workflow: `qwen_image21` (default; the official bf16 weights, variant `int8`; needs a large GPU), `qwen_image21_gguf` (quantized, for smaller GPUs) or `qwen_image_edit_2511_aio`. |
| `MODEL_VARIANT` | Variant of the model, e.g. `Q8_0` for `qwen_image21_gguf` (`Q4_0`, `Q4_K_M` (default), `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`: the uncensored third-party version; `standard-Q4_0` to `standard-Q8_0`: the unmodified model). Larger is better and needs more memory. |
| `WORKFLOW_FILE` | Path of a workflow file in the container to run instead (mount it). |
| `ACCEPT_LICENSES` | Licenses you accept, comma-separated (`all` for every one). A model whose files have a license is not downloaded without it; the error names the license and its URL. `qwen_image21_gguf` needs `qwen-research`. |
| `DOWNLOAD_CONNECTIONS` | Parts of a file downloaded at once (default 8, at most 32). |
| `HF_TOKEN` | Hugging Face token, if a file needs a login. Sent to Hugging Face only. |

The same can be given as arguments: `docker run ... ai-image-edit python -m
ai_image_edit --model qwen_image21_gguf --variant Q8_0`.

The model folders are below `/home/user/app/models`. **Mount a volume there**
(as above), or the weights (about 15 GB for `qwen_image21_gguf`) are
downloaded again by every new container. Any other location works the same
way: it is just the volume you mount. `MODEL_FILE`, `TEXT_ENCODER_FILE` and
`VAE_FILE` (paths under `models/diffusion_models`, `models/text_encoders`
and `models/vae`) can still name other files that exist in the container
instead of the ones from the workflow file; set `MODEL_VERSION` to label them.
`COMFY_GENERATION_TIMEOUT` (seconds, default 600) is for slow GPUs.

The GPU needs at least 16 GB of VRAM for `qwen_image21_gguf`. `qwen_image21` with the bf16 weights needs far more; it was tested on an A100, and `MODEL_VARIANT=int8` uses less.

The Qwen-Image-2.1 weights are under the Qwen RESEARCH LICENSE AGREEMENT
(non-commercial use only), which is why they are not in the image and why you
accept it with `ACCEPT_LICENSES` when you run it.

Build arguments: `APP_REPO` and `APP_REF` (where the app is installed from:
`pulb/ai_image_edit` at `main` by default; the image is not built from the
build context, so to try a change, push it to a branch and pass its name),
`COMFYUI_REF`, and the PyTorch versions.

### ComfyUI arguments (`COMFY_EXTRA_ARGS`)

Set `COMFY_EXTRA_ARGS` at run time to pass arguments to ComfyUI, for example
`-e COMFY_EXTRA_ARGS="--highvram --disable-dynamic-vram"`. The variable also
applies to the `qwen_image_edit_2511_aio` image, but the advice below was
tested only with this one:

| Arguments | Use when |
|---|---|
| `--lowvram` | The GPU has little memory (about 8 GB or less). |
| `--highvram --disable-dynamic-vram` | The GPU has about 24 GB or more (estimate; see below). By default ComfyUI re-stages the text encoder for every new prompt, which added roughly 10-15 s per generation in our tests; with these flags the weights stay on the GPU and generation time matches the former Diffusers-based implementation. `--disable-dynamic-vram` is deprecated in ComfyUI and will be removed. |
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
`ghcr.io/<owner>/<repo>` tagged `latest`, the git tag and the short commit
SHA.
