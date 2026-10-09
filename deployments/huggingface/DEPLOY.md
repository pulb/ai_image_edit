# Hugging Face Space

Runs the ComfyUI image ([`../docker/DEPLOY.md`](../docker/DEPLOY.md)) as a
Docker Space on a **paid GPU**. ZeroGPU is not supported: the models run in a
ComfyUI process, which needs a regular GPU. The UI is served on port `7860`.

## Space setup

1. Create a Space with the **Docker** SDK and pick a GPU under
   **Settings → Space hardware**: at least 48 GB of VRAM for `qwen_image21`,
   at least 16 GB for `qwen_image21_gguf`.
2. Copy [`../docker/Dockerfile`](../docker/Dockerfile) to the Space root as
   `Dockerfile`, and add a `README.md` card with the front matter:

   ```yaml
   ---
   title: AI Image Edit
   sdk: docker
   app_port: 7860
   ---
   ```
3. Under **Settings → Variables and secrets** set `MODEL_BACKEND` and
   `ACCEPT_LICENSES` (e.g. `qwen-research`), plus any optional variable from
   [Docker](../docker/DEPLOY.md). The weights download on first start.

The Space runs without a password login: don't set `APP_PASSWORD`. Access is
controlled by the Space's visibility; keep it private, since the weights are
under the Qwen RESEARCH LICENSE AGREEMENT (see the
[License](../../README.md#license) section).
