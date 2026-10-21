# Hugging Face Space

Runs AI Image Edit ([Docker image](../docker/DEPLOY.md)) as a
Docker Space on a **paid GPU**. ZeroGPU is not supported: the models run in a
ComfyUI process, which needs a regular GPU. The UI is served on port `7860`.

## Space setup

1. Create a Space with the **Docker** SDK and pick a GPU under
   **Settings → Space hardware**: at least 48 GB of VRAM for `qwen_image21` and `qwen_image21_turbo`,
   at least 16 GB for the GGUF workflows.
2. Copy the contents of [`space/`](space/) into the root of the Space repo: the
   `README.md` card (`sdk: docker`, `app_port: 7860`) and the `Dockerfile`, which
   is a symlink to [`../docker/Dockerfile`](../docker/Dockerfile). Copy the
   symlink's target, not the link itself (`cp -L`).
3. Optional: under **Settings → Variables and secrets** set `MODEL_WORKFLOW` to change the model from
   the default `qwen_image21`, plus any other optional variable from [Docker](../docker/DEPLOY.md). The
   weights download on first start. The Qwen-Image-2.1 weights are for non-commercial use only
   ([license](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE)).

The image needs a password (`REQUIRE_PASSWORD=1`); without one, the Space shows a page
saying that `APP_PASSWORD` is not set. Either set
`APP_PASSWORD`, or set `REQUIRE_PASSWORD` to an empty value and restrict access
by setting the Space's visibility to private in Hugging Face.

The weights are downloaded when the Space starts and are not part of the Space
repository, but they may be under the Qwen RESEARCH LICENSE AGREEMENT
(non-commercial use only) or another license that restricts use; check the
[licenses](../../README.md#license) before making the Space public.
