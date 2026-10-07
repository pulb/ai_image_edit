# Hugging Face Space

Runs the `qwen_image21` backend (diffusers pipeline for Qwen-Image-2.1) in a
Gradio-SDK Space. The app is installed as a package from this repo; the
Space itself only holds three small files. It serves on port `7860`.

## Space files

Copy the contents of [`space/`](space/) into the **root** of the Space repo:

| File | Purpose |
|---|---|
| `requirements.txt` | installs the app and its dependencies |
| `app.py` | entry point that sets the defaults |
| `README.md` | the Space card |

`requirements.txt` installs `ai-image-edit` from GitHub together with the
separately licensed [`ai-image-edit-qwen`](https://github.com/pulb/ai_image_edit_qwen)
package, so the Space combines GPL and Qwen-licensed code: keep the Space
private (see the [License](../../README.md#license) section).

## Backend and frontend

`app.py` sets the defaults `MODEL_BACKEND=qwen_image21` and `FRONTEND=gradio`.
To change them, edit those two lines, or set `MODEL_BACKEND` / `FRONTEND`
under **Settings → Variables and secrets** in the Space, which take
precedence over the defaults. Valid frontends are `gradio` and `nicegui`.
Both are installed; the Gradio SDK provides Gradio itself.

The Space runs without a password login: don't set `APP_PASSWORD`. Access is
controlled by the Space's visibility (keep it private).

Optional variables are the same as for the `qwen_image21` image (see
[`../docker/DEPLOY.md`](../docker/DEPLOY.md#qwen_image21-image)), except
`HF_HUB_CACHE`.

## Hardware

### ZeroGPU

- **Settings → Space hardware → ZeroGPU** (needs a PRO account).
- Keep `FRONTEND=gradio`. ZeroGPU (`spaces.GPU`) only works with the Gradio
  SDK Space type, which the card in `space/README.md` already declares.

### Paid GPU

- Pick a GPU under **Settings → Space hardware**. Recommended: A100 or L40S.
- `FRONTEND` can be `gradio` or `nicegui`. `nicegui` is the snappier
  frontend and also runs under the Gradio SDK, because the Space just runs
  `app.py` on port `7860`.
- To run it as a Docker Space instead, see
  [`../docker/DEPLOY.md`](../docker/DEPLOY.md): the same image works, with
  `sdk: docker` in the card and the Dockerfile copied to the Space root as
  `Dockerfile`.
