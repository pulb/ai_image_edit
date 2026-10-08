# Adding a model or frontend

## Adding a ComfyUI model

A model that runs through ComfyUI needs no Python. Create a folder under
`src/ai_image_edit/models/<name>/` with

- `workflow_api.json`: the workflow, exported from ComfyUI with "Save (API Format)", and
- `manifest.json`: which workflow inputs the prompt, seed, steps, images, weight files and so on go to.

The folder is registered automatically, and `<name>` is the value for
`MODEL_BACKEND`. The manifest format is described in
[`MANIFEST.md`](../src/ai_image_edit/models/_shared/comfyui/MANIFEST.md). Run the tests
(`PYTHONPATH=src python -m unittest discover -s tests/comfy_models`): they pick up the new
model automatically and check the manifest against the workflow.

## Adding any other model

Implement `ModelBackend` (see `src/ai_image_edit/models/base.py`) and
register one loader in `src/ai_image_edit/models/__init__.py`.

## Adding a frontend

Write a module that exposes `run(model: ModelBackend) -> None` and register it in
`src/ai_image_edit/frontends/__init__.py`.

## For all of them

Add the new backend's or frontend's dependencies as an extra in
`pyproject.toml`, named `<name>_backend` or `<name>_frontend`. (A ComfyUI model
needs no new extra: it runs on the ComfyUI stack the existing ones use.)

A hand-written backend that runs `imaging.run_masked_generation` should wrap its infer
callback with `result_cache.cached_infer(...)`, passing every value that
affects the output, so a repeated generation (for example after changing only
the mask) reuses the previous result. The cache is currently single-user.
ComfyUI models get this from `ComfyWorkflowModel`.
