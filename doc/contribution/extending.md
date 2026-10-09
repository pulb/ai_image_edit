# Adding a model

## Adding a ComfyUI model

A model that runs through ComfyUI needs no Python. Add one JSON file to `src/ai_image_edit/data/workflows/`.
It holds the workflow (exported from ComfyUI with "Save (API Format)") and a manifest that says
which workflow inputs the prompt, seed, steps, images and so on go to. The file name is the value
for `MODEL_WORKFLOW`, and the file is registered automatically. The format is described in
[workflow files](workflow_files.md). Run the tests
(`PYTHONPATH=src python -m unittest discover -s tests/comfy_models`): they pick up the new file
automatically and check it against the JSON Schema and the workflow.

## Adding any other model

Implement `Model` (see `src/ai_image_edit/models/base.py`) and
register one loader in `src/ai_image_edit/models/__init__.py`.

## Dependencies

A ComfyUI model needs no new dependencies. Add the dependencies of any other model
to `pyproject.toml`.

A hand-written model that runs `imaging.run_masked_generation` should wrap its infer
callback with `result_cache.cached_infer(...)`, passing every value that
affects the output, so a repeated generation (for example after changing only
the mask) reuses the previous result. The cache is currently single-user.
ComfyUI models get this from `ComfyWorkflowModel`.
