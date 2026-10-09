# Adding a model

A model needs no Python. Add one JSON file to `src/ai_image_edit/data/workflows/`.
It holds the workflow (exported from ComfyUI with "Save (API Format)") and a manifest that says
which workflow inputs the prompt, seed, steps, images and so on go to. The file name is the value
for `MODEL_WORKFLOW` (`--workflow`), and the file is registered automatically. The format is described in
[workflow files](workflow_files.md). Run the tests
(`PYTHONPATH=src python -m unittest discover -s tests/comfy_models`): they pick up the new file
automatically and check it against the JSON Schema and the workflow.
