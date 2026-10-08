# ComfyUI model manifests

A ComfyUI model is a folder under `models/` with two files; no Python is needed:

```
models/<name>/workflow_api.json   the workflow, exported from ComfyUI with "Save (API Format)"
models/<name>/manifest.json       how the app's inputs map onto that workflow
```

The folder name is the backend name (`MODEL_BACKEND=<name>`). The folder is
found automatically (`models/__init__.py`), and `ComfyWorkflowModel` loads it.
Mistakes such as a target that names a node missing from the workflow fail at
load time.

A **target** is `"<node id>.<input name>"`, e.g. `"6.seed"`. The input name may
contain dots (`"5.images.image_2"`). Every target the app fills in (everything
except the reference image slots) must already exist in the workflow with a
placeholder value, so a mistyped node id or input name fails when the model loads.

| Key | Meaning |
|---|---|
| `model_name` | Name shown in the UI. |
| `model_version` | Optional. `env` (environment variable), `regex` (group 1 is the version), `basename`, `ignore_case`, `upper`, and `prefixes` (`[{regex, text}]` prepended when matching). |
| `workflow` | Workflow file, default `workflow_api.json`. |
| `required_nodes`, `missing_nodes_hint` | Node classes ComfyUI must provide, checked on the first generation, and the advice shown if they are missing. |
| `files` | Weight files: `env` names a variable holding a path relative to `folder`; the path is written to `set`. Existence is checked at startup. |
| `bind` | Where each app input goes: `prompt`, `seed`, `steps`, `sampler`, `scheduler` (required), `cfg`, `denoise`, `negative_prompt` (optional). An input that is not bound is not offered in the UI (`cfg`, `denoise`, `negative_prompt`). |
| `constants` | Fixed values written into the workflow: `{target: value}`. |
| `seed` | `min`, `max` of random seeds; `wrap` reduces a chosen seed modulo `max + 1`. |
| `images` | `source`: target for the source image. `references`: further images, each a `LoadImage` node wired to `target` (`{n}` counts from `first_n`), with node ids from `loader_ids` (a list) or `loader_id_start`, up to `max`. Unused slots in the template are removed. |
| `size` | `policy` (see `size_policies.py`: `qwen21_tiers`, `fixed_area`), its settings, and `bind`, which maps the policy's named values (`resolution`, or `width`/`height`) to targets. |
| `loras` | Omit if the model has none. Chains one `loader_class` node per selected file from `first_id`; `from` are the links it starts from, `outputs` the names of its outputs, `feeds` the targets that receive the final links, `strength_inputs` the inputs that get the strength. `dir` is searched for LoRA files. |
| `capabilities` | What the UI offers: `supports_inpainting`, `num_annotation_colors`, `sampler`/`scheduler` (`default`, optional `choices`), and ranges (`min`, `max`, `default`, `step`) for `steps`, `cfg`, `denoise`, `lora_strength`. |

Tests: `PYTHONPATH=src python -m unittest discover -s tests/comfy_models`
loads every manifest it finds, builds workflows with 0, 1 and the maximum number of
reference images, and checks they are well formed. It cannot check the workflow
itself (a wrong node class or input name): only running it in ComfyUI does.
