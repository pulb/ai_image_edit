# Workflow files

A model is one JSON file in `src/ai_image_edit/data/workflows/`; no Python is
needed. The files are package data, so they are installed with the app. The file name is the
model name (`--workflow <name>` or `MODEL_WORKFLOW=<name>`), and the file is found automatically. A file anywhere else is run by giving its path instead of a name; its name is then the file name without `.json`.

```json
{
  "format_version": 2,
  "manifest": { ... how the app's inputs map onto the workflow ... },
  "workflow": { ... the workflow, exported from ComfyUI with "Save (API Format)" ... }
}
```

The structure is defined by [`workflow.schema.json`](../../src/ai_image_edit/data/workflow.schema.json) (JSON
Schema), which the app checks every file against when it loads it (and the tests check the bundled ones). `format_version` changes only when the
format changes incompatibly; the app refuses files of another version.

## Targets

A **target** is `"<node id>.<input name>"`, e.g. `"6.seed"`. The input name may contain
dots (`"5.images.image_2"`). Every target the app fills in (everything except the reference
image slots) must already exist in the `workflow` section with a placeholder value, so a
mistyped node id or input name fails when the model loads.

## Manifest

| Key | Meaning |
|---|---|
| `model_name` | Name shown in the UI. |
| `model_version` | Optional. The version shown after the name, e.g. `"UC Q4_K_M"`. |
| `vram_headroom_gb` | Optional number. GPU memory in GB the generation needs on top of the model files; the app starts ComfyUI with `--highvram --disable-dynamic-vram` if the free GPU memory covers the files plus this headroom. Default 6. See `deployments/docker/DEPLOY.md`. |
| `custom_nodes` | `[{name, git, ref, subdir}]`: ComfyUI custom nodes installed to `custom_nodes/<name>` if that folder is missing, from the `https://` git repository at the exact commit `ref` (40 hex digits), plus its `requirements.txt`. With `subdir`, only that folder of the repository is installed (as `custom_nodes/<name>`). This is how a patched fork is referenced. An existing folder is left as it is. |
| `variants` | `{id: {model_version, files: {<set target>: {name, url, sha256, size, license}}}}`: alternatives selected with `--variant id` / `MODEL_VARIANT`. A variant replaces the keys it gives in the `files` entry with that `set` target, and the `model_version`. If it gives a `name` or `url`, the entry's `sha256` and `size` are dropped unless the variant gives its own. |
| `required_nodes`, `missing_nodes_hint` | Node classes ComfyUI must provide, checked on the first generation, and the advice shown if they are missing. |
| `files` | Weight files, each written to `set`. `name` is the file name relative to `folder` (a folder below the ComfyUI root, e.g. `models/vae`) and `url` where to get it from (`https://`); optional `sha256` and `size` are verified after the download, `license` names an entry of `licenses`. See [Downloads](#downloads). |
| `licenses` | `{id: {name, url}}` for the licenses the files refer to. |
| `bind` | Where the inputs every model has go: `prompt`, `seed` and `steps`, all required. The progress bar follows the node `steps` is bound to. |
| `inputs` | Optional. The model's own further inputs, shown in Advanced settings in the order given: `{name: {target, range, choices, label}}`, see [Optional inputs](#optional-inputs). |
| `constants` | Fixed values written into the workflow: `{target: value}`. |
| `seed` | `min`, `max` of random seeds; `wrap` reduces a chosen seed modulo `max + 1`. |
| `images` | `source`: target for the source image. `references`: further images, each a `LoadImage` node wired to `target` (`{n}` counts from `first_n`), with node ids from `loader_ids` (a list) or `loader_id_start`, up to `max`. Unused slots in the template are removed. |
| `size` | `policy` (see `sizes.py`: `qwen21_tiers`, `fixed_area`), its settings, and `bind`, which maps the policy's named values to targets: `resolution`, `width` and `height` for `qwen21_tiers` (width and height go to an `EmptyLatentImage` that feeds the sampler, so the output size does not depend on the source image; the app scales every input image itself and sends `resolution` 0, so the encode node keeps their sizes: where the source keeps its framing (a mask, the "Original" ratio, or a chosen ratio within 2% of its own) it is fitted to exactly that size, otherwise it keeps its own ratio at the tier and the model recomposes it), `width` and `height` for `fixed_area`. A resolution is the side length in pixels of a square with the area to generate (1024 is "1K"); `fixed_area` takes `resolutions` (e.g. `[1024]`), an optional `default_resolution` and `multiple`. |
| `loras` | Omit if the model has none. Chains one `loader_class` node per selected file from `first_id`; `from` are the links it starts from, `outputs` the names of its outputs, `feeds` the targets that receive the final links, `strength_inputs` the inputs that get the strength. `dir` is searched for LoRA files. `files` (`[{dir, name, url, sha256, size, license}]`) are LoRA files downloaded like `files` above if missing: `dir` is the subfolder of the LoRA (its display name; one LoRA can have several files), `name` the file. A LoRA downloaded in the background is added to the UI list as soon as its files are there. `DOWNLOAD_LORAS=0` skips these downloads. |
| `capabilities` | What the UI offers: `supports_inpainting`, `num_annotation_colors`, and ranges (`min`, `max`, `default`, `step`) for `steps` and `lora_strength`. |

## Optional inputs

`inputs` lists the workflow inputs a model offers beyond the three in `bind`,
such as a CFG scale, a negative prompt or the sampler; the app knows nothing
about them except what the entry says:

```json
"inputs": {
  "cfg":             {"target": "6.cfg", "range": {"min": 1.0, "max": 10.0, "default": 1.0, "step": 0.1}, "label": "CFG scale"},
  "negative_prompt": {"target": "5.negative_prompt"},
  "sampler":         {"target": "6.sampler_name", "choices": "comfy_samplers"},
  "scheduler":       {"target": "6.scheduler", "choices": "comfy_schedulers"}
}
```

- An input with a `range` (`min`, `max`, `default`, optional `step`) is a
  slider. One with `choices` is a dropdown: `comfy_samplers` and
  `comfy_schedulers` are the names ComfyUI's KSampler takes, or the entry
  gives its own list. One with neither is a text field. The dropdown and the
  text field start with the workflow's placeholder at `target`.
- The placeholder must be a number for a slider, text for a text field, and
  one of the choices for a dropdown, or the model does not load. An entry
  cannot have both a `range` and `choices`.
- An input the file leaves out, such as a sampler the model needs fixed,
  keeps the workflow's own value.
- A slider's value is a whole number when the range and the placeholder are
  all whole numbers, otherwise a decimal.
- `label` is optional: without it the name is shown with spaces for
  underscores and a capital first letter (`negative_prompt` is "Negative
  prompt"). Names are lower case letters, digits and underscores, and must
  not be one of the three `bind` names.

## Converting a workflow from the editor

The `workflow` section is the API format ("Save (API Format)"). A workflow saved with plain "Save"
can be converted with `tools/convert_workflow.py`. It needs the node definitions to name the values
the editor stores by position; those of ComfyUI's built-in nodes come with it, from the ComfyUI commit
the Docker image pins:

```
python tools/convert_workflow.py my_workflow.json -o model.json
```

A workflow with custom nodes, or one saved in a ComfyUI whose built-in nodes differ, needs the
definitions of a ComfyUI that has its nodes installed (its URL, or its `/object_info` saved to a file):
`--object-info http://127.0.0.1:8188`. When the Docker image moves to another ComfyUI commit, renew the
bundled ones from that ComfyUI's `/object_info`:
`--update-builtin-nodes object_info.json --comfyui-ref <commit>` (a test checks they match the pin).

It writes a workflow file whose manifest binds the first sampler, the prompt its conditioning comes
from, the first `LoadImage` and an empty latent image (as a `fixed_area` size), offers the sampler's
CFG, denoise, sampler and scheduler and the negative prompt as [optional inputs](#optional-inputs), and lists the files
the model loaders name, with an empty `url` (`https://`). That is a starting point: the notes it prints
say what is left (download URLs and licenses, custom nodes, reference images, LoRAs). `--api-only`
writes just the graph. Subgraphs are not supported.

## Tests

`PYTHONPATH=src python -m unittest discover -s tests` checks every file in
`src/ai_image_edit/data/workflows/` against the schema and loads it, builds workflows with 0, 1 and the
maximum number of reference images, and checks they are well formed. The tests cannot check the workflow
itself (a wrong node class or input name): only running it in ComfyUI does.

## Downloads

When the app starts a model, every file in `files` that is not in its folder yet is downloaded
in the background; ComfyUI is launched when the last one is complete, and until then the Generate button
stays off while the page says what is being downloaded. Custom nodes are installed after the downloads, before ComfyUI launches. A failed download or
install is reported the same way and retried on the next start.

- Files are fetched to `<name>.part` in 64 MB pieces, 16 at a time (`DOWNLOAD_CONNECTIONS`, at
  most 32; a server that does not support range requests gets one connection), and renamed when
  complete. An interrupted download continues with the missing pieces, and a file under its real
  name is never partial.
- Folders are relative to the directory the app runs in (the ComfyUI root). To keep the weights
  elsewhere, mount a volume at `<ComfyUI root>/models`.
- `HF_TOKEN` is sent as a bearer token to huggingface.co only.
- A file's `license` names an entry of `licenses`; it is documentation, the download does not
  depend on it. Say in the workflow's documentation when a license limits its use.
- `sha256` and `size` are optional; use them for files you have checked, so a corrupt or replaced
  file is rejected.
