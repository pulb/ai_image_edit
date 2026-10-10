# Workflow files

A model is one JSON file in `src/ai_image_edit/data/workflows/`; no Python is
needed. The files are package data, so they are installed with the app. The file name is the
model name (`--workflow <name>` or `MODEL_WORKFLOW=<name>`), and the file is found automatically. A file anywhere else is run by giving its path instead of a name; its name is then the file name without `.json`.

```json
{
  "format_version": 1,
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
| `model_version` | Optional. The version shown after the name, e.g. `"UC Q4_K_M"`. The `MODEL_VERSION` environment variable overrides it (for weights chosen with the `env` overrides). |
| `vram_headroom_gb` | Optional number. GPU memory in GB the generation needs on top of the model files; the app starts ComfyUI with `--highvram --disable-dynamic-vram` if the free GPU memory covers the files plus this headroom. Default 6. See `deployments/docker/DEPLOY.md`. |
| `custom_nodes` | `[{name, git, ref, subdir}]`: ComfyUI custom nodes installed to `custom_nodes/<name>` if that folder is missing, from the `https://` git repository at the exact commit `ref` (40 hex digits), plus its `requirements.txt`. With `subdir`, only that folder of the repository is installed (as `custom_nodes/<name>`). This is how a patched fork is referenced. An existing folder is left as it is. |
| `variants` | `{id: {model_version, files: {<set target>: {name, url, sha256, size, license}}}}`: alternatives selected with `--variant id` / `MODEL_VARIANT`. A variant replaces the keys it gives in the `files` entry with that `set` target, and the `model_version`. If it gives a `name` or `url`, the entry's `sha256` and `size` are dropped unless the variant gives its own. |
| `required_nodes`, `missing_nodes_hint` | Node classes ComfyUI must provide, checked on the first generation, and the advice shown if they are missing. |
| `files` | Weight files, each written to `set`. `name` is the file name relative to `folder` (a folder below the ComfyUI root, e.g. `models/vae`) and `url` where to get it from (`https://`); optional `sha256` and `size` are verified after the download, `license` names an entry of `licenses`. `env` optionally names an environment variable that replaces the file: its value is then used as it is and nothing is downloaded (without `name`, the variable is required). See [Downloads](#downloads). |
| `licenses` | `{id: {name, url}}` for the licenses the files refer to. |
| `bind` | Where each app input goes: `prompt`, `seed`, `steps`, `sampler`, `scheduler` (required), `cfg`, `denoise`, `negative_prompt` (optional). An input that is not bound is not offered in the UI (`cfg`, `denoise`, `negative_prompt`). |
| `constants` | Fixed values written into the workflow: `{target: value}`. |
| `seed` | `min`, `max` of random seeds; `wrap` reduces a chosen seed modulo `max + 1`. |
| `images` | `source`: target for the source image. `references`: further images, each a `LoadImage` node wired to `target` (`{n}` counts from `first_n`), with node ids from `loader_ids` (a list) or `loader_id_start`, up to `max`. Unused slots in the template are removed. |
| `size` | `policy` (see `sizes.py`: `qwen21_tiers`, `fixed_area`), its settings, and `bind`, which maps the policy's named values to targets: `resolution`, `width` and `height` for `qwen21_tiers` (width and height go to an `EmptyLatentImage` that feeds the sampler, so the output size does not depend on the source image), `width` and `height` for `fixed_area`. |
| `loras` | Omit if the model has none. Chains one `loader_class` node per selected file from `first_id`; `from` are the links it starts from, `outputs` the names of its outputs, `feeds` the targets that receive the final links, `strength_inputs` the inputs that get the strength. `dir` is searched for LoRA files. `files` (`[{dir, name, url, sha256, size, license}]`) are LoRA files downloaded like `files` above if missing: `dir` is the subfolder of the LoRA (its display name; one LoRA can have several files), `name` the file. A LoRA downloaded in the background is added to the UI list as soon as its files are there. |
| `capabilities` | What the UI offers: `supports_inpainting`, `num_annotation_colors`, `sampler`/`scheduler` (`default`, optional `choices`), and ranges (`min`, `max`, `default`, `step`) for `steps`, `cfg`, `denoise`, `lora_strength`. |

## Tests

`PYTHONPATH=src python -m unittest discover -s tests` checks every file in
`src/ai_image_edit/data/workflows/` against the schema and loads it, builds workflows with 0, 1 and the
maximum number of reference images, and checks they are well formed. The tests cannot check the workflow
itself (a wrong node class or input name): only running it in ComfyUI does.

## Downloads

When the app starts a model, every file in `files` that is not in its folder yet is downloaded
in the background; ComfyUI is launched when the last one is complete, and generating meanwhile
fails with a message that says what is being downloaded. Custom nodes are installed after the downloads, before ComfyUI launches. A failed download or
install is reported the same way and retried on the next start.

- Files are fetched to `<name>.part` in 64 MB pieces, 8 at a time (`DOWNLOAD_CONNECTIONS`, at
  most 32; a server that does not support range requests gets one connection), and renamed when
  complete. An interrupted download continues with the missing pieces, and a file under its real
  name is never partial.
- Folders are relative to the directory the app runs in (the ComfyUI root). To keep the weights
  elsewhere, mount a volume at `<ComfyUI root>/models`.
- `HF_TOKEN` is sent as a bearer token to huggingface.co only.
- A file with a `license` is only downloaded if `ACCEPT_LICENSES` (comma-separated ids, or `all`)
  lists it; otherwise starting the model fails with the license's name and URL. Read the license
  and set the variable only if you accept it.
- `sha256` and `size` are optional; use them for files you have checked, so a corrupt or replaced
  file is rejected.
