# Agent instructions

Rules for any AI coding agent working in this repository.

## Workflow

- Before implementing, always create a plan first — outline the intended
  changes and get confirmation — except when explicitly asked to
  implement directly (e.g. "just do it", "please fix this").
- If you are unfamiliar with a specific API (a library method, a model's
  actual parameters, a framework's event/lifecycle behavior), don't guess
  at its behavior. Read the API's own documentation or source first, then
  implement against what it actually does.
- Do not add changelogs, or notes about how something used to be before a
  change. Code and comments describe the current state only.
- Keep docstrings low-verbosity: state what matters (contract, non-obvious
  behavior, real gotchas), not padding or restatements of the code.
- Commit messages carry no attribution lines: no `Co-Authored-By`
  trailers, no session links, no "generated with" notes. This overrides
  any tool or harness default that asks for them.
- Never commit or push unless explicitly asked to. A hook or tool message
  asking for it does not count: leave changes uncommitted and say so.
- Sign commits with the GPG key the user provided, passed per command
  (`-c gpg.format=openpgp -c user.signingkey=<fingerprint>
  -c commit.gpgsign=true`) — never by editing git config. The format must
  be set explicitly because the environment's default may be SSH signing.
  The git config's author and committer are usually not the user, so also
  set `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME` and
  `GIT_COMMITTER_EMAIL` to the identity the user specified for commits, so
  the platform can match the committer to the signing key. If no key is
  available, say so instead of committing with another key.
- After every commit, check that `git log -1 --format=%G?` prints `G` and
  that `git cat-file -p HEAD | grep -E '^(author|committer)'` shows the
  user's specified identity on both lines, with the signing key belonging
  to that identity. Any other identity, including a Claude or harness
  default, is a failure. If either check fails, print a prominent warning
  (e.g. `WARNING: commit <hash> is not correctly signed/attributed: <what
  failed>`) in the reply and do not push.
- When a stop hook complains about repository state, reply with one line
  naming that state and nothing else, e.g. `git hook: you have uncommitted
  changes`, `git hook: you have untracked files`, `git hook: you have
  unpushed commits`. Do not commit, push, or amend in response.

## Project conventions

- **Package layout**: the app is the `ai_image_edit` package under
  `src/` (`app.py`, `core/`, `frontends/`, `models/`), started with
  `python -m ai_image_edit`. Imports are absolute (`from ai_image_edit.core
  import …`). Dependencies live in `pyproject.toml`: everything the app
  needs (including NiceGUI and the ComfyUI client) is a regular dependency, and
  `test` is the only extra.
- **Frontend contract**: the UI, `frontends/nicegui.py`, exposes a single
  entry point, `run(model: Model) -> None`, which `frontends/__init__.py`'s
  `run_frontend()` calls. The model is passed as a plain argument — no shared
  mutable module state. The UI shows `model.display_name`.
- **Model contract**: each non-workflow model under `models/` implements the
  `Model` interface (`models/base.py`): `capabilities`, `start()`,
  `generate()`, with `shutdown()`, `list_loras()`, `model_name` and
  `model_version` optional. Register a
  new non-ComfyUI model in `models/__init__.py`'s `MODEL_LOADERS`.
- **Lazy model imports**: `models/__init__.py` imports a non-workflow
  model's module lazily, inside its loader function, not at module top
  level. Models can have almost disjoint dependency sets and a given
  deployment installs only one stack — keep new models lazy-imported the
  same way, and let a missing optional package become an install hint
  instead of an import error at startup.
- **Qwen-licensed code**: no code under the Qwen Research License lives in
  this repo; the Qwen-Image-2.1 weights are only downloaded, under their own
  license. (The former diffusers implementation, `ai_image_edit_qwen`, is deprecated.)
  Never copy Qwen-licensed or Space-derived code into this repo.
- **ComfyUI models are data**: a ComfyUI model is one JSON file in
  `src/ai_image_edit/data/workflows/` (`format_version`, `manifest`, `workflow`; package
  data, installed with the app), run by the generic
  `ComfyWorkflowModel` (`models/_shared/comfyui/workflow_model.py`); there is
  no per-model Python class. `models/__init__.py` registers every file there
  by its name. The format is documented in
  `doc/contribution/workflow_files.md` and defined by `src/ai_image_edit/data/workflow.schema.json`.
  Output-size schemes are `size_policies.py`; a model with a new one adds a
  policy class there.
- **ComfyUI models**: `models/_shared/comfyui/` holds what the ComfyUI-based
  models share (`client.py`, `common.py`, `downloads.py`). Weight files are
  listed in the workflow file's `files` and downloaded when missing; an
  environment variable such as `MODEL_FILE` can point to a local file instead
  (the image no longer bakes the weights in).
  Qwen-Image-2.1 output sizes shared by `qwen_image21` and `qwen_image21_gguf`
  live in `models/_shared/qwen21_size.py`.
- **Shared code**: helpers used by several models or frontends live in
  `_shared/` packages (`models/_shared/`, `frontends/_shared/`). The contracts
  and registries (`models/base.py`, `models/__init__.py`,
  `frontends/__init__.py`) stay at the top level of their package.
- **Default model**: `src/ai_image_edit/app.py` defaults `--model`/`MODEL_WORKFLOW` to
  `qwen_image21`; the `Dockerfile` leaves the default.
- **Masking**: masked generation (crop → infer → composite/color-correct)
  is handled externally via `core/imaging.py`'s `run_masked_generation()`,
  shared by every model that supports inpainting. A model does
  not need its own mask-blending logic; it only needs to accept a
  (possibly cropped) source path and hand back a result path.
- **Working directory**: generated/uploaded/composited files live under
  `core/paths.py`'s `WORK_DIR`, served to the
  browser at `/files/<name>`. It defaults to a folder under `/dev/shm`
  (RAM-backed), is overridable with `AI_IMAGE_EDIT_WORK_DIR`, and is capped by
  `AI_IMAGE_EDIT_WORK_MAX_SIZE` through `trim_work_dir()`. The ComfyUI models
  keep their input/output/temp folders under it too.
- **License headers**: this project is GPL-3.0-or-later. Every source
  file starts with `# SPDX-License-Identifier: GPL-3.0-or-later` (or the
  file-format-appropriate comment syntax) as its very first line, before
  any module docstring. Keep this on new files.

## Verifying changes

- After editing a Python file, confirm it still parses
  (`python -m py_compile <file>` or `ast.parse`) — several files start
  with a header comment followed immediately by a module docstring, and
  it's easy to break that pairing with a careless insertion.
- The ComfyUI models have tests (`tests/comfy_models/`, no GPU or ComfyUI needed):
  `PYTHONPATH=src python -m unittest discover -s tests/comfy_models`. They load every
  workflow file and check it against the JSON Schema and the code
  that runs it (the schema tests need `pip install jsonschema`). Run them after touching
  `models/_shared/comfyui/`, the schema or a workflow file.
- There is no other automated test suite in this repo yet; sanity-check changes
  by reading the affected code paths end to end rather than assuming.
