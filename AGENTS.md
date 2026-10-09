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
  `src/`: `app.py` (entry point), `workflow_model.py` (the model),
  `workflows.py` (finding, reading and schema-checking workflow files),
  `sizes.py` (output-size policies), `comfyui/` (client, downloads, custom
  nodes, GPU arguments), `ui/` (the NiceGUI UI: `gui.py` is the page, with `login.py`,
  `components.py`, `mask_editor.py` and `compare_slider.py`; its CSS and JS are
  in `ui/static/`) and `core/` (types, imaging, paths, result cache). It is started with `python -m ai_image_edit`.
  Imports are absolute (`from ai_image_edit.core import …`). Dependencies
  live in `pyproject.toml`: everything the app needs (including NiceGUI and
  the ComfyUI client) is a regular dependency.
- **UI contract**: `ui/gui.py` exposes a single entry point,
  `run(model: WorkflowModel) -> None`, which `ui/__init__.py`'s `run()`
  calls (NiceGUI is imported only then). The model is passed as a plain
  argument — no shared mutable module state. The UI shows `model.display_name`.
- **Model contract**: the UI uses a `WorkflowModel` (`workflow_model.py`):
  `capabilities`, `start()`, `generate()`, `setup_progress()`, `shutdown()`,
  `list_loras()`, `model_name`, `model_version` and `display_name`.
  `get_model()` in the same module finds a workflow file by bundled name or
  by path.
- **Qwen-licensed code**: no code under the Qwen Research License lives in
  this repo; the Qwen-Image-2.1 weights are only downloaded, under their own
  license. Never copy Qwen-licensed code into this repo.
- **Models are data**: a model is one JSON file in
  `src/ai_image_edit/data/workflows/` (`format_version`, `manifest`, `workflow`; package
  data, installed with the app), run by the generic `WorkflowModel`; there is
  no per-model Python class. `get_model()` finds every file there
  by its name. The format is documented in
  `doc/contribution/workflow_files.md` and defined by `src/ai_image_edit/data/workflow.schema.json`.
  Output-size schemes (including the Qwen-Image-2.1 tiers) are in `sizes.py`;
  a model with a new one adds a policy class there.
- **ComfyUI code**: `comfyui/` holds the client (`client.py`), downloads,
  custom node installation and the GPU argument choice. Weight files are
  listed in the workflow file's `files` and downloaded when missing; an
  environment variable such as `MODEL_FILE` can point to a local file instead.
- **Default model**: `src/ai_image_edit/app.py` defaults `--workflow`/`MODEL_WORKFLOW` to
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
  `AI_IMAGE_EDIT_WORK_MAX_SIZE` through `trim_work_dir()`. ComfyUI keeps
  its input/output/temp folders under it too.
- **License headers**: this project is GPL-3.0-or-later. Every source
  file starts with `# SPDX-License-Identifier: GPL-3.0-or-later` (or the
  file-format-appropriate comment syntax) as its very first line, before
  any module docstring. Keep this on new files.

## Verifying changes

- After editing a Python file, confirm it still parses
  (`python -m py_compile <file>` or `ast.parse`) — several files start
  with a header comment followed immediately by a module docstring, and
  it's easy to break that pairing with a careless insertion.
- Tests live in `tests/` (no GPU, ComfyUI or network needed):
  `PYTHONPATH=src python -m unittest discover -s tests`. They cover the
  workflow files (schema and the code that runs them), downloads, custom
  node installation, the entry points, the GPU argument choice, the ComfyUI client, image compositing and the session secret. Run them
  after any change; for code they don't reach (the UI, the ComfyUI client),
  also read the affected code paths end to end.
