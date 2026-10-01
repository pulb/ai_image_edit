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
  Check the result with `git log -1 --format=%G?`; if no key is available,
  say so instead of committing with another key. Never re-author or amend
  commits to satisfy a hook.
- When a stop hook complains about repository state, reply with one line
  naming that state and nothing else, e.g. `git hook: you have uncommitted
  changes`, `git hook: you have untracked files`, `git hook: you have
  unpushed commits`. Do not commit, push, or amend in response.

## Project conventions

- **Frontend contract**: each module under `frontends/` exposes a single
  entry point, `run(model: ModelBackend, model_backend: str) -> None`.
  Model and backend name are passed as plain arguments — no shared
  mutable module state. Register a new frontend in
  `frontends/__init__.py`'s `FRONTEND_LOADERS`.
- **Model contract**: each backend under `models/` implements the
  `ModelBackend` interface (`models/base.py`): `capabilities`, `start()`,
  `generate()`, with `shutdown()` and `list_loras()` optional. Register a
  new backend in `models/__init__.py`'s `MODEL_LOADERS`.
- **Lazy model imports**: `models/__init__.py` imports each backend's
  module lazily, inside its loader function, not at module top level.
  The two shipped backends (`qwen_image`, `qwen_image_edit_comfy`) have
  almost disjoint dependency sets, and a given deployment only installs
  one of them — keep new backends lazy-imported the same way.
- **Masking**: masked generation (crop → infer → composite/color-correct)
  is handled externally via `core/imaging.py`'s `run_masked_generation()`,
  shared by every backend that supports inpainting. A model backend does
  not need its own mask-blending logic; it only needs to accept a
  (possibly cropped) source path and hand back a result path.
- **Working directory**: generated/uploaded/composited files live under
  `core/paths.py`'s `WORK_DIR`, shared by both frontends, served to the
  browser at `/files/<name>`.
- **License headers**: this project is GPL-3.0-or-later. Every source
  file starts with `# SPDX-License-Identifier: GPL-3.0-or-later` (or the
  file-format-appropriate comment syntax) as its very first line, before
  any module docstring. Keep this on new files.

## Verifying changes

- After editing a Python file, confirm it still parses
  (`python -m py_compile <file>` or `ast.parse`) — several files start
  with a header comment followed immediately by a module docstring, and
  it's easy to break that pairing with a careless insertion.
- There is no automated test suite in this repo yet; sanity-check changes
  by reading the affected code paths end to end rather than assuming.
