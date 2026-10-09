# SPDX-License-Identifier: GPL-3.0-or-later
"""
Finding and reading workflow files (ai_image_edit/data/workflows/*.json, format
described in doc/contribution/workflow_files.md). The files are package data, so
they are installed with the app. Reading a file checks it against the JSON Schema
(ai_image_edit/data/workflow.schema.json).
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import jsonschema

FORMAT_VERSION = 1

# The workflow files shipped with the app, inside the installed package.
PACKAGE_DIR = Path(__file__).resolve().parent / "data" / "workflows"
SCHEMA_PATH = PACKAGE_DIR.parent / "workflow.schema.json"


def available_workflows() -> Dict[str, Path]:
    """Workflow name (the file name without .json) -> workflow file."""
    return {p.stem: p for p in sorted(PACKAGE_DIR.glob("*.json"))}


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def read_workflow_file(path: Path) -> Tuple[dict, dict]:
    """The (manifest, workflow) of a workflow file; rejects files of another format version or that do not match the schema."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("format_version") if isinstance(data, dict) else None
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format_version {version!r} (this app reads version {FORMAT_VERSION})")
    for key in ("manifest", "workflow"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"{path}: missing {key!r} section")
    errors = sorted(_validator().iter_errors(data), key=lambda e: [str(part) for part in e.path])
    if errors:
        details = "\n".join(f"  {'/'.join(map(str, e.path)) or '(top level)'}: {e.message}" for e in errors[:10])
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(f"{path}: does not match the workflow file schema:\n{details}{more}")
    return data["manifest"], data["workflow"]
