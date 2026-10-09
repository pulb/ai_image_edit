# SPDX-License-Identifier: GPL-3.0-or-later
"""
Finding and reading workflow files (ai_image_edit/data/workflows/*.json, format
described in doc/contribution/workflow_files.md). The files are package data, so
they are installed with the app. Standard library only, so the available
files can be listed without loading anything heavy.
"""
import json
from pathlib import Path
from typing import Dict, Tuple

FORMAT_VERSION = 1

# The workflow files shipped with the app, inside the installed package.
PACKAGE_DIR = Path(__file__).resolve().parents[3] / "data" / "workflows"


def available_workflows() -> Dict[str, Path]:
    """Workflow name (the file name without .json) -> workflow file."""
    return {p.stem: p for p in sorted(PACKAGE_DIR.glob("*.json"))}


def read_workflow_file(path: Path) -> Tuple[dict, dict]:
    """The (manifest, workflow) of a workflow file; rejects files of another format version."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("format_version") if isinstance(data, dict) else None
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format_version {version!r} (this app reads version {FORMAT_VERSION})")
    for key in ("manifest", "workflow"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"{path}: missing {key!r} section")
    return data["manifest"], data["workflow"]
