# SPDX-License-Identifier: GPL-3.0-or-later
"""
Installs the ComfyUI custom nodes a workflow file lists (manifest "custom_nodes"):
a git repository at an exact commit, cloned to custom_nodes/<name> in the ComfyUI
root, plus its requirements.txt if it has one. A node that is already there is
left alone, whatever its version.
"""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List

CUSTOM_NODES_DIR = Path("custom_nodes")


def missing(specs: List[dict], root: Path = CUSTOM_NODES_DIR) -> List[dict]:
    return [spec for spec in specs if not (root / spec["name"]).is_dir()]


def _run(*cmd: str, cwd: Path = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{result.stdout.strip()}")


def install(spec: dict, root: Path = CUSTOM_NODES_DIR) -> None:
    """Clones spec["git"] at spec["ref"] (a full commit hash) to root/<name>, atomically."""
    dest = root / spec["name"]
    tmp = root / (spec["name"] + ".installing")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _run("git", "init", "-q", cwd=tmp)
        _run("git", "fetch", "-q", "--depth", "1", spec["git"], spec["ref"], cwd=tmp)
        _run("git", "checkout", "-q", "FETCH_HEAD", cwd=tmp)
        requirements = tmp / "requirements.txt"
        if requirements.is_file():
            _run(sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements))
        tmp.replace(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
