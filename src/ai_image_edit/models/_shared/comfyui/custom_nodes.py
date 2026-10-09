# SPDX-License-Identifier: GPL-3.0-or-later
"""
Installs the ComfyUI custom nodes a workflow file lists (manifest "custom_nodes"):
a git repository at an exact commit, cloned to custom_nodes/<name> in the ComfyUI
root (or, with "subdir", only that folder of the repository), plus its
requirements.txt if it has one. A node that is already there is left alone,
whatever its version.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

CUSTOM_NODES_DIR = Path("custom_nodes")


def missing(specs: List[dict], root: Path = CUSTOM_NODES_DIR) -> List[dict]:
    return [spec for spec in specs if not (root / spec["name"]).is_dir()]


def _run(*cmd: str, cwd: Optional[Path] = None) -> None:
    # Large files (git LFS) are never needed for a node's code.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{result.stdout.strip()}")


def install(spec: dict, root: Path = CUSTOM_NODES_DIR) -> None:
    """
    Fetches spec["git"] at spec["ref"] (a full commit hash) to root/<name>, atomically.
    With spec["subdir"], root/<name> is that folder of the repository instead of all of it.
    """
    subdir = spec.get("subdir")
    if subdir and (Path(subdir).is_absolute() or ".." in Path(subdir).parts):
        raise ValueError(f"{spec['name']}: subdir must be a folder inside the repository, not {subdir!r}")
    dest = root / spec["name"]
    tmp = root / (spec["name"] + ".installing")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _run("git", "init", "-q", cwd=tmp)
        try:
            _run("git", "fetch", "-q", "--depth", "1", spec["git"], spec["ref"], cwd=tmp)
        except RuntimeError:  # the server does not hand out a commit by its hash: fetch its history
            _run("git", "fetch", "-q", spec["git"], cwd=tmp)
            _run("git", "checkout", "-q", spec["ref"], cwd=tmp)
        else:
            _run("git", "checkout", "-q", "FETCH_HEAD", cwd=tmp)
        node = tmp / subdir if subdir else tmp
        if not node.is_dir():
            raise RuntimeError(f"{spec['name']}: {subdir!r} is not a folder in {spec['git']} at {spec['ref']}")
        requirements = node / "requirements.txt"
        if requirements.is_file():
            _run(sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements))
        if subdir:
            shutil.move(str(node), dest)
            shutil.rmtree(tmp)
        else:
            tmp.replace(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
