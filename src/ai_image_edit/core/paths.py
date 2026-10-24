# SPDX-License-Identifier: GPL-3.0-or-later
"""
Shared filesystem/URL plumbing for the app's runtime working directory:
where uploaded and generated files live so the browser can load them.
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

# Where uploaded, generated, and composited images live for the lifetime of
# the process. Served to the browser at /files/<name>
# (see ui/gui.py's app.add_static_files) so client-side
# <canvas>/<img> elements can load them by URL.
#
# Defaults to a folder under /dev/shm (RAM-backed, so images never reach
# persistent storage) and falls back to ./app_work where /dev/shm is not
# usable. AI_IMAGE_EDIT_WORK_DIR overrides it. AI_IMAGE_EDIT_WORK_MAX_SIZE is
# the cap trim_work_dir() enforces, itself limited by the size of the
# filesystem (e.g. docker run --shm-size); read_settings() applies it.
_SHM = Path("/dev/shm")


def _default_work_dir() -> Path:
    if _SHM.is_dir() and os.access(_SHM, os.W_OK):
        return _SHM / "ai_image_edit"
    return Path("app_work")


WORK_DIR = Path(os.environ.get("AI_IMAGE_EDIT_WORK_DIR") or _default_work_dir())
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WORK_MAX_SIZE = "512m"
_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_size(text: str) -> int:
    """Bytes for '16g', '512m', '1.5g' or a plain byte count."""
    text = text.strip().lower().rstrip("b")
    if text and text[-1] in _UNITS:
        return int(float(text[:-1]) * _UNITS[text[-1]])
    return int(text)


def format_size(n: int) -> str:
    """A byte count as text, e.g. '512.0 MiB'."""
    n = max(0, n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _effective_max_bytes() -> int:
    """
    The cap trim_work_dir() enforces: AI_IMAGE_EDIT_WORK_MAX_SIZE (default
    512m), limited to 90% of the filesystem WORK_DIR is on — /dev/shm is often
    far smaller than expected (Docker's default is 64 MB) and a full one makes
    writes fail before any cap is reached. 0 means "as much as the
    filesystem allows". Prints a warning when the setting had to be lowered.
    """
    raw = os.environ.get("AI_IMAGE_EDIT_WORK_MAX_SIZE", DEFAULT_WORK_MAX_SIZE)
    try:
        configured = parse_size(raw or "0")
    except ValueError:
        raise ValueError(f"AI_IMAGE_EDIT_WORK_MAX_SIZE={raw!r} is not a size such as 512m or 2g.") from None
    limit = int(shutil.disk_usage(WORK_DIR).total * 0.9)
    if configured <= 0:
        return limit
    if configured > limit:
        print(
            f"WARNING: AI_IMAGE_EDIT_WORK_MAX_SIZE ({format_size(configured)}) is larger than "
            f"90% of the filesystem holding {WORK_DIR}; the effective cap is {format_size(limit)}. "
            "Enlarge the filesystem (docker run --shm-size) or lower the setting.",
            flush=True,
        )
    return min(configured, limit)


WORK_MAX_BYTES = parse_size(DEFAULT_WORK_MAX_SIZE)


def read_settings() -> None:
    """Applies AI_IMAGE_EDIT_WORK_MAX_SIZE (see _effective_max_bytes()); raises ValueError for an invalid value."""
    global WORK_MAX_BYTES
    WORK_MAX_BYTES = _effective_max_bytes()


def trim_work_dir(keep: int = 0, protect: Iterable[str] = ()) -> None:
    """
    Deletes the oldest files under WORK_DIR (ComfyUI's folders included)
    until the total is within WORK_MAX_BYTES. The `keep` newest files are
    never deleted, so the files of the generation in progress survive even
    a tiny cap, and neither are the `protect`ed paths, the files the page
    still uses (its input image, however old, and the result on show). They
    count towards the total, so the folder stays above the cap if nothing
    else is left to delete.
    """
    protected = {Path(p).resolve() for p in protect}
    files = []
    for p in WORK_DIR.rglob("*"):
        try:
            if p.is_file():
                st = p.stat()
                files.append((st.st_mtime, st.st_size, p))
        except OSError:
            continue
    files.sort(key=lambda f: f[0])  # oldest first
    total = sum(size for _, size, _ in files)
    for _, size, p in files[:max(0, len(files) - keep)]:
        if total <= WORK_MAX_BYTES:
            break
        if p.resolve() in protected:
            continue
        try:
            p.unlink()
            total -= size
        except OSError:
            continue


def new_work_file(prefix: str, suffix: str = ".png") -> Path:
    """A fresh, unused path in WORK_DIR: `<prefix>_<random hex><suffix>`."""
    return WORK_DIR / f"{prefix}_{uuid.uuid4().hex}{suffix}"


def to_url(path_str: str) -> str:
    """Converts a local WORK_DIR file path into the URL it's served at."""
    return f"/files/{quote(Path(path_str).name, safe='')}"
