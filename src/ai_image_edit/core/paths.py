# SPDX-License-Identifier: GPL-3.0-or-later
"""
Shared filesystem/URL plumbing for the app's runtime working directory.

Owned by the UI/app layer, not by any model: it answers "where do
generated/uploaded files live so the browser can load them", which model
backends are simply handed plain local paths for and never need a URL
scheme for.
"""
from pathlib import Path

# Where uploaded, generated, and composited images live for the lifetime of
# the process. Shared by both frontends (gradio_ui.py exposes it through
# allowed_paths=[str(WORK_DIR)]). Served to the browser at /files/<name>
# (see frontends/nicegui.py's app.add_static_files) so client-side
# <canvas>/<img> elements can load them by URL. This also doubles as the
# single source of truth for "what's the current result" — see
# create_compare_slider()'s get_after_path() in frontends/nicegui.py.
WORK_DIR = Path("app_work")
WORK_DIR.mkdir(exist_ok=True)


def to_url(path_str: str) -> str:
    """Converts a local WORK_DIR file path into the URL it's served at."""
    return f"/files/{Path(path_str).name}"


def from_url(url: str) -> str:
    """Inverse of to_url() — recovers the local WORK_DIR path from a (possibly absolute) served URL."""
    filename = url.rstrip("/").split("/")[-1]
    return str(WORK_DIR / filename)
