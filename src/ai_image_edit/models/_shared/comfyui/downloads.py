# SPDX-License-Identifier: GPL-3.0-or-later
"""
Downloads the weight files a workflow file lists (manifest "files"), so a model
can be started on a machine that does not have them yet.

Files are fetched to "<name>.part" next to their destination, resumed with an
HTTP Range request if a previous attempt was cut off, verified (if the manifest
gives a size and/or sha256) and only then renamed, so a file under its final
name is always complete.
"""
import hashlib
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


def _headers(url: str) -> Dict[str, str]:
    """HF_TOKEN is only ever sent to Hugging Face."""
    host = (urlparse(url).hostname or "").lower()
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and (host in ("huggingface.co", "hf.co") or host.endswith(".huggingface.co")):
        return {"Authorization": f"Bearer {token}"}
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str, dest: Path, *, sha256: Optional[str] = None, size: Optional[int] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> None:
    """Downloads url to dest (no-op if dest exists). progress(done, total) is called per chunk."""
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(2):
        done = part.stat().st_size if part.exists() else 0
        headers = _headers(url)
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416 and done:  # the partial file is already everything (or stale)
                    part.unlink()
                    continue
                if r.status_code in (401, 403):
                    raise DownloadError(
                        f"{url}: access denied (HTTP {r.status_code}). If the file needs a login, set HF_TOKEN."
                    )
                if r.status_code >= 400:
                    raise DownloadError(f"{url}: HTTP {r.status_code}")
                if done and r.status_code != 206:  # the server ignored Range: start over
                    done = 0
                length = int(r.headers.get("Content-Length", 0)) or None
                total = size or (done + length if length else None)
                with open(part, "ab" if done else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise DownloadCancelled("download cancelled")
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
        except requests.RequestException as exc:
            raise DownloadError(f"{url}: {exc} (run again to resume)") from exc
        break

    actual = part.stat().st_size
    if size is not None and actual != size:
        part.unlink()
        raise DownloadError(f"{dest.name}: expected {size} bytes, got {actual}")
    if sha256 and _sha256(part) != sha256.lower():
        part.unlink()
        raise DownloadError(f"{dest.name}: sha256 does not match the manifest")
    os.replace(part, dest)


class DownloadStatus:
    """Thread-safe progress of the background download, shown to the user while it runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._message = ""
        self._error: Optional[str] = None
        self.finished = threading.Event()

    def set_message(self, message: str) -> None:
        with self._lock:
            self._message = message

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error
        self.finished.set()

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error


def run_downloads(
    items: List[dict], status: DownloadStatus, cancel: threading.Event,
    on_done: Optional[Callable[[], None]] = None,
) -> None:
    """
    items: [{"url", "dest": Path, "sha256", "size"}]. Calls on_done after the last file, then
    sets status.finished (also set on failure, with status.error). Meant to run in a thread.
    """
    try:
        for index, item in enumerate(items, 1):
            label = f"{item['dest'].name} ({index}/{len(items)})"
            last = [-1]

            def progress(done: int, total: Optional[int], label: str = label) -> None:
                pct = int(done * 100 / total) if total else None
                if pct is None or pct != last[0]:
                    last[0] = pct if pct is not None else last[0]
                    gb = done / 1e9
                    status.set_message(f"Downloading {label}: {pct}%" if pct is not None else f"Downloading {label}: {gb:.1f} GB")

            status.set_message(f"Downloading {label}")
            download_file(item["url"], item["dest"], sha256=item.get("sha256"), size=item.get("size"),
                          progress=progress, cancel=cancel)
        status.set_message("")
        if on_done:
            on_done()
        status.finished.set()
    except DownloadCancelled:
        status.fail("download cancelled")
    except Exception as exc:  # reported to the user by the model, not lost in the thread
        status.fail(str(exc))
