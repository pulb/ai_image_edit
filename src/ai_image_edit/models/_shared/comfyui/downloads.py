# SPDX-License-Identifier: GPL-3.0-or-later
"""
Downloads the weight files a workflow file lists (manifest "files"), so a model
can be started on a machine that does not have them yet.

Files are fetched to "<name>.part" next to their destination, in pieces over
several connections at once (HTTP Range requests; DOWNLOAD_CONNECTIONS, default 8),
verified (if the manifest gives a size and/or sha256) and only then renamed, so a file
under its final name is always complete. A download that was cut off continues with
the pieces that are missing.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

CHUNK = 1024 * 1024
PART_SIZE = 64 * 1024 * 1024  # a file is fetched in pieces of this size, several at a time


class DownloadError(RuntimeError):
    fatal = False  # True: trying again cannot help (a refused or missing file)


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


def _connections() -> int:
    """How many parts of a file are fetched at once (DOWNLOAD_CONNECTIONS, default 8, at most 32)."""
    try:
        return max(1, min(32, int(os.environ.get("DOWNLOAD_CONNECTIONS", "8"))))
    except ValueError:
        return 8


def _check_status(url: str, status: int) -> None:
    if status in (401, 403):
        error = DownloadError(f"{url}: access denied (HTTP {status}). If the file needs a login, set HF_TOKEN.")
    elif status >= 400:
        error = DownloadError(f"{url}: HTTP {status}")
    else:
        return
    error.fatal = status < 500
    raise error


def _probe(url: str) -> Optional[int]:
    """The file's size if the server supports Range requests, else None."""
    try:
        with requests.get(url, headers={**_headers(url), "Range": "bytes=0-0"}, stream=True, timeout=60) as r:
            _check_status(url, r.status_code)
            if r.status_code != 206:
                return None
            total = r.headers.get("Content-Range", "").rpartition("/")[2]
            return int(total) if total.isdigit() else None
    except requests.RequestException as exc:
        raise DownloadError(f"{url}: {exc} (run again to resume)") from exc


def remote_size(url: str) -> Optional[int]:
    """The file's size according to the server, or None if it does not say (or cannot be reached)."""
    try:
        return _probe(url)
    except DownloadError:
        return None


class _Progress:
    """Bytes downloaded so far, shared by the worker threads; reports whole percents."""

    def __init__(self, total: Optional[int], done: int, callback: Optional[Callable[[int, Optional[int]], None]]) -> None:
        self._lock = threading.Lock()
        self._total, self._done, self._callback = total, done, callback

    def add(self, n: int) -> None:
        with self._lock:
            self._done += n
            if self._callback:
                self._callback(self._done, self._total)

    def subtract(self, n: int) -> None:  # a part is fetched again
        with self._lock:
            self._done -= n


def _fetch_part(url: str, part: Path, start: int, end: int, progress: _Progress, cancel: Optional[threading.Event]) -> None:
    """Writes bytes start..end (inclusive) of url into the same offsets of `part`, retrying a few times."""
    fetched = 0
    for attempt in range(4):
        try:
            with requests.get(url, headers={**_headers(url), "Range": f"bytes={start}-{end}"},
                              stream=True, timeout=60) as r:
                _check_status(url, r.status_code)
                if r.status_code != 206:
                    raise DownloadError(f"{url}: the server stopped honouring Range requests")
                with open(part, "r+b") as f:
                    f.seek(start)
                    for data in r.iter_content(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise DownloadCancelled("download cancelled")
                        f.write(data)
                        fetched += len(data)
                        progress.add(len(data))
            if fetched != end - start + 1:
                raise DownloadError(f"{url}: received {fetched} of {end - start + 1} bytes")
            return
        except DownloadCancelled:
            raise
        except (requests.RequestException, DownloadError) as exc:
            if getattr(exc, "fatal", False):
                raise
            progress.subtract(fetched)
            fetched = 0
            if attempt == 3:
                raise DownloadError(f"{url}: {exc} (run again to resume)") from exc
            time.sleep(2 ** attempt)


def _download_parts(
    url: str, part: Path, total: int, progress: Optional[Callable[[int, Optional[int]], None]],
    cancel: Optional[threading.Event],
) -> None:
    """
    Fetches `total` bytes in PART_SIZE pieces, several at a time, into one file. The pieces
    that are complete are recorded next to it ("<name>.part.json"), so an interrupted
    download continues with the missing ones.
    """
    state_path = part.with_name(part.name + ".json")
    count = -(-total // PART_SIZE)
    done_parts = set()
    try:
        state = json.loads(state_path.read_text())
        if part.is_file() and state["total"] == total and state["part_size"] == PART_SIZE:
            done_parts = {i for i in state["done"] if 0 <= i < count}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not done_parts:
        with open(part, "wb") as f:
            f.truncate(total)
    todo = [i for i in range(count) if i not in done_parts]
    already = sum(min((i + 1) * PART_SIZE, total) - i * PART_SIZE for i in done_parts)
    shared = _Progress(total, already, progress)
    lock = threading.Lock()
    failure: List[BaseException] = []

    def save_state() -> None:
        state_path.write_text(json.dumps({"total": total, "part_size": PART_SIZE, "done": sorted(done_parts)}))

    save_state()

    def worker() -> None:
        while True:
            with lock:
                if failure or not todo:
                    return
                i = todo.pop(0)
            try:
                _fetch_part(url, part, i * PART_SIZE, min((i + 1) * PART_SIZE, total) - 1, shared, cancel)
            except BaseException as exc:  # reported once, after the other workers have stopped
                with lock:
                    failure.append(exc)
                return
            with lock:
                done_parts.add(i)
                save_state()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(_connections(), len(todo)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failure:
        raise failure[0]
    state_path.unlink(missing_ok=True)


def _download_single(
    url: str, part: Path, progress: Optional[Callable[[int, Optional[int]], None]],
    cancel: Optional[threading.Event],
) -> None:
    """One plain request, for servers without Range support. Cannot be resumed."""
    try:
        with requests.get(url, headers=_headers(url), stream=True, timeout=60) as r:
            _check_status(url, r.status_code)
            total = int(r.headers.get("Content-Length", 0)) or None
            done = 0
            with open(part, "wb") as f:
                for data in r.iter_content(CHUNK):
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled("download cancelled")
                    f.write(data)
                    done += len(data)
                    if progress:
                        progress(done, total)
    except requests.RequestException as exc:
        raise DownloadError(f"{url}: {exc}") from exc


def download_file(
    url: str, dest: Path, *, sha256: Optional[str] = None, size: Optional[int] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> None:
    """
    Downloads url to dest (no-op if dest exists), over several connections if the server
    allows ranges. progress(done, total) is called as data arrives, from any thread.
    """
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    total = _probe(url)
    if total:
        _download_parts(url, part, total, progress, cancel)
    else:
        part.unlink(missing_ok=True)
        _download_single(url, part, progress, cancel)

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
        self._fraction: Optional[float] = None
        self.finished = threading.Event()

    def set_fraction(self, fraction: Optional[float]) -> None:
        with self._lock:
            self._fraction = fraction

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
    def fraction(self) -> Optional[float]:
        """How much of all downloads is done (0 to 1), or None while that is not known."""
        with self._lock:
            return self._fraction

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
    status.fraction follows the bytes of all files together, if every file's size is known
    (from the item or the server).
    """
    try:
        status.set_message("Checking the file sizes")
        sizes = [item.get("size") or remote_size(item["url"]) for item in items]
        grand_total = sum(sizes) if items and all(sizes) else None
        completed = 0
        for index, item in enumerate(items, 1):
            label = f"{item['dest'].name} ({index}/{len(items)})"
            last = [-1]

            def progress(done: int, total: Optional[int], label: str = label, before: int = completed) -> None:
                if grand_total:
                    status.set_fraction(min(1.0, (before + done) / grand_total))
                pct = int(done * 100 / total) if total else None
                if pct is None or pct != last[0]:
                    last[0] = pct if pct is not None else last[0]
                    gb = done / 1e9
                    status.set_message(f"Downloading {label}: {pct}%" if pct is not None else f"Downloading {label}: {gb:.1f} GB")

            status.set_message(f"Downloading {label}")
            download_file(item["url"], item["dest"], sha256=item.get("sha256"), size=item.get("size"),
                          progress=progress, cancel=cancel)
            completed += sizes[index - 1] or 0
        status.set_message("")
        if on_done:
            on_done()
        status.finished.set()
    except DownloadCancelled:
        status.fail("download cancelled")
    except Exception as exc:  # reported to the user by the model, not lost in the thread
        status.fail(str(exc))
