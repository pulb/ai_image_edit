# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI wire protocol: launching the server process, uploading images,
submitting a workflow and waiting for it to finish, and fetching the
result.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Callable, List, Optional

import requests
import websocket

from ai_image_edit.comfy import gpu
from ai_image_edit.core.errors import GenerationCancelled, GenerationError
from ai_image_edit.core.paths import WORK_DIR
from ai_image_edit.core.types import GenerationProgress, ProgressStage

PORT = 8188
SERVER_ADDRESS = f"127.0.0.1:{PORT}"

# Seconds a single HTTP request to ComfyUI may take (the generation itself is
# waited for separately, see STALL_TIMEOUT_SECONDS).
REQUEST_TIMEOUT_SECONDS = 60

# Safety nets for submit_workflow_and_wait's websocket loop, so a ComfyUI job
# that stalls, fails without an execution_error message, or a websocket that
# silently drops cannot leave the app waiting forever.
# Seconds a running job may go without reporting anything (a node starting or
# a sampler step). A single node that reports nothing, such as loading the
# weights from a slow disk, must finish within it. COMFY_GENERATION_TIMEOUT,
# applied by read_settings().
STALL_TIMEOUT_SECONDS = 300
# Seconds a job may take in all, from submission, including the wait behind
# other jobs in ComfyUI's queue. COMFY_GENERATION_MAX_TIME, applied by read_settings().
MAX_GENERATION_SECONDS = 1800
# How often submit_workflow_and_wait checks whether it was cancelled.
CANCEL_POLL_SECONDS = 0.5


def _seconds(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"{name}={raw!r} is not a number of seconds.")
    return int(raw)


def read_settings() -> None:
    """Applies COMFY_GENERATION_TIMEOUT and COMFY_GENERATION_MAX_TIME; raises ValueError for an invalid value."""
    global STALL_TIMEOUT_SECONDS, MAX_GENERATION_SECONDS
    STALL_TIMEOUT_SECONDS = _seconds("COMFY_GENERATION_TIMEOUT", 300)
    MAX_GENERATION_SECONDS = _seconds("COMFY_GENERATION_MAX_TIME", 1800)


def _unreachable(error: Exception) -> GenerationError:
    return GenerationError(
        f"Cannot reach ComfyUI ({error}). ComfyUI may still be starting or has crashed. Please wait a moment and try again."
    )


def _stall_error() -> GenerationError:
    return GenerationError(
        f"ComfyUI reported no progress for {STALL_TIMEOUT_SECONDS}s — it may be stuck. Check the ComfyUI server logs."
    )


def _max_time_error() -> GenerationError:
    return GenerationError(
        f"The generation took longer than {MAX_GENERATION_SECONDS}s and was abandoned. Check the ComfyUI server logs."
    )


def launch_comfy_process(extra_args: Optional[List[str]] = None) -> subprocess.Popen:
    """
    Starts the ComfyUI server in the background with the given extra command
    line arguments (see gpu.py). -u forces unbuffered output, so startup
    progress isn't silently swallowed. Returns the Popen handle so the caller
    can terminate it.
    """
    print("Starting ComfyUI server in the background...", flush=True)
    # ComfyUI's input/output/temp folders live under WORK_DIR too, so
    # everything the app handles sits in one place (RAM-backed by default).
    comfy_dir = WORK_DIR.resolve() / "comfy"
    args = ["--port", str(PORT), *(extra_args or [])]
    for name in ("input", "output", "temp"):
        (comfy_dir / name).mkdir(parents=True, exist_ok=True)
        args += [f"--{name}-directory", str(comfy_dir / name)]
    return subprocess.Popen([sys.executable, "-u", "main.py", *args], env=gpu.environment())


def is_ready() -> bool:
    """True once the ComfyUI server answers requests."""
    try:
        return requests.get(f"http://{SERVER_ADDRESS}/system_stats", timeout=2).ok
    except requests.exceptions.RequestException:
        return False


def missing_nodes(class_names: List[str]) -> List[str]:
    """The node classes among `class_names` that the running ComfyUI does not know."""
    try:
        return [
            name for name in class_names
            if not requests.get(f"http://{SERVER_ADDRESS}/object_info/{name}", timeout=REQUEST_TIMEOUT_SECONDS).json()
        ]
    except (requests.exceptions.RequestException, ValueError) as e:
        raise _unreachable(e)


def upload_image(filepath: Optional[str]) -> Optional[str]:
    """Uploads an image file path to ComfyUI, returning the server-side filename."""
    if not filepath:
        return None
    try:
        with open(filepath, "rb") as f:
            files = {"image": f}
            res = requests.post(f"http://{SERVER_ADDRESS}/upload/image", files=files, timeout=REQUEST_TIMEOUT_SECONDS)
            res.raise_for_status()
            return res.json()["name"]
    except requests.exceptions.RequestException as e:
        raise _unreachable(e)


def _queue_prompt(workflow: dict, client_id: str) -> str:
    """Queues the workflow and returns its prompt_id; ComfyUI's own message if it refuses the workflow."""
    try:
        res = requests.post(
            f"http://{SERVER_ADDRESS}/prompt", json={"prompt": workflow, "client_id": client_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        body = res.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        raise GenerationError(f"Error sending prompt to ComfyUI: {e}")
    if "prompt_id" not in body:
        error = body.get("error")
        message = error.get("message", error) if isinstance(error, dict) else error
        details = [
            f"node {node}: {err.get('message', '')} {err.get('details', '')}".strip()
            for node, info in (body.get("node_errors") or {}).items() for err in info.get("errors", [])
        ]
        raise GenerationError(f"ComfyUI rejected the workflow: {'; '.join([str(message or res.status_code)] + details)}")
    return body["prompt_id"]


def cancel_prompt(prompt_id: str) -> None:
    """
    Stops one prompt: removes it from ComfyUI's queue if it still waits there,
    and interrupts it if it is running.

    Other users' and sessions' generations are not affected: the queue entry
    is deleted by its prompt_id, and the interrupt names the prompt_id too, so
    ComfyUI interrupts only if this prompt is the one running. ComfyUI's
    interrupt itself is a single global flag, though, which the running job
    picks up at its next check. If this prompt finishes in the instant between
    ComfyUI's check and the flag being set, and another prompt starts at once,
    that other prompt is interrupted instead. The window is very short, and the
    flag is cleared whenever a prompt starts, so it never outlives that case.
    Best effort: a ComfyUI that does not answer is ignored.
    """
    for path, body in (("queue", {"delete": [prompt_id]}), ("interrupt", {"prompt_id": prompt_id})):
        try:
            requests.post(f"http://{SERVER_ADDRESS}/{path}", json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as e:
            print(f"[cancel] POST /{path} failed: {e}", flush=True)


def submit_workflow_and_wait(
    workflow: dict,
    client_id: str,
    sampler_node: Optional[str] = None,
    on_progress: Optional[Callable[[GenerationProgress], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> str:
    """
    Submits a workflow to ComfyUI and blocks (via websocket) until this
    prompt finishes executing; returns the prompt_id. Raises GenerationError
    if submission fails, ComfyUI reports an error or an interruption, the
    running job reports nothing for STALL_TIMEOUT_SECONDS, or the whole job
    takes longer than MAX_GENERATION_SECONDS.

    on_progress, if given, is called (on this thread) whenever the stage
    changes and on every step of sampler_node; progress from other nodes is
    not reported as sampling. Without a sampler_node, any node's steps are.

    Setting cancel stops the prompt (see cancel_prompt()) within about
    CANCEL_POLL_SECONDS and raises GenerationCancelled.
    """
    last: List[Optional[GenerationProgress]] = [None]  # repeats are not reported

    def report(stage: ProgressStage, step: Optional[int] = None, total: Optional[int] = None) -> None:
        progress = GenerationProgress(stage, step, total)
        if on_progress is not None and progress != last[0]:
            last[0] = progress
            on_progress(progress)

    ws = websocket.WebSocket()
    # Connected before the prompt is queued: a job that finishes at once (everything
    # cached) would otherwise report its end before anyone is listening.
    try:
        ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    except (websocket.WebSocketException, OSError) as e:
        ws.close()
        raise _unreachable(e)
    try:
        prompt_id = _queue_prompt(workflow, client_id)
        give_up_at = time.time() + MAX_GENERATION_SECONDS
        # None while the job waits in the queue: a wait behind another job is
        # only limited by MAX_GENERATION_SECONDS.
        stall_at: Optional[float] = None
        sampled = False
        report(ProgressStage.PREPARING)

        while True:
            if cancel is not None and cancel.is_set():
                cancel_prompt(prompt_id)
                raise GenerationCancelled("Generation cancelled.")
            now = time.time()
            if now >= give_up_at:
                raise _max_time_error()
            if stall_at is not None and now >= stall_at:
                raise _stall_error()
            # Re-armed every iteration to the time actually left, so a stream
            # of unrelated messages cannot stretch the wait past either limit.
            wait = min(give_up_at, stall_at or give_up_at) - now
            ws.settimeout(min(wait, CANCEL_POLL_SECONDS) if cancel is not None else wait)

            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue  # the checks at the top of the loop say which limit it was

            if not isinstance(out, str):
                continue  # binary frames are previews
            message = json.loads(out)
            data = message.get("data")
            if not isinstance(data, dict):
                continue
            if message.get("type") == "status" and stall_at is None:
                # Broadcast whenever the queue changes; it counts the running job and
                # the waiting ones, so more than this one means it waits its turn.
                remaining = (data.get("status") or {}).get("exec_info", {}).get("queue_remaining")
                if isinstance(remaining, int) and remaining > 1:
                    report(ProgressStage.QUEUED)
                continue
            if data.get("prompt_id") != prompt_id:
                continue
            stall_at = time.time() + STALL_TIMEOUT_SECONDS
            msg_type = message.get("type")

            if msg_type == "execution_start":
                report(ProgressStage.PREPARING)
            elif msg_type == "progress":
                if sampler_node is None or str(data.get("node")) == sampler_node:
                    sampled = True
                    report(ProgressStage.SAMPLING, data.get("value"), data.get("max"))
            elif msg_type == "executing":
                if data.get("node") is None:
                    break
                if sampled and str(data["node"]) != sampler_node:
                    report(ProgressStage.FINISHING)
            elif msg_type == "execution_error":
                raise GenerationError(f"ComfyUI Error: {data.get('exception_message', 'Unknown error')}")
            elif msg_type == "execution_interrupted":
                raise GenerationError("The generation was interrupted in ComfyUI.")
    except GenerationError:
        # A real execution failure or timeout — surface it, don't let the
        # narrower except below swallow it.
        raise
    except (websocket.WebSocketException, OSError) as e:
        # Genuinely flaky websocket/network trouble (connection reset,
        # protocol error, ...) — the prompt may still have completed on the
        # ComfyUI side, so don't fail outright; let the history fetch
        # afterward be the final word. Deliberately narrower than a bare
        # `except Exception`: a bug in *this* loop (e.g. malformed JSON
        # from json.loads, a missing key in message["data"]) should not be
        # swallowed and reported as "just a flaky websocket" — it should
        # propagate so it's actually visible and fixed.
        print(f"Websocket error: {e}", flush=True)
    finally:
        ws.close()

    return prompt_id


def fetch_generated_image(prompt_id: str) -> str:
    """
    Downloads the first output image from a completed ComfyUI prompt's
    history to a local path. Raises GenerationError if none is found.
    """
    try:
        history = requests.get(f"http://{SERVER_ADDRESS}/history/{prompt_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()
        outputs = history[prompt_id]["outputs"]

        for node_id in outputs:
            if "images" in outputs[node_id]:
                img_data = outputs[node_id]["images"][0]
                query = urllib.parse.urlencode({k: img_data[k] for k in ("filename", "subfolder", "type")})
                output_path = WORK_DIR / f"output_{prompt_id}.png"
                res = requests.get(f"http://{SERVER_ADDRESS}/view?{query}", timeout=REQUEST_TIMEOUT_SECONDS)
                res.raise_for_status()
                output_path.write_bytes(res.content)
                return str(output_path)
    except (requests.RequestException, OSError, ValueError, LookupError) as e:
        raise GenerationError(f"Error retrieving generated image: {e}")

    raise GenerationError("ComfyUI finished but produced no output image.")
