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
import time
import urllib.parse
from typing import List, Optional

import requests
import websocket

from ai_image_edit.comfyui import gpu
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR

PORT = 8188
SERVER_ADDRESS = f"127.0.0.1:{PORT}"

# Seconds a single HTTP request to ComfyUI may take (the generation itself is
# waited for separately, see GENERATION_TIMEOUT_SECONDS).
REQUEST_TIMEOUT_SECONDS = 60

# Safety net for submit_workflow_and_wait's websocket loop — without this, a
# ComfyUI job that stalls, fails without emitting a proper execution_error
# message, or a websocket connection that silently drops would leave the
# app hanging forever with no feedback.
# COMFY_GENERATION_TIMEOUT overrides the default (seconds); slower GPUs need more.
GENERATION_TIMEOUT_SECONDS = int(os.environ.get("COMFY_GENERATION_TIMEOUT", "120"))


def _unreachable(error: Exception) -> GenerationError:
    return GenerationError(
        f"Cannot reach ComfyUI ({error}). ComfyUI may still be starting or has crashed. Please wait a moment and try again."
    )


def _timeout_error() -> GenerationError:
    return GenerationError(
        f"Generation timed out after {GENERATION_TIMEOUT_SECONDS}s waiting for ComfyUI — it may be stuck or stalled. Check the ComfyUI server logs."
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


def submit_workflow_and_wait(workflow: dict, client_id: str) -> str:
    """
    Submits a workflow to ComfyUI and blocks (via websocket) until this
    prompt finishes executing, up to GENERATION_TIMEOUT_SECONDS. Raises
    GenerationError if submission fails, ComfyUI reports an execution error,
    or the wait times out; returns the prompt_id on success.
    """
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
        deadline = time.time() + GENERATION_TIMEOUT_SECONDS

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise _timeout_error()

            # Re-armed every iteration to the time actually left, not the
            # full budget — a fixed settimeout(GENERATION_TIMEOUT_SECONDS)
            # here would let recv() block for a fresh full timeout on every message
            # (e.g. a stream of ComfyUI "progress" events), so a job that
            # keeps sending *something* without ever finishing could stall
            # far past the documented timeout before the deadline check
            # above ever caught it.
            ws.settimeout(remaining)

            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                raise _timeout_error()

            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get("type")

                if msg_type == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break

                elif msg_type == "execution_error":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        raise GenerationError(f"ComfyUI Error: {data.get('exception_message', 'Unknown error')}")
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
