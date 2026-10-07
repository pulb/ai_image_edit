# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI wire protocol: launching the server process, uploading images,
submitting a workflow and waiting for it to finish, and fetching the
result.

Private to qwen_image_edit_comfy — nothing outside models/qwen_image_edit_comfy/
imports this. Any future ComfyUI-based model can reuse it directly; a model
like qwen_image that doesn't run through ComfyUI never touches it, and
never has to.
"""
import json
import subprocess
import time
import urllib.request
from typing import Optional

import requests
import websocket

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.paths import WORK_DIR

SERVER_ADDRESS = "127.0.0.1:8188"

# Safety net for submit_workflow_and_wait's websocket loop — without this, a
# ComfyUI job that stalls, fails without emitting a proper execution_error
# message, or a websocket connection that silently drops would leave the
# app hanging forever with no feedback.
GENERATION_TIMEOUT_SECONDS = 120


def launch_comfy_process() -> subprocess.Popen:
    """
    Starts the ComfyUI server in the background. -u forces unbuffered
    output, so startup progress isn't silently swallowed. Returns the Popen
    handle so the caller (QwenImageEditComfyModel.shutdown) can terminate it.
    """
    print("Starting ComfyUI server in the background...", flush=True)
    # ComfyUI's input/output/temp folders live under WORK_DIR too, so
    # everything the app handles sits in one place (RAM-backed by default).
    comfy_dir = WORK_DIR.resolve() / "comfy"
    args = ["--port", "8188"]
    for name in ("input", "output", "temp"):
        (comfy_dir / name).mkdir(parents=True, exist_ok=True)
        args += [f"--{name}-directory", str(comfy_dir / name)]
    return subprocess.Popen(["python", "-u", "main.py", *args])


def upload_image(filepath: Optional[str]) -> Optional[str]:
    """Uploads an image file path to ComfyUI, returning the server-side filename."""
    if not filepath:
        return None
    try:
        with open(filepath, "rb") as f:
            files = {"image": f}
            res = requests.post(f"http://{SERVER_ADDRESS}/upload/image", files=files)
            res.raise_for_status()
            return res.json()["name"]
    except requests.exceptions.RequestException as e:
        raise GenerationError(
            f"Cannot reach ComfyUI ({e}). The backend may still be starting or has crashed. Please wait a moment and try again."
        )


def submit_workflow_and_wait(workflow: dict, client_id: str) -> str:
    """
    Submits a workflow to ComfyUI and blocks (via websocket) until this
    prompt finishes executing, up to GENERATION_TIMEOUT_SECONDS. Raises
    GenerationError if submission fails, ComfyUI reports an execution error,
    or the wait times out; returns the prompt_id on success.
    """
    payload = {"prompt": workflow, "client_id": client_id}
    try:
        response = requests.post(f"http://{SERVER_ADDRESS}/prompt", json=payload).json()
        prompt_id = response["prompt_id"]
    except Exception as e:
        raise GenerationError(f"Error sending prompt to ComfyUI: {e}")

    ws = websocket.WebSocket()
    try:
        ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}")
        deadline = time.time() + GENERATION_TIMEOUT_SECONDS

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise GenerationError(
                    f"Generation timed out after {GENERATION_TIMEOUT_SECONDS}s waiting for ComfyUI — it may be stuck or stalled. Check the ComfyUI server logs."
                )

            # Re-armed every iteration to the time actually left, not the
            # full budget — a fixed settimeout(GENERATION_TIMEOUT_SECONDS)
            # here would let recv() block for a fresh 120s on every message
            # (e.g. a stream of ComfyUI "progress" events), so a job that
            # keeps sending *something* without ever finishing could stall
            # far past the documented timeout before the deadline check
            # above ever caught it.
            ws.settimeout(remaining)

            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                raise GenerationError(
                    f"Generation timed out after {GENERATION_TIMEOUT_SECONDS}s waiting for ComfyUI — it may be stuck or stalled. Check the ComfyUI server logs."
                )

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
        history = requests.get(f"http://{SERVER_ADDRESS}/history/{prompt_id}").json()
        outputs = history[prompt_id]["outputs"]

        for node_id in outputs:
            if "images" in outputs[node_id]:
                img_data = outputs[node_id]["images"][0]
                img_url = f"http://{SERVER_ADDRESS}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
                output_path = WORK_DIR / f"output_{prompt_id}.png"
                urllib.request.urlretrieve(img_url, output_path)
                return str(output_path)
    except Exception as e:
        raise GenerationError(f"Error retrieving generated image: {e}")

    raise GenerationError("ComfyUI finished but produced no output image.")
