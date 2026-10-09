# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tests the weight downloads against a local HTTP server (no internet):

    PYTHONPATH=src python -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    import websocket  # noqa: F401
except ImportError:
    sys.modules["websocket"] = types.ModuleType("websocket")

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import GenerationParams
from ai_image_edit.models.base import SetupProgress
from ai_image_edit.models._shared.comfyui import client, downloads
from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class Handler(BaseHTTPRequestHandler):
    hits = []
    gate = None  # an Event the response waits for, to keep a download "in progress"
    ranges = True  # False: behave like a server that ignores Range
    fail_once = set()  # range starts whose first request is answered with a 500
    delay = 0.0  # seconds each ranged response takes
    active = peak = 0  # requests being served at once, and the most there have been
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        Handler.hits.append((self.path, rng, self.headers.get("Authorization")))
        if self.path.startswith("/missing"):
            self.send_error(404)
            return
        if rng and Handler.ranges and rng != "bytes=0-0":
            with Handler.lock:
                Handler.active += 1
                Handler.peak = max(Handler.peak, Handler.active)
            time.sleep(Handler.delay)
            with Handler.lock:
                Handler.active -= 1
        if rng and Handler.ranges:
            first, _, last = rng.split("=")[1].partition("-")
            first, last = int(first), int(last) if last else len(PAYLOAD) - 1
            if first in Handler.fail_once:
                Handler.fail_once.discard(first)
                self.send_error(500)
                return
            body = PAYLOAD[first:last + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{first + len(body) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if Handler.gate is not None and len(body) > 1:
            self.wfile.write(body[:1000])
            self.wfile.flush()
            Handler.gate.wait(10)
            self.wfile.write(body[1000:])
        else:
            self.wfile.write(body)


class ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        Handler.hits, Handler.gate, Handler.ranges, Handler.fail_once = [], None, True, set()
        Handler.delay, Handler.active, Handler.peak = 0.0, 0, 0
        patch = mock.patch.object(downloads, "PART_SIZE", 100_000)  # ~11 parts of the test payload
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()



class DownloadTests(ServerCase):
    def test_download_and_verify(self):
        dest = self.tmp / "a/b.bin"
        downloads.download_file(self.url + "/f", dest, sha256=hashlib.sha256(PAYLOAD).hexdigest(), size=len(PAYLOAD))
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertFalse(dest.with_name("b.bin.part").exists())

    def test_existing_file_is_not_downloaded_again(self):
        dest = self.tmp / "b.bin"
        dest.write_bytes(b"x")
        downloads.download_file(self.url + "/f", dest)
        self.assertEqual(Handler.hits, [])

    def test_file_is_fetched_in_parts_over_several_connections(self):
        dest = self.tmp / "b.bin"
        downloads.download_file(self.url + "/f", dest, sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        ranges = [h[1] for h in Handler.hits if h[1] != "bytes=0-0"]
        self.assertEqual(len(ranges), -(-len(PAYLOAD) // 100_000))
        self.assertEqual(sorted(downloads.os.listdir(self.tmp)), ["b.bin"])  # no .part or .json left

    def test_parts_are_fetched_at_the_same_time(self):
        Handler.delay = 0.2
        with mock.patch.dict(os.environ, {"DOWNLOAD_CONNECTIONS": "4"}):
            downloads.download_file(self.url + "/f", self.tmp / "b.bin")
        self.assertEqual(Handler.peak, 4)

    def test_progress_reaches_the_total(self):
        seen = []
        downloads.download_file(self.url + "/f", self.tmp / "b.bin", progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(max(seen), (len(PAYLOAD), len(PAYLOAD)))

    def test_continues_with_the_missing_parts(self):
        dest, part = self.tmp / "b.bin", self.tmp / "b.bin.part"
        with open(part, "wb") as f:  # the first three parts are there, the rest is zeros
            f.write(PAYLOAD[:300_000])
            f.truncate(len(PAYLOAD))
        part.with_name("b.bin.part.json").write_text(
            json.dumps({"total": len(PAYLOAD), "part_size": 100_000, "done": [0, 1, 2]}))
        downloads.download_file(self.url + "/f", dest, sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        starts = {int(h[1].split("=")[1].split("-")[0]) for h in Handler.hits if h[1] != "bytes=0-0"}
        self.assertEqual(min(starts), 300_000)

    def test_stale_partial_file_starts_over(self):
        part = self.tmp / "b.bin.part"
        part.write_bytes(b"junk")
        downloads.download_file(self.url + "/f", self.tmp / "b.bin")
        self.assertEqual((self.tmp / "b.bin").read_bytes(), PAYLOAD)

    def test_a_failed_part_is_retried(self):
        Handler.fail_once = {200_000}
        with mock.patch.object(downloads.time, "sleep"):
            downloads.download_file(self.url + "/f", self.tmp / "b.bin", sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual((self.tmp / "b.bin").read_bytes(), PAYLOAD)

    def test_server_without_range_support_is_fetched_in_one_piece(self):
        Handler.ranges = False
        downloads.download_file(self.url + "/f", self.tmp / "b.bin", sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual((self.tmp / "b.bin").read_bytes(), PAYLOAD)

    def test_wrong_checksum_or_size_leaves_no_file(self):
        dest = self.tmp / "b.bin"
        with self.assertRaisesRegex(downloads.DownloadError, "sha256"):
            downloads.download_file(self.url + "/f", dest, sha256="0" * 64)
        with self.assertRaisesRegex(downloads.DownloadError, "expected 5 bytes"):
            downloads.download_file(self.url + "/f", dest, size=5)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_http_error(self):
        with self.assertRaisesRegex(downloads.DownloadError, "404"):
            downloads.download_file(self.url + "/missing", self.tmp / "b.bin")
        self.assertEqual(len(Handler.hits), 1)  # not retried

    def test_token_only_goes_to_huggingface(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "secret"}):
            self.assertEqual(downloads._headers("https://huggingface.co/x"), {"Authorization": "Bearer secret"})
            self.assertEqual(downloads._headers("https://cdn-lfs.huggingface.co/x"), {"Authorization": "Bearer secret"})
            self.assertEqual(downloads._headers("https://evilhuggingface.co/x"), {})
            self.assertEqual(downloads._headers(self.url + "/f"), {})


class ModelStartTests(ServerCase):
    def model(self, **file_extra) -> ComfyWorkflowModel:
        from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows, read_workflow_file
        manifest, workflow = read_workflow_file(available_workflows()["qwen_image_edit_2511_aio"])
        manifest["loras"]["files"] = []
        manifest["custom_nodes"] = []
        manifest["licenses"] = {"lic": {"name": "Test License", "url": "https://example.org/lic"}}
        manifest["files"] = [dict(
            name="w.safetensors", url=self.url + "/f", folder="models/checkpoints", set="1.ckpt_name",
            env="TEST_WEIGHTS", **file_extra,
        )]
        return ComfyWorkflowModel("test", manifest, workflow)

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("TEST_WEIGHTS", None)
        os.environ.pop("ACCEPT_LICENSES", None)
        launch = mock.patch.object(client, "launch_comfy_process", return_value=mock.Mock())
        self.launch = launch.start()
        self.addCleanup(launch.stop)

    def params(self):
        return mock.Mock(spec=GenerationParams)  # the download check comes before any use

    def test_missing_file_is_downloaded_then_comfy_starts(self):
        model = self.model()
        Handler.gate = threading.Event()
        model.start()
        self.launch.assert_not_called()
        with self.assertRaisesRegex(GenerationError, "still being set up"):
            model.generate(self.params())
        deadline = time.time() + 10
        while "w.safetensors" not in model.setup_progress().message and time.time() < deadline:
            time.sleep(0.01)
        progress = model.setup_progress()
        self.assertFalse(progress.done)
        self.assertIn("w.safetensors", progress.message)
        self.assertLess(progress.fraction or 0, 1.0)
        Handler.gate.set()
        self.assertTrue(model._download_status.finished.wait(10))
        self.assertIsNone(model._download_status.error)
        final = model.setup_progress()
        self.assertTrue(final.done)
        self.assertEqual((final.message, final.error, final.fraction), ("", None, 1.0))
        self.assertEqual((self.tmp / "models/checkpoints/w.safetensors").read_bytes(), PAYLOAD)
        self.launch.assert_called_once()
        model._check_downloads()  # no longer raises

    def test_nothing_downloaded_if_file_exists(self):
        (self.tmp / "models/checkpoints").mkdir(parents=True)
        (self.tmp / "models/checkpoints/w.safetensors").touch()
        self.model().start()
        self.assertEqual(Handler.hits, [])
        self.launch.assert_called_once()

    def test_env_overrides_name_and_must_exist(self):
        os.environ["TEST_WEIGHTS"] = "mine.safetensors"
        with self.assertRaisesRegex(RuntimeError, "File not found"):
            self.model().start()
        (self.tmp / "models/checkpoints").mkdir(parents=True)
        (self.tmp / "models/checkpoints/mine.safetensors").touch()
        self.model().start()
        self.assertEqual(Handler.hits, [])

    def test_license_must_be_accepted(self):
        model = self.model(license="lic")
        with self.assertRaisesRegex(RuntimeError, r"ACCEPT_LICENSES=lic.*example.org/lic|example.org/lic.*ACCEPT_LICENSES=lic"):
            model.start()
        self.assertEqual(Handler.hits, [])
        os.environ["ACCEPT_LICENSES"] = "other, lic"
        model.start()
        self.assertTrue(model._download_status.finished.wait(10))
        self.assertIsNone(model._download_status.error)

    def test_failed_download_is_reported(self):
        model = self.model()
        model._manifest["files"][0]["url"] = self.url + "/missing"
        model.start()
        model._download_status.finished.wait(10)
        self.launch.assert_not_called()
        with self.assertRaisesRegex(GenerationError, "failed.*404"):
            model.generate(self.params())
        progress = model.setup_progress()
        self.assertTrue(progress.done)
        self.assertRegex(progress.error, "failed.*404")

    def test_fraction_covers_all_files_together(self):
        status = downloads.DownloadStatus()
        seen = []
        original = status.set_fraction
        status.set_fraction = lambda value: (seen.append(value), original(value))
        items = [
            {"url": self.url + "/f", "dest": self.tmp / "a.bin"},
            {"url": self.url + "/f", "dest": self.tmp / "b.bin", "size": len(PAYLOAD)},
        ]
        downloads.run_downloads(items, status, threading.Event())
        self.assertTrue(status.finished.is_set() and status.error is None)
        self.assertEqual(seen, sorted(seen))
        self.assertTrue(0 < seen[0] < 0.5)
        self.assertAlmostEqual(seen[-1], 1.0)

    def test_fraction_is_unknown_if_the_server_gives_no_size(self):
        Handler.ranges = False
        status = downloads.DownloadStatus()
        downloads.run_downloads([{"url": self.url + "/f", "dest": self.tmp / "a.bin"}], status, threading.Event())
        self.assertTrue(status.finished.is_set() and status.error is None)
        self.assertIsNone(status.fraction)

    def test_no_setup_means_done(self):
        self.assertEqual(self.model().setup_progress(), SetupProgress(done=True))

    def test_missing_lora_folder_is_warned_about_once(self):
        model = self.model()
        model._manifest["loras"]["dir"] = str(self.tmp / "nowhere")
        with mock.patch("builtins.print") as printed:
            model.list_loras()
            model.list_loras()
        self.assertEqual(printed.call_count, 1)

    def test_lora_files_are_downloaded_into_their_folders(self):
        model = self.model()
        model._manifest["files"] = []
        model._manifest["loras"]["files"] = [
            {"dir": "Pack", "name": "a b.safetensors", "url": self.url + "/f"},
            {"dir": "Pack", "name": "c.safetensors", "url": self.url + "/f"},
        ]
        model.start()
        self.assertTrue(model._download_status.finished.wait(10))
        self.assertIsNone(model._download_status.error)
        self.assertEqual(sorted(model.list_loras()), ["Pack"])
        self.assertEqual(model.list_loras()["Pack"], ["Pack/a b.safetensors", "Pack/c.safetensors"])

    def test_unknown_license_is_rejected_at_load(self):
        with self.assertRaisesRegex(ValueError, "unknown license"):
            self.model(license="nope")


if __name__ == "__main__":
    unittest.main()
