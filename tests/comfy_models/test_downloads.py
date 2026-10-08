# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tests the weight downloads against a local HTTP server (no internet):

    PYTHONPATH=src python -m unittest discover -s tests/comfy_models -v
"""
import hashlib
import os
import sys
import tempfile
import threading
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
from ai_image_edit.models._shared.comfyui import client, downloads
from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class Handler(BaseHTTPRequestHandler):
    hits = []
    gate = None  # an Event the response waits for, to keep a download "in progress"

    def log_message(self, *args):
        pass

    def do_GET(self):
        Handler.hits.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
        if self.path.startswith("/missing"):
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if Handler.gate is not None:
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
        Handler.hits, Handler.gate = [], None

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

    def test_resumes_partial_file(self):
        dest = self.tmp / "b.bin"
        dest.with_name("b.bin.part").write_bytes(PAYLOAD[:5000])
        downloads.download_file(self.url + "/f", dest, sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertEqual(Handler.hits[0][1], "bytes=5000-")

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
        Handler.gate.set()
        self.assertTrue(model._download_status.finished.wait(10))
        self.assertIsNone(model._download_status.error)
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
