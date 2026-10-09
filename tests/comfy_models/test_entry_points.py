# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tests the command line / environment selection of the model, workflow variants,
the version override and the custom node installation (against a local git
repository, no network):

    PYTHONPATH=src python -m unittest discover -s tests/comfy_models -v
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import websocket  # noqa: F401
except ImportError:
    sys.modules["websocket"] = types.ModuleType("websocket")

from ai_image_edit import app
from ai_image_edit.models._shared.comfyui import client, custom_nodes
from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows, read_workflow_file
from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

GGUF = available_workflows()["qwen_image21_gguf"]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("MODEL_WORKFLOW", "MODEL_VARIANT", "REQUIRE_PASSWORD", "MODEL_VERSION"):
            os.environ.pop(name, None)
        self.model = mock.Mock()
        self.run = mock.patch.object(app, "run_frontend").start()
        self.get = mock.patch.object(app, "get_model", return_value=self.model).start()
        self.addCleanup(mock.patch.stopall)

    def test_default_workflow(self):
        app.main([])
        self.get.assert_called_once_with("qwen_image21", None)
        self.model.start.assert_called_once()
        self.run.assert_called_once_with(self.model)

    def test_environment_is_the_fallback(self):
        os.environ.update(MODEL_WORKFLOW="x", MODEL_VARIANT="Q8_0")
        app.main([])
        self.get.assert_called_once_with("x", "Q8_0")
        self.run.assert_called_once_with(self.model)

    def test_command_line_beats_environment(self):
        os.environ.update(MODEL_WORKFLOW="x", MODEL_VARIANT="a")
        app.main(["--workflow", "y", "--variant", "b"])
        self.get.assert_called_once_with("y", "b")

    def test_list_workflows(self):
        with mock.patch("builtins.print") as out:
            app.main(["--list-workflows"])
        self.assertEqual(out.call_args[0][0].split("\n"), sorted(available_workflows()))
        self.get.assert_not_called()

    def test_bad_workflow_is_a_clean_exit(self):
        self.get.side_effect = ValueError("Unknown workflow 'z'")
        with self.assertRaisesRegex(SystemExit, "Unknown workflow"):
            app.main(["--workflow", "z"])


class GetModelTests(unittest.TestCase):
    def test_bundled_name_and_file_path(self):
        from ai_image_edit.models import get_model

        with mock.patch.object(ComfyWorkflowModel, "from_file") as from_file:
            get_model("qwen_image21_gguf", "Q8_0")
            from_file.assert_called_with(available_workflows()["qwen_image21_gguf"], "Q8_0")
            get_model(str(GGUF))
            from_file.assert_called_with(GGUF, None)

    def test_unknown_workflow_lists_the_bundled_ones(self):
        from ai_image_edit.models import get_model

        with self.assertRaisesRegex(ValueError, "Unknown workflow 'nope'. Available: qwen_image21,"):
            get_model("nope")


class VariantTests(unittest.TestCase):
    def test_variant_replaces_file_and_version(self):
        model = ComfyWorkflowModel.from_file(GGUF, "Q8_0")
        unet = model._manifest["files"][0]
        self.assertEqual(unet["name"], "qwen-image-2.1-UC-Q8_0.gguf")
        self.assertTrue(unet["url"].endswith("qwen-image-2.1-UC-Q8_0.gguf?download=true"))
        self.assertEqual(unet["license"], "qwen-research")  # keys the variant does not give are kept
        self.assertEqual(model.model_version, "UC Q8_0")
        self.assertEqual(ComfyWorkflowModel.from_file(GGUF).model_version, "UC Q4_K_M")

    def test_variants_do_not_change_the_shared_manifest(self):
        manifest, workflow = read_workflow_file(GGUF)
        before = copy.deepcopy(manifest)
        ComfyWorkflowModel("x", manifest, workflow, "Q8_0")
        self.assertEqual(manifest, before)

    def test_unknown_variant_lists_the_options(self):
        with self.assertRaisesRegex(ValueError, r"unknown variant 'nope' \(available: Q4_0"):
            ComfyWorkflowModel.from_file(GGUF, "nope")
        with self.assertRaisesRegex(ValueError, "available: none"):
            ComfyWorkflowModel.from_file(available_workflows()["qwen_image_edit_2511_aio"], "x")

    def test_variant_must_change_an_existing_file(self):
        manifest, workflow = read_workflow_file(GGUF)
        manifest["variants"]["bad"] = {"files": {"9.nothing": {"name": "x"}}}
        with self.assertRaisesRegex(ValueError, "not in files"):
            ComfyWorkflowModel("x", manifest, workflow)

    def test_version_environment_override(self):
        with mock.patch.dict(os.environ, {"MODEL_VERSION": "custom"}):
            self.assertEqual(ComfyWorkflowModel.from_file(GGUF).model_version, "custom")


class CustomNodeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        repo = self.tmp / "repo"
        repo.mkdir()
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-C", str(repo)]
        subprocess.run([*git, "init", "-q"], check=True)
        (repo / "node.py").write_text("one")
        (repo / "pack").mkdir()
        (repo / "pack/inner.py").write_text("inner")
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "commit", "-qm", "one"], check=True)
        self.first = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        (repo / "node.py").write_text("two")
        subprocess.run([*git, "commit", "-qam", "two"], check=True)
        self.spec = {"name": "My-Node", "git": str(repo), "ref": self.first}
        self.root = self.tmp / "custom_nodes"
        self.root.mkdir()

    def test_installs_the_pinned_commit(self):
        self.assertEqual(custom_nodes.missing([self.spec], self.root), [self.spec])
        with mock.patch.object(subprocess, "run", wraps=subprocess.run):
            custom_nodes.install(self.spec, self.root)
        self.assertEqual((self.root / "My-Node/node.py").read_text(), "one")
        self.assertEqual(custom_nodes.missing([self.spec], self.root), [])
        self.assertEqual([p.name for p in self.root.iterdir()], ["My-Node"])

    def test_installs_only_a_subfolder(self):
        custom_nodes.install({**self.spec, "name": "inner-node", "subdir": "pack"}, self.root)
        self.assertEqual([p.name for p in (self.root / "inner-node").iterdir()], ["inner.py"])
        self.assertEqual([p.name for p in self.root.iterdir()], ["inner-node"])

    def test_subdir_must_exist_and_stay_inside(self):
        with self.assertRaisesRegex(RuntimeError, "not a folder"):
            custom_nodes.install({**self.spec, "subdir": "nope"}, self.root)
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            custom_nodes.install({**self.spec, "subdir": "../x"}, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_falls_back_to_a_full_fetch(self):
        real = custom_nodes._run

        def refuse_shallow(*cmd, cwd=None):
            if "--depth" in cmd:
                raise RuntimeError("server does not allow fetching by hash")
            real(*cmd, cwd=cwd)

        with mock.patch.object(custom_nodes, "_run", refuse_shallow):
            custom_nodes.install(self.spec, self.root)
        self.assertEqual((self.root / "My-Node/node.py").read_text(), "one")

    def test_failed_install_leaves_nothing(self):
        with self.assertRaises(RuntimeError):
            custom_nodes.install({**self.spec, "ref": "0" * 40}, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_model_installs_before_launching_comfy(self):
        manifest, workflow = read_workflow_file(available_workflows()["qwen_image_edit_2511_aio"])
        manifest["loras"]["files"] = []
        manifest["custom_nodes"] = [self.spec]
        manifest["files"] = [{"name": "w.safetensors", "url": "https://example.org/w", "folder": "models/checkpoints",
                              "set": "1.ckpt_name"}]
        (self.tmp / "models/checkpoints").mkdir(parents=True)
        (self.tmp / "models/checkpoints/w.safetensors").touch()
        model = ComfyWorkflowModel("t", manifest, workflow)
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            with mock.patch.object(client, "launch_comfy_process") as launch:
                launch.side_effect = lambda: self.assertTrue((Path("custom_nodes/My-Node/node.py")).is_file()) or mock.Mock()
                model.start()
                self.assertTrue(model._download_status.finished.wait(20))
                self.assertIsNone(model._download_status.error)
                launch.assert_called_once()
        finally:
            os.chdir(old)


if __name__ == "__main__":
    unittest.main()
