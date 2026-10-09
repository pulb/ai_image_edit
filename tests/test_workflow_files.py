# SPDX-License-Identifier: GPL-3.0-or-later
"""
Checks every workflow file in src/ai_image_edit/data/workflows/ against the JSON Schema and
against the code that runs it, so a new file is covered without adding a test:

    PYTHONPATH=src python -m unittest discover -s tests -v

No ComfyUI, GPU or network is needed: image uploads are stubbed and the weight
files are empty placeholders in a temporary folder. What this cannot catch is a
mistake inside the workflow itself (a wrong node class or input name); only
running the workflow in ComfyUI shows that. The schema tests need the
jsonschema package (pip install jsonschema) and are skipped without it.
"""
import copy
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import websocket  # noqa: F401
except ImportError:  # only the ComfyUI client needs it, and these tests never connect
    sys.modules["websocket"] = types.ModuleType("websocket")

from PIL import Image

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.types import GenerationParams
from ai_image_edit.models._shared.comfyui.workflow_files import available_workflows
from ai_image_edit.models._shared.comfyui import client, workflow_files
from ai_image_edit.models._shared.comfyui.workflow_model import ComfyWorkflowModel

import jsonschema

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "ai_image_edit" / "data"
SCHEMA_PATH = DATA_DIR / "workflow.schema.json"
MANIFESTS = sorted((DATA_DIR / "workflows").glob("*.json"))
LORAS = ["a.safetensors", "pack/unet.safetensors", "pack/clip.safetensors"]


def is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


class ManifestModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._old_cwd, cls._old_env = os.getcwd(), dict(os.environ)
        cls._old_upload = client.upload_image
        cls.tmp = Path(cls._tmp.name)
        os.chdir(cls.tmp)
        client.upload_image = lambda path: "uploaded_" + os.path.basename(path)
        for lora in LORAS:
            (cls.tmp / "models/loras" / lora).parent.mkdir(parents=True, exist_ok=True)
            (cls.tmp / "models/loras" / lora).touch()
        Image.new("RGB", (1000, 777), (10, 20, 30)).save(cls.tmp / "source.png")

    @classmethod
    def tearDownClass(cls):
        client.upload_image = cls._old_upload
        os.chdir(cls._old_cwd)
        os.environ.clear()
        os.environ.update(cls._old_env)
        cls._tmp.cleanup()

    def load(self, path: Path) -> ComfyWorkflowModel:
        """The model, with the weight files its manifest names created as empty placeholders."""
        model = ComfyWorkflowModel.from_file(path)
        for spec in model._manifest.get("files", []):
            # A file with an env override is exercised through it; the others by their own name.
            name = f"{spec['env'].lower()}.bin" if "env" in spec else spec["name"]
            (self.tmp / spec["folder"] / name).parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / spec["folder"] / name).touch()
            if "env" in spec:
                os.environ[spec["env"]] = name
        return model

    def build(self, model: ComfyWorkflowModel, n_refs: int) -> dict:
        m = model._manifest
        size_values = {name: 1024 for name in m["size"].get("bind", {})}
        return model.build_workflow(
            prompt="a prompt", negative_prompt="not this", source_path="/x/source.png",
            reference_paths=[f"/x/ref{i}.png" for i in range(n_refs)], seed=1, steps=4, cfg=1.0, denoise=1.0,
            sampler="euler", scheduler="simple", lora_files=LORAS if m.get("loras") else [], size_values=size_values,
        )

    def test_workflow_files_exist_and_are_listed(self):
        self.assertTrue(MANIFESTS, "no workflow file found in ai_image_edit/data/workflows")
        for path in MANIFESTS:
            self.assertIn(path.stem, available_workflows())

    def test_files_match_the_schema(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        for path in MANIFESTS:
            with self.subTest(file=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
                self.assertEqual([], [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors])

    def test_loading_rejects_a_file_that_breaks_the_schema(self):
        data = json.loads(MANIFESTS[0].read_text(encoding="utf-8"))
        del data["manifest"]["bind"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "(?s)does not match the workflow file schema.*bind"):
                workflow_files.read_workflow_file(path)

    def test_schema_rejects_mistakes(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        good = json.loads(next(p for p in MANIFESTS if "aio" in p.stem).read_text(encoding="utf-8"))
        self.assertTrue(validator.is_valid(good))

        def broken(mutate):
            data = copy.deepcopy(good)
            mutate(data)
            return data

        mistakes = {
            "unknown format version": lambda d: d.update(format_version=2),
            "no workflow": lambda d: d.pop("workflow"),
            "unknown manifest key": lambda d: d["manifest"].update(modle_name="x"),
            "no name": lambda d: d["manifest"].pop("model_name"),
            "no prompt binding": lambda d: d["manifest"]["bind"].pop("prompt"),
            "unknown binding": lambda d: d["manifest"]["bind"].update(temperature="1.name"),
            "target without input name": lambda d: d["manifest"]["bind"].update(seed="2"),
            "unknown size policy": lambda d: d["manifest"]["size"].update(policy="whatever"),
            "size without width/height": lambda d: d["manifest"]["size"].update(bind={"width": "9.width"}),
            "node without class": lambda d: d["workflow"]["1"].pop("class_type"),
            "range without default": lambda d: d["manifest"]["capabilities"]["steps"].pop("default"),
            "references without loader ids": lambda d: d["manifest"]["images"]["references"].pop("loader_ids"),
            "loras without strength range": lambda d: d["manifest"]["capabilities"].pop("lora_strength"),
        }
        for what, mutate in mistakes.items():
            with self.subTest(what):
                self.assertFalse(validator.is_valid(broken(mutate)), f"accepted: {what}")

    def test_format_version_is_checked_when_loading(self):
        path = self.tmp / "wrong_version.json"
        path.write_text(json.dumps({"format_version": 99, "manifest": {}, "workflow": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported format_version 99"):
            workflow_files.read_workflow_file(path)

    def test_workflows_are_well_formed_for_every_number_of_references(self):
        for path in MANIFESTS:
            model = self.load(path)
            refs = model._manifest["images"].get("references")
            for n in sorted({0, 1, refs["max"] if refs else 0}):
                with self.subTest(model=path.stem, references=n):
                    wf = self.build(model, n)
                    for node_id, node in wf.items():
                        self.assertIn("class_type", node, f"node {node_id}")
                        self.assertIsInstance(node.get("inputs"), dict, f"node {node_id}")
                        for name, value in node["inputs"].items():
                            if is_link(value):
                                self.assertIn(value[0], wf, f"node {node_id} input {name!r} links to a missing node")
                    if refs:
                        for i in range(refs["max"]):
                            loader = refs["loader_ids"][i] if "loader_ids" in refs else str(refs["loader_id_start"] + i)
                            node, key = refs["target"].format(n=i + refs.get("first_n", 2)).split(".", 1)
                            used = i < n
                            self.assertEqual(loader in wf, used, f"loader {loader}")
                            self.assertEqual(key in wf[node]["inputs"], used, f"input {key!r}")

    def test_capabilities_and_listing_work(self):
        for path in MANIFESTS:
            with self.subTest(model=path.stem):
                model = self.load(path)
                caps = model.capabilities
                refs = model._manifest["images"].get("references")
                self.assertEqual(caps.max_reference_images, 1 + (refs["max"] if refs else 0))
                self.assertEqual(caps.supports_loras, bool(model._manifest.get("loras")))
                self.assertIn(caps.default_sampler, caps.sampler_choices)
                self.assertIn(caps.default_scheduler, caps.scheduler_choices)
                self.assertIn(caps.default_megapixels, caps.supported_megapixels)
                self.assertIn(caps.default_aspect_ratio, caps.supported_aspect_ratios)
                self.assertIsInstance(model.list_loras(), dict)
                model.model_version  # must not raise

    def test_lora_chain_feeds_the_workflow(self):
        for path in MANIFESTS:
            model = self.load(path)
            cfg = model._manifest.get("loras")
            if not cfg:
                continue
            with self.subTest(model=path.stem):
                wf = self.build(model, 0)
                last = str(cfg["first_id"] + len(LORAS) - 1)
                self.assertEqual(wf[last]["class_type"], cfg["loader_class"])
                for targets in cfg["feeds"].values():
                    for target in targets:
                        node, key = target.split(".", 1)
                        self.assertEqual(wf[node]["inputs"][key][0], last)

    def run_generate(self, model, **params) -> int:
        real = imaging.run_masked_generation
        imaging.run_masked_generation = lambda *args: "/out.png"
        try:
            return model.generate(GenerationParams(
                prompt="a hat", source_image_path=str(self.tmp / "source.png"), mask_path=None, steps=10, **params
            )).actual_seed
        finally:
            imaging.run_masked_generation = real

    def test_seed_handling(self):
        real = random.randint
        try:
            for path in MANIFESTS:
                model = self.load(path)
                spec = model._manifest.get("seed", {"min": 0, "max": 2 ** 32 - 1})
                with self.subTest(model=path.stem):
                    seen = []
                    random.randint = lambda a, b: seen.append((a, b)) or a
                    self.run_generate(model, randomize_seed=True)
                    self.assertEqual(seen, [(spec["min"], spec["max"])])
                    random.randint = real
                    chosen = 2 ** 40 + 5
                    expected = chosen % (spec["max"] + 1) if spec.get("wrap") else chosen
                    self.assertEqual(self.run_generate(model, randomize_seed=False, seed=chosen), expected)
        finally:
            random.randint = real

    def test_too_many_input_images_is_rejected(self):
        for path in MANIFESTS:
            model = self.load(path)
            refs = model._manifest["images"].get("references")
            limit = refs["max"] if refs else 0
            with self.subTest(model=path.stem):
                with self.assertRaisesRegex(GenerationError, f"Up to {limit + 1} input images"):
                    self.run_generate(model, reference_images=[f"/x/r{i}.png" for i in range(limit + 1)])


if __name__ == "__main__":
    unittest.main()
