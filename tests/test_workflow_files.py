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
import threading
import types
import unittest
from unittest import mock
from pathlib import Path

try:
    import websocket  # noqa: F401
except ImportError:  # only the ComfyUI client needs it, and these tests never connect
    sys.modules["websocket"] = types.ModuleType("websocket")

from PIL import Image

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationCancelled, GenerationError
from ai_image_edit.core.types import GenerationParams, GenerationProgress, Images, Inference, Postprocessing, ProgressStage, Size
from ai_image_edit.workflows import available_workflows
from ai_image_edit import workflows as workflow_files
from ai_image_edit.comfy import client
from ai_image_edit import sizes
from ai_image_edit.sizes import POLICIES
from ai_image_edit.workflow_model import REQUIRED_BINDINGS, SAMPLER_CHOICES, SCHEDULER_CHOICES, WorkflowModel

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

    def load(self, path: Path) -> WorkflowModel:
        """The model, with the weight files its manifest names created as empty placeholders."""
        model = WorkflowModel.from_file(path)
        for spec in model._manifest.get("files", []):
            (self.tmp / spec["folder"] / spec["name"]).parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / spec["folder"] / spec["name"]).touch()
        return model

    @staticmethod
    def inference(model: WorkflowModel = None, **overrides) -> Inference:
        """An Inference with a value other than the default for every option the model has (none without a model)."""
        def value(spec):
            if spec.range is not None:
                return 1.0
            if spec.choices:
                return next(c for c in spec.choices if c != spec.default)
            return "not this"
        options = {spec.name: value(spec) for spec in (model.capabilities.options if model else ())}
        values = dict(prompt="a prompt", steps=4, seed=1, options=options)
        return Inference(**(values | overrides))

    def request(self, source="source.png", mask=None, aspect="Original", resolution=1024, **inference) -> GenerationParams:
        return GenerationParams(
            Images(str(self.tmp / source), mask, reference_paths=tuple(inference.pop("reference_paths", ()))),
            Size(aspect, resolution), self.inference(**inference), Postprocessing(),
        )

    def build(self, model: WorkflowModel, n_refs: int) -> dict:
        m = model._manifest
        size_values = {name: 1024 for name in m["size"].get("bind", {})}
        inference = self.inference(model, lora_files=LORAS if m.get("loras") else (), lora_strength=0.7)
        return model.build_workflow(inference, "/x/source.png", [f"/x/ref{i}.png" for i in range(n_refs)], size_values)

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
            "unknown format version": lambda d: d.update(format_version=3),
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
            "optional input in bind": lambda d: d["manifest"]["bind"].update(cfg="2.cfg"),
            "cfg range in capabilities": lambda d: d["manifest"]["capabilities"].update(cfg=d["manifest"]["inputs"]["cfg"]["range"]),
            "input without target": lambda d: d["manifest"]["inputs"]["cfg"].pop("target"),
            "input with an unknown key": lambda d: d["manifest"]["inputs"]["cfg"].update(kind="slider"),
            "input name that is not lower case": lambda d: d["manifest"]["inputs"].update(CFG=d["manifest"]["inputs"].pop("cfg")),
            "empty input label": lambda d: d["manifest"]["inputs"]["cfg"].update(label=""),
            "input with a range and choices": lambda d: d["manifest"]["inputs"]["cfg"].update(choices=["a"]),
            "unknown named choices": lambda d: d["manifest"]["inputs"]["sampler"].update(choices="comfy_nodes"),
            "sampler in bind": lambda d: d["manifest"]["bind"].update(sampler="2.sampler_name"),
            "sampler in capabilities": lambda d: d["manifest"]["capabilities"].update(sampler={"default": "euler"}),
        }
        for what, mutate in mistakes.items():
            with self.subTest(what):
                self.assertFalse(validator.is_valid(broken(mutate)), f"accepted: {what}")

    def test_format_version_is_checked_when_loading(self):
        path = self.tmp / "wrong_version.json"
        path.write_text(json.dumps({"format_version": 99, "manifest": {}, "workflow": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported format_version 99"):
            workflow_files.read_workflow_file(path)

    def test_the_required_bindings_are_the_schemas(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        bind = schema["$defs"]["manifest"]["properties"]["bind"]
        self.assertEqual(set(REQUIRED_BINDINGS), set(bind["required"]))
        self.assertEqual(set(REQUIRED_BINDINGS), set(bind["properties"]))

    # The optional inputs every bundled workflow offers: (name, label, slider?, default), in order.
    EXPECTED_OPTIONS = {
        **{
            name: [("cfg", "CFG scale", True, 1.0), ("negative_prompt", "Negative prompt", False, ""),
                   ("sampler", "Sampler", False, "euler"), ("scheduler", "Scheduler", False, "simple")]
            for name in ("qwen_image21", "qwen_image21_gguf", "qwen_image21_turbo", "qwen_image21_turbo_gguf")
        },
        "qwen_image_edit_2511_aio": [("cfg", "CFG scale", True, 1.0), ("denoise", "Denoise", True, 1.0),
                                     ("sampler", "Sampler", False, "sa_solver"), ("scheduler", "Scheduler", False, "beta")],
    }

    def test_optional_inputs_become_controls(self):
        self.assertEqual(set(self.EXPECTED_OPTIONS), {p.stem for p in MANIFESTS})
        for path in MANIFESTS:
            with self.subTest(model=path.stem):
                options = self.load(path).capabilities.options
                self.assertEqual(
                    [(o.name, o.label, o.range is not None, o.default) for o in options], self.EXPECTED_OPTIONS[path.stem]
                )
                for option in options:
                    self.assertFalse(option.integer, option.name)  # CFG and denoise are fractional
                    self.assertIsInstance(option.convert(1), float if option.range else str)
                choices = {o.name: o.choices for o in options}
                self.assertEqual(choices["sampler"], tuple(SAMPLER_CHOICES))
                self.assertEqual(choices["scheduler"], tuple(SCHEDULER_CHOICES))

    def edited(self, change) -> tuple:
        manifest, workflow = workflow_files.read_workflow_file(available_workflows()["qwen_image21"])
        change(manifest, workflow)
        return manifest, workflow

    def test_an_input_must_match_its_workflow_placeholder(self):
        cases = {
            "has a range, but its workflow input holds": lambda m, w: m["inputs"]["negative_prompt"].update(
                range={"min": 0, "max": 1, "default": 0}
            ),
            "needs a range": lambda m, w: m["inputs"]["cfg"].pop("range"),
            "a binding every workflow has": lambda m, w: m["inputs"].update(seed={"target": "6.seed", "range": {"min": 0, "max": 9, "default": 0}}),
            "has no input 'nope'": lambda m, w: m["inputs"].update(extra={"target": "6.nope"}),
            "not one of its choices": lambda m, w: m["inputs"]["scheduler"].update(choices=["karras", "normal"]),
        }
        for message, change in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    WorkflowModel("x", *self.edited(change))

    def test_labels_and_whole_numbers_are_inferred(self):
        def add(m, w):
            w["6"]["inputs"]["extra_steps"] = 2
            m["inputs"]["extra_steps"] = {"target": "6.extra_steps", "range": {"min": 1, "max": 8, "default": 2, "step": 1}}
        spec = next(o for o in WorkflowModel("x", *self.edited(add)).capabilities.options if o.name == "extra_steps")
        self.assertEqual((spec.label, spec.integer, spec.default), ("Extra steps", True, 2))
        self.assertEqual(spec.convert(3.0), 3)
        self.assertIsInstance(spec.convert(3.0), int)

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
                for option in caps.options:
                    if option.choices:
                        self.assertIn(option.default, option.choices)
                self.assertIn(caps.default_resolution, caps.supported_resolutions)
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

    def test_loras_are_rejected_by_a_workflow_without_a_loras_section(self):
        for path in MANIFESTS:
            model = self.load(path)
            if "loras" in model._manifest:
                continue
            with self.subTest(model=path.stem):
                with self.assertRaisesRegex(ValueError, "no 'loras' section"):
                    model._splice_loras({}, LORAS, 0.7)

    def run_generate(self, model, **inference) -> int:
        real = imaging.run_generation
        imaging.run_generation = lambda *args, **kwargs: "/out.png"
        try:
            return model.generate(self.request(prompt="a hat", steps=10, **inference)).actual_seed
        finally:
            imaging.run_generation = real

    def test_generate_reports_progress_with_the_sampler_node(self):
        model = self.load(workflow_files.available_workflows()["qwen_image21"])
        events, calls = [], []

        def submit(workflow, client_id, sampler_node=None, on_progress=None, cancel=None):
            calls.append(sampler_node)
            on_progress(GenerationProgress(ProgressStage.SAMPLING, 1, 10))
            return "p"

        real = imaging.run_generation
        imaging.run_generation = lambda source, mask, annotated, size, infer, *a, **k: infer(source)
        try:
            with mock.patch.object(model, "_check_nodes"), mock.patch.object(model, "_check_setup"), \
                    mock.patch.object(client, "submit_workflow_and_wait", submit), \
                    mock.patch.object(client, "fetch_generated_image", return_value="/out.png"):
                model.generate(self.request(prompt="a new hat", steps=10, seed=987), events.append)
        finally:
            imaging.run_generation = real
        self.assertEqual(calls, ["6"])
        self.assertEqual([e.stage for e in events],
                         [ProgressStage.PREPARING, ProgressStage.SAMPLING, ProgressStage.POSTPROCESSING])

    def test_lora_downloads_can_be_turned_off(self):
        found = False
        for path in MANIFESTS:
            lora_files = {spec["name"] for spec in WorkflowModel.from_file(path)._manifest.get("loras", {}).get("files", [])}
            if not lora_files:
                continue
            found = True
            with self.subTest(model=path.stem):
                self.assertTrue(lora_files <= {i["dest"].name for i in WorkflowModel.from_file(path)._pending_downloads()})
                off = WorkflowModel.from_file(path, lora_downloads=False)
                self.assertFalse(lora_files & {i["dest"].name for i in off._pending_downloads()})
                self.assertTrue(off.capabilities.supports_loras)
        self.assertTrue(found, "no bundled workflow downloads LoRAs")

    def test_a_cancelled_generate_submits_nothing(self):
        model = self.load(workflow_files.available_workflows()["qwen_image21"])
        cancel = threading.Event()
        cancel.set()
        real = imaging.run_generation
        imaging.run_generation = lambda source, mask, annotated, size, infer, *a, **k: infer(source)
        try:
            with mock.patch.object(model, "_check_nodes"), mock.patch.object(model, "_check_setup"), \
                    mock.patch.object(client, "submit_workflow_and_wait") as submit:
                with self.assertRaises(GenerationCancelled):
                    model.generate(self.request(prompt="a cancelled hat", steps=10), cancel=cancel)
        finally:
            imaging.run_generation = real
        submit.assert_not_called()

    def test_turbo_workflows_default_to_eight_steps_without_guidance(self):
        for name in ("qwen_image21_turbo", "qwen_image21_turbo_gguf"):
            with self.subTest(workflow=name):
                model = self.load(workflow_files.available_workflows()[name])
                self.assertEqual(8, model.capabilities.step_range.default)
                cfg = next(spec for spec in model.capabilities.options if spec.name == "cfg")
                self.assertEqual(1.0, cfg.default)
                self.assertEqual(8, model._workflow["6"]["inputs"]["steps"])

    def test_qwen21_output_size_comes_from_an_empty_latent(self):
        for name in ("qwen_image21", "qwen_image21_gguf", "qwen_image21_turbo", "qwen_image21_turbo_gguf"):
            with self.subTest(workflow=name):
                model = self.load(workflow_files.available_workflows()[name])
                size = model._size.resolve(Size("16:9", 1024), str(self.tmp / "source.png"), masked=False)
                self.assertEqual((1376, 768), (size.width, size.height))
                wf = model.build_workflow(self.inference(model), "/x/source.png", [], size.values)
                latent_id = wf["6"]["inputs"]["latent_image"][0]
                self.assertEqual("EmptyLatentImage", wf[latent_id]["class_type"])
                self.assertEqual((1376, 768), (wf[latent_id]["inputs"]["width"], wf[latent_id]["inputs"]["height"]))
                self.assertEqual(0, wf["5"]["inputs"]["resolution"])

    def test_qwen21_fits_the_source_to_the_output_only_where_it_keeps_its_framing(self):
        Image.new("RGB", (1200, 896), (10, 20, 30)).save(self.tmp / "four_by_three.png")
        cases = [  # source file, mask, aspect ratio, fitted?
            ("source.png", "m.png", "16:9", True),  # a mask: always
            ("source.png", None, "Original", True),
            ("source.png", None, "16:9", False),  # 1000 x 777 recomposed into 16:9
            ("four_by_three.png", None, "4:3", True),  # already the chosen ratio
            ("four_by_three.png", None, "16:9", False),
        ]
        for name in ("qwen_image21", "qwen_image21_gguf", "qwen_image21_turbo", "qwen_image21_turbo_gguf"):
            model = self.load(workflow_files.available_workflows()[name])
            for source, mask, aspect, fitted in cases:
                with self.subTest(workflow=name, source=source, mask=mask, aspect=aspect):
                    source_path = str(self.tmp / source)
                    size = model._size.resolve(Size(aspect, 1024), source_path, masked=bool(mask))
                    own_ratio = sizes.dimensions_from_source(source_path, size.reference_resolution)
                    self.assertEqual((size.width, size.height) if fitted else own_ratio, size.input_size)
                    self.assertEqual(model._size.resolution_for_source(source_path) if mask else 1024, size.reference_resolution)
                    self.assertEqual(0, size.values["resolution"])
                    self.assertEqual((size.width, size.height), (size.values["width"], size.values["height"]))
                    wf = model.build_workflow(self.inference(model), "/x/source.png", [], size.values)
                    self.assertEqual(size.values["resolution"], wf["5"]["inputs"]["resolution"])

    def test_size_policies_resolve_exactly_the_values_they_declare(self):
        source = str(self.tmp / "source.png")
        for name, policy_class in POLICIES.items():
            config = {"resolutions": [1024]} if name == "fixed_area" else {}
            policy = policy_class(config)
            for aspect in policy.aspect_ratios:
                with self.subTest(policy=name, aspect=aspect):
                    resolved = policy.resolve(Size(aspect, policy.default_resolution), source, masked=False)
                    self.assertEqual(policy.value_names, set(resolved.values))

    def test_loading_rejects_a_size_binding_that_leaves_a_value_unused(self):
        data = json.loads(workflow_files.available_workflows()["qwen_image21"].read_text(encoding="utf-8"))
        del data["manifest"]["size"]["bind"]["height"]
        with self.assertRaisesRegex(ValueError, "size.bind must bind exactly"):
            WorkflowModel("broken", data["manifest"], data["workflow"])

    def test_variant_does_not_inherit_the_checksum_of_another_file(self):
        manifest, workflow = workflow_files.read_workflow_file(available_workflows()["qwen_image21"])
        manifest["files"][0].update(sha256="a" * 64, size=1)
        manifest["variants"] = {
            "other": {"files": {"1.unet_name": {"name": "o.safetensors", "url": "https://example.org/o"}}},
            "same": {"files": {"1.unet_name": {"size": 2}}},
        }
        other = WorkflowModel("t", manifest, workflow, "other")._manifest["files"][0]
        self.assertEqual(other["name"], "o.safetensors")
        self.assertNotIn("sha256", other)
        self.assertNotIn("size", other)
        same = WorkflowModel("t", manifest, workflow, "same")._manifest["files"][0]
        self.assertEqual((same["sha256"], same["size"]), ("a" * 64, 2))

    def test_seed_handling(self):
        real = random.randint
        try:
            for path in MANIFESTS:
                model = self.load(path)
                spec = model._manifest.get("seed", {"min": 0, "max": 2 ** 32 - 1})
                with self.subTest(model=path.stem):
                    seen = []
                    random.randint = lambda a, b: seen.append((a, b)) or a
                    self.run_generate(model, seed=None)
                    self.assertEqual(seen, [(spec["min"], spec["max"])])
                    random.randint = real
                    chosen = 2 ** 40 + 5
                    expected = chosen % (spec["max"] + 1) if spec.get("wrap") else chosen
                    self.assertEqual(self.run_generate(model, seed=chosen), expected)
        finally:
            random.randint = real

    def test_options_reach_their_workflow_inputs(self):
        for path in MANIFESTS:
            model = self.load(path)
            with self.subTest(model=path.stem):
                inference = self.inference(model)
                self.assertTrue(inference.options, "every bundled workflow has optional inputs")
                wf = model.build_workflow(inference, "/x/source.png", [], {n: 1024 for n in model._manifest["size"]["bind"]})
                for name, value in inference.options.items():
                    node, _, key = model._option_targets[name].partition(".")
                    self.assertEqual(wf[node]["inputs"][key], value)

    def test_a_choice_must_be_one_of_the_choices(self):
        model = self.load(MANIFESTS[0])
        with self.assertRaisesRegex(ValueError, "not one of the choices of sampler"):
            self.run_generate(model, options={"sampler": "nope"})

    def test_a_model_without_a_sampler_input_keeps_the_workflows(self):
        model = WorkflowModel("x", *self.edited(lambda m, w: m["inputs"].pop("sampler")))
        self.assertNotIn("sampler", [o.name for o in model.capabilities.options])
        wf = model.build_workflow(self.inference(model), "/x/source.png", [], {n: 1024 for n in model._manifest["size"]["bind"]})
        self.assertEqual(wf["6"]["inputs"]["sampler_name"], "euler")

    def test_an_option_the_model_does_not_have_is_rejected(self):
        for path in MANIFESTS:
            with self.subTest(model=path.stem):
                with self.assertRaisesRegex(ValueError, "no optional input"):
                    self.run_generate(self.load(path), options={"nope": 1})

    def test_too_many_input_images_is_rejected(self):
        for path in MANIFESTS:
            model = self.load(path)
            refs = model._manifest["images"].get("references")
            limit = refs["max"] if refs else 0
            with self.subTest(model=path.stem):
                with self.assertRaisesRegex(GenerationError, f"Up to {limit + 1} input images"):
                    self.run_generate(model, reference_paths=[f"/x/r{i}.png" for i in range(limit + 1)])


if __name__ == "__main__":
    unittest.main()
