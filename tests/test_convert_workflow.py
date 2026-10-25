# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tests for tools/convert_workflow.py on a small hand-written editor workflow and
the node definitions it needs (no ComfyUI required).

    PYTHONPATH=src python -m unittest discover -s tests -v
"""
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import jsonschema
except ImportError:
    jsonschema = None

TOOL = Path(__file__).resolve().parents[1] / "tools" / "convert_workflow.py"
spec = importlib.util.spec_from_file_location("convert_workflow", TOOL)
convert_workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_workflow)

OBJECT_INFO = {
    "KSampler": {"input": {"required": {
        "model": ["MODEL"], "seed": ["INT", {"control_after_generate": True}], "steps": ["INT", {}],
        "cfg": ["FLOAT", {}], "sampler_name": [["euler", "dpmpp_2m"]], "scheduler": [["simple", "karras"]],
        "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"], "denoise": ["FLOAT", {}]}},
        "python_module": "nodes"},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}}, "python_module": "nodes"},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}},
                         "python_module": "nodes"},
    "LoadImage": {"input": {"required": {"image": [["a.png"], {"image_upload": True}]}}, "python_module": "nodes"},
    "UNETLoader": {"input": {"required": {"unet_name": [["x.safetensors"]], "weight_dtype": [["default", "fp8"]]}},
                   "python_module": "nodes"},
    "CLIPLoader": {"input": {"required": {"clip_name": [["c.safetensors"]], "type": [["sd3"]]}}, "python_module": "nodes"},
    "VAELoader": {"input": {"required": {"vae_name": [["v.safetensors"]]}}, "python_module": "nodes"},
    "Custom": {"input": {"required": {"image": ["IMAGE"]}}, "python_module": "custom_nodes.thing"},
    "Plain": {"input": {"required": {"image": ["IMAGE"]}}, "python_module": "nodes"},
}


def node(node_id, kind, inputs=None, outputs=None, widgets=None, mode=0):
    return {"id": node_id, "type": kind, "mode": mode, "inputs": inputs or [], "outputs": outputs or [],
            "widgets_values": widgets or []}


def inp(name, kind, link=None):
    return {"name": name, "type": kind, "link": link}


def editor_workflow():
    """UNET 1, CLIP 2, VAE 3, positive 4, negative 5 (through a Reroute 9), latent 6, sampler 7, LoadImage 8."""
    return {
        "nodes": [
            node(1, "UNETLoader", outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}], widgets=["x.safetensors", "default"]),
            node(2, "CLIPLoader", outputs=[{"name": "CLIP", "type": "CLIP", "links": [2, 3]}], widgets=["c.safetensors", "sd3"]),
            node(3, "VAELoader", widgets=["v.safetensors"]),
            node(4, "CLIPTextEncode", inputs=[inp("clip", "CLIP", 2)], outputs=[{"name": "C", "type": "CONDITIONING", "links": [4]}],
                 widgets=["a cat"]),
            node(5, "CLIPTextEncode", inputs=[inp("clip", "CLIP", 3)], outputs=[{"name": "C", "type": "CONDITIONING", "links": [9]}],
                 widgets=["ugly"]),
            node(6, "EmptyLatentImage", outputs=[{"name": "L", "type": "LATENT", "links": [6]}], widgets=[1024, 1024, 1]),
            node(7, "KSampler", inputs=[inp("model", "MODEL", 1), inp("positive", "CONDITIONING", 4),
                                        inp("negative", "CONDITIONING", 5), inp("latent_image", "LATENT", 6)],
                 widgets=[42, "randomize", 30, 4.0, "euler", "simple", 1.0]),
            node(8, "LoadImage", widgets=["a.png", "image"]),
        ],
        "links": [[1, 1, 0, 7, 0, "MODEL"], [2, 2, 0, 4, 0, "CLIP"], [3, 2, 0, 5, 0, "CLIP"], [4, 4, 0, 7, 1, "CONDITIONING"],
                  [5, 5, 0, 7, 2, "CONDITIONING"], [6, 6, 0, 7, 3, "LATENT"]],
    }


class GraphTests(unittest.TestCase):
    def convert(self, editor=None):
        return convert_workflow.convert_graph(editor or editor_workflow(), OBJECT_INFO)

    def test_widget_values_links_and_control_values(self):
        api, notes = self.convert()
        self.assertEqual(notes, [])
        self.assertEqual(api["7"]["class_type"], "KSampler")
        self.assertEqual(api["7"]["inputs"], {
            "seed": 42, "steps": 30, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]})
        self.assertEqual(api["8"]["inputs"], {"image": "a.png"})  # the upload widget's value is skipped
        self.assertEqual(api["4"]["inputs"], {"text": "a cat", "clip": ["2", 0]})

    def test_notes_and_muted_nodes_are_dropped(self):
        editor = editor_workflow()
        editor["nodes"].append(node(20, "Note", widgets=["hello"]))
        editor["nodes"][4]["mode"] = convert_workflow.MODE_MUTED
        api, notes = self.convert(editor)
        self.assertNotIn("20", api)
        self.assertNotIn("5", api)
        self.assertNotIn("negative", api["7"]["inputs"])
        self.assertTrue(any("muted" in n for n in notes))

    def test_reroute_is_followed(self):
        editor = editor_workflow()
        editor["nodes"].append(node(9, "Reroute", inputs=[inp("", "CONDITIONING", 5)], outputs=[{"type": "CONDITIONING", "links": [7]}]))
        editor["nodes"][6]["inputs"][2]["link"] = 7
        editor["links"].append([7, 9, 0, 7, 2, "CONDITIONING"])
        api, _ = self.convert(editor)
        self.assertNotIn("9", api)
        self.assertEqual(api["7"]["inputs"]["negative"], ["5", 0])

    def test_bypassed_node_passes_its_input_through(self):
        editor = editor_workflow()
        editor["nodes"].append(node(10, "Plain", inputs=[inp("image", "IMAGE", 8)], outputs=[{"type": "IMAGE", "links": [9]}],
                                    mode=convert_workflow.MODE_BYPASSED))
        editor["nodes"].append(node(11, "Custom", inputs=[inp("image", "IMAGE", 9)]))
        editor["links"] += [[8, 8, 0, 10, 0, "IMAGE"], [9, 10, 0, 11, 0, "IMAGE"]]
        api, _ = self.convert(editor)
        self.assertNotIn("10", api)
        self.assertEqual(api["11"]["inputs"]["image"], ["8", 0])

    def test_primitive_node_gives_its_value(self):
        editor = editor_workflow()
        editor["nodes"].append(node(30, "PrimitiveNode", outputs=[{"type": "INT", "links": [10]}], widgets=[99, "fixed"]))
        editor["nodes"][6]["inputs"].append(inp("steps", "INT", 10))
        editor["links"].append([10, 30, 0, 7, 4, "INT"])
        api, _ = self.convert(editor)
        self.assertEqual(api["7"]["inputs"]["steps"], 99)

    def test_unknown_class_and_subgraphs_are_errors(self):
        editor = editor_workflow()
        editor["nodes"].append(node(40, "NotInstalled"))
        with self.assertRaisesRegex(convert_workflow.ConvertError, "NotInstalled"):
            self.convert(editor)
        with self.assertRaisesRegex(convert_workflow.ConvertError, "subgraphs"):
            self.convert({"nodes": [], "links": [], "definitions": {"subgraphs": [{}]}})

    def test_links_as_objects(self):
        editor = editor_workflow()
        editor["links"] = [{"id": l[0], "origin_id": l[1], "origin_slot": l[2], "target_id": l[3], "target_slot": l[4], "type": l[5]}
                           for l in editor["links"]]
        api, _ = self.convert(editor)
        self.assertEqual(api["7"]["inputs"]["model"], ["1", 0])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        api, _ = convert_workflow.convert_graph(editor_workflow(), OBJECT_INFO)
        self.api = api
        self.manifest, self.notes = convert_workflow.scaffold_manifest(api, OBJECT_INFO, "My model")

    def test_bindings(self):
        m = self.manifest
        self.assertEqual(m["bind"], {"seed": "7.seed", "steps": "7.steps", "prompt": "4.text"})
        self.assertEqual(m["inputs"], {
            "cfg": {"target": "7.cfg", "label": "CFG scale", "range": {"min": 1.0, "max": 10.0, "default": 4.0, "step": 0.1}},
            "negative_prompt": {"target": "5.text"},
            "denoise": {"target": "7.denoise", "range": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01}},
            "sampler": {"target": "7.sampler_name", "choices": "comfy_samplers"},
            "scheduler": {"target": "7.scheduler", "choices": "comfy_schedulers"},
        })
        # the order Advanced settings shows
        self.assertEqual(list(m["inputs"]), ["cfg", "negative_prompt", "denoise", "sampler", "scheduler"])
        self.assertNotIn("sampler", m["capabilities"])
        self.assertEqual(m["images"], {"source": "8.image"})
        self.assertEqual(m["size"]["bind"], {"width": "6.width", "height": "6.height"})
        self.assertEqual(m["size"]["resolutions"], [1024])
        self.assertEqual(m["capabilities"]["steps"]["default"], 30)

    def test_model_files_are_listed_with_empty_placeholders(self):
        self.assertEqual(sorted(f["set"] for f in self.manifest["files"]), ["1.unet_name", "2.clip_name", "3.vae_name"])
        self.assertEqual(next(f for f in self.manifest["files"] if f["set"] == "3.vae_name"),
                         {"name": "v.safetensors", "url": "https://", "folder": "models/vae", "set": "3.vae_name"})
        self.assertEqual(self.api["1"]["inputs"]["unet_name"], "")
        self.assertTrue(any("x.safetensors" in n for n in self.notes))

    def test_custom_node_classes_are_listed(self):
        api = {"1": {"class_type": "Custom", "inputs": {}}, "2": {"class_type": "Plain", "inputs": {}}}
        manifest, notes = convert_workflow.scaffold_manifest(api, OBJECT_INFO, "x")
        self.assertEqual(manifest["required_nodes"], ["Custom"])
        self.assertTrue(any("no KSampler" in n for n in notes))

    def test_result_loads_as_a_model(self):
        try:
            import websocket  # noqa: F401
        except ImportError:  # only the ComfyUI client needs it, and loading never connects
            sys.modules["websocket"] = types.ModuleType("websocket")
        from ai_image_edit.workflow_model import WorkflowModel

        result, _ = convert_workflow.convert(editor_workflow(), OBJECT_INFO, "My model")
        model = WorkflowModel("converted", result["manifest"], result["workflow"])
        self.assertEqual(model.capabilities.supported_resolutions, [1024])
        self.assertEqual([(o.name, o.label) for o in model.capabilities.options],
                         [("cfg", "CFG scale"), ("negative_prompt", "Negative prompt"), ("denoise", "Denoise"),
                          ("sampler", "Sampler"), ("scheduler", "Scheduler")])
        defaults = {o.name: o.default for o in model.capabilities.options}
        self.assertEqual((defaults["sampler"], defaults["scheduler"]), ("euler", "simple"))

    @unittest.skipUnless(jsonschema, "pip install jsonschema")
    def test_result_matches_the_schema(self):
        result, notes = convert_workflow.convert(editor_workflow(), OBJECT_INFO, "My model")
        self.assertEqual([n for n in notes if n.startswith("schema:")], [])
        with open(convert_workflow.SCHEMA_PATH, "r", encoding="utf-8") as f:
            jsonschema.validate(result, json.load(f))


class BuiltinNodesTests(unittest.TestCase):
    def test_a_workflow_of_built_in_nodes_converts_with_the_bundled_definitions(self):
        nodes, _ = convert_workflow.load_builtin_nodes()
        api, _ = convert_workflow.convert_graph(editor_workflow(), nodes)
        self.assertEqual(api["7"]["inputs"], {
            "seed": 42, "steps": 30, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
        })
        self.assertEqual(api["8"]["inputs"], {"image": "a.png"})
        self.assertEqual(api["1"]["inputs"], {"unet_name": "x.safetensors", "weight_dtype": "default"})

    def test_a_custom_node_needs_the_object_info_of_its_comfyui(self):
        editor = {"nodes": [node(1, "Custom")], "links": []}
        with self.assertRaisesRegex(convert_workflow.ConvertError, "'Custom'.*give --object-info"):
            convert_workflow.convert_graph(editor, convert_workflow.load_builtin_nodes()[0])

    def test_the_bundled_definitions_hold_only_what_reading_widgets_needs(self):
        nodes, ref = convert_workflow.load_builtin_nodes()
        dockerfile = (TOOL.parents[1] / "deployments" / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"ARG COMFYUI_REF={ref}\n", dockerfile, "renew them with --update-builtin-nodes for the pinned ComfyUI")
        for name in ("KSampler", "LoadImage", "CheckpointLoaderSimple", "TextEncodeQwenImageEditPlus", "EmptyLatentImage"):
            self.assertIn(name, nodes)
        for name, info in nodes.items():
            self.assertFalse(info["python_module"].startswith("custom_nodes"), name)
            self.assertLessEqual(set(info), {"input", "input_order", "python_module"}, name)
            for section in info["input"].values():
                for entry in section.values():
                    self.assertFalse(isinstance(entry[0], list) and entry[0], f"{name}: dropdown options")
                    self.assertLessEqual(set(entry[1] if len(entry) > 1 else {}), set(convert_workflow.WIDGET_CONFIG_KEYS), name)

    def test_trimming_keeps_the_built_in_nodes_widget_order_and_markers(self):
        full = copy.deepcopy(OBJECT_INFO)
        full["KSampler"]["input"]["required"]["steps"] = ["INT", {"default": 20, "tooltip": "how many"}]
        full["KSampler"]["input_order"] = {"required": list(full["KSampler"]["input"]["required"])}
        trimmed = convert_workflow.trim_object_info(full)
        self.assertNotIn("Custom", trimmed)
        self.assertEqual(trimmed["LoadImage"]["input"], {"required": {"image": [[], {"image_upload": True}]}})
        self.assertEqual(trimmed["KSampler"]["input"]["required"]["steps"], ["INT"])
        self.assertEqual(trimmed["KSampler"]["input"]["required"]["seed"], ["INT", {"control_after_generate": True}])
        self.assertEqual(trimmed["KSampler"]["input_order"], full["KSampler"]["input_order"])
        api, _ = convert_workflow.convert_graph(editor_workflow(), trimmed)
        self.assertEqual(api, convert_workflow.convert_graph(editor_workflow(), OBJECT_INFO)[0])


class CommandLineTests(unittest.TestCase):
    def test_converts_without_object_info_and_says_which_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "in.json").write_text(json.dumps(editor_workflow()))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = convert_workflow.main([str(tmp / "in.json"), "--api-only", "-o", str(tmp / "api.json")])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((tmp / "api.json").read_text())["7"]["inputs"]["seed"], 42)
            self.assertIn("built-in node definitions of ComfyUI 87c3282", err.getvalue())

    def test_renews_the_bundled_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "info.json").write_text(json.dumps(OBJECT_INFO))
            target = tmp / "builtin.json"
            with mock.patch.object(convert_workflow, "BUILTIN_NODES_PATH", target), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(convert_workflow.main(["--update-builtin-nodes", str(tmp / "info.json"), "--comfyui-ref", "abc"]), 0)
                with self.assertRaises(SystemExit):
                    convert_workflow.main(["--update-builtin-nodes", str(tmp / "info.json")])
            data = json.loads(target.read_text())
            self.assertEqual(data["comfyui_ref"], "abc")
            self.assertEqual(data["nodes"], convert_workflow.trim_object_info(OBJECT_INFO))

    def test_writes_a_workflow_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "in.json").write_text(json.dumps(editor_workflow()))
            (tmp / "info.json").write_text(json.dumps(OBJECT_INFO))
            code = convert_workflow.main([str(tmp / "in.json"), "--object-info", str(tmp / "info.json"), "-o", str(tmp / "out.json")])
            self.assertEqual(code, 0)
            out = json.loads((tmp / "out.json").read_text())
            self.assertEqual((out["format_version"], out["manifest"]["model_name"]), (2, "in"))
            code = convert_workflow.main([str(tmp / "in.json"), "--object-info", str(tmp / "info.json"), "--api-only", "-o", str(tmp / "api.json")])
            self.assertEqual(json.loads((tmp / "api.json").read_text())["7"]["class_type"], "KSampler")

    def test_api_format_input_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "api.json").write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}))
            (tmp / "info.json").write_text("{}")
            self.assertEqual(convert_workflow.main([str(tmp / "api.json"), "--object-info", str(tmp / "info.json")]), 1)


if __name__ == "__main__":
    unittest.main()
