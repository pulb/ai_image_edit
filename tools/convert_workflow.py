# SPDX-License-Identifier: GPL-3.0-or-later
"""
Converts a ComfyUI workflow saved from the editor ("Save") into a workflow file
(doc/contributing/workflow_files.md): the API-format graph plus a manifest
scaffold.

    python tools/convert_workflow.py my_workflow.json --object-info http://127.0.0.1:8188 -o model.json
    python tools/convert_workflow.py my_workflow.json --object-info object_info.json --api-only

The editor format stores widget values by position, so the node definitions of
the ComfyUI that has the workflow's nodes installed are needed
(`--object-info`: a running ComfyUI's URL, or its /object_info saved to a file).

Converted: node links, widget values (including the extra "control after
generate" and upload values), muted nodes (dropped), bypassed nodes (skipped),
Reroute and PrimitiveNode. Not supported: subgraphs and group nodes.

The manifest is a starting point, not a finished model. It binds the first
sampler, the text encoders its conditioning comes from, the first LoadImage and
an empty latent image, and lists the files the model loaders name, without a
download URL. The notes the tool prints say what is left to check or fill in
(download URLs, licenses, custom nodes, reference images, LoRAs).
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FORMAT_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "ai_image_edit" / "data" / "workflow.schema.json"

UI_ONLY = {"Note", "MarkdownNote"}
MODE_MUTED, MODE_BYPASSED = 2, 4
SCALAR_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
UPLOAD_KEYS = ("image_upload", "video_upload", "audio_upload")

# (class, input) -> folder below the ComfyUI root, for the files a model needs.
FILE_INPUTS = {
    ("UNETLoader", "unet_name"): "models/diffusion_models",
    ("UnetLoaderGGUF", "unet_name"): "models/diffusion_models",
    ("CLIPLoader", "clip_name"): "models/text_encoders",
    ("DualCLIPLoader", "clip_name1"): "models/text_encoders",
    ("DualCLIPLoader", "clip_name2"): "models/text_encoders",
    ("VAELoader", "vae_name"): "models/vae",
    ("CheckpointLoaderSimple", "ckpt_name"): "models/checkpoints",
    ("ControlNetLoader", "control_net_name"): "models/controlnet",
    ("CLIPVisionLoader", "clip_name"): "models/clip_vision",
    ("UpscaleModelLoader", "model_name"): "models/upscale_models",
}
LORA_LOADERS = {"LoraLoader", "LoraLoaderModelOnly"}
SAMPLERS = {"KSampler": "seed", "KSamplerAdvanced": "noise_seed"}
LATENT_SIZE_NODES = {"EmptyLatentImage", "EmptySD3LatentImage"}
# A file's `url` until it is filled in; the schema requires one.
URL_PLACEHOLDER = "https://"


class ConvertError(Exception):
    pass


def load_object_info(source: str) -> dict:
    """A running ComfyUI's /object_info (give its URL) or a file saved from it."""
    if source.startswith(("http://", "https://")):
        url = source.rstrip("/")
        if not url.endswith("/object_info"):
            url += "/object_info"
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def _links(editor: dict) -> Dict[int, Tuple[int, int]]:
    """link id -> (origin node id, origin slot); links are lists or, in newer files, objects."""
    result = {}
    for link in editor.get("links", []):
        if isinstance(link, dict):
            result[link["id"]] = (link["origin_id"], link["origin_slot"])
        else:
            result[link[0]] = (link[1], link[2])
    return result


def _widgets(info: dict) -> List[Tuple[str, str, dict]]:
    """(name, type, config) of the inputs that are widgets, in the order of widgets_values."""
    spec = info.get("input", {})
    order = info.get("input_order") or {}
    result = []
    for section in ("required", "optional"):
        names = order.get(section) or list(spec.get(section, {}))
        for name in names:
            entry = spec.get(section, {}).get(name)
            if not entry:
                continue
            kind = entry[0]
            config = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if config.get("forceInput"):
                continue
            if isinstance(kind, list):
                result.append((name, "COMBO", config))
            elif kind in SCALAR_TYPES:
                result.append((name, kind, config))
    return result


def _widget_values(node: dict, widgets: List[Tuple[str, str, dict]]) -> Tuple[Dict[str, Any], List[str]]:
    values = node.get("widgets_values", [])
    notes = []
    if isinstance(values, dict):
        return {name: values[name] for name, _, _ in widgets if name in values}, notes
    result, i = {}, 0
    for name, kind, config in widgets:
        if i >= len(values):
            notes.append(f"node {node['id']} ({node['type']}): no value for '{name}'")
            continue
        result[name] = values[i]
        i += 1
        if kind == "INT" and (config.get("control_after_generate") or name in ("seed", "noise_seed")):
            i += 1  # the "randomize / fixed / ..." value of the frontend's control widget
        elif any(config.get(key) for key in UPLOAD_KEYS):
            i += 1  # the upload button's value
    if i < len(values):
        notes.append(f"node {node['id']} ({node['type']}): {len(values) - i} widget value(s) not used")
    return result, notes


def convert_graph(editor: dict, object_info: dict) -> Tuple[Dict[str, dict], List[str]]:
    """The API-format graph of an editor-format workflow, and notes about what was dropped or guessed."""
    if editor.get("definitions", {}).get("subgraphs"):
        raise ConvertError("The workflow uses subgraphs, which are not supported: unpack them in the editor first.")
    nodes = {n["id"]: n for n in editor.get("nodes", [])}
    links = _links(editor)
    notes: List[str] = []

    def input_link(node: dict, index: int) -> Optional[int]:
        inputs = node.get("inputs", [])
        return inputs[index].get("link") if index < len(inputs) else None

    def resolve(origin: int, slot: int, depth: int = 0) -> Optional[Tuple[str, Any]]:
        """Follow Reroute, bypassed and Primitive nodes: ('link', [id, slot]), ('value', v) or None (muted)."""
        if depth > 100:
            raise ConvertError("Nodes are connected in a loop.")
        node = nodes[origin]
        mode = node.get("mode", 0)
        if mode == MODE_MUTED:
            return None
        if node["type"] == "PrimitiveNode":
            return "value", (node.get("widgets_values") or [None])[0]
        if node["type"] == "Reroute" or mode == MODE_BYPASSED:
            wanted = None
            if node["type"] != "Reroute":
                outputs = node.get("outputs", [])
                wanted = outputs[slot].get("type") if slot < len(outputs) else None
            for index, inp in enumerate(node.get("inputs", [])):
                link = input_link(node, index)
                if link is not None and (wanted is None or inp.get("type") == wanted):
                    return resolve(*links[link], depth + 1)
            raise ConvertError(f"Bypassed node {origin} ({node['type']}) has nothing to pass through.")
        return "link", [str(origin), slot]

    api: Dict[str, dict] = {}
    for node_id, node in nodes.items():
        kind = node["type"]
        if kind in UI_ONLY or kind in ("Reroute", "PrimitiveNode") or node.get("mode", 0) in (MODE_MUTED, MODE_BYPASSED):
            continue
        info = object_info.get(kind)
        if info is None:
            raise ConvertError(f"Node {node_id} is of class '{kind}', which the object_info does not know "
                               "(a custom node that is not installed in that ComfyUI?).")
        widgets = _widgets(info)
        values, widget_notes = _widget_values(node, widgets)
        notes += widget_notes
        inputs: Dict[str, Any] = {}
        widget_names = {name for name, _, _ in widgets}
        for name in widget_names:
            if name in values:
                inputs[name] = values[name]
        for index, inp in enumerate(node.get("inputs", [])):
            link = input_link(node, index)
            if link is None:
                continue
            resolved = resolve(*links[link])
            if resolved is None:
                notes.append(f"node {node_id} ({kind}): input '{inp['name']}' comes from a muted node and was left out")
                inputs.pop(inp["name"], None)
            else:
                inputs[inp["name"]] = resolved[1]
        api[str(node_id)] = {"class_type": kind, "inputs": inputs}
    return api, notes


def _is_ref(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _text_input(node: dict, preferred: Tuple[str, ...]) -> Optional[str]:
    for name in preferred:
        if isinstance(node["inputs"].get(name), str):
            return name
    return None


def scaffold_manifest(api: Dict[str, dict], object_info: dict, name: str) -> Tuple[dict, List[str]]:
    """A manifest for the converted graph; the notes say what could not be derived."""
    notes: List[str] = []
    manifest: Dict[str, Any] = {"model_name": name}

    custom = sorted({n["class_type"] for n in api.values()
                     if str(object_info.get(n["class_type"], {}).get("python_module", "")).startswith("custom_nodes")})
    if custom:
        manifest["required_nodes"] = custom
        notes.append("custom node classes used: " + ", ".join(custom) +
                     ". Add them to `custom_nodes` (git URL and exact commit).")

    files = []
    for node_id, node in api.items():
        for input_name, value in list(node["inputs"].items()):
            folder = FILE_INPUTS.get((node["class_type"], input_name))
            if folder is None or not isinstance(value, str):
                continue
            files.append({"name": value, "url": URL_PLACEHOLDER, "folder": folder, "set": f"{node_id}.{input_name}"})
            node["inputs"][input_name] = ""
            notes.append(f"{node_id}.{input_name} ('{value}'): fill in its `url`, and `sha256`, `size` and `license` "
                         "if known.")
        if node["class_type"] in LORA_LOADERS:
            notes.append(f"node {node_id} is a LoRA loader: remove it and describe the LoRAs in `loras` instead.")
    manifest["files"] = files

    samplers = [(i, n) for i, n in api.items() if n["class_type"] in SAMPLERS]
    bind: Dict[str, str] = {}
    capabilities: Dict[str, Any] = {"supports_inpainting": True, "num_annotation_colors": 3}
    if not samplers:
        notes.append("no KSampler or KSamplerAdvanced found: fill in `bind` (seed, steps, sampler, scheduler) and `capabilities`.")
    else:
        if len(samplers) > 1:
            notes.append(f"{len(samplers)} samplers found; the first (node {samplers[0][0]}) is bound.")
        sampler_id, sampler = samplers[0]
        inputs = sampler["inputs"]
        for key, input_name in (("seed", SAMPLERS[sampler["class_type"]]), ("steps", "steps"), ("sampler", "sampler_name"),
                                ("scheduler", "scheduler"), ("cfg", "cfg"), ("denoise", "denoise")):
            if input_name in inputs and not _is_ref(inputs[input_name]):
                bind[key] = f"{sampler_id}.{input_name}"
        steps = inputs.get("steps") if isinstance(inputs.get("steps"), int) else 20
        capabilities["sampler"] = {"default": inputs.get("sampler_name", "euler")}
        capabilities["scheduler"] = {"default": inputs.get("scheduler", "normal")}
        capabilities["steps"] = {"min": 1, "max": max(60, steps), "default": steps, "step": 1}
        if "cfg" in bind:
            cfg = float(inputs["cfg"])
            capabilities["cfg"] = {"min": min(1.0, cfg), "max": max(10.0, cfg), "default": cfg, "step": 0.1}
        if "denoise" in bind:
            capabilities["denoise"] = {"min": 0.0, "max": 1.0, "default": float(inputs["denoise"]), "step": 0.01}
        positive, negative = inputs.get("positive"), inputs.get("negative")
        if _is_ref(positive) and positive[0] in api:
            encoder = api[positive[0]]
            field = _text_input(encoder, ("prompt", "text"))
            if field:
                bind["prompt"] = f"{positive[0]}.{field}"
        if _is_ref(negative) and negative[0] in api:
            encoder = api[negative[0]]
            field = _text_input(encoder, ("negative_prompt",) if negative[0] == positive[0] else ("prompt", "text"))
            if field:
                bind["negative_prompt"] = f"{negative[0]}.{field}"
        if "prompt" not in bind:
            notes.append("could not find the positive prompt: set `bind.prompt`.")
    manifest["bind"] = bind
    manifest["capabilities"] = capabilities
    manifest["seed"] = {"min": 0, "max": 2147483647, "wrap": True}

    loaders = sorted((i for i, n in api.items() if n["class_type"] == "LoadImage"), key=int)
    if loaders:
        manifest["images"] = {"source": f"{loaders[0]}.image"}
        if len(loaders) > 1:
            notes.append(f"LoadImage nodes {', '.join(loaders[1:])} could be reference images: describe them in `images.references`.")
    else:
        notes.append("no LoadImage node found: set `images.source`.")

    sizes = [(i, n) for i, n in api.items() if n["class_type"] in LATENT_SIZE_NODES and
             all(isinstance(n["inputs"].get(k), int) for k in ("width", "height"))]
    if sizes:
        size_id, size_node = sizes[0]
        resolution = round(math.sqrt(size_node["inputs"]["width"] * size_node["inputs"]["height"]))
        manifest["size"] = {"policy": "fixed_area", "resolutions": [resolution], "multiple": 8,
                            "bind": {"width": f"{size_id}.width", "height": f"{size_id}.height"}}
    else:
        notes.append("no empty latent image with a width and height found: set `size` (see sizes.py).")
    return manifest, notes


def validate(workflow_file: dict) -> List[str]:
    """Problems according to the JSON Schema, if jsonschema is installed."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed, the result was not validated (pip install jsonschema)."]
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(schema)
    return [f"schema: {'/'.join(map(str, e.absolute_path)) or '(file)'}: {e.message}" for e in validator.iter_errors(workflow_file)]


def convert(editor: dict, object_info: dict, name: str, api_only: bool = False) -> Tuple[dict, List[str]]:
    api, notes = convert_graph(editor, object_info)
    if api_only:
        return api, notes
    manifest, manifest_notes = scaffold_manifest(api, object_info, name)
    result = {"format_version": FORMAT_VERSION, "manifest": manifest, "workflow": api}
    return result, notes + manifest_notes + validate(result)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert an editor-format ComfyUI workflow into a workflow file.")
    parser.add_argument("workflow", help="the workflow saved from the ComfyUI editor")
    parser.add_argument("--object-info", required=True, metavar="URL_OR_FILE",
                        help="a running ComfyUI's URL, or its /object_info saved to a file")
    parser.add_argument("--name", help="model name for the manifest (default: the file name)")
    parser.add_argument("--api-only", action="store_true", help="write only the API-format graph")
    parser.add_argument("-o", "--output", help="output file (default: standard output)")
    args = parser.parse_args(argv)
    try:
        with open(args.workflow, "r", encoding="utf-8") as f:
            editor = json.load(f)
        if "nodes" not in editor:
            raise ConvertError("This is not an editor-format workflow (no 'nodes'); API-format graphs need no conversion.")
        result, notes = convert(editor, load_object_info(args.object_info), args.name or Path(args.workflow).stem, args.api_only)
    except (ConvertError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
