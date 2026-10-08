# SPDX-License-Identifier: GPL-3.0-or-later
"""
A ModelBackend that is described by data instead of code: a folder with a
ComfyUI workflow (workflow_api.json, API format) and a manifest.json that says
how the app's inputs map onto that workflow.

    models/<name>/manifest.json
    models/<name>/workflow_api.json

A model is registered by creating such a folder (see models/__init__.py). The
manifest format is documented in doc/contribution/comfy_manifest.md.

Workflow targets are written "<node id>.<input name>", e.g. "6.seed"; the
input name may itself contain dots ("5.images.image_2").
"""
import copy
import json
import os
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.result_cache import cached_infer
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.models.base import ModelBackend
from ai_image_edit.models._shared.comfyui import client as comfy_client
from ai_image_edit.models._shared.comfyui.common import SAMPLER_CHOICES, SCHEDULER_CHOICES, configured_file
from ai_image_edit.models._shared.comfyui.size_policies import POLICIES, ResolvedSize, SizePolicy

_NAMED_CHOICES = {"comfy_samplers": SAMPLER_CHOICES, "comfy_schedulers": SCHEDULER_CHOICES}
_BINDABLE = {"prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "denoise"}


def _split(target: str) -> Tuple[str, str]:
    node, _, key = target.partition(".")
    if not node or not key:
        raise ValueError(f"invalid workflow target {target!r}: expected '<node id>.<input name>'")
    return node, key


def _range(spec: dict) -> RangeSpec:
    return RangeSpec(spec["min"], spec["max"], spec["default"], step=spec.get("step", 1.0))


def _choices(value: Any) -> List[str]:
    return list(_NAMED_CHOICES[value]) if isinstance(value, str) else list(value)


class ComfyWorkflowModel(ModelBackend):
    """A model run through a local ComfyUI server, as described by a manifest."""

    def __init__(self, name: str, manifest: dict, workflow: dict) -> None:
        self.name = name
        self._manifest = manifest
        self._workflow = workflow
        self._process: Optional[subprocess.Popen] = None
        self._nodes_checked = False

        size = manifest["size"]
        if size["policy"] not in POLICIES:
            raise ValueError(f"{name}: unknown size policy {size['policy']!r} (known: {', '.join(POLICIES)})")
        self._size: SizePolicy = POLICIES[size["policy"]](size)
        self._validate()

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ComfyWorkflowModel":
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(manifest_path.parent / manifest.get("workflow", "workflow_api.json"), "r", encoding="utf-8") as f:
            workflow = json.load(f)
        return cls(manifest_path.parent.name, manifest, workflow)

    # ------------------------------------------------------------ validation

    def _validate(self) -> None:
        """Catches manifest mistakes (typos in node ids, unknown settings) at load time."""
        m, wf = self._manifest, self._workflow

        def need_node(target: str, what: str) -> None:
            node, key = _split(target)
            if node not in wf:
                raise ValueError(f"{self.name}: {what} {target!r} refers to node {node!r}, which is not in the workflow")
            # The template carries a placeholder for every input the app fills in, so a name
            # that is not there is a typo (it would otherwise silently add an unknown input).
            if key not in wf[node].get("inputs", {}):
                raise ValueError(f"{self.name}: {what} {target!r}: node {node!r} has no input {key!r} in the workflow")

        unknown = set(m.get("bind", {})) - _BINDABLE
        if unknown:
            raise ValueError(f"{self.name}: unknown bind name(s) {sorted(unknown)} (known: {sorted(_BINDABLE)})")
        for key in ("prompt", "seed", "steps", "sampler", "scheduler"):
            if key not in m.get("bind", {}):
                raise ValueError(f"{self.name}: bind.{key} is required")
        for key, target in m["bind"].items():
            need_node(target, f"bind.{key}")
        for target in m.get("constants", {}):
            need_node(target, "constants")
        for spec in m.get("files", []):
            need_node(spec["set"], "files")
        need_node(m["images"]["source"], "images.source")
        refs = m["images"].get("references")
        if refs:
            ref_node, _ = _split(refs["target"].format(n=refs.get("first_n", 2)))
            if ref_node not in wf:
                raise ValueError(f"{self.name}: images.references.target refers to node {ref_node!r}, which is not in the workflow")
            ids = refs.get("loader_ids")
            if ids is None and "loader_id_start" not in refs:
                raise ValueError(f"{self.name}: images.references needs loader_ids or loader_id_start")
            if ids is not None and len(ids) < refs["max"]:
                raise ValueError(f"{self.name}: images.references.loader_ids has fewer entries than max")
        for key, target in m["size"].get("bind", {}).items():
            need_node(target, f"size.bind.{key}")
        loras = m.get("loras")
        if loras:
            for targets in loras["feeds"].values():
                for target in targets:
                    need_node(target, "loras.feeds")

    # ------------------------------------------------------------ identity

    @property
    def model_name(self) -> str:
        return self._manifest["model_name"]

    @property
    def model_version(self) -> Optional[str]:
        spec = self._manifest.get("model_version")
        if not spec:
            return None
        value = os.environ.get(spec["env"], "")
        if spec.get("basename"):
            value = Path(value).name
        flags = re.IGNORECASE if spec.get("ignore_case") else 0
        match = re.search(spec["regex"], value, flags)
        if not match:
            return None
        version = match.group(1).upper() if spec.get("upper") else match.group(1)
        prefix = "".join(p["text"] for p in spec.get("prefixes", []) if re.search(p["regex"], value))
        return prefix + version

    @property
    def _max_references(self) -> int:
        refs = self._manifest["images"].get("references")
        return refs["max"] if refs else 0

    @property
    def capabilities(self) -> ModelCapabilities:
        m, caps = self._manifest, self._manifest["capabilities"]
        bind = m["bind"]
        kwargs: Dict[str, Any] = dict(
            max_reference_images=1 + self._max_references,  # the source image counts
            supports_loras=bool(m.get("loras")),
            supports_inpainting=caps["supports_inpainting"],
            supported_aspect_ratios=list(self._size.aspect_ratios),
            default_aspect_ratio=self._size.default_aspect_ratio,
            supported_megapixels=list(self._size.megapixels),
            default_megapixels=self._size.default_megapixels,
            sampler_choices=_choices(caps["sampler"].get("choices", "comfy_samplers")),
            default_sampler=caps["sampler"]["default"],
            scheduler_choices=_choices(caps["scheduler"].get("choices", "comfy_schedulers")),
            default_scheduler=caps["scheduler"]["default"],
            supports_cfg="cfg" in bind,
            supports_denoise="denoise" in bind,
            supports_seed=True,
            supports_negative_prompt="negative_prompt" in bind,
            num_annotation_colors=caps.get("num_annotation_colors", 0),
            step_range=_range(caps["steps"]),
        )
        for field, key in (("cfg_range", "cfg"), ("denoise_range", "denoise"), ("lora_strength_range", "lora_strength")):
            if key in caps:
                kwargs[field] = _range(caps[key])
        return ModelCapabilities(**kwargs)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        # Checked before ComfyUI starts, so a missing setting or file fails fast.
        for spec in self._manifest.get("files", []):
            configured_file(spec["env"], Path(spec["folder"]))
        self._process = comfy_client.launch_comfy_process()

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.terminate()

    def megapixels_for_source(self, source_image_path: str) -> Optional[float]:
        return self._size.megapixels_for_source(source_image_path)

    def list_loras(self) -> Dict[str, List[str]]:
        """
        Available LoRAs in the manifest's loras.dir, keyed by display name, each
        mapped to its weight file(s) as paths relative to that folder, which is
        what ComfyUI's LoraLoader node expects. A LoRA is either a single
        .safetensors/.pt file directly in the folder (display name = filename),
        or a subfolder with one or more such files (display name = subfolder
        name) — some LoRAs ship as several files that all need loading together.
        """
        loras_cfg = self._manifest.get("loras")
        if not loras_cfg:
            return {}
        lora_dir = loras_cfg["dir"]
        if not os.path.exists(lora_dir):
            print(f"[WARNING] LoRA folder not found at expected path: {lora_dir}", flush=True)
            return {}

        loras: Dict[str, List[str]] = {}
        for entry in sorted(os.listdir(lora_dir)):
            entry_path = os.path.join(lora_dir, entry)
            if os.path.isdir(entry_path):
                files = sorted(
                    os.path.join(entry, f)
                    for f in os.listdir(entry_path)
                    if f.endswith((".safetensors", ".pt"))
                )
                if files:
                    loras[entry] = files
            elif entry.endswith((".safetensors", ".pt")):
                loras[entry] = [entry]

        if not loras:
            print(f"[WARNING] No LoRA files found in {lora_dir}", flush=True)
        return loras

    def _check_nodes(self) -> None:
        """
        Fails with a clear message if ComfyUI lacks a node the workflow needs.
        Done on the first generation rather than in start(), because ComfyUI
        is still booting then; while it is, this raises the usual "still
        starting" error and is retried on the next generation.
        """
        required = self._manifest.get("required_nodes", [])
        if self._nodes_checked or not required:
            return
        missing = comfy_client.missing_nodes(required)
        if missing:
            hint = self._manifest.get("missing_nodes_hint", "install the custom node(s) that provide them.")
            raise GenerationError(f"ComfyUI is missing node(s) {', '.join(missing)}: {hint}")
        self._nodes_checked = True

    # ------------------------------------------------------------ workflow

    def build_workflow(
        self, *, prompt: str, negative_prompt: str = "", source_path: str, reference_paths: List[str],
        seed: int, steps: int, cfg: float = 1.0, denoise: float = 1.0, sampler: str, scheduler: str,
        lora_files: Optional[List[str]] = None, lora_strength: float = 0.7, size_values: Dict[str, Any],
    ) -> dict:
        """The workflow template with the fields for one generation filled in."""
        m = self._manifest
        wf = copy.deepcopy(self._workflow)

        def put(target: str, value: Any) -> None:
            node, key = _split(target)
            wf[node]["inputs"][key] = value

        for spec in m.get("files", []):
            put(spec["set"], configured_file(spec["env"], Path(spec["folder"])))
        for target, value in m.get("constants", {}).items():
            put(target, value)

        put(m["images"]["source"], comfy_client.upload_image(source_path))
        self._place_references(wf, reference_paths)

        values = {
            "prompt": prompt.strip(), "negative_prompt": negative_prompt or "", "seed": seed, "steps": int(steps),
            "cfg": float(cfg), "denoise": float(denoise), "sampler": sampler, "scheduler": scheduler,
        }
        for name, target in m["bind"].items():
            put(target, values[name])
        for name, target in m["size"].get("bind", {}).items():
            put(target, size_values[name])

        if lora_files:
            self._splice_loras(wf, lora_files, lora_strength)
        return wf

    def _place_references(self, wf: dict, reference_paths: List[str]) -> None:
        """
        Each reference image is a LoadImage node wired to the next free image
        input of the encode node. Slots the request does not use are removed,
        in case the template ships them.
        """
        refs = self._manifest["images"].get("references")
        if not refs:
            return
        first_n = refs.get("first_n", 2)
        for i in range(refs["max"]):
            loader_id = refs["loader_ids"][i] if "loader_ids" in refs else str(refs["loader_id_start"] + i)
            node, key = _split(refs["target"].format(n=i + first_n))
            if i < len(reference_paths):
                uploaded = comfy_client.upload_image(reference_paths[i])
                if loader_id in wf:
                    wf[loader_id]["inputs"]["image"] = uploaded
                else:
                    wf[loader_id] = {"class_type": "LoadImage", "inputs": {"image": uploaded, "upload": "image"}}
                wf[node]["inputs"][key] = [loader_id, 0]
            else:
                wf.pop(loader_id, None)
                wf[node]["inputs"].pop(key, None)

    def _splice_loras(self, wf: dict, lora_files: List[str], strength: float) -> None:
        """
        Chains one loader node per file after the checkpoint, each taking the
        previous one's outputs, and points everything that consumed the
        checkpoint's outputs at the last loader. Only the outputs listed in the
        manifest (model/clip) are touched; the VAE is not.
        """
        cfg = self._manifest["loras"]
        links = {name: list(link) for name, link in cfg["from"].items()}
        for i, lora_file in enumerate(lora_files):
            node_id = str(cfg["first_id"] + i)
            inputs: Dict[str, Any] = dict(links)
            inputs["lora_name"] = lora_file
            for strength_input in cfg["strength_inputs"]:
                inputs[strength_input] = float(strength)
            wf[node_id] = {"class_type": cfg["loader_class"], "inputs": inputs}
            links = {name: [node_id, slot] for slot, name in enumerate(cfg["outputs"])}
        for name, targets in cfg["feeds"].items():
            for target in targets:
                node, key = _split(target)
                wf[node]["inputs"][key] = links[name]

    # ------------------------------------------------------------ generation

    def _pick_seed(self, params: GenerationParams) -> int:
        spec = self._manifest.get("seed", {"min": 0, "max": 2 ** 32 - 1})
        if params.randomize_seed:
            return random.randint(spec["min"], spec["max"])
        seed = int(params.seed)
        return seed % (spec["max"] + 1) if spec.get("wrap") else seed

    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Resolves the target size, then hands off to imaging.run_masked_generation
        for the crop -> infer -> composite sequence; _infer() builds and submits
        the ComfyUI workflow and fetches the result. The source is cropped to
        the generation size first, so the output matches; reference images are
        passed as they are.
        """
        prompt = (params.prompt or "").strip()
        if not prompt:
            raise GenerationError("Please enter a prompt.")
        max_images = 1 + self._max_references
        if 1 + len(params.reference_images) > max_images:
            raise GenerationError(f"Up to {max_images} input images are supported.")

        seed = self._pick_seed(params)
        size: ResolvedSize = self._size.resolve(params)
        print(
            f"[{self.name}] mask={'yes' if params.mask_path else 'no'} images={1 + len(params.reference_images)} "
            f"gen_dims={size.width}x{size.height} {size.description} steps={params.steps} cfg={params.cfg} "
            f"sampler={params.sampler_name} scheduler={params.scheduler}",
            flush=True,
        )

        def _infer(model_input_path: str) -> str:
            self._check_nodes()
            workflow = self.build_workflow(
                prompt=prompt, negative_prompt=params.negative_prompt, source_path=model_input_path,
                reference_paths=params.reference_images, seed=seed, steps=params.steps, cfg=params.cfg,
                denoise=params.denoise, sampler=params.sampler_name, scheduler=params.scheduler,
                lora_files=params.lora_files, lora_strength=params.lora_strength, size_values=size.values,
            )
            t0 = time.time()
            prompt_id = comfy_client.submit_workflow_and_wait(workflow, str(uuid.uuid4()))
            print(f"[{self.name}] ComfyUI generation took {time.time() - t0:.1f}s", flush=True)
            return comfy_client.fetch_generated_image(prompt_id)

        infer = cached_infer(_infer, self.display_name, params, seed=seed, **size.cache_kwargs, width=size.width, height=size.height)

        final_output_path = imaging.run_masked_generation(
            params.source_image_path,
            params.mask_path,
            size.width,
            size.height,
            params.feather_amount,
            params.apply_color_correction_enabled,
            infer,
            params.annotated_image_path,
        )
        return GenerationResult(before_path=params.source_image_path, after_path=final_output_path, actual_seed=seed)
