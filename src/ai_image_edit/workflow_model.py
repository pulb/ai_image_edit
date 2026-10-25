# SPDX-License-Identifier: GPL-3.0-or-later
"""
The model the app runs: a workflow file (ai_image_edit/data/workflows/<name>.json) with a
ComfyUI workflow in API format and a manifest that says how the app's inputs map
onto that workflow. A bundled file is found by its name, any other by its path
(see get_model()); the format is documented in doc/contributing/workflow_files.md.

Workflow targets are written "<node id>.<input name>", e.g. "6.seed"; the
input name may itself contain dots ("5.images.image_2").
"""
import copy
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ai_image_edit.comfy import client as comfy_client
from ai_image_edit.comfy import custom_nodes, downloads, gpu
from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationCancelled, GenerationError
from ai_image_edit.core.result_cache import cached_infer
from ai_image_edit.core.types import (
    GenerationParams, GenerationProgress, GenerationResult, Inference, InputSpec, ModelCapabilities, ProgressStage,
    RangeSpec,
)
from ai_image_edit.sizes import POLICIES, ResolvedSize, SizePolicy, dimensions_from_source
from ai_image_edit.workflows import available_workflows, read_workflow_file


# Valid ComfyUI KSampler scheduler names (from comfy/samplers.py's SCHEDULER_HANDLERS).
SCHEDULER_CHOICES = ["normal", "karras", "exponential", "simple", "ddim_uniform", "beta", "sgm_uniform", "linear_quadratic", "kl_optimal"]

# Valid ComfyUI KSampler sampler names (comfy/samplers.py's KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]).
SAMPLER_CHOICES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2"
]


@dataclass(frozen=True)
class SetupProgress:
    """How far the model's background setup (downloads, installs, the start of ComfyUI) is."""

    done: bool = True
    message: str = ""  # what is being done right now, e.g. "Downloading x.safetensors (2/5): 42%"
    error: Optional[str] = None
    fraction: Optional[float] = None  # 0 to 1 over the whole setup, None while that is not known


# Seconds between the checks whether a freshly launched ComfyUI answers yet.
COMFY_POLL_SECONDS = 0.5

_NAMED_CHOICES = {"comfy_samplers": SAMPLER_CHOICES, "comfy_schedulers": SCHEDULER_CHOICES}
# The bindings every workflow has; the app fills them in from Inference's own fields.
REQUIRED_BINDINGS = ("prompt", "seed", "steps")


def _split(target: str) -> Tuple[str, str]:
    node, _, key = target.partition(".")
    if not node or not key:
        raise ValueError(f"invalid workflow target {target!r}: expected '<node id>.<input name>'")
    return node, key


def _range(spec: dict) -> RangeSpec:
    return RangeSpec(spec["min"], spec["max"], spec["default"], step=spec.get("step", 1.0))


def _choices(value: Any) -> List[str]:
    return list(_NAMED_CHOICES[value]) if isinstance(value, str) else list(value)


class WorkflowModel:
    """
    A model run through a local ComfyUI server, as described by a workflow file.

    The UI needs four things from it: what it can do (capabilities), how to
    start and stop it (start(), shutdown(); setup_progress() reports the
    downloads and the start of ComfyUI that start() continues in the
    background), and how to run one generation (generate()), plus a name
    and version to show.

    lora_downloads=False skips downloading the workflow's own LoRA files
    (LoRAs already in the folder are still offered).
    """

    def __init__(
        self, name: str, manifest: dict, workflow: dict, variant: Optional[str] = None, *, lora_downloads: bool = True,
    ) -> None:
        self.name = name
        self._check_variants(name, manifest)
        self._manifest = self._with_variant(name, manifest, variant)
        self._workflow = workflow
        self._lora_downloads = lora_downloads
        self._process: Optional[subprocess.Popen] = None
        self._nodes_checked = False
        self._download_status: Optional[downloads.DownloadStatus] = None
        self._cancel_downloads = threading.Event()
        self._comfy_ready = threading.Event()  # set once the launched ComfyUI answers
        self._comfy_error: Optional[str] = None  # set if its process ends before that
        self._lora_warned: set = set()

        size = manifest["size"]
        if size["policy"] not in POLICIES:
            raise ValueError(f"{name}: unknown size policy {size['policy']!r} (known: {', '.join(POLICIES)})")
        self._size: SizePolicy = POLICIES[size["policy"]](size)
        self._validate()

    @classmethod
    def from_file(cls, path: Path, variant: Optional[str] = None, **options: bool) -> "WorkflowModel":
        """The model described by a workflow file; its name is the file name without .json. options are __init__'s."""
        path = Path(path)
        manifest, workflow = read_workflow_file(path)
        return cls(path.stem, manifest, workflow, variant, **options)

    @staticmethod
    def _check_variants(name: str, manifest: dict) -> None:
        """Every variant may only change files the manifest lists, and only name licenses it defines."""
        targets = {spec["set"] for spec in manifest["files"]}
        for variant, chosen in manifest.get("variants", {}).items():
            for target, changes in chosen.get("files", {}).items():
                if target not in targets:
                    raise ValueError(f"{name}: variant {variant!r} changes {target!r}, which is not in files")
                if changes.get("license") and changes["license"] not in manifest.get("licenses", {}):
                    raise ValueError(f"{name}: variant {variant!r} changes {target!r}: unknown license {changes['license']!r}")

    @staticmethod
    def _with_variant(name: str, manifest: dict, variant: Optional[str]) -> dict:
        """
        The manifest with a variant applied: its files replace the entries of the
        same 'set' target (only the keys it gives; a variant that names another
        file or url does not inherit the sha256 and size of the original), and
        its model_version replaces the file's.
        """
        if not variant:
            return manifest
        variants = manifest.get("variants", {})
        if variant not in variants:
            options = ", ".join(variants) or "none"
            raise ValueError(f"{name}: unknown variant {variant!r} (available: {options})")
        manifest = copy.deepcopy(manifest)
        chosen = variants[variant]
        for target, changes in chosen.get("files", {}).items():
            spec = next(spec for spec in manifest["files"] if spec["set"] == target)  # see _check_variants()
            if "name" in changes or "url" in changes:
                for key in ("sha256", "size"):
                    spec.pop(key, None)
            spec.update(changes)
        if "model_version" in chosen:
            manifest["model_version"] = chosen["model_version"]
        return manifest

    # ------------------------------------------------------------ validation

    def _validate(self) -> None:
        """
        Catches the manifest mistakes the schema cannot see (typos in node ids,
        unknown licenses) at load time. Of what the schema checks, only size.bind
        is checked again, against the policy class, which keeps sizes.py and the
        schema in step.
        """
        m, wf = self._manifest, self._workflow

        def need_node(target: str, what: str) -> None:
            node, key = _split(target)
            if node not in wf:
                raise ValueError(f"{self.name}: {what} {target!r} refers to node {node!r}, which is not in the workflow")
            # The template carries a placeholder for every input the app fills in, so a name
            # that is not there is a typo (it would otherwise silently add an unknown input).
            if key not in wf[node].get("inputs", {}):
                raise ValueError(f"{self.name}: {what} {target!r}: node {node!r} has no input {key!r} in the workflow")

        for key, target in m["bind"].items():
            need_node(target, f"bind.{key}")
        for name, spec in m.get("inputs", {}).items():
            need_node(spec["target"], f"inputs.{name}")
            if name in REQUIRED_BINDINGS:
                raise ValueError(f"{self.name}: inputs.{name}: {name!r} is a binding every workflow has, not an optional input")
            node, key = _split(spec["target"])
            placeholder = wf[node]["inputs"][key]
            is_number = isinstance(placeholder, (int, float)) and not isinstance(placeholder, bool)
            if "range" in spec and not is_number:
                raise ValueError(f"{self.name}: inputs.{name} has a range, but its workflow input holds {placeholder!r}, not a number")
            if "range" not in spec and not isinstance(placeholder, str):
                raise ValueError(f"{self.name}: inputs.{name} needs a range: its workflow input holds {placeholder!r}, not text")
            if "choices" in spec and placeholder not in _choices(spec["choices"]):
                raise ValueError(f"{self.name}: inputs.{name}: its workflow input holds {placeholder!r}, which is not one of its choices")
        for target in m.get("constants", {}):
            need_node(target, "constants")
        for spec in m["files"]:
            need_node(spec["set"], "files")
        loras = m.get("loras")
        for spec in m["files"] + (loras.get("files", []) if loras else []):
            if spec.get("license") and spec["license"] not in m.get("licenses", {}):
                raise ValueError(f"{self.name}: files entry {spec.get('set') or spec['name']!r}: unknown license {spec['license']!r}")
        need_node(m["images"]["source"], "images.source")
        refs = m["images"].get("references")
        if refs:
            ref_node, _ = _split(refs["target"].format(n=refs.get("first_n", 2)))
            if ref_node not in wf:
                raise ValueError(f"{self.name}: images.references.target refers to node {ref_node!r}, which is not in the workflow")
            ids = refs.get("loader_ids")
            if ids is not None and len(ids) < refs["max"]:
                raise ValueError(f"{self.name}: images.references.loader_ids has fewer entries than max")
        bound = set(m["size"].get("bind", {}))
        if bound != self._size.value_names:
            raise ValueError(
                f"{self.name}: size.bind must bind exactly {sorted(self._size.value_names)} "
                f"for policy {m['size']['policy']!r}, not {sorted(bound)}"
            )
        for key, target in m["size"].get("bind", {}).items():
            need_node(target, f"size.bind.{key}")
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
        """The manifest's (or variant's) version."""
        return self._manifest.get("model_version")

    @property
    def display_name(self) -> str:
        """model_name followed by model_version, if any."""
        return f"{self.model_name} {self.model_version}" if self.model_version else self.model_name

    @property
    def _max_references(self) -> int:
        refs = self._manifest["images"].get("references")
        return refs["max"] if refs else 0

    @cached_property
    def capabilities(self) -> ModelCapabilities:
        m, caps = self._manifest, self._manifest["capabilities"]
        kwargs: Dict[str, Any] = dict(
            max_reference_images=1 + self._max_references,  # the source image counts
            supports_loras=bool(m.get("loras")),
            supports_inpainting=caps["supports_inpainting"],
            supported_aspect_ratios=list(self._size.aspect_ratios),
            default_aspect_ratio=self._size.default_aspect_ratio,
            supported_resolutions=list(self._size.resolutions),
            default_resolution=self._size.default_resolution,
            options=tuple(self._input_spec(name, spec) for name, spec in m.get("inputs", {}).items()),
            num_annotation_colors=caps.get("num_annotation_colors", 0),
            step_range=_range(caps["steps"]),
        )
        if "lora_strength" in caps:
            kwargs["lora_strength_range"] = _range(caps["lora_strength"])
        return ModelCapabilities(**kwargs)

    def _input_spec(self, name: str, spec: dict) -> InputSpec:
        """
        An optional input as the UI shows it: a slider over its range, or a
        dropdown of its choices or a text field, both starting with the workflow's
        placeholder. The label defaults to the name, with spaces for underscores.
        A slider takes whole numbers when its range and the placeholder are all
        whole numbers.
        """
        label = spec.get("label") or name.replace("_", " ").capitalize()
        node, key = _split(spec["target"])
        placeholder = self._workflow[node]["inputs"][key]
        if "range" not in spec:
            return InputSpec(name, label, default=placeholder, choices=tuple(_choices(spec.get("choices", []))))
        values = [spec["range"][k] for k in ("min", "max", "default", "step") if k in spec["range"]]
        integer = isinstance(placeholder, int) and all(float(v).is_integer() for v in values)
        rng = _range(spec["range"])
        return InputSpec(name, label, default=int(rng.default) if integer else rng.default, range=rng, integer=integer)

    # ------------------------------------------------------------ lifecycle

    def _model_files(self) -> List[Tuple[dict, Path]]:
        """The model's weight files (not the LoRAs), with where each one goes."""
        return [(spec, Path(spec["folder"]) / spec["name"]) for spec in self._manifest["files"]]

    def _pending_downloads(self) -> List[dict]:
        wanted = self._model_files()
        # LoRA files: each goes into its own subfolder of loras.dir (the subfolder is the display name).
        loras = self._manifest.get("loras", {})
        if self._lora_downloads:
            wanted += [(spec, Path(loras["dir"]) / spec["dir"] / spec["name"]) for spec in loras.get("files", [])]
        return [
            {"url": spec["url"], "dest": dest, "sha256": spec.get("sha256"), "size": spec.get("size")}
            for spec, dest in wanted if not dest.is_file()
        ]

    def _weights_bytes(self) -> int:
        """Total size of the model files (not the LoRAs), which ComfyUI loads for a generation."""
        return sum(path.stat().st_size for _, path in self._model_files() if path.is_file())

    def _launch_comfy(self) -> None:
        args, reason = gpu.comfy_extra_args(self._weights_bytes(), self._manifest.get("vram_headroom_gb"))
        print(f"ComfyUI arguments: {' '.join(args) or '(none)'} ({reason})", flush=True)
        self._process = comfy_client.launch_comfy_process(args)
        threading.Thread(target=self._watch_comfy, args=(self._process,), name=f"{self.name}-comfy-start", daemon=True).start()

    def _watch_comfy(self, process: subprocess.Popen) -> None:
        """Waits for the launched ComfyUI to answer, or notes that its process ended first."""
        while not self._cancel_downloads.is_set():
            code = process.poll()
            if code is not None:
                self._comfy_error = f"ComfyUI stopped while starting (exit code {code}). Check its log, then restart the app."
                return
            if comfy_client.is_ready():
                self._comfy_ready.set()
                return
            time.sleep(COMFY_POLL_SECONDS)

    def start(self) -> None:
        # Missing files and custom nodes are fetched in the background, then ComfyUI is launched.
        pending = self._pending_downloads()
        nodes = custom_nodes.missing(self._manifest.get("custom_nodes", []))
        if not pending and not nodes:
            self._launch_comfy()
            return
        status = self._download_status = downloads.DownloadStatus()

        def prepare_and_launch() -> None:
            for spec in nodes:
                if self._cancel_downloads.is_set():
                    return
                status.message = f"Installing {spec['name']}"
                custom_nodes.install(spec)
            if not self._cancel_downloads.is_set():
                self._launch_comfy()

        threading.Thread(
            target=downloads.run_downloads, args=(pending, status, self._cancel_downloads, prepare_and_launch),
            name=f"{self.name}-setup", daemon=True,
        ).start()

    def setup_progress(self) -> SetupProgress:
        status = self._download_status
        if status is not None:
            if status.error:
                return SetupProgress(done=True, error=f"Setting up the model failed: {status.error}. Restart the app to retry.")
            if not status.finished.is_set():
                return SetupProgress(done=False, message=status.message, fraction=status.fraction)
        if self._process is None:
            return SetupProgress()
        if self._comfy_error:
            return SetupProgress(done=True, error=self._comfy_error)
        if not self._comfy_ready.is_set():
            return SetupProgress(done=False, message="Starting ComfyUI")
        return SetupProgress(fraction=status.fraction if status is not None else None)

    def _check_setup(self) -> None:
        progress = self.setup_progress()
        if progress.error:
            raise GenerationError(progress.error)
        if not progress.done:
            detail = f" ({progress.message})" if progress.message else ""
            raise GenerationError(f"The model is still being set up{detail}. Please wait a moment and try again.")

    def shutdown(self) -> None:
        self._cancel_downloads.set()
        if self._process is not None:
            self._process.terminate()

    def resolution_for_source(self, source_image_path: str) -> Optional[int]:
        return self._size.resolution_for_source(source_image_path)

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
        lora_dir = Path(loras_cfg["dir"])
        if not lora_dir.exists():
            self._warn_once(f"LoRA folder not found at expected path: {lora_dir}")
            return {}

        def is_weights(path: Path) -> bool:
            return path.suffix in (".safetensors", ".pt")

        loras: Dict[str, List[str]] = {}
        for entry in sorted(lora_dir.iterdir()):
            if entry.is_dir():
                files = sorted(str(Path(entry.name) / f.name) for f in entry.iterdir() if is_weights(f))
                if files:
                    loras[entry.name] = files
            elif is_weights(entry):
                loras[entry.name] = [entry.name]

        if not loras:
            self._warn_once(f"No LoRA files found in {lora_dir}")
        return loras

    def _warn_once(self, message: str) -> None:
        """list_loras() is called repeatedly while the LoRAs are being downloaded."""
        if message not in self._lora_warned:
            self._lora_warned.add(message)
            print(f"[WARNING] {message}", flush=True)

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

    @property
    def _option_targets(self) -> Dict[str, str]:
        """The optional inputs (Inference.options) by name, with their workflow targets."""
        return {name: spec["target"] for name, spec in self._manifest.get("inputs", {}).items()}

    def build_workflow(
        self, inference: Inference, source_path: str, reference_paths: Sequence[str], size_values: Dict[str, Any],
    ) -> dict:
        """The workflow template with the fields for one generation filled in; inference.seed must be resolved."""
        m = self._manifest
        wf = copy.deepcopy(self._workflow)

        def put(target: str, value: Any) -> None:
            node, key = _split(target)
            wf[node]["inputs"][key] = value

        for spec in m.get("files", []):
            put(spec["set"], spec["name"])
        for target, value in m.get("constants", {}).items():
            put(target, value)

        put(m["images"]["source"], comfy_client.upload_image(source_path))
        self._place_references(wf, reference_paths)

        required = {"prompt": inference.prompt.strip(), "seed": inference.seed, "steps": int(inference.steps)}
        for name in REQUIRED_BINDINGS:
            put(m["bind"][name], required[name])
        targets = self._option_targets
        for name, value in inference.options.items():
            put(targets[name], value)
        for name, target in m["size"].get("bind", {}).items():
            put(target, size_values[name])

        if inference.lora_files:
            self._splice_loras(wf, inference.lora_files, inference.lora_strength)
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
        cfg = self._manifest.get("loras")
        if cfg is None:
            raise ValueError(f"{self.name}: the workflow file has no 'loras' section, so it cannot use LoRAs")
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

    def _pick_seed(self, seed: Optional[int]) -> int:
        """The seed to use: a random one for None."""
        spec = self._manifest.get("seed", {"min": 0, "max": 2 ** 32 - 1})
        if seed is None:
            return random.randint(spec["min"], spec["max"])
        seed = int(seed)
        return seed % (spec["max"] + 1) if spec.get("wrap") else seed

    def generate(
        self,
        params: GenerationParams,
        on_progress: Optional[Callable[[GenerationProgress], None]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> GenerationResult:
        """
        Resolves the target size, then hands off to imaging.run_generation
        for the crop -> infer -> composite sequence; _infer() builds and submits
        the ComfyUI workflow and fetches the result. The source is fitted to
        size.input_size when the size policy sets one (the reference images are
        then scaled too, to size.reference_resolution), and to the generation
        size for any masked request. Otherwise the images are passed as they
        are and the model scales them. The output size comes from the
        workflow, through the values the manifest binds (size.bind).

        on_progress, if given, is called on this thread as the generation
        moves on (see ProgressStage); the sampler's steps are those of the
        node the steps are bound to. Setting cancel stops the generation
        (GenerationCancelled); only this request's ComfyUI job is stopped, see
        comfy_client.cancel_prompt().
        """
        def report(stage: ProgressStage) -> None:
            if on_progress is not None:
                on_progress(GenerationProgress(stage))

        def check_cancel() -> None:
            if cancel is not None and cancel.is_set():
                raise GenerationCancelled("Generation cancelled.")

        self._check_setup()
        images = params.images
        if not params.inference.prompt.strip():
            raise GenerationError("Please enter a prompt.")
        max_images = 1 + self._max_references
        if 1 + len(images.reference_paths) > max_images:
            raise GenerationError(f"Up to {max_images} input images are supported.")
        unknown = set(params.inference.options) - set(self._option_targets)
        if unknown:
            raise ValueError(f"{self.name}: no optional input(s) {sorted(unknown)} (it has {sorted(self._option_targets)})")
        for spec in self.capabilities.options:
            value = params.inference.options.get(spec.name)
            if spec.choices and value is not None and value not in spec.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of the choices of {spec.name}")

        report(ProgressStage.PREPARING)
        inference = replace(params.inference, seed=self._pick_seed(params.inference.seed))
        size: ResolvedSize = self._size.resolve(params.size, images.source_path, masked=bool(images.mask_path))
        options = " ".join(f"{name}={value!r}" for name, value in inference.options.items())
        print(
            f"[{self.name}] mask={'yes' if images.mask_path else 'no'} images={1 + len(images.reference_paths)} "
            f"gen_dims={size.width}x{size.height} {size.description} steps={inference.steps} {options}".rstrip(),
            flush=True,
        )

        def _infer(model_input_path: str) -> str:
            check_cancel()
            self._check_nodes()
            reference_paths = images.reference_paths
            if size.reference_resolution:
                # Each reference keeps its own aspect ratio at reference_resolution's
                # area, in multiples of 32: the Qwen-Image 2.1 encode node's own rule
                # (dimensions_from_source() in sizes.py), the only policy that sets
                # it. Another policy setting it may need a rule of its own.
                reference_paths = [
                    imaging.fit_image_to_size(
                        p, *dimensions_from_source(p, size.reference_resolution), Image.Resampling.LANCZOS
                    )
                    for p in reference_paths
                ]
            workflow = self.build_workflow(inference, model_input_path, reference_paths, size.values)
            t0 = time.time()
            sampler_node = _split(self._manifest["bind"]["steps"])[0]
            prompt_id = comfy_client.submit_workflow_and_wait(workflow, str(uuid.uuid4()), sampler_node, on_progress, cancel)
            print(f"[{self.name}] ComfyUI generation took {time.time() - t0:.1f}s", flush=True)
            return comfy_client.fetch_generated_image(prompt_id)

        cached = cached_infer(
            _infer, self.display_name, inference, images.reference_paths,
            **size.cache_kwargs, width=size.width, height=size.height,
        )

        def infer(model_input_path: str) -> str:
            output_path = cached(model_input_path)
            check_cancel()
            report(ProgressStage.POSTPROCESSING)
            return output_path

        final_output_path = imaging.run_generation(
            images.source_path, images.mask_path, images.annotated_path, (size.width, size.height), infer,
            params.postprocessing, input_size=size.input_size,
        )
        return GenerationResult(before_path=images.source_path, after_path=final_output_path, actual_seed=inference.seed)


def get_model(workflow: str, variant: Optional[str] = None, **options: bool) -> WorkflowModel:
    """
    The model of a bundled workflow (by name) or of a workflow file (by path).
    `variant` selects one of the workflow's variants; options are
    WorkflowModel's (lora_downloads).
    """
    bundled = available_workflows()
    if workflow in bundled:
        path = bundled[workflow]
    elif Path(workflow).is_file():
        path = Path(workflow)
    else:
        raise ValueError(
            f"Unknown workflow '{workflow}'. Available: {', '.join(sorted(bundled))}; "
            "or give the path of a workflow file."
        )
    return WorkflowModel.from_file(path, variant, **options)
