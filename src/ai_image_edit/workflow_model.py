# SPDX-License-Identifier: GPL-3.0-or-later
"""
The model the app runs: a workflow file (ai_image_edit/data/workflows/<name>.json) with a
ComfyUI workflow in API format and a manifest that says how the app's inputs map
onto that workflow. A bundled file is found by its name, any other by its path
(see get_model()); the format is documented in doc/contribution/workflow_files.md.

Workflow targets are written "<node id>.<input name>", e.g. "6.seed"; the
input name may itself contain dots ("5.images.image_2").
"""
import copy
import os
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_image_edit.comfyui import client as comfy_client
from ai_image_edit.comfyui import custom_nodes, downloads, gpu
from ai_image_edit.core import imaging
from ai_image_edit.core.errors import GenerationError
from ai_image_edit.core.result_cache import cached_infer
from ai_image_edit.core.types import GenerationParams, GenerationResult, ModelCapabilities, RangeSpec
from ai_image_edit.sizes import POLICIES, ResolvedSize, SizePolicy
from ai_image_edit.workflows import available_workflows, read_workflow_file


def configured_file(env_var: str, folder: Path) -> str:
    """
    The path (forward slashes, relative to `folder`) that environment variable
    `env_var` names. Raises RuntimeError if it is unset or the file does not exist.
    """
    value = os.environ.get(env_var, "").strip().replace("\\", "/")
    if not value:
        raise RuntimeError(
            f"{env_var} is not set: set it to the file's path relative to ComfyUI's {folder} folder."
        )
    if not (folder / value).is_file():
        raise RuntimeError(f"File not found: {folder / value} (set by {env_var}).")
    return value


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
    """How far the model's background setup (downloads, installs) is."""

    done: bool = True
    message: str = ""  # what is being done right now, e.g. "Downloading x.safetensors (2/5): 42%"
    error: Optional[str] = None
    fraction: Optional[float] = None  # 0 to 1 over the whole setup, None while that is not known


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


class WorkflowModel:
    """
    A model run through a local ComfyUI server, as described by a workflow file.

    The UI needs four things from it: what it can do (capabilities), how to
    start and stop it (start(), shutdown(); setup_progress() reports the
    downloads start() continues in the background), and how to run one
    generation (generate()), plus a name and version to show.
    """

    def __init__(self, name: str, manifest: dict, workflow: dict, variant: Optional[str] = None) -> None:
        self.name = name
        self._manifest = self._with_variant(name, manifest, variant)
        self._workflow = workflow
        self._process: Optional[subprocess.Popen] = None
        self._nodes_checked = False
        self._download_status: Optional[downloads.DownloadStatus] = None
        self._cancel_downloads = threading.Event()
        self._lora_warned: set = set()

        size = manifest["size"]
        if size["policy"] not in POLICIES:
            raise ValueError(f"{name}: unknown size policy {size['policy']!r} (known: {', '.join(POLICIES)})")
        self._size: SizePolicy = POLICIES[size["policy"]](size)
        self._validate()

    @classmethod
    def from_file(cls, path: Path, variant: Optional[str] = None) -> "WorkflowModel":
        """The model described by a workflow file; its name is the file name without .json."""
        path = Path(path)
        manifest, workflow = read_workflow_file(path)
        return cls(path.stem, manifest, workflow, variant)

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
            matches = [spec for spec in manifest["files"] if spec["set"] == target]
            if not matches:
                raise ValueError(f"{name}: variant {variant!r} changes {target!r}, which is not in files")
            if "name" in changes or "url" in changes:
                for key in ("sha256", "size"):
                    matches[0].pop(key, None)
            matches[0].update(changes)
        if "model_version" in chosen:
            manifest["model_version"] = chosen["model_version"]
        return manifest

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
            if "name" not in spec and "env" not in spec:
                raise ValueError(f"{self.name}: files entry for {spec['set']!r} needs 'name' or 'env'")
            if "name" in spec and "url" not in spec and "env" not in spec:
                raise ValueError(f"{self.name}: files entry {spec['name']!r} needs a 'url'")
            if spec.get("license") and spec["license"] not in m.get("licenses", {}):
                raise ValueError(f"{self.name}: files entry {spec['set']!r}: unknown license {spec['license']!r}")
        for variant, chosen in m.get("variants", {}).items():
            for target in chosen.get("files", {}):
                if target not in {spec["set"] for spec in m["files"]}:
                    raise ValueError(f"{self.name}: variant {variant!r} changes {target!r}, which is not in files")
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
        bound = set(m["size"].get("bind", {}))
        if bound != self._size.value_names:
            raise ValueError(
                f"{self.name}: size.bind must bind exactly {sorted(self._size.value_names)} "
                f"for policy {m['size']['policy']!r}, not {sorted(bound)}"
            )
        for key, target in m["size"].get("bind", {}).items():
            need_node(target, f"size.bind.{key}")
        loras = m.get("loras")
        if loras:
            for spec in loras.get("files", []):
                if spec.get("license") and spec["license"] not in m.get("licenses", {}):
                    raise ValueError(f"{self.name}: loras.files {spec['name']!r}: unknown license {spec['license']!r}")
            for targets in loras["feeds"].values():
                for target in targets:
                    need_node(target, "loras.feeds")

    # ------------------------------------------------------------ identity

    @property
    def model_name(self) -> str:
        return self._manifest["model_name"]

    @property
    def model_version(self) -> Optional[str]:
        """The manifest's (or variant's) version; MODEL_VERSION overrides it, e.g. for weights set by MODEL_FILE."""
        return os.environ.get("MODEL_VERSION", "").strip() or self._manifest.get("model_version")

    @property
    def display_name(self) -> str:
        """model_name followed by model_version, if any."""
        return f"{self.model_name} {self.model_version}" if self.model_version else self.model_name

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
            supports_negative_prompt="negative_prompt" in bind,
            num_annotation_colors=caps.get("num_annotation_colors", 0),
            step_range=_range(caps["steps"]),
        )
        for field, key in (("cfg_range", "cfg"), ("denoise_range", "denoise"), ("lora_strength_range", "lora_strength")):
            if key in caps:
                kwargs[field] = _range(caps[key])
        return ModelCapabilities(**kwargs)

    # ------------------------------------------------------------ lifecycle

    def _file_value(self, spec: dict) -> str:
        """
        The file name (relative to its folder) for a files entry: the environment
        variable if it is set (that file is used as it is and never downloaded),
        otherwise the manifest's name.
        """
        env_set = bool(spec.get("env") and os.environ.get(spec["env"], "").strip())
        if env_set or "name" not in spec:  # without a name the variable is required
            return configured_file(spec["env"], Path(spec["folder"]))
        return spec["name"]

    def _pending_downloads(self) -> List[dict]:
        wanted = [(spec, Path(spec["folder"]) / self._file_value(spec)) for spec in self._manifest.get("files", [])]
        # LoRA files: each goes into its own subfolder of loras.dir (the subfolder is the display name).
        loras = self._manifest.get("loras", {})
        wanted += [(spec, Path(loras["dir"]) / spec["dir"] / spec["name"]) for spec in loras.get("files", [])]
        return [
            {"url": spec["url"], "dest": dest, "sha256": spec.get("sha256"), "size": spec.get("size"), "license": spec.get("license")}
            for spec, dest in wanted if not dest.is_file()
        ]

    def _check_licenses(self, pending: List[dict]) -> None:
        """Downloading a file whose license needs accepting requires ACCEPT_LICENSES to name it."""
        accepted = {part.strip() for part in os.environ.get("ACCEPT_LICENSES", "").split(",")}
        for item in pending:
            lic = item["license"]
            if lic and lic not in accepted and "all" not in accepted:
                info = self._manifest["licenses"][lic]
                raise RuntimeError(
                    f"{self.name} needs {item['dest'].name}, which is under the {info['name']} ({info['url']}). "
                    f"Read it and, if you accept it, set ACCEPT_LICENSES={lic} to let the app download the file."
                )

    def _weights_bytes(self) -> int:
        """Total size of the model files (not the LoRAs), which ComfyUI loads for a generation."""
        total = 0
        for spec in self._manifest.get("files", []):
            path = Path(spec["folder"]) / self._file_value(spec)
            if path.is_file():
                total += path.stat().st_size
        return total

    def _launch_comfy(self) -> None:
        args, reason = gpu.comfy_extra_args(self._weights_bytes(), self._manifest.get("vram_headroom_gb"))
        print(f"ComfyUI arguments: {' '.join(args) or '(none)'} ({reason})", flush=True)
        self._process = comfy_client.launch_comfy_process(args)

    def start(self) -> None:
        # Checked before anything starts, so a missing setting or file fails fast.
        pending = self._pending_downloads()
        nodes = custom_nodes.missing(self._manifest.get("custom_nodes", []))
        if not pending and not nodes:
            self._launch_comfy()
            return
        self._check_licenses(pending)
        status = self._download_status = downloads.DownloadStatus()

        def prepare_and_launch() -> None:
            for spec in nodes:
                if self._cancel_downloads.is_set():
                    return
                status.set_message(f"Installing {spec['name']}")
                custom_nodes.install(spec)
            if not self._cancel_downloads.is_set():
                self._launch_comfy()

        threading.Thread(
            target=downloads.run_downloads, args=(pending, status, self._cancel_downloads, prepare_and_launch),
            name=f"{self.name}-setup", daemon=True,
        ).start()

    def setup_progress(self) -> SetupProgress:
        status = self._download_status
        if status is None:
            return SetupProgress()
        if status.error:
            return SetupProgress(done=True, error=f"Setting up the model failed: {status.error}")
        done = status.finished.is_set()
        return SetupProgress(done=done, message="" if done else status.message, fraction=status.fraction)

    def _check_downloads(self) -> None:
        status = self._download_status
        if status is None:
            return
        if status.error:
            raise GenerationError(f"Setting up the model failed: {status.error}. Restart the app to retry.")
        if not status.finished.is_set():
            raise GenerationError(f"The model is still being set up. {status.message}".strip())

    def shutdown(self) -> None:
        self._cancel_downloads.set()
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
            self._warn_once(f"LoRA folder not found at expected path: {lora_dir}")
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
            put(spec["set"], self._file_value(spec))
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
        the ComfyUI workflow and fetches the result. Only a masked request has
        its source cropped to the generation size; otherwise the output size
        comes from the workflow, through the values the manifest binds
        (size.bind). Reference images are passed as they are.
        """
        self._check_downloads()
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


def get_model(workflow: str, variant: Optional[str] = None) -> WorkflowModel:
    """The model of a bundled workflow (by name) or of a workflow file (by path).

    `variant` selects one of the workflow's variants.
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
    return WorkflowModel.from_file(path, variant)
