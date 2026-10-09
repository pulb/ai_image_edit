# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings and helpers shared by the ComfyUI-based models."""
import os
from pathlib import Path


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
