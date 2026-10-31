# SPDX-License-Identifier: GPL-3.0-or-later
# One image for AI Image Edit and all of its models (the workflow files under
# ai_image_edit/data/workflows/). It contains ComfyUI, PyTorch and the app,
# but no model weights: the model is chosen when the container starts
# (MODEL_WORKFLOW, MODEL_VARIANT, or a workflow file of your own), and its
# weights, LoRAs and custom nodes are fetched on first start, as its workflow
# file lists them. Mount a volume at /home/user/app/models to keep the weights
# between containers.
#
# Base image: plain Python (Debian slim), no bundled CUDA/PyTorch — torch
# is installed manually below via pip, which pulls its own bundled CUDA
# runtime libs (cuDNN, cuBLAS, NCCL, etc.) from the cu126 wheel index. No
# system CUDA toolkit is needed since nothing here compiles CUDA code.
FROM python:3.11-slim

# git: the app installs custom nodes with it (the workflow files list them), and this file installs the app.
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Matches Hugging Face Spaces' expected non-root container user (uid 1000);
# harmless on RunPod, which doesn't require it.
RUN useradd -m -u 1000 user
USER user
ENV MKL_THREADING_LAYER=GNU

# PIP_USER: the packages a custom node needs are installed by the app when it
# starts, as this user, who cannot write to /usr/local.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_USER=1 \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Images (uploads, results, ComfyUI's input/output/temp folders) live in RAM
# under /dev/shm, never on disk. This is the soft cap the app enforces by
# deleting the oldest files first. It cannot exceed 90% of the container's
# /dev/shm (docker run --shm-size; Docker's default is 64 MB), and the app
# warns at startup if it had to lower it.
ENV AI_IMAGE_EDIT_WORK_MAX_SIZE=256m

# This image is meant for public hosts (RunPod's HTTP proxy has no login of
# its own), so the app refuses to start unless APP_PASSWORD is set.
ENV REQUIRE_PASSWORD=1

# Install PyTorch (CUDA 12.6 wheels) manually — this base image has no
# CUDA/PyTorch preinstalled, unlike the pytorch/pytorch image. Before ComfyUI,
# so that bumping COMFYUI_REF leaves these gigabytes as they are: a host that has
# the previous image then only downloads the layers after them.
#
# The CUDA libraries PyTorch brings come in several layers rather than one, so
# a pull downloads and unpacks them in parallel. pip first reports what PyTorch
# installs (a dry run); the steps below install those exact files group by
# group, and the last one installs the rest and PyTorch itself, which finds
# everything in place.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
# The groups, by package name (whatever matches none goes with PyTorch itself):
# cuDNN, cuBLAS, cuSPARSE, the solvers and NCCL, the remaining CUDA libraries.
RUN pip install --dry-run --ignore-installed --quiet --report /tmp/torch.json \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${TORCH_INDEX} && \
    python -c 'import json, re; \
groups = [("cudnn", "nvidia-cudnn"), ("cublas", "nvidia-cublas"), ("cusparse", "nvidia-cusparse"), \
          ("solvers", "nvidia-(cufft|curand|cusolver|nccl|nvshmem)"), ("cuda", "nvidia-|triton")]; \
items = json.load(open("/tmp/torch.json"))["install"]; \
group = lambda name: next((g for g, pattern in groups if re.match(pattern, name.lower().replace("_", "-"))), "rest"); \
[open(f"/tmp/torch-{g}.txt", "w").write("".join(i["download_info"]["url"] + "\n" for i in items if group(i["metadata"]["name"]) == g)) \
 for g in [*dict(groups), "rest"]]'
RUN if [ -s /tmp/torch-cudnn.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-cudnn.txt; fi
RUN if [ -s /tmp/torch-cublas.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-cublas.txt; fi
RUN if [ -s /tmp/torch-cusparse.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-cusparse.txt; fi
RUN if [ -s /tmp/torch-solvers.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-solvers.txt; fi
RUN if [ -s /tmp/torch-cuda.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-cuda.txt; fi
RUN if [ -s /tmp/torch-rest.txt ]; then pip install --no-cache-dir --no-deps -r /tmp/torch-rest.txt; fi && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${TORCH_INDEX}

# Clone ComfyUI, pinned to a known commit (0.39.0) so that a later ComfyUI
# change cannot silently break a model's nodes. Bump it deliberately and re-test.
ARG COMFYUI_REF=87c32827017c50c6a629da941439015b4ad656e6
RUN git clone https://github.com/Comfy-Org/ComfyUI.git . && \
    git checkout ${COMFYUI_REF}

# ComfyUI's own Python dependencies, from the git clone above.
RUN pip install --no-cache-dir -r requirements.txt

# COMFY_EXTRA_ARGS is deliberately not set here: unset means the app chooses the
# ComfyUI arguments from the free GPU memory (deployments/docker/DEPLOY.md).

# Install the app from GitHub: the branch, tag or commit APP_REF of APP_REPO
# (the CI build passes the commit it was started for). To try a change, push it
# to a branch and build with --build-arg APP_REF=<branch>. Not copied from the
# build context, so the context can stay empty. The app is started from the
# ComfyUI root, which it relies on (main.py, models/).
ARG APP_REPO=pulb/ai_image_edit
ARG APP_REF=main
RUN pip install --no-cache-dir \
    "ai-image-edit @ git+https://github.com/${APP_REPO}@${APP_REF}"

EXPOSE 7860

# MODEL_WORKFLOW (default qwen_image21) and MODEL_VARIANT
# are read from the environment.
CMD ["ai-image-edit"]
