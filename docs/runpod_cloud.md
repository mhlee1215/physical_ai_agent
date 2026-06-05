# RunPod Cloud Workstation

This project uses RunPod as the Linux/NVIDIA cloud workstation for Isaac, LIBERO, and GPU-heavy SmolVLA experiments after the Mac-local gates pass.

## Current Baseline

- Product: Pods
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- GPU: `1x RTX 4090`
- Persistent workspace: `/workspace`
- Repo path: `/workspace/physical-ai/physical_ai_agent`

Do not rely on `/` for project data. The container root filesystem is small and should be treated as disposable.

## Connect

Set the Pod SSH target in the shell:

```bash
export RUNPOD_SSH='v605dhuhdkjbfm-64410b15@ssh.runpod.io'
```

Then run:

```bash
sh scripts/runpod_check.sh
```

For manual access:

```bash
ssh -tt -i ~/.ssh/id_ed25519 "$RUNPOD_SSH"
```

Avoid SSH agent forwarding (`-A`) unless there is a specific reason. The normal key-based connection is enough for setup and keeps the local SSH agent from being exposed to the cloud machine.

## Update Repo On RunPod

Inside the Pod:

```bash
cd /workspace/physical-ai/physical_ai_agent
git fetch origin
git checkout main
git pull --ff-only origin main
```

The current known-good cloud commit is `59f25a1`.

## Verified Smoke

The template was verified on the Pod with:

```text
torch 2.4.1+cu124
cuda_available True
cuda_version 12.4
gpu NVIDIA GeForce RTX 4090
```

Run a quick local cloud check with:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Verified CP24 Status

The current official PyTorch 2.4 / Python 3.11 template is usable as a first
cloud workstation, but it is not yet a full ManiSkill manipulation evaluation
machine.

Verified:

```bash
cd /workspace/physical-ai/physical_ai_agent
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,sim,maniskill]'
PYTHONPATH=src .venv/bin/python -B -m physical_ai_agent.checkpoints.checkpoint_24 \
  --require-maniskill \
  --no-fallback-env \
  --env-id Empty-v1 \
  --episodes 1 \
  --steps 1 \
  --policy zero \
  --output-dir _workspace/checkpoints/runpod_checkpoint_24_empty_zero_1ep_1step
```

Result:

```text
checkpoint_24_maniskill_hab_smolvla_eval_planning: passed
metrics=env:Empty-v1 episodes:2 rollout:passed success_rate:0/2 smolvla_ready:False
```

Blocked on this template:

- `.[smolvla]` does not install under Python 3.11 because `lerobot>=0.5.1`
  requires Python `>=3.12`.
- `PickCube-v1` is blocked by SAPIEN/Vulkan renderer initialization:
  `vk::createInstanceUnique: ErrorIncompatibleDriver`.
- Installing `libvulkan1` and `vulkan-tools` makes `vulkaninfo` available, but
  the NVIDIA ICD still does not load. `vulkaninfo --summary` reports CPU
  `llvmpipe` only, not the RTX 4090 Vulkan device.

The generated blocker artifact is:

```text
_workspace/checkpoints/runpod_checkpoint_24_pickcube_zero_after_vulkan_1ep_1step/maniskill_blocker.md
```

For full SmolVLA evaluation, use a Python 3.12-capable image or build a Python
3.12 environment with CUDA PyTorch. For full ManiSkill manipulation tasks, use a
RunPod template or host configuration where NVIDIA Vulkan appears in
`vulkaninfo --summary`, not only CUDA in `nvidia-smi`.

## Cost Hygiene

- Stop the Pod when not actively running experiments.
- Keep repo, environments, model cache, datasets, logs, and videos under `/workspace`.
- Download large results locally or to external storage before deleting volumes.
- Delete the Network Volume only after confirming no needed data remains.
