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

The current known-good cloud commit is `40bbd38`.

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

## Cost Hygiene

- Stop the Pod when not actively running experiments.
- Keep repo, environments, model cache, datasets, logs, and videos under `/workspace`.
- Download large results locally or to external storage before deleting volumes.
- Delete the Network Volume only after confirming no needed data remains.
