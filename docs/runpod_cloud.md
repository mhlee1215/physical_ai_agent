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

## Lifecycle Script

Create a RunPod API key in RunPod account settings and keep it only in your
local shell or uncommitted `.env` file:

```bash
export RUNPOD_API_KEY='...'
export RUNPOD_POD_ID='v605dhuhdkjbfm'
```

Then manage the Pod without opening the RunPod console:

```bash
sh scripts/runpod_pod.sh status
sh scripts/runpod_pod.sh stop
sh scripts/runpod_pod.sh start
sh scripts/runpod_pod.sh list
```

There is also an explicit terminate command, but it is intentionally guarded:

```bash
sh scripts/runpod_pod.sh terminate --yes-terminate
```

Use `stop` first when the goal is to pause GPU billing. RunPod may reject stop
for Pods attached to network volumes; in that case, terminate only after
confirming all useful files are under the persistent `/workspace` network
volume. Container-root data under `/` should be treated as disposable.

## Probe Alternative GPU Types

When COMMUNITY pool is saturated, try SECURE pool with the same image/volume and
probe for first available GPU type:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_NETWORK_VOLUME_ID=tchm4gxfvd
RUNPOD_CLOUD_TYPE=SECURE sh scripts/runpod_probe_gpus.sh
```

Useful follow-up:

- If it succeeds, keep the returned `RUNPOD_POD_ID` for the run.
- If it fails, rerun later or add more GPU names to the probe list in
  `scripts/runpod_probe_gpus.sh`.

## Result Handoff Before Stop

Maintain the durable RunPod handoff journal while working:

```text
docs/runpod_worklog.md
```

Update it with the active Pod, result paths, exact commands, current status,
and next actions before ending a work session or stopping the Pod.

Before stopping a Pod after an evaluation run, collect a self-contained result
bundle under the network volume:

```bash
cd /workspace/physical-ai/physical_ai_agent
mkdir -p _workspace/runpod_results
cp -R _workspace/checkpoints _workspace/runpod_results/checkpoints
cp -R outputs _workspace/runpod_results/outputs 2>/dev/null || true
```

Also write a report such as:

```text
_workspace/runpod_results/smolvla_libero_report.md
```

The report should include the git commit, exact command, model id, benchmark
suite, episode count, success-rate table, blockers, and artifact paths.

Fetch the bundle to the local repo before stopping:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_fetch_results.sh
```

During long debugging sessions, archive completed result directories locally and
clear them from the network volume without stopping the active Pod:

```bash
set -a
. ./.env
set +a
RUNPOD_ACTIVE_RESULT_DIR=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/<active-run> \
  sh scripts/runpod_archive_results.sh --delete-remote --yes-delete
```

Keep model caches and LIBERO assets on the network volume while iterating; they
are usually more expensive to redownload than to store. Delete only completed
run outputs after a successful local archive.

Only after the local fetch succeeds:

```bash
sh scripts/runpod_pod.sh stop
```

## Update Repo On RunPod

Inside the Pod:

```bash
cd /workspace/physical-ai/physical_ai_agent
git fetch origin
git checkout main
git pull --ff-only origin main
```

The current known-good cloud commit is `59f25a1`.

## LeRobot / SmolVLA RunPod Environment

Use the unified LeRobot runner for LIBERO and Meta-World:

```bash
cd /workspace/physical-ai/physical_ai_agent
sh scripts/eval_smolvla_lerobot_linux.sh --benchmark libero --agentic-layer baseline
sh scripts/eval_smolvla_lerobot_linux.sh --benchmark metaworld --agentic-layer baseline
```

Environment defaults are set to avoid previous RunPod failure modes:

- `PY312_VENV=/root/physical-ai/envs/lerobot_py312` by default. This uses the
  faster container disk for the Python environment; keep caches/results under
  `/workspace`.
- `PIP_CACHE_DIR=/workspace/physical-ai/pip_cache`.
- `HF_HOME=/workspace/physical-ai/hf_home`.
- `MUJOCO_GL=egl`.
- `MUJOCO_VERSION=3.3.2` for LIBERO unless explicitly overridden.
- `REQUIRE_CUDA=1` by default. If CUDA is not visible to PyTorch, the runner
  fails before producing misleading CPU-baseline numbers.

Every run writes:

```text
debug_artifacts/runpod_preflight.txt
debug_artifacts/environment_probe.txt
debug_artifacts/eval_manifest.json
debug_artifacts/command_argv.json
debug_artifacts/agentic_layer.json
debug_artifacts/events.jsonl
```

Read those files first when debugging setup drift. The most important checks
are Python `>=3.12`, `torch.cuda.is_available() == True`, the evaluated git
commit, MuJoCo version, and the generated `lerobot-eval` argv.

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

## Retired ManiSkill Checkpoint Note

The former CP24 ManiSkill smoke gate has been retired from the active repo surface. Historical RunPod notes remain useful for environment selection: the official PyTorch 2.4 / Python 3.11 template verified CUDA/PyTorch, but Python 3.11 blocked `lerobot>=0.5.1` SmolVLA and the tested host exposed CUDA without a working NVIDIA Vulkan device for manipulation rendering.

For current benchmark work, use a committed evaluation branch on RunPod with a Python 3.12-capable LeRobot environment and record results under `_workspace/runpod_results/` before stopping the Pod.

## Cost Hygiene

- Stop the Pod when not actively running experiments.
- Keep repo, environments, model cache, datasets, logs, and videos under `/workspace`.
- Download large results locally or to external storage before deleting volumes.
- Delete the Network Volume only after confirming no needed data remains.
