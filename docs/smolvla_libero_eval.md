# SmolVLA LIBERO Evaluation

## Goal

Generate paper-comparable SmolVLA baseline numbers on LIBERO before testing
agentic wrappers.

## Comparable Protocol

Use LeRobot's LIBERO benchmark integration:

- policy: `lerobot/smolvla_libero`
- environment: `libero`
- suites: `libero_spatial,libero_object,libero_goal,libero_10`
- episodes: `10` per task
- tasks: all 10 tasks per suite
- total trials: `4 suites * 10 tasks * 10 episodes = 400`
- metric: binary task success rate, reported per suite and averaged
- rendering: `MUJOCO_GL=egl` on headless cloud
- control mode: use the checkpoint's default unless the model card or config
  specifies otherwise

The quick smoke protocol may use one suite and one episode per task, but those
numbers are not paper-comparable.

## RunPod Command

After starting the Pod and pulling the latest repo:

```bash
cd /workspace/physical-ai/physical_ai_agent
git pull --ff-only origin main
sh scripts/runpod_smolvla_libero_eval.sh
```

The RunPod command is a wrapper around the Linux evaluator:

```bash
sh scripts/eval_smolvla_libero_linux.sh
```

For a quick smoke:

```bash
LIBERO_TASKS=libero_spatial \
LIBERO_N_EPISODES=1 \
sh scripts/eval_smolvla_libero_linux.sh
```

## Mac Local Preflight

macOS cannot produce paper-comparable LIBERO numbers because LeRobot's LIBERO
benchmark requires Linux. Use the Mac script only to produce a local readiness
report:

```bash
sh scripts/eval_smolvla_libero_mac.sh
```

## Result Handoff

The script writes to:

```text
_workspace/runpod_results/smolvla_libero_<timestamp>/
```

Before stopping RunPod, fetch results locally:

```bash
set -a
. ./.env
set +a
RUNPOD_REMOTE_RESULT_DIR=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results \
  sh scripts/runpod_fetch_results.sh
sh scripts/runpod_pod.sh stop
```

## Blocking Conditions

- If RunPod cannot start the stopped Pod, create a new Pod with the same
  network volume attached.
- The previous official PyTorch 2.4 / Python 3.11 template is insufficient for
  current LeRobot SmolVLA because `lerobot>=0.5.1` requires Python `>=3.12`.
- Prefer a Python 3.12-capable image. If unavailable, bootstrap Python 3.12
  inside the Pod and create the LeRobot environment under `/workspace`.

## Data And Cache Size

Checked through the Hugging Face tree API on 2026-06-05:

| Artifact | Approx size | Recommendation |
| --- | ---: | --- |
| `lerobot/smolvla_libero` model | 0.91 GB | Fine to download once; cache under `/workspace` |
| `lerobot/smolvla_base` model | 0.92 GB | Fine to download once; cache under `/workspace` |
| `lerobot/libero` dataset repo | 1.94 GB | Cache under `/workspace`; acceptable to redownload if needed |
| `HuggingFaceVLA/libero` dataset repo | 34.93 GB | Do not redownload repeatedly; keep on network volume |

For evaluation, the first required download should primarily be the finetuned
policy plus LIBERO/MuJoCo assets. Full training datasets are not necessarily
needed for policy evaluation, but any Hugging Face cache should still point at
the network volume:

```bash
export HF_HOME=/workspace/physical-ai/hf_home
export HF_HUB_CACHE=/workspace/physical-ai/hf_home/hub
export TRANSFORMERS_CACHE=/workspace/physical-ai/hf_home/transformers
```

If a run starts downloading `HuggingFaceVLA/libero`, preserve it on the network
volume and do not place it on the disposable container disk.

## Create Replacement RunPod

If the stopped Pod cannot start because its old host has no available GPU, make
a replacement Pod attached to the same network volume:

```bash
set -a
. ./.env
set +a
export RUNPOD_NETWORK_VOLUME_ID=tchm4gxfvd
sh scripts/runpod_create_pod.sh
sh scripts/runpod_create_pod.sh --yes-create
```

The dry run prints the request body without starting billing. The `--yes-create`
form creates a new Pod and starts billing.
