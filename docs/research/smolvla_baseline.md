# SmolVLA Baseline

Stable lookup alias for the current SmolVLA LIBERO baseline execution method.
Use this file when you remember "SmolVLA baseline" but not the dated handoff
filename.

Canonical handoff:

```text
docs/research/smolvla_baseline_handoff_2026_06_07.md
```

Physical AI orchestrator anchor:

```text
.agents/skills/physical-ai-orchestrator/SKILL.md
```

## Baseline Snapshot

- Model: `lerobot/smolvla_libero`
- Environment: LeRobot LIBERO through MuJoCo/robosuite on RunPod Linux GPU
- Focused weak-baseline task: `libero_goal`, task id `6`
- Seeds: `1200`, `1201`, `1202`
- Condition: policy-only SmolVLA rollout, no agentic intervention
- Required camera mapping:
  `--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'`
- Required policy camera setting: `--policy.empty_cameras=0`

## Minimal RunPod Command

```bash
cd /workspace/physical-ai/physical_ai_agent

export LIBERO_CONFIG_PATH=$HOME/.libero
export MUJOCO_GL=egl
export HF_HOME=/workspace/physical-ai/hf_home
export TRANSFORMERS_CACHE=/workspace/physical-ai/hf_home/transformers
export HF_HUB_CACHE=/workspace/physical-ai/hf_home/hub

PY=/root/physical-ai/envs/lerobot_py312/bin/python
OUT=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/libero_goal_task6_baseline_example/baseline_seed1200
mkdir -p "$OUT"

$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_no_progress \
  --intervention-mode none \
  --semantic-min-step 220 \
  --semantic-window 20 \
  --semantic-progress-threshold 0.002 \
  --output_dir="$OUT/eval_logs" \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[6]" \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.empty_cameras=0 \
  --seed=1200 \
  > "$OUT/instrumented_eval.log" 2>&1
```

Repeat with `--seed=1201` and `--seed=1202` for the current paired smoke.

The focused baseline is frozen unless the user explicitly asks to change the
baseline protocol. Wrapper experiments should compare against this policy-only
reference and report benchmark success separately from any internal verifier
or intervention metrics.

## RunPod Reproduction Scripts

Use the repo-local bootstrap script instead of repeating ad hoc environment
repair steps:

```bash
cd /workspace/physical-ai/physical_ai_agent
sh scripts/install/runpod_install.sh --component libero-smolvla
```

The known-good RunPod path pins `torch==2.5.1+cu124` on CUDA 12.4, then installs
LeRobot editable with `--no-deps` and adds LIBERO runtime packages under
constraints. This avoids the failed torch 2.11/CUDA 13 drift path that made
CUDA unavailable on the tested RunPod driver.

For baseline-preserving Imagine-Then-Act parity checks, use:

```bash
PYTHONPATH=src /root/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/runpod_libero_goal_baseline_parity_eval.py \
  --preset breadth \
  --methods policy_only,ita_baseline_fallback \
  --monitor-gpu \
  --early-stop-zero-at-half \
  --output-dir _workspace/runpod_results/libero_goal_breadth_seed1201 \
  --json
```

`ita_baseline_fallback` is a baseline-preserving control: it chooses
`candidate_00_policy_only`, records the ITA plumbing metadata, and keeps
`method_claim_ready=false`. It is not a true imagined selector and should not be
reported as an agentic improvement claim.

Current 2026-06-09 RunPod full-result evidence on commit
`7e3a18eb643b2f23220271e280091a7877e7dc81`:

- Direct policy-only: `37/50 = 74.0%`
- ITA `baseline_fallback`: `39/50 = 78.0%`

This is baseline-near/better implementation evidence only. The +4 percentage
point delta may be variance and should not be overclaimed as method performance.
