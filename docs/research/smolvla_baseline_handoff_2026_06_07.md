# SmolVLA LIBERO Baseline Handoff

This note captures the current SmolVLA baseline and the exact execution path
used for the in-episode agentic-wrapper experiments. Use this first when
starting a new thread.

## Baseline Definition

- Model: `lerobot/smolvla_libero`
- Benchmark path: LeRobot `lerobot-eval`
- Environment: LIBERO through MuJoCo/robosuite
- Current focused weak-baseline task: `libero_goal`, task id `6`
- Seeds used for the paired smoke: `1200`, `1201`, `1202`
- Baseline condition: policy-only SmolVLA rollout, no non-trivial in-episode
  intervention, one rollout per seed, one environment reset per episode.
- Metrics reported:
  - benchmark success / `pc_success`
  - action steps
  - eval seconds
  - environment resets
  - verifier triggers
  - intervention count
  - success per action step
  - success per eval minute

Current focused baseline result on `libero_goal` task id `6`:

| Seed | Success | Action steps | Eval seconds | Env resets |
| ---: | --- | ---: | ---: | ---: |
| 1200 | false | 300 | 10.1237 | 1 |
| 1201 | false | 300 | 10.4005 | 1 |
| 1202 | false | 300 | 10.2122 | 1 |

Aggregate: baseline `0/3`.

## RunPod Environment

Known working RunPod setup:

- Image family: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Evaluation Python: `/root/physical-ai/envs/lerobot_py312/bin/python`
- Repo path: `/workspace/physical-ai/physical_ai_agent`
- Branch: `codex/real-so100-green-doll-dryrun`
- Required environment variables:

```bash
export LIBERO_CONFIG_PATH=$HOME/.libero
export MUJOCO_GL=egl
export HF_HOME=/workspace/physical-ai/hf_home
export TRANSFORMERS_CACHE=/workspace/physical-ai/hf_home/transformers
export HF_HUB_CACHE=/workspace/physical-ai/hf_home/hub
```

Required camera mapping for this policy/env pairing:

```bash
--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
--policy.empty_cameras=0
```

Without the camera mapping, LeRobot raises a visual-feature mismatch between
`observation.images.camera1/camera2/camera3` and
`observation.images.image/image2`.

## Baseline Command Shape

The baseline was run through the instrumented wrapper so that cost metrics and
semantic traces are logged, but with `--intervention-mode none`.

```bash
cd /workspace/physical-ai/physical_ai_agent

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

## Wrapper Commands Used For Comparison

Best current positive wrapper:

```bash
$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_near_receptacle \
  --intervention-mode semantic_place_receptacle \
  --semantic-min-step 220 \
  --semantic-distance-threshold 0.07 \
  --semantic-contact-threshold 0.08 \
  --semantic-reach-gain 2.0 \
  --semantic-push-gain 5.0 \
  --semantic-place-z-command -1.0 \
  --semantic-gripper-command 1.0 \
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
  --seed=1200
```

Diagnostic negative wrapper:

```bash
$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_no_progress \
  --intervention-mode semantic_place_receptacle \
  --semantic-min-step 220 \
  --semantic-window 20 \
  --semantic-progress-threshold 0.002 \
  --semantic-distance-threshold 0.07 \
  --semantic-contact-threshold 0.08 \
  --semantic-reach-gain 2.0 \
  --semantic-push-gain 5.0 \
  --semantic-place-z-command -1.0 \
  --semantic-gripper-command 1.0 \
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
  --seed=1200
```

## Current Comparison Results

Focused `libero_goal` task id `6`, seeds `1200/1201/1202`:

| Condition | Successes | Success rate | Mean interventions | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline policy-only | 0/3 | 0.00% | 0.00 | fixed weak baseline |
| near-receptacle placement macro | 1/3 | 33.33% | 1.33 | seed 1200 converts fail to success |
| no-progress placement macro | 0/3 | 0.00% | 73.33 | improves subgoal distance on some seeds, over-intervenes |

The positive result is not reset retry. It is an in-episode intervention:
seed `1200` baseline failed at `300` steps, while the near-receptacle wrapper
succeeded at `224` steps with `4` interventions and one environment reset.

## Important Files

- Runner:
  `scripts/run_libero_in_episode_smolvla_instrumented.py`
- Report builder:
  `scripts/build_libero_in_episode_ablation_report.py`
- Baseline/wrapper paired report:
  `docs/research/libero_goal_task6_place_multiseed_report_2026_06_07.md`
- Trigger coverage report:
  `docs/research/libero_goal_task6_trigger_coverage_report_2026_06_07.md`
- Worklog:
  `docs/runpod_worklog.md`

## Current Research Interpretation

The SmolVLA policy-only baseline is stable enough to freeze for this focused
agentic-wrapper study. The next work should not keep changing the baseline.
It should improve the wrapper.

Most promising next direction:

- Use no-progress detection to identify approach/reach failure.
- Do not apply the placement macro continuously.
- Gate placement under contact or near-receptacle conditions.
- Keep reporting cost-normalized metrics: success, action steps, eval seconds,
  env resets, verifier triggers, intervention count, success/action step, and
  success/eval minute.
