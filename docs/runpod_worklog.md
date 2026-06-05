# RunPod Worklog

This file is the durable handoff journal for RunPod work. Update it whenever a
RunPod evaluation, setup fix, result fetch, or lifecycle decision changes the
state another conversation would need to recover.

Do not record API keys, SSH private keys, full `.env` contents, or raw RunPod
API responses that may contain secrets.

## Operating Rules

- Keep a Pod running during an active evaluation or setup milestone.
- Do not stop the Pod after every command; stop only after a meaningful
  milestone is complete, results are fetched locally, and no immediate follow-up
  run needs the warm environment.
- Keep repo, virtualenvs, datasets, model cache, logs, videos, and outputs under
  `/workspace` on RunPod.
- For paper-comparable numbers, use a committed git revision that was pulled on
  RunPod. Debug-only remote edits are not reportable benchmark evidence.
- Before stopping or terminating a Pod, fetch result bundles to the local repo
  with `scripts/runpod_fetch_results.sh`.

## 2026-06-05 SmolVLA LIBERO Full Eval

### Current State

- Status: completed and fetched locally.
- Goal: evaluate `lerobot/smolvla_libero` on LIBERO suites with a protocol close
  to paper-comparable reporting.
- Pod: `t8eqsuj7nzaou8`.
- GPU: RTX 3090 24GB.
- Local SSH target is stored in uncommitted `.env`; do not copy it into this
  file.
- Remote repo path: `/workspace/physical-ai/physical_ai_agent`.
- Remote result root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_full_20260605T162959Z`.
- Driver log:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/full_eval_driver_20260605T162959Z.log`.

### Command

```bash
lerobot-eval \
  --output_dir=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_full_20260605T162959Z/eval_logs \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.empty_cameras=1
```

### Protocol

- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Expected task coverage: 10 tasks per suite.
- Episodes: 10 per task.
- Expected total: 400 episodes.

### Latest Observations

- Completion:
  - exit code: 0
  - local fetched bundle:
    `_workspace/runpod_results/20260605T173150Z/smolvla_libero_full_20260605T162959Z`
  - metric file:
    `_workspace/runpod_results/20260605T173150Z/smolvla_libero_full_20260605T162959Z/eval_logs/eval_info.json`
  - videos: 400
  - elapsed eval time: 3186.6 s
  - overall success: 64.75%
- Suite success:
  - `libero_spatial`: 57.0%
  - `libero_object`: 68.0%
  - `libero_goal`: 81.0%
  - `libero_10`: 53.0%
- Interpretation: completed correctly, but the result is not close to published
  SmolVLA numbers. The likely mismatch is checkpoint/config. This run used
  `lerobot/smolvla_libero` and loaded `n_action_steps=50`; the benchmark
  checkpoint path is `HuggingFaceVLA/smolvla_libero` with `n_action_steps=1`.
- The evaluation is using GPU, but it is CPU-bound by LIBERO/MuJoCo stepping.
- Evidence from the running process:
  - GPU memory: about 1994 MiB allocated by the `lerobot-eval` Python process.
  - GPU SM utilization sampled around 3-25%.
  - CPU usage sampled around 1184-1667% for the same process.
- Generated video count reached 79 while processing `libero_spatial_7`.
- Task video counts at that point:
  - `libero_spatial_0`: 10
  - `libero_spatial_1`: 10
  - `libero_spatial_2`: 10
  - `libero_spatial_3`: 10
  - `libero_spatial_4`: 10
  - `libero_spatial_5`: 10
  - `libero_spatial_6`: 10
  - `libero_spatial_7`: 9 in progress
- The latest sampled `libero_spatial_7` task progress was 9/10 eval batches
  with `running_success_rate=55.6%`.
- Disk was healthy: `/workspace` about 12G used of 40G.

### Important Interpretation

The low GPU utilization does not mean the policy is running on CPU only. The
policy process holds GPU memory, but the conservative evaluation settings run a
single environment with batch size 1, so simulation and environment stepping are
the bottleneck. Throughput optimization should be tested separately with
parallel environments or larger batches after this correctness-oriented run
finishes.

### Next Actions

1. Continue the second full run using `HuggingFaceVLA/smolvla_libero` and
   `--policy.num_steps=10 --policy.n_action_steps=1`.
2. Fetch the second run's result bundle when it completes.
3. Replace the headline comparison table with the second run if it matches the
   canonical SmolVLA configuration better.
4. Stop the Pod only after the comparable result and report are fetched locally.

## 2026-06-05 HuggingFaceVLA SmolVLA LIBERO Full Eval

### Current State

- Status: first attempt failed before rollout; corrected attempt is running.
- Goal: rerun the full 400-episode LIBERO evaluation with the benchmark-oriented
  checkpoint/config.
- Pod: `t8eqsuj7nzaou8`.
- Failed output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_full_20260605T173749Z`.
- Failed driver log:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/hfvla_full_eval_driver_20260605T173749Z.log`.
- Corrected output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_full_20260605T173842Z`.
- Corrected driver log:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/hfvla_full_eval_driver_20260605T173842Z.log`.

### Command Difference From Previous Run

```bash
SMOLVLA_MODEL_ID=HuggingFaceVLA/smolvla_libero
LIBERO_EXTRA_ARGS="--policy.num_steps=10 --policy.n_action_steps=1"
```

This is expected to be closer to published SmolVLA LIBERO settings than the
previous `lerobot/smolvla_libero` run.

### Failure And Fix

- First attempt failed before any videos were created.
- Error: feature mismatch. The HuggingFaceVLA checkpoint expected
  `observation.images.image`, `observation.images.image2`, and
  `observation.images.empty_camera_0`, but the previous camera mapping produced
  `observation.images.camera1` and `observation.images.camera2`.
- Fix: rerun with `LIBERO_CAMERA_NAME_MAPPING=none` so LeRobot keeps the default
  image feature names for this checkpoint.
