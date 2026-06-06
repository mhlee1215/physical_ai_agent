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

- Status: corrected batch-10 attempt completed and fetched locally.
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
- Third failed output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_full_20260605T174046Z`.
- Third failed driver log:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/hfvla_full_eval_driver_20260605T174046Z.log`.
- Completed output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_full_b10_20260605T174446Z`.
- Completed local bundle:
  `_workspace/runpod_results/20260605T205127Z/smolvla_hfvla_libero_full_b10_20260605T174446Z`.

### Command Difference From Previous Run

```bash
SMOLVLA_MODEL_ID=HuggingFaceVLA/smolvla_libero
LIBERO_EXTRA_ARGS="--policy.num_steps=10 --policy.n_action_steps=1"
```

This is expected to be closer to published SmolVLA LIBERO settings than the
previous `lerobot/smolvla_libero` run.

### Failure And Fix

- First and second attempts failed before any videos were created.
- Initial error: feature mismatch. The HuggingFaceVLA checkpoint expected
  `observation.images.image`, `observation.images.image2`, and
  `observation.images.empty_camera_0`, but the previous camera mapping produced
  `observation.images.camera1` and `observation.images.camera2`.
- Follow-up bug: the shell expression
  `LIBERO_CAMERA_NAME_MAPPING="${LIBERO_CAMERA_NAME_MAPPING:-{...}}"` appended
  a stray `}` when `LIBERO_CAMERA_NAME_MAPPING=none`, producing `none}` and a
  draccus decode error.
- Fix: replace that default assignment with an explicit `if [ -z
  "${LIBERO_CAMERA_NAME_MAPPING+x}" ]` block, then rerun with
  `LIBERO_CAMERA_NAME_MAPPING=none` so LeRobot keeps the default image feature
  names for this checkpoint.

### Completed Result

- exit code: 0
- videos: 400
- metric file:
  `_workspace/runpod_results/20260605T205127Z/smolvla_hfvla_libero_full_b10_20260605T174446Z/eval_logs/eval_info.json`
- comparison report:
  `_workspace/runpod_results/20260605T205127Z/smolvla_hfvla_libero_full_b10_20260605T174446Z/comparison_report.md`
- overall success: 68.5%
- suite success:
  - `libero_spatial`: 75.0%
  - `libero_object`: 78.0%
  - `libero_goal`: 82.0%
  - `libero_10`: 39.0%
- interpretation: the cloud pipeline is verified, but this is not a close
  reproduction of published SmolVLA LIBERO numbers. The next debugging target is
  protocol parity: exact checkpoint identity, LeRobot commit/version, LIBERO
  assets/init states, action normalization/control mode, and whether the
  published table used a different SmolVLA checkpoint.

### 2026-06-05 Model Identity Check

- User concern: 68.5% overall success seemed high for a baseline.
- Verified from run command/log: the evaluated policy path was
  `HuggingFaceVLA/smolvla_libero`.
- Verified from both Hugging Face config and local `lerobot_eval.log`:
  `type=smolvla`, visual inputs are `observation.images.image` and
  `observation.images.image2`, action output is 7D, and runtime device is CUDA.
- Important baseline interpretation: this is a LIBERO-finetuned SmolVLA policy,
  not a random/zero/scripted baseline. It is a strong policy-only baseline for
  later agentic-wrapper comparisons.
- Config details explaining the baseline character: `train_expert_only=True`,
  `num_vlm_layers=0`, `vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct`,
  `num_steps=10`, and `n_action_steps=1`.
- Reference comparison source to keep with any table: ActionX Table 1 reports
  SmolVLA LIBERO success rates of Goal 91, Object 94, Spatial 93, Long 77,
  Average 88.8 under 10 tasks per suite and 10 trials per task.
- Current result remains materially below that reference, especially on
  `libero_10` Long: 39.0% versus 77.0%.

## 2026-06-05 Cloud Resourcing + Debug Plan

- I verified `RUNPOD_POD_ID=t8eqsuj7nzaou8` is in `EXITED` state and cannot be
  started (`There are not enough free GPUs on the host machine`).
- Attempted replacement Pod creation on the same network volume `tchm4gxfvd`:
  - `NVIDIA GeForce RTX 4090` (requested): `There are no instances currently available`.
  - `NVIDIA H200`: `could not find any pods with required specifications`.
  - `NVIDIA A40`: same error.
  - `NVIDIA H100 80GB HBM3`, `NVIDIA H100 SXM`, `NVIDIA A100-SXM4-80GB`,
    `NVIDIA RTX A2000`, `NVIDIA GeForce RTX 3070`, `NVIDIA RTX A4000`:
    no instances/invalid host availability errors from API.
- Result: 재현 비교 재실행은 현재 클라우드 자원 부족으로 보류.

### Prepared debug preset for next run

- Added `scripts/runpod_smolvla_libero_eval_paper.sh` and docs entry with a
  high-likelihood paper-parity preset:
  - `SMOLVLA_MODEL_ID=lerobot/smolvla_libero`
  - `POLICY_EMPTY_CAMERAS=0`
  - `LIBERO_BATCH_SIZE=10`
  - `LIBERO_EXTRA_ARGS="--policy.num_steps=10 --policy.n_action_steps=50"`
- Run order once pod is available:
 1. `libero_spatial LIBERO_TASK_IDS='[0,1,2,3,4]' LIBERO_N_EPISODES=5` for
     quick sanity.
 2. Full `paper-comparable` 400-episode run with the preset above.

### 2026-06-05 to 2026-06-06 Cloud Resourcing Escalation

- Community pool probing (same image/volume) continued to fail with:
  - `There are no instances currently available`
  - `could not find any pods with required specifications`
- Escalation to `RUNPOD_CLOUD_TYPE=SECURE` succeeded on
  `NVIDIA GeForce RTX 4090` with the existing volume `tchm4gxfvd`, creating:
  - `z1py6r6em95aa5` (running, 16 vCPU, costPerHr 0.69)
- Additional probe result still visible: `3f6so5bs0zz6k6` (running, started during
  same search, 6 vCPU, costPerHr 0.39, has mapped SSH port in RunPod status).
- Status now: there are live SECURE 4090 candidates, so benchmark continuation can
  proceed if SSH access is confirmed for one of them; otherwise keep one and
  convert its SSH path before the next evaluation.
- Post-check action:
  - Stopped the SECURE 4090 probe pod (`z1py6r6em95aa5`) to avoid a duplicate running Pod while continuing with
    `3f6so5bs0zz6k6` as the single active experiment Pod.

### 2026-06-05 Pod Cost Cleanup

- User correctly pointed out that stopped Pods can still carry small residual
  cost.
- Terminated stopped Pods:
  - `z1py6r6em95aa5` probe Pod
  - `t8eqsuj7nzaou8` previous eval Pod
- Verified current RunPod list now contains only:
  - `3f6so5bs0zz6k6` running, `NVIDIA L4`, `0.39/hr`, network volume
    `tchm4gxfvd`, SSH `root@213.173.105.22:30187`.
- Updated local uncommitted `.env` RunPod target to this active Pod.
- Current blocker for immediate rerun: active L4 Pod has repo checkout but does
  not have the prior Python 3.12 LeRobot environment at
  `/workspace/physical-ai/envs/lerobot_py312`; bootstrap is required before the
  next sanity evaluation.

### 2026-06-06 SmolVLA LIBERO Bootstrap Bottleneck

- Started a sanity evaluation on the active L4 Pod with:
  - `SMOLVLA_MODEL_ID=lerobot/smolvla_libero`
  - `LIBERO_TASKS=libero_spatial`
  - `LIBERO_TASK_IDS=[0,1]`
  - `LIBERO_N_EPISODES=5`
  - paper-parity policy args: `--policy.num_steps=10 --policy.n_action_steps=50`
- The run stayed in `pip install -e /workspace/physical-ai/vendor/lerobot[smolvla,libero]`
  for over 25 minutes before any GPU evaluation began.
- Evidence collected while running:
  - GPU memory/utilization remained `0 MiB / 0%`, so evaluation had not started.
  - `venv` on `/workspace/physical-ai/envs/lerobot_py312` grew from roughly
    `4.4G` to `11G`.
  - Newest venv files were robosuite robot mesh assets under
    `site-packages/robosuite/models/assets/...`, showing install was still
    writing many small files rather than hard-hanging.
- Likely cause: placing the Python venv on the RunPod network volume preserves
  it across Pods, but it is slow for Python package installs that write many
  small files.
- Follow-up change prepared for the next run:
  - export `PIP_CACHE_DIR` before bootstrap installs.
  - keep pip/HF/model/data caches under `/workspace/physical-ai`.
  - default the RunPod paper preset venv to
    `/root/physical-ai/envs/lerobot_py312` to use faster container disk for
    small-file package installs.
  - raise the create/probe script default container disk from `20GB` to `60GB`
    because the SmolVLA/LIBERO venv alone can exceed `13GB` before model and
    asset caches are considered.
- Operating rule: keep only the active experiment Pod running; terminate stopped
  Pods after results are in git or the network volume because stopped Pods can
  still carry residual hourly cost.

### 2026-06-06 SmolVLA LIBERO Spatial Sanity Results

- Completed paper-preset plumbing sanity:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_paper_sanity_20260605T235334Z`
  - model: `lerobot/smolvla_libero`
  - suite/tasks: `libero_spatial`, task ids `[0,1]`
  - episodes: `5` per task, `10` total
  - result: `80.0%` success, task 0 `4/5`, task 1 `4/5`
- Completed larger Spatial subset:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_spatial_5tasks_5eps_20260606T003702Z`
  - model: `lerobot/smolvla_libero`
  - suite/tasks: `libero_spatial`, task ids `[0,1,2,3,4]`
  - episodes: `5` per task, `25` total
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=50`
  - result: `72.0%` success
  - per-task successes: task 0 `4/5`, task 1 `4/5`, task 2 `4/5`,
    task 3 `4/5`, task 4 `2/5`
- Interpretation:
  - The evaluation pipeline is now real: generated videos and `eval_info.json`
    for every episode.
  - The result is still materially below the external Spatial reference of
    roughly `93%`, so this is not yet a satisfactory baseline parity run.
  - High-probability mismatch: the current paper preset forces
    `--policy.n_action_steps=50`, while LeRobot LIBERO default evaluation does
    not require that for SmolVLA and historical SmolVLA reproduction commands
    use `--policy.n_action_steps=1`.
- Next debug run launched:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_5tasks_5eps_nas1_20260606T005516Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - same `libero_spatial` task ids `[0,1,2,3,4]`, `5` episodes per task
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=1`
- Additional blocker found:
  - the existing network-volume venv installed `torch 2.11.0+cu130`.
  - RunPod L4 driver reports CUDA driver version `12080`, so PyTorch CUDA 13
    reports `torch.cuda.is_available() == False`.
  - As a result, attempted CUDA evals fell back to CPU despite
    `--policy.device=cuda`.
- Attempted repair:
  - checked PyTorch wheel availability and found `torch==2.11.0+cu128`,
    `torchvision==0.26.0+cu128`, and `torchcodec==0.11.1+cu128`.
  - deleted unused repo `.venv` and failed HuggingFaceVLA partial cache,
    reducing `/workspace/physical-ai` from about `22GB` to `19GB`.
  - `pip install --force-reinstall --no-cache-dir --index-url
    https://download.pytorch.org/whl/cu128 ...` still failed with
    `Disk quota exceeded` while replacing packages inside the network-volume
    venv.
- Decision:
  - stop repairing the network-volume venv.
  - create the next Pod with `RUNPOD_CONTAINER_DISK_GB=60`.
  - keep venv/build/temp under container disk (`/root/physical-ai/...`) and
    preserve only repo, HF cache, LIBERO assets, videos, metrics, and reports
    under `/workspace`.

### 2026-06-06 New 60GB Container Disk Pod

- Created a replacement SECURE Pod after Community pool availability failed:
  - Pod id: `nu2iyu4s8nqmbl`
  - GPU: `NVIDIA GeForce RTX 4090`
  - driver: `580.126.20`
  - container disk: `60GB`
  - cost: `$0.69/hr`
  - network volume: `tchm4gxfvd`
  - SSH: `root@213.173.109.89:18329`
- Verified SSH, GPU, and disk on the new Pod.
- Terminated the old L4 Pod `3f6so5bs0zz6k6` after confirming the new Pod, so
  only one billable Pod remains.
- Updated the paper preset default toward the current best debug hypothesis:
  - `SMOLVLA_MODEL_ID=HuggingFaceVLA/smolvla_libero`
  - `--policy.num_steps=10`
  - `--policy.n_action_steps=1`
  - `--policy.device=cuda`
  - `--policy.empty_cameras=0`

### 2026-06-06 GPU SmolVLA Spatial Subset

- Verified new root/container-disk venv on the 4090 Pod:
  - torch: `2.11.0+cu130`
  - CUDA available: `True`
  - GPU: `NVIDIA GeForce RTX 4090`
- First HuggingFaceVLA run failed due feature mismatch:
  - policy expected `observation.images.image` and
    `observation.images.image2`.
  - env mapping produced `observation.images.camera1` and
    `observation.images.camera2`.
- Fixed by overriding camera mapping:
  - `LIBERO_CAMERA_NAME_MAPPING={"agentview_image": "image",
    "robot0_eye_in_hand_image": "image2"}`
- Completed GPU subset:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_5tasks_5eps_cuda_imagenames_20260606T013004Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - suite/tasks: `libero_spatial`, task ids `[0,1,2,3,4]`
  - episodes: `5` per task, `25` total
  - result: `80.0%` success
  - per-task successes: task 0 `3/5`, task 1 `4/5`, task 2 `5/5`,
    task 3 `4/5`, task 4 `4/5`
  - eval speed: `484.7s` total, `19.4s/episode`
- Interpretation:
  - CPU fallback, old-driver CUDA mismatch, network-volume venv quota, and
    feature-name mismatch are now resolved for the current best-known setup.
  - The 25-episode subset is still below the external Spatial reference
    (`93%`), so full Spatial 10-task/10-episode evaluation is required before
    judging baseline parity.
- Launched full Spatial evaluation:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_full_10eps_cuda_20260606T014347Z`
  - task suite: all `libero_spatial`
  - episodes: `10` per task, `100` total

### 2026-06-06 Full Spatial Result

- Completed full Spatial run:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_full_10eps_cuda_20260606T014347Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - suite: all `libero_spatial`
  - episodes: `10` per task, `100` total
  - result: `69.0%` success
  - eval speed: `1422.4s` total, `14.2s/episode`
  - per-task successes:
    - task 0: `6/10`
    - task 1: `8/10`
    - task 2: `7/10`
    - task 3: `8/10`
    - task 4: `6/10`
    - task 5: `4/10`
    - task 6: `5/10`
    - task 7: `10/10`
    - task 8: `8/10`
    - task 9: `7/10`
- Interpretation:
  - This is now a valid GPU evaluation with CUDA active and correct feature
    names for the HuggingFaceVLA checkpoint.
  - The result is close to historical LeRobot reproduction reports around the
    low 70s on Spatial, but far below the ActionX table reference of `93%`.
  - Next high-probability cause to test is checkpoint/model-id mismatch.
- Launched checkpoint comparison:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_5tasks_5eps_nas1_cuda_20260606T021418Z`
  - model: `lerobot/smolvla_libero`
  - suite/tasks: `libero_spatial`, task ids `[0,1,2,3,4]`
  - episodes: `5` per task, `25` total
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=1
    --policy.device=cuda --policy.empty_cameras=0`

### 2026-06-06 Checkpoint Identity Debug Result

- The first checkpoint-comparison launch used the deleted network-volume venv
  and failed before evaluation. Relaunched with the root/container-disk venv:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_5tasks_5eps_nas1_cuda_rootvenv_20260606T022020Z`
  - model: `lerobot/smolvla_libero`
  - suite/tasks: `libero_spatial`, task ids `[0,1,2,3,4]`
  - episodes: `5` per task, `25` total
  - batch size: `1`
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=1
    --policy.device=cuda --policy.empty_cameras=0`
  - result: `96.0%` success
  - per-task successes: task 0 `5/5`, task 1 `5/5`, task 2 `5/5`,
    task 3 `5/5`, task 4 `4/5`
  - eval speed: `372.4s` total, `14.9s/episode`
- Interpretation:
  - The largest mismatch was checkpoint identity, not GPU plumbing.
  - `HuggingFaceVLA/smolvla_libero` is valid to evaluate, but it is not the
    current best candidate for matching the external `SmolVLA` Spatial `93%`
    reference.
  - The paper-parity preset should default back to `lerobot/smolvla_libero`,
    `--policy.n_action_steps=1`, and CUDA.
- Launched full Spatial validation:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_10eps_cuda_20260606T022852Z`
  - suite: all `libero_spatial`
  - episodes: `10` per task, `100` total
  - model: `lerobot/smolvla_libero`
  - batch size: `10`

### 2026-06-06 MuJoCo Version Debug

- Full Spatial with `lerobot/smolvla_libero` on MuJoCo `3.9.0` completed:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_10eps_cuda_20260606T022852Z`
  - result: `78.0%`
  - per-task successes:
    - task 0: `9/10`
    - task 1: `9/10`
    - task 2: `9/10`
    - task 3: `10/10`
    - task 4: `7/10`
    - task 5: `0/10`
    - task 6: `9/10`
    - task 7: `8/10`
    - task 8: `7/10`
    - task 9: `10/10`
- Version check:
  - MuJoCo: `3.9.0`
  - robosuite: `1.4.0`
  - LeRobot: `0.5.2`
  - LeRobot checkout: `09808183ca72c30cbb41b653586f6d0632a4bcca`
- Task 5 instruction:
  `pick up the black bowl on the ramekin and place it on the plate`
- Downgraded the root/container venv to MuJoCo `3.3.2`, matching the LeRobot
  maintainer note that older/newer MuJoCo versions can render colors
  differently.
- Re-ran task 5 only:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_task5_mj332_10eps_cuda_20260606T025325Z`
  - result: `90.0%`, task 5 `9/10`
- Interpretation:
  - MuJoCo version/rendering mismatch is a confirmed high-impact baseline
    parity issue.
  - The Linux evaluator now defaults to `MUJOCO_VERSION=3.3.2`.
- Launched full Spatial validation on MuJoCo `3.3.2`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_mj332_10eps_cuda_20260606T025657Z`
  - suite: all `libero_spatial`
  - episodes: `10` per task, `100` total

### 2026-06-06 MuJoCo 3.3.2 Full Spatial Result

- Completed full Spatial run on MuJoCo `3.3.2`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_mj332_10eps_cuda_20260606T025657Z`
  - model: `lerobot/smolvla_libero`
  - result: `87.0%`
  - eval speed: `1144.5s` total, `11.4s/episode`
  - per-task successes:
    - task 0: `9/10`
    - task 1: `8/10`
    - task 2: `9/10`
    - task 3: `10/10`
    - task 4: `7/10`
    - task 5: `8/10`
    - task 6: `10/10`
    - task 7: `8/10`
    - task 8: `8/10`
    - task 9: `10/10`
- Compared to the MuJoCo `3.9.0` run, Spatial improved from `78.0%` to
  `87.0%`, and task 5 recovered from `0/10` to `8/10`.
- Tested `POLICY_EMPTY_CAMERAS=1` on hard task ids `[4,5,7,8]`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_hardtasks_mj332_emptycam1_10eps_cuda_20260606T032203Z`
  - result: `82.5%` across 40 episodes
  - per-task successes: task 4 `9/10`, task 5 `8/10`, task 7 `9/10`,
    task 8 `7/10`
- Interpretation:
  - `empty_cameras=1` did not clearly improve the hard subset total; it changed
    which tasks failed but not the aggregate.
  - MuJoCo `3.3.2` is a confirmed necessary setting, but it is not sufficient
    to match the ActionX Spatial `93%` reference.
- Launched the official LeRobot ported checkpoint under the same MuJoCo `3.3.2`
  condition:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_full_mj332_10eps_cuda_20260606T033221Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - camera mapping: `{"agentview_image": "image",
    "robot0_eye_in_hand_image": "image2"}`
  - suite: all `libero_spatial`
  - episodes: `10` per task, `100` total

### 2026-06-06 HuggingFaceVLA Checkpoint Result

- Completed full Spatial run on MuJoCo `3.3.2`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_spatial_full_mj332_10eps_cuda_20260606T033221Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - result: `72.0%`
  - eval speed: `1517.1s` total, `15.2s/episode`
  - per-task successes:
    - task 0: `6/10`
    - task 1: `9/10`
    - task 2: `7/10`
    - task 3: `8/10`
    - task 4: `5/10`
    - task 5: `7/10`
    - task 6: `6/10`
    - task 7: `9/10`
    - task 8: `9/10`
    - task 9: `6/10`
- Interpretation:
  - `HuggingFaceVLA/smolvla_libero` remains below the current best
    `lerobot/smolvla_libero` result under this harness.
  - The best validated Spatial result so far is `87.0%` with
    `lerobot/smolvla_libero`, MuJoCo `3.3.2`, `batch_size=10`.
  - The next high-probability parity issue is batch handling: an earlier
    `batch_size=1` 25-episode subset reached `96.0%`, while `batch_size=10`
    full Spatial reached `87.0%`.
- Launched full Spatial validation with `batch_size=1`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_mj332_b1_10eps_cuda_20260606T040105Z`
  - model: `lerobot/smolvla_libero`
  - suite: all `libero_spatial`
  - episodes: `10` per task, `100` total

### 2026-06-06 Batch-Size And Action-Step Debug

- Completed full Spatial run with `batch_size=1`:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_spatial_full_mj332_b1_10eps_cuda_20260606T040105Z`
  - result: `88.0%`
  - eval speed: `1733.4s` total, `17.3s/episode`
  - per-task successes:
    - task 0: `10/10`
    - task 1: `9/10`
    - task 2: `10/10`
    - task 3: `9/10`
    - task 4: `6/10`
    - task 5: `8/10`
    - task 6: `10/10`
    - task 7: `9/10`
    - task 8: `8/10`
    - task 9: `9/10`
- Interpretation:
  - `batch_size=1` improves Spatial by only one point over `batch_size=10`
    (`88.0%` vs `87.0%`), so batch handling is not the main remaining gap.
  - The persistent gap is task 4 (`top drawer`) and partially tasks 5/8.
- Ran task 4 action-step sweep:
  - `n_action_steps=5`: task 4 `5/10`
  - `n_action_steps=10`: task 4 `7/10`
  - `n_action_steps=50`: task 4 `5/10`
- Interpretation:
  - larger action chunks do not solve task 4.
  - The current best internal Spatial baseline is `88.0%` using
    `lerobot/smolvla_libero`, MuJoCo `3.3.2`, `batch_size=1`,
    `n_action_steps=1`, CUDA, and image inputs.
  - This remains below ActionX Table 1's SmolVLA Spatial reference of `93%`.
    The likely reason is protocol/checkpoint/control-mode mismatch: ActionX
    reports fine-tuned LIBERO baselines with normalized absolute Cartesian pose
    actions, while the LeRobot run logs `control_mode=relative`.
- Launched full 4-suite LIBERO evaluation with the current best setting:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z`
  - suites: `libero_spatial,libero_object,libero_goal,libero_10`
  - episodes: `10` per task, expected `400` total

### 2026-06-06 Full 4-Suite LIBERO Result

- Completed the full 4-suite, 400-episode run:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z`
  - model: `lerobot/smolvla_libero`
  - MuJoCo: `3.3.2`
  - batch size: `1`
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=1 --policy.device=cuda`
  - total result: `76.25%`
  - eval speed: `11066.5s` total, `27.7s/episode`
- Suite results:
  - `libero_spatial`: `89.0%`
  - `libero_object`: `78.0%`
  - `libero_goal`: `79.0%`
  - `libero_10`: `59.0%`
- External reference used for side-by-side comparison:
  - ActionX Table 1 SmolVLA: Goal `91.0`, Object `94.0`, Spatial `93.0`,
    Long `77.0`, Average `88.8`
  - reference URL:
    `https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full`
- Local handoff:
  - copied `eval_info.json` and `smolvla_libero_report.md` locally under:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z`
  - left the 400 rollout videos on the RunPod network volume to avoid filling
    the local Mac workspace.
- Interpretation:
  - The run is now format-comparable with the reference table: 4 suites, 10
    tasks per suite, 10 episodes per task.
  - The result is not performance-comparable yet. The largest remaining gap is
    Long/`libero_10` (`59.0%` internal vs `77.0%` reference), followed by
    Object and Goal.
  - Current working hypothesis remains checkpoint/control protocol mismatch:
    the LeRobot hub checkpoint logs `control_mode=relative`, while the external
    reference describes normalized absolute Cartesian pose actions.

### 2026-06-06 HF Checkpoint Parity Run And Volume Cleanup

- Operating rule update:
  - If the baseline still needs parity debugging, keep the active RunPod Pod
    running instead of stopping it between attempts.
  - Stop the Pod only after a satisfactory milestone, local handoff, and report
    update, or when explicitly requested.
- The previous stopped Pod could not restart because its host had no free GPU:
  - stopped Pod id: `nu2iyu4s8nqmbl`
  - error: `There are not enough free GPUs on the host machine to start this pod`
- Created a replacement SECURE 4090 Pod attached to the same network volume:
  - new Pod id: `ldwpvij20awxqi`
  - SSH: `root@213.173.109.208`, port `19632`
  - network volume id: `tchm4gxfvd`
  - GPU: `NVIDIA GeForce RTX 4090`
- Tried to bootstrap the reusable network-volume venv:
  - path: `/workspace/physical-ai/envs/lerobot_py312`
  - issue: `pip install -e lerobot[smolvla,libero]` stayed slow on network
    volume I/O.
  - decision: terminate that bootstrap and use a root/container-disk venv for
    the current continuous debugging session.
- Root/container venv smoke passed:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_smoke_mj332_rootvenv_b1_20260606T0810Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - camera mapping:
    `{"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}`
  - MuJoCo: `3.3.2`
  - result: `1/1`, success `100.0%`
- Launched current HF checkpoint 4-suite parity run:
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_all_mj332_rootvenv_b1_10eps_cuda_20260606T0814Z`
  - model: `HuggingFaceVLA/smolvla_libero`
  - suites: `libero_spatial,libero_object,libero_goal,libero_10`
  - episodes: `10` per task, expected `400` total
  - batch size: `1`
  - policy args: `--policy.num_steps=10 --policy.n_action_steps=1 --policy.device=cuda`
  - camera mapping:
    `{"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}`
  - latest checked progress: `78/400` videos at `2026-06-06T09:04:06Z`
  - status: running
- Cleaned the RunPod network volume while preserving the active run:
  - local backup:
    `_workspace/runpod_results/remote_archives/runpod_results_before_cleanup_20260606T0820Z.tar.gz`
  - deleted old remote run result directories after backup.
  - deleted the incomplete network-volume venv and pip cache:
    `/workspace/physical-ai/envs/lerobot_py312`,
    `/workspace/physical-ai/pip_cache`
  - retained HF model cache and LIBERO assets for the active and next runs.
  - `/workspace/physical-ai` usage dropped from roughly `19G` to `7.1G`.
- Added `scripts/runpod_archive_results.sh` as the standard mid-run cleanup
  tool:
  - fetches remote `runpod_results` into local
    `_workspace/runpod_results/remote_archives/`
  - deletes only completed remote result directories with
    `eval_logs/eval_info.json` when called with
    `--delete-remote --yes-delete`
  - preserves `RUNPOD_ACTIVE_RESULT_DIR`
- Volume check at `2026-06-06T09:51:41Z`:
  - `/workspace/physical-ai`: `7.1G`
  - active run:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_all_mj332_rootvenv_b1_10eps_cuda_20260606T0814Z`
  - active run size: `41M`
  - active videos: `148`
  - status: still running
  - retained data: HF cache `5.9G`, LIBERO assets `710M`, vendor `463M`
- Archive script verification:
  - first no-delete full fetch completed locally at
    `_workspace/runpod_results/remote_archives/runpod_results_20260606T095321Z`
  - then changed the script default to completed-result-only archives so it
    does not copy an active run during evaluation.
  - verified completed-only mode with the active run excluded:
    `no_completed_remote_results=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results`
  - follow-up progress check at `2026-06-06T09:57:38Z`: active videos `158`,
    active size `43M`, `/workspace/physical-ai` still `7.1G`, GPU memory
    `2406/24564 MiB`.
- References currently tracked:
  - ActionX Table 1 SmolVLA: Goal `91.0`, Object `94.0`, Spatial `93.0`,
    Long `77.0`, Average `88.8`
  - LeRobot/HF reproduction issue `huggingface/lerobot#2354` for public
    checkpoint reality checks.
  - LeRobot LIBERO docs for protocol and command shape.
