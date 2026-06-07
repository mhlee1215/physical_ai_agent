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

## 2026-06-06 Low-Cost Affordance Overlay Probe Pod

### Current State

- Status: stopped after blocker capture.
- Purpose: create a separate low-cost Pod so CP24 affordance-overlay probing did
  not interfere with the active LIBERO evaluation.
- Pod: `ynyy1lwc5qnnef`.
- GPU: NVIDIA RTX 2000 Ada Generation 16GB.
- Cost at creation: `$0.24/hr`.
- Cloud type: SECURE.
- Container disk: 20GB.
- Network volume: `tchm4gxfvd`.
- Separate clone path:
  `/workspace/physical-ai/physical_ai_agent_affordance_probe`.

### Commands

```bash
PYTHONPATH=src .venv/bin/python -B -m physical_ai_agent.checkpoints.checkpoint_24 \
  --require-maniskill \
  --episodes 1 \
  --steps 1 \
  --policy affordance_oracle_probe \
  --real-images \
  --output-dir _workspace/checkpoints/runpod_affordance_oracle_probe_rtx2000_1step
```

### Result

- Result: blocked before rollout.
- Output path:
  `/workspace/physical-ai/physical_ai_agent_affordance_probe/_workspace/checkpoints/runpod_affordance_oracle_probe_rtx2000_1step`.
- Blocker:
  `vk::createInstanceUnique: ErrorIncompatibleDriver`.
- `vulkaninfo --summary` saw only CPU `llvmpipe`, not the RTX 2000 Ada Vulkan
  device.
- Installing `libvulkan1` and `vulkan-tools` did not expose NVIDIA Vulkan.

### Interpretation

This low-cost Pod confirmed the same limitation as earlier RunPod templates:
CUDA is available, but ManiSkill/SAPIEN RGB manipulation tasks need a working
NVIDIA Vulkan device. The overlay code should be tested on a template or host
where `vulkaninfo --summary` lists the NVIDIA GPU, or on a benchmark path that
does not require SAPIEN Vulkan rendering.

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
- Parallelization check:
  - current active run remains conservative on purpose:
    `batch_size=1`, `use_async_envs=false`, `max_parallel_tasks=1`.
  - RunPod resources at `2026-06-06T10:01:29Z`: 32 CPU cores, 124GiB RAM,
    RTX 4090 using about `2406/24564 MiB` VRAM and about `33%` GPU util.
  - LeRobot source confirms `eval.batch_size` creates multiple envs,
    `eval.use_async_envs` uses async envs, and `env.max_parallel_tasks` runs
    task-level work with a thread pool.
  - Added `experiments/libero_runpod_variants/runpod_smolvla_libero_parallel_probe.sh` to compare:
    `b1_sync_t1`, `b4_async_t1`, `b8_async_t1`, and `b4_async_t2` on a small
    fixed subset before changing a full benchmark run.
  - Preferred next acceleration path is `batch_size + async`; task-thread
    parallelism is higher risk because it shares one policy across task
    workers and must be regression-checked before full-run use.
- Registered a RunPod wait-wrapper so the active 400-episode run is not
  interrupted:
  - waiter pid: `6339`
  - waiter log:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/parallel_probe_waiter_20260606T100356Z.log`
  - probe root after active completion:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_parallel_probe_after_hfvla_20260606T100356Z`
  - latest active progress at `2026-06-06T10:04:15Z`: `167/400` videos,
    still running, GPU memory `2406/24564 MiB`.
- Added `scripts/compare_smolvla_libero_results.py` to generate comparison
  tables from any completed LeRobot `eval_info.json`.
  - Verified it against the previous 400-episode
    `lerobot/smolvla_libero` result.
  - It reports ActionX Table 1 deltas and LeRobot/HF issue `#2354` public
    checkpoint repro/paper deltas separately.
  - Important interpretation from the previous run: average was `-12.55` vs
    ActionX, but `+3.5` vs the HF public-checkpoint repro line; those are not
    the same external baseline.
- Active progress check at `2026-06-06T10:14:05Z`: `181/400` videos, still
  running, GPU memory `2404/24564 MiB`.
- References currently tracked:
  - ActionX Table 1 SmolVLA: Goal `91.0`, Object `94.0`, Spatial `93.0`,
    Long `77.0`, Average `88.8`
  - LeRobot/HF reproduction issue `huggingface/lerobot#2354` for public
    checkpoint reality checks.
  - LeRobot LIBERO docs for protocol and command shape.

### 2026-06-06 Steps10 Parity And Process-Lane Scaling

- Completed `HuggingFaceVLA/smolvla_libero` full 400-episode run:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_hfvla_libero_all_mj332_rootvenv_b1_10eps_cuda_20260606T0814Z`
  - result: Goal `66.0`, Object `92.0`, Spatial `70.0`, Long `39.0`, Avg
    `66.75`
  - interpretation: not the ActionX-parity path. Object is good, but Spatial,
    Goal, and Long are much worse than the `lerobot/smolvla_libero` baseline.
- Completed process-level parallel probes:
  - HF checkpoint probe:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_parallel_probe_after_hfvla_20260606T100356Z`
    showed little speedup from async/batched envs and worse result with task
    thread parallelism.
  - `lerobot/smolvla_libero` probe:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_parallel_probe_lerobot_fixed_20260606T144929Z`
    showed `b1_sync_t1` stayed best: `100.0%`, `96s`; async/batched variants
    dropped to `75.0%`.
  - conclusion: do not use intra-eval batching/async/task-thread parallelism
    for reported baseline numbers.
- Completed Long protocol sweep on hard Long subset `[0,1,6,7,8]`, 5 episodes
  per task:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_long_protocol_sweep_fixed_20260606T150943Z`
  - `seed=1000, n_action_steps=1`: `40.0%`
  - `seed=0, n_action_steps=1`: `24.0%`
  - `seed=10000, n_action_steps=1`: `32.0%`
  - `seed=1000, n_action_steps=10`: `68.0%`
  - `seed=1000, n_action_steps=50`: `28.0%`
  - conclusion: seed changes did not help; `n_action_steps=10` is the first
    strong Long improvement signal.
- Added `n_action_steps=15` to future protocol sweeps because the user noted
  this setting from the π0.7 paper. Do not interrupt the active full run for
  this; run it as the next targeted check if steps10 full result leaves a gap.
- Completed dual-process canary:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_dual_process_probe_20260606T164816Z`
  - sequential lanes: `116s` wall-clock
  - concurrent lanes: `61s` wall-clock
  - all lanes kept `100.0%` success.
  - conclusion: independent process-level lanes can reduce wall-clock while
    preserving each lane's conservative eval settings.
- Completed Long full split run with `n_action_steps=10`:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_long_full_steps10_split_20260606T165555Z`
  - lane A `[0,1,2,3,4]`: `80.0%`
  - lane B `[5,6,7,8,9]`: `70.0%`
  - aggregate Long: `75.0%`, close to ActionX Long `77.0`
  - wall-clock: `726s`
  - interpretation: `n_action_steps=10` is now the current best candidate for
    ActionX-parity full 4-suite evaluation.
- Launched active full 4-suite two-lane validation with
  `n_action_steps=10`:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z`
  - lane A: `libero_spatial,libero_object`
  - lane B: `libero_goal,libero_10`
  - eval settings per lane: `batch_size=1`, `use_async_envs=false`,
    `max_parallel_tasks=1`
  - latest checked progress at `2026-06-06T17:18:36Z`: `83/400` videos,
    still running; GPU was `32%` with `4237/24564 MiB` used.
- Added `experiments/libero_runpod_variants/runpod_smolvla_libero_multi_process_probe.sh` to test
  triple-process scaling after the active full run completes. Do not run it
  concurrently with the active reported evaluation because it could contaminate
  timing and possibly success.
- Completed the full 4-suite two-lane `n_action_steps=10` run:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z`
  - local compact artifact:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z`
  - result: Goal `85.0`, Object `92.0`, Spatial `91.0`, Long `73.0`, Avg
    `85.25`
  - delta vs ActionX Table 1 SmolVLA: Goal `-6.0`, Object `-2.0`, Spatial
    `-2.0`, Long `-4.0`, Avg `-3.55`
  - delta vs LeRobot/HF issue `#2354` paper line: Goal `-7.0`, Object `-4.0`,
    Spatial `+1.0`, Long `+2.0`, Avg `-2.0`
  - delta vs LeRobot/HF issue `#2354` public repro line: Goal `+2.0`, Object
    `+1.0`, Spatial `+18.0`, Long `+30.0`, Avg `+12.5`
  - interpretation: this is the first close policy-only baseline. It is not an
    exact ActionX reproduction, but the remaining ActionX average gap is now
    small enough to make targeted protocol checks meaningful.
- Added `scripts/merge_lerobot_eval_infos.py` to merge split-lane
  `eval_info.json` files into a single 400-episode artifact before running
  comparison tables.
- Next debug checks after this result:
  - run the already-registered `n_action_steps=15` protocol check because the
    user noted pi0.7 uses this setting.
  - run triple-process canary only as an acceleration check, not as a reported
    metric change.
- Completed the full 4-suite two-lane `n_action_steps=15` run:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_full_steps15_two_lane_20260606T1753Z`
  - local compact artifact:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_full_steps15_two_lane_20260606T1753Z`
  - result: Goal `89.0`, Object `93.0`, Spatial `86.0`, Long `74.0`, Avg
    `85.5`
  - delta vs ActionX Table 1 SmolVLA: Goal `-2.0`, Object `-1.0`, Spatial
    `-7.0`, Long `-3.0`, Avg `-3.3`
  - delta vs LeRobot/HF issue `#2354` paper line: Goal `-3.0`, Object `-3.0`,
    Spatial `-4.0`, Long `+3.0`, Avg `-1.75`
  - interpretation: `n_action_steps=15` is the best-average internal baseline
    so far, but `n_action_steps=10` is more balanced because Spatial stays
    much closer to ActionX.
  - next question: whether Spatial can be recovered while keeping the
    Goal/Object/Long gains. Candidate checks are per-suite action-step routing
    (`Spatial=10`, other suites `15`) or a small `n_action_steps=12` full run.
- Completed routed full run with `Spatial n_action_steps=10` and
  `Object/Goal/Long n_action_steps=15`:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_routed_spatial10_rest15_20260606T1829Z`
  - local compact artifact:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_routed_spatial10_rest15_20260606T1829Z`
  - result: Goal `91.0`, Object `94.0`, Spatial `91.0`, Long `75.0`, Avg
    `87.75`
  - delta vs ActionX Table 1 SmolVLA: Goal `0.0`, Object `0.0`, Spatial
    `-2.0`, Long `-2.0`, Avg `-1.05`
  - delta vs LeRobot/HF issue `#2354` paper line: Goal `-1.0`, Object `-2.0`,
    Spatial `+1.0`, Long `+4.0`, Avg `+0.5`
  - interpretation: this is now the closest internal policy-only baseline.
    It matches ActionX Goal/Object exactly and leaves only Spatial/Long gaps.
  - remaining targeted debug candidates:
    - Long task-focused routing, because Long tasks `6` and `8` remain weak
      at `3/10` and `4/10`.
    - Spatial-focused check only if we need to close the last two Spatial
      points; steps10 already appears best for Spatial among tested settings.
- Completed Long-only task-routed probe:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_long_task_routed_probe_20260606T1915Z`
  - local compact artifact:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_long_task_routed_probe_20260606T1915Z`
  - protocol: Long tasks `[4,8]` used `n_action_steps=10`; tasks
    `[0,1,2,3,5,6,7,9]` used `n_action_steps=15`.
  - result: Long `71.0`, below the current routed full baseline Long `75.0`.
  - per-task counts: task `0` `4/10`, `1` `10/10`, `2` `10/10`, `3` `7/10`,
    `4` `2/10`, `5` `10/10`, `6` `7/10`, `7` `7/10`, `8` `5/10`, `9`
    `9/10`.
  - interpretation: task-level routing is not safe to adopt from subset
    probes. Task `6` improved, but tasks `3`, `4`, and `7` regressed. The
    subset run also changes task order and likely episode/RNG details, so it is
    useful as debug evidence but not as a reportable full-baseline replacement.
  - next recommended step: repeat the current best routed full run to estimate
    variance and confirm whether the ActionX average gap around `-1.05` is
    stable.
- Completed routed full repeat with the same `Spatial=10`, `Object/Goal/Long=15`
  protocol:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_routed_spatial10_rest15_repeat_20260606T1933Z`
  - local compact artifact:
    `_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_routed_spatial10_rest15_repeat_20260606T1933Z`
  - result: Goal `91.0`, Object `94.0`, Spatial `91.0`, Long `75.0`, Avg
    `87.75`
  - delta vs ActionX Table 1 SmolVLA: Goal `0.0`, Object `0.0`, Spatial
    `-2.0`, Long `-2.0`, Avg `-1.05`
  - repeat matched the previous routed full result exactly, including per-task
    counts. This is now the repeat-confirmed internal policy-only baseline.
  - conclusion: baseline parity is close enough for the next stage of
    agentic-wrapper experiments, while reporting the remaining `-1.05` average
    gap and the fact that ActionX may use a different exact checkpoint/control
    protocol.

### 2026-06-06 Basic Agentic Retry Wrapper

- Added the first LIBERO/SmolVLA agentic retry layer:
  - source: `src/physical_ai_agent/agent_core/libero_agentic_retry.py`
  - runner: `experiments/libero_runpod_variants/runpod_smolvla_libero_agentic_retry_probe.sh`
  - test: `tests/test_libero_agentic_retry.py`
  - report: `docs/smolvla_libero_agentic_retry_report.md`
- Semantics:
  - verifier reads the LIBERO benchmark `success` flag from `eval_info.json`.
  - planner schedules retry for task ids that have failed baseline episodes.
  - retry executor reruns those failed task ids once.
  - aggregator reports `success_once_rate`, counting an episode as successful
    if either baseline or retry succeeds for the same task/episode index.
  - This is an episode-level retry wrapper, not yet a subgoal-level in-episode
    controller.
- Completed same-protocol RunPod probe:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_probe_20260606T2036Z`
  - local:
    `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_probe_20260606T2036Z`
  - subset: `libero_10`, task ids `[0,6,8]`, `2` episodes per task
  - baseline and retry args: `n_action_steps=15`, `seed=1000`
  - result: baseline `50.0`, success-once `50.0`, recovered `0/3`
  - interpretation: plumbing passed, but identical retry protocol did not
    recover failures.
- Completed alternate-protocol RunPod probe:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_probe_20260606T2042Z`
  - local:
    `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_probe_20260606T2042Z`
  - baseline args: `n_action_steps=15`, `seed=1000`
  - retry args: `n_action_steps=10`, `seed=1001`
  - result: baseline `50.0`, success-once `66.67`, recovered `1/3`
  - interpretation: first positive SmolVLA/LIBERO agentic retry signal. The
    sample is too small for a paper-scale claim, but it proves the wrapper can
    schedule, execute, trace, and recover a failed task-episode index.
- Next recommended run:
  - same alternate retry protocol on `libero_10` task ids `[0,6,8]` with `10`
    episodes per task.
  - if recovery remains positive, scale to the full Long suite before comparing
    against the repeat-confirmed policy-only baseline.
- Completed 30-episode alternate-protocol sanity run:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_30ep_20260606T2049Z`
  - local:
    `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_30ep_20260606T2049Z`
  - subset: `libero_10`, task ids `[0,6,8]`, `10` episodes per task
  - baseline args: `n_action_steps=15`, `seed=1000`
  - retry args: `n_action_steps=10`, `seed=1001`
  - result: baseline `50.0`, success-once `70.0`, recovered `6/15`
  - interpretation: positive retry signal persisted at a larger subset scale.
    Next step is full Long-suite agentic retry, still with explicit disclosure
    that this wrapper retries task/episode indexes rather than intervening
    mid-episode.
- Started a full Long-suite run at
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_long_full_20260606T2105Z`,
  but stopped it because `LIBERO_TASK_IDS=` fell back to the script default
  `[0,6,8]`. Added `LIBERO_TASK_IDS=all` support in
  `experiments/libero_runpod_variants/runpod_smolvla_libero_agentic_retry_probe.sh` before relaunching.
- Completed full Long-suite alternate-protocol agentic retry:
  - output:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_long_full_20260606T2108Z`
  - local:
    `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_long_full_20260606T2108Z`
  - suite: `libero_10`, all task ids, `10` episodes per task
  - baseline args: `n_action_steps=15`, `seed=1000`
  - retry args: `n_action_steps=10`, `seed=1001`
  - result: baseline `71.0`, success-once `86.0`, recovered `15/29`
  - strongest per-task recovery:
    - task `6`: baseline `1/10`, recovered `6/9`, success-once `7/10`
    - task `8`: baseline `7/10`, recovered `3/3`, success-once `10/10`
    - task `4`: baseline `4/10`, recovered `2/6`, success-once `6/10`
  - interpretation: first full Long-suite positive agentic retry result. It
    should be repeated before paper-scale claims, and reported against its
    same-run baseline because the policy-only routed parity baseline used a
    different split-suite protocol.

### 2026-06-06 Post-LIBERO Evaluation Candidate Scan

- Added `docs/evaluation_candidate_matrix_after_libero.md`.
- Main conclusion:
  - keep LIBERO as the immediate paper-comparable anchor because the
    repeat-confirmed SmolVLA baseline is already within `-1.05` average points
    of the ActionX Table 1 SmolVLA reference.
  - next executable comparison should be a repeat of the full Long-suite
    agentic retry run, then all-suite LIBERO agentic retry if the recovery
    signal persists.
  - best next external benchmark is RoboCasa365/RoboCasa because the current
    leaderboard is a 50-task household manipulation benchmark with atomic,
    composite-seen, and composite-unseen splits, which matches the
    planner/verifier/retry research question better than generic short-horizon
    manipulation suites.
  - ManiSkill-HAB remains useful for smaller subtask success-once experiments,
    but it is lower priority for SmolVLA paper comparability because the
    published baselines are mostly RL/IL low-level mobile manipulation policies,
    and serious GPU-scale execution needs Linux/NVIDIA/SAPIEN validation.
  - vla-evaluation-harness is promising infrastructure for cross-benchmark
    evaluation, but SmolVLA may require a custom model server before it can
    replace the repo's working LIBERO path.

### 2026-06-07 LIBERO Long Retry Control Series

- Added and ran a Long-suite series runner:
  - `scripts/runpod_smolvla_libero_agentic_retry_series.sh`
  - `scripts/build_agentic_retry_series_report.py`
- Remote output:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_series_long_3seed_20260606T220158Z`
- Local archive without videos:
  `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z_no_videos.tar.gz`
- Local extracted report:
  `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z/agentic_retry_series_report.md`
- Protocol:
  - suite: `libero_10`
  - episodes: `10` per task, `100` per baseline seed
  - baseline seeds: `1000`, `1001`, `1002`
  - baseline action horizon: `n_action_steps=15`
  - retry seeds: `1100`, `1101`, `1102`
  - condition `blind_new_seed`: retry with `n_action_steps=15`
  - condition `alternate_steps10`: retry with `n_action_steps=10`
- Per-run results:
  - `alternate_steps10`, seed `1000`: baseline `79.00`, success-once `90.00`,
    delta `+11.00`, recovered `11/21`
  - `alternate_steps10`, seed `1001`: baseline `66.00`, success-once `81.00`,
    delta `+15.00`, recovered `15/34`
  - `alternate_steps10`, seed `1002`: baseline `70.00`, success-once `79.00`,
    delta `+9.00`, recovered `9/30`
  - `blind_new_seed`, seed `1000`: baseline `79.00`, success-once `85.00`,
    delta `+6.00`, recovered `6/21`
  - `blind_new_seed`, seed `1001`: baseline `66.00`, success-once `75.00`,
    delta `+9.00`, recovered `9/34`
  - `blind_new_seed`, seed `1002`: baseline `70.00`, success-once `86.00`,
    delta `+16.00`, recovered `16/30`
- Summary:
  - `alternate_steps10`: baseline `71.67 +/- 5.44`, success-once
    `83.33 +/- 4.78`, delta `+11.67 +/- 2.49`, recovery `42.17 +/- 9.24`
  - `blind_new_seed`: baseline `71.67 +/- 5.44`, success-once
    `82.00 +/- 4.97`, delta `+10.33 +/- 4.19`, recovery `36.13 +/- 12.20`
- Interpretation:
  - retry budget itself is a strong baseline, not a weak strawman.
  - `alternate_steps10` has a slightly higher mean than `blind_new_seed`, but
    the margin is small and seed `1002` favored blind retry.
  - next paper-useful experiment should be a real verifier-guided retry policy
    that chooses retry strategy from failure/task predicates, and it must be
    compared against the blind retry control.

### 2026-06-07 Task-Guided Retry Selection Analysis

- Added offline trace analysis:
  - `scripts/build_agentic_retry_selection_report.py`
  - local report:
    `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z/agentic_retry_selection_report.md`
- This analysis used the completed 3-seed Long blind and alternate retry traces;
  it did not launch extra GPU rollouts.
- Summary:
  - `alternate_steps10`: success-once `83.33 +/- 4.78`, delta
    `+11.67 +/- 2.49`, recovery `42.17 +/- 9.24`
  - `blind_new_seed`: success-once `82.00 +/- 4.97`, delta
    `+10.33 +/- 4.19`, recovery `36.13 +/- 12.20`
  - `portfolio_budget2`: success-once `87.33 +/- 3.68`, delta
    `+15.67 +/- 1.89`, recovery `56.19 +/- 4.87`
  - `task_guided_loso`: success-once `80.00 +/- 3.74`, delta
    `+8.33 +/- 1.70`, recovery `29.33 +/- 0.59`
  - `task_oracle_same_seed`: success-once `85.67 +/- 3.68`, delta
    `+14.00 +/- 2.16`, recovery `49.94 +/- 4.14`
- Interpretation:
  - task identity alone is not enough for a robust verifier-guided selector.
  - a two-retry portfolio is more promising than task-id-only routing: run both
    blind and alternate retry variants for baseline failures and count success
    if either retry succeeds.
  - next GPU experiment should formalize `retry_budget=2` portfolio retry on
    Long first, then expand to all four LIBERO suites if still strong.

### 2026-06-07 Remaining-Suite Portfolio Probe

- Ran remaining LIBERO suites for seed `1000`:
  - remote root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z`
  - local archive without videos:
    `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z_no_videos.tar.gz`
  - local extracted portfolio report:
    `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z/agentic_retry_portfolio_report.md`
- Spatial protocol:
  - baseline `n_action_steps=10`
  - blind retry `n_action_steps=10`, seed `1100`
  - alternate retry `n_action_steps=15`, seed `1100`
  - result: baseline `91.00`, blind `98.00`, alternate `94.00`,
    portfolio `98.00`
- Object protocol:
  - baseline `n_action_steps=15`
  - blind retry `n_action_steps=15`, seed `1100`
  - alternate retry `n_action_steps=10`, seed `1100`
  - result: baseline `94.00`, blind `96.00`, alternate `96.00`,
    portfolio `97.00`
- Goal protocol:
  - baseline `n_action_steps=15`
  - blind retry `n_action_steps=15`, seed `1100`
  - alternate retry `n_action_steps=10`, seed `1100`
  - result: baseline `89.00`, blind `97.00`, alternate `97.00`,
    portfolio `99.00`
- Four-suite seed-1000 portfolio table, using the existing Long seed-1000 row
  from the 3-seed Long series:
  - Spatial: baseline `91.00`, best single `98.00`, portfolio `98.00`,
    delta `+7.00`
  - Object: baseline `94.00`, best single `96.00`, portfolio `97.00`,
    delta `+3.00`
  - Goal: baseline `89.00`, best single `97.00`, portfolio `99.00`,
    delta `+10.00`
  - Long: baseline `79.00`, best single `90.00`, portfolio `92.00`,
    delta `+13.00`
  - Macro average: baseline `88.25`, best single `95.25`, portfolio `96.50`,
    delta `+8.25`
- Interpretation:
  - This is the strongest current four-suite agentic retry result.
  - It is a `retry_budget=2` portfolio result, not a fair policy-only
    comparison against ActionX or base SmolVLA.
  - The next paper-useful step is repeating the four-suite portfolio probe over
    more seeds, or implementing a stronger verifier-guided selector that can
    beat the blind/portfolio controls with less retry budget.

## 2026-06-07 CP24B LIBERO Oracle Overlay Readiness

### Current State

- Status: completed, fetched locally, and probe Pod stopped.
- Goal: verify real LIBERO/MuJoCo oracle affordance overlay generation on diverse actual simulation frames without interfering with baseline evaluation.
- Probe Pod: `v5yckik4qv9y85`.
- GPU: L4-class low-cost probe lane at `$0.39/hr` when running.
- Network volume: `tchm4gxfvd`.
- Separate clone path:
  `/workspace/physical-ai/physical_ai_agent_affordance_probe`.
- Final local artifact root:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/`.

### Result

- Broad pool: 19 real LIBERO/MuJoCo samples across `libero_spatial`, `libero_object`, and `libero_goal`.
- Curated evidence set: 13 visually distinct samples after excluding near-duplicates and a weak edge-clipped case.
- Targeting method: raw MuJoCo segmentation for semantic object geoms plus a cabinet drawer/handle resolver for open-drawer tasks.
- Figure orientation: native `agentview` frames looked vertically inverted relative to LIBERO paper figures, so the paper-facing figure pack applies a visualization-only vertical flip and transforms overlay points as `y_fixed = H - 1 - y_native`.
- Policy-input note: native simulator coordinates are preserved in the manifest; the fixed orientation is for report/figure artifacts.

### Evidence

- Corrected contact sheet:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/libero_mujoco_oracle_curated_diverse_contact_sheet.jpg`
- Corrected manifest:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/libero_mujoco_oracle_curated_manifest.json`
- Corrected report:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/verification_report_final.md`

### Claim Boundary

This is CP24B policy-input readiness evidence. It proves that real LIBERO/MuJoCo frames can be annotated with simulator-oracle affordance points in a visually inspected, paper-facing orientation. It does not prove SmolVLA success-rate improvement. The next checkpoint should be CP24C, an overlay ablation comparing `smolvla_rgb_only` against `smolvla_rgb_oracle_point`, optionally with `agentic_retry`.

## 2026-06-07 LIBERO Retry Cost Normalization and Claim Boundary

### Current State

- Status: in progress.
- Active RunPod evaluation root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z`
- Current run purpose: finish Spatial/Object/Goal seeds `1001` and `1002` so
  the Long 3-seed retry series can be joined into a 4-suite, 3-seed table.
- Direction change: success-only retry tables are not enough for the paper
  claim. The current retry wrapper uses the same frozen `lerobot/smolvla_libero`
  policy and should not be described as a better one-shot model.

### Completed Partial Results

- Spatial seeds `1001,1002`:
  - baseline mean `88.00`
  - `blind_new_seed` success-once mean `96.50`, delta `+8.50`
  - `alternate_steps15` success-once mean `94.50`, delta `+6.50`
- Object seeds `1001,1002`:
  - baseline mean `92.50`
  - `alternate_steps10` success-once mean `97.00`, delta `+4.50`
  - `blind_new_seed` success-once mean `95.00`, delta `+2.50`
- Goal seed `1001`:
  - baseline `87.00`
  - `alternate_steps10` success-once `97.00`, delta `+10.00`
  - `blind_new_seed` success-once `94.00`, delta `+7.00`

### Cost-Normalized Metric Update

- Added cost fields to `physical_ai_agent.agent_core.libero_agentic_retry`:
  - total attempts
  - extra environment resets
  - total eval seconds
  - success-once per attempt
  - recovered episodes per retry attempt
  - success-once per eval minute
- Added cost fields to the four-suite portfolio report builder:
  `scripts/build_agentic_retry_four_suite_portfolio_report.py`.
- Current LeRobot `eval_info.json` records `overall.eval_s` but does not record
  per-episode action-step counts, so action-step-normalized metrics require an
  instrumented rollout path before being used as paper evidence.

### Claim Boundary

- Current retry evidence supports:
  "same frozen SmolVLA plus explicit reset/retry budget improves realized
  benchmark success."
- Current retry evidence does not support:
  "SmolVLA model weights improved" or "one-shot policy success improved."
- A stronger agentic physical AI claim needs either:
  - better performance than blind retry under the same retry budget,
  - better success per attempt/eval minute than blind retry,
  - or in-episode verifier intervention before reset.

### Next Required Experiment

- Build or reuse an instrumented LIBERO rollout loop that logs per-step action
  counts and exposes an online verifier hook.
- Test an in-episode verifier that detects stagnation, failed grasp progress,
  or timeout risk and intervenes before environment reset.
- Compare against policy-only, blind retry, and horizon-switch retry with the
  same success, attempt, reset, eval-time, and action-step metrics.

### In-Episode Intervention Readiness Audit

- Added executable readiness audit:
  `scripts/build_libero_in_episode_intervention_readiness_report.py`.
- Report:
  `docs/research/libero_in_episode_intervention_readiness_report_2026_06_07.md`.
- Summary:
  - pass `7`
  - warn `1`
  - fail `2`
  - missing `0`
- Findings:
  - LeRobot `rollout()` already returns stacked action, success, and done
    tensors, so action-step-normalized metrics are feasible in a custom or
    patched rollout path.
  - The online verifier can be inserted immediately before or after
    `env.step(action_numpy)`.
  - `LiberoEnv.step()` exposes `info["is_success"]`, but auto-resets after
    terminal states; intervention must trigger before `terminated=True`.
  - Default `eval_info.json` does not include per-episode action-step counts.
- Next gate:
  implement a custom LIBERO rollout wrapper that logs `action_step_count`,
  `verifier_triggered`, `trigger_step`, `intervention_type`, and final benchmark
  success before launching more paper-facing GPU runs.

### In-Episode Instrumentation Smoke

- Added dependency-light core:
  `src/physical_ai_agent/agent_core/libero_in_episode.py`.
- Added executable smoke:
  `scripts/run_libero_in_episode_instrumented_smoke.py`.
- Local smoke command:
  `PYTHONPATH=src:. python3 scripts/run_libero_in_episode_instrumented_smoke.py --output-dir _workspace/libero_in_episode_smoke_20260607`
- Smoke evidence:
  - report:
    `_workspace/libero_in_episode_smoke_20260607/in_episode_report.md`
  - metrics:
    `_workspace/libero_in_episode_smoke_20260607/in_episode_metrics.json`
  - trace:
    `_workspace/libero_in_episode_smoke_20260607/in_episode_trace.jsonl`
- Smoke result:
  - success `true`
  - action steps `6`
  - verifier triggers `1`
  - interventions `1`
  - environment resets `1`
  - success/action-step `0.166667`
- RunPod Linux smoke:
  - executed from a temporary `git archive FETCH_HEAD` checkout to avoid
    touching the dirty remote working tree
  - output root:
    `/workspace/physical-ai/physical_ai_agent/_workspace/libero_in_episode_smoke_runpod_20260607`
  - success `true`, action steps `6`, verifier triggers `1`, interventions
    `1`, environment resets `1`, success/action-step `0.166667`
- Interpretation:
  this is not a LIBERO benchmark result yet. It proves the in-episode logging
  contract: verifier trigger and intervention occur before terminal reset, and
  the result reports action-step-normalized cost.

### Real LIBERO/SmolVLA In-Episode Hook Smoke

- Added real-path wrapper:
  `scripts/run_libero_in_episode_smolvla_instrumented.py`.
- Method:
  monkeypatch LeRobot `rollout()` at runtime, keep the standard
  `lerobot-eval` environment/policy setup, and append per-step trace rows.
- RunPod command shape:
  `python /tmp/run_libero_in_episode_smolvla_instrumented.py --trace-path ... --intervention-step 3 --intervention-scale 1.0 --output_dir=... --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=libero_goal --env.task_ids=[0] --eval.n_episodes=1 ...`
- Remote root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/libero_in_episode_smolvla_smoke_20260607T034632`
- Local archive:
  `_workspace/runpod_results/in_episode_20260607/libero_in_episode_smolvla_smoke_20260607T034632_no_videos.tar.gz`
- Result:
  - benchmark success `true`
  - action steps `131`
  - verifier triggers `1`
  - interventions `1`
  - environment resets `1`
  - eval seconds `7.3373`
  - success/action-step `0.007634`
- Claim boundary:
  intervention scale was `1.0`, so this is a real LIBERO hook/logging smoke,
  not an intervention-improvement result. The next comparison should run
  no-op hook vs non-trivial intervention under identical task/seed/action
  budget.

### Real LIBERO Same-Seed In-Episode Ablation

- Added ablation report builder:
  `scripts/build_libero_in_episode_ablation_report.py`.
- Added report:
  `docs/research/libero_in_episode_smolvla_ablation_2026_06_07.md`.
- Conditions:
  - no-op hook: scale `1.0`, `libero_goal`, task `[0]`, seed `1200`
  - non-trivial intervention: scale `0.5`, same task and seed
- Result:
  - no-op hook: success `true`, action steps `131`, eval seconds `7.3373`,
    success/action-step `0.007634`
  - scale `0.5`: success `true`, action steps `132`, eval seconds `7.3378`,
    success/action-step `0.007576`
- Interpretation:
  the scale `0.5` intervention preserved success but did not improve action
  cost or eval-time cost in this one-episode smoke. This is useful negative
  evidence: the project now has a same-task, same-seed, cost-normalized
  comparison protocol, but not yet a positive in-episode intervention result.

### Real LIBERO Same-Seed Spike Intervention Matrix

- Extended real-path wrapper:
  `scripts/run_libero_in_episode_smolvla_instrumented.py`.
- Added verifier/intervention options:
  - trigger modes: `fixed_step`, `action_norm_threshold`,
    `fixed_or_action_norm`
  - intervention modes: `none`, `scale`, `clamp`, `smooth`
  - per-step trace records pre/post intervention action norms
- RunPod task:
  - policy: `lerobot/smolvla_libero`
  - suite: `libero_goal`
  - task ids: `[0]`
  - seed: `1200`
  - episodes: `1`
  - `n_action_steps`: `15`
- Local archive:
  `_workspace/runpod_results/in_episode_20260607/libero_in_episode_matrix_20260607T035853_no_videos.tar.gz`
- Report:
  `docs/research/libero_in_episode_smolvla_spike_intervention_matrix_2026_06_07.md`
- Result:
  - `hook_none`: success `true`, action steps `131`, eval seconds `7.4206`
  - `spike_clamp145_to120`: success `true`, action steps `135`, delta `+4`
  - `spike_smooth145_a070`: success `true`, action steps `132`, delta `+1`
  - `spike_scale145_085`: success `true`, action steps `134`, delta `+3`
- Interpretation:
  conditional action-norm spike interventions preserved success but did not
  improve action-step efficiency on this same-seed one-task matrix. This is a
  stronger negative result than the earlier fixed-step scale smoke: the
  in-episode protocol is real and cost-normalized, but the current simple
  action-space interventions are not yet a positive agentic improvement.

### LIBERO Goal Task Scout and Task 3 Intervention Follow-Up

- Ran a 5-task baseline scout with no in-episode intervention:
  - suite: `libero_goal`
  - task ids: `0, 1, 2, 3, 4`
  - seed: `1200`
  - episodes per task: `1`
- Scout result:
  - task `0`: success `true`, action steps `131`
  - task `1`: success `true`, action steps `85`
  - task `2`: success `true`, action steps `87`
  - task `3`: success `true`, action steps `177`
  - task `4`: success `true`, action steps `90`
- Task `3` was selected for follow-up because it was the longest successful
  rollout in the scout.
- Task `3` baseline action-norm spikes were concentrated at steps `148-156`
  with threshold `>=1.45`, suggesting a late execution/approach phase rather
  than early instability.
- Task `3` intervention follow-up:
  - `task3_hook_none`: success `true`, action steps `177`, eval seconds
    `8.5576`
  - `task3_spike_scale145_110`: success `true`, action steps `178`, delta
    `+1`
  - `task3_spike_scale145_120`: success `true`, action steps `177`, delta
    `+0`
  - `task3_spike_clamp145_to135`: success `true`, action steps `177`, delta
    `+0`
  - `task3_spike_smooth145_a050`: success `true`, action steps `177`, delta
    `+0`
- Reports:
  - `docs/research/libero_goal_task3_in_episode_intervention_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task3_in_episode_intervention_matrix_summary_2026_06_07.json`
- Interpretation:
  task `3` confirms the same pattern: simple action-space interventions
  preserve success but do not yet reduce action-step cost. The next positive
  candidate should move beyond scalar action post-processing toward a semantic
  verifier/intervention, such as detecting no-progress windows and forcing a
  policy reset/action-queue refresh inside the episode, or selecting known weak
  seeds/tasks where baseline fails before comparing intervention variants.

### LIBERO Goal Task 3 Policy Reset Matrix

- Added `policy_reset` intervention mode to the real-path wrapper.
- Mechanism:
  when the verifier triggers, call `policy.reset()` inside the same episode
  and reselect an action from the same observation before `env.step()`.
- This is closer to the intended agentic physical AI loop than scalar action
  post-processing because it refreshes the policy action queue/state without
  resetting the environment.
- Task:
  - suite: `libero_goal`
  - task id: `3`
  - seed: `1200`
  - baseline action steps: `177`
- Result:
  - `task3_reset_step80`: success `true`, action steps `178`, delta `+1`
  - `task3_reset_step120`: success `true`, action steps `177`, delta `+0`
  - `task3_reset_step145`: success `true`, action steps `177`, delta `+0`
  - `task3_reset_norm145`: success `true`, action steps `177`, delta `+0`
- Report:
  - `docs/research/libero_goal_task3_policy_reset_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task3_policy_reset_matrix_summary_2026_06_07.json`
- Interpretation:
  in-episode policy reset/queue refresh preserved benchmark success but did not
  reduce action-step cost on the selected long successful rollout. This is a
  verified negative/neutral result, not a positive improvement. The next
  experiment should first identify weak seeds/tasks where baseline fails or
  times out, then compare policy reset or semantic retry against that fixed
  subset.

### LIBERO Goal Task 5-9 Scout and Task 6 Failure Intervention Matrix

- Ran a second no-intervention scout over `libero_goal` task ids `5-9`.
- Scout result:
  - task `5`: success `true`, action steps `130`
  - task `6`: success `false`, action steps `300`
  - task `7`: success `true`, action steps `86`
  - task `8`: success `true`, action steps `78`
  - task `9`: success `true`, action steps `142`
- Task `6` is the first identified same-seed baseline failure case in the
  in-episode intervention search.
- Task `6` baseline action-norm distribution:
  - max norm: `1.4562`
  - threshold `>=1.35`: `48` trigger candidates
  - threshold `>=1.45`: `1` trigger candidate
- Task `6` intervention matrix:
  - `task6_hook_none`: success `false`, action steps `300`
  - `task6_reset_step30`: success `false`, action steps `300`
  - `task6_reset_step60`: success `false`, action steps `300`
  - `task6_reset_step120`: success `false`, action steps `300`
  - `task6_reset_norm135`: success `false`, action steps `300`
  - `task6_smooth_norm135_a050`: success `false`, action steps `300`
- Report:
  - `docs/research/libero_goal_task6_failure_intervention_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task6_failure_intervention_matrix_summary_2026_06_07.json`
- Interpretation:
  this is a verified negative result on a true baseline failure case. The
  current in-episode interventions can preserve success on successful tasks,
  but they do not recover this failure. To get a positive paper-facing result,
  the next intervention likely needs semantic state feedback from LIBERO/MuJoCo
  info or rendered observations, not only action norm or policy queue reset.

### LIBERO Goal Task 6 Semantic State Probe and Late Placement Intervention

- Added semantic probe:
  `scripts/probe_libero_env_semantic_state.py`.
- Probe result:
  LeRobot/LIBERO exposes raw object and robot state through the underlying
  `LiberoEnv._env.env._get_observations()` path inside rollout.
- Available task `6` semantic keys:
  - `cream_cheese_1_pos`
  - `akita_black_bowl_1_pos`
  - `robot0_eef_pos`
  - `robot0_gripper_qpos`
- Extended real in-episode wrapper with:
  - `semantic_no_progress` trigger
  - semantic trace fields: target position, receptacle position, EEF position,
    target-to-receptacle distance, EEF-to-target distance
  - `semantic_reach_target` intervention
  - `semantic_push_receptacle` intervention
- Early semantic intervention result:
  - `task6_semantic_probe_none`: success `false`, action steps `300`
  - `task6_semantic_reach_g2`: success `false`, action steps `300`
  - `task6_semantic_reach_g4`: success `false`, action steps `300`
  - `task6_semantic_reach_g4_open`: success `false`, action steps `300`
- Late-stage semantic intervention result:
  - `task6_late_probe_none`: success `false`, action steps `300`
  - `task6_late_push_g5_close`: success `false`, action steps `300`
  - `task6_late_push_g10_close`: success `false`, action steps `300`
  - `task6_late_push_g5_open`: success `false`, action steps `300`
  - `task6_late_reach_g2_open`: success `false`, action steps `300`
- Semantic diagnostic:
  - baseline minimum target-to-bowl distance: `0.062636` at step `229`
  - baseline final target-to-bowl distance: `0.062821`
  - `late_reach_g2_open` final EEF-to-target improved from `0.080481` to
    `0.047316`, but target-to-bowl stayed `0.062821`
- Report:
  - `docs/research/libero_goal_task6_semantic_intervention_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task6_semantic_intervention_matrix_summary_2026_06_07.json`
- Interpretation:
  small semantic interventions reveal improvement potential at the diagnostic
  level: the wrapper can identify a late placement stall and move the gripper
  closer to the object. However, it has not yet produced benchmark success or
  action-step efficiency improvement. The next useful intervention should be
  contact-aware placement/release rather than generic reach or push vectors.

### LIBERO Goal Task 6 Reach-Then-Push Phase Intervention

- Added `semantic_reach_then_push` intervention mode.
- Mechanism:
  - trigger on late semantic no-progress
  - if EEF-to-target is above a contact threshold, move EEF toward the target
  - once EEF-to-target is below the threshold, push the target toward the bowl
- RunPod matrix:
  - suite: `libero_goal`
  - task id: `6`
  - seed: `1200`
  - baseline: `task6_late_probe_none`
- Result:
  - all phase-intervention conditions still failed benchmark success at `300`
    action steps
  - `task6_phase_push_c06_g2p5_close`: final EEF-to-target improved from
    `0.080481` to `0.059052`, but object placement did not improve
  - `task6_phase_push_c08_g2p5_close`: first tiny object-placement improvement,
    minimum target-to-bowl distance improved from `0.062636` to `0.062615`
  - open-gripper variants worsened final target-to-bowl distance
- Report:
  - `docs/research/libero_goal_task6_phase_intervention_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task6_phase_intervention_matrix_summary_2026_06_07.json`
- Interpretation:
  this is still not a positive task-level agentic result. It is the first
  diagnostic evidence that a contact-seeking phase intervention can nudge the
  object-placement metric in the desired direction. The next intervention
  should use raw MuJoCo contact state or a short placement macro rather than a
  single vector push.

### LIBERO Goal Task 6 Near-Receptacle Placement Macro

- Tried to probe raw MuJoCo contacts, but the exposed `sim` candidates in the
  current LeRobot/LIBERO wrapper did not surface named geom/contact pairs
  through the quick probe.
- Added `semantic_near_receptacle` trigger:
  trigger once target-to-receptacle distance is below a threshold after a
  minimum step.
- Added `semantic_place_receptacle` intervention:
  - if EEF is far from target, reach target
  - if EEF is within contact threshold, push target toward receptacle in XY
    and apply a configurable downward Z command
  - optionally close/open gripper via action dimension 6
- RunPod task:
  - suite: `libero_goal`
  - task id: `6`
  - seed: `1200`
  - baseline: `task6_late_probe_none`
- Benchmark result:
  all tested placement-macro conditions still failed benchmark success at
  `300` action steps.
- Subgoal metric result:
  - baseline min target-to-bowl distance: `0.062636`
  - `task6_place_c06_p5_zm02_close`: `0.058806`
  - `task6_place_c08_p8_zm02_close`: `0.060582`
  - `task6_place_c08_p5_zm05_close`: `0.055952`
- Report:
  - `docs/research/libero_goal_task6_place_macro_matrix_2026_06_07.md`
  - `docs/research/libero_goal_task6_place_macro_matrix_summary_2026_06_07.json`
- Interpretation:
  this is still not a positive benchmark success result, but it is the
  strongest in-episode evidence so far: a semantic subgoal intervention
  improves an intermediate physical metric by moving the target closer to the
  bowl. The next search should tune this macro or test nearby weak seeds/tasks.

### LIBERO Goal Task 6 Placement Macro Positive Tuning

- Tuned the best near-receptacle placement macro around:
  - trigger step: `220`
  - contact threshold: `0.08`
  - push gain: `5.0`
  - downward Z command: `-0.7` and `-1.0`
  - gripper command: `1.0`
- Baseline:
  - `task6_late_probe_none`
  - benchmark success: `false`
  - action steps: `300`
  - eval seconds: `10.0566`
  - environment resets: `1`
- Positive in-episode intervention results:
  - `task6_tune_s220_zm07`
    - benchmark success: `true`
    - action steps: `224`
    - interventions: `4`
    - eval seconds: `9.3361`
    - min target-to-bowl: `0.058218`
  - `task6_tune_s220_zm10`
    - benchmark success: `true`
    - action steps: `224`
    - interventions: `4`
    - eval seconds: `9.3117`
    - min target-to-bowl: `0.053997`
- Negative controls:
  - delaying trigger to step `224` or `228` failed despite improving placement
    metrics
  - weaker push gain `3.0` failed despite transient placement improvement
- Report:
  - `docs/research/libero_goal_task6_place_tuning_success_2026_06_07.md`
  - `docs/research/libero_goal_task6_place_tuning_success_summary_2026_06_07.json`
- Interpretation:
  this is the first positive benchmark result for the in-episode agentic layer.
  It is not a reset retry: the wrapper detects a semantic near-receptacle
  condition inside the rollout and switches to a placement subgoal. The result
  is still a single task/seed smoke, but it gives a real seed for paper-scale
  follow-up.

### LIBERO Goal Task 6 Placement Macro Multi-Seed Check

- Expanded the positive placement macro from one task/seed to seeds `1200`,
  `1201`, and `1202` on the same `libero_goal` task id `6`.
- Baseline condition:
  - SmolVLA policy-only rollout
  - no in-episode intervention
  - environment resets: `1`
- Wrapper condition:
  - same `lerobot/smolvla_libero` weights
  - `semantic_near_receptacle` trigger
  - `semantic_place_receptacle` macro
  - environment resets: `1`
- Paired result:
  - seed `1200`: baseline failed at `300` steps; wrapper succeeded at `224`
    steps with `4` interventions
  - seed `1201`: baseline failed; wrapper also failed because the object never
    entered the near-receptacle trigger threshold
  - seed `1202`: baseline failed; wrapper also failed because the object never
    entered the near-receptacle trigger threshold
- Aggregate paired success:
  - baseline: `0/3`
  - wrapper: `1/3`
- Report:
  - `docs/research/libero_goal_task6_place_multiseed_report_2026_06_07.md`
  - `docs/research/libero_goal_task6_place_multiseed_summary_2026_06_07.json`
- Interpretation:
  the small in-episode intervention did produce a benchmark-success conversion
  on one seed without adding an environment reset, so there is improvement
  signal. The blocker is trigger coverage: the current wrapper only helps when
  the base policy has already moved the target close enough to the receptacle.
  The next improvement should broaden trigger/reach coverage inside the same
  episode rather than using reset retry.

### LIBERO Goal Task 6 No-Progress Trigger Coverage Check

- Tested a no-code trigger-coverage variant on the same `libero_goal` task id
  `6` and seeds `1200`, `1201`, `1202`.
- Variant:
  - trigger: `semantic_no_progress`
  - intervention: `semantic_place_receptacle`
  - min step: `220`
  - progress window: `20`
  - progress threshold: `0.002`
  - environment resets: `1`
- Aggregate benchmark result:
  - baseline: `0/3`
  - near-receptacle placement macro: `1/3`
  - no-progress placement macro: `0/3`
- Intervention load:
  - near-receptacle placement macro: mean `1.33` interventions across seeds
  - no-progress placement macro: mean `73.33` interventions across seeds
- Subgoal metric:
  - seed `1201` min target-to-bowl improved from `0.146379` to `0.112685`
  - seed `1202` min target-to-bowl improved from `0.142073` to `0.061886`
  - seed `1200` regressed from the near-receptacle success case to failure
- Report:
  - `docs/research/libero_goal_task6_trigger_coverage_report_2026_06_07.md`
  - `docs/research/libero_goal_task6_trigger_coverage_summary_2026_06_07.json`
- Interpretation:
  no-progress detection fixes the trigger coverage problem only partially. It
  creates in-episode interventions on previously untouched seeds and improves
  an intermediate physical metric, but it over-intervenes and hurts benchmark
  success. The next candidate should be a phased/gated trigger: use
  no-progress to recover approach/reach, then switch to placement only under
  contact or near-receptacle conditions.

### ManiSkill3 Shared LeRobot Runner Smoke

- Checked the previous LIBERO SmolVLA path and confirmed it uses the full
  LeRobot processor contract:
  `env_preprocessor -> preprocessor -> policy.select_action -> postprocessor -> env_postprocessor`.
- Added a shared custom-benchmark policy path:
  - `src/physical_ai_agent/policies/lerobot_policy_runner.py`
  - `scripts/run_maniskill3_smolvla_eval.py`
  - `tests/test_lerobot_policy_runner.py`
- Purpose:
  - prevent one-off custom scripts from skipping policy processors
  - preserve `UnnormalizerProcessorStep` for SmolVLA actions
  - make future ManiSkill3 and other custom benchmark evals use one runner
- RunPod checkpoint inspected:
  - `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count10_100step/checkpoints/last/pretrained_model`
  - preprocessor steps:
    `RenameObservationsProcessorStep`, `AddBatchDimensionProcessorStep`,
    `NewLineTaskProcessorStep`, `TokenizerProcessorStep`,
    `DeviceProcessorStep`, `NormalizerProcessorStep`
  - postprocessor steps:
    `UnnormalizerProcessorStep`, `DeviceProcessorStep`
- Debug fixes made:
  - aligned mismatched processor stats to declared feature shape for the small
    probe checkpoint
  - converted ManiSkill RGB observations to batched CHW LeRobot image format
  - moved preprocessed tensors to the policy device before action selection
- Smoke result:
  - env: `PushCube-v1`
  - episodes: `3`
  - success: `0/3`
  - output:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count10_100step_shared_runner_eval_3ep/metrics.json`
  - metrics confirm:
    `preprocessor_applied=true`, `postprocessor_applied=true`,
    `postprocessor_steps=["UnnormalizerProcessorStep","DeviceProcessorStep"]`
- Interpretation:
  - the earlier raw custom `select_action()` eval snippets are invalid for
    comparison because they skipped LeRobot processors
  - this shared-runner smoke is a valid pipeline proof, but the `0/3` result is
    not a paper-comparable STARE number because it uses only a count10,
    100-step PushCube probe checkpoint rather than paper-like task-specific
    SFT with `1000` trajectory samples

### ManiSkill3 PushCube SFT Scale-Up

- Goal:
  calibrate the policy-only SmolVLA fine-tuning baseline against the STARE
  ManiSkill3 Table 2 `PushCube-v1` reference before making any agentic-wrapper
  claim.
- Reference:
  STARE Table 2 reports `SmolVLA (fine-tuning)` on `PushCube-v1` at `86.3%`
  success and states SmolVLA uses `1000` trajectory samples for SFT per task.
- Data:
  official `PushCube-v1` RGB replay was generated without `--use-env-states`,
  saving `1000/1000=100.00%` demos.
- Converted dataset:
  `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_no_env_states`
  with `1000` episodes and `68978` frames.
- Training:
  `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_feature_override_9000step`
  completed `9000` steps, about `1.04` epochs, with final logged loss `0.029`.
- Evaluation:
  `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_feature_override_9000step_qpos_horizon100_shared_runner_eval_50ep`
  scored `29/50`, `58.0%`.
- Delta:
  `58.0%` vs STARE `86.3%`, so current gap is `-28.3pp`.
- Earlier scale probe:
  count100/1000-step SFT scored `20/50`, `40.0%`; a shorter 20ep run scored
  `11/20`, `55.0%`.
- Correctness fixes:
  - `scripts/run_maniskill3_smolvla_eval.py` now passes
    `max_episode_steps=args.max_steps` into `gym.make(...)`; without this,
    `PushCube-v1` truncated at the default 50-step horizon.
  - The eval observation builder now prefers `obs["agent"]["qpos"]` as
    `observation.state`, matching the LeRobot conversion path.
  - The eval path uses `LeRobotPolicyRunner`, preserving the
    `UnnormalizerProcessorStep` postprocessor.
- Interpretation:
  this is meaningful as policy-only baseline calibration, not as the project
  contribution. The next wrapper comparison must use this or a better-matched
  checkpoint as the fixed baseline. The remaining work is to narrow or explain
  the `-28.3pp` gap before expanding to `StackCube-v1`, `PullCube-v1`, and
  `LiftPegUpright-v1`.

### ManiSkill3 PushCube Negative Parity Ablations

- Action chunk ablation on the count1000/9000-step checkpoint:
  - default checkpoint config: `n_action_steps=50`
  - temporary config copy with `n_action_steps=1`:
    `1/20`, `5.0%`
  - temporary config copy with `n_action_steps=10`:
    `8/20`, `40.0%`
  - temporary config copy with `n_action_steps=15`:
    `8/20`, `40.0%`
  - interpretation:
    unlike the LIBERO parity sweep, shorter action chunks do not help
    `PushCube-v1`; keep the default `50` for this checkpoint.
- Longer SFT ablation:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_feature_override_27000step`
  - config source:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_feature_override_9000step/checkpoints/last/pretrained_model/train_config.json`
  - changed fields:
    `steps=27000`, `save_freq=27000`, `eval_freq=50000`, output dir and
    repo id only
  - training completed:
    `27000` steps, about `3.13` epochs, final logged loss `0.017`
  - 50-episode eval result:
    `26/50`, `52.0%`
  - artifact:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_feature_override_27000step_qpos_horizon100_shared_runner_eval_50ep/metrics.json`
  - interpretation:
    longer training reduced train loss but worsened eval relative to the
    9000-step result (`58.0%`), so simple under-training is not the main
    explanation for the STARE `86.3%` gap.
- Operational note:
  - the RunPod clone used for these commands reported git revision `808aa6a`
    during the ablation session, while the local branch later contained newer
    documentation commits. Before the next paper-facing run, run `git pull` on
    RunPod and record the exact git revision in the metrics/report.
- Next likely parity checks:
  - exact STARE eval seed/distribution and number of episodes
  - whether STARE used the same official motion-planning demo source or a
    different successful trajectory source
  - observation preprocessing, especially camera resolution and any image
    resize/padding semantics between replay conversion, training, and eval
  - whether the STARE row uses a different ManiSkill task/control-mode variant

### ManiSkill3 PushCube Horizon-30 / Idle-Filtered Parity Round

- Status: completed on RunPod; Pod left running for follow-up parity debugging.
- Reference protocol update:
  - STARE Appendix B.5 reports ManiSkill3 evaluation with horizon `30`, `300`
    episodes, and `5` random seeds.
  - STARE Appendix C.2 reports filtering idle actions before training.
- Raw checkpoint paper-horizon check:
  - checkpoint:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_feature_override_9000step/checkpoints/last/pretrained_model`
  - eval:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_feature_override_9000step_qpos_horizon30_shared_runner_eval_50ep`
  - result:
    `0/50`, `0.0%` at horizon `30`.
  - interpretation:
    the earlier horizon-100 `58.0%` number is not paper-comparable; most raw
    SFT successes happen too late for STARE's horizon.
- Qpos idle-filtered dataset:
  - source:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_no_env_states`
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_qpos_filter005`
  - filter:
    keep frames with qpos delta at least `0.05`.
  - size:
    `1000` episodes, `21802` frames, mean length `21.8`, p50 `21`, p90 `24`,
    max `31`.
  - loader smoke:
    `LeRobotDataset` loaded `1000` episodes and `21802` frames; sample state
    shape `[9]`, action shape `[8]`.
- Qpos idle-filtered SFT:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_qpos_filter005_9000step`
  - source config:
    raw count1000 9000-step train config.
  - training completed:
    `9000` steps, about `3.30` epochs, final logged loss `0.023`.
- Evaluation:
  - 50ep horizon-30 eval:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_50ep`
    scored `34/50`, `68.0%`.
  - 300ep horizon-30 eval, seed `1000`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_300ep_seed1000`
    scored `182/300`, `60.7%`.
  - 300ep horizon-30 eval, seed `1001`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_300ep_seed1001`
    scored `183/300`, `61.0%`.
  - 300ep horizon-30 eval, seed `1002`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_300ep_seed1002`
    scored `189/300`, `63.0%`.
  - 300ep horizon-30 eval, seed `1003`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_300ep_seed1003`
    scored `193/300`, `64.3%`.
  - 300ep horizon-30 eval, seed `1004`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter005_9000step_qpos_horizon30_shared_runner_eval_300ep_seed1004`
    scored `186/300`, `62.0%`.
- Delta:
  the five-seed aggregate is `933/1500`, `62.2%` vs STARE PushCube `86.3%`, so
  the current paper-horizon gap is `-24.1pp`.
- Interpretation:
  fine-tuning here is baseline calibration, not the agentic contribution.
  Action/idle filtering is a real protocol lever: it recovers horizon-30
  success from `0.0%` to a stable `61-64%` band across five 300-episode seeds,
  matching STARE's reported seed count. The remaining gap is still too large
  for a parity claim, so continue by matching STARE's exact filtering/source
  trajectory protocol before expanding to the other three ManiSkill3 tasks.

### ManiSkill3 PushCube Qpos Filter 0.04 Negative Ablation

- Status: completed on RunPod.
- Purpose:
  test whether qpos threshold `0.05` was too aggressive by trying a looser
  idle-frame filter.
- Script:
  `scripts/filter_lerobot_idle_frames.py` now provides a reusable repo-local
  idle-frame filtering command for LeRobot datasets. It keeps a frame when the
  state delta from the last kept frame exceeds the threshold, rewrites parquet
  metadata, and reuses source videos through symlinks by default.
- Dataset:
  - source:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_no_env_states`
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_qpos_filter004`
  - threshold:
    qpos delta `0.04`
  - size:
    `1000` episodes, `25104` frames, mean length `25.1`, p50 `24`, p90 `30`,
    max `37`.
  - loader smoke:
    `LeRobotDataset` loaded `1000` episodes and `25104` frames; sample state
    shape `[9]`, action shape `[8]`.
- Training:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_qpos_filter004_9000step`
  - steps:
    `9000`
  - final logged loss:
    `0.022`
  - approximate epochs:
    `2.87`
- Evaluation:
  - artifact:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter004_9000step_qpos_horizon30_shared_runner_eval_50ep`
  - result:
    `20/50`, `40.0%`
  - delta vs STARE PushCube `86.3%`:
    `-46.3pp`
- Interpretation:
  looser filtering is worse than threshold `0.05` (`68.0%` over the same 50ep
  horizon-30 check). The next likely filter ablation is a stronger threshold,
  or a closer implementation of STARE's exact idle-action filtering rule.

### ManiSkill3 PushCube Qpos Filter 0.06 Ablation

- Status: completed on RunPod.
- Purpose:
  test whether a stronger qpos idle-frame filter improves horizon-30 success
  relative to the current best threshold `0.05`.
- Dataset:
  - source:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_no_env_states`
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_rgb_count1000_rgbmode_128_qpos_filter006`
  - threshold:
    qpos delta `0.06`
  - size:
    `1000` episodes, `18850` frames, mean length `18.85`, p50 `19`, p90 `21`,
    max `25`.
  - loader smoke:
    `LeRobotDataset` loaded `1000` episodes and `18850` frames; sample state
    shape `[9]`, action shape `[8]`.
- Training:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pushcube_count1000_qpos_filter006_9000step`
  - steps:
    `9000`
  - final logged loss:
    `0.022`
  - approximate epochs:
    `3.82`
- Evaluation:
  - artifact:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/pushcube_count1000_qpos_filter006_9000step_qpos_horizon30_shared_runner_eval_50ep`
  - result:
    `31/50`, `62.0%`
  - delta vs STARE PushCube `86.3%`:
    `-24.3pp`
- Interpretation:
  stronger filtering is better than the looser `0.04` threshold but still below
  the threshold `0.05` 50ep result (`68.0%`). The current qpos-filter best
  remains threshold `0.05`, with the stronger five-seed 300ep estimate
  `62.2%`. Further improvement likely requires matching STARE's exact
  action-filter/source-demo protocol rather than sweeping qpos thresholds
  alone.

### ManiSkill3 StackCube Qpos Filter 0.05 Two-Camera SFT

- Status: completed on RunPod; results fetched locally; Pod left running for
  follow-up parity work.
- Purpose:
  add a second STARE-style non-LIBERO SmolVLA fine-tuning row after PushCube.
- Reference:
  STARE Table 2 reports `StackCube-v1` SmolVLA fine-tuning success `12.7%`.
  Appendix B.5 reports horizon `30`, `300` episodes, and `5` random seeds.
- Repo/runtime note:
  the eval script was updated so StackCube observations populate both
  `base_camera/camera1` and `hand_camera/camera2`. This avoids evaluating a
  two-camera checkpoint with a zero-filled wrist/hand camera stream.
- RGB replay:
  - source:
    `/root/physical-ai/tmp_maniskill_official_demos/StackCube-v1/motionplanning/trajectory.h5`
  - output:
    `/root/physical-ai/tmp_maniskill_official_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_pos.physx_cpu.h5`
  - result:
    `999/1000` demos saved successfully; one replay episode was skipped, so
    this is close to but not exact `1000`-trajectory parity.
- LeRobot conversion:
  - output:
    `/root/physical-ai/tmp_lerobot_stare_stackcube_rgb_count999_rgbmode_128_no_env_states`
  - result:
    `999` episodes, `107420` frames.
  - features:
    action shape `[8]`, state shape `[9]`, cameras `base_camera` and
    `hand_camera`.
- Qpos idle-filtered dataset:
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_stackcube_rgb_count999_rgbmode_128_qpos_filter005`
  - threshold:
    qpos delta `0.05`.
  - size:
    `999` episodes, `26728` frames, mean length `26.8`, p50 `26`, p90 `34`,
    max `76`.
- Training:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_stackcube_count999_qpos_filter005_9000step_2cam`
  - checkpoint:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_stackcube_count999_qpos_filter005_9000step_2cam/checkpoints/009000/pretrained_model`
  - base:
    `lerobot/smolvla_base`
  - steps:
    `9000`
  - rename map:
    `base_camera -> camera1`, `hand_camera -> camera2`
  - training completion:
    `9000/9000`, final logged loss `0.092`.
- Evaluation:
  - 1ep smoke:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/stackcube_count999_qpos_filter005_9000step_2cam_smoke_1ep_seed1000`
    scored `0/1`, `0.0%`.
  - 300ep horizon-30 eval, seed `1000`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/stackcube_count999_qpos_filter005_9000step_2cam_horizon30_eval_300ep_seed1000`
    scored `1/300`, `0.33%`.
  - 100ep horizon-100 debug eval, seed `1000`:
    `_workspace/runpod_results/maniskill3_stare_sft_scale_probe_20260607/stackcube_count999_qpos_filter005_9000step_2cam_horizon100_eval_100ep_seed1000`
    scored `0/100`, `0.0%`.
- Local fetched bundle:
  `_workspace/runpod_results/20260607_maniskill3_stackcube_qpos005_9000_2cam`.
- Delta:
  paper-horizon seed `1000` is `0.33%` vs STARE StackCube `12.7%`, a
  `-12.37pp` gap.
- Interpretation:
  this is a real table-backed row but not a parity result. The horizon-100
  debug run also failed, so the low StackCube score is not merely a
  short-horizon timing issue. Continue by checking StackCube source trajectory,
  action/control-mode details, exact idle-action filtering, and whether the
  skipped replay episode or two-camera preprocessing differs from STARE.

### ManiSkill3 PullCube Qpos Filter 0.05 SFT

- Status: completed on RunPod; five-seed 300ep results fetched locally; Pod
  left running for follow-up parity work.
- Purpose:
  add a third STARE-style non-LIBERO SmolVLA fine-tuning row after PushCube and
  StackCube.
- Reference:
  STARE Table 2 reports `PullCube-v1` SmolVLA fine-tuning success `90.7%`.
  Appendix B.5 reports horizon `30`, `300` episodes, and `5` random seeds.
- Replay finding:
  action replay without env states saved only `130/1000`, `13.0%`, so it was
  rejected for SFT. Replaying the same official `pd_joint_delta_pos` RL source
  with `--use-env-states` saved `989/1000`, `98.90%`.
- LeRobot conversion:
  - output:
    `/root/physical-ai/tmp_lerobot_stare_pullcube_rgb_count989_rgbmode_128_use_env_states`
  - result:
    `989` episodes, `20941` frames.
  - features:
    action shape `[8]`, state shape `[9]`, camera `base_camera`.
- Qpos idle-filtered dataset:
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_pullcube_rgb_count989_rgbmode_128_use_env_states_qpos_filter005`
  - threshold:
    qpos delta `0.05`.
  - size:
    `989` episodes, `19753` frames, mean length `20.0`, p50 `19`, p90 `24`,
    max `37`.
- Training:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pullcube_count989_qpos_filter005_9000step`
  - checkpoint:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_pullcube_count989_qpos_filter005_9000step/checkpoints/009000/pretrained_model`
  - base:
    `lerobot/smolvla_base`
  - steps:
    `9000`
  - rename map:
    `base_camera -> camera1`
  - training completion:
    `9000/9000`, final logged loss `0.083`.
- Evaluation:
  - 10ep smoke seed `1000`:
    `1/10`, `10.0%`.
  - 300ep seed `1000`:
    `29/300`, `9.67%`.
  - 300ep seed `1001`:
    `29/300`, `9.67%`.
  - 300ep seed `1002`:
    `34/300`, `11.33%`.
  - 300ep seed `1003`:
    `28/300`, `9.33%`.
  - 300ep seed `1004`:
    `25/300`, `8.33%`.
  - five-seed aggregate:
    `145/1500`, `9.67%`.
- Local fetched bundle:
  `_workspace/runpod_results/20260607_maniskill3_pullcube_qpos005_9000_5seed`.
- Delta:
  five-seed aggregate is `9.67%` vs STARE PullCube `90.7%`, a `-81.03pp`
  gap.
- Interpretation:
  this is now a paper-scale row by episode and seed count, but not a parity
  result. Because `--use-env-states` was required to recover the replay
  success rate, likely suspects are source-demo type, control-mode/action
  replay semantics, exact STARE idle-action filtering, and training-protocol
  mismatch rather than evaluation sample count.

### ManiSkill3 LiftPegUpright Qpos Filter 0.05 SFT

- Status: completed on RunPod; five-seed 300ep results fetched locally; Pod
  left running for follow-up parity/debug work.
- Purpose:
  add the fourth STARE-style non-LIBERO SmolVLA fine-tuning row.
- Reference:
  STARE Table 2 reports `LiftPegUpright-v1` SmolVLA fine-tuning success
  `16.3%`. Appendix B.5 reports horizon `30`, `300` episodes, and `5` random
  seeds.
- Source audit:
  official `LiftPegUpright-v1` RL demos include `pd_ee_delta_pose` with `1015`
  successes and `pd_joint_delta_pos` with `993` successes. The
  `pd_joint_delta_pos` source was selected to match the current 8D action
  SFT/eval lane.
- Replay audit:
  - `pd_ee_delta_pose` count100 no-env-state replay:
    `3/100`, `3.0%`.
  - `pd_joint_delta_pos` count100 no-env-state replay:
    `0/100`, `0.0%`.
  - `pd_ee_delta_pose` count100 `--use-env-states` replay:
    `90/100`, `90.0%`.
  - `pd_joint_delta_pos` count100 `--use-env-states` replay:
    `93/100`, `93.0%`.
  - full selected `pd_joint_delta_pos` `--use-env-states` replay:
    `878/993`, `88.42%` demos saved.
- LeRobot conversion:
  - output:
    `/root/physical-ai/tmp_lerobot_stare_liftpeg_rgb_count878_rgbmode_128_use_env_states`
  - result:
    `878` episodes, `35726` frames.
  - features:
    action shape `[8]`, state shape `[9]`, camera `base_camera`.
- Qpos idle-filtered dataset:
  - destination:
    `/root/physical-ai/tmp_lerobot_stare_liftpeg_rgb_count878_rgbmode_128_use_env_states_qpos_filter005`
  - threshold:
    qpos delta `0.05`.
  - size:
    `878` episodes, `34663` frames, mean length `39.48`, p50 `41`, p90 `48`,
    max `50`.
- Training:
  - output:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_liftpeg_count878_qpos_filter005_9000step`
  - checkpoint:
    `/root/physical-ai/tmp_train_smolvla_base_maniskill3_liftpeg_count878_qpos_filter005_9000step/checkpoints/009000/pretrained_model`
  - base:
    `lerobot/smolvla_base`
  - steps:
    `9000`
  - rename map:
    `base_camera -> camera1`
  - training completion:
    `9000/9000`.
- Evaluation:
  - 10ep smoke seed `1000`:
    `0/10`, `0.0%`.
  - 300ep seed `1000`:
    `0/300`, `0.0%`.
  - 300ep seed `1001`:
    `0/300`, `0.0%`.
  - 300ep seed `1002`:
    `0/300`, `0.0%`.
  - 300ep seed `1003`:
    `0/300`, `0.0%`.
  - 300ep seed `1004`:
    `0/300`, `0.0%`.
  - five-seed aggregate:
    `0/1500`, `0.0%`.
  - horizon-100 debug seed `1000`:
    `0/100`, `0.0%`.
- Local fetched bundle:
  `_workspace/runpod_results/20260607_maniskill3_liftpeg_qpos005_9000_5seed`.
- Delta:
  five-seed aggregate is `0.0%` vs STARE LiftPegUpright `16.3%`, a `-16.3pp`
  gap.
- Interpretation:
  this is now a paper-scale row by episode and seed count, but not a parity
  result. The horizon-100 debug run also scored `0/100`, so the failure is not
  only a short-horizon issue. Because the selected source required
  `--use-env-states` and still dropped `115` of `993` episodes, likely suspects
  are exact source-demo selection, control-mode/action replay semantics,
  STARE's idle-action filter, and training-protocol details rather than
  evaluation sample count.
