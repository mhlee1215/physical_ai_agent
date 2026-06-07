# Non-LIBERO Evaluation Status

Date: 2026-06-07

This note tracks evaluation options after freezing the LIBERO SmolVLA baseline.
The goal is to produce non-LIBERO numbers that can eventually be compared with
paper or official leaderboard values.

## Current Decision

The best table-backed external benchmark is now **Meta-World MT50**. RoboCasa
remains the best household-manipulation lane, but no public SmolVLA RoboCasa
reference table has been found yet.

For the next evaluation cycles, prioritize benchmarks with an existing paper
or official leaderboard table that can be placed side by side with our result.
Use benchmarks without a direct reference row as plumbing or qualitative
evidence only, not as the main paper-comparison target.

Why:

- ManiSkill/HAB already has repo-local execution plumbing and Mac-local pilot
  artifacts, but the current strict Mac renderer path is blocked and the
  available policy rows are random/zero controls.
- Meta-World is directly reported in the SmolVLA paper Table 2 and has a
  released checkpoint, `lerobot/smolvla_metaworld`.
- RoboCasa365 is more paper-relevant for household long-horizon agentic
  wrappers, and the RunPod path now supports strict reset/step plus
  `lerobot/smolvla_robocasa` evaluation over 3-task and 5-task subsets.
- RoboTwin/Isaac/SimplerEnv/CALVIN are plausible later lanes, but they require
  new model/action-space compatibility work before a fair SmolVLA comparison.
- VLA evaluation harness is now the best infrastructure lane for future
  cross-benchmark table reproduction, but the current RunPod image does not
  include Docker, and the public harness model-server configs do not include
  SmolVLA yet.

## Primary Reference Anchors

### RoboCasa365 / RoboCasa

Official RoboCasa365 leaderboard scope:

- 50-task multi-task benchmark.
- 3 evaluation splits: Atomic-Seen, Composite-Seen, Composite-Unseen.
- Overall score is the published average task success rate.

Current leaderboard values:

| Policy | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen |
| --- | ---: | ---: | ---: | ---: |
| RLDX-1 | 33.2 | 63.0 | 27.5 | 5.4 |
| GR00T N1.5 | 23.9 | 50.7 | 14.8 | 2.7 |
| GR00T N1.6 | 21.9 | 51.1 | 9.4 | 1.7 |
| GigaWorld-Policy 0.1 | 20.7 | 44.4 | 11.8 | 2.9 |
| pi0.5 | 16.9 | 39.6 | 7.1 | 1.2 |
| pi0 | 14.8 | 34.6 | 6.1 | 1.1 |
| Diffusion Policy | 6.1 | 15.7 | 0.2 | 1.3 |

Reference:

- https://robocasa.ai/leaderboard.html
- https://robocasa.ai/
- https://huggingface.co/docs/lerobot/main/robocasa

### ManiSkill / ManiSkill-HAB

Official ManiSkill macOS note:

- macOS supports CPU simulation and standard rendering.
- GPU simulation is not supported on macOS.
- Rendering-intensive workflows are recommended for CUDA-capable machines.

Reference:

- https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/getting_started/macos_install.html

ManiSkill-HAB / MS-HAB reference:

- ICLR 2025 benchmark for low-level manipulation in home rearrangement tasks.
- Project page includes Pick, Place, Open, and Close subtasks and failure mode
  examples.
- The supplementary page reports a rendering benchmark on a single RTX 4090:
  ManiSkill-HAB `69.90 +/- 0.25` SPS and Behavior-1k `19.92 +/- 0.04` SPS.

Reference:

- https://sites.google.com/view/maniskill-hab
- https://openreview.net/forum?id=6bKEWevgSd

### Meta-World MT50

SmolVLA paper Table 2 reference:

| Policy | Easy | Medium | Hard | Very Hard | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diffusion Policy | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| SmolVLA 0.45B | 82.5 | 41.8 | 45.0 | 60.0 | 57.3 |
| SmolVLA 2.25B | 87.14 | 51.82 | 70.0 | 64.0 | 68.24 |

Protocol notes:

- SmolVLA paper reports 10 trials per task.
- LeRobot Meta-World groups are `easy`, `medium`, `hard`, and `very_hard`.
- The released checkpoint is `lerobot/smolvla_metaworld`.

Reference:

- https://arxiv.org/abs/2506.01844
- https://huggingface.co/docs/lerobot/v0.4.3/metaworld
- https://huggingface.co/lerobot/smolvla_metaworld

### VLA evaluation harness

The `allenai/vla-evaluation-harness` project is a unified evaluation harness
with public leaderboard coverage across many robot simulation benchmarks. Its
README lists Dockerized benchmark support for LIBERO, CALVIN, SimplerEnv,
ManiSkill2, RoboCasa, RoboTwin, VLABench, RLBench, and others, plus official
model-server configs for OpenVLA, pi0, pi0-FAST, GR00T, X-VLA, CogACT, and
related models. SmolVLA is not currently listed as an official model-server
config in the checked `v0.2.0` source tree.

Reference:

- https://github.com/allenai/vla-evaluation-harness
- https://allenai.github.io/vla-evaluation-harness/leaderboard/
- https://arxiv.org/abs/2603.13966

## Our Current Non-LIBERO Evidence

### 1. Current strict PickCube target is blocked on this Mac

Command:

```bash
sh scripts/checkpoint_24.sh \
  --require-maniskill \
  --no-fallback-env \
  --env-id PickCube-v1 \
  --episodes 1 \
  --steps 1 \
  --policy zero \
  --output-dir _workspace/checkpoints/checkpoint_24_non_libero_pickcube_nofallback_1ep_1step
```

Result:

| Field | Value |
| --- | --- |
| status | failed |
| requested env | `PickCube-v1` |
| executed env | `PickCube-v1` |
| rollout status | blocked |
| episodes | 0 |
| blocker | `vk::createInstanceUnique: ErrorIncompatibleDriver` |
| report | `_workspace/checkpoints/checkpoint_24_non_libero_pickcube_nofallback_1ep_1step/checkpoint_report.json` |
| blocker artifact | `_workspace/checkpoints/checkpoint_24_non_libero_pickcube_nofallback_1ep_1step/maniskill_blocker.md` |

Interpretation:

This is the authoritative current-state blocker for strict `PickCube-v1` on the
Mac path. It does not contradict prior pilot artifacts; it says the current
strict target-env path needs a renderer/driver-compatible machine before a
paper-comparable ManiSkill task-success run.

### 2. Current CP24 fallback pipeline still executes

Command:

```bash
sh scripts/checkpoint_24.sh \
  --require-maniskill \
  --episodes 2 \
  --steps 20 \
  --policy zero \
  --output-dir _workspace/checkpoints/checkpoint_24_non_libero_verify_2ep_20step
```

Result:

| Field | Value |
| --- | --- |
| status | passed |
| requested env | `PickCube-v1` |
| executed fallback env | `Empty-v1` |
| total rollout episodes | 4 |
| policies | `random`, `zero` |
| success | 0/4 |
| blocker for requested env | `PickCube-v1` Vulkan incompatible driver |
| report | `_workspace/checkpoints/checkpoint_24_non_libero_verify_2ep_20step/checkpoint_report.json` |

Interpretation:

This confirms the CP24 logging/reporting path still works, but it is not a
ManiSkill manipulation benchmark result because fallback `Empty-v1` executed.

### 3. Existing PickCube pilot artifact

Artifact:

```text
_workspace/checkpoints/checkpoint_24_pickcube_baselines_20ep_100step/maniskill_rollout/metrics.json
```

Observed metrics:

| Policy | Episodes | Success | Success rate | Mean reward sum | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 20 | 0 | 0.000 | 2.527653 | 50.0 |
| zero | 20 | 1 | 0.050 | 3.152852 | 47.55 |

Interpretation:

This is a useful old Mac-local pilot, but it is not a paper-comparable
ManiSkill number. The directory name says `100step`, while the saved metrics
show 50-step episodes, and the policy is random/zero rather than a trained
policy such as RDT/OpenVLA/DP. Treat it as pipeline evidence, not a final
comparison.

### 4. Existing ManiSkill-HAB pilot artifacts

Report:

```text
_workspace/reports/hab_success_once_20ep_2026-06-04.md
```

Results:

| Task | Policy | Episodes | Horizon | Success once |
| --- | --- | ---: | ---: | ---: |
| `ReplicaCADSetTableVal_SceneManipulation-v1` | random | 20 | 200 | 0/20 |
| `ReplicaCADSetTableVal_SceneManipulation-v1` | zero | 20 | 200 | 0/20 |
| `ReplicaCADPrepareGroceriesVal_SceneManipulation-v1` | random | 20 | 200 | 0/20 |
| `ReplicaCADPrepareGroceriesVal_SceneManipulation-v1` | zero | 20 | 200 | 0/20 |

Interpretation:

This proves partial HAB tasks can execute locally with authoritative task ids
and fallback disabled. It is not comparable to MS-HAB policy results because
random/zero are weak controls and the run is 20 episodes, not paper scale.

### 5. RoboCasa365 / RoboCasa CP25 probe gate

Commands:

```bash
sh scripts/checkpoint_25.sh
sh scripts/checkpoint_25.sh --probe-reset-step --require-robocasa --task CloseFridge
```

RunPod strict result:

| Field | Value |
| --- | --- |
| status | passed |
| task | `CloseFridge` |
| probe language | `Close the fridge door.` |
| steps executed | 1 |
| success | not reported by one-step probe |
| artifact | `_workspace/runpod_results/checkpoint_25_robocasa_strict_assets_ready_20260607T061729Z/checkpoint_report.json` |

Expected/generated artifacts:

| Artifact | Path |
| --- | --- |
| checkpoint report | `_workspace/checkpoints/checkpoint_25_robocasa/checkpoint_report.json` |
| blocker or no-blocker note | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa_blocker.md` |
| install/eval command handoff | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa_install_and_eval.md` |
| reference comparison table | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa365_reference_table.md` |

### 6. RoboCasa SmolVLA 3-task subset evaluation

Command family:

```bash
PY=/root/physical-ai/envs/lerobot_py312/bin/python
COMBO=/workspace/physical-ai/robocasa:/workspace/physical-ai/robosuite:/workspace/physical-ai/vendor/lerobot/src
export MUJOCO_GL=egl
export HF_HOME=/workspace/physical-ai/hf_home
export TRANSFORMERS_CACHE=/workspace/physical-ai/hf_home/transformers
export HF_HUB_CACHE=/workspace/physical-ai/hf_home/hub

PYTHONPATH="$COMBO:${PYTHONPATH:-}" "$PY" -m lerobot.scripts.lerobot_eval \
  --output_dir="_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z" \
  --policy.path=lerobot/smolvla_robocasa \
  --env.type=robocasa \
  --env.task=CloseFridge,OpenCabinet,OpenDrawer \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --rename_map='{"observation.images.robot0_agentview_left":"observation.images.camera1","observation.images.robot0_eye_in_hand":"observation.images.camera2","observation.images.robot0_agentview_right":"observation.images.camera3"}' \
  --seed=0
```

Result:

| Task | Policy | Episodes | Horizon | Success | Success rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `CloseFridge` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `OpenCabinet` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `OpenDrawer` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| Overall 3-task subset | `lerobot/smolvla_robocasa` | 60 | default LeRobot/RoboCasa `1000` | 0/60 | 0.0% |

Artifacts:

| Artifact | Path |
| --- | --- |
| metrics | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/eval_info.json` |
| command | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/run_command.txt` |
| `CloseFridge` representative video | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/videos/CloseFridge_0/eval_episode_0.mp4` |
| `OpenCabinet` representative video | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/videos/OpenCabinet_0/eval_episode_0.mp4` |
| `OpenDrawer` representative video | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/videos/OpenDrawer_0/eval_episode_0.mp4` |

Visual inspection:

- `CloseFridge_0/frame_000.png`: RoboCasa kitchen scene with open fridge door
  and robot arm.
- `OpenCabinet_0/frame_000.png`: RoboCasa kitchen sink/cabinet scene with
  robot arm.
- `OpenDrawer_0/frame_000.png`: RoboCasa kitchen fixture scene with robot arm.

Interpretation:

This is a protocol-compatible small-subset RoboCasa policy evaluation: real
SmolVLA weights, real visual observations, 20 episodes per task, default
RoboCasa horizon, and benchmark success metric. It is still **not** directly
comparable to the RoboCasa365 leaderboard overall number because the
leaderboard averages 50 tasks over Atomic-Seen, Composite-Seen, and
Composite-Unseen splits. It is, however, now a concrete external benchmark row
we can scale from.

### 7. RoboCasa 5-task scale-up blocker

Attempted task list:

```text
CloseFridge,OpenCabinet,OpenDrawer,TurnOnMicrowave,TurnOffStove
```

Result:

| Field | Value |
| --- | --- |
| completed before blocker | `CloseFridge`, `OpenCabinet`, `OpenDrawer` |
| blocker task | `TurnOnMicrowave` |
| error | `ValueError: a cannot be empty unless no samples are taken` |
| source line | `robocasa/models/objects/kitchen_object_utils.py`, `rng.choice(valid_categories)` |
| artifact | `_workspace/runpod_results/robocasa_smolvla_5task_20ep_default_horizon_20260607T065632Z/eval.log` |

Interpretation:

The 5-task run is blocked by lightweight asset coverage, not by SmolVLA
inference or RoboCasa runtime. The current RunPod setup installed lightweight
`tex`, `tex_generative`, `fixtures_lw`, and `objs_lw` assets. Some microwave
or stove task categories require fuller object assets or a task subset that is
known to be covered by the lightweight registry.

### 8. RoboCasa lightweight-compatible 5-task subset

Task list:

```text
OpenFridge,AdjustWaterTemperature,OpenBlenderLid,CloseBlenderLid,TurnOnToaster
```

Result:

| Task | Policy | Episodes | Horizon | Success | Success rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `OpenFridge` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `AdjustWaterTemperature` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `OpenBlenderLid` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `CloseBlenderLid` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 0/20 | 0.0% |
| `TurnOnToaster` | `lerobot/smolvla_robocasa` | 20 | default LeRobot/RoboCasa `1000` | 4/20 | 20.0% |
| Overall 5-task subset | `lerobot/smolvla_robocasa` | 100 | default LeRobot/RoboCasa `1000` | 4/100 | 4.0% |

Artifacts:

| Artifact | Path |
| --- | --- |
| metrics | `_workspace/runpod_results/robocasa_smolvla_5task_20ep_default_horizon_20260607T094500Z/eval_info.json` |
| command | `_workspace/runpod_results/robocasa_smolvla_5task_20ep_default_horizon_20260607T094500Z/run_command.txt` |

Interpretation:

This completed run confirms SmolVLA/RoboCasa is not uniformly broken: one
lightweight-compatible atomic task, `TurnOnToaster`, produced nonzero success.
It remains a secondary baseline because no public SmolVLA RoboCasa reference
table has been found.

### 9. Meta-World MT50 SmolVLA Table 2 catch-up run

Command family:

```bash
PY=/root/physical-ai/envs/lerobot_py312/bin/python
COMBO=/workspace/physical-ai/vendor/lerobot/src
RENAME_MAP='{"observation.image":"observation.images.camera1"}'
export MUJOCO_GL=egl
export HF_HOME=/workspace/physical-ai/hf_home
export TRANSFORMERS_CACHE=/workspace/physical-ai/hf_home/transformers
export HF_HUB_CACHE=/workspace/physical-ai/hf_home/hub

PYTHONPATH="$COMBO:${PYTHONPATH:-}" "$PY" -m lerobot.scripts.lerobot_eval \
  --output_dir="_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z" \
  --policy.path=lerobot/smolvla_metaworld \
  --env.type=metaworld \
  --env.task=easy,medium,hard,very_hard \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.empty_cameras=2 \
  --rename_map="$RENAME_MAP" \
  --seed=0
```

The rename and empty-camera arguments are required because the Meta-World env
exposes one visual observation key, `observation.image`, while the released
SmolVLA checkpoint expects `observation.images.camera1`,
`observation.images.camera2`, and `observation.images.camera3`.

Result versus SmolVLA paper Table 2:

| Split | Ours | SmolVLA paper 0.45B | Delta | Episodes |
| --- | ---: | ---: | ---: | ---: |
| Easy | 65.4 | 82.5 | -17.1 | 280 |
| Medium | 38.2 | 41.8 | -3.6 | 110 |
| Hard | 38.3 | 45.0 | -6.7 | 60 |
| Very Hard | 20.0 | 60.0 | -40.0 | 50 |
| Table-style avg | 40.5 | 57.3 | -16.9 | 500 |
| Episode-weighted overall | 51.6 | n/a | n/a | 500 |

Artifacts:

| Artifact | Path |
| --- | --- |
| metrics | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/eval_info.json` |
| command | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/run_command.txt` |
| easy representative success video | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/videos/easy_0/eval_episode_0.mp4` |
| medium representative success video | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/videos/medium_0/eval_episode_5.mp4` |
| hard representative success video | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/videos/hard_0/eval_episode_2.mp4` |
| very hard representative success video | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_table2_20260607T115500Z/videos/very_hard_0/eval_episode_3.mp4` |

Visual inspection:

Representative frames from all four difficulty groups were inspected locally.
They show valid Meta-World Sawyer tabletop scenes with task objects and are not
blank renderer artifacts.

Interpretation:

This is the first non-LIBERO result in this thread with the right scale and a
direct SmolVLA paper reference table. The result is not yet parity: medium and
hard are close, but easy and especially very hard are far below the paper
reference. The highest-probability parity issues are the one-camera
`observation.image` mapping with two empty camera placeholders, possible
version differences in LeRobot/Meta-World, and seed/task reset differences.

#### Meta-World parity ablations

The first targeted parity check focused on `very_hard`, because that split has
the largest gap against SmolVLA Table 2 (`20.0%` ours versus `60.0%` reported).

| Run | Intended check | Result | Delta vs full MT50 `very_hard` | Delta vs Table 2 |
| --- | --- | ---: | ---: | ---: |
| `metaworld_smolvla_veryhard_10ep_ep400_seed0_fixedrename_20260607T102444Z` | Try the checkpoint `train_config.json` `episode_length=400` clue with correct `rename_map` parsing | 10/50, 20.0% | 0.0 | -40.0 |
| `metaworld_smolvla_veryhard_10ep_seed1000_20260607T103256Z` | Try checkpoint/training seed `1000` as a reset-distribution candidate | 13/50, 26.0% | +6.0 | -34.0 |
| `metaworld_smolvla_veryhard_10ep_seed1000_empty0_20260607T104608Z` | Match checkpoint `empty_cameras=0` instead of padding missing `camera2/3` with blank images | 13/50, 26.0% | +6.0 | -34.0 |
| `metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_20260607T105427Z` | Match `n_episodes=10` with `batch_size=10`, so each task uses one vectorized batch of seeds `1000..1009` | 18/50, 36.0% | +16.0 | -24.0 |
| `metaworld_smolvla_veryhard_10ep_seed0_v051_smolvlaonly3_20260607T120405Z` | Run LeRobot `v0.5.1` source instead of current HEAD to test version drift; Python 3.12 required a SmolVLA-only patch that removes unrelated GR00T registry imports | 6/50, 12.0% | -8.0 | -48.0 |
| `metaworld_smolvla_veryhard_50ep_seed1000_batch50_empty0_20260607T121323Z` | Match the released checkpoint `train_config.json` eval hints more closely: `seed=1000`, `empty_cameras=0`, `eval.n_episodes=50`, `eval.batch_size=50`; `very_hard` only | 61/250, 24.4% | +4.4 | -35.6 |
| `metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_topcam_20260607T124317Z` | Temporarily patch the LeRobot Meta-World wrapper from `corner2` to `top` because the feature key is named `pixels/top`; restore source after run | 0/50, 0.0% | -20.0 | -60.0 |
| `metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_mt1seed1000_20260607T130000Z` | Temporarily patch Meta-World suite construction from `MT1(..., seed=42)` to `seed=1000` while keeping eval seed, batch size, camera, and empty-camera settings fixed | 18/50, 36.0% | +16.0 | -24.0 |
| `metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_actual400_20260607T134500Z` | Temporarily patch the LeRobot Meta-World wrapper from `_max_episode_steps = 500` to `400` to test the released checkpoint `train_config.json` horizon clue directly; restore source after run | 17/50, 34.0% | +14.0 | -26.0 |

Notes:

- A malformed local quoting attempt produced
  `rename_map={'observation.image:observation.images.camera1': None}` and
  failed before rollout with missing image features. It is not counted as an
  evaluation result.
- The corrected runs show `rename_map={'observation.image':
  'observation.images.camera1'}` in `eval.log`.
- The current LeRobot Meta-World wrapper still logs `Running rollout with at
  most 500 steps` even when `--env.episode_length=400` is passed. Remote code
  inspection showed `lerobot/envs/metaworld.py` hardcodes
  `_max_episode_steps = 500`, so this CLI flag does not currently reproduce a
  400-step horizon.
- Seed `1000` improves `very_hard` from `20.0%` to `26.0%`, but this is far
  short of the paper's `60.0%`. Seed/reset variance alone is therefore not
  enough to explain the parity gap.
- Matching checkpoint `empty_cameras=0` produced the same `26.0%` as
  `empty_cameras=2` under seed `1000`. Empty-camera padding is therefore not a
  leading explanation for the `very_hard` gap.
- `batch_size=10` improved the standalone `very_hard` split to `36.0%`, but a
  full MT50 rerun with the same setting did not preserve that improvement.
  Treat standalone split ablations as useful probes, not final parity numbers.
- The LeRobot `v0.5.1` source check did not recover the reported `very_hard`
  score. It ran after removing only unrelated GR00T top-level registry/import
  paths that fail under Python 3.12 before SmolVLA evaluation starts. The
  completed SmolVLA-only `v0.5.1` result was `6/50 = 12.0%`, worse than the
  current HEAD seed-0 `very_hard` result (`20.0%`). This weakens the hypothesis
  that the main parity gap is explained by moving from LeRobot `v0.5.1` to the
  current LeRobot source alone.
- The 250-episode `train_config`-style `very_hard` run also did not recover
  Table 2 parity. It produced `61/250 = 24.4%`. Per-task success was highly
  uneven: task `0` `18/50 = 36.0%`, task `1` `30/50 = 60.0%`, task `2`
  `5/50 = 10.0%`, task `3` `1/50 = 2.0%`, and task `4` `7/50 = 14.0%`.
  This means one `very_hard` task can match the paper's aggregate `60.0%`,
  but the group average remains far below it. The parity gap is therefore more
  likely tied to task/reset/protocol details for specific `very_hard` tasks,
  not merely to the number of trials per task.
- The temporary top-camera patch was a negative control. `top` rendered
  successfully under `MUJOCO_GL=egl`, but policy success fell to `0/50`.
  The remote source was restored to `camera_name="corner2"` after the run.
  This rules out the simple hypothesis that the checkpoint feature key
  `pixels/top` implies the released policy should be evaluated with the actual
  Meta-World top camera.
- The temporary `MT1(..., seed=1000)` patch was also a negative control for
  overall parity. It matched the same-condition `MT1(..., seed=42)` standalone
  run at `18/50 = 36.0%`. The per-task distribution changed only slightly:
  with `seed=42`, tasks `0..4` were `60.0`, `80.0`, `30.0`, `10.0`, `0.0`;
  with `seed=1000`, they were `60.0`, `80.0`, `10.0`, `20.0`, `10.0`.
  This means the suite seed changes individual task samples but does not
  explain the gap to the paper's `60.0%` very-hard aggregate on its own.
- The actual 400-step horizon patch was a direct check of the checkpoint
  `train_config.json` `episode_length=400` clue. The log confirmed
  `Running rollout with at most 400 steps`, but the result was
  `17/50 = 34.0%`, slightly below the same-condition 500-step standalone run
  (`18/50 = 36.0%`). Per-task success was `60.0`, `60.0`, `20.0`, `20.0`,
  and `10.0` for tasks `0..4`. This rules out episode horizon as the
  standalone explanation for the Table 2 very-hard parity gap.
- Checkpoint processor stats were inspected in
  `lerobot/smolvla_metaworld`. Although `policy_preprocessor.json` lists
  `observation.state` with feature shape `[6]`, the saved normalizer tensors
  contain `observation.state.mean/std/min/max` with shape `(4,)`, matching the
  current Meta-World wrapper's `agent_pos = raw_obs[:4]`. This weakens the
  hypothesis that the current low score is caused by a simple state-dimension
  mismatch between training and evaluation.
- Dataset metadata from `lerobot/metaworld_mt50` was inspected without
  downloading the full dataset. `meta/info.json` reports `total_episodes=2500`
  and `total_tasks=49`, while episode metadata contains 50 distinct `task_id`
  values (`0..49`) with 50 episodes each. The `total_tasks=49` count comes from
  description-level `task_index`: two tasks share the same instruction text
  (`Push the puck to a goal`) and collapse to one `task_index`. The current
  `very_hard` task names still match dataset `task_id` values by instruction
  text: `shelf-place-v3 -> 42`, `disassemble-v3 -> 12`,
  `stick-pull-v3 -> 44`, `stick-push-v3 -> 45`, and
  `pick-place-wall-v3 -> 32`. This rules out a simple very-hard task-id
  mismatch between the current LeRobot config and the dataset metadata.
- Meta-World action clipping was probed on RunPod for representative
  `very_hard` tasks. For `shelf-place-v3`, `disassemble-v3`, and
  `stick-pull-v3`, stepping once from the same reset with large actions such as
  `[10, -10, 10, 1]` produced the same next observation, reward, and info as
  the clipped action `[1, -1, 1, 1]`. The same held for the negative direction.
  This weakens the hypothesis that the dataset's large action statistics are
  causing a simple action-scale mismatch during evaluation.
- A no-CLI-rename smoke run was executed with `--rename_map` omitted. It failed
  before rollout with a policy/environment feature mismatch: missing
  `observation.images.camera1/2/3` and extra `observation.image`. Inspection of
  `lerobot_eval.py` shows the eval script overrides the checkpoint
  `rename_observations_processor` with `cfg.rename_map`; therefore the explicit
  CLI rename `{"observation.image": "observation.images.camera1"}` is required
  for Meta-World evaluation and is not an accidental duplicate preprocessing
  step.

Artifacts:

| Artifact | Path |
| --- | --- |
| ep400/seed0 metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_ep400_seed0_fixedrename_20260607T102444Z/eval_info.json` |
| ep400/seed0 command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_ep400_seed0_fixedrename_20260607T102444Z/run_command.txt` |
| seed1000 metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_20260607T103256Z/eval_info.json` |
| seed1000 command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_20260607T103256Z/run_command.txt` |
| seed1000/empty0 metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_empty0_20260607T104608Z/eval_info.json` |
| seed1000/empty0 command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_empty0_20260607T104608Z/run_command.txt` |
| seed1000/batch10/empty0 very hard metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_20260607T105427Z/eval_info.json` |
| seed1000/batch10/empty0 very hard command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_20260607T105427Z/run_command.txt` |
| v0.5.1 SmolVLA-only very hard metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed0_v051_smolvlaonly3_20260607T120405Z/eval_info.json` |
| v0.5.1 SmolVLA-only very hard command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed0_v051_smolvlaonly3_20260607T120405Z/run_command.txt` |
| v0.5.1 SmolVLA-only representative video | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed0_v051_smolvlaonly3_20260607T120405Z/videos/very_hard_0/eval_episode_0.mp4` |
| train-config-style 250ep very hard metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_50ep_seed1000_batch50_empty0_20260607T121323Z/eval_info.json` |
| train-config-style 250ep very hard command | `_workspace/runpod_results/metaworld_smolvla_veryhard_50ep_seed1000_batch50_empty0_20260607T121323Z/run_command.txt` |
| train-config-style 250ep representative video | `_workspace/runpod_results/metaworld_smolvla_veryhard_50ep_seed1000_batch50_empty0_20260607T121323Z/videos/very_hard_0/eval_episode_0.mp4` |
| top-camera negative-control metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_topcam_20260607T124317Z/eval_info.json` |
| top-camera negative-control command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_topcam_20260607T124317Z/run_command.txt` |
| MT1 seed1000 negative-control metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_mt1seed1000_20260607T130000Z/eval_info.json` |
| MT1 seed1000 negative-control command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_mt1seed1000_20260607T130000Z/run_command.txt` |
| actual 400-step horizon metrics | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_actual400_20260607T134500Z/eval_info.json` |
| actual 400-step horizon command | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_actual400_20260607T134500Z/run_command.txt` |
| actual 400-step horizon patch backup | `_workspace/runpod_results/metaworld_smolvla_veryhard_10ep_seed1000_batch10_empty0_actual400_20260607T134500Z/metaworld.py.before_actual400_patch` |
| checkpoint preprocessor stats inspected on RunPod | `/workspace/physical-ai/hf_home/hub/models--lerobot--smolvla_metaworld/snapshots/cd6778d2cfa724c1bf5fc637490548e54d81dc4c/policy_preprocessor_step_5_normalizer_processor.safetensors` |
| dataset metadata inspected on RunPod | `/workspace/physical-ai/hf_home/hub/datasets--lerobot--metaworld_mt50/snapshots/a59f742d218c903328164257ecf180f9b18018a1/meta/info.json` |
| dataset task metadata inspected on RunPod | `/workspace/physical-ai/hf_home/hub/datasets--lerobot--metaworld_mt50/snapshots/a59f742d218c903328164257ecf180f9b18018a1/meta/tasks.parquet` |
| dataset episode metadata inspected on RunPod | `/workspace/physical-ai/hf_home/hub/datasets--lerobot--metaworld_mt50/snapshots/a59f742d218c903328164257ecf180f9b18018a1/meta/episodes/chunk-000/file-000.parquet` |
| no-rename smoke failure log | `_workspace/runpod_results/metaworld_smolvla_veryhard_1ep_seed1000_norenamemap_20260607T133000Z/eval.log` |

#### Full MT50 batch-size parity rerun

The most paper-like rerun after the targeted probes used all 50 tasks again
with `seed=1000`, `batch_size=10`, `empty_cameras=0`, and 10 trials per task.

| Split | Ours, batch10/seed1000/empty0 | SmolVLA paper 0.45B | Delta | Episodes |
| --- | ---: | ---: | ---: | ---: |
| Easy | 67.1 | 82.5 | -15.4 | 280 |
| Medium | 43.6 | 41.8 | +1.8 | 110 |
| Hard | 23.3 | 45.0 | -21.7 | 60 |
| Very Hard | 20.0 | 60.0 | -40.0 | 50 |
| Table-style avg | 38.5 | 57.3 | -18.8 | 500 |
| Episode-weighted overall | 52.0 | n/a | n/a | 500 |

Artifacts:

| Artifact | Path |
| --- | --- |
| full MT50 batch10 metrics | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_seed1000_batch10_empty0_20260607T110031Z/eval_info.json` |
| full MT50 batch10 command | `_workspace/runpod_results/metaworld_smolvla_mt50_10ep_seed1000_batch10_empty0_20260607T110031Z/run_command.txt` |

Interpretation:

This rerun does not improve the main comparison table. It slightly improves
`medium` but worsens `hard` and leaves `very_hard` at `20.0%`. The best current
paper-comparable Meta-World result remains the first full MT50 run
(`table-style avg 40.5%`, episode-weighted `51.6%`). Remaining parity
candidates are now more likely to be LeRobot/Meta-World version differences,
task reset distributions beyond the exposed seed, or a training/eval protocol
detail not encoded in the released model card.

### 10. VLA evaluation harness RunPod install audit

Command family:

```bash
cd /workspace/physical-ai
python3.11 -m venv /root/physical-ai/envs/vla_eval_py311
/root/physical-ai/envs/vla_eval_py311/bin/python -m pip install vla-eval==0.2.0
git clone --depth 1 --branch v0.2.0 https://github.com/allenai/vla-evaluation-harness.git
cd /workspace/physical-ai/vla-evaluation-harness
/root/physical-ai/envs/vla_eval_py311/bin/python -m pip install -e .
/root/physical-ai/envs/vla_eval_py311/bin/vla-eval test --list
/root/physical-ai/envs/vla_eval_py311/bin/vla-eval test --validate
/root/physical-ai/envs/vla_eval_py311/bin/vla-eval test --benchmark maniskill2 --timeout 30 --verbose
```

Result:

| Check | Result |
| --- | --- |
| RunPod Docker | not installed: `docker: command not found` |
| RunPod `uv` | not installed |
| `vla-eval==0.2.0` isolated install | passed in `/root/physical-ai/envs/vla_eval_py311` |
| source checkout | passed at `/workspace/physical-ai/vla-evaluation-harness`, tag `v0.2.0` |
| config validation | passed: `155/155 configs valid` |
| server inventory | 16 configs, but blocked for smoke because `uv` is missing |
| benchmark inventory | 17 configs, but blocked for smoke because Docker is missing |
| ManiSkill2 benchmark smoke | skipped, not failed: `docker not found on PATH` |

Installation footprint:

| Path | Size |
| --- | ---: |
| `/workspace/physical-ai/vla-evaluation-harness` | 124M |
| `/root/physical-ai/envs/vla_eval_py311` | 230M |

Artifacts:

| Artifact | Path |
| --- | --- |
| harness inventory log | `_workspace/runpod_results/vla_eval_harness_audit_20260607/test_list.log` |
| harness validation log | `_workspace/runpod_results/vla_eval_harness_audit_20260607/validate.log` |
| ManiSkill2 benchmark smoke blocker log | `_workspace/runpod_results/vla_eval_harness_audit_20260607/maniskill2_benchmark_smoke.log` |

Relevant harness configs observed:

| Benchmark | Config | Harness scope |
| --- | --- | --- |
| ManiSkill2 | `configs/benchmarks/maniskill2/eval.yaml` | `PickCube-v0`, `StackCube-v0`, `PickSingleYCB-v0`, `PickSingleEGAD-v0`, `PickClutterYCB-v0`; `50` episodes/task; `max_steps=400`; `base_camera` |
| RoboCasa | `configs/benchmarks/robocasa/eval.yaml` | Dockerized RoboCasa quick eval config; README describes full RoboCasa evaluation as `24` tasks x `50` episodes/task |
| SimplerEnv | `configs/benchmarks/simpler/*.yaml` | Google Robot and WidowX task configs |
| CALVIN | `configs/benchmarks/calvin/eval.yaml` | chained language-conditioned benchmark config |

Interpretation:

This is a useful infrastructure lane, but not yet a direct SmolVLA number. The
current RunPod image can validate the harness and inspect configs, but cannot
execute Dockerized benchmark smoke tests. To use this for paper-comparable
non-LIBERO numbers, the next cloud setup needs Docker-enabled RunPod or a
compatible host image, plus a SmolVLA model-server adapter because upstream
`v0.2.0` does not ship one.

Interpretation:

The non-strict CP25 run is allowed to pass with a documented missing-dependency
blocker. The strict RunPod gate now imports `robocasa` and `robosuite`, creates
a `CloseFridge` environment through `robocasa.utils.env_utils.create_env()`,
resets it, and executes at least one zero-action step. This is an installation
and runtime proof, not a policy score.

### 6. RoboCasa `CloseFridge` SmolVLA 20-episode evaluation

Command shape:

```bash
PYTHONPATH=/workspace/physical-ai/robocasa:/workspace/physical-ai/robosuite:/workspace/physical-ai/vendor/lerobot/src:$PYTHONPATH \
/root/physical-ai/envs/lerobot_py312/bin/python -m lerobot.scripts.lerobot_eval \
  --output_dir _workspace/runpod_results/robocasa_smolvla_closefridge_20ep_default_horizon_20260607T063345Z \
  --policy.path=lerobot/smolvla_robocasa \
  --env.type=robocasa \
  --env.task=CloseFridge \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --rename_map='{"observation.images.robot0_agentview_left":"observation.images.camera1","observation.images.robot0_eye_in_hand":"observation.images.camera2","observation.images.robot0_agentview_right":"observation.images.camera3"}' \
  --seed=0
```

Result:

| Field | Value |
| --- | --- |
| policy | `lerobot/smolvla_robocasa` |
| task | `CloseFridge` |
| episodes | 20 |
| horizon | default LeRobot/RoboCasa `1000` |
| successes | 0/20 |
| success rate | 0.0% |
| avg sum reward | 0.0 |
| eval seconds | 821.637 |
| eval seconds per episode | 41.082 |
| metrics artifact | `_workspace/runpod_results/robocasa_smolvla_closefridge_20ep_default_horizon_20260607T063345Z/eval_info.json` |
| representative video | `_workspace/runpod_results/robocasa_smolvla_closefridge_20ep_default_horizon_20260607T063345Z/videos/CloseFridge_0/eval_episode_0.mp4` |

Visual check:

- Representative frame inspected locally:
  `_workspace/runpod_results/robocasa_smolvla_closefridge_20ep_default_horizon_20260607T063345Z/videos/CloseFridge_0/frame_000.png`
- The frame shows a valid RoboCasa kitchen/fridge scene with the robot arm, not
  a blank or broken render.

Interpretation:

This is a real non-LIBERO SmolVLA policy evaluation and follows the LeRobot
single-task evaluation shape for `CloseFridge`: `lerobot/smolvla_robocasa`,
`env.type=robocasa`, `eval.n_episodes=20`, and the three-camera rename map.
It is not the same as the RoboCasa365 leaderboard because the leaderboard is a
50-task multi-task benchmark over three splits. Treat it as the first
official-protocol single-task number and a scale-up gate.

## RoboCasa Comparison Table

| Evaluation | Policy | Scale | Metric | Our Number | Reference Number | Delta / Boundary |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Single-task quick iteration | `lerobot/smolvla_robocasa` | `CloseFridge`, 20 episodes | task success | 0.0% | LeRobot documents this as the recommended quick-iteration command, but does not publish a success target on the docs page | protocol match, no public target |
| RoboCasa365 leaderboard | RLDX-1 | 50 tasks, 3 splits | overall task success | n/a | 33.2 | not directly comparable to single-task |
| RoboCasa365 leaderboard | GR00T N1.5 | 50 tasks, 3 splits | overall task success | n/a | 23.9 | not directly comparable to single-task |
| RoboCasa365 leaderboard | pi0.5 | 50 tasks, 3 splits | overall task success | n/a | 16.9 | not directly comparable to single-task |
| RoboCasa365 leaderboard | pi0 | 50 tasks, 3 splits | overall task success | n/a | 14.8 | not directly comparable to single-task |
| RoboCasa365 leaderboard | Diffusion Policy | 50 tasks, 3 splits | overall task success | n/a | 6.1 | not directly comparable to single-task |

## Recommendation

Next executable path:

1. Use RunPod only if the selected Pod exposes a working NVIDIA Vulkan/SAPIEN
   renderer for ManiSkill. Validate with strict `PickCube-v1`, no fallback.
2. If strict `PickCube-v1` works, run a small trained-policy-compatible
   ManiSkill table over `PickCube`, `PushCube`, `StackCube`,
   `PegInsertionSide`, and `PlugCharger` only if model/action compatibility is
   available.
3. Expand RoboCasa from the completed `CloseFridge` 20ep run to a small
   multi-task subset or `atomic_seen`.
4. Do not compare random/zero pilots to trained-policy paper numbers except as
   sanity controls. A fair table needs either the same policy family or an
   explicitly labeled weak-control row.

## Claim Boundary

Current non-LIBERO state is **partially paper-facing but not yet leaderboard
comparable**:

- ManiSkill/HAB: local pilots and current renderer blocker are documented.
- RoboCasa: CP25 strict reset/step passed on RunPod, and
  `lerobot/smolvla_robocasa` ran on `CloseFridge` for 20 episodes with a
  measured `0/20` success rate.
- RoboCasa365: full leaderboard comparability still requires 20 episodes per
  task across the target benchmark groups, not just the single `CloseFridge`
  task.
