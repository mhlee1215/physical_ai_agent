# Meta-World Public LeRobot Full MT50 Result - 2026-06-08

## Summary

Status: **completed**.

This run evaluated the public `lerobot/smolvla_metaworld` checkpoint on the full
LeRobot Meta-World MT50 suite using a fresh public LeRobot clone on RunPod.

The run is comparable in scale to the SmolVLA paper's Meta-World Table 2 row:
all 50 Meta-World tasks, grouped as Easy/Medium/Hard/Very Hard, with 10
episodes per task for 500 total episodes.

Important caveat: the paper table reports the arithmetic mean of the four split
success rates. LeRobot also reports an episode-weighted `overall.pc_success`.
Because the split sizes differ (`28/11/6/5` tasks), these two averages are not
the same.

## Evidence

Local evidence root:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_20260608T0650Z/metaworld_public_full_mt50_10ep_20260608T0650Z/
```

Key files:

- `eval/eval_info.json`: authoritative metrics.
- `eval.log`: full `lerobot-eval` log.
- `environment_probe.txt`: Python/package/CUDA probe.
- `run_command.txt`: exact evaluation command.
- `lerobot_commit.txt`: public LeRobot commit.
- `summary.txt`: compact JSON summary.
- `frames/`: locally extracted representative frames from videos.

Visual artifact inspected:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_20260608T0650Z/metaworld_public_full_mt50_10ep_20260608T0650Z/frames/eval_videos_very_hard_4_eval_episode_9_frame30.png
```

The frame shows a valid Meta-World Sawyer tabletop scene with robot arm, table,
object, and target. It is not a blank renderer artifact.

## Runtime

RunPod:

- Pod ID: `r6jzvw1osez5pm`
- GPU: NVIDIA GeForce RTX 4090
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Output was written under `/tmp` and then fetched locally.

Python/package probe:

```text
python 3.12.13
lerobot 0.5.2
torch 2.11.0
metaworld 3.0.0
gymnasium 1.3.0
mujoco 3.9.0
cuda_available True
cuda_device NVIDIA GeForce RTX 4090
```

LeRobot source:

```text
commit: 09808183ca72c30cbb41b653586f6d0632a4bcca
describe: v0.5.1-115-g09808183
status: clean
```

## Command

```bash
MUJOCO_GL=egl lerobot-eval \
  --output_dir="/tmp/metaworld_public_repro_results/metaworld_public_full_mt50_10ep_20260608T0650Z/eval" \
  --policy.path=lerobot/smolvla_metaworld \
  --env.type=metaworld \
  --env.task=easy,medium,hard,very_hard \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.empty_cameras=0 \
  --rename_map='{"observation.image":"observation.images.camera1"}' \
  --seed=0
```

## Result

Exit code: `0`.

LeRobot weighted overall:

| Metric | Value |
| --- | ---: |
| Episodes | 500 |
| `overall.pc_success` | 55.00 |
| `overall.eval_s` | 1094.41 |
| `overall.eval_ep_s` | 2.19 |

Paper-style split table:

| Meta-World split | Episodes | Ours success % | SmolVLA 0.45B Table 2 % | Delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 280 | 68.21 | 82.50 | -14.29 |
| Medium | 110 | 52.73 | 41.80 | +10.93 |
| Hard | 60 | 23.33 | 45.00 | -21.67 |
| Very Hard | 50 | 24.00 | 60.00 | -36.00 |
| Arithmetic split avg | 500 | 42.07 | 57.30 | -15.23 |

The run is closest to the paper row on Medium and still farthest on Very Hard.
The environment path is now fixed and full-scale executable; the remaining gap
is a benchmark-parity gap, not an environment-start failure.

## Follow-Up: `n_action_steps=15`

Status: **completed**.

After the default-checkpoint run above, we reran the same public LeRobot MT50
protocol with `--policy.n_action_steps=15`.

Local evidence root:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/
```

Representative frames were visually inspected from all four splits:

```text
frames/eval_videos_easy_0_eval_episode_0.mp4_mid.png
frames/eval_videos_medium_0_eval_episode_0.mp4_mid.png
frames/eval_videos_hard_0_eval_episode_0.mp4_mid.png
frames/eval_videos_very_hard_0_eval_episode_0.mp4_mid.png
```

All four frames show valid Meta-World tabletop scenes with robot arm, task
objects, and targets; none are blank renderer artifacts.

Runtime caveat: this second run used public LeRobot `main` resolving to
`torch 2.11.0`. The RunPod driver exposed CUDA 12.8, so the environment probe
reported `cuda_available False` after a CUDA-driver warning. The evaluation
still completed successfully, but runtime was slower (`7248.79s`) than the
earlier default run (`1094.41s`). This should not change the deterministic
success metric, but it is relevant for reproducibility and cost.

Command delta:

```bash
--policy.n_action_steps=15
```

LeRobot weighted overall:

| Metric | Default/checkpoint run | `n_action_steps=15` |
| --- | ---: | ---: |
| Episodes | 500 | 500 |
| `overall.pc_success` | 55.00 | 65.20 |
| `overall.eval_s` | 1094.41 | 7248.79 |
| `overall.eval_ep_s` | 2.19 | 14.50 |

Paper-style split comparison:

| Meta-World split | SmolVLA 0.45B Table 2 % | Default/checkpoint % | `n_action_steps=15` % | Delta vs paper |
| --- | ---: | ---: | ---: | ---: |
| Easy | 82.50 | 68.21 | 78.21 | -4.29 |
| Medium | 41.80 | 52.73 | 58.18 | +16.38 |
| Hard | 45.00 | 23.33 | 40.00 | -5.00 |
| Very Hard | 60.00 | 24.00 | 38.00 | -22.00 |
| Arithmetic split avg | 57.30 | 42.07 | 53.60 | -3.70 |

Interpretation:

- `n_action_steps=15` materially improves the public LeRobot reproduction:
  `+10.20pp` weighted overall and `+11.53pp` paper-style split average versus
  the default/checkpoint run.
- The gap to the SmolVLA paper Table 2 row narrows from `-15.23pp` to
  `-3.70pp` on the arithmetic split average.
- The remaining gap is concentrated in Very Hard (`-22.00pp`), while Medium is
  above the paper reference in this run.

## Follow-Up: CUDA-Pinned `n_action_steps=10` and `20`

Status: **completed**.

After the `n_action_steps=15` run exposed a CUDA-driver mismatch with
`torch 2.11.0`, we added optional torch/CUDA pin support to the RunPod runner
and reran full MT50 with CUDA-enabled `torch 2.5.1+cu124`.

Important caveat: current public LeRobot `0.5.2` declares
`torch>=2.7` and `torchvision>=0.22`. The CUDA-pinned runs intentionally used
`torch 2.5.1+cu124` and `torchvision 0.20.1+cu124`, so they are best treated
as speed/cost-oriented reproducibility checks, not a cleaner dependency match
than the public resolver. Both runs completed with exit code `0` and
`cuda_available True`.

Local evidence roots:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z/
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z/
```

Representative frames were visually inspected from all four splits for both
runs:

```text
frames/easy_0_eval_episode_0_mid.png
frames/medium_0_eval_episode_0_mid.png
frames/hard_0_eval_episode_0_mid.png
frames/very_hard_0_eval_episode_0_mid.png
```

All inspected frames show valid Meta-World tabletop scenes with robot arm, task
objects, and targets; none are blank renderer artifacts.

Runtime probe for both CUDA-pinned runs:

```text
lerobot 0.5.2
torch 2.5.1+cu124
torchvision 0.20.1+cu124
torchcodec 0.11.1
metaworld 3.0.0
gymnasium 1.3.0
mujoco 3.9.0
cuda_available True
cuda_device NVIDIA GeForce RTX 4090
```

Command deltas:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
TORCH_VERSION_SPEC='torch==2.5.1+cu124'
TORCHVISION_VERSION_SPEC='torchvision==0.20.1+cu124'
--policy.n_action_steps=10
--policy.n_action_steps=20
```

LeRobot weighted overall:

| Metric | Default/checkpoint | `n_action_steps=10` CUDA | `n_action_steps=15` | `n_action_steps=20` CUDA |
| --- | ---: | ---: | ---: | ---: |
| Episodes | 500 | 500 | 500 | 500 |
| `overall.pc_success` | 55.00 | 66.20 | 65.20 | 65.40 |
| `overall.eval_s` | 1094.41 | 2742.43 | 7248.79 | 1425.98 |
| `overall.eval_ep_s` | 2.19 | 5.48 | 14.50 | 2.85 |

Paper-style split comparison:

| Meta-World split | SmolVLA 0.45B Table 2 % | Default/checkpoint % | `n_action_steps=10` CUDA % | `n_action_steps=15` % | `n_action_steps=20` CUDA % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 82.50 | 68.21 | 82.86 | 78.21 | 80.36 |
| Medium | 41.80 | 52.73 | 50.91 | 58.18 | 52.73 |
| Hard | 45.00 | 23.33 | 43.33 | 40.00 | 43.33 |
| Very Hard | 60.00 | 24.00 | 34.00 | 38.00 | 36.00 |
| Arithmetic split avg | 57.30 | 42.07 | 52.77 | 53.60 | 53.10 |

Best current readings:

- Best weighted overall: `n_action_steps=10` CUDA at `66.20%`.
- Best paper-style arithmetic split average: `n_action_steps=15` at `53.60%`,
  `-3.70pp` below the paper's `57.30%` reference.
- The remaining paper-style gap is dominated by Very Hard. Easy/Medium/Hard are
  now close enough that the reproduction is useful as a baseline, but Very Hard
  is still materially below the paper row.

## Per-Task Success

Each task uses 10 episodes.

| Split | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 10/10 | 8/10 | 10/10 | 7/10 |
| Medium | 2/10 | 8/10 | 7/10 | 7/10 | 6/10 | 6/10 | 1/10 | 7/10 | 1/10 | 9/10 |
| Hard | 1/10 | 2/10 | 1/10 | 1/10 | 6/10 | 3/10 |  |  |  |  |
| Very Hard | 2/10 | 7/10 | 0/10 | 1/10 | 2/10 |  |  |  |  |  |

Easy has 28 tasks, so the remaining Easy tasks are listed separately:

| Split | Task 10 | Task 11 | Task 12 | Task 13 | Task 14 | Task 15 | Task 16 | Task 17 | Task 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 10/10 | 8/10 | 5/10 | 8/10 | 10/10 | 10/10 | 1/10 | 0/10 | 1/10 |

| Split | Task 19 | Task 20 | Task 21 | Task 22 | Task 23 | Task 24 | Task 25 | Task 26 | Task 27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 8/10 | 2/10 | 7/10 | 10/10 | 2/10 | 6/10 | 8/10 | 8/10 | 2/10 |

Medium has 11 tasks:

| Split | Task 10 |
| --- | ---: |
| Medium | 4/10 |

## Reference Links

- LeRobot Meta-World docs:
  <https://huggingface.co/docs/lerobot/en/metaworld>
- Public model card:
  <https://huggingface.co/lerobot/smolvla_metaworld>
- SmolVLA paper:
  <https://arxiv.org/abs/2506.01844>

The LeRobot docs specify the Meta-World split sizes, `--env.task` difficulty
groups, and 10 episodes per task for reproducible benchmarking. The SmolVLA
paper Table 2 provides the reference Meta-World numbers used above.
