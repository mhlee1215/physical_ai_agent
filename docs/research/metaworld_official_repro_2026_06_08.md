# Meta-World Official LeRobot Reproduction - 2026-06-08

## Latest Result

Status: **full public-code MT50 evaluation succeeded on RunPod**.

Latest completed paper-style runs: `n_action_steps=10`, `15`, and `20`.

Current best paper-style reproduction remains `n_action_steps=15`, because it
has the smallest arithmetic split-average gap to the SmolVLA Table 2 reference.
The `n_action_steps=10` CUDA-pinned run has the best LeRobot weighted overall.

The full-scale run used the same public-code environment fix as the minimal
run, then evaluated `lerobot/smolvla_metaworld` on all LeRobot Meta-World
difficulty groups with 10 episodes per task:

```text
easy,medium,hard,very_hard
```

Local evidence for the original default/checkpoint run:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_20260608T0650Z/metaworld_public_full_mt50_10ep_20260608T0650Z/
```

Local evidence for the `n_action_steps=15` rerun:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/
```

Local evidence for CUDA-pinned `n_action_steps=10` and `20` reruns:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z/
_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z/
```

Detailed report:

```text
docs/research/metaworld_public_full_mt50_2026_06_08.md
```

Full MT50 result, default/checkpoint action horizon:

| Meta-World split | Episodes | Ours success % | SmolVLA 0.45B Table 2 % | Delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 280 | 68.21 | 82.50 | -14.29 |
| Medium | 110 | 52.73 | 41.80 | +10.93 |
| Hard | 60 | 23.33 | 45.00 | -21.67 |
| Very Hard | 50 | 24.00 | 60.00 | -36.00 |
| Arithmetic split avg | 500 | 42.07 | 57.30 | -15.23 |

LeRobot's episode-weighted `overall.pc_success` is `55.00%` over 500 episodes.
The arithmetic split average is the paper-style comparison number because the
four difficulty groups have different task counts.

Full MT50 result, `--policy.n_action_steps=15`:

| Meta-World split | Episodes | Ours success % | SmolVLA 0.45B Table 2 % | Delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 280 | 78.21 | 82.50 | -4.29 |
| Medium | 110 | 58.18 | 41.80 | +16.38 |
| Hard | 60 | 40.00 | 45.00 | -5.00 |
| Very Hard | 50 | 38.00 | 60.00 | -22.00 |
| Arithmetic split avg | 500 | 53.60 | 57.30 | -3.70 |

LeRobot's episode-weighted `overall.pc_success` for this rerun is `65.20%`
over 500 episodes. Compared with the default/checkpoint run, the
`n_action_steps=15` setting improves the paper-style arithmetic split average
by `+11.53pp`.

Runtime caveat: the `n_action_steps=15` rerun resolved to public LeRobot `main`
with `torch 2.11.0`; on this RunPod image, the CUDA driver was too old for that
wheel and the environment probe reported `cuda_available False`. The benchmark
still completed with exit code `0`, but took `7248.79s`.

Full MT50 result, CUDA-pinned `--policy.n_action_steps=10`:

| Meta-World split | Episodes | Ours success % | SmolVLA 0.45B Table 2 % | Delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 280 | 82.86 | 82.50 | +0.36 |
| Medium | 110 | 50.91 | 41.80 | +9.11 |
| Hard | 60 | 43.33 | 45.00 | -1.67 |
| Very Hard | 50 | 34.00 | 60.00 | -26.00 |
| Arithmetic split avg | 500 | 52.77 | 57.30 | -4.53 |

LeRobot's episode-weighted `overall.pc_success` is `66.20%` over 500 episodes.
The environment probe reports `torch 2.5.1+cu124` and `cuda_available True`.

Full MT50 result, CUDA-pinned `--policy.n_action_steps=20`:

| Meta-World split | Episodes | Ours success % | SmolVLA 0.45B Table 2 % | Delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 280 | 80.36 | 82.50 | -2.14 |
| Medium | 110 | 52.73 | 41.80 | +10.93 |
| Hard | 60 | 43.33 | 45.00 | -1.67 |
| Very Hard | 50 | 36.00 | 60.00 | -24.00 |
| Arithmetic split avg | 500 | 53.10 | 57.30 | -4.20 |

LeRobot's episode-weighted `overall.pc_success` is `65.40%` over 500 episodes.
The environment probe reports `torch 2.5.1+cu124` and `cuda_available True`.

CUDA pin caveat: public LeRobot `0.5.2` declares `torch>=2.7`, while the CUDA
pin used `torch 2.5.1+cu124` to match the RunPod driver and restore GPU
execution. These runs are valid completed rollouts, but the `n_action_steps=15`
run is still the cleaner public-resolver comparison despite its slow CPU
runtime.

Representative visual frame inspected locally:

```text
_workspace/runpod_results/metaworld_public_full_mt50_10ep_20260608T0650Z/metaworld_public_full_mt50_10ep_20260608T0650Z/frames/eval_videos_very_hard_4_eval_episode_9_frame30.png
```

The frame shows a valid Meta-World Sawyer tabletop scene with the robot arm,
table, object, and target.

## Earlier Minimal Result

Status: **minimum public-code Meta-World evaluation succeeded on RunPod**.

The successful run used a fresh public LeRobot clone from GitHub, installed
`lerobot[smolvla,metaworld]`, did **not** apply the stale
`gymnasium==1.1.0` downgrade, and completed one `assembly-v3` episode with the
public `lerobot/smolvla_metaworld` checkpoint.

This proves the environment path is runnable. The single episode did not
succeed at the task, so the score is not useful as a benchmark number.

Local evidence:

```text
_workspace/runpod_results/metaworld_public_minimal_no_gym_pin_20260608T0625Z/
```

Representative video frame inspected locally:

```text
_workspace/runpod_results/metaworld_public_minimal_no_gym_pin_20260608T0625Z/frames/assembly_v3_frame30.png
```

The frame shows a valid Meta-World Sawyer tabletop scene with the robot arm and
task object; it is not a blank renderer artifact.

### Successful Run Environment

RunPod:

- Pod ID: `3ix18qgdxc4v21`
- GPU: NVIDIA GeForce RTX 4090
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Result output was written under `/tmp`, not `/workspace`, to avoid network
  volume quota failures.

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

Command:

```bash
MUJOCO_GL=egl lerobot-eval \
  --output_dir="/tmp/metaworld_public_minimal_results/metaworld_public_minimal_no_gym_pin_20260608T0625Z/eval" \
  --policy.path=lerobot/smolvla_metaworld \
  --env.type=metaworld \
  --env.task=assembly-v3 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.empty_cameras=0 \
  --rename_map='{"observation.image":"observation.images.camera1"}' \
  --seed=0
```

Metrics:

```text
exit_code: 0
n_episodes: 1
pc_success: 0.0
avg_sum_reward: 981.219751207511
avg_max_reward: 6.94472599029541
eval_s: 5.140444040298462
```

### What Fixed The Environment

The critical fix was **not** pinning Gymnasium to `1.1.0`.

The previous failed attempt followed the current LeRobot docs' workaround and
forced `gymnasium==1.1.0`; with `metaworld==3.0.0` and `mujoco==3.9.0`, that
still failed at the render mode assertion. Letting current public LeRobot's
dependency resolver install `gymnasium==1.3.0` allowed environment construction
and rollout to complete.

Also, write results/logs to `/tmp` first. The network volume currently has quota
pressure and failed when logs or full environment copies were written directly
under `/workspace`.

### Repeat Command

Use the durable script:

```bash
RUN_ID=metaworld_public_minimal_no_gym_pin_$(date -u +%Y%m%dT%H%M%SZ) \
TASK=assembly-v3 \
EPISODES=1 \
OUT=/tmp/metaworld_public_repro_results/$RUN_ID \
sh scripts/runpod_smolvla_metaworld_official_repro.sh
```

For future scale-up, increase `EPISODES` or set
`TASK=easy,medium,hard,very_hard`, but keep the no-Gymnasium-downgrade rule
unless upstream changes.

## Summary

This run attempted to reproduce Meta-World SmolVLA evaluation using the public
LeRobot code path and the public `lerobot/smolvla_metaworld` checkpoint on a
RunPod RTX 4090.

Earlier result: **evaluation did not start**. The run failed during Meta-World
environment construction, before any episode rollout or success metric could be
computed.

This earlier failed run is preserved below because it documents the stale
Gymnasium workaround that should not be repeated.

## Evidence

Local evidence bundle:

```text
_workspace/runpod_results/metaworld_official_repro/metaworld_official_v051_evidence/
```

Key files:

- `repro.log`: full installation log and official Gymnasium pin.
- `eval_medium.log`: LeRobot evaluation config and traceback.
- `environment_probe.txt`: Python/package/CUDA versions.
- `run_command_medium.txt`: exact evaluation command.
- `lerobot_commit.txt`: exact LeRobot commit.
- `lerobot_describe.txt`: exact tag.
- `exit_code_medium.txt`: process exit code.

## Official Reference Used

LeRobot Meta-World docs:

```text
https://huggingface.co/docs/lerobot/metaworld
```

The docs describe Meta-World evaluation through:

```bash
lerobot-eval \
  --policy.path="your-policy-id" \
  --env.type=metaworld \
  --env.task=medium \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

The docs also state that if `AssertionError: ['human', 'rgb_array',
'depth_array']` appears, install:

```bash
pip install "gymnasium==1.1.0"
```

## Run Environment

Pod:

- RunPod Pod ID: `bd7p82wu1yn2xg`
- GPU: NVIDIA GeForce RTX 4090
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Network volume: `tchm4gxfvd`

Python/package probe:

```text
python 3.12.13
lerobot 0.5.1
torch 2.10.0
metaworld 3.0.0
gymnasium 1.1.0
mujoco 3.9.0
cuda_available True
cuda_device NVIDIA GeForce RTX 4090
```

LeRobot source:

```text
tag: v0.5.1
commit: 1396b9fab7aecddd10006c33c47a487ffdcb54b4
```

## Exact Command

The run used the official default medium split with the SmolVLA checkpoint and
the necessary rename map for this checkpoint's camera feature name:

```bash
MUJOCO_GL=egl lerobot-eval \
  --output_dir="/tmp/official_lerobot_metaworld_repro/metaworld_official_v051_medium_20260608T0604Z/results" \
  --policy.path=lerobot/smolvla_metaworld \
  --env.type=metaworld \
  --env.task=medium \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --rename_map='{"observation.image":"observation.images.camera1"}'
```

## Failure

The run failed while building the first medium Meta-World task:

```text
Creating Meta-World envs | task_groups=['medium'] | n_envs(per task)=1
Building vec env | group=medium | task_id=0 | task=basketball-v3
...
File ".../gymnasium/envs/mujoco/mujoco_env.py", line 85, in __init__
    assert self.metadata["render_modes"] == [
AssertionError: ['human', 'rgb_array', 'depth_array']
```

Exit code:

```text
1
```

## Debug Chronology

1. `main` branch fresh clone was attempted first.
   - LeRobot commit was `v0.5.1-115-g09808183`.
   - It installed and reached Meta-World environment construction.
   - First blocker was missing EGL/OpenGL system libraries:
     `libEGL.so.0`, `libOpenGL.so.0`.

2. System rendering libraries were installed on the Pod.
   - Installed: `libegl1`, `libopengl0`, `libgl1`, `libosmesa6`,
     `libglfw3`.
   - This moved the failure past the EGL loader.

3. `main` branch plus `gymnasium==1.1.0` still failed with the same
   render mode assertion.

4. `gymnasium==1.0.0` was tested as a fallback.
   - This avoided the render mode assertion shape, but broke MetaWorld import:
     `module 'gymnasium.vector' has no attribute 'AutoresetMode'`.
   - Therefore `gymnasium==1.0.0` is not viable with `metaworld==3.0.0`.

5. Fresh official-docs-shaped run was created at LeRobot `v0.5.1`.
   - Installed `lerobot[smolvla,metaworld]`.
   - Applied official documented fix: `gymnasium==1.1.0`.
   - Still failed at the same render mode assertion before rollout.

## Interpretation

The current official LeRobot Meta-World docs and current package resolution do
not produce a runnable Meta-World evaluation environment on this RunPod image.
The failure happens before policy inference and before benchmark scoring, so no
success-rate comparison against SmolVLA reference numbers can be made from this
official-code run.

The likely issue is a version mismatch among:

- LeRobot `v0.5.1`
- MetaWorld `3.0.0`
- Gymnasium `1.1.0`
- MuJoCo `3.9.0`

The important practical point is that the public instruction "install
Gymnasium 1.1.0" did not fix the assertion in this environment.

## Do Not Repeat

- Do not copy a full venv or full LeRobot checkout to the network volume for
  failure evidence. It can exceed the volume quota. Preserve small evidence
  files only.
- Do not treat this as a 0% benchmark score. It is an environment construction
  failure.
- Do not use `gymnasium==1.0.0` with `metaworld==3.0.0`; MetaWorld import fails.
- Do not use unquoted `--rename_map` JSON through nested SSH commands. Use a
  remote script file or an environment variable.

## Next Viable Options

1. Open or search LeRobot/Farama issues for the exact render mode assertion
   against `metaworld==3.0.0` and `gymnasium==1.1.x`.
2. If paper parity matters more than official-code purity, use the already
   working repo-local Meta-World evaluator path recorded in
   `docs/evaluation_candidate_matrix_after_libero.md`, then report it as
   repo-patched/reproduction-engineering rather than official-docs-only.
3. Try an older LeRobot docs version and dependency lock if the goal is exact
   historical reproduction.
