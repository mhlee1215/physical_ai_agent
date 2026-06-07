# Non-LIBERO Evaluation Status

Date: 2026-06-07

This note tracks evaluation options after freezing the LIBERO SmolVLA baseline.
The goal is to produce non-LIBERO numbers that can eventually be compared with
paper or official leaderboard values.

## Current Decision

The best next external benchmark is **RoboCasa365 / RoboCasa**, but the most
immediately executable non-LIBERO lane in this repo is **ManiSkill /
ManiSkill-HAB CP24**.

Why:

- ManiSkill/HAB already has repo-local execution plumbing and Mac-local pilot
  artifacts.
- RoboCasa365 is more paper-relevant for household long-horizon agentic
  wrappers, and now has a separate CP25 install/reset-step probe gate before
  full evaluation.
- RoboTwin/Isaac/SimplerEnv/CALVIN are plausible later lanes, but they require
  new model/action-space compatibility work before a fair SmolVLA comparison.

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

Expected artifacts:

| Artifact | Path |
| --- | --- |
| checkpoint report | `_workspace/checkpoints/checkpoint_25_robocasa/checkpoint_report.json` |
| blocker or no-blocker note | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa_blocker.md` |
| install/eval command handoff | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa_install_and_eval.md` |
| reference comparison table | `_workspace/checkpoints/checkpoint_25_robocasa/robocasa365_reference_table.md` |

Interpretation:

The non-strict CP25 run is allowed to pass with a documented missing-dependency
blocker. The strict run is the first real RoboCasa execution gate: it must
import `robocasa` and `robosuite`, create a `CloseFridge` environment through
`robocasa.utils.env_utils.create_env()`, reset it, and execute at least one
zero-action step. This is still not a paper-comparable score; it is the
installation/runtime gate before `lerobot/smolvla_robocasa` evaluation.

## Recommendation

Next executable path:

1. Use RunPod only if the selected Pod exposes a working NVIDIA Vulkan/SAPIEN
   renderer for ManiSkill. Validate with strict `PickCube-v1`, no fallback.
2. If strict `PickCube-v1` works, run a small trained-policy-compatible
   ManiSkill table over `PickCube`, `PushCube`, `StackCube`,
   `PegInsertionSide`, and `PlugCharger` only if model/action compatibility is
   available.
3. Run CP25 on RunPod. If the non-strict probe only records missing
   dependencies, install RoboCasa/robosuite plus lightweight assets and rerun
   the strict reset-step gate.
4. Do not compare random/zero pilots to trained-policy paper numbers except as
   sanity controls. A fair table needs either the same policy family or an
   explicitly labeled weak-control row.

## Claim Boundary

Current non-LIBERO state is **not yet paper-comparable**. It is a readiness
audit:

- ManiSkill/HAB: local pilots and current renderer blocker are documented.
- RoboCasa365: reference numbers are identified; CP25 probe gate exists, while
  strict reset-step and SmolVLA policy evaluation still require installed
  RoboCasa assets.
- Next required milestone: a strict non-LIBERO task-success run on a
  renderer-compatible ManiSkill machine, or a RoboCasa strict reset-step smoke
  followed by `lerobot/smolvla_robocasa` evaluation.
