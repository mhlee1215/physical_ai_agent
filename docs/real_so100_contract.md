# Real SO-100 Contract Module

The current real SO-100 path is centralized in:

```text
src/physical_ai_agent/real_so100/contract.py
```

Use this module before adding or changing real-robot camera, action, or
execution logic. When older SO101/SO-100 code conflicts with this file, this
contract is the current source of truth.

## Current Camera Routing

| Index / name | Role | SmolVLA policy input |
| --- | --- | --- |
| real camera `0` | wrist/end-effector policy view | yes |
| real camera `1` | object/context policy view | yes |
| real camera `3` | Codex observer/debug evidence | no |
| sim `wrist_cam` | policy view equivalent | yes |
| sim `egocentric_cam` | policy context equivalent | yes |
| sim `top_down` | debug/observer equivalent | no |

Legacy camera index `2` is not part of the current policy loop.

## Execution Defaults

- Default action chunk length: `10`
- Joint order:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper`
- Physical execution stays blocked unless calibration, observer evidence,
  SmolVLA postprocessing/unnormalization, clipping, home-return, and torque-off
  requirements are satisfied.

## Tests

Run the dependency-light contract tests with:

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_real_so100_contract
```

The older SO101 sim checkpoint tests still exist, but CP17 now follows the
latest split: `wrist_cam` and `egocentric_cam` are policy inputs; `top_down` is
debug/observer only.
