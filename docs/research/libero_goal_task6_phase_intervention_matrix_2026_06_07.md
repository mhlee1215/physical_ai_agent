# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task6_late_probe_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task6_late_probe_none | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.0566 | +0.0000 | 0.000000 | 0.000000 |
| task6_phase_push_c06_g2p5_close | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 9.9652 | -0.0913 | 0.000000 | 0.000000 |
| task6_phase_push_c08_g2p5_close | false | 0.00 | 300 | +0 | 22 | 22 | 1 | 9.9412 | -0.1154 | 0.000000 | 0.000000 |
| task6_phase_push_c06_g2p10_close | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 10.2174 | +0.1608 | 0.000000 | 0.000000 |
| task6_phase_push_c06_g2p5_open | false | 0.00 | 300 | +0 | 30 | 30 | 1 | 10.2024 | +0.1458 | 0.000000 | 0.000000 |
| task6_phase_push_c08_g2p5_open | false | 0.00 | 300 | +0 | 25 | 25 | 1 | 10.1708 | +0.1142 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.00 | 300.00 | 10.0922 | 0.000000 | 0.000000 |

## Semantic Diagnostics

The phase intervention switches behavior after a semantic no-progress trigger:
first move EEF back toward the target object; once EEF-to-target is below the
configured contact threshold, push the target toward the receptacle.

| Condition | Min target-to-bowl step | Min target-to-bowl | Final target-to-bowl | Final EEF-to-target | Diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| `task6_late_probe_none` | 229 | 0.062636 | 0.062821 | 0.080481 | Baseline stalls after moving the object near the bowl. |
| `task6_phase_push_c06_g2p5_close` | 229 | 0.062636 | 0.062821 | 0.059052 | Improves final EEF-to-target but not object placement. |
| `task6_phase_push_c08_g2p5_close` | 298 | 0.062615 | 0.062632 | 0.086103 | Produces the first tiny object-placement improvement, but no benchmark success. |
| `task6_phase_push_c06_g2p10_close` | 229 | 0.062636 | 0.062821 | 0.059733 | Stronger push gain does not improve placement. |
| `task6_phase_push_c06_g2p5_open` | 229 | 0.062636 | 0.062832 | 0.061097 | Opening gripper slightly worsens final placement. |
| `task6_phase_push_c08_g2p5_open` | 229 | 0.062636 | 0.062867 | 0.084995 | Opening gripper worsens final placement. |

## Current Interpretation

- No condition recovered benchmark success or action-step efficiency on task
  `6`.
- `task6_phase_push_c08_g2p5_close` is the first condition that nudged the
  object-placement metric in the desired direction, from minimum target-to-bowl
  distance `0.062636` to `0.062615`.
- The improvement is too small to claim task-level benefit. It is useful only
  as a diagnostic sign that a contact-seeking phase intervention can affect
  object placement, while the current controller is still too weak for success.
- Next candidate: use raw contact information or a short learned/heuristic
  placement macro around steps `220-300`, rather than a single vector push.

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
