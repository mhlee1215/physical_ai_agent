# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task6_late_probe_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task6_late_probe_none | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.0566 | +0.0000 | 0.000000 | 0.000000 |
| task6_place_c06_p5_zm02_close | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.2295 | +0.1729 | 0.000000 | 0.000000 |
| task6_place_c08_p5_zm02_close | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.6021 | +0.5455 | 0.000000 | 0.000000 |
| task6_place_c08_p8_zm02_close | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.6088 | +0.5523 | 0.000000 | 0.000000 |
| task6_place_c08_p5_zm05_close | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.6344 | +0.5779 | 0.000000 | 0.000000 |
| task6_place_c08_p5_zm02_open | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.4206 | +0.3641 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.00 | 300.00 | 10.4253 | 0.000000 | 0.000000 |

## Semantic Diagnostics

The `semantic_place_receptacle` macro activates once the target object is near
the receptacle, then applies a contact-seeking placement action. This is an
in-episode subgoal intervention: it does not reset the environment.

| Condition | Min target-to-bowl step | Min target-to-bowl | Final target-to-bowl | Final EEF-to-target | Diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| `task6_late_probe_none` | 229 | 0.062636 | 0.062821 | 0.080481 | Baseline stalls near the bowl and fails. |
| `task6_place_c06_p5_zm02_close` | 288 | 0.058806 | 0.059147 | 0.067679 | Clear object-placement improvement, but no benchmark success. |
| `task6_place_c08_p5_zm02_close` | 230 | 0.062426 | 0.062512 | 0.078382 | Small early placement improvement. |
| `task6_place_c08_p8_zm02_close` | 296 | 0.060582 | 0.060595 | 0.079968 | Moderate placement improvement. |
| `task6_place_c08_p5_zm05_close` | 224 | 0.055952 | 0.056542 | 0.060363 | Best placement metric: about 10.7% lower min target-to-bowl distance than baseline. |
| `task6_place_c08_p5_zm02_open` | 292 | 0.062648 | 0.062661 | 0.073316 | Opening gripper does not improve placement. |

## Current Interpretation

- None of these conditions recover benchmark success or reduce action-step
  cost; task-level success remains `0%`.
- The placement macro gives the strongest subgoal-level evidence so far:
  best min target-to-bowl distance improves from `0.062636` to `0.055952`.
- This is not a paper-facing success-rate result yet. It is evidence that the
  in-episode semantic wrapper can improve an intermediate physical metric.
- Next candidate: keep the successful placement macro but search over timing
  and placement parameters, or run it on adjacent weak seeds/tasks where the
  success threshold may be closer.

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
