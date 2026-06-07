# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task6_late_probe_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task6_late_probe_none | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.0566 | +0.0000 | 0.000000 | 0.000000 |
| task6_tune_s220_zm07 | true | 100.00 | 224 | -76 | 4 | 4 | 1 | 9.3361 | -0.7204 | 0.004464 | 6.426655 |
| task6_tune_s220_zm10 | true | 100.00 | 224 | -76 | 4 | 4 | 1 | 9.3117 | -0.7449 | 0.004464 | 6.443521 |
| task6_tune_s224_zm05 | false | 0.00 | 300 | +0 | 76 | 76 | 1 | 10.6559 | +0.5993 | 0.000000 | 0.000000 |
| task6_tune_s224_zm07 | false | 0.00 | 300 | +0 | 76 | 76 | 1 | 10.4298 | +0.3733 | 0.000000 | 0.000000 |
| task6_tune_s228_zm05 | false | 0.00 | 300 | +0 | 72 | 72 | 1 | 10.6671 | +0.6105 | 0.000000 | 0.000000 |
| task6_tune_s220_p3_zm05 | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 10.6523 | +0.5958 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 28.57 | 278.29 | 10.1585 | 0.001276 | 1.838597 |

## Semantic Diagnostics

Task `6` is `put the cream cheese in the bowl`. The no-reset intervention
uses raw LIBERO object state during rollout:

- target: `cream_cheese_1_pos`
- receptacle: `akita_black_bowl_1_pos`
- robot end-effector: `robot0_eef_pos`

| Condition | Success | First trigger step | Interventions | Min target-to-bowl | Final target-to-bowl | Final EEF-to-target | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `task6_late_probe_none` | false | 269 | 0 | 0.062636 | 0.062821 | 0.080481 | Baseline reaches the bowl area but stalls and times out. |
| `task6_tune_s220_zm07` | true | 220 | 4 | 0.058218 | 0.058218 | 0.057050 | Placement macro recovers success with 76 fewer action steps. |
| `task6_tune_s220_zm10` | true | 220 | 4 | 0.053997 | 0.053997 | 0.052318 | Stronger downward placement also recovers success with best placement metric. |
| `task6_tune_s224_zm07` | false | 224 | 76 | 0.056620 | 0.057507 | 0.062519 | Improves placement but triggers too late to satisfy the benchmark. |
| `task6_tune_s220_p3_zm05` | false | 220 | 80 | 0.057160 | 0.061232 | 0.059635 | Weaker push improves placement transiently but does not recover success. |

## Current Interpretation

- This is the first positive in-episode intervention result in this run:
  `task6_late_probe_none` fails at `300` action steps, while two
  `semantic_place_receptacle` variants succeed at `224` action steps.
- The intervention does not reset the environment. It detects a semantic
  near-receptacle condition during rollout and switches to a placement subgoal.
- Cost-normalized result:
  - success improves from `false` to `true`
  - action steps improve from `300` to `224` (`-76`)
  - eval seconds improve from `10.0566` to `9.3117-9.3361`
  - environment resets remain `1` in both rows
- Claim boundary: this is a single task/seed positive result, not a paper-scale
  benchmark. It supports a narrow claim that an in-episode semantic wrapper can
  recover at least one SmolVLA/LIBERO failure case under fixed environment
  reset budget.

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
