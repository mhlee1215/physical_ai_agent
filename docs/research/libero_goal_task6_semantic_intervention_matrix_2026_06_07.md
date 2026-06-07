# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task6_semantic_probe_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task6_semantic_probe_none | false | 0.00 | 300 | +0 | 56 | 0 | 1 | 10.0911 | +0.0000 | 0.000000 | 0.000000 |
| task6_semantic_reach_g2 | false | 0.00 | 300 | +0 | 174 | 174 | 1 | 10.4492 | +0.3581 | 0.000000 | 0.000000 |
| task6_semantic_reach_g4 | false | 0.00 | 300 | +0 | 203 | 203 | 1 | 10.1006 | +0.0095 | 0.000000 | 0.000000 |
| task6_semantic_reach_g4_open | false | 0.00 | 300 | +0 | 195 | 195 | 1 | 10.2072 | +0.1161 | 0.000000 | 0.000000 |
| task6_late_probe_none | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.0566 | -0.0346 | 0.000000 | 0.000000 |
| task6_late_push_g5_close | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 10.2540 | +0.1628 | 0.000000 | 0.000000 |
| task6_late_push_g10_close | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 10.2287 | +0.1376 | 0.000000 | 0.000000 |
| task6_late_push_g5_open | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 10.0633 | -0.0279 | 0.000000 | 0.000000 |
| task6_late_reach_g2_open | false | 0.00 | 300 | +0 | 31 | 31 | 1 | 10.1579 | +0.0668 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 300.00 | 10.1787 | 0.000000 | 0.000000 |

## Semantic Diagnostics

Task `6` is `put the cream cheese in the bowl`. The semantic probe confirmed
that LeRobot/LIBERO exposes raw object state inside the rollout:
`cream_cheese_1_pos`, `akita_black_bowl_1_pos`, `robot0_eef_pos`, and
`robot0_gripper_qpos`.

| Condition | Min target-to-bowl step | Min target-to-bowl | Final target-to-bowl | Final eef-to-target | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `late_probe_none` | 229 | 0.062636 | 0.062821 | 0.080481 | Baseline gets the object near the bowl, then stalls. |
| `late_reach_g2_open` | 229 | 0.062636 | 0.062821 | 0.047316 | The intervention moves the gripper closer to the target but does not move the object into the bowl. |
| `early_reach_g2` | 299 | 0.072914 | 0.072914 | 0.047005 | Early reach intervention hurts placement distance while improving gripper-target proximity. |

## Current Interpretation

- Small semantic interventions found a real failure phase: late placement
  stalls after the target gets near the bowl.
- The interventions can change robot pose (`eef_to_target` improved in
  `late_reach_g2_open`) but did not improve object placement or benchmark
  success.
- This is not yet a positive agentic result. It narrows the next intervention
  target from generic retry/reach to contact-aware placement or release.

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
