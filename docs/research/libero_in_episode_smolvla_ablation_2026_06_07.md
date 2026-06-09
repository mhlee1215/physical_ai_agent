# LIBERO In-Episode Intervention Ablation Report

## Conditions

| Condition | Success | PC success | Action steps | Triggers | Interventions | Resets | Eval seconds | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| noop_scale1 | true | 100.00 | 131 | 1 | 1 | 1 | 7.3373 | 0.007634 | 8.177374 |
| scale05 | true | 100.00 | 132 | 1 | 1 | 1 | 7.3378 | 0.007576 | 8.176857 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 100.00 | 131.50 | 7.3376 | 0.007605 | 8.177116 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
