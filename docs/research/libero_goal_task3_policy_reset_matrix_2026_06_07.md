# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task3_hook_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task3_hook_none | true | 100.00 | 177 | +0 | 1 | 0 | 1 | 8.5576 | +0.0000 | 0.005650 | 7.011290 |
| task3_reset_step80 | true | 100.00 | 178 | +1 | 1 | 1 | 1 | 8.5501 | -0.0075 | 0.005618 | 7.017478 |
| task3_reset_step120 | true | 100.00 | 177 | +0 | 1 | 1 | 1 | 8.5292 | -0.0284 | 0.005650 | 7.034654 |
| task3_reset_step145 | true | 100.00 | 177 | +0 | 1 | 1 | 1 | 8.4667 | -0.0910 | 0.005650 | 7.086612 |
| task3_reset_norm145 | true | 100.00 | 177 | +0 | 8 | 8 | 1 | 9.1695 | +0.6119 | 0.005650 | 6.543397 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 100.00 | 177.20 | 8.6546 | 0.005643 | 6.938686 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
