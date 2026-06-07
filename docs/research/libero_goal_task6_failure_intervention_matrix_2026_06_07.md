# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task6_hook_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task6_hook_none | false | 0.00 | 300 | +0 | 1 | 0 | 1 | 10.2711 | +0.0000 | 0.000000 | 0.000000 |
| task6_reset_step30 | false | 0.00 | 300 | +0 | 1 | 1 | 1 | 10.5359 | +0.2648 | 0.000000 | 0.000000 |
| task6_reset_step60 | false | 0.00 | 300 | +0 | 1 | 1 | 1 | 10.4445 | +0.1733 | 0.000000 | 0.000000 |
| task6_reset_step120 | false | 0.00 | 300 | +0 | 1 | 1 | 1 | 10.3801 | +0.1089 | 0.000000 | 0.000000 |
| task6_reset_norm135 | false | 0.00 | 300 | +0 | 39 | 39 | 1 | 14.1313 | +3.8602 | 0.000000 | 0.000000 |
| task6_smooth_norm135_a050 | false | 0.00 | 300 | +0 | 52 | 52 | 1 | 10.3311 | +0.0600 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.00 | 300.00 | 11.0157 | 0.000000 | 0.000000 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
