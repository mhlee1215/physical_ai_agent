# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `task3_hook_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task3_hook_none | true | 100.00 | 177 | +0 | 1 | 0 | 1 | 8.5576 | +0.0000 | 0.005650 | 7.011290 |
| task3_spike_scale145_110 | true | 100.00 | 178 | +1 | 8 | 8 | 1 | 8.5737 | +0.0160 | 0.005618 | 6.998183 |
| task3_spike_scale145_120 | true | 100.00 | 177 | +0 | 8 | 8 | 1 | 8.5247 | -0.0329 | 0.005650 | 7.038377 |
| task3_spike_clamp145_to135 | true | 100.00 | 177 | +0 | 8 | 8 | 1 | 8.5000 | -0.0576 | 0.005650 | 7.058835 |
| task3_spike_smooth145_a050 | true | 100.00 | 177 | +0 | 8 | 8 | 1 | 8.3979 | -0.1598 | 0.005650 | 7.144676 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 100.00 | 177.20 | 8.5108 | 0.005643 | 7.050272 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
