# LIBERO In-Episode Intervention Ablation Report

## Conditions

Baseline condition: `hook_none`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hook_none | true | 100.00 | 131 | +0 | 1 | 0 | 1 | 7.4206 | +0.0000 | 0.007634 | 8.085627 |
| spike_clamp145_to120 | true | 100.00 | 135 | +4 | 20 | 20 | 1 | 7.5462 | +0.1257 | 0.007407 | 7.950980 |
| spike_smooth145_a070 | true | 100.00 | 132 | +1 | 19 | 19 | 1 | 7.3394 | -0.0812 | 0.007576 | 8.175104 |
| spike_scale145_085 | true | 100.00 | 134 | +3 | 19 | 19 | 1 | 7.6068 | +0.1862 | 0.007463 | 7.887687 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 100.00 | 133.00 | 7.4782 | 0.007520 | 8.024850 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
