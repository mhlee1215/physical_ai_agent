# LIBERO In-Episode Intervention Ablation Report

## Paired Seed Result

Task: `libero_goal`, task id `6`.

Baseline policy and wrapper policy both use `lerobot/smolvla_libero`. The
wrapper condition keeps the same episode budget and environment reset budget,
but enables an in-episode `semantic_near_receptacle` trigger followed by the
`semantic_place_receptacle` macro.

| Seed | Baseline success | Wrapper success | Success delta | Baseline steps | Wrapper steps | Step delta | Wrapper interventions | Wrapper first trigger | Baseline min target-to-bowl | Wrapper min target-to-bowl |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1200 | false | true | +1 | 300 | 224 | -76 | 4 | 220 | 0.062636 | 0.053997 |
| 1201 | false | false | +0 | 300 | 300 | +0 | 0 | n/a | 0.146379 | 0.146379 |
| 1202 | false | false | +0 | 300 | 300 | +0 | 0 | n/a | 0.142073 | 0.142073 |

Aggregate paired success: baseline `0/3`, wrapper `1/3`.

Interpretation: the small intervention shows a real improvement opportunity,
but only when the base policy already brings the object within the semantic
near-receptacle trigger radius. On seeds `1201` and `1202`, the object never
entered the `0.07` trigger threshold after step `220`, so the wrapper did not
intervene. The next research step is trigger coverage, not more reset retry.

## Conditions

Baseline condition: `baseline_seed1200`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_seed1200 | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.1237 | +0.0000 | 0.000000 | 0.000000 |
| place_zm10_seed1200 | true | 100.00 | 224 | -76 | 4 | 4 | 1 | 9.4327 | -0.6910 | 0.004464 | 6.360830 |
| baseline_seed1201 | false | 0.00 | 300 | +0 | 80 | 0 | 1 | 10.4005 | +0.2768 | 0.000000 | 0.000000 |
| place_zm10_seed1201 | false | 0.00 | 300 | +0 | 0 | 0 | 1 | 10.3540 | +0.2303 | 0.000000 | 0.000000 |
| baseline_seed1202 | false | 0.00 | 300 | +0 | 80 | 0 | 1 | 10.2122 | +0.0885 | 0.000000 | 0.000000 |
| place_zm10_seed1202 | false | 0.00 | 300 | +0 | 0 | 0 | 1 | 10.2757 | +0.1520 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 16.67 | 287.33 | 10.1331 | 0.000744 | 1.060138 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
