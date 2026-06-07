# LIBERO In-Episode Intervention Ablation Report

## Trigger Coverage Result

Task: `libero_goal`, task id `6`. All rows use the same
`lerobot/smolvla_libero` weights, one environment reset, and one rollout per
seed.

This run compares two in-episode trigger policies:

- `near_receptacle`: intervene only after the target object is already close to
  the bowl.
- `no_progress`: intervene after target-object motion stalls, even if the
  target is not yet close to the bowl.

| Seed | Baseline success | Near-receptacle success | No-progress success | Near interventions | No-progress interventions | Baseline min target-to-bowl | Near min target-to-bowl | No-progress min target-to-bowl |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1200 | false | true | false | 4 | 80 | 0.062636 | 0.053997 | 0.139642 |
| 1201 | false | false | false | 0 | 80 | 0.146379 | 0.146379 | 0.112685 |
| 1202 | false | false | false | 0 | 60 | 0.142073 | 0.142073 | 0.061886 |

Aggregate benchmark success:

| Condition | Successes | Success rate | Mean interventions |
| --- | ---: | ---: | ---: |
| baseline | 0/3 | 0.00% | 0.00 |
| near_receptacle | 1/3 | 33.33% | 1.33 |
| no_progress | 0/3 | 0.00% | 73.33 |

Interpretation: `no_progress` does broaden trigger coverage, and it improves
the intermediate target-to-bowl distance on seeds `1201` and `1202`. However,
it is too aggressive: it applies 60-80 interventions and loses the seed `1200`
success that the more surgical near-receptacle trigger produced. The next
paper-useful direction is therefore not reset retry; it is a gated or phased
in-episode trigger that uses no-progress to recover reach/approach coverage,
then switches to the placement macro only when contact or near-receptacle
conditions are satisfied.

## Conditions

Baseline condition: `baseline_seed1200`.

| Condition | Success | PC success | Action steps | Delta steps | Triggers | Interventions | Resets | Eval seconds | Delta eval s | Success/action step | Success/eval min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_seed1200 | false | 0.00 | 300 | +0 | 31 | 0 | 1 | 10.1237 | +0.0000 | 0.000000 | 0.000000 |
| near_receptacle_seed1200 | true | 100.00 | 224 | -76 | 4 | 4 | 1 | 9.4327 | -0.6910 | 0.004464 | 6.360830 |
| no_progress_seed1200 | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 9.3934 | -0.7302 | 0.000000 | 0.000000 |
| baseline_seed1201 | false | 0.00 | 300 | +0 | 80 | 0 | 1 | 10.4005 | +0.2768 | 0.000000 | 0.000000 |
| near_receptacle_seed1201 | false | 0.00 | 300 | +0 | 0 | 0 | 1 | 10.3540 | +0.2303 | 0.000000 | 0.000000 |
| no_progress_seed1201 | false | 0.00 | 300 | +0 | 80 | 80 | 1 | 8.9403 | -1.1834 | 0.000000 | 0.000000 |
| baseline_seed1202 | false | 0.00 | 300 | +0 | 80 | 0 | 1 | 10.2122 | +0.0885 | 0.000000 | 0.000000 |
| near_receptacle_seed1202 | false | 0.00 | 300 | +0 | 0 | 0 | 1 | 10.2757 | +0.1520 | 0.000000 | 0.000000 |
| no_progress_seed1202 | false | 0.00 | 300 | +0 | 60 | 60 | 1 | 8.9045 | -1.2191 | 0.000000 | 0.000000 |

## Summary

| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 11.11 | 291.56 | 9.7819 | 0.000496 | 0.706759 |

## Claim Boundary

- This is a tiny same-task smoke ablation, not a paper-scale benchmark.
- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.
