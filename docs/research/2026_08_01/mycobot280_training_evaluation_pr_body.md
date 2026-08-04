## Summary

This PR completes the myCobot 280 SmolVLA fine-tuning and policy-only
evaluation pipeline. It adds matched close-camera LeRobot datasets, exact-7D
state/action adaptation, reloadable checkpoints, held-out diagnostics,
camera-controlled closed-loop evaluation, three paired training replications,
penetration auditing, figures, and reproducible video evidence.

The results support two scoped claims:

1. **Fine-tuning changes task behavior on the matched control.** The
   unfine-tuned base policy achieved `0/11` pickup-and-hold, while both
   deterministic- and randomized-data fine-tuned policies achieved `11/11`.
2. **Randomized-data fine-tuning improves strict penetration-gated success in
   the tested direction.** It exceeded deterministic-data fine-tuning in `3/3`
   paired training seeds under both nominal and fresh-randomized physics.

The randomized-over-deterministic result is replicated engineering evidence,
not statistical significance: the exact two-sided sign test is `p=0.25` at
`n=3` independently trained seed pairs. Agentic verifier/retry code is
intentionally excluded and will be evaluated separately.

## What The Sample Counts Mean

| Unit | Count | Interpretation |
| --- | ---: | --- |
| Training seeds per dataset | 3 | Independently trained checkpoints: `20260731-20260733` |
| Evaluation episodes per checkpoint and regime | 11 | Matched environment seeds `92000-92010` |
| Pooled rollouts per dataset and regime | 33 | `3` checkpoints x `11` episodes; descriptive total |
| Unfine-tuned base control | 11 | One fixed base policy on the matched fresh-randomized schedule |
| Comparison video | 1 | Seed `92000`; illustrative, not an aggregate result |

The sign test uses `n=3` training-seed pairs. The `x/33` totals pool policy
rollouts, but those episodes are clustered within three checkpoints and are not
33 independent training replications.

## Dataset Dependency And Filtering

This work uses the randomized generator, contract, and source audits from
[#33: myCobot 280 ground-pickup randomized dataset v0](https://github.com/mhlee1215/physical_ai_agent/pull/33):

- source branch: `codex/mycobot-280-ground-pickup-randomized-dataset`;
- source commits: `f9c7569`, `1a2ac84`, and `5ddb73b`;
- accepted randomized corpus: 50 train and 10 validation episodes, or 31,800
  all-frame observations before LeRobot conversion.

The learning comparison is conditioned on successful teacher demonstrations.
For the randomized source this filtering is explicit: 60 of 73 attempted
teacher episodes were retained. All 13 rejected attempts occur in the highest
mass quartile (`0.034-0.036 kg`), where only 2/15 attempts were accepted;
validation contains no accepted high-mass examples and no examples in the first
two friction quartiles.

Teacher-success filtering applies to training and supervised-validation data,
not to closed-loop policy evaluation. Evaluation uses direct, unfiltered
candidate draws and counts every rollout, including failures. Merge PR #33
first so this PR remains a downstream training/evaluation change.

## Dataset And Training Contract

| Input | Episodes | Frames | Camera |
| --- | ---: | ---: | --- |
| Deterministic training | 50 | 26,500 | close, 256x256 |
| Randomized training | 50 | 26,500 | close, 256x256 |
| Common randomized validation | 10 | 5,300 | close, 256x256 |

All six checkpoints use `lerobot/smolvla_base`, exact 7D state/action, 100
optimizer steps, batch size 1, learning rate `1e-5`, CUDA, dataset-derived
processors, and constant-joint normalization safeguards.

| Dataset | Seed | Median train loss | Held-out loss | Action RMSE |
| --- | ---: | ---: | ---: | ---: |
| Deterministic | 20260731 | 0.37815 | 4.98985 | 0.10869 |
| Randomized | 20260731 | 0.35483 | 5.08591 | 0.10652 |
| Deterministic | 20260732 | 0.41996 | 4.60799 | 0.10405 |
| Randomized | 20260732 | 0.43502 | 4.73292 | 0.10105 |
| Deterministic | 20260733 | 0.34436 | 5.28494 | 0.11066 |
| Randomized | 20260733 | 0.37761 | 5.41762 | 0.11105 |
| Mean held-out, det. vs rand. | - | - | 4.9609 vs 5.0788 | 0.10780 vs 0.10621 |

Held-out loss and action RMSE do not consistently rank the two fine-tuned
policies and are not treated as closed-loop task-success proxies.

## Closed-Loop Results

Each cell contains 11 matched 530-step MuJoCo episodes using the same close
camera, no teacher attachment, no object teleport, and the strict 3.0 mm
adaptive-pad-to-cube penetration cap.

| Training seed | Det. nominal | Rand. nominal | Det. fresh | Rand. fresh |
| ---: | ---: | ---: | ---: | ---: |
| 20260731 | 5/11 | 7/11 | 3/11 | 5/11 |
| 20260732 | 1/11 | 4/11 | 1/11 | 5/11 |
| 20260733 | 0/11 | 3/11 | 0/11 | 4/11 |
| Descriptive total | 6/33 | 14/33 | 4/33 | 14/33 |

Strict success includes pickup, lift, contact, hold, and the 3 mm penetration
gate. Pickup-and-hold below excludes only the penetration gate.

| Training data | Evaluation | Strict | Pickup + hold | Penetration-only |
| --- | --- | ---: | ---: | ---: |
| Deterministic | Nominal | 6/33 | 32/33 | 26/33 |
| Randomized | Nominal | 14/33 | 32/33 | 18/33 |
| Deterministic | Fresh randomized | 4/33 | 32/33 | 28/33 |
| Randomized | Fresh randomized | 14/33 | 33/33 | 19/33 |

Randomized training reduced mean maximum pad-cube penetration by `0.087 mm`
under nominal physics and `0.082 mm` under fresh-randomized physics.

## Claim Boundary

| Question | Supported conclusion |
| --- | --- |
| Fine-tuning vs no fine-tuning | Supported for functional pickup-and-hold on one matched 11-episode control: `0/11` base versus `11/11` for both fine-tuned policies. |
| Randomized vs deterministic, strict | Directionally replicated: randomized is higher in `3/3` training-seed pairs and `14/33` versus `4/33` under fresh physics. Not statistically significant at `n=3`. |
| Randomized vs deterministic, penetration excluded | No material functional-task advantage established: `33/33` versus `32/33` fresh and `32/33` versus `32/33` nominal. |
| Contact quality | Randomized training produces fewer penetration-only failures and slightly lower mean maximum penetration under the fixed 3 mm gate. |
| General robustness | Not established because training/validation demonstrations are teacher-success-conditioned and high-mass/low-friction coverage is incomplete. |

Accordingly, the evidence supports `fine-tuning > no fine-tuning` for matched
functional behavior and supports `randomized > deterministic` directionally for
strict contact-valid success. It does not support a broad claim that randomized
training materially improves pickup-and-hold when penetration is excluded.

## Camera Control

The deterministic camera ablation produced base wide/close strict success of
`0/11` and fine-tuned wide/close strict success of `3/11` versus `5/11`. All
deterministic-versus-randomized comparisons use the same close-camera contract,
so camera framing is not confounded with training-data randomization.

## Figures

### Main policy-progression figure

The shared unfine-tuned baseline appears once at `0/11`. Dotted segments connect
that one baseline reference to each independently trained deterministic
checkpoint; solid segments show each paired deterministic-to-randomized result.

![Base to deterministic to randomized progression](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_policy_progression_main.png?raw=1)

### Supporting task-versus-strict claim map

The first panel of the policy-paradigm figure plots penetration-excluded
pickup-and-hold on the x-axis and strict success on the y-axis for the matched
11-episode control. Its second panel shows all three paired training-seed
comparisons.

![Policy paradigm comparison](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_policy_paradigm_claims.png?raw=1)

![Matched camera ablation](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_camera_ablation.png?raw=1)

![Randomized-training pilot](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_randomized_training_pilot.png?raw=1)

![Three-seed replication](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_randomized_training_multiseed.png?raw=1)

## Video Evidence

![Video preview](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_base_vs_finetuned_seed92000_preview.jpg?raw=1)

[Open the 17.67-second base versus fine-tuned comparison](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_base_vs_finetuned_seed92000.mp4)

| Policy | Final lift | Max penetration | Outcome |
| --- | ---: | ---: | --- |
| Base | 31.30 mm | 3.292 mm | task and penetration failure |
| Deterministic fine-tuned | 64.83 mm | 3.178 mm | pickup/hold; penetration-only |
| Randomized fine-tuned | 60.61 mm | 3.113 mm | pickup/hold; penetration-only |

The video is one matched candidate, seed `92000`. It illustrates the behavioral
change from base to fine-tuned policies, but does not by itself establish the
randomized-over-deterministic claim. The 3-seed x 11-episode results above carry
that comparison.

## Evidence And Verification

- [Complete evidence package](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_training_evaluation_pr_package.md)
- [Policy claim boundary](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_policy_claims_evidence.md)
- [Machine-readable policy claim summary](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_08_01/mycobot280_policy_claims_summary.json)
- [Three-seed evidence](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_randomized_training_multiseed_evidence.md)
- [Machine-readable multiseed summary](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_randomized_training_multiseed_summary.json)
- [Penetration audit](https://github.com/mhlee1215/physical_ai_agent/blob/codex/mycobot280-randomized-smolvla-eval/docs/research/2026_07_31/mycobot280_penetration_failure_audit.json)
- 47 focused tests pass: 34 training/evaluation tests and 13 randomized-source
  contract tests.
- All checkpoints contain weights, optimizer state, feature contract,
  processors, logs, and TensorBoard events.
- All videos decode to exactly 530 frames at 30 FPS.
- Generated claim summaries and figures reproduce byte-for-byte from the saved
  reports.

## Limitations

These results do not establish statistical significance, uniform physics
coverage, optimized training convergence, real-robot transfer, recovery from
unseen states, or agentic verifier/retry benefit. The 3 mm metric covers
adaptive-pad-to-cube contact only and does not measure cube-table/mat overlap or
robot self-collision.
