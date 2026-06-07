# SmolVLA LIBERO Agentic Retry Report

## Status

Implemented the first LIBERO/SmolVLA agentic wrapper layer:

- verifier: LIBERO benchmark `success` flag from `eval_info.json`
- planner: schedule retry for task ids with failed baseline episodes
- retry policy: retry failed task/episode indexes once
- aggregate metric: `success_once_rate`, where an episode passes if either the
  baseline pass or retry pass is true

This is a basic episode-level retry wrapper. It is not yet a subgoal-level
environment intervention inside a single LIBERO episode.

## First RunPod Probes

Task subset:

- suite: `libero_10`
- task ids: `[0,6,8]`
- episodes per task: `2`
- total baseline episodes: `6`

| Probe | Baseline retry args | Retry args | Baseline success | Success once | Recovery |
| --- | --- | --- | ---: | ---: | ---: |
| same-protocol retry | steps15, seed1000 | steps15, seed1000 | 50.00 | 50.00 | 0/3 |
| alternate retry | steps15, seed1000 | steps10, seed1001 | 50.00 | 66.67 | 1/3 |
| alternate retry, 30 episodes | steps15, seed1000 | steps10, seed1001 | 50.00 | 70.00 | 6/15 |
| alternate retry, full Long | steps15, seed1000 | steps10, seed1001 | 71.00 | 86.00 | 15/29 |

The same-protocol retry confirmed the wrapper plumbing but did not recover any
failed episode. The alternate retry recovered one failed task-episode index,
raising success-once from `50.00%` to `66.67%` on the smallest subset. Scaling
the same alternate condition to `30` episodes recovered `6/15` failed episodes
and raised success-once from `50.00%` to `70.00%`.

The first full Long-suite run recovered `15/29` failed episodes and raised
success-once from `71.00%` to `86.00%`. This is the first full-suite positive
agentic retry signal. Compare it against its own baseline in the same run; do
not mix it with the earlier routed policy-only full-suite baseline without
disclosing the protocol difference.

## Long-Suite Retry Control Series

The first paper-oriented control series repeated Long-suite episode-level retry
over three baseline seeds. Each baseline used `n_action_steps=15`; retry used
the same failed task/episode indexes from the matching baseline seed.

| Condition | Base seed | Retry seed | Episodes | Baseline | Success once | Delta | Recovery | Recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_steps10 | 1000 | 1100 | 100 | 79.00 | 90.00 | +11.00 | 52.38 | 11/21 |
| alternate_steps10 | 1001 | 1101 | 100 | 66.00 | 81.00 | +15.00 | 44.12 | 15/34 |
| alternate_steps10 | 1002 | 1102 | 100 | 70.00 | 79.00 | +9.00 | 30.00 | 9/30 |
| blind_new_seed | 1000 | 1100 | 100 | 79.00 | 85.00 | +6.00 | 28.57 | 6/21 |
| blind_new_seed | 1001 | 1101 | 100 | 66.00 | 75.00 | +9.00 | 26.47 | 9/34 |
| blind_new_seed | 1002 | 1102 | 100 | 70.00 | 86.00 | +16.00 | 53.33 | 16/30 |

| Condition | Runs | Baseline mean | Success-once mean | Delta mean | Recovery mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| alternate_steps10 | 3 | 71.67 +/- 5.44 | 83.33 +/- 4.78 | +11.67 +/- 2.49 | 42.17 +/- 9.24 |
| blind_new_seed | 3 | 71.67 +/- 5.44 | 82.00 +/- 4.97 | +10.33 +/- 4.19 | 36.13 +/- 12.20 |

Interpretation: blind retry is a strong control, so future agentic claims must
compare against it rather than against policy-only alone. The alternate
`n_action_steps=10` retry had the higher mean gain in this series, but the
margin over blind retry is small and seed `1002` favored blind retry.

## Task-Guided Selection Analysis

Using the completed blind and alternate retry traces, an offline selector was
evaluated without additional GPU rollout. The selector used only task identity
and baseline failure as inputs.

| Selector | Runs | Baseline mean | Success-once mean | Delta mean | Recovery mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| alternate_steps10 | 3 | 71.67 +/- 5.44 | 83.33 +/- 4.78 | +11.67 +/- 2.49 | 42.17 +/- 9.24 |
| blind_new_seed | 3 | 71.67 +/- 5.44 | 82.00 +/- 4.97 | +10.33 +/- 4.19 | 36.13 +/- 12.20 |
| portfolio_budget2 | 3 | 71.67 +/- 5.44 | 87.33 +/- 3.68 | +15.67 +/- 1.89 | 56.19 +/- 4.87 |
| task_guided_loso | 3 | 71.67 +/- 5.44 | 80.00 +/- 3.74 | +8.33 +/- 1.70 | 29.33 +/- 0.59 |
| task_oracle_same_seed | 3 | 71.67 +/- 5.44 | 85.67 +/- 3.68 | +14.00 +/- 2.16 | 49.94 +/- 4.14 |

Interpretation: task identity alone is not a good enough verifier/planner
signal. The leave-one-seed-out task-guided selector underperformed both blind
and alternate retry. However, `portfolio_budget2` is a deployable two-retry
condition if the protocol allows both retry variants after baseline failures:
execute both retry variants and count success if either retry succeeds. This
suggests the next method should be reported as retry-budget scaling or
portfolio retry before claiming intelligent failure diagnosis.

## Four-Suite Seed-1000 Portfolio Probe

After the Long 3-seed analysis, the remaining LIBERO suites were evaluated with
seed `1000` to estimate a first four-suite portfolio-retry table. The Long row
uses the existing 3-seed series seed `1000` run; Spatial/Object/Goal use the
remaining-suite probe.

| Suite | Baseline | Best single retry | Portfolio budget2 | Delta vs baseline | Recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 91.00 | 98.00 (`blind_new_seed`) | 98.00 | +7.00 | 7/9 |
| Object | 94.00 | 96.00 (`alternate_steps10`) | 97.00 | +3.00 | 3/6 |
| Goal | 89.00 | 97.00 (`alternate_steps10`) | 99.00 | +10.00 | 10/11 |
| Long | 79.00 | 90.00 (`alternate_steps10`) | 92.00 | +13.00 | 13/21 |
| Macro avg | 88.25 | 95.25 | 96.50 | +8.25 |  |

This is the strongest current agentic-retry table, but it must be presented as
`retry_budget=2` portfolio evaluation. It is not directly comparable to
policy-only ActionX numbers unless the extra retry budget is disclosed.

## Four-Suite Three-Seed Portfolio With Cost

The follow-up run completed Spatial/Object/Goal seeds `1001` and `1002` and
joins them with the existing Long 3-seed series. This gives a 12-run table:
four LIBERO suites over seeds `1000,1001,1002`, 100 episodes per suite/seed.

| Metric | Mean |
| --- | ---: |
| Baseline success | 85.42 +/- 8.66 |
| Best single retry success | 93.92 +/- 5.17 |
| Portfolio budget2 success | 95.00 +/- 4.85 |
| Portfolio delta vs baseline | +9.58 +/- 4.27 |
| Portfolio recovery | 69.17 +/- 13.23 |
| Portfolio attempts | 235.00 |
| Portfolio success/attempt | 0.4084 |
| Portfolio success/eval minute | 4.52 |

Seed macro results:

| Seed | Baseline macro | Best single macro | Portfolio macro | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 88.25 | 95.25 | 96.50 | +8.25 |
| 1001 | 83.25 | 92.50 | 93.50 | +10.25 |
| 1002 | 84.75 | 94.00 | 95.00 | +10.25 |

Artifact:
`docs/smolvla_libero_agentic_retry_four_suite_portfolio_report.md`

Interpretation: this confirms a strong realized-success gain from retry
budget, but it is not a one-shot policy improvement. The model weights remain
`lerobot/smolvla_libero`. The protocol spends extra environment resets and
retry attempts. For paper claims, present this as a cost-normalized
retry-budget baseline/control, not as a better SmolVLA policy.

## Paper Claim Boundary

The current evidence supports:

- Same frozen SmolVLA plus explicit reset/retry budget improves realized LIBERO
  benchmark success.
- Different retry protocols recover different failure subsets; blind retry is
  a strong control and sometimes wins.

The current evidence does not yet support:

- SmolVLA model weights improved.
- One-shot rollout success improved.
- The wrapper performs intelligent in-episode failure diagnosis.

The next paper-grade agentic experiment must either beat blind retry under the
same retry budget, improve success per attempt/eval minute, or intervene inside
an episode before environment reset.

## Full Long Per-Task Recovery

| Task | Baseline | Recovered | Success once |
| --- | ---: | ---: | ---: |
| 0 | 4/10 | 1/6 | 5/10 |
| 1 | 10/10 | 0/0 | 10/10 |
| 2 | 10/10 | 0/0 | 10/10 |
| 3 | 9/10 | 1/1 | 10/10 |
| 4 | 4/10 | 2/6 | 6/10 |
| 5 | 10/10 | 0/0 | 10/10 |
| 6 | 1/10 | 6/9 | 7/10 |
| 7 | 7/10 | 1/3 | 8/10 |
| 8 | 7/10 | 3/3 | 10/10 |
| 9 | 9/10 | 1/1 | 10/10 |

## Artifacts

- same-protocol remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_probe_20260606T2036Z`
- same-protocol local:
  `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_probe_20260606T2036Z`
- alternate remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_probe_20260606T2042Z`
- alternate local:
  `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_probe_20260606T2042Z`
- alternate 30-episode remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_30ep_20260606T2049Z`
- alternate 30-episode local:
  `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_30ep_20260606T2049Z`
- alternate full-Long remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_alt_long_full_20260606T2108Z`
- alternate full-Long local:
  `_workspace/runpod_results/agentic_retry_probe_20260606/smolvla_agentic_retry_alt_long_full_20260606T2108Z`
- Long retry control series remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_series_long_3seed_20260606T220158Z`
- Long retry control series local archive:
  `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z_no_videos.tar.gz`
- Long retry control series local extracted report:
  `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z/agentic_retry_series_report.md`
- Long retry selection local report:
  `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z/agentic_retry_selection_report.md`
- Remaining-suite portfolio remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z`
- Remaining-suite portfolio local archive:
  `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z_no_videos.tar.gz`
- Remaining-suite portfolio local extracted report:
  `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z/agentic_retry_portfolio_report.md`
- Remaining-suite seed1001/1002 portfolio remote:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z`
- Remaining-suite seed1001/1002 portfolio local archive:
  `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z_no_videos.tar.gz`
- Four-suite 3-seed cost-normalized report:
  `docs/smolvla_libero_agentic_retry_four_suite_portfolio_report.md`

## Next Step

The repeat/control series confirms that retry budget improves realized success,
but it also shows that blind retry is competitive and task-id-only selection is
weak. The next paper-useful experiment should move from reset-level retry to an
instrumented in-episode verifier/intervention loop, while continuing to report
attempt, reset, eval-time, and action-step normalized costs.
