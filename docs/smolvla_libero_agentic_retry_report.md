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

## Next Step

The repeat/control series confirms that retry budget improves realized success
on Long, but it also shows that blind retry is competitive. The next paper-useful
step is to implement a stronger verifier-guided retry condition that selects
retry strategy from failure/task predicates, then compare it against
`blind_new_seed` and `alternate_steps10`.
