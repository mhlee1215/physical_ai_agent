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

## Next Step

Repeat the full Long-suite agentic retry run to estimate variance before making
a paper-scale claim. If recovery remains positive, compare against the
repeat-confirmed policy-only routed baseline and then decide whether to expand
to all four LIBERO suites.
