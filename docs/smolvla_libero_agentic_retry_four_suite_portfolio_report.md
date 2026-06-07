# LIBERO Four-Suite Agentic Retry Portfolio Report

- long_root: `_workspace/runpod_results/agentic_retry_series_20260606/smolvla_agentic_retry_series_long_3seed_20260606T220158Z`
- remaining_roots: `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1000_20260607T001547Z, _workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z`
- portfolio_budget2: baseline success or any retry condition succeeds after a baseline failure
- protocol: episode-level retry budget, not in-episode replanning

## Per-Suite Seed Results

| Suite | Seed | Episodes | Baseline | Best single retry | Portfolio budget2 | Delta | Recovery | Portfolio attempts | Success/attempt | Success/eval min | Conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Goal | 1000 | 100 | 89.00 | 97.00 (`alternate_steps10`) | 99.00 | +10.00 | 10/11 (90.91) | 240 | 0.4125 | 5.38 | `alternate_steps10=97.00, blind_new_seed=97.00` |
| Object | 1000 | 100 | 94.00 | 96.00 (`alternate_steps10`) | 97.00 | +3.00 | 3/6 (50.00) | 220 | 0.4409 | 5.70 | `alternate_steps10=96.00, blind_new_seed=96.00` |
| Spatial | 1000 | 100 | 91.00 | 98.00 (`blind_new_seed`) | 98.00 | +7.00 | 7/9 (77.78) | 240 | 0.4083 | 4.53 | `alternate_steps15=94.00, blind_new_seed=98.00` |
| Long | 1000 | 100 | 79.00 | 90.00 (`alternate_steps10`) | 92.00 | +13.00 | 13/21 (61.90) | 240 | 0.3833 | 2.56 | `alternate_steps10=90.00, blind_new_seed=85.00` |
| Goal | 1001 | 100 | 87.00 | 97.00 (`alternate_steps10`) | 98.00 | +11.00 | 11/13 (84.62) | 220 | 0.4455 | 5.86 | `alternate_steps10=97.00, blind_new_seed=94.00` |
| Object | 1001 | 100 | 93.00 | 96.00 (`alternate_steps10`) | 97.00 | +4.00 | 4/7 (57.14) | 240 | 0.4042 | 5.24 | `alternate_steps10=96.00, blind_new_seed=96.00` |
| Spatial | 1001 | 100 | 87.00 | 96.00 (`blind_new_seed`) | 96.00 | +9.00 | 9/13 (69.23) | 220 | 0.4364 | 4.70 | `alternate_steps15=93.00, blind_new_seed=96.00` |
| Long | 1001 | 100 | 66.00 | 81.00 (`alternate_steps10`) | 83.00 | +17.00 | 17/34 (50.00) | 280 | 0.2964 | 1.92 | `alternate_steps10=81.00, blind_new_seed=75.00` |
| Goal | 1002 | 100 | 88.00 | 95.00 (`alternate_steps10`) | 97.00 | +9.00 | 9/12 (75.00) | 200 | 0.4850 | 5.96 | `alternate_steps10=95.00, blind_new_seed=94.00` |
| Object | 1002 | 100 | 92.00 | 98.00 (`alternate_steps10`) | 98.00 | +6.00 | 6/8 (75.00) | 240 | 0.4083 | 5.27 | `alternate_steps10=98.00, blind_new_seed=94.00` |
| Spatial | 1002 | 100 | 89.00 | 97.00 (`blind_new_seed`) | 98.00 | +9.00 | 9/11 (81.82) | 220 | 0.4455 | 4.91 | `alternate_steps15=96.00, blind_new_seed=97.00` |
| Long | 1002 | 100 | 70.00 | 86.00 (`blind_new_seed`) | 87.00 | +17.00 | 17/30 (56.67) | 260 | 0.3346 | 2.18 | `alternate_steps10=79.00, blind_new_seed=86.00` |

## Suite Summary

| Suite | Runs | Baseline | Best single | Portfolio budget2 | Delta | Recovery | Attempts mean | Success/attempt | Success/eval min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Goal | 3 | 88.00 +/- 0.82 | 96.33 +/- 0.94 | 98.00 +/- 0.82 | +10.00 +/- 0.82 | 83.51 +/- 6.54 | 220.00 | 0.4477 | 5.73 |
| Object | 3 | 93.00 +/- 0.82 | 96.67 +/- 0.94 | 97.33 +/- 0.47 | +4.33 +/- 1.25 | 60.71 +/- 10.51 | 233.33 | 0.4178 | 5.41 |
| Spatial | 3 | 89.00 +/- 1.63 | 97.00 +/- 0.82 | 97.33 +/- 0.94 | +8.33 +/- 0.94 | 76.28 +/- 5.25 | 226.67 | 0.4301 | 4.71 |
| Long | 3 | 71.67 +/- 5.44 | 85.67 +/- 3.68 | 87.33 +/- 3.68 | +15.67 +/- 1.89 | 56.19 +/- 4.87 | 260.00 | 0.3381 | 2.22 |

## Seed Macro Summary

| Seed | Suites | Baseline macro | Best single macro | Portfolio macro | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 4 | 88.25 | 95.25 | 96.50 | +8.25 |
| 1001 | 4 | 83.25 | 92.50 | 93.50 | +10.25 |
| 1002 | 4 | 84.75 | 94.00 | 95.00 | +10.25 |

## Overall Macro Summary

| Runs | Baseline | Best single | Portfolio budget2 | Delta | Recovery | Attempts mean | Success/attempt | Success/eval min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 85.42 +/- 8.66 | 93.92 +/- 5.17 | 95.00 +/- 4.85 | +9.58 +/- 4.27 | 69.17 +/- 13.23 | 235.00 | 0.4084 | 4.52 |

## Interpretation Guardrail

- Compare `portfolio_budget2` against retry-budget controls, not against policy-only alone.
- `Portfolio attempts` counts all baseline episodes plus all retry episodes actually evaluated for both retry conditions.
- The benchmark success flag remains the final success metric; retry traces only decide whether to rerun failed task/episode indexes.
- A strong blind-retry result means this is currently evidence for retry-budget scaling more than evidence for intelligent failure diagnosis.
- Per-episode action-step counts are not recorded in the current LeRobot `eval_info.json`; action-step-normalized metrics require an instrumented rollout path.
