# myCobot 280 Penetration-Failure Audit

Strict pad-cube penetration threshold: **3.0 mm**.

| Evaluation | Episodes | Strict | Penetration gate failures | Penetration-only | Mean peak (mm) | Median steps over gate | Longest run | Peak side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base_fresh | 11 | 0 | 6 | 0 | 3.404 | 6.0 | 9 | right=6 |
| det_s20260731_nominal | 11 | 5 | 6 | 6 | 3.309 | 8.0 | 10 | right=6 |
| rand_s20260731_nominal | 11 | 7 | 4 | 4 | 3.430 | 7.0 | 8 | right=4 |
| det_s20260731_fresh | 11 | 3 | 8 | 8 | 3.225 | 6.0 | 9 | right=8 |
| rand_s20260731_fresh | 11 | 5 | 6 | 6 | 3.287 | 6.0 | 7 | right=6 |
| det_s20260732_nominal | 11 | 1 | 10 | 10 | 3.273 | 6.0 | 12 | right=10 |
| rand_s20260732_nominal | 11 | 4 | 7 | 7 | 3.323 | 5.0 | 9 | right=7 |
| det_s20260732_fresh | 11 | 1 | 10 | 9 | 3.289 | 7.0 | 11 | right=10 |
| rand_s20260732_fresh | 11 | 5 | 6 | 6 | 3.414 | 5.5 | 8 | right=6 |
| det_s20260733_nominal | 11 | 0 | 11 | 10 | 3.288 | 9.0 | 12 | right=11 |
| rand_s20260733_nominal | 11 | 3 | 8 | 7 | 3.301 | 7.5 | 11 | right=8 |
| det_s20260733_fresh | 11 | 0 | 11 | 11 | 3.257 | 7.0 | 10 | right=11 |
| rand_s20260733_fresh | 11 | 4 | 7 | 7 | 3.319 | 7.0 | 9 | right=7 |

## Interpretation Boundary

- The gate covers adaptive-gripper **pad-cube** contacts only.
- Cube-table/mat penetration and robot self-collision are not being mislabeled by this metric.
- Penetration-only means pickup/lift/hold passed, but strict contact quality did not.
- Saved traces identify side, phase, magnitude, and duration in simulation steps.
- They do not log contact force, contact impulse; those need evaluator instrumentation before force-based diagnosis.

> A penetration-only failure is functional pickup/lift/hold success that failed the conservative pad-cube contact-quality gate. This audit characterizes saved policy-only traces; it does not validate a recovery action or agentic improvement.
