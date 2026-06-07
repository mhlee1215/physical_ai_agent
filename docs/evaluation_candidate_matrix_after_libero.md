# Evaluation Candidate Matrix After LIBERO Baseline

Date: 2026-06-07

## Current Anchor Result

The first paper-comparable anchor is the repeat-confirmed SmolVLA LIBERO
baseline:

| Evaluation | Policy | Protocol | Goal | Object | Spatial | Long | Avg | Episodes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal routed baseline | `lerobot/smolvla_libero` | Spatial `n_action_steps=10`; Object/Goal/Long `n_action_steps=15`; seed `1000`; batch `1` | 91.0 | 94.0 | 91.0 | 75.0 | 87.75 | 400 |
| ActionX Table 1 reference | SmolVLA | 10 tasks per suite, 10 trials per task | 91.0 | 94.0 | 93.0 | 77.0 | 88.8 | 400 |
| Delta |  |  | 0.0 | 0.0 | -2.0 | -2.0 | -1.05 |  |

Use this as the policy-only baseline for wrapper experiments.

## Candidate Evaluations

| Candidate | Why It Matters | Comparable Public Numbers | Local/RunPod Viability | Fit For Agentic Wrapper | Recommendation |
| --- | --- | --- | --- | --- | --- |
| LIBERO agentic retry repeat | Same benchmark as the anchor result; isolates wrapper effect without adding a new simulator variable. | Same-run baseline vs success-once can be compared internally; policy-only can still be shown next to ActionX SmolVLA. | Already runnable on RunPod. | High. Current full Long probe recovered `15/29` failed episodes and moved Long from `71.0` to `86.0` success-once over 100 episodes. | Do this first: repeat full Long, then expand to all 4 suites if stable. |
| Meta-World MT50 | SmolVLA paper Table 2 reports direct SmolVLA Meta-World numbers, so this is the cleanest non-LIBERO "reference table catch-up" target. | SmolVLA paper Table 2 reports Meta-World success rates: SmolVLA 0.45B `Easy 82.5`, `Medium 41.8`, `Hard 45.0`, `Very Hard 60.0`, `Avg 57.3`; SmolVLA 2.25B `Avg 68.24`; Diffusion Policy `Avg 10.5`; TinyVLA `Avg 31.6`. | RunPod passed full MT50 10ep/task with `lerobot/smolvla_metaworld` after mapping `observation.image -> observation.images.camera1` and setting `--policy.empty_cameras=2`. `very_hard` ablations show seed `1000` improves `20.0% -> 26.0%`; `empty_cameras=0` stays `26.0%`; standalone `batch_size=10` reaches `36.0%`, but full MT50 batch10 falls to table avg `38.5%`; `episode_length=400` does not apply because the current wrapper still uses 500-step rollouts; a LeRobot `v0.5.1` SmolVLA-only source check scored only `12.0%`; a checkpoint `train_config`-style 250ep run scored `24.4%` on `very_hard`, with only task `1` reaching `60.0%`. | Medium. Good for broad manipulation reference parity; less household-specific than RoboCasa but much stronger for published-table comparison. | Current main external benchmark. Best current full-run table-style average remains `40.5%` vs paper `57.3%` (`-16.9pp`); weighted episode average is `51.6%`. Continue parity debugging, but the strongest remaining clue is task-specific reset/protocol mismatch rather than simple source version, camera padding, or sample count alone. |
| RoboCasa365 / RoboCasa | Strong household-manipulation relevance; current leaderboard is explicitly multi-task generalist robot policy evaluation over atomic and composite kitchen tasks. | RoboCasa365 leaderboard reports 50-task Overall / Atomic-Seen / Composite-Seen / Composite-Unseen, e.g. RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`. LeRobot docs expose `CloseFridge` single-task 20ep evaluation for quick iteration, but no public SmolVLA RoboCasa success table has been found yet. | CP25 probe gate, strict RunPod reset/step, SmolVLA `CloseFridge` 20ep, a 3-task 60ep subset, and a lightweight-compatible 5-task 100ep subset now run. The first 5-task scale-up with microwave/stove hit a lightweight-asset blocker at `TurnOnMicrowave`. | Very high. Long-horizon composite tasks are a natural target for planner/verifier/retry. | Keep as secondary household benchmark. Current 5-task subset scored `4/100`, entirely from `TurnOnToaster` `4/20`; useful for our own baseline, but less useful for public SmolVLA reference catch-up. |
| ManiSkill-HAB / MS-HAB | ICLR 2025 low-level home rearrangement benchmark; metrics include subtask success-once and progressive completion. | Paper reports 1000 episodes per evaluation run; subtask success-once examples include TidyHouse Pick val `77.48` RL-per and PrepareGroceries Pick val `72.32` RL-per. | Mac supports ManiSkill CPU simulation and standard rendering, but no Mac GPU simulation. RunPod may need NVIDIA Vulkan/SAPIEN validation; previous PickCube path hit driver issues. | Medium-high. It is useful for low-level verifier/subtask recovery, but less directly aligned to SmolVLA LIBERO policy checkpoints. | Keep as partial, smaller-scale checkpoint. Use for subtask-level research, not as the next main paper number. |
| vla-evaluation-harness cross-benchmark smoke | New unified VLA evaluation framework with Dockerized benchmarks and model servers; supports LIBERO, CALVIN, SimplerEnv, ManiSkill2, RoboCasa, and more. | Public leaderboard reports coverage across 17+ robot simulation benchmarks. Harness configs include ManiSkill2 `5` tasks x `50` episodes/task, RoboCasa, SimplerEnv, CALVIN, RoboTwin, VLABench, RLBench, and related suites. | RunPod install audit passed for `vla-eval==0.2.0`: source checkout `124M`, isolated env `230M`, validation `155/155 configs valid`. Current Pod lacks Docker and `uv`, so benchmark and server smoke are skipped. Current public model-server configs do not include SmolVLA. | High as infrastructure. It can make multi-benchmark comparison less bespoke if SmolVLA integration is added. | Keep as the next infrastructure milestone: use a Docker-enabled RunPod image or host, then add a SmolVLA model-server adapter before claiming comparable SmolVLA numbers. |
| CALVIN | Classic language-conditioned long-horizon manipulation benchmark; directly relevant to language-conditioned sequence completion. | Public protocols commonly report chained sequence success; vla-evaluation-harness lists `ABC->D`, `1000` chained sequences. | RunPod likely; separate dependency stack. | Medium. Good for long-horizon language policy evaluation, but our current SmolVLA LIBERO checkpoint is not automatically CALVIN-compatible. | Candidate after RoboCasa or harness integration; first task is checkpoint/model compatibility, not success table generation. |
| SimplerEnv | Real-to-sim VLA evaluation benchmark with Google Robot/WidowX style tasks; useful for generalization and real-world proxy claims. | Harness lists `4` WidowX tasks x `24` episodes; recent VLA papers use SimplerEnv alongside LIBERO. | RunPod likely; Vulkan/simulator details need validation. | Medium. More about policy generalization than household planner/retry. | Useful for broader VLA comparison, but not first choice for our agentic-wrapper claim. |
| Meta-World / robosuite classic tasks | Covered above as the current table-backed SmolVLA target. | SmolVLA paper Table 2 and LeRobot Meta-World docs provide direct protocol anchors. | RunPod feasible after dependency install. | Medium. | Treat as the immediate post-RoboCasa execution lane. |

## Non-LIBERO Comparison Table To Build Next

LIBERO now has a frozen policy-only baseline and wrapper evidence. The next
table should be non-LIBERO only:

| Row | Benchmark | Scale | Metric | Our current number | Reference number | Status |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Table-backed result | Meta-World MT50 | 50 tasks, 10 trials per task in SmolVLA paper | task success | Easy `65.4`, Medium `38.2`, Hard `38.3`, Very Hard `20.0`, table avg `40.5`; weighted overall `51.6` | SmolVLA 0.45B Avg `57.3`; Easy `82.5`, Medium `41.8`, Hard `45.0`, Very Hard `60.0` | Completed; comparable scale, but `-16.9pp` table-average delta needs parity debugging |
| Full parity rerun | Meta-World MT50 | 50 tasks, 10 trials/task | task success | seed `1000` + `batch_size=10` + `empty_cameras=0`: Easy `67.1`, Medium `43.6`, Hard `23.3`, Very Hard `20.0`, table avg `38.5`; weighted overall `52.0` | SmolVLA 0.45B Avg `57.3`; Easy `82.5`, Medium `41.8`, Hard `45.0`, Very Hard `60.0` | Completed; does not beat the first full MT50 run |
| Parity ablation | Meta-World `very_hard` | 5 tasks, 10-50 trials/task | task success | seed `0` corrected-rename run `20.0`; seed `1000` run `26.0`; seed `1000` + `empty_cameras=0` run `26.0`; standalone `batch_size=10` run `36.0`; LeRobot `v0.5.1` SmolVLA-only run `12.0`; train-config-style 250ep run `24.4` | SmolVLA Table 2 Very Hard `60.0` | Standalone 10ep probes can improve, but larger 250ep run drops to `24.4`; only one of five `very_hard` tasks reached `60.0`, so the gap is task-specific |
| Mac-local pilot | ManiSkill `PickCube-v1` | 20 episodes per policy | task success | random `0.0`, zero `0.05` | RDT repo reports OpenVLA `8%`, DP `40%`, RDT `77.2%` on PickCube under a different trained-policy protocol | Pilot only; not comparable yet |
| Mac-local pilot | ManiSkill-HAB SetTable / PrepareGroceries validation tasks | 20 episodes per task/policy | success-once | random/zero `0.0` | MS-HAB paper-scale runs use success-once over much larger evaluation; paper baselines are trained RL/IL policies | Executable pilot; not comparable policy |
| Current blocker | ManiSkill `PickCube-v1` no-fallback strict | 1 episode | direct target env execution | blocked by `ErrorIncompatibleDriver` on current Mac renderer path | n/a | Needs RunPod/driver-compatible ManiSkill renderer |
| Harness infra audit | VLA evaluation harness | config validation only | install/readiness | `vla-eval==0.2.0` installed on RunPod; source `v0.2.0` cloned; `155/155` configs valid; ManiSkill2 benchmark smoke skipped because Docker is absent | Harness leaderboard covers 17+ benchmarks; configs include ManiSkill2/RoboCasa/SimplerEnv/CALVIN | Infrastructure ready enough to plan, not ready to execute benchmarks on this Pod |
| CP25 RunPod strict probe | RoboCasa365 / RoboCasa | import/reset-step gate | runtime readiness | passed for `CloseFridge` reset + one zero-action step | n/a | Runtime gate passed with lightweight assets |
| First strict external benchmark | RoboCasa / LeRobot single-task | 20 episodes, default 1000-step horizon | task success | `lerobot/smolvla_robocasa` on `CloseFridge`: `0/20`, `0.0%` | LeRobot documents this exact single-task 20ep protocol; not a leaderboard value | Completed; pipeline runs but baseline is weak on this task |
| First strict multi-task subset | RoboCasa / LeRobot 3-task subset | 20 episodes per task, 60 total, default 1000-step horizon | task success | `CloseFridge` `0/20`, `OpenCabinet` `0/20`, `OpenDrawer` `0/20`; overall `0/60`, `0.0%` | Same LeRobot RoboCasa evaluation protocol; still not the full RoboCasa365 50-task leaderboard | Completed; protocol-compatible subset with video evidence |
| Lightweight-compatible multi-task subset | RoboCasa / LeRobot 5-task subset | 20 episodes per task, 100 total, default 1000-step horizon | task success | overall `4/100`, `4.0%`; `TurnOnToaster` `4/20`, all other selected tasks `0/20` | No public SmolVLA RoboCasa reference table found | Completed; secondary internal baseline |
| Current scale-up blocker | RoboCasa / LeRobot 5-task subset | intended 20 episodes per task | reset/eval creation | completed first 3 task groups, then blocked at `TurnOnMicrowave` | n/a | Lightweight `objs_lw` asset registry lacks required categories; use fuller assets or a covered task subset |
| Planned full external benchmark | RoboCasa365 / RoboCasa | 50-task benchmark, 3 splits | average task success | pending | RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`, DP `6.1` | Next scale-up after single-task validation |

For the current goal, do not spend more cycles on LIBERO repeat tables unless
the user explicitly asks. The next meaningful work is either:

1. debug Meta-World parity: our run is comparable scale but below Table 2,
   mostly due to `easy` and `very_hard`;
2. keep RoboCasa as a secondary household benchmark, using the current subset
   results as our own baseline rather than a public SmolVLA reference catch-up;
3. use VLA evaluation harness as the next infrastructure lane only on a
   Docker-enabled host and after a SmolVLA model-server adapter exists; and
4. make ManiSkill `PickCube-v1` strict rendering work on a suitable RunPod
   host only after choosing between direct CP24 and harness-based execution.

## External Sources

- ActionX Table 1 SmolVLA LIBERO reference:
  <https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full>
- SmolVLA paper Table 2 Meta-World reference:
  <https://arxiv.org/abs/2506.01844>
- LeRobot Meta-World docs:
  <https://huggingface.co/docs/lerobot/v0.4.3/metaworld>
- `lerobot/smolvla_metaworld` model:
  <https://huggingface.co/lerobot/smolvla_metaworld>
- RoboCasa365 leaderboard:
  <https://robocasa.ai/leaderboard.html>
- RoboCasa benchmarking documentation:
  <https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html>
- RoboCasa RSS 2024 project:
  <https://robocasa.ai/>
- ManiSkill-HAB ICLR 2025 paper:
  <https://openreview.net/pdf?id=6bKEWevgSd>
- ManiSkill macOS installation notes:
  <https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/getting_started/macos_install.html>
- VLA evaluation harness:
  <https://github.com/allenai/vla-evaluation-harness>
- VLA evaluation harness paper:
  <https://openreview.net/pdf/3e7c55fe021522dc4ff8171ef4082a307238ab67.pdf>
- CALVIN paper:
  <https://arxiv.org/abs/2112.03227>
