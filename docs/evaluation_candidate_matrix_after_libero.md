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
| RoboCasa365 / RoboCasa | Strong household-manipulation relevance; current leaderboard is explicitly multi-task generalist robot policy evaluation over atomic and composite kitchen tasks. | RoboCasa365 leaderboard reports 50-task Overall / Atomic-Seen / Composite-Seen / Composite-Unseen, e.g. RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`. LeRobot docs expose `CloseFridge` single-task 20ep evaluation for quick iteration. | CP25 probe gate, strict RunPod reset/step, and SmolVLA `CloseFridge` 20ep now run. Full benchmark-group eval still needs more runtime. | Very high. Long-horizon composite tasks are a natural target for planner/verifier/retry. | Current best external benchmark. We have a 20ep single-task result: `lerobot/smolvla_robocasa` on `CloseFridge` default horizon scored `0/20`. Next step is atomic_seen or a small multi-task subset. |
| ManiSkill-HAB / MS-HAB | ICLR 2025 low-level home rearrangement benchmark; metrics include subtask success-once and progressive completion. | Paper reports 1000 episodes per evaluation run; subtask success-once examples include TidyHouse Pick val `77.48` RL-per and PrepareGroceries Pick val `72.32` RL-per. | Mac supports ManiSkill CPU simulation and standard rendering, but no Mac GPU simulation. RunPod may need NVIDIA Vulkan/SAPIEN validation; previous PickCube path hit driver issues. | Medium-high. It is useful for low-level verifier/subtask recovery, but less directly aligned to SmolVLA LIBERO policy checkpoints. | Keep as partial, smaller-scale checkpoint. Use for subtask-level research, not as the next main paper number. |
| vla-evaluation-harness cross-benchmark smoke | New unified VLA evaluation framework with Dockerized benchmarks and model servers; supports LIBERO, CALVIN, SimplerEnv, ManiSkill, RoboCasa, and more. | Harness paper/proposal lists LIBERO `4 suites x 10 tasks x 50 episodes`, CALVIN `1000` chained sequences, SimplerEnv `4` WidowX tasks x `24` episodes. | RunPod only for serious use. Current public model server list does not show SmolVLA, so SmolVLA may need a custom server. | High as infrastructure. It can make multi-benchmark comparison less bespoke if SmolVLA integration is added. | Run an installation/config audit next; do not replace the working LIBERO path until SmolVLA server feasibility is confirmed. |
| CALVIN | Classic language-conditioned long-horizon manipulation benchmark; directly relevant to language-conditioned sequence completion. | Public protocols commonly report chained sequence success; vla-evaluation-harness lists `ABC->D`, `1000` chained sequences. | RunPod likely; separate dependency stack. | Medium. Good for long-horizon language policy evaluation, but our current SmolVLA LIBERO checkpoint is not automatically CALVIN-compatible. | Candidate after RoboCasa or harness integration; first task is checkpoint/model compatibility, not success table generation. |
| SimplerEnv | Real-to-sim VLA evaluation benchmark with Google Robot/WidowX style tasks; useful for generalization and real-world proxy claims. | Harness lists `4` WidowX tasks x `24` episodes; recent VLA papers use SimplerEnv alongside LIBERO. | RunPod likely; Vulkan/simulator details need validation. | Medium. More about policy generalization than household planner/retry. | Useful for broader VLA comparison, but not first choice for our agentic-wrapper claim. |
| Meta-World / robosuite classic tasks | Stable manipulation baselines and easier environment setup. | Many public success-rate tables exist, but policy/action setup differs strongly from VLA household benchmarks. | Mac and RunPod feasible. | Low-medium. Good debugging benchmark, weak paper relevance for SmolVLA agentic household claims. | Do not spend main evaluation budget here unless we need a fast control-system sanity check. |

## Non-LIBERO Comparison Table To Build Next

LIBERO now has a frozen policy-only baseline and wrapper evidence. The next
table should be non-LIBERO only:

| Row | Benchmark | Scale | Metric | Our current number | Reference number | Status |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Mac-local pilot | ManiSkill `PickCube-v1` | 20 episodes per policy | task success | random `0.0`, zero `0.05` | RDT repo reports OpenVLA `8%`, DP `40%`, RDT `77.2%` on PickCube under a different trained-policy protocol | Pilot only; not comparable yet |
| Mac-local pilot | ManiSkill-HAB SetTable / PrepareGroceries validation tasks | 20 episodes per task/policy | success-once | random/zero `0.0` | MS-HAB paper-scale runs use success-once over much larger evaluation; paper baselines are trained RL/IL policies | Executable pilot; not comparable policy |
| Current blocker | ManiSkill `PickCube-v1` no-fallback strict | 1 episode | direct target env execution | blocked by `ErrorIncompatibleDriver` on current Mac renderer path | n/a | Needs RunPod/driver-compatible ManiSkill renderer |
| CP25 RunPod strict probe | RoboCasa365 / RoboCasa | import/reset-step gate | runtime readiness | passed for `CloseFridge` reset + one zero-action step | n/a | Runtime gate passed with lightweight assets |
| First strict external benchmark | RoboCasa / LeRobot single-task | 20 episodes, default 1000-step horizon | task success | `lerobot/smolvla_robocasa` on `CloseFridge`: `0/20`, `0.0%` | LeRobot documents this exact single-task 20ep protocol; not a leaderboard value | Completed; pipeline runs but baseline is weak on this task |
| Planned full external benchmark | RoboCasa365 / RoboCasa | 50-task benchmark, 3 splits | average task success | pending | RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`, DP `6.1` | Next scale-up after single-task validation |

For the current goal, do not spend more cycles on LIBERO repeat tables unless
the user explicitly asks. The next meaningful work is either:

1. make ManiSkill `PickCube-v1` strict rendering work on a suitable RunPod
   host and run a trained-policy-compatible comparison, or
2. expand RoboCasa from the completed `CloseFridge` 20ep run to either a small
   multi-task subset or the full `atomic_seen` group before attempting all
   RoboCasa365 leaderboard splits.

## External Sources

- ActionX Table 1 SmolVLA LIBERO reference:
  <https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full>
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
