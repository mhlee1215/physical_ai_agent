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
| Meta-World MT50 | SmolVLA paper Table 2 reports direct SmolVLA Meta-World numbers, so this is the cleanest non-LIBERO "reference table catch-up" target. | SmolVLA paper Table 2 reports Meta-World success rates: SmolVLA 0.45B `Easy 82.5`, `Medium 41.8`, `Hard 45.0`, `Very Hard 60.0`, `Avg 57.3`; SmolVLA 2.25B `Avg 68.24`; Diffusion Policy `Avg 10.5`; TinyVLA `Avg 31.6`. | RunPod passed full MT50 10ep/task with `lerobot/smolvla_metaworld` after mapping `observation.image -> observation.images.camera1` and setting `--policy.empty_cameras=2`. A second official-docs-shaped full run with `seed=0`, `batch_size=1`, and `empty_cameras=0` scored table avg `39.5%`, below the current best `40.5%`, so empty-camera padding is not the main full-run parity cause. `very_hard` ablations show seed `1000` improves `20.0% -> 26.0%`; standalone `batch_size=10` reaches `36.0%`, but full MT50 batch10 falls to table avg `38.5%`; a fixed reset-vector patch scored only `16.0%`, so disabling Meta-World reset randomization does not recover parity; a direct temporary 400-step wrapper patch scored `34.0%`, so the released checkpoint `episode_length=400` clue does not recover parity; a LeRobot `v0.5.1` SmolVLA-only source check scored only `12.0%`; a checkpoint `train_config`-style 250ep run scored `24.4%` on `very_hard`, with only task `1` reaching `60.0%`; a temporary actual `top` camera patch scored `0.0%`, so `pixels/top` naming does not mean the release should use the Meta-World top camera; a temporary `MT1(..., seed=1000)` suite-construction patch stayed at `36.0%`; checkpoint normalizer stats use 4D `observation.state`, matching the wrapper; dataset metadata has 50 `task_id` values but 49 language `task_index` values, and current `very_hard` task ids match by text; large-action probes matched clipped-action probes, so Meta-World action clipping removes the simplest action-scale mismatch; a no-rename smoke fails before rollout, confirming the explicit CLI rename is required rather than an accidental duplicate. | Medium. Good for broad manipulation reference parity; less household-specific than RoboCasa but much stronger for published-table comparison. | Current main external benchmark. Best current full-run table-style average remains `40.5%` vs paper `57.3%` (`-16.9pp`); weighted episode average is `51.6%`. Continue parity debugging, but the strongest remaining clue is task-specific reset/protocol mismatch rather than simple source version, camera padding, camera name, MT1 suite seed, state dimension, very-hard task-id mapping, action clipping, rename/preprocessor duplication, sample count, episode horizon, full-run empty-camera padding, or fixed reset-vector mode alone. |
| RoboCasa365 / RoboCasa | Strong household-manipulation relevance; current leaderboard is explicitly multi-task generalist robot policy evaluation over atomic and composite kitchen tasks. | RoboCasa365 leaderboard reports 50-task Overall / Atomic-Seen / Composite-Seen / Composite-Unseen, e.g. RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`. LeRobot docs expose `CloseFridge` single-task 20ep evaluation for quick iteration, but no public SmolVLA RoboCasa success table has been found yet. | CP25 probe gate, strict RunPod reset/step, SmolVLA `CloseFridge` 20ep, a 3-task 60ep subset, and a lightweight-compatible 5-task 100ep subset now run. The first 5-task scale-up with microwave/stove hit a lightweight-asset blocker at `TurnOnMicrowave`. | Very high. Long-horizon composite tasks are a natural target for planner/verifier/retry. | Keep as secondary household benchmark. Current 5-task subset scored `4/100`, entirely from `TurnOnToaster` `4/20`; useful for our own baseline, but less useful for public SmolVLA reference catch-up. |
| ManiSkill-HAB / MS-HAB | ICLR 2025 low-level home rearrangement benchmark; metrics include subtask success-once and progressive completion. | Paper reports 1000 episodes per evaluation run; subtask success-once examples include TidyHouse Pick val `77.48` RL-per and PrepareGroceries Pick val `72.32` RL-per. | Mac supports ManiSkill CPU simulation and standard rendering, but no Mac GPU simulation. RunPod may need NVIDIA Vulkan/SAPIEN validation; previous PickCube path hit driver issues. | Medium-high. It is useful for low-level verifier/subtask recovery, but less directly aligned to SmolVLA LIBERO policy checkpoints. | Keep as partial, smaller-scale checkpoint. Use for subtask-level research, not as the next main paper number. |
| ManiSkill3 selected Franka tasks | A recent STARE paper reports a direct SmolVLA row on four ManiSkill3 Franka tasks, so it is table-backed outside LIBERO/Meta-World. | Table 2 reports `SmolVLA (fine-tuning)` with Avg `51.5%`: StackCube `12.7`, PushCube `86.3`, PullCube `90.7`, LiftPegUpright `16.3`. It states Octo and SmolVLA use `1000` trajectory samples for SFT per task. Appendix B.5 reports ManiSkill3 evaluation with horizon `30`, `300` episodes, and `5` random seeds; Appendix C.2 reports filtering idle actions before training. | RunPod runtime audit passed with `mani_skill==3.0.1` and `sapien==3.0.3`: `PushCube-v1`, `PullCube-v1`, `StackCube-v1`, and `LiftPegUpright-v1` all reset, stepped, and rendered RGB frames. The SFT feasibility audit downloaded official demos for all four STARE tasks. `PushCube-v1` and `StackCube-v1` have `1000` successful motion-planning trajectories; `PullCube-v1` has `1024` successful RL trajectories; `LiftPegUpright-v1` has `1015` successful RL `pd_ee_delta_pose` trajectories and `993` successful RL `pd_joint_delta_pos` trajectories. All four selected tasks now pass a `count=1` RGB replay and LeRobot conversion smoke with `action_dim=8` and `state_dim=9`; `StackCube-v1` exposes both `base_camera` and `hand_camera`, while the other three smoke conversions expose `base_camera`. Fresh SmolVLA architecture and pretrained `lerobot/smolvla_base` both pass 1-step train smokes. The custom ManiSkill3 eval script now uses the shared `LeRobotPolicyRunner`, prefers `obs.agent.qpos` for state alignment, and sets `max_episode_steps` from the eval horizon. `PushCube-v1` raw count1000/9000-step SFT scored `58.0%` at horizon 100 but `0.0%` at STARE horizon 30. A qpos idle-filtered count1000 dataset with threshold `0.05` compressed `68978` frames to `21802` frames, trained for 9000 steps, and scored `60.7%` over 300 episodes at horizon 30. Threshold ablations scored `0.04 -> 40.0%`, `0.05 -> 68.0%`, and `0.06 -> 62.0%` on 50ep horizon-30 checks. | Medium. Good for low-level manipulation and stage-aware recovery once the policy/training lane exists. | Active table-backed training benchmark. Fine-tuning is a calibration baseline, not the agentic contribution: first make policy-only SFT close to STARE, then compare wrappers on the same checkpoint/budget. The current best paper-protocol PushCube result is `60.7%` vs STARE `86.3%` (`-25.6pp`), so action filtering fixed the horizon-30 collapse but did not fully recover parity; qpos threshold search now points back to `0.05`, not looser or stronger filtering. |
| SafeVLA-Bench | Post-hoc safety benchmark that keeps host LIBERO/RoboCasa rollouts and adds safety metrics. It can turn our RoboCasa rollouts into a stronger paper-style safety table once native success is credible. | Reports SR, Safety, SBU, and VSI. Public page highlights `13-15%` unsafe episodes in high-SR LIBERO baselines and `36-56%` unsafe successful RoboCasa-365 rollouts. | Requires simulator signal instrumentation beyond current success-only logs. It is not a replacement for native benchmark success and needs RoboCasa full/subset rollouts with contact/object/robot signals. | High for an agentic verifier paper angle, but secondary to native success parity. | Track as a post-hoc evaluation layer after RoboCasa native evaluation is stable. |
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
| Official-docs-shaped rerun | Meta-World MT50 | 50 tasks, 10 trials/task | task success | seed `0` + `batch_size=1` + `empty_cameras=0`: Easy `66.1`, Medium `42.7`, Hard `33.3`, Very Hard `16.0`, table avg `39.5`; weighted overall `52.0` | SmolVLA 0.45B Avg `57.3`; Easy `82.5`, Medium `41.8`, Hard `45.0`, Very Hard `60.0` | Completed; closer on `medium`, worse on `hard`/`very_hard`, does not beat the first full MT50 run |
| Full parity rerun | Meta-World MT50 | 50 tasks, 10 trials/task | task success | seed `1000` + `batch_size=10` + `empty_cameras=0`: Easy `67.1`, Medium `43.6`, Hard `23.3`, Very Hard `20.0`, table avg `38.5`; weighted overall `52.0` | SmolVLA 0.45B Avg `57.3`; Easy `82.5`, Medium `41.8`, Hard `45.0`, Very Hard `60.0` | Completed; does not beat the first full MT50 run |
| Parity ablation | Meta-World `very_hard` | 5 tasks, 10-50 trials/task | task success | seed `0` corrected-rename run `20.0`; seed `1000` run `26.0`; seed `1000` + `empty_cameras=0` run `26.0`; standalone `batch_size=10` run `36.0`; fixed reset-vector patch run `16.0`; direct 400-step wrapper patch run `34.0`; LeRobot `v0.5.1` SmolVLA-only run `12.0`; train-config-style 250ep run `24.4`; temporary actual `top` camera run `0.0`; temporary `MT1 seed=1000` run `36.0` | SmolVLA Table 2 Very Hard `60.0` | Standalone 10ep probes can improve, but larger 250ep run drops to `24.4`; actual `top` camera, MT1 suite seed, fixed reset-vector mode, and 400-step horizon are negative controls; only one of five `very_hard` tasks reached `60.0`, so the gap is task/protocol-specific |
| Mac-local pilot | ManiSkill `PickCube-v1` | 20 episodes per policy | task success | random `0.0`, zero `0.05` | RDT repo reports OpenVLA `8%`, DP `40%`, RDT `77.2%` on PickCube under a different trained-policy protocol | Pilot only; not comparable yet |
| Mac-local pilot | ManiSkill-HAB SetTable / PrepareGroceries validation tasks | 20 episodes per task/policy | success-once | random/zero `0.0` | MS-HAB paper-scale runs use success-once over much larger evaluation; paper baselines are trained RL/IL policies | Executable pilot; not comparable policy |
| Current blocker | ManiSkill `PickCube-v1` no-fallback strict | 1 episode | direct target env execution | blocked by `ErrorIncompatibleDriver` on current Mac renderer path | n/a | Needs RunPod/driver-compatible ManiSkill renderer |
| Active training benchmark | ManiSkill3 `PushCube-v1` | count1000 qpos idle-filtered official-demo replay, 9000 SFT steps, five 300-episode eval seeds, horizon `30` | task success | `62.2%` (`933/1500`; seed1000 `60.7`, seed1001 `61.0`, seed1002 `63.0`, seed1003 `64.3`, seed1004 `62.0`), shared runner, qpos state, postprocessor includes `UnnormalizerProcessorStep` | STARE SmolVLA fine-tuning PushCube `86.3%` | Best paper-protocol-aligned PushCube run so far; still `-24.1pp`, but action filtering recovered the raw-data horizon-30 collapse. Seed count now matches STARE; remaining gap is likely source/filter/training protocol rather than eval sample count |
| Negative filter ablation | ManiSkill3 `PushCube-v1` | count1000 qpos idle-filtered official-demo replay, threshold `0.04`, 9000 SFT steps, 50 eval episodes, horizon `30` | task success | `40.0%` (`20/50`), dataset `25104` frames, mean length `25.1`, p90 `30`, max `37` | STARE SmolVLA fine-tuning PushCube `86.3%` | Looser filtering keeps too many slow frames; worse than threshold `0.05` 50ep result `68.0%` |
| Filter ablation | ManiSkill3 `PushCube-v1` | count1000 qpos idle-filtered official-demo replay, threshold `0.06`, 9000 SFT steps, 50 eval episodes, horizon `30` | task success | `62.0%` (`31/50`), dataset `18850` frames, mean length `18.9`, p90 `21`, max `25` | STARE SmolVLA fine-tuning PushCube `86.3%` | Stronger filtering is better than `0.04` but still below threshold `0.05`; keep `0.05` as current best |
| Protocol-gap confirmation | ManiSkill3 `PushCube-v1` | raw count1000 official-demo replay, 9000 SFT steps, 50 eval episodes, horizon `30` | task success | `0.0%` (`0/50`) | STARE SmolVLA fine-tuning PushCube `86.3%` | Shows the prior horizon-100 `58.0%` number was not paper-comparable; most successes were too slow for STARE horizon `30` |
| Earlier training benchmark | ManiSkill3 `PushCube-v1` | raw count1000 official-demo replay, 9000 SFT steps, 50 eval episodes, horizon `100` | task success | `58.0%` (`29/50`), shared runner, qpos state | STARE SmolVLA fine-tuning PushCube `86.3%` | Useful debugging number, but not comparable to STARE horizon `30` |
| Negative training ablation | ManiSkill3 `PushCube-v1` | count1000 official-demo replay, 27000 SFT steps, 50 eval episodes | task success | `52.0%` (`26/50`), final train loss `0.017` | STARE SmolVLA fine-tuning PushCube `86.3%` | Longer SFT lowered train loss but worsened eval, so simple under-training is not the main gap |
| Negative action-chunk ablation | ManiSkill3 `PushCube-v1` | count1000/9000-step checkpoint with edited `n_action_steps`, 20 eval episodes | task success | `n_action_steps=1`: `5.0%`; `10`: `40.0%`; `15`: `40.0%` | STARE SmolVLA fine-tuning PushCube `86.3%` | Keep default `n_action_steps=50`; LIBERO-style chunk sweep does not explain this gap |
| Smaller scale probe | ManiSkill3 `PushCube-v1` | count100 official-demo replay, 1000 SFT steps, 50 eval episodes | task success | `40.0%` (`20/50`), same shared-runner eval after horizon fix | STARE SmolVLA fine-tuning PushCube `86.3%` | Shows data scale matters; not paper-comparable enough |
| Future safety layer | SafeVLA-Bench on RoboCasa | host RoboCasa rollouts plus STL safety scoring | SR / Safety / SBU / VSI | pending | RoboCasa-365 successful rollouts have `36-56%` unsafe-success rates in public summary | Needs safety instrumentation and stable RoboCasa native rollouts |
| Harness infra audit | VLA evaluation harness | config validation only | install/readiness | `vla-eval==0.2.0` installed on RunPod; source `v0.2.0` cloned; `155/155` configs valid; ManiSkill2 benchmark smoke skipped because Docker is absent | Harness leaderboard covers 17+ benchmarks; configs include ManiSkill2/RoboCasa/SimplerEnv/CALVIN | Infrastructure ready enough to plan, not ready to execute benchmarks on this Pod |
| CP25 RunPod strict probe | RoboCasa365 / RoboCasa | import/reset-step gate | runtime readiness | passed for `CloseFridge` reset + one zero-action step | n/a | Runtime gate passed with lightweight assets |
| First strict external benchmark | RoboCasa / LeRobot single-task | 20 episodes, default 1000-step horizon | task success | `lerobot/smolvla_robocasa` on `CloseFridge`: `0/20`, `0.0%` | LeRobot documents this exact single-task 20ep protocol; not a leaderboard value | Completed; pipeline runs but baseline is weak on this task |
| First strict multi-task subset | RoboCasa / LeRobot 3-task subset | 20 episodes per task, 60 total, default 1000-step horizon | task success | `CloseFridge` `0/20`, `OpenCabinet` `0/20`, `OpenDrawer` `0/20`; overall `0/60`, `0.0%` | Same LeRobot RoboCasa evaluation protocol; still not the full RoboCasa365 50-task leaderboard | Completed; protocol-compatible subset with video evidence |
| Lightweight-compatible multi-task subset | RoboCasa / LeRobot 5-task subset | 20 episodes per task, 100 total, default 1000-step horizon | task success | overall `4/100`, `4.0%`; `TurnOnToaster` `4/20`, all other selected tasks `0/20` | No public SmolVLA RoboCasa reference table found | Completed; secondary internal baseline |
| Current scale-up blocker | RoboCasa / LeRobot 5-task subset | intended 20 episodes per task | reset/eval creation | completed first 3 task groups, then blocked at `TurnOnMicrowave` | n/a | Lightweight `objs_lw` asset registry lacks required categories; use fuller assets or a covered task subset |
| Planned full external benchmark | RoboCasa365 / RoboCasa | 50-task benchmark, 3 splits | average task success | pending | RLDX-1 `33.2`, GR00T N1.5 `23.9`, pi0.5 `16.9`, pi0 `14.8`, DP `6.1` | Next scale-up after single-task validation |

For the current goal, do not spend more cycles on LIBERO repeat tables unless
the user explicitly asks. Also do not promote benchmarks without direct paper
or leaderboard reference rows to the main comparison table; use them only as
runtime gates or secondary evidence. The next meaningful work is either:

1. debug Meta-World parity: our run is comparable scale but below Table 2,
   mostly due to `easy` and `very_hard`;
2. keep RoboCasa as a secondary household benchmark, using the current subset
   results as our own baseline rather than a public SmolVLA reference catch-up;
3. use VLA evaluation harness as the next infrastructure lane only on a
   Docker-enabled host and after a SmolVLA model-server adapter exists; and
4. make ManiSkill `PickCube-v1` strict rendering work on a suitable RunPod
   host only after choosing between direct CP24 and harness-based execution.
5. treat ManiSkill3 selected Franka tasks as a separate table-backed
   fine-tuning benchmark, because the public SmolVLA row uses task-specific
   SFT rather than a released general checkpoint. The RunPod runtime smoke now
   passes for all four selected tasks, official demos are downloadable for all
   four STARE tasks with at least `993` successful trajectories per selected
   source, all four tasks now pass 1-episode RGB replay plus LeRobot
   conversion smokes, and pretrained `lerobot/smolvla_base` reaches completed
   1-step train smokes on all four selected tasks with minimal `rename_map`
   camera adaptation. The next real blocker is not raw data conversion or model
   loading; it is scaling SFT/eval to paper-like trajectory counts and actual
   task success measurement.
6. treat SafeVLA-Bench as a post-hoc safety layer after native RoboCasa
   rollouts are stable enough to provide meaningful SR/Safety/SBU/VSI rows.

## External Sources

- ActionX Table 1 SmolVLA LIBERO reference:
  <https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full>
- SmolVLA paper Table 2 Meta-World reference:
  <https://arxiv.org/abs/2506.01844>
- LeRobot Meta-World docs:
  <https://huggingface.co/docs/lerobot/en/metaworld>
- LeRobot Meta-World docs, earlier pinned version used during bring-up:
  <https://huggingface.co/docs/lerobot/v0.4.3/metaworld>
- `lerobot/smolvla_metaworld` model:
  <https://huggingface.co/lerobot/smolvla_metaworld>
- Public SmolVLA LIBERO / Meta-World reproduction-detail issue:
  <https://github.com/huggingface/lerobot/issues/1316>
- RoboCasa365 leaderboard:
  <https://robocasa.ai/leaderboard.html>
- RoboCasa benchmarking documentation:
  <https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html>
- RoboCasa RSS 2024 project:
  <https://robocasa.ai/>
- LeRobot RoboCasa SmolVLA evaluation docs:
  <https://huggingface.co/docs/lerobot/main/robocasa>
- ManiSkill-HAB ICLR 2025 paper:
  <https://openreview.net/pdf?id=6bKEWevgSd>
- ManiSkill macOS installation notes:
  <https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/getting_started/macos_install.html>
- ManiSkill3 selected-task SmolVLA fine-tuning reference:
  <https://openreview.net/attachment?id=qBcgyxDeMM&name=pdf>
- SafeVLA-Bench:
  <https://safevla.org/>
- VLA evaluation harness:
  <https://github.com/allenai/vla-evaluation-harness>
- VLA evaluation harness paper:
  <https://openreview.net/pdf/3e7c55fe021522dc4ff8171ef4082a307238ab67.pdf>
- CALVIN paper:
  <https://arxiv.org/abs/2112.03227>
