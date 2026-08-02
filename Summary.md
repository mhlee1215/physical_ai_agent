# Project Summary

Last updated: 2026-08-02

This repository is currently being used to build and evaluate an agentic
physical-AI wrapper around lightweight vision-language-action policies. The
near-term paper target is RSS SemRob 2026.

## Paper Target

- Target venue: RSS SemRob 2026, the 3rd Workshop on Semantic Reasoning and
  Goal Understanding in Robotics.
- Website: https://semrob.github.io/
- Submission deadline: 18 June 2026, 23:59 AoE.
- Notification: 25 June 2026.
- Camera-ready: 1 July 2026, 23:59 AoE.
- Workshop date: 17 July 2026.
- Format: 4+N or 8+N pages, double-blind, non-archival workshop report.
- Paper workspace: `papers/rss2026_semrob/`.
- Main draft: `papers/rss2026_semrob/main.tex`.
- Overview figure: `papers/rss2026_semrob/figures/overview.png`.
- TODO/checklist: `papers/rss2026_semrob/manuscript_todo_checklist_2026_06_10.md`.

## Current Research Idea

The working method is **Imagine-Then-Act**: an agentic wrapper around a frozen
lightweight VLA policy.

## Durable Implementation Policy

- Source-backed architecture rule: before implementing or changing model
  architecture, a well-known algorithm, or a named robotics/ML technique, first
  search the web for official docs, papers, and similar open-source code. Base
  the implementation on that evidence and record the source-backed rationale in
  the work note, PR, or code comment as appropriate. Do not invent architecture
  changes from scratch when comparable public implementations exist.
- Deterministic photoreal robot dataset work is routed through
  `.agents/skills/robot-photoreal-dataset-rendering/SKILL.md`. The immutable
  contract is the source dataset identity, trajectory, episode timeline,
  simulator state, world assets, and recorded camera calibration. Render
  engine, samples, lighting, and material JSON are mutable only in a new
  append-only derivative. A full render is blocked until an all-episode replay
  preflight and representative-frame canary pass; probe renders and partial
  sidecars are diagnostics, not training datasets.
- Production photoreal recipes keep `preserve_pinhole_renders: false`. The
  512px pre-distortion PNG is an opt-in camera-calibration diagnostic only;
  disabling its retention must not change the final distorted 256px policy
  image or the frozen camera-rig hash.
- Current `v2_5` replay audit is not full-rerender ready. All 300 episodes have
  snapshots and the source contains 50,456 frames, but the default renderer
  exposes 9 RGB object slots while the recorded report expects 3 green cubes,
  296 episode starts match exactly, episodes 2, 41, 96, and 295 have invalid
  start snapshots, and the recorded model dimensions differ from snapshot
  dimensions (`69/60` versus `27/24`). Seed `31010278` therefore requires a
  version-matched environment or explicit state-recovery path before a full
  deterministic render can proceed.
- New schema-v2 SO101 generation recipes use typed
  `common.inspection_gates`. Grip generation requires one authoritative
  `geometry_contact_alignment` gate and may add one
  `camera2_visual_alignment` gate. Camera2 acceptance uses constructive wrist
  roll refinement followed by a pre-close/early-close probe; a failed probe
  skips the expensive full trajectory. `grip_the_cube_v3` and
  `grip_the_cube_v3_align` are the first recipes using both gates and exact
  teacher-time render-replay capture.
- The unified SO101 full-grasp teacher is
  `skill_mode=grip_the_cube_continuous_v1` with
  `trajectory_variant=auto`. It shares one candidate factory between workspace
  probes and dataset export, uses the near-contact solver below 18 cm, tries
  both solver families over 18-22 cm, and uses the mid fixed-jaw solver above
  22 cm. Executable probes verify a stable 10-30 cm radius envelope from the
  shoulder-pan axis; 30.5 cm is a boundary-only canary and 31 cm or farther is
  currently unsupported. The 8-episode representative canary at
  `_workspace/so101_lerobot/grip_the_cube_continuous_v1_canary` passed audit,
  distribution, registry, and viewer gates with `training_ready=true`.
- `grip_the_cube_v4_4` is the current continuous 10-30 cm dataset: 500 train
  and 50 held-out validation episodes, 100% teacher success, unique seeds and
  workspace positions, full absolute and robot-relative yaw-bin coverage, and
  radial total variation 0.016/0.020 from the declared train/validation target.
  Its review report and figure are
  `docs/research/2026_08_02/so101_grip_the_cube_v4_4_distribution.md`.
  The `move`, `align`, and `grip_lift` materializations are supervision-only;
  the parent validation split owns the single 10-episode end-to-end loop test.

The immediate collaboration goal is to produce manuscript-table experiment data
as quickly and efficiently as possible. Research and orchestration choices should
favor real task rows, metrics, videos, summaries, and artifact bundles over
additional infrastructure diagnostics unless those diagnostics are the shortest
path to unblock a data-producing run.

Paper-facing method framing:

1. Read the current observation and task instruction.
2. Generate a portfolio of plausible subgoal or strategy variants with an
   external multimodal planner, primarily Qwen2.5-VL.
3. Feed each strategy prompt into the same frozen SmolVLA executor to produce
   candidate action chunks.
4. Predict or simulate likely outcomes for candidate chunks when possible.
5. Select the most promising candidate before execution.
6. Verify progress after execution and replan if progress is insufficient.

Internal management labels such as `Risk1`, `Risk1-A`, `Risk1-B`, `Risk1-C`,
`Risk2`, and `Risk5` must not appear in paper prose. Translate them into
paper-facing concepts:

- candidate generation
- template prompt portfolio
- external VLM subgoal generation
- simulation-based candidate selection
- context-capture and rendering gate
- state-restoration validation
- oracle/proxy evaluation boundary

## Current Evidence State

### Active SO101 SmolVLA Fine-Tuning Lane

- RunPod SO101 SmolVLA fine-tuning is being monitored through the training
  dashboard and JSONL metrics under the active run directory.
- Checkpoint and run retention policy for this lane is strict. During training,
  keep only `checkpoints/best_closed_loop`, `checkpoints/best_val_loss`, and
  `checkpoints/best_train_loss`; numeric periodic checkpoint directories are
  temporary save candidates and must be pruned after each checkpoint event. Do
  not keep a separate latest checkpoint unless the user explicitly asks for
  crash-recovery archival storage. After a run finishes, delete its local run
  artifact directory if closed-loop test success rate is exactly `0.0`; runs
  without closed-loop evidence are missing-evidence runs, not success-zero runs.
- Supervised validation loss and closed-loop evaluation are mandatory parts of
  the SO101 training lane. The training process should own the sequence:
  train, run supervised evaluation, save checkpoint, then run the scheduled
  loop test. Do not replace this with an external polling monitor.
- The prepared hardware-locked photoreal phase lane is
  `configs/so101/training/grip_the_cube_v3_hardware_locked_photoreal_phases_v1.json`.
  It virtually combines three train splits (approach 300 episodes, alignment
  500, grip/lift 500) and their three held-out validation splits (50 episodes
  each). Primitive phase datasets are supervision-only and do not own loop
  tests. Automatic periodic closed-loop evaluation runs the single continuous
  chain on fixed parent validation episodes
  `[0,6,12,18,24,25,31,37,43,49]`, five each from source camera1 grid bins 9
  and 10. Final automatic evaluation runs only the continuous chain over all
  50 episodes. Do not materialize phase-specific loop-test sidecars.
- The parent continuous-chain loop restores the selected validation episode
  `sim_snapshot` and never resets to teacher state between phases:
  valid-mask proposes a boundary and the geometric verifier confirms it. A
  rejected proposal continues the same phase instead of failing the episode;
  at the phase cap, a passing verifier still advances/completes while a failing
  verifier records `hard_cap`. Phase caps are
  `ceil(reference episode frames * 1.5)`: approach 66, alignment 66,
  grip/lift 134, and continuous chain 266. The actual policy-created state
  enters the next phase. Policy input remains camera1, camera2, motor state,
  and prompt; simulator object pose is verifier-only.
- Current observed best validation checkpoint is `001490` (`val_loss=0.083125`).
  Checkpoint `002682` saved and validated but did not improve
  (`val_loss=0.139522`), so training was stopped as likely overfitting.
- Final current-best closed-loop evaluation
  `final_best_001490_cuda_ep24_step160` completed on CUDA with
  `success_rate=0.0`, `grasp_rate=0.041667`, over 24 episodes. The best
  checkpoint was downloaded locally to
  `_workspace/so101_smolvla_pure/runpod5_best_checkpoints/001490_pretrained_model`;
  final rollout artifacts were downloaded to
  `_workspace/so101_smolvla_pure/runpod5_final_evals/final_best_001490_cuda_ep24_step160`.
- The next SO101 SmolVLA run must start from the checked pipeline contract in
  `docs/so101_smolvla_training_pipeline.md`: 256x256 SmolVLA inputs,
  512-padding preprocessing, sample-time CUDA/MPS augmentation, doubled
  recovery-aware train data, dataset manifest validation, and monitor-managed
  validation/closed-loop/overfit stopping.
- SO101 SmolVLA training configs must include moderate train-time augmentation
  by default: `state_jitter_std=0.003`, `state_dropout_prob=0.02`,
  `image_patch_mask_ratio=0.15`, `gpu_image_augmentation=true`, no camera
  dropout, and no legacy one-patch dropout. Keep validation and closed-loop test
  inputs unaugmented. Do not use teacher-action dropout for BC runs. If action
  chunks need smoothing, use explicit temporal smoothness loss or inference-time
  temporal ensembling/chunk smoothing instead of corrupting labels.
- User policy: SO101/SmolVLA training runs must execute outside the Codex
  sandbox. On macOS, MPS availability checks performed inside the sandbox are
  not authoritative; verify `torch.backends.mps.is_available()` and launch
  MPS training from an unsandboxed/external runtime. Keep deterministic logs,
  TensorBoard events, checkpoints, and monitor artifacts under `_workspace/`.
- User policy: for this `physical_ai_agent` repo, executable local work that
  materially depends on real runtime access should be launched outside the Codex
  sandbox by default. This standing approval covers repo-local training,
  evaluation, closed-loop tests, MuJoCo/MPS runs, TensorBoard, dataset/viewer
  servers, and executable verification commands. Do not pause to ask again in
  chat before these launches; request the required unsandboxed tool execution
  directly and keep logs/artifacts under `_workspace/`. This does not authorize
  destructive cleanup, secret disclosure, remote spending/resource creation, or
  broad network/download actions unless the user explicitly requests them.
- Local SO101 training standard: use
  `docs/so101_local_training_standard.md` before starting local training. The
  current standard lane is `primitive training with qwen validation v1`: one
  SmolVLA checkpoint trained on the three primitive Qwen edge datasets through
  dataset-config `hf_merge_sources`, macOS runtime outside the sandbox with
  MPS, and `--num_workers=0` unless multiprocessing has been proven safe for
  the current dataset wrappers. `scripts/start_so101_training.py` records this
  standard in every dry-run/start/status payload as `local_training_standard`.
- User policy: SO101 training, supervised evaluation, and loop test are all
  mandatory. They must stay runnable on both local macOS and Linux/RunPod
  through the canonical launcher. The runtime contract is
  `macos => mps + MuJoCo glfw` and `linux/RunPod => cuda + MuJoCo egl`; dry-run
  both profiles or run the targeted command-contract tests before treating a
  training PR as ready.
- User policy: SO101 loop tests must record analyzer artifacts by default.
  Qwen-chain loop validation should preserve raw Qwen request/response payloads,
  rollout `policy_rollout_config`, action-chunk metadata, and seed/action/state
  traces sufficient for local media regeneration. Training-time validation
  should not render PNG/MP4 media by default; generate frames/videos locally
  from saved traces when inspection needs them. Disable recording only for
  explicitly labeled lightweight smoke/debug runs, not for validation evidence.
- User policy: SO101 training-time closed-loop validation must run exactly 10
  episodes by default. Keep `--closed-loop-episodes 10` in launcher and monitor
  paths unless the user explicitly asks for a labeled one-off smoke/debug count.
- Current `grip_the_cube_v2` loop termination uses the checkpoint's jointly
  trained valid-mask sidecar on the exact predicted action chunk being executed.
  Joint training feeds the detached SmolVLA `action_hat` to the valid-mask head;
  silently substituting teacher actions is forbidden because inference sees
  predicted chunks.
  Two consecutive re-query termination predictions are required before an
  episode stops, while 200 environment steps remain the hard safety cap. Report
  boundary-index MAE, stop precision/recall, premature-stop rate, and terminal
  sample fraction; per-action valid-mask accuracy alone is not termination proof.
- User policy: local SO101 training visibility must use one TensorBoard process
  only. The default launcher surface is exactly the training process plus one
  TensorBoard process. Do not start extra dashboards, GPU monitors, progress
  monitors, watchers, alternate TensorBoards, or ad hoc polling services unless
  the user explicitly asks for that one-off tool. If the TensorBoard view is
  stale or wrong, stop and restart only that TensorBoard process for the current
  run logdir. Closed-loop tests must be invoked from the training process after
  checkpoint/evaluation events through `scripts/run_so101_training_loop_test.py`,
  not by a separate polling monitor.
- User policy: whenever TensorBoard is started or reported, provide both the
  local URL and the same-Wi-Fi mobile TensorBoard URL.
- User policy: local dataset/viewer dashboards must reuse the current
  user-visible server and port. If the user is viewing
  `http://127.0.0.1:8769/`, update or restart that same `8769` server rather
  than starting another viewer on a fallback port. Starting extra viewer
  servers or changing ports is allowed only when the user explicitly asks for a
  separate server or the original port is owned by an unrelated process and the
  exception is reported clearly.
- User policy: SO101 Live Training Process Safety Contract. Read-only status
  and root-cause checks may inspect `status --json`, `ps`, `tail`, TensorBoard
  events, `stat`, `find`, `du`, `rg`, and `sed` without another confirmation.
  Mutating/destructive actions require explicit user approval immediately before
  execution: `kill`, `pkill`, SIGTERM/SIGKILL,
  `scripts/start_so101_training.py stop`, training restart/resume,
  TensorBoard event deletion/reset, checkpoint or artifact deletion beyond the
  configured retention policy, and overwriting active run state files such as
  `active_training.json`, `train.pid`, locks, or run metadata. Root-cause
  requests mean gather evidence and report first; do not fix, stop, restart, or
  clean up without approval. Never infer liveness from PID only; report process
  alive, `train/loss` scalar advancing, validation/closed-loop cadence, and
  `train.log` stdout progress separately.
- User policy: every SO101 retraining/restart must begin with a clean
  TensorBoard view. Before launching the new training process, delete old
  TensorBoard event files for that run logdir so graphs and images reflect only
  the current run. During an already-active run, preserve the active writer's
  event file and restart only TensorBoard when the display needs refreshing.
- User policy: SO101 training launches are Hydra/Pydantic config-first. Runtime
  defaults live in the selected Hydra entrypoint's `launcher:` block under
  `configs/so101/hydra/training/`, while dataset/training/augmentation/loop
  contracts live in the referenced JSON under `configs/so101/training/`. After
  the user approves a default entrypoint, do not edit that default again unless
  the user directly asks for a default-policy change. Do not reconstruct stable
  behavior by dynamically adding/removing CLI flags for prompt, dataset,
  loop-test cases, RMSE sweep, camera/media, augmentation, action contract,
  checkpoint cadence, runner, device, or ports. CLI overrides are only for
  clearly labeled smoke/debug commands, runtime/port/lock plumbing, or explicit
  one-off user requests; repeated overrides must be promoted into a Hydra
  entrypoint or JSON config before the next run. Code should fail before
  training when required values are missing rather than silently choosing Python
  fallback defaults.
- User policy: loop-test GIFs used as PR/research evidence must be generated
  with the same TensorBoard closed-loop media renderer as
  `closed_loop/<test_id>/rollout_camera1_camera2_episode_*`. Do not attach raw
  rollout GIFs when TensorBoard shows labeled side-by-side camera1/camera2
  policy-input media.
- User policy: SO101 loop-test TensorBoard evidence must include playable
  rollout media and RMSE diagnostics, not only scalar metrics or static images.
  Required tags include the stable user-facing
  `closed_loop/<test_id>/rollout_episode_<NNN>` for every episode and
  `closed_loop/<test_id>/action_rmse_sweep` for action-chunk policies. RMSE
  sweep is mandatory training-result evidence unless a clearly named
  smoke/debug command explicitly disables it. The canonical rollout tag must be
  generated from side-by-side camera1=egocentric and camera2=wrist policy-input
  traces. Do not create duplicate
  `extra/closed_loop/<test_id>/raw_rollout_gif_episode_*` TensorBoard tags.
  Rollout frames should show
  episode/frame, prompt, phase/primitive, active camera/servo state, target
  overlays and dx/dy values when available, terminal success/failure context,
  and a green border on model inference/re-query frames. Training-time
  loop-test result generation must go through the canonical
  `write_so101_training_loop_test_results(run_dir, row, report)` function;
  do not create runner-specific TensorBoard/video writers.
- User policy: `training_config.closed_loop.tensorboard_media.render_test_cases`
  is the explicit allowlist for TensorBoard rollout and train-reference media.
  Tests outside the allowlist still run and publish scalar/RMSE diagnostics but
  must not publish rollout videos. For the phase-split grip experiment, only
  the periodic and final continuous-chain tests are allowlisted. A continuous
  chain uses `chain_rollout_layout=per_episode` and publishes one long,
  time-ordered `closed_loop/<test_id>/rollout_episode_<NNN>` video for each
  evaluated episode. Never concatenate different episodes into one tag and
  never tile phases or episodes into a spatial grid. The matching train
  reference concatenates the same source episode's approach, alignment, and
  grip/lift teacher trajectories in phase order. Every frame preserves the
  side-by-side camera1/camera2 view, episode/frame, prompt, phase, inference
  borders, and termination cues. A failed rollout remains partial at the phase
  where it terminated; visualization must not fabricate later phases.
- Phase-split photoreal runs use
  `observation_renderer.render_policy_inference_only=true`: TensorBoard rollout
  media and RMSE sweeps include fresh policy-query frames only. Their periodic
  RMSE sweep uses `n_action_steps=[5,15,30,50]` and fixes the plot range with
  `action_rmse_sweep.y_axis_max=0.2` so checkpoints remain visually comparable.
  The phase-chain RMSE diagnostic
  runs the aligned approach, alignment, and grip/lift primitive starts, joins
  them into one 177-frame teacher timeline, marks the 44/88 transition frames,
  and reports overall plus per-phase RMSE in one table. Attach it only to the
  parent dataset's single end-to-end continuous-chain loop. Set the renderer
  value to `false` only when the user
  explicitly requests full environment-step rendering.
- User policy: static loop-test train-reference media is generated once per
  run/test by default, controlled by
  `training_config.closed_loop.tensorboard_media.train_reference_frequency`.
  Supported values are `once_per_run`, `every_checkpoint`, and `disabled`.
  `once_per_run` is invalidated when its TensorBoard event files are removed.
  Inference rollout capture is independently controlled by
  `observation_renderer.render_policy_inference_only`. `true` captures policy
  query frames only; `false` captures every environment step.
- User policy: canonical rollout media draws a red outer border on the final
  frame only when the report confirms `termination.reason=valid_mask_stop`, and
  a blue outer border for `termination.reason=env_success`. Hard-cap and failure
  termination frames use neither terminal border.
- User policy: closed-loop success criteria live in the main JSON training
  config. Post-checkpoint evaluation must refresh the selected test case's
  `success_metric` from that config so stale launch-time arguments cannot win.
- User policy: whenever TensorBoard is started or reported, provide the
  TensorBoard access set together: local URL, same-Wi-Fi mobile URL, and an
  external-access URL. Use `cloudflared` quick tunnel for the external URL when
  available; if unavailable, report the external URL as unavailable with the
  reason instead of omitting it.
- User policy: Robot Experiment Manager / dataset viewer servers must be
  launched with `launchctl` through
  `sh scripts/launch_so101_dataset_viewer.sh restart`. Do not use
  `nohup ... &` as the standard path for this server; Codex command cleanup can
  reap that child process after the tool call, which looks like a silent
  server death even when the app did not crash. The durable LaunchAgent label is
  `com.physical-ai-agent.dataset-viewer`; logs live under `_workspace/logs/`.
- User policy: whenever the Robot Experiment Manager / dataset viewer is
  started, opened, or reported, provide the access set together: localhost,
  same-Wi-Fi mobile, and external URL. Keep the external `cloudflared` tunnel
  under `launchctl`, verify its `/api/datasets` response before reporting it,
  and never silently omit one of the three links.
- User policy: Qwen-chain SO101 loop tests must use the valid-mask termination
  head, not fixed-length primitive execution. Provide
  `closed_loop.valid_mask_checkpoint` in dataset config or
  `--closed-loop-valid-mask-checkpoint` on the launcher; missing valid-mask
  configuration is a contract failure for validation loop tests.
- User policy: SO101 training must use camera1 object-position 4x4 grid-bin
  balanced sampling. Every train split must provide or auto-generate
  `meta/camera_grid_bins/observation_images_camera1_4x4_frame0.parquet`; the
  launcher must fail before model setup if it cannot provide this sidecar.
  Validation and closed-loop test splits remain unbalanced for evaluation.
- RunPod experiment-data storage policy: past remote experiment results are not
  needed. Starting now, every new RunPod data-generation, training, evaluation,
  and closed-loop run must end with a local download, local verification, and
  remote deletion of the completed dataset/result/checkpoint artifact. The local
  repo is the preservation point for experiment data. Keep reusable Python
  environments on the Pod's local disk when practical; use the network volume
  only for active handoff/cache, not long-term experiment-data storage.
- SO101 dataset handoff policy: generate or re-export teacher datasets locally,
  verify checksums/manifests, upload the checked LeRobot exports to the HF
  dataset bundle `mhlee1215/so101-nexus-sim-dataset`, then start training
  through `scripts/start_so101_training.py`. Training configs now point at HF
  subfolders; the launcher downloads the configured subfolder from HF before
  training and passes that downloaded path to LeRobot. For private HF dataset
  upload/download, use `HF_TOKEN` from `.env` or the active runtime; do not rely
  on `HF_API_TOKEN` as the canonical token name. For test/debug work only,
  `--use-local-dataset-roots` keeps the config's local `root` values and skips
  HF resolution. RunPod should be used as the CUDA training/evaluation worker,
  not the primary MuJoCo/LeRobot dataset exporter, unless local export is
  blocked or the user explicitly asks for remote generation.
- Multi-train-split SO101 training should use `train_datasets[]` with virtual
  LeRobot concat at training time. Physical merged train roots under
  `_workspace/so101_lerobot_merged/` are fallback/debug artifacts, not the
  canonical training path.
- The Experiment Manager owns persistent, completion-gated dataset-role
  selections at `_workspace/so101_training/dataset_role_selection.json` for
  training, validation, and executable loop-test inputs.
  `scripts/start_so101_training.py start --use-marked-dataset-set` consumes all
  three roles together and fails closed if any role is missing. Catalog pages
  are cached by filter/sort/page in the browser and invalidated after role or
  delete operations.
- Schema-v2 datasets with `splits.<name>.closed_loop` are not complete with a
  start-state report alone. Generation writes a sibling executable contract
  containing the exact prompt, environment, action horizon, and camera renderer;
  completion and catalog role marking fail closed when it is missing. Marked
  loop-test launch also promotes that renderer contract so evaluation cannot
  silently use a preset camera pipeline that differs from the dataset.
- SO101 dataset camera1 contract: `camera1` is the real-hardware-aligned
  `egocentric_cam`, not `top_down`. Current approved camera1 pose is
  `{"type":"free","lookat":[0.245,0.11,0.035],"distance":0.63,"azimuth":270,"elevation":-82,"rotation_degrees":90}`;
  do not change it during data generation without explicit user approval.
- SO101 pre-export dataset material is tracked as code/config through
  `configs/so101/training_datasets/export_recipes.json` and
  `scripts/export_so101_training_datasets.py`. Raw datasets remain local
  `_workspace` artifacts; regenerate them from the recipe and refresh
  `configs/so101/training_datasets/checksums.json` before RunPod upload.
- SO101 dataset lifecycle is append-only by default. Unless the user explicitly
  requests replacement or cleanup in the current turn, every changed dataset
  gets a new versioned recipe, local root, and HF path; existing datasets are
  never overwritten, renamed, repointed, or deleted. Completed datasets are
  registered through `configs/so101/dataset_generation/*.json`
  `splits.<name>.output_root`, not through per-dataset edits to the viewer's
  static training-config list. Sign-off requires export/merge/audit evidence,
  required sidecars/loop starts, presence in `/api/datasets`, and a successful
  sample `/api/frame` response for every intended split. Every schema-v2 split
  must also pass the mandatory post-generation distribution gate and retain
  `meta/distribution/distribution.json`, `.md`, and `.html`. The report covers
  success/seed uniqueness, episode lengths, camera1 visibility and 4x4
  occupancy, prompt counts, and available world-XY/radius/workspace-cell
  statistics. Area-sampled workspace datasets must additionally gate radius
  span, angle span, two-dimensional polar-cell coverage/balance, and median
  nearest-neighbor spacing; source-cell coverage alone is not evidence of a
  spatially broad dataset. Cube orientation is checked in both world yaw and
  robot-relative yaw `(object_yaw - angle_from_base) mod symmetry_period`, so
  rotating the cube and spawn position together cannot masquerade as face
  diversity. JSON statistics produce the canonical Markdown
  record, and the HTML visualization is generated from that Markdown;
  completion verifies the Markdown SHA-256 across all three artifacts.
- SO101 dataset inventory and training readiness now use the shared registry
  `src/physical_ai_agent/so101_dataset_registry.py` through
  `scripts/so101_dataset_registry.py`. The recipe directory is the only
  registry, the generator and viewer share its resolver, and generation is
  complete only when `validate --require-training-ready` passes. Use
  `training-manifest --dataset-id <id>` to retrieve validated train/validation
  roots, episode/frame counts, camera-grid sidecar, and validation loop-start
  before selecting the dataset in a training experiment config.
- User policy: every recipe-backed dataset generation must finish with the
  generator's mandatory `completion:registry-viewer` stage. It runs
  `scripts/verify_so101_dataset_completion.py`, restarts the existing
  launchctl-managed Robot Experiment Manager so no stale Pydantic schema remains
  loaded, and rejects completion unless every selected split is present in
  `/api/datasets` and episode 0/frame 0 returns prompt plus camera1/camera2 from
  `/api/frame`. It also rechecks that the JSON/Markdown/HTML distribution
  report exists and records a passed gate. A registry-only pass is not a
  completed dataset handoff.
- User policy: all newly authored SO101 generation recipes use Pydantic recipe
  schema v2. They explicitly choose `source.mode=from_scratch`,
  `source.mode=from_spawn_catalog`, or `source.mode=from_existing_dataset`.
  `from_spawn_catalog` is the default when only object placement is reused. A
  checked-in catalog may contain seed-free `bin -> world [x, y]` candidates or
  typed workspace candidates with `world_xy_m`, candidate-specific
  `object_yaw_deg`, distance/sampling metadata, and camera-bin diagnostics. It
  must never contain source frames/actions/states/seeds; the new export creates
  all simulator state, trajectories, images, and seeds.
  `from_existing_dataset` additionally chooses `operation=regenerate_teacher`,
  `render_derivative`, or `episode_subset` and lists every source dataset root.
  Episode subsets materialize retained episodes unchanged into a new append-only
  root and rebuild metadata, audit, and sampling sidecars. New
  `grip_the_cube_v1` recipes must configure the exact
  jaw-line versus contacted-face-normal alignment gate. The generator forwards
  geometry and camera2 close-trace limits into teacher episode acceptance, so a
  dataset cannot be declared aligned only by a later filtering pass. Existing
  schema-v1 recipes stay immutable and readable for historical reproduction.
- The `grip_the_cube_v4_2` audit found that all 500 episodes used
  `object_yaw_deg = angle_from_base_deg - 80`, so the world yaw varied while
  the same cube face always faced the robot. Preserve v4.2 for reproducibility
  but do not treat it as orientation-randomized training data. The corrected
  catalog contract stratifies robot-relative cube yaw and rejects coupled yaw
  through pre-export and post-export distribution gates.

### What We Have Learned

- A plain frozen SmolVLA policy is not enough by itself for the paper story; it
  gives us a useful baseline but weak candidate diversity.
- Template-based strategy portfolios can create more diverse SmolVLA action
  chunks. This supports the idea that language-conditioned variation can widen
  the action space of a frozen policy.
- External VLM-generated strategy variants are feasible, but still inconsistent:
  some tasks produce usable variants and policy-generated chunks, while others
  fail schema, grounding, or task-relation checks.
- Simulation-based candidate selection is partially implemented, but current
  selector evidence is still diagnostic. It is not yet strong enough to claim a
  reliable selector improvement.
- The system can now run paper-shaped diagnostic sweeps and produce comparison
  tables, videos, summaries, and artifact bundles. The current evidence is
  useful for deciding the next research direction, not yet for claiming benchmark
  gains.

### Current Claim Boundary

- Do claim: we are testing whether external strategy generation plus frozen VLA
  execution can produce a useful set of candidate first-action chunks.
- Do claim: diagnostic runs show modest candidate diversity improvements on some
  tasks.
- Do not claim yet: benchmark success, deployment-ready rendering, reliable
  selector improvement, or full-task improvement.
- Do not describe internal blockers or management labels in the manuscript.

### Current Research Direction

- The top priority is fast, efficient production of experiment data that can
  support or rule out paper-table claims.
- Prioritize real experimental rows for the paper table: task coverage, candidate
  validity, action-chunk diversity, selector behavior, and full-task or smoke
  outcomes when available.
- Stop spending research time on repeated environment or smoke diagnostics unless
  they are the smallest necessary step before a real data-producing run.
- Use the current shallow-rendering route as a diagnostic data-production lane.
  Treat it as useful for research iteration, not as deployment or benchmark
  renderer evidence.
- Keep full EGL/deployment evidence separate. It can be revisited later, but it
  is not the active path for generating the next paper table.

## Collaboration Model

This summary is for collaborators who need research context, not operational
details. Low-level orchestration rules, thread IDs, install commands, cloud
volume paths, and RunPod lifecycle policy live in
`docs/harness/physical-ai/team-spec.md`.

Working collaboration assumptions:

- New conversations start as ordinary Codex collaboration unless the user
  explicitly asks for postdoc/orchestrator mode.
- The postdoc/orchestrator owns high-level research judgment, claim boundaries,
  and user-facing synthesis.
- Do not modify installed third-party library source code, vendored package
  internals, `site-packages`, TensorBoard frontend bundles, or framework
  runtime files as a normal fix path. Prefer public APIs, repo-local wrappers,
  launcher contracts, logging/tag layout changes, or monitor-side adaptations.
  If a temporary monkey patch or library-source patch is truly unavoidable,
  ask first and document it as temporary with the tested dependency version.
- PM owns coordination and concise status tracking.
- Tech Lead owns code fixes and tests.
- RunPod Manager owns cloud resource setup and handoff.
- RunPod Researcher owns experiment execution and artifact-backed result reports.
- Evaluation Results Manager classifies final evaluation packages before they
  are used as paper evidence.

Research mode means an active cloud experiment environment should be preserved
for researcher/tech-lead debugging unless the user or PM explicitly requests
cleanup or an urgent cost/security risk appears.

Any future orchestration policy change must be recorded in repo docs, not only
in chat. Use this file for collaborator-facing current state and
`docs/harness/physical-ai/team-spec.md` for the detailed operating rules.

## Main vs Legacy

Current state is managed through:

- `Summary.md`
- `AGENTS.md`
- `docs/harness/physical-ai/team-spec.md`
- `papers/rss2026_semrob/main.tex`
- `papers/rss2026_semrob/manuscript_todo_checklist_2026_06_10.md`

Legacy by default:

- dated research notes under `docs/research/`
- old paper outlines, progress ledgers, handoff reports, and HTML reports
- old RunPod worklogs and setup-recovery notes
- old `_workspace/` checkpoint or run artifacts
- prior 40GB network-volume notes and quota history

Legacy artifacts are still useful for provenance, but they must not define the
current plan. If a legacy result matters for the paper, the Evaluation Results
Manager must explicitly classify it before the paper-writing thread uses it.
New development logs should stay out of the main state surface. Put run
artifacts under `_workspace/`, historical notes under `docs/legacy/` or dated
archive locations, and promote only distilled current facts back into this
summary.

## Immediate Next Steps

1. Continue the shallow-OSMesa paper-data lane: run actual manifest rows that
   produce Qwen variants, SmolVLA action chunks, selector outputs, metrics, and
   table-ready artifacts.
2. Do not reopen COMMUNITY/EGL render-device hunting unless the user explicitly
   asks for EGL/deployment evidence.
3. If a run blocks, assign the concrete blocker to Tech Lead or RunPod Manager,
   then resume data production after the fix; do not replace the experiment with
   more smoke diagnostics.
4. Paper writing should continue with conservative claims: describe the method
   clearly, use the overview figure, and avoid claiming final benchmark gains
   until the evaluation manager marks evidence paper-ready.
