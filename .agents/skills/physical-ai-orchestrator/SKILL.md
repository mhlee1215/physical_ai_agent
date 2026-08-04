---
name: physical-ai-orchestrator
description: Orchestrate checkpoint-driven development for the agentic physical AI simulation stack, including mandatory executable verification steps.
---

# Physical AI Orchestrator

## When to Use

Use this skill when implementing or reviewing milestones for the agentic physical AI project, especially LIBERO, LeRobot, policy evaluation, planner/verifier/retry, or local VLM checkpoints.

## Required Inputs

- Current checkpoint from `docs/agentic_physical_ai_plan.md`
- Team spec: `docs/harness/physical-ai/team-spec.md`
- Existing code/config under `src/physical_ai_agent/` and `configs/`

## Fast Context Lookup

### SmolVLA Baseline Execution

When the task is "SmolVLA baseline", "baseline SmolVLA 실행 방법",
"SmolVLA 어떻게 돌렸지?", "LIBERO baseline", "RunPod baseline", or any
similar request where the file name is not known, use these anchors first:

```text
docs/research/smolvla_baseline.md
docs/research/smolvla_baseline_handoff_2026_06_07.md
```

The stable alias file `docs/research/smolvla_baseline.md` exists so the user
and future agents do not need to remember the dated handoff filename.

Canonical policy-only baseline:

- model: `lerobot/smolvla_libero`
- environment: LeRobot LIBERO through MuJoCo/robosuite on RunPod Linux GPU
- focused weak-baseline task: `libero_goal`, task id `6`
- seeds: `1200`, `1201`, `1202`
- condition: SmolVLA policy-only rollout, no agentic intervention
- important mapping:
  `--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'`

Minimal command shape:

```bash
cd /workspace/physical-ai/physical_ai_agent
PY=/root/physical-ai/envs/lerobot_py312/bin/python
OUT=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/libero_goal_task6_baseline_example/baseline_seed1200
mkdir -p "$OUT"

$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_no_progress \
  --intervention-mode none \
  --semantic-min-step 220 \
  --semantic-window 20 \
  --semantic-progress-threshold 0.002 \
  --output_dir="$OUT/eval_logs" \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[6]" \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.empty_cameras=0 \
  --seed=1200 \
  > "$OUT/instrumented_eval.log" 2>&1
```

Do not change this focused baseline unless the user explicitly asks. Use it as
the fixed policy-only reference while improving or comparing agentic wrappers.

When the task mentions the real SO-100 follower, camera indexes `0`/`1`/`3`,
Innomaker U20CAM, iPhone observer camera, real green-object grasp/relocation,
or the real agentic SmolVLA loop, open this skill first:

```text
.agents/skills/real-so100-agentic-smolvla/SKILL.md
```

Then read its hardware contract reference before any hardware action or code
change that could affect the real robot:

```text
.agents/skills/real-so100-agentic-smolvla/references/hardware_contract.md
```

When the task mentions SmolVLA baseline execution, LIBERO baseline parity,
RunPod SmolVLA evaluation, `lerobot/smolvla_libero`, baseline command,
baseline seed protocol, camera-name mapping, policy-only reference, or
"how did we run SmolVLA?", open this handoff first:

```text
docs/research/smolvla_baseline_handoff_2026_06_07.md
```

That file records the frozen focused baseline, RunPod environment variables,
required camera mapping, exact `lerobot/smolvla_libero` command shape, current
`libero_goal` task 6 seed results, and the wrapper comparison commands.

Search aliases for this context:

```text
smolvla baseline
smolvla execution
LIBERO baseline
RunPod baseline
policy-only SmolVLA
lerobot/smolvla_libero
camera_name_mapping
baseline handoff
```

### Robot Photoreal Dataset Rendering

When the task mentions photoreal robot datasets, deterministic rerendering,
Blender/Cycles dataset images, SO101 render profiles, camera-preserving dataset
conversion, or MyCobot photoreal output, open and follow:

```text
.agents/skills/robot-photoreal-dataset-rendering/SKILL.md
```

That skill owns the source/replay/render contract. The PM must track the source
identity, replay-preflight verdict, canary evidence, render profile, full-render
artifact, image-only derivative build, registry validation, and dashboard
frame/playback checks as separate gates. A probe render or incomplete frame
sidecar is diagnostic evidence and must not be reported as a completed training
dataset.

### SO101 SmolVLA Fine-Tuning

When the task mentions SO101 SmolVLA training, RunPod SO101 fine-tuning,
SO101 dataset configs, augmentation, validation loss, closed-loop rollouts,
action chunk jitter, or smoothness, open these anchors first:

```text
docs/so101_smolvla_training_pipeline.md
configs/so101/training_datasets/README.md
docs/harness/physical-ai/team-spec.md
```

Durable SO101 fine-tuning contract:

- SO101 dataset creation is append-only unless the user explicitly authorizes
  overwrite, replacement, rename, or deletion in the current turn. A changed
  teacher, filter, camera contract, prompt, split composition, or quality gate
  must produce a new versioned recipe and a new local/HF dataset path. Never
  pass `--overwrite`, remove an existing root, rewrite an existing HF split, or
  repoint an established dataset name as an inferred convenience;
- the single durable registration path for newly generated SO101 datasets is a
  versioned JSON recipe under `configs/so101/dataset_generation/` with every
  generated split declared as `splits.<name>.output_root`. The Robot Experiment
  Manager discovers completed recipe-backed roots automatically. Dataset
  contracts continue to define stable semantics, and training configs select
  active train/validation inputs; neither should be used as an ad hoc viewer-
  only registry;
- a dataset-generation request is complete only after: the versioned recipe
  exists; unique output roots are generated; export/merge/audit reports pass;
  required camera-grid sidecars and loop starts are produced; the mandatory
  distribution stage writes and passes `meta/distribution/distribution.json`,
  `.md`, and `.html`; overlap checks
  against protected datasets pass when requested; `/api/datasets` lists every
  intended split; and `/api/frame` succeeds for at least episode 0/frame 0 of
  each split. Report the root, episode/frame count, size, sidecar status, and
  viewer URLs as one verified localhost, same-Wi-Fi mobile, and external access
  set. A raw directory alone is not a completed handoff;
- every recipe run must retain the generator's final
  `completion:registry-viewer` stage. That stage invokes
  `scripts/verify_so101_dataset_completion.py`, restarts the existing
  launchctl-managed Robot Experiment Manager to evict stale schema code, and
  fails unless each selected split is listed by `/api/datasets` and its first
  frame exposes prompt, camera1, and camera2 through `/api/frame`. It must also
  reject missing or failed distribution reports. Do not bypass
  it with `--no-restart-viewer` outside an explicitly labeled unit/debug test;
- author every new SO101 recipe with Pydantic `schema_version: 2`. Select
  `source.mode=from_scratch` when simulator state and trajectories are newly
  constructed without reusable placement inputs. Select
  `source.mode=from_spawn_catalog` when only object placement is reused. The
  checked-in catalog may use seed-free camera-bin `[x, y]` candidates or typed
  workspace candidates containing XY, candidate-specific yaw, and sampling
  weight, but never source frames/actions/states/seeds. Generated splits create
  new state, trajectories, images, and seeds. Select
  `source.mode=from_existing_dataset` with exact roots and
  `operation=regenerate_teacher`, `render_derivative`, or `episode_subset` only
  when the dataset itself is an input. An episode subset writes a new append-only
  root while preserving retained frame/action/state values and source episode
  provenance. For new `grip_the_cube_v1` recipes, require one typed
  `geometry_contact_alignment` entry in `common.inspection_gates`; allow at
  most one `camera2_visual_alignment` entry and use
  `constructive_refine_then_probe` before full episode export. Forward all
  declared thresholds into exporter acceptance. Keep schema-v1 recipes
  immutable and readable for reproduction;
- schema-v2 generation always runs
  `scripts/build_so101_dataset_distribution_report.py` after sidecar/loop-start
  materialization and before final audits/completion. Inspect the Markdown or
  standalone HTML report before sign-off. The canonical order is JSON
  statistics -> Markdown research record -> HTML visualization; completion
  verifies the recorded Markdown SHA-256 in both JSON and HTML. At minimum,
  enforce episode count, teacher success, unique seeds, camera1 visibility,
  and any recipe-declared workspace-cell/radial-distribution thresholds.
  For area-sampled workspaces, source-cell coverage is insufficient by itself:
  also gate radius span, angle span, two-dimensional polar-cell
  coverage/balance, and median nearest-neighbor spacing so a narrow arc cannot
  be reported as broad spatial coverage. When cube orientation is randomized,
  also gate the absolute yaw distribution and the robot-relative face
  orientation `(object_yaw - angle_from_base) mod symmetry_period`. Do not
  accept absolute yaw span as a proxy for face diversity: by default generate
  robot-relative yaw strata directly, and reject a deterministic coupling
  between placement angle and the face presented to the robot unless the
  recipe declares a reviewed physical-feasibility exception;
- the shared registry implementation is
  `src/physical_ai_agent/so101_dataset_registry.py`, and the only operator CLI
  is `scripts/so101_dataset_registry.py`. Run `list` for inventory,
  `validate --require-training-ready` as the generation completion gate, and
  `training-manifest --dataset-id <id>` to obtain the exact train/validation
  roots, counts, grid sidecar, and closed-loop start that a training config can
  consume. The generator must run the same readiness gate after its pipeline;
- do not add each generated dataset to the viewer's static `TRAINING_CONFIGS`
  list. Active training datasets belong in `configs/so101/training/*.json`;
  durable generated datasets belong in generation recipes. Temporary smoke
  roots may use `SO101_TEMP_DATASETS`, but that mechanism is never the canonical
  registration path for a completed dataset;
- training configs use explicit train-time augmentation by default. For
  canonical `grip_the_cube_v2`, affine is off:
  `state_jitter_std=0.01`, `state_jitter_prob=0.35`,
  `state_dropout_prob=0.02`,
  `image_patch_mask_ratio=0.0`, `image_affine_degrees=0.0`,
  `image_affine_translate=0.0`, `image_noise_std=0.01`,
  `image_motion_blur_prob=0.1`, `gpu_image_augmentation=true`;
- canonical grip training exposes only camera1 egocentric and camera2 wrist to
  the policy; camera3 is a configurable future opt-in, not a duplicate active
  input;
- canonical grip LoRA uses LeRobot PEFT with rank 8 on VLM text/vision Q/V
  projections and keeps the action expert plus state/action/time projections
  fully trainable. PEFT evaluators must restore the declared base and adapter;
  the base must use the adapter directory's saved policy config so active
  cameras cannot silently revert;
- resume a LoRA run from the retained checkpoint's saved `train_config.json`
  and exact training-state directory. `policy_repo_id` maps to output
  `policy.repo_id`, never input `policy.path`; reactivate the loaded adapter as
  trainable before optimizer restoration and verify resumed step plus trainable
  parameter count;
- a checkpoint-triggered loop-test subprocess must receive
  `SO101_CHECKPOINT_DIR` as an exact `--checkpoint-name`; never rescan and
  evaluate unrelated retained aliases inside that event;
- validation and closed-loop test inputs remain unaugmented;
- SO101 training, supervised evaluation, and loop test are mandatory phases;
  the training process owns the sequence and invokes loop tests directly after
  checkpoint/evaluation events;
- phase-split SO101 experiments use virtual `train_datasets[]` and
  `validation_datasets[]`. Their official suite is three primitive tests plus
  one continuous chain. Periodic evaluation uses ten fixed held-out episodes
  balanced five/five over the approved source bins; final evaluation uses the
  complete validation set. Oracle handoff is a manual diagnostic only;
- phase loop starts restore the selected validation `sim_snapshot`. In an
  official continuous chain, do not reset between phases: valid-mask proposes
  termination and the phase verifier confirms it. A rejected stop proposal
  continues the same phase. At a phase cap, a passing verifier still
  advances/completes and only a failing verifier records `hard_cap`. Derive the
  phase cap as `ceil(reference frames * reference_length_multiplier)` and fail
  when it disagrees with the explicit configured cap. The current phase lane
  uses multiplier 1.5 and caps 66/66/134/266 for
  approach/alignment/grip-lift/chain. The policy-created state flows into the
  next prompt. Camera1, camera2, motor state, and prompt are policy inputs;
  simulator object pose is verifier-only;
- checkpoint, validation, and closed-loop timing must be one aligned step
  event: `validation_interval_steps == save_freq ==
  steps_per_epoch * closed_loop_every_epochs`, and total `steps` must be
  divisible by `save_freq`. Monitored runs must log
  `train/checkpoint_steps_remaining` and
  `important/checkpoint_steps_remaining`;
- training-time SO101 closed-loop validation defaults to exactly 10 episodes;
  keep `--closed-loop-episodes 10` unless the user explicitly requests a
  labeled one-off smoke/debug count;
- valid-mask early termination must consume the exact normalized action chunk
  that will be postprocessed and executed. Require explicit consecutive
  re-query confirmations and retain an environment-step hard cap. The canonical
  `grip_the_cube_v2` lane uses two confirmations and a 200-step cap, and logs
  boundary MAE, stop precision/recall, premature-stop rate, and terminal sample
  fraction instead of trusting slot accuracy alone. During joint training,
  feed the detached SmolVLA predicted `action_hat` to the valid-mask head;
  teacher actions define valid/padding labels but are not a silent substitute
  for the predicted-chunk input;
- when a SO101 training dataset has camera1 object-position grid-bin sidecar
  metadata, use grid-bin balanced sampling for training; keep validation and
  closed-loop sampling unbalanced;
- SO101 training launches are Hydra/Pydantic config-first. For repeated or
  user-facing runs, edit the Hydra entrypoint under
  `configs/so101/hydra/training/` and the referenced JSON config under
  `configs/so101/training/` first, then run
  `scripts/start_so101_training.py start --hydra-config <name>` or a named
  preset against that config unchanged. Do not rebuild stable behavior with ad
  hoc CLI flags for prompt, dataset, loop-test cases, RMSE sweep, media,
  augmentation, action contract, checkpoint cadence, or runner. One-off CLI
  overrides must be clearly labeled smoke/debug or explicitly requested by the
  user; repeated overrides must be promoted into the config or preset;
- SO101 launcher/runtime defaults live in the selected Hydra entrypoint's
  `launcher:` block. After the user approves a default entrypoint, do not edit
  that default again unless the user directly asks to change the default
  policy. Do not keep hidden Python fallback values for prompt, dataset,
  loop-test cases, RMSE sweep, media resolution, augmentation, action contract,
  checkpoint cadence, runner, device, or ports; missing required values should
  fail before training instead of being guessed in code;
- SO101 training loss, augmentation, and runtime knobs must have an explicit
  config-to-CLI contract. If a training code path exposes a knob such as action
  prefix weighting, teacher-importance weighting, smoothness, consistency,
  action-overlap consistency, visual-servo loss, augmentation, or valid-mask loss, the selected training
  config/default config must declare the value and the launcher must forward it
  explicitly. Zero/disabled values should be explicit when they define the
  experiment contract. Add or update tests when adding a knob so code and
  config cannot silently drift;
- `action_overlap_consistency` uses the same-episode future teacher action as a
  detached target and does not add a second model forward;
- `action_requery_consistency` is a separate, explicitly configured train-only
  loss. It runs the same policy on the same-episode `t+offset` observation,
  aligns the flow time/noise over the overlapping horizon, and penalizes the
  disagreement between the current chunk tail and detached future-observation
  chunk prefix. Gradients flow through the current prediction only so the
  second forward does not double the retained backward graph. Do not silently
  substitute it for teacher overlap consistency;
- SO101 training config edits must pass the Pydantic/Hydra validator
  `PYTHONPATH=src .venv/bin/python scripts/validate_so101_training_configs.py`;
  the launcher also validates the selected config before command construction;
- local SO101 training launches default to exactly two processes: the training
  process and one TensorBoard process. Extra dashboards, GPU monitors, progress
  monitors, watchers, alternate TensorBoards, or polling helpers require an
  explicit user request;
- TensorBoard must be launched with `--reload_multifile true` for SO101
  training runs. The active training writer and post-checkpoint validation or
  loop-test writers can append separate event files in the same run logdir; the
  launcher must make TensorBoard poll all active event files instead of only the
  newest one;
- SO101 Live Training Process Safety Contract: read-only status/debug commands
  are allowed without another confirmation, including `status --json`, `ps`,
  `tail`, TensorBoard event reads, `stat`, `find`, `du`, `rg`, and `sed`.
  Mutating or destructive actions require explicit user approval immediately
  before execution. This includes `kill`, `pkill`, SIGTERM, SIGKILL,
  `scripts/start_so101_training.py stop`, restarting/resuming training,
  deleting or resetting TensorBoard event data, pruning/deleting checkpoints
  beyond the configured retention aliases, deleting artifacts, overwriting
  `active_training.json`, `train.pid`, lock files, or active run metadata.
  Root-cause analysis requests such as "why", "check", "find cause", or
  "debug" mean gather evidence and report it first; do not fix, restart, stop,
  or clean up unless the user explicitly approves that mutation. Never infer
  liveness from PID only. Report process alive, `train/loss` scalar advancing,
  validation/closed-loop cadence, and `train.log` stdout progress separately.
  If training appears hung, collect those four evidence streams and ask before
  terminating or restarting anything;
- TensorBoard reports must include both the local URL and the same-Wi-Fi mobile
  URL;
- SO101 loop-test TensorBoard evidence must include animated rollout media and
  RMSE diagnostics, not only static images or scalar success rates:
  `closed_loop/<test_id>/rollout_episode_<NNN>` for every episode,
  `closed_loop/<test_id>/action_rmse_sweep` for action-chunk policies, and
  matching train reference when policy-input camera frames are available. The
  train reference defaults to
  `training_config.closed_loop.tensorboard_media.train_reference_frequency=once_per_run`;
  do not regenerate it at every checkpoint unless the config explicitly says
  `every_checkpoint`. A TensorBoard cleanup invalidates its event-file marker,
  so the next loop test writes it once again. The
  `observation_renderer.render_policy_inference_only` config controls rollout
  frame capture. `true` captures fresh policy-query frames only and `false`
  captures every environment step. It must not change the model's
  `n_action_steps` re-query cadence. The action RMSE sweep is mandatory training-result evidence
  unless a clearly
  named smoke/debug command explicitly disables it. The canonical rollout tag
  must be generated from the labeled camera1=egocentric/camera2=wrist
  policy-input trace. Do not mirror evaluator raw GIFs into TensorBoard under
  `extra/closed_loop/<test_id>/raw_rollout_gif_episode_*`;
- use `tensorboard_media.render_test_cases` as the explicit rollout and
  train-reference media allowlist. Tests outside it still run scalar and RMSE
  diagnostics but publish no rollout video. Phase-split grip runs allowlist only
  the periodic and final continuous-chain tests;
- continuous phase-chain tests use
  `tensorboard_media.chain_rollout_layout=per_episode` to publish one long
  `closed_loop/<test_id>/rollout_episode_<NNN>` video per evaluated episode.
  Never concatenate different episodes into one tag or tile phases/episodes
  into a spatial grid. Build the matching train reference by concatenating the
  same source episode's approach, alignment, and grip/lift teacher
  trajectories. Keep failed policy rollouts partial at their real terminating
  phase; never fabricate later phases. Use the canonical side-by-side camera
  renderer;
- phase-split photoreal configs use
  `observation_renderer.render_policy_inference_only=true`; configure RMSE
  sweeps the same way with `n_action_steps=[5,15,30,50]`. Set the observation
  renderer value to `false` only for explicitly requested full environment-step
  media;
- phase-split action RMSE uses `timeline_mode=phase_chain`: evaluate the same
  aligned source episode for approach, alignment, and grip/lift, concatenate
  the three teacher timelines, mark phase transitions, and show overall plus
  per-phase RMSE in one table. Allowlist this diagnostic to periodic/final
  continuous-chain tests so primitive tests do not rerun the same sweep;
- action-chunk loop-test diagnostics must report teacher-reference drift,
  re-query-boundary action jump, non-boundary action jump, their ratio, and
  overlapping chunk prediction RMSE as separate metrics. Teacher-reference
  drift is not an oracle error at the rollout state and must not be labeled as
  one;
- loop-test GIF/video frames must show episode/frame, prompt, camera names,
  phase/primitive or active camera when available, target overlays when
  available, dx/dy values when available, terminal success/failure context, and
  a green border on model inference/re-query frames, plus a red outer border on
  the final frame only when `termination.reason=valid_mask_stop` is confirmed,
  or a blue outer border when `termination.reason=env_success` is confirmed;
  hard-cap and failure termination receive neither terminal border. Repeat a
  colored terminal frame briefly so TensorBoard GIF encoding preserves it;
- training-time loop-test result generation must call
  `write_so101_training_loop_test_results(run_dir, row, report)` instead of
  creating runner-specific TensorBoard/video writers;
- do not use teacher-action dropout in behavior cloning;
- action chunk jitter is handled through explicit predicted-action temporal
  smoothness loss or inference-time temporal ensembling/chunk smoothing, not by
  corrupting teacher labels.

Short anchor:

```bash
cd /workspace/physical-ai/physical_ai_agent
PY=/root/physical-ai/envs/lerobot_py312/bin/python

$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_no_progress \
  --intervention-mode none \
  --semantic-min-step 220 \
  --semantic-window 20 \
  --semantic-progress-threshold 0.002 \
  --output_dir="$OUT/eval_logs" \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[6]" \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.empty_cameras=0 \
  --seed=1200
```

## Workflow

1. Select the smallest unchecked checkpoint that moves the MVP forward.
2. Implement only the code, config, docs, and tests needed for that checkpoint.
3. Register or update the executable verification command in the team spec when the checkpoint adds a new runnable path.
4. Run the required verification command before marking the checkpoint complete.
5. Report whether the checkpoint is passed, failed, or blocked by missing external dependencies.

## Image Artifact Verification

When a checkpoint or milestone produces images, videos, overlays, contact
sheets, GIFs, plots, or visual reports, do not call it passed from file
existence, JSON metrics, or command exit status alone.

Required behavior:

- Visually inspect representative artifacts before claiming success.
- State what was inspected in the response.
- If the visual artifact contradicts the claim, report failed or blocked even
  when the manifest says `passed`.
- For projection/overlay work, treat wrong-object, wrong-side, vertical mirror,
  center-bias, and visually implausible points as bugs, not acceptable evidence.
- Paper-facing visual evidence requires both same-timestep metadata provenance
  and human-visible semantic alignment.

## Checkpoint 01 Required Verification

Always run:

```bash
sh scripts/checkpoint_01.sh
```

When claiming LIBERO itself is executable, also run:

```bash
sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env
```

When claiming checkpoint 01 works on the target Mac, run:

```bash
sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco
```

If the Mac-local command fails because MuJoCo is missing, treat checkpoint 01 as not fully complete. If the LIBERO strict command fails on macOS because LIBERO/LeRobot requires Linux, treat that as a future Linux/cloud blocker rather than a Mac-local checkpoint failure.

## Expected Outputs

- Updated repo files
- Verification command results
- Clear next blocker or next checkpoint

## Validation Notes

- Do not claim Mac-local simulation readiness from import-free tests alone.
- Do not install or download simulation dependencies without user approval.
- Keep validation commands deterministic and repo-local.
