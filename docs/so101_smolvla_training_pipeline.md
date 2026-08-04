# SO101 SmolVLA Training Pipeline

This page records the checked training contract for SO101 SmolVLA fine-tuning.
It exists so PRs can test the pipeline before launching another expensive
RunPod run.

## Model Contract

- Base checkpoint: `lerobot/smolvla_base`
- Canonical grip visual inputs: `observation.images.camera1`, `camera2`
- Optional stored compatibility input: `camera3` (inactive unless the config opts in)
- Image tensor shape: `[3, 256, 256]`
- SmolVLA preprocessing resize target: `[512, 512]`
- State input: `observation.state`, shape `[6]`
- Action output: `action`, shape `[6]`
- Training chunk: `chunk_size=50`, `n_action_steps=50`
- Rollout default: `policy_n_action_steps=15`, `policy_num_steps=10`

The enforceable contract lives in
`src/physical_ai_agent/so101_smolvla_pipeline.py`.

## Augmentation And Cache

Training should enter through `scripts/start_so101_training.py` and the selected
Hydra config. Its canonical trainer is
`scripts/lerobot_train_so101_lightning.py`, not a bare `lerobot-train` command.
The trainer keeps the LeRobot policy path intact while adding SO101 sample-time
controls:

- predecoded image cache via `--so101-image-cache-dir`;
- GPU/MPS-side image augmentation via `--so101-gpu-image-augmentation`;
- image color/sharpness jitter and optional affine jitter with
  `--so101-image-affine-degrees` / `--so101-image-affine-translate`;
- camera-level image dropout with `--so101-image-camera-dropout-prob`;
- patch image dropout with `--so101-image-patch-dropout-prob`;
- patch-ratio image masking with `--so101-image-patch-mask-ratio`;
- motor-state jitter with `--so101-state-jitter-std` and per-sample
  `--so101-state-jitter-prob`;
- motor-state dropout with `--so101-state-dropout-prob`.

Image augmentation should run after the batch is moved to CUDA/MPS whenever
possible. CPU-side decoding should be avoided during repeated epochs by using
the predecoded image cache. Training configs should use moderate augmentation
by default. Validation and closed-loop test datasets should remain unaugmented.
Do not use teacher-action dropout for SO101 BC runs; it corrupts the label.

Default SO101 training configs use this moderate preset unless an experiment
explicitly overrides it:

```json
{
  "state_jitter_std": 0.01,
  "state_jitter_prob": 0.35,
  "state_dropout_prob": 0.02,
  "state_dropout_keep_gripper": true,
  "image_camera_dropout_prob": 0.0,
  "image_patch_dropout_prob": 0.0,
  "image_patch_mask_ratio": 0.0,
  "image_color_jitter": true,
  "image_sharpness_jitter": true,
  "image_affine_degrees": 0.0,
  "image_affine_translate": 0.0,
  "gpu_image_augmentation": true
}
```

For `grip_the_cube_v2`, state jitter is a deliberately small in-training
recovery approximation: the observed motor state is perturbed while the clean
teacher action remains the target. It improves tolerance to local state error,
but it does not rerender cameras from the perturbed robot pose. Larger recovery
errors still require simulator-generated correction trajectories.

The canonical grip config presents only `camera1=egocentric` and
`camera2=wrist` to SmolVLA. The duplicate wrist `camera3` remains in stored
datasets for backward compatibility but is removed from the active policy
feature contract and predecoded cache reads. Future three-camera experiments
must opt in by changing `training_config.model_inputs.active_image_features`.

The canonical PEFT path uses LeRobot's official `policy.wrap_with_peft` route:
rank-8 LoRA is attached to VLM text/vision Q/V projections, while the action
expert and state/action/time projections remain fully trainable through
`full_training_modules`. A LoRA run starts from checkpoint weights without
restoring the old full-finetune optimizer state. Evaluators must detect
`adapter_config.json`, construct the base with the adapter directory's saved
policy `config.json`, and then load the adapter. This preserves the camera1/2
contract instead of restoring camera3 from an older base checkpoint.
Resume an existing LoRA run through its retained
`<checkpoint>/pretrained_model/train_config.json`, not `--policy.path`: the
saved train config owns optimizer and scheduler reconstruction. The launcher
must keep `training.policy_repo_id` mapped to output `policy.repo_id`, resolve
the exact retained checkpoint for training state, and reactivate the loaded
PEFT adapter as trainable before constructing the optimizer.

Grip-focused loss uses teacher-derived movement, gripper-transition, and final
valid-horizon weights. Wrist roll additionally uses a circular loss with a
`2*pi` physical period, so equivalent angles across the wrap boundary are not
penalized as far apart. Train and supervised validation use the same loss
composition.

`image_patch_mask_ratio` masks a fraction of an 8x8 image patch grid for every
training image sample. It is distinct from legacy `image_patch_dropout_prob`,
which only masks one random patch for selected samples and should stay `0.0`
unless a specific ablation requires it.

`image_color_jitter` and `image_sharpness_jitter` mirror the LeRobot SmolVLA
image transform recipe while keeping validation and closed-loop inputs
unchanged.

`image_affine_degrees` and `image_affine_translate` are explicit config keys.
For `grip_the_cube_v2` they default to `0.0` so color-referenced prompts do not
also fight geometry-changing augmentation. If an ablation enables affine
augmentation, the implementation uses Torch `affine_grid` / `grid_sample` on
the current tensor device, so CUDA and MPS use the same augmentation path.

## Action Smoothness

Action smoothness is not data augmentation. Do not use action-label dropout to
make predicted chunks smoother. If generated action chunks are jittery, prefer:

- training-side temporal jerk/smoothness loss on the differentiable SmolVLA
  flow action estimate `action_hat = noise - v_t`, such as
  `lambda_smooth * mean((action_hat[t+1] - 2 * action_hat[t] + action_hat[t-1]) ** 2)`,
  starting with a small weight like `0.01`;
- inference-side temporal ensembling or chunk-boundary smoothing for rollout
  execution.

Report smoothness loss separately from supervised BC loss in TensorBoard when it
is enabled.

`action_overlap_consistency` is a train-only auxiliary loss for action-chunk
re-query stability. When enabled, the dataset wrapper attaches the same-episode
teacher action at `t + offset`; training penalizes the overlap between
`action_hat[t][offset:offset+horizon]` and the normalized teacher action
`action[t+offset][:horizon]`. The future teacher target is detached and does not
run a second model forward. It must be declared in config and forwarded by the
launcher with `--so101-action-overlap-consistency-*`.

`action_requery_consistency` is the paired future-observation variant. The
dataset wrapper supplies the same episode at `t + offset`; SmolVLA runs a
second forward on that observation with the same flow time and overlap-aligned
noise. Training applies Smooth L1 between the current prediction tail and the
detached future-observation prediction prefix, so the second forward does not
retain a second backward graph. Keep its offset, horizon, and weight in
the training config and forward them with
`--so101-action-requery-consistency-*`.

For inference-only mitigation, `closed_loop.temporal_ensemble` blends all
postprocessed action chunks that cover the current step with ACT-style
exponential weights. Compare it against baseline on the same checkpoint,
snapshot, seed, prompt, cameras, resolution, and rollout horizon before making
it the default.

## Optional Subgoal Termination Head

The SmolVLA baseline policy must remain unchanged. Train a lightweight
valid-mask head from LeRobot `action_is_pad` labels and enable it only during
closed-loop evaluation with explicit flags.

The head predicts which positions in a predicted SmolVLA action chunk are still
valid for the current subgoal. At inference time the evaluator can execute the
current subgoal until the head predicts padding/end, then switch to the next
subgoal. This is an experimental substitute for a full verifier; it should be
reported separately from policy-only baseline rollouts.

For a single-policy episode without an explicit subgoal sequence, valid-mask
mode is a debounced global early-stop signal. It must inspect the same normalized
action chunk that is postprocessed and executed, require the configured number
of consecutive re-query confirmations, and retain a finite environment-step
hard cap. The canonical `grip_the_cube_v2` lane uses two confirmations and a
200-step cap. Log boundary-index MAE, stop precision/recall, premature-stop
rate, and terminal-sample fraction; token accuracy by itself is insufficient.

Train the head:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_so101_valid_mask_head.py \
  --policy-path <smolvla_checkpoint>/pretrained_model \
  --dataset-root <train_lerobot_root> \
  --dataset-repo-id <train_repo_id> \
  --validation-dataset-root <val_lerobot_root> \
  --validation-dataset-repo-id <val_repo_id> \
  --output-dir _workspace/so101_valid_mask_head/pick_from_top_cube
```

Run closed-loop with the optional chain:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_so101_picklift_smolvla_policy.py \
  --policy-path <smolvla_checkpoint>/pretrained_model \
  --eval-skill-mode pick_from_top_cube \
  --subgoal-chain-mode valid-mask \
  --subgoal-sequence move_over_cube,pick_from_top_cube \
  --valid-mask-checkpoint _workspace/so101_valid_mask_head/pick_from_top_cube/valid_mask_head.pt
```

For ablations, use `--subgoal-chain-mode off` for the baseline and
`--subgoal-chain-mode fixed` to switch subgoals after a fixed number of action
chunks. Closed-loop reports include a `subgoal_chain` section and per-step
subgoal metadata.

## Dataset Expansion

The next train split target is a doubled dataset:

```text
configs/so101/smolvla_pickplace_contact_train100_manifest.json
```

The manifest requires:

- at least `100` train episodes from the previous `50` episode base;
- `[3, 256, 256]` image inputs;
- 6D state/action;
- no sticky grasp;
- recovery/off-nominal frames, because the teacher is privileged and the student
  is a visual policy.

Use `scripts/so101_dataset_manifest.py validate <manifest>` in CI and after
dataset generation. Use `from-export-report` to turn an exporter report into a
validated manifest.

### Move-And-Align V2 Dataset-Generation Augmentation

For `move_and_align_cube_edge`, keep train-time augmentation separate from
dataset-generation augmentation. The v2 dataset is:

```text
move_and_align_cube_edge_train_v2
  - generated teacher trajectories
  - terminal hold included
  - near-target correction included
```

The reproducible export recipe is
`configs/so101/training_datasets/export_recipes.json` entry
`move_and_align_cube_edge_train_v2`. It writes:

```text
_workspace/so101_lerobot/move_and_align_cube_edge_train_v2_300_ego_wrist_256_seed124000
```

The intended composition is 300 episodes:

- about half standard generated teacher trajectories from the home-closed
  start distribution;
- about half near-target correction trajectories that start close to the
  aligned edge pose with joint/XY perturbations;
- 20 terminal hold frames after the target edge-aligned pose.

This is not on-the-fly image/state augmentation. It changes the teacher
trajectory distribution so the policy sees target-near correction and
goal-hold behavior during supervised training.

## Teacher/Student Gap

The pick-place teacher can use privileged simulator state to generate stable
actions. The student only sees SmolVLA runtime inputs. New datasets therefore
must include recovery/off-nominal states, enabled by:

```bash
scripts/export_so101_pickplace_teacher_rollouts_lerobot.py \
  --recovery-steps <N> \
  --recovery-joint-std <STD>
```

Default `--recovery-steps=0` preserves legacy exports. New train data should
turn it on.

## Evaluation Taxonomy

Closed-loop evaluation records must separate the task scenario from the
execution policy. A scenario names what the robot must accomplish; an execution
policy names how the system tries to accomplish it.

Use these SO101 scenario names for training and evaluation tables:

- `pick_up_cube`: grasp the visible cube and lift it up.
- `pick_from_top_cube`: start above the visible cube, then grasp and lift.
- `pick_place_cube`: pick up the small red cube and place it on the blue circle.
- `move_over_cube`: move the gripper over the visible cube.

Use these execution-policy names separately from the scenario:

- `single_smolvla`: one SmolVLA checkpoint runs the whole scenario.
- `fixed_chain`: a hand-specified primitive chain, such as
  `move_over_cube -> pick_from_top_cube`.
- `valid_mask_chain`: a primitive chain that switches subgoals using the
  optional valid-mask head.
- `qwen_edge_chain`: Qwen plans the edge-grasp primitive chain
  `move -> align -> pick_up`, and SmolVLA primitive checkpoints execute robot
  actions. Validation loop tests for this execution policy must use the
  valid-mask termination head via `closed_loop.valid_mask_checkpoint` or
  `--closed-loop-valid-mask-checkpoint`; fixed-length primitive execution is
  not authoritative for this lane.

Do not list `qwen_edge_chain` as a scenario. It is an execution policy/planner
policy for a scenario such as `pick_up_cube`. A closed-loop row should therefore
look like:

```text
scenario=pick_up_cube
execution_policy=qwen_edge_chain
dataset_or_checkpoint=<primitive checkpoint set>
```

## Evaluation Schedule

Use `scripts/monitor_so101_training_dashboard.py` for validation and closed-loop
scheduling. Important flags:

- `--closed-loop-policy best_only`
- `--closed-loop-policy periodic`
- `--closed-loop-policy best_or_periodic`
- `--stop-training-on-overfit`
- `--overfit-patience-checkpoints`
- `--overfit-min-delta`

For `primitive training with qwen validation v1`, supervised validation loss and
Qwen-chain closed-loop validation run on the same checkpoint cadence. In the
standard local setup this means `validation_interval_steps == save_freq ==
steps_per_epoch` and `closed-loop-every-epochs=1`.

Monitored SO101 training must fail before training when checkpoint,
validation, and loop-test cadences drift apart. The effective schedule must
satisfy `validation_interval_steps == save_freq ==
steps_per_epoch * closed_loop_every_epochs`, and total `steps` must be
divisible by `save_freq` so the final checkpoint also has validation and
loop-test evidence. TensorBoard exposes the countdown as
`train/checkpoint_steps_remaining` and
`important/checkpoint_steps_remaining`.

For older expensive CUDA-only closed-loop lanes, best-only or manual final
evaluation may still be used when the user has not requested per-validation
closed-loop evidence.

## Config-First Launch Contract

Do not rebuild the same SO101 training command by adding and removing ad hoc
CLI flags from chat memory. For any repeated or user-facing training run, edit
the main dataset config first, then launch that config unchanged through the
canonical launcher.

Default rule:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_so101_training.py start
```

or, equivalently:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_so101_training.py start --preset default
```

When a run needs a different dataset, prompt, loop-test case, RMSE sweep,
camera/media setting, augmentation setting, action contract, checkpoint
cadence, or closed-loop runner, update the corresponding JSON config under
`configs/so101/training/` first. Dataset contracts, export recipes, checksums,
and dataset-only manifests stay under `configs/so101/training_datasets/`.
Then run the launcher against that
config or preset. Do not silently patch those values at launch time with one-off
CLI flags.

Allowed CLI exceptions:

- `--dry-run`, `--json`, `--replace`, `--lock-file`, ports, host, and runtime
  platform/device selection;
- a clearly labeled smoke/debug command that is not presented as a training
  result;
- explicit user-requested overrides for one-off investigation, recorded in the
  handoff or final report.

If an override becomes useful more than once, move it into the main config or a
named preset before launching another training run.

The post-checkpoint loop runner refreshes each test case's `success_metric` from
that main JSON config when evaluation starts. This lets an approved metric fix
take effect on the next checkpoint without stopping an otherwise healthy live
training process, and prevents a stale launch-time CLI value from overriding the
config source of truth.

## Live Training Process Safety Contract

Read-only status/debug commands are allowed without another confirmation:
`status --json`, `ps`, `tail`, TensorBoard event reads, `stat`, `find`, `du`,
`rg`, and `sed`.

Mutating or destructive actions require explicit user approval immediately
before execution. This includes `kill`, `pkill`, SIGTERM, SIGKILL,
`scripts/start_so101_training.py stop`, restarting/resuming training, deleting
or resetting TensorBoard event data, pruning/deleting checkpoints beyond the
configured retention aliases, deleting artifacts, overwriting
`active_training.json`, `train.pid`, lock files, or active run metadata.

Root-cause analysis requests such as "why", "check", "find cause", or "debug"
mean gather evidence and report it first. Do not fix, restart, stop, or clean
up unless the user explicitly approves that mutation.

Never infer liveness from PID only. Report process alive, `train/loss` scalar
advancing, validation/closed-loop cadence, and `train.log` stdout progress
separately. If training appears hung, collect those four evidence streams and
ask before terminating or restarting anything.

## Closed-Loop Evidence Contract

Loop tests are not complete when they only write scalar metrics or static
input-grid images. Every SO101 closed-loop evaluation used for training
decisions, PR evidence, or research notes must leave enough TensorBoard media
to inspect what the policy actually did.

This section is the stable training-result visualization contract. Treat it as
mandatory harness behavior, not as a reminder. When a training run has a loop
test, the run is not reviewable until the TensorBoard evidence below exists.
Do not re-decide the visualization format per runner, per checkpoint, or per
debug session.

### Training Loop-Test Result Generation Guidelines

During training, do not hand-write TensorBoard loop-test output in a new place.
The training monitor must append the closed-loop metrics row, then call exactly
one canonical result-generation function:

```python
write_so101_training_loop_test_results(run_dir, row, report)
```

That function owns the stable closed-loop TensorBoard/media contract. If a new
runner, evaluator, report field, RMSE diagnostic, overlay, or video format is
needed, update that function or the private helpers it calls. Do not create a
second rollout visualizer, a runner-specific TensorBoard writer, or a separate
ad hoc media attachment path.

Required behavior for the function:

- write scalar loop-test metrics;
- attach the action RMSE sweep plot for action-chunk policies;
- attach canonical user-facing rollout videos at stable tags;
- attach raw/debug media only under `extra/closed_loop/...`;
- attach train reference media when available, using the configured frequency;
- keep camera naming and rollout frame overlays consistent across runners.

Closed-loop starts must be dataset-backed. Official training-time loop tests
must use the train/validation/loop-validation dataset named in the active JSON
config. For picklift-style evaluators, `closed_loop.test_cases[].start_dataset`
or `start_report_path` is not just metadata: the evaluator must restore the
episode `sim_snapshot` from that export report before policy rollout. Do not
create a new random reset state, a new cube color, or a new object distribution
inside the evaluator. If a test cannot restore the configured dataset snapshot,
fail the loop test instead of silently falling back to a fresh reset.

### Phase-Split Primitive And Chain Evaluation

The hardware-locked photoreal phase lane virtually combines approach,
alignment, and grip/lift train splits and validates on the three matching
held-out splits. The phase datasets are supervision-only. At each scheduled
checkpoint, run one continuous approach -> alignment -> grip/lift chain; do
not create separate primitive loop tests.

Periodic evaluation uses fixed validation source episode indices
`[0,6,12,18,24,25,31,37,43,49]`, five from camera1 grid bin 9 and five from
bin 10. Final evaluation uses all 50 validation episodes. The optional oracle
handoff chain is `schedule=manual` and never contributes to official success.

The continuous chain starts from the parent validation episode snapshot only.
Change prompts at
verified phase boundaries without resetting the environment, so the next phase
receives the actual state produced by the previous policy. Valid-mask proposes
the boundary, but the geometric verifier owns advance/completion. If a
valid-mask proposal is rejected, continue the same phase and re-query instead
of ending the episode. At a phase cap, advance/complete when the verifier
passes and record `hard_cap` only when it fails. Policy inputs are camera1,
camera2, motor state, and prompt. Simulator object pose is available only to
the verifier.

Phase caps are derived from each reference episode and checked against the
declared config value: `ceil(reference frames * 1.5)`. The current phase
lengths 44/44/89 therefore use caps 66/66/134 and a 266-step full-chain cap.
Static reference media uses
`train_reference_frequency=once_per_run`; the normal training profile uses
`render_policy_inference_only=true`.

Required TensorBoard outputs:

- `closed_loop/<test_id>/success_rate`, `grasp_rate`, `episodes`, and
  `duration_s`;
- `important/closed_loop_success_rate/<test_id>`;
- `closed_loop/<test_id>/action_rmse_sweep` for action-chunk policies. The
  default sweep is `n_action_steps=[1,3,5,10,15,30,40,50]`. A missing sweep is
  a missing diagnostic, not an acceptable visualization. The only allowed
  exception is a clearly named smoke/debug command that explicitly disables it
  and is not used as training-result evidence;
- one canonical animated rollout media tag per episode:
  `closed_loop/<test_id>/rollout_episode_<NNN>`;
- for an allowlisted continuous phase-chain test, each per-episode tag is one
  time-ordered side-by-side video spanning every phase the policy actually
  reached. A failed episode remains partial at its terminating phase;
- do not publish duplicate side-by-side rollout media under
  `extra/closed_loop/...`;
- when the source train/validation root is available, a matching train
  reference video per episode:
  `closed_loop/<test_id>/train_reference_camera1_camera2_episode_<NNN>`. For a
  phase chain, this reference concatenates the corresponding approach,
  alignment, and grip/lift teacher trajectories for the same source episode.

Train-reference media is static for a run/test pair. Configure its cadence at
`training_config.closed_loop.tensorboard_media.train_reference_frequency`:

- `once_per_run` (default): write it once, record a marker tied to the
  TensorBoard event files, and skip regeneration at later checkpoints;
- `every_checkpoint`: regenerate it with every loop-test result;
- `disabled`: do not generate or require it.

If TensorBoard event files are deliberately cleared, an old marker is not
sufficient evidence and `once_per_run` writes the reference once again.

The inference rollout render cadence is controlled by
`training_config.closed_loop.observation_renderer.render_policy_inference_only`.
With `true`, only fresh policy-query frames are captured. With `false`, every
environment step is captured for an explicitly requested full review. This
setting does not change policy inference/re-query cadence, and the green frame
border remains limited to actual inference frames.

`training_config.closed_loop.tensorboard_media.render_test_cases` explicitly
selects which tests may publish rollout and train-reference media. Tests not in
the allowlist still run and publish scalar/RMSE diagnostics. For phase-split
grip training, only the periodic and final continuous-chain tests are listed.

`training_config.closed_loop.tensorboard_media.chain_rollout_layout` controls
continuous phase-chain presentation. The canonical value is `per_episode`:
each evaluated episode gets one long time-axis video. It must not concatenate
different episodes or build a spatial grid. Each video preserves side-by-side
camera1/camera2, episode/frame, prompt, phase, inference, and termination
overlays.

Phase-split photoreal configs use
`observation_renderer.render_policy_inference_only=true`.
`action_rmse_sweep.render_policy_inference_only=true` applies the same
policy-query-only capture contract to sweep subprocesses, with
`n_action_steps=[5,15,30,50]`. Set the observation renderer value to `false`
only when full-step TensorBoard rendering is explicitly requested.

The user-facing rollout visualization must not change tag names or renderer
priority across runs. The canonical `rollout_episode_<NNN>` tag must be built
from the side-by-side camera1=egocentric and camera2=wrist policy-input trace.
Do not silently swap display labels or camera sources to make a viewer look
plausible. If the evaluator report only contains raw `rollout_gif` paths, the
canonical rollout is missing evidence; fix the evaluator to emit policy-input
traces. Do not mirror raw evaluator GIFs into TensorBoard under
`extra/closed_loop/<test_id>/raw_rollout_gif_episode_*`.

The closed-loop metrics row is written only after
`write_so101_training_loop_test_results(run_dir, row, report)` successfully
attaches the required TensorBoard evidence. If success scalar, canonical
per-episode rollout media (or the configured combined chain montage), required
RMSE sweep, or configured train-reference evidence is missing, the loop-test
subprocess must fail instead of leaving behind a `closed_loop_metrics.jsonl`
row that looks complete.

Every rollout frame should be self-describing:

- episode id and frame/global step;
- prompt, shortened only for display;
- camera names;
- phase/primitive and active camera/servo state when available;
- predicted/ground-truth target overlays and dx/dy values when available;
- success/failure state or enough terminal state information to understand the
  result;
- green border exactly on frames where model inference/re-query happens;
- red outer border on the final frame only when the episode is confirmed to
  terminate with `termination.reason=valid_mask_stop`;
- blue outer border on the final frame when the episode terminates with
  `termination.reason=env_success`. Failure and hard-cap termination do not
  receive either terminal border.

The action RMSE sweep is a default diagnostic for action-chunk policies. Use the
configured sweep, normally `n_action_steps=[1,3,5,10,15,30,40,50]`, and make
the plot x-axis explicit as teacher episode frame index. The y-axis compares
postprocessed policy action against teacher action at that frame. Missing RMSE
sweep media is missing diagnostic evidence, not a complete loop-test result.
For the phase-split grip lane, use `timeline_mode=phase_chain`: evaluate the
same aligned source episode for approach, alignment, and grip/lift, concatenate
their 44/44/89 teacher timelines, mark transition frames 44 and 88, and place
overall plus per-phase RMSE in one table. Restrict this combined sweep to the
single continuous-chain periodic/final test ID.
The teacher-frame curve is teacher-reference drift, not oracle error at the
actual rollout state. Also log re-query-boundary action jump RMSE,
non-boundary action jump RMSE, their ratio, and overlapping chunk prediction
RMSE separately so closed-loop discontinuity is not conflated with state drift.

## Runtime Platforms

The canonical launcher must support both local macOS and Linux/RunPod for
training, supervised validation, and closed-loop evaluation:

- macOS local: `--runtime-platform macos` defaults to `policy.device=mps`,
  `--lightning-accelerator=mps`, and closed-loop `--mujoco-gl=glfw`.
- Linux/RunPod: `--runtime-platform linux` defaults to `policy.device=cuda`,
  `--lightning-accelerator=cuda`, and closed-loop `--mujoco-gl=egl`.
- `--runtime-platform auto` detects the host and chooses the matching profile.
- User-provided `--policy.device`, `--lightning-accelerator`,
  `--lightning-devices`, or `--closed-loop-mujoco-gl` remain explicit
  overrides, but monitored training still fails early if validation or
  closed-loop monitoring is disabled.

The local Mac standard is recorded in
`docs/so101_local_training_standard.md`. Every `scripts/start_so101_training.py`
dry-run/start/status payload includes `local_training_standard` so future
training launches see the standard before acting.

Multi-train-split configs should use `train_datasets[]`. The launcher resolves
each HF subfolder independently and the training script uses a
dataset-balanced random sampler over a virtual `ConcatDataset`; it should not
build a physical merged LeRobot root for the default path. This keeps each
source split equally likely during training even when frame counts differ.

Before launching a long run, inspect the dry-run `runtime_contract`:

```bash
PYTHONPATH=src python3 scripts/start_so101_training.py start --dry-run \
  --runtime-platform macos --dataset-config <config.json> -- --policy.type=smolvla

PYTHONPATH=src python3 scripts/start_so101_training.py start --dry-run \
  --runtime-platform linux --dataset-config <config.json> -- --policy.type=smolvla
```

## Required Tests

Before opening or updating a PR for this pipeline:

```bash
PYTHONPATH=src python3 -B -m unittest \
  tests.test_so101_smolvla_pipeline \
  tests.test_lerobot_sampling_augmentation
```

Before launching RunPod training, run the full suite:

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests
```
