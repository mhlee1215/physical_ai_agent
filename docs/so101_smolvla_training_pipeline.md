# SO101 SmolVLA Training Pipeline

This page records the checked training contract for SO101 SmolVLA fine-tuning.
It exists so PRs can test the pipeline before launching another expensive
RunPod run.

## Model Contract

- Base checkpoint: `lerobot/smolvla_base`
- Visual inputs: `observation.images.camera1`, `camera2`, `camera3`
- Image tensor shape: `[3, 256, 256]`
- SmolVLA preprocessing resize target: `[512, 512]`
- State input: `observation.state`, shape `[6]`
- Action output: `action`, shape `[6]`
- Training chunk: `chunk_size=50`, `n_action_steps=50`
- Rollout default: `policy_n_action_steps=15`, `policy_num_steps=10`

The enforceable contract lives in
`src/physical_ai_agent/so101_smolvla_pipeline.py`.

## Augmentation And Cache

Training should use `scripts/lerobot_train_so101_sampling_aug.py`, not a bare
`lerobot-train` command. The wrapper keeps the LeRobot train path intact while
adding SO101 sample-time controls:

- predecoded image cache via `--so101-image-cache-dir`;
- GPU/MPS-side image augmentation via `--so101-gpu-image-augmentation`;
- image color, sharpness, and affine jitter with
  `--so101-image-affine-degrees` / `--so101-image-affine-translate`;
- camera-level image dropout with `--so101-image-camera-dropout-prob`;
- patch image dropout with `--so101-image-patch-dropout-prob`;
- patch-ratio image masking with `--so101-image-patch-mask-ratio`;
- motor-state jitter with `--so101-state-jitter-std`;
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
  "state_jitter_std": 0.003,
  "state_dropout_prob": 0.02,
  "state_dropout_keep_gripper": true,
  "image_camera_dropout_prob": 0.0,
  "image_patch_dropout_prob": 0.0,
  "image_patch_mask_ratio": 0.15,
  "image_color_jitter": true,
  "image_sharpness_jitter": true,
  "image_affine_degrees": 5.0,
  "image_affine_translate": 0.05,
  "gpu_image_augmentation": true
}
```

`image_patch_mask_ratio` masks a fraction of an 8x8 image patch grid for every
training image sample. It is distinct from legacy `image_patch_dropout_prob`,
which only masks one random patch for selected samples and should stay `0.0`
unless a specific ablation requires it.

`image_color_jitter` and `image_sharpness_jitter` mirror the LeRobot SmolVLA
image transform recipe while keeping validation and closed-loop inputs
unchanged.

`image_affine_degrees` and `image_affine_translate` apply a small random affine
transform after the image batch is on the training device. The implementation
uses Torch `affine_grid` / `grid_sample` on the current tensor device, so CUDA
and MPS use the same augmentation path.

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

## Optional Subgoal Termination Head

The SmolVLA baseline policy must remain unchanged. For agentic subgoal chaining,
train a separate lightweight valid-mask head from LeRobot `action_is_pad`
labels and enable it only during closed-loop evaluation with explicit flags.

The head predicts which positions in a predicted SmolVLA action chunk are still
valid for the current subgoal. At inference time the evaluator can execute the
current subgoal until the head predicts padding/end, then switch to the next
subgoal. This is an experimental substitute for a full verifier; it should be
reported separately from policy-only baseline rollouts.

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
- attach train reference media when available;
- keep camera naming and rollout frame overlays consistent across runners.

Closed-loop starts must be dataset-backed. Official training-time loop tests
must use the train/validation/loop-validation dataset named in the active JSON
config. For picklift-style evaluators, `closed_loop.test_cases[].start_dataset`
or `start_report_path` is not just metadata: the evaluator must restore the
episode `sim_snapshot` from that export report before policy rollout. Do not
create a new random reset state, a new cube color, or a new object distribution
inside the evaluator. If a test cannot restore the configured dataset snapshot,
fail the loop test instead of silently falling back to a fresh reset.

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
- when policy-input camera frames are recorded, debug side-by-side policy-input
  video per episode:
  `extra/closed_loop/<test_id>/rollout_camera_trace_camera1_camera2_episode_<NNN>`;
- when the source train/validation root is available, a matching train
  reference video per episode:
  `closed_loop/<test_id>/train_reference_camera1_camera2_episode_<NNN>`.

The user-facing rollout visualization must not change tag names or renderer
priority across runs. The canonical `rollout_episode_<NNN>` tag must be built
from the side-by-side camera1=egocentric and camera2=wrist policy-input trace.
Do not silently swap display labels or camera sources to make a viewer look
plausible. If the evaluator report only contains raw `rollout_gif` paths, the
canonical rollout is missing evidence; keep raw/debug GIFs only under
`extra/closed_loop/...` and fix the evaluator to emit policy-input traces.

Every rollout frame should be self-describing:

- episode id and frame/global step;
- prompt, shortened only for display;
- camera names;
- phase/primitive and active camera/servo state when available;
- predicted/ground-truth target overlays and dx/dy values when available;
- success/failure state or enough terminal state information to understand the
  result;
- green border exactly on frames where model inference/re-query happens.

The action RMSE sweep is a default diagnostic for action-chunk policies. Use the
configured sweep, normally `n_action_steps=[1,3,5,10,15,30,40,50]`, and make
the plot x-axis explicit as teacher episode frame index. The y-axis compares
postprocessed policy action against teacher action at that frame. Missing RMSE
sweep media is missing diagnostic evidence, not a complete loop-test result.

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
