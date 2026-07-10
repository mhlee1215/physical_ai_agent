# SO101 Local Training Standard

This is the local standard for SO101 SmolVLA training on the user Mac.

## Name

`primitive training with qwen validation v1`

Use this name for the current local primitive-training lane unless the user
explicitly chooses another experiment.

## Runtime

- Launch local SO101/SmolVLA training outside the Codex sandbox.
- Use the repo virtualenv Python, not bare system Python:
  `/Users/minhaeng/workspace/physical_ai_agent/.venv/bin/python`.
- On macOS, use MPS:
  `--runtime-platform macos`, `--policy.device=mps`, and
  `--lightning-accelerator=mps`.
- Do not trust MPS availability checks performed inside the Codex sandbox.
- Use `--num_workers=0` for local macOS training unless a specific run proves
  multiprocessing is safe with the current dataset wrappers.

## Dataset

- Use dataset-config virtual merge declarations, not ad hoc manual pre-merge
  commands.
- SO101 training must use camera1 object-position 4x4 grid-bin balanced
  sampling. Every train dataset entry must have
  `grid_bin_sidecar` pointing to
  `meta/camera_grid_bins/observation_images_camera1_4x4_frame0.parquet`.
  `scripts/start_so101_training.py` is responsible for creating this sidecar
  when it is missing and for forwarding it to the Lightning trainer. A local
  training run that cannot provide a grid-bin sidecar must fail before model
  setup instead of silently falling back to ordinary random sampling.
- Validation and closed-loop test splits are not rebalanced by the sampler;
  their sidecars may be generated for diagnostics, but evaluation should keep
  the declared validation/test distribution intact.
- For the current Qwen primitive lane, the config is:
  `configs/so101/training/qwen_edge_primitives.json`.
- The primitive train splits are used together in one training run:
  `move_over_cube_edge/train`, `align_fixed_jaw_cube_edge/train`, and
  `grip_from_edge_cube/train`.
- The validation splits are also composed through `hf_merge_sources`.

## Evaluation

- Local training visibility must use one TensorBoard process only. The default
  launcher surface is exactly the training process plus one TensorBoard process.
  Do not start extra dashboard, GPU monitor, progress monitor, watcher,
  alternate TensorBoard, or ad hoc polling service unless the user explicitly
  asks for that one-off tool. If the TensorBoard view is stale or wrong, stop
  and restart only the TensorBoard process attached to the current run logdir.
  Always report the TensorBoard access set together: local URL, same-Wi-Fi
  mobile URL, and external-access URL. Use a `cloudflared` quick tunnel for the
  external URL when available; if unavailable, report the external URL as
  unavailable with the reason.
- Every SO101 retraining/restart starts with a clean TensorBoard view. Before
  launching the new training process, delete old TensorBoard event files for
  that run logdir. During an already-active run, preserve the active writer's
  event file and restart only TensorBoard when the display needs refreshing.
- Training, supervised evaluation, and loop test are all mandatory. The training
  process owns the sequence; do not use an external polling monitor as the
  default mechanism for discovering checkpoints or triggering loop tests.
  Checkpoint-triggered loop tests use the one-shot
  `scripts/run_so101_training_loop_test.py` entrypoint from inside the training
  callback.
- `pick_up_cube` is the scenario.
- `qwen_edge_chain` is the execution/validation policy.
- Qwen routes `move -> align -> pick_up`; SmolVLA produces robot actions.
- The final Qwen validation command should route all primitive prompts to the
  same trained checkpoint with `--policy-path`.
- Do not use sentinel values such as `--closed-loop-every-epochs=999` to
  effectively disable training-time closed-loop checks.
- For the current local lane, every validation-loss checkpoint must also run a
  closed-loop test. Keep `validation_interval_steps == save_freq ==
  steps_per_epoch * closed_loop_every_epochs`.
- Because validation loss is computed every epoch in this lane, checkpoint save
  and closed-loop test cadence must also be every epoch.
- Do not pass `--closed-loop-every-epochs 2` for normal user training requests;
  use the launcher default of `1` unless the user explicitly asks otherwise.
- Use `--closed-loop-policy periodic` or `--closed-loop-policy best_or_periodic`;
  do not use `best_only`, because it can skip non-best validation checkpoints.
- For Qwen primitive training, the training-time closed-loop monitor must use
  `--closed-loop-runner qwen_chain` so the `pick_up_cube` scenario is evaluated
  through the Qwen `move -> align -> pick_up` chain, not the legacy single-skill
  picklift smoke evaluator.
- Training-time closed-loop tests must always run exactly 10 episodes by
  default. Keep `--closed-loop-episodes 10` for launcher and monitor defaults;
  use a different count only when the user explicitly requests a labeled
  one-off run.
- Loop tests must record analyzer artifacts by default. Keep
  `--record-loop-artifacts` enabled for Qwen-chain validation so every loop test
  preserves raw Qwen payloads, rollout config, action-chunk metadata, and the
  seed/action/state trace needed to regenerate visual media locally. Do not
  render PNG/MP4 media during training-time validation by default; use
  `scripts/build_loop_test_analyzer_export.py --generate-media` when visual
  frames/videos are needed for inspection. Use `--no-record-loop-artifacts` only
  for explicitly labeled lightweight smoke or debugging runs, not for validation
  evidence.
- The training-time Qwen chain should default to the saved mock response
  `configs/agent/qwen3_so101_tool_planner_mock_response.json` so the epoch
  validation/closed-loop cadence is not blocked when LM Studio or Qwen is not
  running. Use live Qwen for the final/current-best validation command.
- Checkpoint/run retention is strict. During training, keep only:
  `checkpoints/best_closed_loop`, `checkpoints/best_val_loss`, and
  `checkpoints/best_train_loss`. Periodic numeric checkpoint directories are
  temporary save candidates and must be pruned after each checkpoint event.
- After a training run finishes, if its closed-loop test success rate is exactly
  `0.0`, delete that run's local artifact directory instead of archiving it.
  Runs without any closed-loop result are not treated as success-zero; label
  them as missing evidence before deciding whether to delete them.

## Required Launcher Behavior

Every local training launch should go through `scripts/start_so101_training.py`
or a plan script that emits a command for that launcher. The launcher records
this standard in its dry-run/start/status payload as `local_training_standard`
so future training work sees the rule before acting.
