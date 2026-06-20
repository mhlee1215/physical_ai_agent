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
- For the current Qwen primitive lane, the config is:
  `configs/so101/training_datasets/qwen_edge_primitives.json`.
- The primitive train splits are used together in one training run:
  `move_over_cube_edge/train`, `align_fixed_jaw_cube_edge/train`, and
  `grip_from_edge_cube/train`.
- The validation splits are also composed through `hf_merge_sources`.

## Evaluation

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
- Use `--closed-loop-policy periodic` or `--closed-loop-policy best_or_periodic`;
  do not use `best_only`, because it can skip non-best validation checkpoints.
- For Qwen primitive training, the training-time closed-loop monitor must use
  `--closed-loop-runner qwen_chain` so the `pick_up_cube` scenario is evaluated
  through the Qwen `move -> align -> pick_up` chain, not the legacy single-skill
  picklift smoke evaluator.
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

## Required Launcher Behavior

Every local training launch should go through `scripts/start_so101_training.py`
or a plan script that emits a command for that launcher. The launcher records
this standard in its dry-run/start/status payload as `local_training_standard`
so future training work sees the rule before acting.
