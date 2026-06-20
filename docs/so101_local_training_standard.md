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

## Required Launcher Behavior

Every local training launch should go through `scripts/start_so101_training.py`
or a plan script that emits a command for that launcher. The launcher records
this standard in its dry-run/start/status payload as `local_training_standard`
so future training work sees the rule before acting.
