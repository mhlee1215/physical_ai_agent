# SO101 Training Run Configs

This directory holds user-facing SO101 training run configs.

Training configs may reference train/validation dataset roots, loop-test cases,
augmentation, checkpoint cadence, TensorBoard media, and closed-loop evaluation
settings. Dataset-only contracts, export recipes, checksum manifests, and raw
dataset registration stay under `configs/so101/training_datasets/`.

For monitored SO101 training, checkpoint creation, supervised validation, and
closed-loop evaluation must resolve to the same training step. In practice:

- `save_freq` is the checkpoint cadence;
- validation must use the same step cadence;
- `closed_loop_every_epochs * training.steps_per_epoch` must equal `save_freq`;
- total `steps` must be divisible by `save_freq`;
- TensorBoard logs `train/checkpoint_steps_remaining` and
  `important/checkpoint_steps_remaining` so the next checkpoint/validation/loop
  event is visible during training.

Launch a training config through the canonical launcher with a Hydra entrypoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_so101_training.py start \
  --hydra-config training/grip_the_cube_v1
```

Do not move a run config back into `training_datasets/`; keep runtime behavior
and dataset-generation metadata separate.

## Pydantic + Hydra Contract

Training configs are governed in code by the Pydantic models in:

```text
src/physical_ai_agent/so101_training_config_schema.py
```

Hydra entrypoints live under:

```text
configs/so101/hydra/training/
```

Each Hydra entrypoint points at one JSON run config under
`configs/so101/training/` and may define the default forwarded trainer args.
The JSON schema artifact at `configs/so101/schemas/training_config.schema.json`
is a compatibility/reference artifact; the Pydantic model is the source of
truth used by tests and the launcher.

Runtime defaults belong in the Hydra entrypoint's `launcher:` block, not in
launcher code. After a default Hydra entrypoint is approved, do not edit it
again unless the user explicitly asks for that default to change. For an
experiment, keep the approved default config intact and change only the
experiment-specific fields by one of these routes:

- create or edit a named Hydra entrypoint under `configs/so101/hydra/training/`;
- edit the referenced JSON run config under `configs/so101/training/` when the
  dataset, augmentation, validation, checkpoint, or closed-loop contract itself
  is changing;
- use CLI overrides only for one-off smoke/debug runs or for a user-requested
  temporary override, and report that override explicitly.

Do not add SO101 training defaults such as prompt, dataset, loop-test cases,
RMSE sweep, media resolution, augmentation, action contract, checkpoint cadence,
runner, device, or ports as hardcoded fallback values in Python. If a value is
missing, the launcher should fail before training rather than silently choosing
one.

Validate all configs before changing or launching a run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_so101_training_configs.py
```

The validator checks both `configs/so101/training/*.json` and Hydra
entrypoints. The launcher also validates the selected config before building the
training command. In particular, a config must define exactly one of
`train_dataset` or `train_datasets`, must keep `camera1=egocentric_cam` and
`camera2=wrist_cam`, and must keep TensorBoard, augmentation, validation, and
closed-loop settings in the documented shape.
