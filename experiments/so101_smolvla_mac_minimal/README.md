# SO-101 SmolVLA Mac Minimal

Compact, separate workspace for running a SmolVLA-style policy on a local Mac
with an SO-101 follower arm. This intentionally does not import or reuse the
current `physical_ai_agent` implementation.

Reference design:

- `Shaibk/so101-smolvla-thesis` for the real-robot evaluation pattern:
  LeRobot policy loading, preprocessor/postprocessor, `predict_action`, and
  bounded trial logging.
- This folder keeps only the small Mac-local path: setup, camera checks,
  policy-load dry run, and an explicit-gated robot rollout.

## Layout

```text
experiments/so101_smolvla_mac_minimal/
  config.example.toml
  requirements.txt
  scripts/setup_lerobot_mac.sh
  tools/so101_smolvla_mac.py
  runs/                         # generated, ignored
```

## Setup

From this folder:

```bash
sh scripts/setup_lerobot_mac.sh
source .venv/bin/activate
cp config.example.toml config.local.toml
```

Edit `config.local.toml`:

- `robot.port`: your SO-101 follower serial port.
- `camera.scene_index`: the external scene camera to feed to SmolVLA as `camera1`.
- `policy.path`: a local checkpoint path or a Hugging Face policy id.

The setup script pins LeRobot to `d9ec3a6`, matching the reference repo's
reproduction guide, then installs SmolVLA and Feetech extras.

## Dry Checks

```bash
python tools/so101_smolvla_mac.py doctor --config config.local.toml
python tools/so101_smolvla_mac.py snap-cameras --config config.local.toml
python tools/so101_smolvla_mac.py dry-policy --config config.local.toml
```

`dry-policy` loads the policy config and tries to instantiate the policy. It
does not connect to the robot and does not move hardware.

The default camera routing mirrors the compact path from
`Shaibk/so101-smolvla-thesis`:

```text
scene camera -> observation.images.camera1
robot state  -> observation.state
```

## Real Robot Rollout

The rollout command refuses to move unless both gates are present:

```bash
python tools/so101_smolvla_mac.py run-policy \
  --config config.local.toml \
  --execute \
  --confirm "I understand this can move the SO-101"
```

Default runtime is intentionally short and slow. Change `runtime.max_seconds`,
`runtime.fps`, and `runtime.policy_num_steps` only after dry checks pass.

## What Is Not Included

This compact path does not include Shaibk's full thesis workflow:

- checkerboard camera warp;
- table/block-top calibration;
- deterministic green-block demonstrator;
- placement-plan benchmark;
- training replay.

Those are useful once the minimal policy path is stable, but they add too much
surface area for the first local bring-up.

## Safety Notes

- Keep the workspace clear.
- Know how to cut power before using `run-policy`.
- Prefer `dry-policy` until the checkpoint's action semantics are known.
- This script uses LeRobot's pre/postprocessor path; do not convert raw model
  tensors into motor commands manually.
