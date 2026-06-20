# SO-101 Interactive Sim-to-Real Contract

This contract defines the lightweight interactive SO101-Nexus simulation lane
used for Codex-driven control development before any real SO-101 actuation.

## Purpose

- Let Codex step a SO101-Nexus MuJoCo environment through a small command loop.
- Preserve every observation, action candidate, safety decision, and blocker as
  repo-local artifacts.
- Keep the action surface close to the real SO-100/SO-101 six-joint convention
  without pretending that simulation actions are real motor commands.

## Command Surface

The interactive simulator accepts:

```text
observe
sample [fraction]
center
action [a0,a1,a2,a3,a4,a5]
nudge <joint_name> <value>
chunk [[a0,...,a5],[a0,...,a5]]
reset [seed]
quit
```

Joint order is fixed:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

## Safety Boundary

- This path never writes to real hardware.
- `real_robot_safe_to_execute` is always `false`.
- `send_action_called` is always `false`.
- Actions are checked only against the SO101-Nexus action space.
- Simulation actions are not Dynamixel raw ticks, calibrated raw positions, or
  verified SmolVLA postprocessed real-robot commands.

Before using any simulation candidate on the real SO-101, a separate real
adapter must provide:

- live readback;
- calibration and joint-limit manifest;
- camera role contract;
- bounded execution packet;
- user confirmation;
- observer evidence;
- home-return and torque-off report.

## Example

```bash
sh scripts/so101_interactive_sim.sh \
  --output-dir _workspace/so101_interactive/demo \
  --command observe \
  --command 'nudge shoulder_pan 0.1' \
  --command 'action [0,0,0,0,0,0]'
```

Artifacts are written under the selected output directory:

- `session_manifest.json`
- `latest_observation.json`
- `events.jsonl`

Use this lane to develop the Codex/agent decision loop and sim-side verifier
before adding a real SO-101 hardware adapter.
