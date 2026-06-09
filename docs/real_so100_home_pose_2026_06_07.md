# Real SO-100 Home Pose

Canonical home pose was defined by the user from the current physical robot
state on 2026-06-07 at 10:53:30 -0700.

Use motor raw state, not visual appearance or calibration midpoint, as the
home-pose definition.

## Target Raw Positions

| Joint | Raw target | Calibration fraction |
| --- | ---: | ---: |
| shoulder_pan | 2042 | 0.596630 |
| shoulder_lift | 2049 | 0.019400 |
| elbow_flex | 2046 | 0.998655 |
| wrist_flex | 2028 | 0.961486 |
| wrist_roll | 2052 | 0.501099 |
| gripper | 1789 | 0.028324 |

## Artifacts

- Home pose definition:
  `_workspace/real_so100/home_pose/canonical_home_pose_2026_06_07.json`
- Motor state snapshot:
  `_workspace/real_so100/home_pose/home_pose_motor_state_2026_06_07.json`
- Camera 0/1 reference observation:
  `_workspace/real_so100/home_pose/home_pose_observation_2026_06_07/manifest.json`

## Operational Rule

When the user asks to move the SO-100 to home pose, command the six joints to
the raw targets above, clipped to the active calibration file and interpolated
with bounded per-step raw deltas. After the movement, disable torque on all
motors and record `post_task_torque_disabled`.

Every real robot task must end with the same sequence:

1. Finish the task motion.
2. Return to this canonical home pose.
3. Disable torque on all motors.
4. Record the home-return artifact and `post_task_torque_disabled`.
