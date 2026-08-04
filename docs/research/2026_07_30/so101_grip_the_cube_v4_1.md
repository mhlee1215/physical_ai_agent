# SO101 Grip the Cube v4.1

## Purpose

`grip_the_cube_v4_1` replaces image-bin rejection sampling with deterministic
sampling over a physically verified, base-relative grasp workspace. The
dataset is append-only and does not modify earlier SO101 datasets.

## Generation Contract

- Episodes: 500 successful teacher trajectories
- Start pose: hardware-aligned home pose
- Object: 30 mm green cube
- Prompt: `grip the green cube and lift`
- Cameras: camera1 egocentric and camera2 wrist, both 256 x 256
- Workspace sector: radius 25 to 28 cm, azimuth -20 to +80 degrees
- Spatial strata: 4 radial bins by 11 angular bins
- Sampling: deterministic within-cell variation with mild distance decay
- Teacher sequence: approach, alignment, close, lift, terminal hold
- Minimum gripper-to-floor clearance: 1 cm
- Render replay: exact simulator state, geom transforms, and camera transforms

Recipe:

```text
configs/so101/dataset_generation/grip_the_cube_v4_1.json
```

Dataset:

```text
_workspace/so101_lerobot/grip_the_cube_v4_1
```

## Verified Result

- Episodes: 500
- Frames: 91,771
- Teacher success: 500 / 500
- Unique seeds: 500 / 500
- Unique workspace positions: 500 / 500
- Declared workspace cells covered: 44 / 44
- Polar cells covered: 44 / 44
- Polar cell count coefficient of variation: 0.2943
- Radius span: 3.0 cm
- Angular span: 100 degrees
- Median nearest-neighbor spacing: 2.48 mm
- Camera1 invisible episodes: 0
- Minimum floor clearance: 10.002 mm
- Minimum lift height: 64.361 mm
- Render replay maximum state error: 0

Radial counts intentionally decrease mildly with distance:

| Base radius | Episodes |
| --- | ---: |
| 25 cm | 132 |
| 26 cm | 126 |
| 27 cm | 121 |
| 28 cm | 121 |

This distribution is uniform over the verified base-relative workspace cells,
subject to the requested distance decay. It is not intended to be uniform over
camera1's 4 x 4 image grid; perspective projection and robot occlusion make
those two distributions different.

## Completion Gates

The following gates passed:

- dataset schema and training-ready registry validation
- exact episode count
- action/state 6D contract
- camera1/camera2 256 x 256 contract
- prompt and terminal-hold audit
- successful grasp/lift outcomes
- unique seeds and positions
- workspace and polar coverage
- radial distribution match and distance decay
- floor-clearance gate
- render-replay state and final-outcome validation
- JSON to Markdown to HTML distribution report integrity
- live dataset viewer catalog and frame load

Canonical distribution artifacts:

```text
_workspace/so101_lerobot/grip_the_cube_v4_1/meta/distribution/
  distribution.json
  distribution.md
  distribution.html
```

The recorded Markdown SHA-256 is
`bf091268b3acd91e9940f56521b0ae2f27c9da7373cd6338efa4b05fee32b56b`.
