# Paper Progress Ledger

Date: 2026-06-06

## Current paper direction

The project should be framed as:

> Agentic lightweight VLA control with verifier/retry and spatial-cue ablations.

It should not be framed as:

> A new visual overlay or affordance point prompting method.

Reason: close prior work already covers visual prompting, traces, visual
primitives, affordance points, and affordance-aware VLA representations.

## Current evidence state

### Supported now

- The repository has a SmolVLA/ManiSkill CP24 path.
- Actual simulator RGB frames exist from prior CP24 rollouts.
- Actual simulator RGB can be used for fallback and visual-heuristic overlay
  reports.
- The overlay/tensor codepath can produce SmolVLA-style image-input previews.
- The augmentation/projection codepath works when actual RGB is paired with
  synthetic pose/camera metadata.
- A cheap zero-action probe path exists to isolate renderer/env metadata
  capture from SmolVLA model loading.
- Mac-local probe failure is documented as a renderer/Vulkan blocker, not a
  SmolVLA blocker.
- A two-stage remote handoff exists: probe first, then SmolVLA policy only if
  probe reaches 10 strict samples.

### Not supported yet

- Tier O actual-simulation true-oracle projection is not achieved.
- No current artifact proves same-step actual RGB + real env object pose + real
  camera metadata + projected overlay for 10 samples.
- No current artifact proves SmolVLA benchmark improvement from overlays.
- No current artifact proves agentic retry improves final environment success.

## Claim ledger

### Claim 1: Lightweight VLA is a valid base policy target

Status: supported as a paper motivation.

Evidence:

- SmolVLA is explicitly lightweight and deployment-oriented.
- Repo has SmolVLA adapter and CP24 planning path.

Next evidence:

- Baseline success table under fixed task IDs and seeds.

### Claim 2: Visual overlay is not the main novelty

Status: supported.

Evidence:

- Related-work scan found VP-VLA, TraceVLA, AVP, RoboPoint, AffordVLA,
  AffordanceVLA, CoA-VLA, CLIPort, and VoxPoser.

Next evidence:

- Intro and Related Work must cite these papers prominently.

### Claim 3: Actual-sim RGB evidence exists

Status: supported.

Evidence:

- CP24 saved actual simulator RGB frames.
- Actual-sim fallback, heuristic, temporal, policy-input, and readiness-gap
  reports show 10+ actual RGB samples.

Next evidence:

- Keep synthetic and actual-sim reports separated in every figure caption.

### Claim 4: Current true-oracle evidence is blocked

Status: supported.

Evidence:

- Readiness gap report shows actual RGB exists but same-step pose/camera
  metadata is missing.
- Mac-local zero-action probe fails at renderer initialization with
  `vk::createInstanceUnique: ErrorIncompatibleDriver`.

Next evidence:

- Renderer-capable Linux/GPU probe pass with
  `affordance_oracle_probe_true_oracle_steps.json`.

### Claim 5: Projection codepath works if metadata is available

Status: supported as diagnostic only.

Evidence:

- Actual RGB + synthetic metadata codepath diagnostic produces 12/12
  `projected_object_pose` samples.

Claim boundary:

- This is not Tier O because pose/camera metadata are synthetic.

### Claim 6: Cheap probe separates renderer metadata from SmolVLA

Status: implemented and Mac-local blocker documented.

Evidence:

- `scripts/run_actual_sim_true_oracle_probe_cp24.sh`
- Probe blocker report shows SmolVLA ready but renderer blocked.

Next evidence:

- Run the same probe in renderer-capable environment.

### Claim 7: Two-stage remote path is ready

Status: implemented.

Evidence:

- `scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh`
- Remote handoff report and two-stage result importer exist.

Next evidence:

- Execute in approved renderer-capable environment and import summary.

### Claim 8: Agentic wrapper improves success

Status: not yet supported.

Evidence:

- None yet.

Next evidence:

- Run controlled matrix: policy-only, overlay-only, agentic-only,
  agentic+overlay, oracle upper bound.

## Immediate next checkpoint

Run the two-stage true-oracle gate in a renderer-capable environment:

```bash
ROOT_OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage \
EPISODES=1 \
STEPS=12 \
MIN_STRICT_STEPS=10 \
sh scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh
```

Then rebuild/import:

```bash
PYTHONPATH=src .venv/bin/python -B scripts/build_actual_sim_true_oracle_two_stage_result_report.py \
  --summary _workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage/two_stage_summary.json \
  --output-dir _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/actual_sim_true_oracle_two_stage_result
```

