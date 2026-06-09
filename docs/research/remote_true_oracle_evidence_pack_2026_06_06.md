# Remote True-Oracle Evidence Pack

Date: 2026-06-06

## Purpose

This is the single command to run inside an already-approved renderer-capable
Linux/GPU environment.

It does not create, stop, delete, or resize pods.

## Command

```bash
ORACLE_OVERLAY_ROOT=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z \
ROOT_OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage \
EPISODES=1 \
STEPS=12 \
MIN_STRICT_STEPS=10 \
PYTHON_BIN=python3 \
sh scripts/run_remote_true_oracle_evidence_pack.sh
```

## Sequence

1. Run renderer preflight.
2. Render preflight HTML.
3. Run zero-action true-oracle probe.
4. Run SmolVLA true-oracle policy only if probe passes.
5. Render two-stage result HTML.
6. Rebuild milestone dashboard and artifact audit.
7. Write evidence-pack summary JSON.

## Outputs

```text
_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_summary.json
_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_dashboard.html
_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/artifact_completeness_audit/artifact_completeness_audit.html
```

## Claim boundary

The pack runner itself proves nothing. It only orchestrates evidence collection.
Paper-facing Tier O remains valid only if the generated manifests report:

```text
strict_true_oracle_step_count >= 10
true_oracle_projection == true
status == passed
```

