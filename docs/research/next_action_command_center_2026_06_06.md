# Next Action Command Center

Date: 2026-06-06

## Current blocking condition

Mac-local actual RGB probe is blocked by SAPIEN/Vulkan:

```text
vk::createInstanceUnique: ErrorIncompatibleDriver
```

The next real progress requires a renderer-capable Linux/GPU environment.

## One command for renderer-capable environment

```bash
ORACLE_OVERLAY_ROOT=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z \
ROOT_OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage \
EPISODES=1 \
STEPS=12 \
MIN_STRICT_STEPS=10 \
PYTHON_BIN=python3 \
sh scripts/run_remote_true_oracle_evidence_pack.sh
```

## Expected pass evidence

Tier O is not complete unless the generated manifests show:

```text
strict_true_oracle_step_count >= 10
true_oracle_projection == true
status == passed
```

## Command order inside the evidence pack

1. `scripts/run_renderer_env_preflight.sh`
2. `scripts/build_renderer_env_preflight_report.py`
3. `scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh`
4. `scripts/build_actual_sim_true_oracle_two_stage_result_report.py`
5. `scripts/rebuild_oracle_overlay_milestones.sh`

## Important report links

- Main dashboard: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_dashboard.html`
- Artifact audit: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/artifact_completeness_audit/artifact_completeness_audit.html`
- Paper claim gate: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/paper_claim_gate/paper_claim_gate_report.html`
- Experiment matrix: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/agentic_smolvla_experiment_matrix_result/agentic_smolvla_experiment_matrix_result.html`
- Remote evidence pack: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_report.html`

## Do not do automatically

- Do not create a new pod.
- Do not stop the baseline/parity pod.
- Do not delete pods.
- Do not claim Tier O before strict manifest evidence exists.
- Do not claim agentic success improvement before C0-C5 environment success flags exist.

