#!/bin/sh
set -eu

# Remote/renderer-capable evidence pack.
#
# Runs the full non-destructive evidence sequence:
#   1. Renderer environment preflight.
#   2. Two-stage actual-sim true-oracle gate.
#   3. Two-stage result report.
#   4. Milestone rebuild.
#
# This script does not create, stop, delete, or resize Pods.

ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-_workspace/checkpoints/renderer_env_preflight}"
TWO_STAGE_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage}"
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-12}"
MIN_STRICT_STEPS="${MIN_STRICT_STEPS:-10}"
SUMMARY_PATH="$TWO_STAGE_OUTPUT_DIR/two_stage_summary.json"
PACK_SUMMARY="$ROOT/remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_summary.json"

mkdir -p "$ROOT/remote_true_oracle_evidence_pack"

echo "stage=preflight"
set +e
OUTPUT_DIR="$PREFLIGHT_OUTPUT_DIR" PYTHON_BIN="$PYTHON_BIN" sh scripts/run_renderer_env_preflight.sh
preflight_status=$?
set -e

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_renderer_env_preflight_report.py \
  --preflight-json "$PREFLIGHT_OUTPUT_DIR/renderer_env_preflight.json" \
  --output-dir "$ROOT/renderer_env_preflight"

echo "stage=two_stage"
set +e
ROOT_OUTPUT_DIR="$TWO_STAGE_OUTPUT_DIR" \
ORACLE_OVERLAY_ROOT="$ROOT" \
EPISODES="$EPISODES" \
STEPS="$STEPS" \
MIN_STRICT_STEPS="$MIN_STRICT_STEPS" \
sh scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh
two_stage_status=$?
set -e

if [ -f "$SUMMARY_PATH" ]; then
  PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_two_stage_result_report.py \
    --summary "$SUMMARY_PATH" \
    --output-dir "$ROOT/actual_sim_true_oracle_two_stage_result" || true
fi

PYTHON_BIN="$PYTHON_BIN" ORACLE_OVERLAY_ROOT="$ROOT" sh scripts/rebuild_oracle_overlay_milestones.sh

cat > "$PACK_SUMMARY" <<EOF
{
  "status": "$(if [ "$preflight_status" -eq 0 ] && [ "$two_stage_status" -eq 0 ]; then echo passed; else echo blocked_or_failed; fi)",
  "source_type": "remote_true_oracle_evidence_pack",
  "preflight_status": $preflight_status,
  "two_stage_status": $two_stage_status,
  "preflight_json": "$PREFLIGHT_OUTPUT_DIR/renderer_env_preflight.json",
  "two_stage_summary": "$SUMMARY_PATH",
  "dashboard": "$ROOT/milestone_dashboard.html",
  "artifact_audit": "$ROOT/artifact_completeness_audit/artifact_completeness_audit.html",
  "claim_boundary": "Evidence pack runner only; does not create/stop/delete Pods."
}
EOF

cat "$PACK_SUMMARY"
exit "$two_stage_status"
