#!/bin/sh
set -eu

# Two-stage actual-simulation true-oracle gate for renderer-capable environments.
#
# Stage 1 runs the cheap zero-action affordance_oracle_probe. This isolates RGB
# rendering plus env pose/camera metadata capture from SmolVLA model loading.
# Stage 2 runs smolvla_affordance_oracle only if Stage 1 produces >=10 strict
# true-oracle samples.
#
# This script does not create, stop, or modify RunPod Pods.

ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage}"
ORACLE_OVERLAY_ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-12}"
MIN_STRICT_STEPS="${MIN_STRICT_STEPS:-10}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-$ROOT_OUTPUT_DIR/probe}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-$ROOT_OUTPUT_DIR/policy}"

PROBE_MANIFEST="$PROBE_OUTPUT_DIR/maniskill_rollout/affordance_oracle_probe_true_oracle_steps.json"
POLICY_MANIFEST="$POLICY_OUTPUT_DIR/maniskill_rollout/smolvla_affordance_true_oracle_steps.json"
SUMMARY_PATH="$ROOT_OUTPUT_DIR/two_stage_summary.json"

mkdir -p "$ROOT_OUTPUT_DIR"

echo "stage=probe output_dir=$PROBE_OUTPUT_DIR"
set +e
OUTPUT_DIR="$PROBE_OUTPUT_DIR" \
ORACLE_OVERLAY_ROOT="$ORACLE_OVERLAY_ROOT" \
EPISODES="$EPISODES" \
STEPS="$STEPS" \
sh scripts/run_actual_sim_true_oracle_probe_cp24.sh
probe_status=$?
set -e

probe_strict_count=0
probe_manifest_status="missing"
if [ -f "$PROBE_MANIFEST" ]; then
  probe_strict_count="$(
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("strict_true_oracle_step_count", 0) or 0))' "$PROBE_MANIFEST"
  )"
  probe_manifest_status="$(
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status", "missing"))' "$PROBE_MANIFEST"
  )"
fi

if [ "$probe_status" -ne 0 ] || [ "$probe_strict_count" -lt "$MIN_STRICT_STEPS" ]; then
  cat > "$SUMMARY_PATH" <<EOF
{
  "status": "blocked_at_probe",
  "probe_status": $probe_status,
  "probe_manifest": "$PROBE_MANIFEST",
  "probe_manifest_status": "$probe_manifest_status",
  "probe_strict_true_oracle_step_count": $probe_strict_count,
  "min_strict_steps": $MIN_STRICT_STEPS,
  "policy_stage_ran": false,
  "policy_manifest": "$POLICY_MANIFEST"
}
EOF
  cat "$SUMMARY_PATH"
  exit "$probe_status"
fi

echo "stage=policy output_dir=$POLICY_OUTPUT_DIR"
set +e
OUTPUT_DIR="$POLICY_OUTPUT_DIR" \
ORACLE_OVERLAY_ROOT="$ORACLE_OVERLAY_ROOT" \
EPISODES="$EPISODES" \
STEPS="$STEPS" \
sh scripts/run_actual_sim_true_oracle_cp24.sh
policy_status=$?
set -e

policy_strict_count=0
policy_manifest_status="missing"
if [ -f "$POLICY_MANIFEST" ]; then
  policy_strict_count="$(
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("strict_true_oracle_step_count", 0) or 0))' "$POLICY_MANIFEST"
  )"
  policy_manifest_status="$(
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status", "missing"))' "$POLICY_MANIFEST"
  )"
fi

if [ "$policy_status" -eq 0 ] && [ "$policy_strict_count" -ge "$MIN_STRICT_STEPS" ]; then
  final_status="passed"
else
  final_status="blocked_at_policy"
fi

cat > "$SUMMARY_PATH" <<EOF
{
  "status": "$final_status",
  "probe_status": $probe_status,
  "probe_manifest": "$PROBE_MANIFEST",
  "probe_manifest_status": "$probe_manifest_status",
  "probe_strict_true_oracle_step_count": $probe_strict_count,
  "policy_status": $policy_status,
  "policy_manifest": "$POLICY_MANIFEST",
  "policy_manifest_status": "$policy_manifest_status",
  "policy_strict_true_oracle_step_count": $policy_strict_count,
  "min_strict_steps": $MIN_STRICT_STEPS,
  "policy_stage_ran": true
}
EOF

cat "$SUMMARY_PATH"
exit "$policy_status"
