#!/bin/sh
set -eu

# Cheap CP24 actual-simulation true-oracle capture probe.
#
# This uses the zero-action affordance_oracle_probe policy, so it does not load
# SmolVLA weights and does not download model files. It only checks whether the
# renderer can produce actual RGB frames and whether the environment exposes
# object pose + camera metadata strongly enough for projected_object_pose.

OUTPUT_DIR="${OUTPUT_DIR:-_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_probe}"
ORACLE_OVERLAY_ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-12}"
CHECKPOINT_SCRIPT="${CHECKPOINT_SCRIPT:-scripts/checkpoint_24.sh}"
TRUE_ORACLE_MANIFEST="$OUTPUT_DIR/maniskill_rollout/affordance_oracle_probe_true_oracle_steps.json"

set +e
sh "$CHECKPOINT_SCRIPT" \
  --require-maniskill \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --policy affordance_oracle_probe \
  --real-images \
  --output-dir "$OUTPUT_DIR"
checkpoint_status=$?
set -e

if [ -f "$TRUE_ORACLE_MANIFEST" ]; then
  ORACLE_OVERLAY_ROOT="$ORACLE_OVERLAY_ROOT" \
    sh scripts/import_actual_sim_true_oracle_milestone.sh "$TRUE_ORACLE_MANIFEST"
  echo "actual_sim_true_oracle_probe_manifest=$TRUE_ORACLE_MANIFEST"
  exit "$checkpoint_status"
fi

cat <<EOF
actual_sim_true_oracle_probe_manifest_missing=true
checkpoint_status=$checkpoint_status
expected_manifest=$TRUE_ORACLE_MANIFEST
output_dir=$OUTPUT_DIR

No probe true-oracle manifest was produced. This isolates renderer/env metadata
capture from SmolVLA model loading. Inspect:
  $OUTPUT_DIR/checkpoint_report.json
  $OUTPUT_DIR/maniskill_blocker.md
EOF

exit "$checkpoint_status"
