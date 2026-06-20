#!/bin/sh
set -eu

# Run the CP24 actual-simulation true-oracle projection probe and import the
# resulting evidence into the oracle-overlay milestone dashboard.
#
# This script does not create or manage RunPod Pods. Use it inside an already
# approved local/remote environment that can render ManiSkill RGB observations.

OUTPUT_DIR="${OUTPUT_DIR:-_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle}"
ORACLE_OVERLAY_ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-12}"
POLICY="${POLICY:-smolvla_affordance_oracle}"
CHECKPOINT_SCRIPT="${CHECKPOINT_SCRIPT:-scripts/checkpoint_24.sh}"
TRUE_ORACLE_MANIFEST="$OUTPUT_DIR/maniskill_rollout/smolvla_affordance_true_oracle_steps.json"

if [ "$POLICY" != "smolvla_affordance_oracle" ]; then
  echo "POLICY must be smolvla_affordance_oracle for true-oracle evidence, got: $POLICY" >&2
  exit 2
fi

set +e
sh "$CHECKPOINT_SCRIPT" \
  --require-maniskill \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --policy "$POLICY" \
  --allow-download \
  --real-images \
  --output-dir "$OUTPUT_DIR"
checkpoint_status=$?
set -e

if [ -f "$TRUE_ORACLE_MANIFEST" ]; then
  ORACLE_OVERLAY_ROOT="$ORACLE_OVERLAY_ROOT" \
    sh scripts/import_actual_sim_true_oracle_milestone.sh "$TRUE_ORACLE_MANIFEST"
  echo "actual_sim_true_oracle_manifest=$TRUE_ORACLE_MANIFEST"
  exit "$checkpoint_status"
fi

cat <<EOF
actual_sim_true_oracle_manifest_missing=true
checkpoint_status=$checkpoint_status
expected_manifest=$TRUE_ORACLE_MANIFEST
output_dir=$OUTPUT_DIR

No true-oracle manifest was produced. This usually means ManiSkill/SAPIEN RGB
rendering is still blocked, SmolVLA loading failed before rollout, or the policy
did not reach action-input observations. Inspect:
  $OUTPUT_DIR/checkpoint_report.json
  $OUTPUT_DIR/maniskill_blocker.md
EOF

exit "$checkpoint_status"
