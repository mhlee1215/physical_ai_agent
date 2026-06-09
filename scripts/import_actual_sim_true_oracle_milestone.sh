#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
LIVE_ROOT="${LIVE_ROOT:-_workspace/runpod_results/live_oracle_probe_20260606T2008Z}"
TRUE_ORACLE_MANIFEST="${1:-${TRUE_ORACLE_MANIFEST:-}}"

if [ -z "$TRUE_ORACLE_MANIFEST" ]; then
  echo "usage: sh scripts/import_actual_sim_true_oracle_milestone.sh <smolvla_affordance_true_oracle_steps.json>" >&2
  echo "or set TRUE_ORACLE_MANIFEST=/path/to/smolvla_affordance_true_oracle_steps.json" >&2
  exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_report.py \
  --manifest "$TRUE_ORACLE_MANIFEST" \
  --output-dir "$ROOT/live_true_oracle_projection" \
  --limit 12 || true

PYTHONPATH=src "$PYTHON_BIN" -B scripts/audit_oracle_overlay_milestone_artifacts.py \
  --root "$ROOT" \
  --output-dir "$ROOT/artifact_completeness_audit"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_milestone_dashboard.py \
  --root "$ROOT" \
  --live-root "$LIVE_ROOT" \
  --output "$ROOT/milestone_dashboard.html"

cat <<EOF
actual_sim_true_oracle_imported=true
manifest=$TRUE_ORACLE_MANIFEST
report=$ROOT/live_true_oracle_projection/actual_sim_true_oracle_report.html
dashboard=$ROOT/milestone_dashboard.html
EOF
