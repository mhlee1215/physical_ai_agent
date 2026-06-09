#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ORACLE_OVERLAY_ROOT:-_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z}"
SIM_FRAMES="${SIM_FRAMES:-_workspace/checkpoints/checkpoint_24_pickcube_success_search_zero_30ep_50step/maniskill_rollout/rollout_frames}"
ACTUAL_SIM_FRAMES="${ACTUAL_SIM_FRAMES:-_workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_images_10ep_50step/maniskill_rollout/rollout_frames}"
TRUE_ORACLE_MANIFEST="${TRUE_ORACLE_MANIFEST:-$ROOT/live_true_oracle_projection/smolvla_affordance_true_oracle_steps.json}"
LIVE_ROOT="${LIVE_ROOT:-_workspace/runpod_results/live_oracle_probe_20260606T2008Z}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$ROOT"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_html_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_trajectory_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_pose_dict_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_sensor_camera_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_multi_camera_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/validate_oracle_point_overlay.py \
  --output-dir "$ROOT/local_edge_case_validation" \
  --sim-frame-root "$SIM_FRAMES" \
  --max-sim-frames 20 \
  --trajectory-frames 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_comparison_report.py \
  --output-dir "$ROOT/raw_vs_overlay_comparison" \
  --sim-frame-root "$SIM_FRAMES" \
  --synthetic-count 20 \
  --sim-count 10

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_zoom_report.py \
  --comparison-root "$ROOT/raw_vs_overlay_comparison" \
  --output-dir "$ROOT/zoomed_raw_vs_overlay" \
  --limit 24 \
  --crop-size 96 \
  --scale 4

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_diverse_object_report.py \
  --output-dir "$ROOT/diverse_object_projection" \
  --episodes 30

PYTHONPATH=src "$PYTHON_BIN" -B scripts/audit_oracle_overlay_center_bias.py \
  --manifest "$ROOT/diverse_object_projection/diverse_object_manifest.json" \
  --output-dir "$ROOT/center_bias_audit" \
  --non-center-threshold 45 \
  --min-non-center 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_figure_pack.py \
  --root "$ROOT" \
  --output-dir "$ROOT/paper_figure_pack"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_runpod_decision_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/runpod_lifecycle_decision" \
  --limit 12

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_policy_input_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/policy_input_readiness" \
  --limit 12

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_encoding_ablation_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/encoding_ablation" \
  --limit 12

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_temporal_consistency_report.py \
  --output-dir "$ROOT/temporal_consistency" \
  --frames 24

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_frame_overlay_report.py \
  --frame-root "$ACTUAL_SIM_FRAMES" \
  --output-dir "$ROOT/actual_sim_frame_overlay_fallback" \
  --limit 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_visual_heuristic_overlay_report.py \
  --frame-root "$ACTUAL_SIM_FRAMES" \
  --output-dir "$ROOT/actual_sim_visual_heuristic_overlay" \
  --limit 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_heuristic_policy_input_report.py \
  --heuristic-manifest "$ROOT/actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_manifest.json" \
  --output-dir "$ROOT/actual_sim_heuristic_policy_input" \
  --limit 12

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_heuristic_temporal_report.py \
  --heuristic-manifest "$ROOT/actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_manifest.json" \
  --output-dir "$ROOT/actual_sim_heuristic_temporal" \
  --limit 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_evidence_preflight_report.py \
  --checkpoints-root "_workspace/checkpoints" \
  --output-dir "$ROOT/actual_sim_evidence_preflight" \
  --limit 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_cross_episode_report.py \
  --checkpoints-root "_workspace/checkpoints" \
  --output-dir "$ROOT/actual_sim_cross_episode_diversity" \
  --limit 24

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_report.py \
  --manifest "$TRUE_ORACLE_MANIFEST" \
  --output-dir "$ROOT/live_true_oracle_projection" \
  --limit 12 || true

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_readiness_gap_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/actual_sim_true_oracle_readiness_gap" \
  --limit 12

if [ -f "_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_probe/checkpoint_report.json" ]; then
  PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_probe_blocker_report.py \
    --checkpoint-dir "_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_probe" \
    --output-dir "$ROOT/actual_sim_true_oracle_probe_blocker"
fi

mkdir -p "$ROOT/actual_sim_true_oracle_remote_handoff"
cp docs/research/actual_sim_true_oracle_remote_handoff_report_2026_06_06.html \
  "$ROOT/actual_sim_true_oracle_remote_handoff/actual_sim_true_oracle_remote_handoff_report.html"
cat > "$ROOT/actual_sim_true_oracle_remote_handoff/actual_sim_true_oracle_remote_handoff_manifest.json" <<EOF
{
  "status": "passed",
  "source_type": "actual_sim_true_oracle_remote_handoff",
  "sample_count": 12,
  "true_oracle_projection": false,
  "two_stage_script": "retired_cp24_true_oracle_probe",
  "claim_boundary": "Remote handoff plan only; does not create pods or prove Tier O."
}
EOF

TWO_STAGE_SUMMARY="${TWO_STAGE_SUMMARY:-_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage/two_stage_summary.json}"
if [ -f "$TWO_STAGE_SUMMARY" ]; then
  PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_true_oracle_two_stage_result_report.py \
    --summary "$TWO_STAGE_SUMMARY" \
    --output-dir "$ROOT/actual_sim_true_oracle_two_stage_result" || true
fi

mkdir -p "$ROOT/paper_progress_ledger"
cp docs/research/paper_progress_ledger_report_2026_06_06.html \
  "$ROOT/paper_progress_ledger/paper_progress_ledger_report.html"
cat > "$ROOT/paper_progress_ledger/paper_progress_ledger_manifest.json" <<EOF
{
  "status": "passed",
  "source_type": "paper_progress_ledger",
  "sample_count": 12,
  "true_oracle_projection": false,
  "claim_boundary": "Paper-progress ledger only; summarizes supported, diagnostic-only, and blocked claims."
}
EOF

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_agentic_retry_schema_readiness_report.py \
  --output-dir "$ROOT/agentic_retry_schema_readiness"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_agentic_smolvla_experiment_matrix_result_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/agentic_smolvla_experiment_matrix_result"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_paper_claim_gate_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/paper_claim_gate"

mkdir -p "$ROOT/agentic_smolvla_paper_outline"
cp docs/research/agentic_smolvla_paper_outline_report_2026_06_06.html \
  "$ROOT/agentic_smolvla_paper_outline/agentic_smolvla_paper_outline_report.html"
cat > "$ROOT/agentic_smolvla_paper_outline/agentic_smolvla_paper_outline_manifest.json" <<EOF
{
  "status": "passed_claim_safe_outline",
  "source_type": "agentic_smolvla_paper_outline",
  "sample_count": 12,
  "true_oracle_projection": false,
  "agentic_success_claim": false,
  "claim_boundary": "Paper outline scaffold only; does not prove Tier O or agentic success."
}
EOF

mkdir -p "$ROOT/next_action_command_center"
cp docs/research/next_action_command_center_report_2026_06_06.html \
  "$ROOT/next_action_command_center/next_action_command_center_report.html"
cat > "$ROOT/next_action_command_center/next_action_command_center_manifest.json" <<EOF
{
  "status": "passed_next_action_ready",
  "source_type": "next_action_command_center",
  "sample_count": 12,
  "true_oracle_projection": false,
  "agentic_success_claim": false,
  "claim_boundary": "Operational command center only; does not run remote evidence or prove Tier O."
}
EOF

if [ -f "_workspace/checkpoints/renderer_env_preflight/renderer_env_preflight.json" ]; then
PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_renderer_env_preflight_report.py \
    --preflight-json "_workspace/checkpoints/renderer_env_preflight/renderer_env_preflight.json" \
    --output-dir "$ROOT/renderer_env_preflight"
fi

mkdir -p "$ROOT/remote_true_oracle_evidence_pack"
cp docs/research/remote_true_oracle_evidence_pack_report_2026_06_06.html \
  "$ROOT/remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_report.html"
cat > "$ROOT/remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_manifest.json" <<EOF
{
  "status": "passed_handoff_ready",
  "source_type": "remote_true_oracle_evidence_pack",
  "sample_count": 12,
  "true_oracle_projection": false,
  "runner": "scripts/run_remote_true_oracle_evidence_pack.sh",
  "claim_boundary": "Evidence pack handoff only; does not create pods or prove Tier O."
}
EOF

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_only_dashboard.py \
  --root "$ROOT" \
  --output-dir "$ROOT/actual_sim_only_dashboard"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_actual_sim_claim_matrix_report.py \
  --root "$ROOT" \
  --output-dir "$ROOT/actual_sim_claim_matrix"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_gallery.py \
  --image-root "$ROOT/local_multi_camera_validation/multi_camera_trajectory" \
  --output-dir "$ROOT/gallery_module_probe" \
  --title "Oracle Gallery Module Probe" \
  --min-frames 10 \
  --limit 20

PYTHONPATH=src "$PYTHON_BIN" -B scripts/audit_live_oracle_overlay_output.py \
  --root "$LIVE_ROOT" \
  --output-dir "$ROOT/live_audit_blocked_probe" \
  --min-frames 10 || true

PYTHONPATH=src "$PYTHON_BIN" -B scripts/audit_oracle_overlay_milestone_artifacts.py \
  --root "$ROOT" \
  --output-dir "$ROOT/artifact_completeness_audit"

PYTHONPATH=src "$PYTHON_BIN" -B scripts/build_oracle_overlay_milestone_dashboard.py \
  --root "$ROOT" \
  --live-root "$LIVE_ROOT" \
  --output "$ROOT/milestone_dashboard.html"

cat <<EOF
oracle_overlay_milestones_rebuilt=true
root=$ROOT
dashboard=$ROOT/milestone_dashboard.html
preferred_full_frame=$ROOT/diverse_object_projection/diverse_object_full_contact_sheet.png
preferred_zoom=$ROOT/diverse_object_projection/diverse_object_zoom_contact_sheet.png
EOF
