from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_imagine_then_act_risk1bc_difficulty_eval import (
    build_config,
    build_parser,
    build_row_plan,
    collect_row_result,
    load_manifest,
    selected_rows,
)


class Risk1BcDifficultyEvalTest(unittest.TestCase):
    def test_manifest_filters_hard_rows_and_preserves_baseline_evidence(self) -> None:
        manifest = load_manifest(Path("configs/eval/risk1bc_baseline_difficulty_manifest.json"))
        rows = selected_rows(manifest, ("baseline_fail_hard",))

        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(row["baseline_category"] == "baseline_fail_hard" for row in rows))
        self.assertTrue(all(row["baseline_success"] is False for row in rows))
        self.assertIn("paths", rows[0]["evidence"])

    def test_row_plan_uses_argv_lists_and_records_baseline_category(self) -> None:
        manifest = load_manifest(Path("configs/eval/risk1bc_baseline_difficulty_manifest.json"))
        args = build_parser().parse_args(
            [
                "--manifest",
                "configs/eval/risk1bc_baseline_difficulty_manifest.json",
                "--categories",
                "baseline_fail_hard",
                "--output-dir",
                "/tmp/risk1bc_difficulty_test",
            ]
        )
        config = build_config(args, manifest)
        row = selected_rows(manifest, ("baseline_fail_hard",))[0]
        plan = build_row_plan(config, row)

        self.assertFalse(plan["used_shell"])
        self.assertEqual(plan["baseline_category"], "baseline_fail_hard")
        self.assertEqual(plan["commands"]["context_capture"][2], "scripts/capture_risk1b_context.py")
        self.assertIn("--risk1b-vlm-subgoals", plan["commands"]["risk1bc_probe"])
        self.assertIn("--risk1c-sim-selector", plan["commands"]["risk1bc_probe"])
        self.assertIn(str(row["task_id"]), plan["commands"]["risk1bc_probe"])
        self.assertIn(str(row["seed"]), plan["commands"]["risk1bc_probe"])
        subgoals_path = plan["expected_artifacts"]["risk1b_subgoals_json"]
        self.assertIn("risk1b_subgoals_qwen2_5_vl_7b_instruct_libero_goal_task6_seed", subgoals_path)
        self.assertIn(subgoals_path, plan["commands"]["risk1bc_probe"])

    def test_collect_row_result_keeps_env_success_separate_from_risk_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            probe = base / "risk1bc_probe"
            probe.mkdir(parents=True)
            (probe / "summary.json").write_text(
                json.dumps(
                    {
                        "risk1b": {"diversity_metrics": {"mean_normalized_pairwise_l2": 0.12}},
                        "candidate_provenance": "policy_generated",
                    }
                ),
                encoding="utf-8",
            )
            (probe / "risk1c_sim_selector.json").write_text(
                json.dumps(
                    {
                        "c1": {
                            "mode": "c1",
                            "selected_candidate_id": "candidate_03",
                            "selected_vs_policy_l2": 0.4,
                            "score_source": "observation_object_target_distance_proxy",
                            "score_spread": 0.7,
                            "per_candidate_details": [{"candidate_id": "candidate_03", "score": 0.9}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            eval_dir = probe / "eval_logs"
            eval_dir.mkdir()
            (eval_dir / "eval_info.json").write_text(json.dumps({"overall": {"pc_success": 100.0}}), encoding="utf-8")
            plan = {
                "row_id": "row",
                "suite": "libero_goal",
                "task_id": 6,
                "seed": 1200,
                "baseline_category": "baseline_fail_hard",
                "baseline_evidence": {"provenance": "test"},
                "baseline_success": False,
                "baseline_pc_success": 0.0,
                "output_dir": str(base),
                "expected_artifacts": {
                    "risk1bc_summary": str(probe / "summary.json"),
                    "risk1c_selector": str(probe / "risk1c_sim_selector.json"),
                },
            }

            record = collect_row_result(plan)

        self.assertEqual(record["baseline_category"], "baseline_fail_hard")
        self.assertEqual(record["risk1b_provenance"], "policy_generated")
        self.assertEqual(record["risk1b_diversity_metrics"]["mean_normalized_pairwise_l2"], 0.12)
        self.assertEqual(record["selected_candidate_id"], "candidate_03")
        self.assertEqual(record["score_source"], "observation_object_target_distance_proxy")
        self.assertEqual(record["pc_success"], 100.0)


if __name__ == "__main__":
    unittest.main()
