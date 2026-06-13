from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_risk1bc_libero10_comparison_table import aggregate, main


class Risk1BcLibero10ComparisonTableTest(unittest.TestCase):
    def test_aggregate_separates_candidate_selector_and_smoke_success(self) -> None:
        rows = [
            {
                "suite": "libero_10",
                "task_id": 0,
                "seed": 1201,
                "risk1b_provenance": "external_vlm_json_policy_generated",
                "risk1b_diversity_metrics": {
                    "mean_normalized_pairwise_l2": 0.1,
                    "mean_pairwise_cosine_distance": 0.01,
                    "min_pairwise_l2": 0.2,
                },
                "score_spread": 0.03,
                "score_source": "observation_object_target_distance_proxy",
                "selected_candidate_id": "candidate_01",
                "pc_success": 0.0,
            },
            {
                "suite": "libero_10",
                "task_id": 1,
                "seed": 1201,
                "risk1b_provenance": "external_vlm_json_policy_generated",
                "risk1b_diversity_metrics": {
                    "mean_normalized_pairwise_l2": 0.2,
                    "mean_pairwise_cosine_distance": 0.03,
                    "min_pairwise_l2": 0.4,
                },
                "score_spread": 0.07,
                "score_source": "observation_object_target_distance_proxy",
                "selected_candidate_id": "candidate_00_policy_only",
                "pc_success": 100.0,
            },
        ]

        summary = aggregate(rows)

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["suites"], ["libero_10"])
        self.assertEqual(summary["task_ids"], [0, 1])
        self.assertEqual(summary["risk1b_policy_generated_rows"], 2)
        self.assertEqual(summary["risk1b_mean_normalized_pairwise_l2"], 0.15)
        self.assertEqual(summary["risk1c_mean_score_spread"], 0.05)
        self.assertEqual(summary["risk1c_non_baseline_selection_rate"], 0.5)
        self.assertEqual(summary["smoke_pc_success_mean"], 50.0)

    def test_main_writes_markdown_and_json_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run"
            root.mkdir()
            (root / "results.jsonl").write_text(
                json.dumps(
                    {
                        "suite": "libero_10",
                        "task_id": 0,
                        "seed": 1201,
                        "risk1b_provenance": "external_vlm_json_policy_generated",
                        "risk1b_diversity_metrics": {"mean_normalized_pairwise_l2": 0.123},
                        "score_spread": 0.04,
                        "score_source": "observation_object_target_distance_proxy",
                        "selected_candidate_id": "candidate_01",
                        "pc_success": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            code = main(["--risk1bc-root", str(root), "--json"])

            self.assertEqual(code, 0)
            self.assertTrue((root / "risk1bc_libero10_comparison_table.json").exists())
            markdown = (root / "risk1bc_libero10_comparison_table.md").read_text(encoding="utf-8")
            payload = json.loads((root / "risk1bc_libero10_comparison_table.json").read_text(encoding="utf-8"))
            self.assertIn("Our SmolVLA baseline", markdown)
            self.assertIn("Reference paper number", markdown)
            self.assertIn("ActionX Table 1, SmolVLA", markdown)
            self.assertIn("frontiersin.org", markdown)
            self.assertIn("Risk1-B/C alternative-goal experiment", markdown)
            self.assertIn("not a fair full benchmark success comparison", markdown)
            self.assertEqual(payload["baseline_libero10_pc_success"], 75.0)
            self.assertEqual(payload["reference_libero10_pc_success"], 77.0)
            self.assertEqual(payload["reference_label"], "ActionX Table 1, SmolVLA")


if __name__ == "__main__":
    unittest.main()
