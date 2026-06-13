from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_risk1b_libero10_result_package import build_package, main


class Risk1BLibero10ResultPackageTest(unittest.TestCase):
    def test_package_marks_below_baseline_as_negative_table_ready_result(self) -> None:
        summary = {
            "status": "completed",
            "lane": "SECURE/shallow OSMesa data-production lane; non-EGL",
            "total_episodes": 100,
            "pc_success": 69.0,
            "fallback_rows": [],
            "invalid_qwen_rows": [],
            "rows": [
                {"task_id": 6, "success_count": 2, "n_episodes": 10, "pc_success": 20.0, "status": "completed"},
                {"task_id": 8, "success_count": 3, "n_episodes": 10, "pc_success": 30.0, "status": "completed"},
                {"task_id": 0, "success_count": 4, "n_episodes": 10, "pc_success": 40.0, "status": "completed"},
            ],
        }

        package = build_package(summary=summary)

        self.assertTrue(package["paper_table_ready"])
        self.assertEqual(package["verdict"], "NEGATIVE_OR_INCOMPLETE")
        self.assertEqual(package["risk1b"]["delta_vs_baseline_pp"], -6.0)
        self.assertEqual([row["task_id"] for row in package["weak_rows"]], [6, 8, 0])
        self.assertIn("did not exceed", package["allowed_claims"][1])
        self.assertIn("non-EGL", package["risk1b"]["lane"])

    def test_package_blocks_fallback_rows_from_paper_table_ready(self) -> None:
        summary = {
            "status": "partial_or_blocked",
            "lane": "SECURE/shallow OSMesa data-production lane; non-EGL",
            "total_episodes": 100,
            "pc_success": 80.0,
            "fallback_rows": [4],
            "invalid_qwen_rows": [],
            "rows": [],
        }

        package = build_package(summary=summary)

        self.assertFalse(package["paper_table_ready"])
        self.assertEqual(package["verdict"], "INCOMPLETE_OR_BLOCKED")
        self.assertIn("not full paper-table evidence", package["allowed_claims"][0])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "lane": "SECURE/shallow OSMesa data-production lane; non-EGL",
                        "total_episodes": 100,
                        "pc_success": 76.0,
                        "fallback_rows": [],
                        "invalid_qwen_rows": [],
                        "rows": [],
                    }
                ),
                encoding="utf-8",
            )

            code = main(["--summary", str(summary)])

            self.assertEqual(code, 0)
            self.assertTrue((root / "risk1b_libero10_result_package.json").exists())
            markdown = (root / "risk1b_libero10_result_package.md").read_text(encoding="utf-8")
            self.assertIn("Risk1-B LIBERO-10 Result Package", markdown)
            self.assertIn("+1.0pp", markdown)


if __name__ == "__main__":
    unittest.main()
