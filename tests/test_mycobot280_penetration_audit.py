from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_mycobot280_penetration_failures import audit_reports


class MyCobotPenetrationAuditTests(unittest.TestCase):
    def test_audits_crossing_duration_phase_and_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "trace.jsonl"
            depths = [0.0029, 0.0031, 0.0034, 0.0028, 0.0032]
            sides = ["left", "left", "left", "right", "right"]
            records = []
            for step, (depth, side) in enumerate(
                zip(depths, sides, strict=True)
            ):
                records.append(
                    {
                        "step": step,
                        "phase": "close" if step < 4 else "lift",
                        "ground_pickup": {
                            "pad_cube_contact_depth": {
                                "max_penetration_m": depth,
                                "checks": [
                                    {
                                        "penetration_m": depth,
                                        "side": side,
                                    }
                                ],
                            }
                        },
                    }
                )
            trace_path.write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            episode = {
                "episode": 0,
                "seed": 10,
                "success": False,
                "failed_gates": ["max_pad_cube_penetration_exceeded"],
                "max_pad_cube_penetration_m": 0.0034,
                "trace_path": str(trace_path),
            }
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps({"episode_summaries": [episode]}),
                encoding="utf-8",
            )

            audit = audit_reports([("test", report_path)])

        item = audit["rows"]["test"]["audited_episodes"][0]
        self.assertEqual(item["above_gate_steps"], 3)
        self.assertEqual(item["longest_above_gate_run_steps"], 2)
        self.assertEqual(item["peak_step"], 2)
        self.assertEqual(item["peak_phase"], "close")
        self.assertEqual(item["peak_sides"], ["left"])
        self.assertEqual(
            audit["scope"]["included_contact_pair"],
            "myCobot adaptive-gripper pad geoms against cube geom",
        )

    def test_rejects_report_trace_peak_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "step": 0,
                        "phase": "close",
                        "ground_pickup": {
                            "pad_cube_contact_depth": {
                                "max_penetration_m": 0.0032,
                                "checks": [
                                    {
                                        "penetration_m": 0.0032,
                                        "side": "right",
                                    }
                                ],
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "episode_summaries": [
                            {
                                "episode": 0,
                                "seed": 10,
                                "success": False,
                                "failed_gates": [
                                    "max_pad_cube_penetration_exceeded"
                                ],
                                "max_pad_cube_penetration_m": 0.0035,
                                "trace_path": str(trace_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "trace/report penetration mismatch"
            ):
                audit_reports([("test", report_path)])


if __name__ == "__main__":
    unittest.main()
