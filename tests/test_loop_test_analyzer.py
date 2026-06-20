from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path("scripts").resolve()))

import build_loop_test_analyzer_export as exporter
import serve_loop_test_analyzer as server


class LoopTestAnalyzerTest(unittest.TestCase):
    def test_qwen_chain_loop_test_id_keeps_nact15_variant_distinct(self) -> None:
        self.assertEqual(
            exporter._loop_test_id_from_report_path(
                Path("closed_loop_evals/qwen_chain_seed98100_001792/qwen_closed_loop_eval_report.json"),
                "001792",
            ),
            "qwen_chain_001792",
        )
        self.assertEqual(
            exporter._loop_test_id_from_report_path(
                Path("closed_loop_evals/qwen_chain_seed98100_nact15_001792/qwen_closed_loop_eval_report.json"),
                "001792",
            ),
            "qwen_chain_nact15_001792",
        )

    def test_build_export_converts_qwen_chain_trace_to_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            metrics_dir = run_dir / "metrics"
            report_dir.mkdir(parents=True)
            metrics_dir.mkdir(parents=True)
            trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(_record(0, "move", "move_over_cube_edge")),
                        json.dumps(_record(1, "move", "move_over_cube_edge")),
                        json.dumps(_record(2, "align", "align_fixed_jaw_cube_edge")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (metrics_dir / "validation_metrics.jsonl").write_text(
                json.dumps({"checkpoint": "000224", "step": 224, "loss": 0.1}) + "\n",
                encoding="utf-8",
            )
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "success_rate": 0.0,
                        "episodes_requested": 1,
                        "episodes_completed": 1,
                        "seed": 98100,
                        "policy_rollout_config": {
                            "chunk_size": 50,
                            "n_action_steps": 15,
                            "num_steps": 10,
                        },
                        "plan": {
                            "model": "qwen3-vl-8b-instruct-mlx",
                            "task": "pick and lift the green cube",
                            "thinking_mode": "non-thinking",
                            "calls": [{"fn": "move"}, {"fn": "align"}],
                        },
                        "episodes": [
                            {
                                "episode": 0,
                                "final_success": False,
                                "total_reward": 3.0,
                                "steps": 3,
                                "trace_path": str(trace_path),
                                "final_info": {"success": False},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = exporter.build_export(run_dir, output_dir, copy_source=True)
            payload = server._loop_tests_payload(output_dir)
            detail = server._loop_test_detail(output_dir, "qwen_chain_000224")

        self.assertEqual(manifest["summary"]["loop_tests"], 1)
        self.assertEqual(payload["loop_tests"][0]["validation_loss"], 0.1)
        self.assertEqual(detail["loop_test"]["status_meaning"], "evaluator_completed")
        self.assertIn("system", detail["loop_test"]["qwen_prompts"])
        timeline = detail["episodes"][0]["timeline"]
        self.assertEqual(timeline[0]["type"], "planner_call")
        self.assertIn("system_prompt", timeline[0]["policy_input"])
        self.assertIn("tool_call_start", [row["type"] for row in timeline])
        self.assertIn("tool_call_end", [row["type"] for row in timeline])
        self.assertEqual([row["iteration"] for row in timeline if row["type"] == "tool_call_start"], [1, 2])
        first_tool = next(row for row in timeline if row["type"] == "tool_call_start")
        self.assertEqual(first_tool["tool_parameters"]["primitive_id"], "move_over_cube_edge")
        first_step = next(row for row in timeline if row["type"] == "policy_step")
        self.assertEqual(first_step["policy_output"]["action_chunk"]["generated_count"], 50)
        self.assertEqual(first_step["policy_output"]["action_chunk"]["used_per_chunk"], 15)
        self.assertTrue(first_step["policy_output"]["action_chunk"]["confirmed_in_rollout"])
        self.assertEqual(detail["episodes"][0]["iterations"][0]["action_chunk_summary"]["chunk_count"], 1)


def _record(step: int, fn: str, primitive_id: str) -> dict:
    return {
        "episode": 0,
        "global_step": step,
        "primitive_step": step,
        "fn": fn,
        "primitive_id": primitive_id,
        "prompt": f"{fn} prompt",
        "policy_path": "/policy",
        "observation": [float(step)],
        "action": [0.1, 0.2],
        "reward": 1.0,
        "info": {"tcp_to_target_dist": 0.1, "success": False},
        "terminated": False,
        "truncated": False,
        "image_feature_mapping": {"observation.images.camera1": "wrist_cam"},
    }


if __name__ == "__main__":
    unittest.main()
