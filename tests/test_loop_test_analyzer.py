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

    def test_media_generation_command_uses_export_parent_as_run_dir(self) -> None:
        repo_root = Path("/repo")
        export_dir = Path("/run/qwen_edge_primitives/loop_test_analyzer_export")
        command = server._media_generation_command(repo_root, export_dir, python_executable="/python")

        self.assertEqual(command[0], "/python")
        self.assertIn(str(repo_root / "scripts" / "build_loop_test_analyzer_export.py"), command)
        self.assertIn("--copy-source", command)
        self.assertIn("--generate-media", command)
        self.assertEqual(command[command.index("--run-dir") + 1], str(export_dir.parent))
        self.assertEqual(command[command.index("--output-dir") + 1], str(export_dir))

    def test_media_generation_command_targets_selected_loop(self) -> None:
        repo_root = Path("/repo")
        export_dir = Path("/run/qwen_edge_primitives/loop_test_analyzer_export")
        command = server._media_generation_command(
            repo_root,
            export_dir,
            python_executable="/python",
            loop_test_id="qwen_chain_nact15_003136",
        )

        self.assertIn("--loop-test-id", command)
        self.assertEqual(command[command.index("--loop-test-id") + 1], "qwen_chain_nact15_003136")

    def test_filtered_export_preserves_existing_loop_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            _write_loop_report(run_dir, "qwen_chain_seed98100_000224", [_record(0, "move", "move_over_cube_edge")])
            _write_loop_report(run_dir, "qwen_chain_seed98100_000448", [_record(0, "align", "align_fixed_jaw_cube_edge")])

            exporter.build_export(run_dir, output_dir)
            manifest = exporter.build_export(run_dir, output_dir, loop_test_ids=["qwen_chain_000224"])

        self.assertEqual(manifest["summary"]["loop_tests"], 2)
        self.assertEqual(
            [row["loop_test_id"] for row in manifest["loop_tests"]],
            ["qwen_chain_000224", "qwen_chain_000448"],
        )

    def test_export_refuses_to_relabel_mismatched_camera_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            report_dir.mkdir(parents=True)
            trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
            bad_record = _record(0, "move", "move_over_cube_edge")
            bad_record["image_feature_mapping"] = {"observation.images.camera1": "wrist_cam"}
            trace_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "success_rate": 0.0,
                        "episodes_requested": 1,
                        "episodes_completed": 1,
                        "seed": 98100,
                        "camera_contract": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "plan": {"model": "qwen3-vl-8b-instruct-mlx", "task": "pick and lift the green cube", "calls": []},
                        "episodes": [{"episode": 0, "steps": 1, "trace_path": str(trace_path)}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "camera mapping mismatch"):
                exporter.build_export(run_dir, output_dir)

    def test_replay_env_config_falls_back_to_green_cube_plan(self) -> None:
        report = {
            "plan": {
                "task": "pick and lift the green cube",
                "calls": [
                    {
                        "fn": "move",
                        "object": "green cube",
                        "prompt": "Move the gripper above one visible green cube edge.",
                    }
                ],
            }
        }

        env_config = exporter._env_config_for_replay(report, {})

        self.assertEqual(env_config["object_shape"], "cube")
        self.assertEqual(env_config["object_color"], "green")
        self.assertEqual(env_config["n_distractors"], 0)

    def test_replay_env_config_prefers_recorded_env_config(self) -> None:
        report = {"env_config": {"object_shape": "cube", "object_color": "green", "n_distractors": 0}}
        replay = {"env_config": {"object_shape": "cube", "object_color": "red", "n_distractors": 0}}

        env_config = exporter._env_config_for_replay(report, replay)

        self.assertEqual(env_config["object_color"], "green")

    def test_media_generation_repo_root_prefers_export_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "physical_ai_agent"
            export_dir = workspace / "_workspace" / "run" / "loop_test_analyzer_export"
            (workspace / "scripts").mkdir(parents=True)
            (workspace / "src").mkdir()
            (workspace / "scripts" / "build_loop_test_analyzer_export.py").write_text("", encoding="utf-8")

            self.assertEqual(server._media_generation_repo_root(Path("/server"), export_dir), workspace)

    def test_media_generation_python_prefers_export_workspace_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "physical_ai_agent"
            export_dir = workspace / "_workspace" / "run" / "loop_test_analyzer_export"
            python_path = workspace / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")

            self.assertEqual(
                server._media_generation_python(Path("/server"), export_dir, python_executable="/fallback"),
                str(python_path),
            )

    def test_media_job_status_round_trips_in_export_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "export"
            payload = {"status": "running", "progress": {"percent": 12.5}}

            server._save_media_job_status(export_dir, payload)

            self.assertEqual(server._load_media_job_status(export_dir), payload)

    def test_media_artifact_progress_counts_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            export_dir = run_dir / "loop_test_analyzer_export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            other_report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000448"
            episode_dir = export_dir / "loop_tests" / "qwen_chain_000224" / "episodes" / "episode_000" / "media"
            report_dir.mkdir(parents=True)
            other_report_dir.mkdir(parents=True)
            trace_path = report_dir / "trace.jsonl"
            trace_path.write_text(json.dumps(_record(0, "move", "move_over_cube_edge")) + "\n", encoding="utf-8")
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps({"episodes": [{"trace_path": str(trace_path)}]}),
                encoding="utf-8",
            )
            other_trace_path = other_report_dir / "trace.jsonl"
            other_trace_path.write_text(
                json.dumps(_record(0, "align", "align_fixed_jaw_cube_edge"))
                + "\n"
                + json.dumps(_record(1, "align", "align_fixed_jaw_cube_edge"))
                + "\n",
                encoding="utf-8",
            )
            (other_report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps({"episodes": [{"trace_path": str(other_trace_path)}]}),
                encoding="utf-8",
            )
            for folder, name in [
                ("policy_inputs", "step_0000_egocentric_cam.png"),
                ("policy_inputs", "step_0000_wrist_cam.png"),
                ("robot_frames", "step_0000_top_down.png"),
                ("videos", "iteration_01_move.gif"),
            ]:
                path = episode_dir / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            progress = server._media_artifact_progress(export_dir, loop_test_id="qwen_chain_000224")

            self.assertEqual(progress["source_rollout_records"], 1)
            self.assertEqual(progress["expected_png_files"], 3)
            self.assertEqual(progress["png_files"], 3)
            self.assertEqual(progress["gif_files"], 1)
            self.assertEqual(progress["percent"], 100.0)
            self.assertEqual(progress["loop_test_id"], "qwen_chain_000224")
            self.assertEqual(server._media_artifact_progress(export_dir)["source_rollout_records"], 3)


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
        "image_feature_mapping": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam",
        },
    }


def _write_loop_report(run_dir: Path, report_name: str, records: list[dict]) -> None:
    report_dir = run_dir / "closed_loop_evals" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    (report_dir / "qwen_closed_loop_eval_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "success_rate": 0.0,
                "episodes_requested": 1,
                "episodes_completed": 1,
                "seed": 98100,
                "plan": {"model": "qwen3-vl-8b-instruct-mlx", "task": "pick and lift the green cube", "calls": []},
                "episodes": [
                    {
                        "episode": 0,
                        "final_success": False,
                        "total_reward": 1.0,
                        "steps": len(records),
                        "trace_path": str(trace_path),
                        "final_info": {"success": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
