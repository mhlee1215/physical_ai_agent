from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path("scripts").resolve()))

import build_loop_test_analyzer_export as exporter
import serve_loop_test_analyzer as server
import serve_so101_dataset_viewer as dataset_viewer


VALID_1X1_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


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

    def test_dataset_viewer_exposes_closed_loop_prompt_and_start_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            config_dir = repo_root / "configs" / "so101" / "training_datasets"
            config_dir.mkdir(parents=True)
            (config_dir / "qwen_edge_primitives.json").write_text(
                json.dumps(
                    {
                        "closed_loop": {
                            "test_cases": [
                                {
                                    "id": "move_align_pick_up",
                                    "description": "Requires move, align, and pick up primitives to succeed.",
                                    "episodes": 5,
                                    "steps": 120,
                                    "seed": 98100,
                                    "start_contract": "full_chain_reset",
                                    "task_prompt": "pick and lift the green cube",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_move_align_pick_up_green_cube.json",
                                },
                                {
                                    "id": "align_pick_up",
                                    "description": "Starts close enough that align and pick up primitives should be sufficient.",
                                    "episodes": 5,
                                    "steps": 120,
                                    "seed": 98200,
                                    "start_contract": "align_pick_reset",
                                    "task_prompt": "align with and lift the green cube",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_align_pick_up_green_cube.json",
                                },
                                {
                                    "id": "pick_up_only",
                                    "description": "Starts from the pick-up primitive distribution and only requires pick up.",
                                    "episodes": 5,
                                    "steps": 120,
                                    "seed": 98300,
                                    "start_contract": "pick_up_reset",
                                    "task_prompt": "lift the green cube",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_pick_up_green_cube.json",
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_dir = repo_root / "_workspace" / "so101_training" / "runs" / "debug_run"
            loop_dir = run_dir / "closed_loop_evals" / "qwen_chain_pick_up_only_seed98300_000086"
            episode_dir = loop_dir
            media_dir = episode_dir / "media"
            policy_dir = media_dir / "policy_inputs"
            robot_dir = media_dir / "robot_frames"
            video_dir = media_dir / "videos"
            policy_dir.mkdir(parents=True)
            robot_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            tiny_png = b"\x89PNG\r\n\x1a\n"
            (policy_dir / "step_0000_egocentric_cam.png").write_bytes(tiny_png)
            (robot_dir / "step_0000_top_down.png").write_bytes(tiny_png)
            (video_dir / "iteration_01_grip_from_edge_cube.gif").write_bytes(b"GIF89a")
            timeline_path = episode_dir / "qwen_closed_loop_episode_000.jsonl"
            timeline_path.write_text(
                json.dumps(
                    {
                        "episode": 0,
                        "global_step": 0,
                        "primitive_step": 0,
                        "fn": "pick_up",
                        "primitive_id": "grip_from_edge_cube",
                        "prompt": "Close the gripper on the green cube edge and lift.",
                        "image_feature_mapping": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "action": [0.0],
                        "policy_rollout_config": {"n_action_steps": 15},
                        "info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0},
                        "media": {
                            "policy_input_images": {
                                "egocentric_cam": str(policy_dir / "step_0000_egocentric_cam.png"),
                            },
                            "robot_frame": str(robot_dir / "step_0000_top_down.png"),
                            "iteration_video_gif": str(video_dir / "iteration_01_grip_from_edge_cube.gif"),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (loop_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "env_id": "MuJoCoPickLift-v1",
                        "seed": 98300,
                        "start_contract": "pick_up_reset",
                        "success_rate": 0.0,
                        "episodes_completed": 1,
                        "camera_contract": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "plan": {"task": "lift the green cube", "calls": []},
                        "qwen_prompts": {"user": "Task: lift the green cube"},
                        "episodes": [
                            {
                                "episode": 0,
                                "final_success": False,
                                "total_reward": 0.1,
                                "steps": 1,
                                "reset_info": {"tcp_to_obj_dist": 0.16},
                                "final_info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0, "lift_height": 0.0},
                                "trace_path": str(timeline_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            listing = dataset_viewer._loop_tests_payload(repo_root)
            pick_item = listing["exports"][0]["loop_tests"][2]
            detail = dataset_viewer._loop_frame_payload(
                repo_root,
                listing["exports"][0]["id"],
                pick_item["loop_test_id"],
                0,
                0,
            )

        self.assertEqual(listing["source"], "official_closed_loop_test_cases")
        self.assertEqual(listing["exports"][0]["id"], "official_qwen_edge_test_cases")
        self.assertEqual(
            [item["test_case_id"] for item in listing["exports"][0]["loop_tests"]],
            ["move_align_pick_up", "align_pick_up", "pick_up_only"],
        )
        self.assertEqual(listing["exports"][0]["loop_tests"][0]["status"], "not_run")
        self.assertEqual(pick_item["checkpoint"], "000086")
        self.assertEqual(pick_item["configured_seed"], 98300)
        self.assertEqual(pick_item["start_contract"], "pick_up_reset")
        self.assertEqual(pick_item["loop_test_id"], "qwen_chain_pick_up_only_seed98300_000086")
        self.assertEqual(detail["plan"]["task"], "lift the green cube")
        self.assertEqual(detail["step"]["policy_input_prompt"], "Close the gripper on the green cube edge and lift.")
        self.assertIn("egocentric_cam", detail["images"]["policy_inputs"])
        self.assertIn("data:image/png;base64", detail["images"]["policy_inputs"]["egocentric_cam"])
        self.assertIn("egocentric_cam", detail["start_images"]["policy_inputs"])
        self.assertIn("top_down", detail["start_images"]["robot_frames"])
        self.assertIn("top_down", detail["images"]["robot_frames"])
        self.assertEqual(detail["start_video"]["name"], "iteration_01_grip_from_edge_cube.gif")

    def test_dataset_viewer_closed_loop_preview_http_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            _write_viewer_closed_loop_fixture(repo_root)
            html = _handle_viewer_request(repo_root, "/")
            listing = _handle_viewer_request(repo_root, "/api/loop-tests")
            preview = _handle_viewer_request(
                repo_root,
                "/api/loop-frame"
                "?export=official_qwen_edge_test_cases"
                "&loop=qwen_chain_pick_up_only_seed98300_000086"
                "&episode=0"
                "&step=0",
            )

        self.assertIn('id="loopPolicyCameras"', html)
        self.assertIn('id="loopStartCameras"', html)
        self.assertIn('id="viewKind"', html)
        self.assertIn('value="train" selected>train', html)
        self.assertIn('value="valid">valid', html)
        self.assertIn('value="closed_loop">closed loop test case', html)
        self.assertNotIn('id="datasetTab"', html)
        self.assertNotIn('id="loopTab"', html)
        self.assertIn("Episode start images", html)
        self.assertIn("loopPlaybackTick", html)
        self.assertEqual(listing["source"], "official_closed_loop_test_cases")
        pick_item = listing["exports"][0]["loop_tests"][2]
        self.assertEqual(pick_item["test_case_id"], "pick_up_only")
        self.assertEqual(pick_item["configured_seed"], 98300)
        self.assertEqual(pick_item["start_contract"], "pick_up_reset")
        self.assertFalse(pick_item["loop_test_id"] == "")
        self.assertEqual(preview["loop_test"]["id"], "qwen_chain_pick_up_only_seed98300_000086")
        self.assertEqual(preview["step"]["policy_input_prompt"], "Close the gripper on the green cube edge and lift.")
        self.assertIn("egocentric_cam", preview["images"]["policy_inputs"])
        self.assertIn("egocentric_cam", preview["start_images"]["policy_inputs"])
        self.assertIn("wrist_cam", preview["images"]["policy_inputs"])
        self.assertIn("wrist_cam", preview["start_images"]["policy_inputs"])
        self.assertIn("top_down", preview["images"]["robot_frames"])
        self.assertIn("top_down", preview["start_images"]["robot_frames"])
        self.assertTrue(preview["images"]["policy_inputs"]["egocentric_cam"].startswith("data:image/png;base64,"))
        self.assertTrue(preview["start_images"]["policy_inputs"]["egocentric_cam"].startswith("data:image/png;base64,"))
        self.assertEqual(preview["step"]["media_available"], True)

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


def _write_viewer_closed_loop_fixture(repo_root: Path) -> None:
    config_dir = repo_root / "configs" / "so101" / "training_datasets"
    config_dir.mkdir(parents=True)
    (config_dir / "qwen_edge_primitives.json").write_text(
        json.dumps(
            {
                "closed_loop": {
                    "test_cases": [
                        {
                            "id": "move_align_pick_up",
                            "description": "Requires move, align, and pick up primitives to succeed.",
                            "episodes": 5,
                            "steps": 120,
                            "seed": 98100,
                            "start_contract": "full_chain_reset",
                            "task_prompt": "pick and lift the green cube",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_move_align_pick_up_green_cube.json",
                        },
                        {
                            "id": "align_pick_up",
                            "description": "Starts close enough that align and pick up primitives should be sufficient.",
                            "episodes": 5,
                            "steps": 120,
                            "seed": 98200,
                            "start_contract": "align_pick_reset",
                            "task_prompt": "align with and lift the green cube",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_align_pick_up_green_cube.json",
                        },
                        {
                            "id": "pick_up_only",
                            "description": "Starts from the pick-up primitive distribution and only requires pick up.",
                            "episodes": 5,
                            "steps": 120,
                            "seed": 98300,
                            "start_contract": "pick_up_reset",
                            "task_prompt": "lift the green cube",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_pick_up_green_cube.json",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    run_dir = repo_root / "_workspace" / "so101_training" / "runs" / "debug_run"
    loop_dir = run_dir / "closed_loop_evals" / "qwen_chain_pick_up_only_seed98300_000086"
    media_dir = loop_dir / "media"
    policy_dir = media_dir / "policy_inputs"
    robot_dir = media_dir / "robot_frames"
    policy_dir.mkdir(parents=True)
    robot_dir.mkdir(parents=True)
    (policy_dir / "step_0000_egocentric_cam.png").write_bytes(VALID_1X1_PNG)
    (policy_dir / "step_0000_wrist_cam.png").write_bytes(VALID_1X1_PNG)
    (robot_dir / "step_0000_top_down.png").write_bytes(VALID_1X1_PNG)
    trace_path = loop_dir / "qwen_closed_loop_episode_000.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "episode": 0,
                "global_step": 0,
                "primitive_step": 0,
                "fn": "pick_up",
                "primitive_id": "grip_from_edge_cube",
                "prompt": "Close the gripper on the green cube edge and lift.",
                "image_feature_mapping": {
                    "observation.images.camera1": "egocentric_cam",
                    "observation.images.camera2": "wrist_cam",
                },
                "policy_rollout_config": {"n_action_steps": 15},
                "action": [0.0],
                "info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0},
                "media": {
                    "policy_input_images": {
                        "egocentric_cam": str(policy_dir / "step_0000_egocentric_cam.png"),
                        "wrist_cam": str(policy_dir / "step_0000_wrist_cam.png"),
                    },
                    "robot_frame": str(robot_dir / "step_0000_top_down.png"),
                    "render_mode": "inline",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (loop_dir / "qwen_closed_loop_eval_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "env_id": "MuJoCoPickLift-v1",
                "env_config": {"object_shape": "cube", "object_color": "green", "n_distractors": 0},
                "seed": 98300,
                "start_contract": "pick_up_reset",
                "success_rate": 0.0,
                "episodes_completed": 1,
                "camera_contract": {
                    "observation.images.camera1": "egocentric_cam",
                    "observation.images.camera2": "wrist_cam",
                },
                "plan": {"task": "lift the green cube", "calls": []},
                "episodes": [
                    {
                        "episode": 0,
                        "final_success": False,
                        "total_reward": 0.1,
                        "steps": 1,
                        "reset_info": {"tcp_to_obj_dist": 0.16},
                        "final_info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0, "lift_height": 0.0},
                        "trace_path": str(trace_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _handle_viewer_request(repo_root: Path, path: str) -> Any:
    handler_cls = dataset_viewer.make_handler(repo_root)
    request = f"GET {path} HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n\r\n".encode("utf-8")
    fake_socket = _FakeSocket(request)
    handler_cls(fake_socket, ("127.0.0.1", 12345), _FakeServer())
    raw_response = fake_socket.output.getvalue()
    header_bytes, body = raw_response.split(b"\r\n\r\n", 1)
    headers = header_bytes.decode("iso-8859-1")
    if "application/json" in headers:
        return json.loads(body.decode("utf-8"))
    return body.decode("utf-8")


class _FakeSocket:
    def __init__(self, request: bytes) -> None:
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, mode: str, buffering: int | None = None):
        del buffering
        if "r" in mode:
            return self.input
        return self.output

    def sendall(self, payload: bytes) -> None:
        self.output.write(payload)


class _FakeServer:
    server_name = "example.test"
    server_port = 80


if __name__ == "__main__":
    unittest.main()
