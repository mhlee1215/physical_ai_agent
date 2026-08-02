from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from physical_ai_agent.so101_training_config_schema import (
    load_so101_training_schema,
    normalize_so101_training_config,
    parse_so101_training_config,
    resolve_so101_training_config_defaults,
    validate_so101_training_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_so101_picklift_smolvla_policy as evaluator  # noqa: E402
import monitor_so101_training_dashboard as dashboard  # noqa: E402
import start_so101_training as launcher  # noqa: E402


CONFIG_PATH = (
    REPO_ROOT
    / "configs/so101/training/grip_the_cube_v3_hardware_locked_photoreal_phases_v1.json"
)
PERIODIC_INDICES = [0, 6, 12, 18, 24, 25, 31, 37, 43, 49]


def _resolved_config() -> dict[str, object]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return normalize_so101_training_config(
        resolve_so101_training_config_defaults(
            payload,
            path=CONFIG_PATH,
            repo_root=REPO_ROOT,
        )
    )


def _arg_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _report_episode(index: int, *, seed: int | None = None) -> dict[str, object]:
    return {
        "seed": int(seed if seed is not None else 1000 + index),
        "task": "phase prompt",
        "object_color": "green",
        "object_shape": "cube",
        "forced_spawn_xy": [0.2 + index * 0.001, 0.1],
        "source_provenance": {"episode_index": index},
        "frames": 10,
        "phase_counts": {"fixture": 10},
        "sim_snapshot": {
            "qpos": [0.0],
            "qvel": [0.0],
            "ctrl": [0.0],
        },
    }


class SO101PhaseChainTrainingTests(unittest.TestCase):
    def test_checked_in_training_schema_matches_pydantic_model(self) -> None:
        schema_path = REPO_ROOT / "configs/so101/schemas/training_config.schema.json"
        self.assertEqual(
            json.loads(schema_path.read_text(encoding="utf-8")),
            load_so101_training_schema(REPO_ROOT),
        )

    def test_phase_training_config_is_strict_and_virtual(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            validate_so101_training_config(
                payload,
                path=CONFIG_PATH,
                repo_root=REPO_ROOT,
            ),
            [],
        )
        parsed = parse_so101_training_config(
            payload,
            path=CONFIG_PATH,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(len(parsed.train_datasets or []), 3)
        self.assertEqual(len(parsed.validation_datasets or []), 3)
        self.assertIsNone(parsed.train_dataset)
        self.assertIsNone(parsed.validation_dataset)

    def test_automatic_schedules_run_only_the_continuous_chain(self) -> None:
        config = _resolved_config()
        test_cases = config["closed_loop"]["test_cases"]
        periodic = [case for case in test_cases if case["schedule"] == "periodic"]
        final = [case for case in test_cases if case["schedule"] == "final"]
        manual = [case for case in test_cases if case["schedule"] == "manual"]

        self.assertEqual(
            [case["id"] for case in periodic],
            ["grip_chain_continuous_periodic10"],
        )
        self.assertEqual(
            [case["id"] for case in final],
            ["grip_chain_continuous_final50"],
        )
        self.assertEqual(len(manual), 7)
        self.assertEqual(periodic[0]["episodes"], 10)
        self.assertEqual(periodic[0]["episode_indices"], PERIODIC_INDICES)
        self.assertEqual(periodic[0]["source_grid_bins"], [9, 10])
        self.assertEqual(final[0]["episodes"], 50)
        self.assertNotIn("episode_indices", final[0])
        self.assertEqual(
            [
                case["id"]
                for case in manual
                if case["phase_contract"]["handoff_mode"] == "oracle_reset"
            ],
            ["grip_chain_oracle_handoff_debug10"],
        )

    def test_chain_prompts_caps_and_continuous_handoff_match_contract(self) -> None:
        config = _resolved_config()
        chain = next(
            case
            for case in config["closed_loop"]["test_cases"]
            if case["id"] == "grip_chain_continuous_periodic10"
        )
        phases = chain["phase_contract"]["phases"]

        self.assertEqual(chain["steps"], 266)
        self.assertEqual(chain["phase_contract"]["handoff_mode"], "continuous")
        self.assertEqual(
            [
                (
                    phase["id"],
                    phase["max_steps"],
                    phase["reference_length_multiplier"],
                    phase["prompt"],
                )
                for phase in phases
            ],
            [
                (
                    "approach",
                    66,
                    1.5,
                    "Open the gripper and move it above the visible green cube.",
                ),
                (
                    "alignment",
                    66,
                    1.5,
                    "Align the open gripper jaws with the visible green cube edge.",
                ),
                (
                    "grip_lift",
                    134,
                    1.5,
                    "Close the aligned gripper on the visible green cube and lift it.",
                ),
            ],
        )
        self.assertTrue(
            config["closed_loop"]["observation_renderer"]["render_policy_inference_only"]
        )
        self.assertEqual(
            config["closed_loop"]["tensorboard_media"]["train_reference_frequency"],
            "once_per_run",
        )
        self.assertEqual(
            config["closed_loop"]["tensorboard_media"]["chain_rollout_layout"],
            "per_episode",
        )
        self.assertEqual(
            config["closed_loop"]["tensorboard_media"]["render_test_cases"],
            [
                "grip_chain_continuous_periodic10",
                "grip_chain_continuous_final50",
            ],
        )
        self.assertTrue(
            config["closed_loop"]["action_rmse_sweep"][
                "render_policy_inference_only"
            ]
        )
        self.assertEqual(
            config["closed_loop"]["action_rmse_sweep"]["n_action_steps"],
            [5, 15, 30, 50],
        )
        self.assertEqual(
            config["closed_loop"]["action_rmse_sweep"]["y_axis_max"],
            0.2,
        )
        self.assertEqual(
            config["closed_loop"]["action_rmse_sweep"]["timeline_mode"],
            "phase_chain",
        )
        self.assertEqual(
            config["closed_loop"]["action_rmse_sweep"]["test_cases"],
            [
                "grip_chain_continuous_periodic10",
                "grip_chain_continuous_final50",
            ],
        )
        self.assertEqual(
            config["closed_loop"]["action_rmse_sweep"][
                "phase_contract_test_case_id"
            ],
            "grip_chain_continuous_periodic10",
        )

    def test_phase_reference_multiplier_is_checked_against_declared_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "phase.json"
            report_path.write_text(
                json.dumps({"episodes": [_report_episode(0)]}),
                encoding="utf-8",
            )
            phase = {
                "id": "approach",
                "prompt": "approach",
                "max_steps": 15,
                "reference_length_multiplier": 1.5,
                "reference_report_path": str(report_path),
                "verifier": {"kind": "approach"},
            }
            loaded = evaluator._load_phase_contract_episodes(
                {
                    "mode": "primitive",
                    "handoff_mode": "continuous",
                    "phases": [phase],
                },
                episode_indices=[0],
                episodes=1,
            )
            episode_contract = evaluator._phase_contract_for_episode(loaded, episode=0)

            self.assertEqual(episode_contract["phases"][0]["reference_steps"], 10)
            self.assertEqual(
                evaluator._phase_contract_episode_max_steps(
                    episode_contract,
                    configured_max_steps=15,
                ),
                15,
            )
            phase["max_steps"] = 14
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluator._load_phase_contract_episodes(
                    {
                        "mode": "primitive",
                        "handoff_mode": "continuous",
                        "phases": [phase],
                    },
                    episode_indices=[0],
                    episodes=1,
                )

    def test_chain_tensorboard_rollout_is_one_video_per_episode(self) -> None:
        import torch

        report = {
            "phase_contract": {"mode": "chain"},
            "episodes": [{"episode": 0}, {"episode": 1}],
        }
        fake_videos = {
            "camera1_camera2_episode_000": torch.zeros(
                (1, 3, 3, 32, 64), dtype=torch.uint8
            ),
            "camera1_camera2_episode_001": torch.ones(
                (1, 2, 3, 32, 64), dtype=torch.uint8
            ),
        }

        self.assertEqual(
            dashboard._closed_loop_rollout_layout(REPO_ROOT, report),
            "per_episode",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "training_run_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_config": {
                            "config_path": str(CONFIG_PATH),
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                dashboard._closed_loop_rollout_layout(run_dir, report),
                "per_episode",
            )
        rollout_videos = dashboard._canonical_closed_loop_rollout_videos(
            side_by_side_videos=fake_videos
        )
        self.assertEqual(
            sorted(rollout_videos),
            ["episode_000", "episode_001"],
        )
        self.assertEqual(tuple(rollout_videos["episode_000"].shape), (1, 3, 3, 32, 64))
        self.assertEqual(tuple(rollout_videos["episode_001"].shape), (1, 2, 3, 32, 64))

        dashboard._require_closed_loop_tensorboard_evidence(
            {"test_id": "grip_chain"},
            {
                **report,
                "action_rmse_sweep": {"skipped": "test_fixture"},
            },
            {
                "status": "ok",
                "media_enabled": True,
                "rollout_layout": "per_episode",
                "scalars": ["closed_loop/grip_chain/success_rate"],
                "images": [],
                "videos": [
                    "closed_loop/grip_chain/rollout_episode_000",
                    "closed_loop/grip_chain/rollout_episode_001",
                    "closed_loop/grip_chain/train_reference_camera1_camera2_episode_000",
                ],
            },
        )

    def test_chain_train_reference_concatenates_all_phases_per_episode(self) -> None:
        phase_roots = {
            phase: Path(f"/tmp/{phase}")
            for phase in ("approach", "alignment", "grip_lift")
        }
        report = {
            "phase_contract": {"mode": "chain"},
            "episodes": [
                {
                    "episode": 0,
                    "reset_meta": {"source_report_episode_index": 7},
                    "phase_contract": {
                        "mode": "chain",
                        "phases": [
                            {
                                "id": phase,
                                "prompt": f"{phase} prompt",
                                "max_steps": 10,
                                "reference_report_path": str(
                                    phase_roots[phase] / "so101_lerobot_export_report.json"
                                ),
                                "reference_report_episode_index": 7,
                            }
                            for phase in ("approach", "alignment", "grip_lift")
                        ],
                    },
                }
            ],
        }

        def fake_root(value: str) -> Path:
            return Path(value).parent

        def fake_frames(
            dataset_root: Path,
            *,
            max_episodes: int,
            max_frames_per_episode: int,
            episode_indices: list[int] | None = None,
        ) -> dict[str, dict[int, list[tuple[object, ...]]]]:
            del max_episodes, max_frames_per_episode
            source_episode = int((episode_indices or [7])[0])
            phase_value = {
                "approach": 30,
                "alignment": 90,
                "grip_lift": 150,
            }[dataset_root.name]
            image = np.full((16, 16, 3), phase_value, dtype=np.uint8)
            frames = [(image, "source", False, None, None)] * 2
            return {
                "camera1": {source_episode: frames},
                "camera2": {source_episode: frames},
            }

        with (
            mock.patch.object(
                dashboard,
                "_phase_reference_dataset_root",
                side_effect=fake_root,
            ),
            mock.patch.object(
                dashboard,
                "_training_reference_camera_frames_by_episode",
                side_effect=fake_frames,
            ),
        ):
            videos = dashboard._phase_chain_reference_camera_side_by_side_videos(
                report
            )

        self.assertEqual(sorted(videos), ["camera1_camera2_episode_000"])
        self.assertEqual(
            tuple(videos["camera1_camera2_episode_000"].shape),
            (1, 6, 3, 16, 38),
        )

    def test_launcher_maps_all_train_validation_and_loop_schedules(self) -> None:
        config = _resolved_config()
        training_args = launcher._with_dataset_config(
            [],
            config,
            runtime_platform="macos",
        )
        train_entries = json.loads(
            launcher._arg_value(training_args, "train-datasets-json")
        )
        validation_entries = json.loads(
            launcher._arg_value(training_args, "validation-datasets-json")
        )
        self.assertEqual(len(train_entries), 3)
        self.assertEqual(len(validation_entries), 3)

        base = [
            sys.executable,
            str(SCRIPTS / "monitor_so101_training_dashboard.py"),
            "--closed-loop-episodes",
            "10",
            "--closed-loop-steps",
            "240",
            "--closed-loop-observation-renderer-json",
            json.dumps(
                config["closed_loop"]["observation_renderer"],
                sort_keys=True,
            ),
        ]
        periodic = launcher._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
            schedule="periodic",
        )
        final = launcher._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
            schedule="final",
        )
        manual = launcher._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
            schedule="manual",
        )
        self.assertEqual((len(periodic), len(final), len(manual)), (1, 1, 7))
        self.assertIn("--skip-validation", periodic[0])
        self.assertEqual(
            _arg_value(periodic[0], "--closed-loop-episode-indices"),
            ",".join(str(index) for index in PERIODIC_INDICES),
        )
        for command in periodic:
            renderer = json.loads(
                _arg_value(command, "--closed-loop-observation-renderer-json")
            )
            self.assertTrue(renderer["render_policy_inference_only"])
        self.assertIn("--record-loop-artifacts", periodic[0])
        self.assertIn("--render-loop-media", periodic[0])
        self.assertIsNotNone(
            _arg_value(periodic[0], "--closed-loop-phase-contract-json")
        )
        oracle = next(
            command
            for command in manual
            if _arg_value(command, "--closed-loop-test-id")
            == "grip_chain_oracle_handoff_debug10"
        )
        self.assertEqual(
            json.loads(_arg_value(oracle, "--closed-loop-phase-contract-json"))[
                "handoff_mode"
            ],
            "oracle_reset",
        )

    def test_runtime_refresh_renders_only_allowlisted_chain_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "training_run_summary.json").write_text(
                json.dumps({"dataset_config": {"config_path": str(CONFIG_PATH)}}),
                encoding="utf-8",
            )
            self.assertFalse(
                dashboard._closed_loop_test_case_media_enabled(
                    run_dir,
                    "approach_phase_periodic10",
                )
            )
            self.assertTrue(
                dashboard._closed_loop_test_case_media_enabled(
                    run_dir,
                    "grip_chain_continuous_periodic10",
                )
            )
            args = SimpleNamespace(
                repo_root=REPO_ROOT,
                closed_loop_test_id="approach_phase_periodic10",
                closed_loop_record_rollout_gif=True,
                record_loop_artifacts=True,
                render_loop_media=True,
            )
            dashboard._refresh_closed_loop_runtime_from_config(args, run_dir)
            self.assertFalse(args.closed_loop_record_rollout_gif)
            self.assertFalse(args.record_loop_artifacts)
            self.assertFalse(args.render_loop_media)

            args.closed_loop_test_id = "grip_chain_continuous_periodic10"
            dashboard._refresh_closed_loop_runtime_from_config(args, run_dir)
            self.assertTrue(args.closed_loop_record_rollout_gif)
            self.assertTrue(args.record_loop_artifacts)
            self.assertTrue(args.render_loop_media)

    def test_rmse_sweep_forces_inference_only_rendering(self) -> None:
        renderer = {
            "mode": "blender_cycles_live",
            "render_policy_inference_only": False,
        }
        resolved = json.loads(
            dashboard._action_rmse_sweep_observation_renderer_json(
                json.dumps(renderer),
                render_policy_inference_only=True,
            )
        )
        self.assertTrue(resolved["render_policy_inference_only"])

    def test_rmse_phase_chain_timeline_contains_all_three_phase_datasets(self) -> None:
        config = _resolved_config()
        chain = next(
            case
            for case in config["closed_loop"]["test_cases"]
            if case["id"] == "grip_chain_continuous_periodic10"
        )
        segments = dashboard._action_rmse_timeline_segments(
            SimpleNamespace(
                repo_root=REPO_ROOT,
                closed_loop_phase_contract_json=json.dumps(chain["phase_contract"]),
                closed_loop_start_report_path=Path(chain["start_report_path"]),
            ),
            timeline_mode="phase_chain",
        )

        self.assertEqual(
            [(segment["id"], segment["steps"]) for segment in segments],
            [("approach", 66), ("alignment", 66), ("grip_lift", 134)],
        )
        self.assertEqual(
            [Path(segment["dataset_root"]).name for segment in segments],
            [
                "grip_the_cube_v3_hardware_locked_photoreal_approach_phase_v1_validation",
                "grip_the_cube_v3_hardware_locked_photoreal_alignment_phase_v1_validation",
                "grip_the_cube_v3_hardware_locked_photoreal_grip_lift_phase_v1_validation",
            ],
        )

    def test_rmse_plot_marks_phase_transitions_and_uses_one_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "combined.png"
            dashboard._plot_action_rmse_sweep(
                plot_path=plot_path,
                series_by_n={5: [0.1] * 9, 15: [0.2] * 9},
                rows=[
                    {
                        "n_action_steps": value,
                        "success": True,
                        "reference_drift_mean": 0.1,
                        "approach_mean": 0.08,
                        "alignment_mean": 0.1,
                        "grip_lift_mean": 0.12,
                    }
                    for value in (5, 15)
                ],
                teacher_frames=9,
                phase_segments=[
                    {"id": "approach", "start_frame": 0, "end_frame_exclusive": 3},
                    {"id": "alignment", "start_frame": 3, "end_frame_exclusive": 6},
                    {"id": "grip_lift", "start_frame": 6, "end_frame_exclusive": 9},
                ],
                y_axis_max=0.2,
            )

            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_rmse_sweep_runs_every_phase_and_combines_one_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "eval_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "skill_success": True,
                                "reset_meta": {
                                    "source_report_episode_index": 7,
                                },
                                "records": [],
                                "final_info": {
                                    "tcp_to_obj_dist": 0.01,
                                    "lift_height": 0.07,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            phase_contract = {
                "mode": "chain",
                "handoff_mode": "continuous",
                "phases": [
                    {
                        "id": phase_id,
                        "prompt": f"{phase_id} prompt",
                        "max_steps": max_steps,
                        "reference_report_path": str(
                            root / phase_id / "so101_lerobot_export_report.json"
                        ),
                        "verifier": {"kind": phase_id},
                    }
                    for phase_id, max_steps in (
                        ("approach", 66),
                        ("alignment", 66),
                        ("grip_lift", 134),
                    )
                ],
            }
            args = SimpleNamespace(
                repo_root=root,
                closed_loop_runner="picklift",
                closed_loop_action_rmse_sweep_n_action_steps="5,15,30,50",
                closed_loop_action_rmse_sweep_y_axis_max=0.2,
                closed_loop_action_rmse_sweep_timeline_mode="phase_chain",
                closed_loop_action_rmse_sweep_phase_contract_json=json.dumps(
                    phase_contract
                ),
            )

            with (
                mock.patch.object(
                    dashboard,
                    "resolve_lerobot_root_for_start_report",
                    side_effect=lambda path, repo_root: Path(path).parent,
                ),
                mock.patch.object(
                    dashboard,
                    "_closed_loop_sweep_command",
                    return_value=["fake-evaluator"],
                ) as sweep_command,
                mock.patch.object(
                    dashboard.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                ),
                mock.patch.object(
                    dashboard,
                    "_runtime_env",
                    return_value={},
                ),
                mock.patch.object(
                    dashboard,
                    "_closed_loop_eval_report_path",
                    return_value=report_path,
                ),
                mock.patch.object(
                    dashboard,
                    "_teacher_action_rmse_series_from_records",
                    return_value=([0.1, 0.2], 2),
                ),
                mock.patch.object(
                    dashboard,
                    "_plot_action_rmse_sweep",
                ) as plot,
            ):
                payload = dashboard._run_action_rmse_sweep(
                    args=args,
                    checkpoint="053013",
                    policy_path=root / "policy",
                    valid_mask_checkpoint=None,
                    closed_loop_test_id="grip_chain_continuous_periodic10",
                    output_dir=root / "output",
                )

            self.assertEqual(sweep_command.call_count, 12)
            self.assertEqual(payload["y_axis_max"], 0.2)
            self.assertEqual(plot.call_args.kwargs["y_axis_max"], 0.2)
            self.assertEqual(
                payload["phase_segments"],
                [
                    {
                        "id": "approach",
                        "start_frame": 0,
                        "end_frame_exclusive": 2,
                        "teacher_frames": 2,
                    },
                    {
                        "id": "alignment",
                        "start_frame": 2,
                        "end_frame_exclusive": 4,
                        "teacher_frames": 2,
                    },
                    {
                        "id": "grip_lift",
                        "start_frame": 4,
                        "end_frame_exclusive": 6,
                        "teacher_frames": 2,
                    },
                ],
            )
            self.assertEqual(len(payload["rows"]), 4)
            for row in payload["rows"]:
                self.assertEqual(
                    sorted(row["phase_metrics"]),
                    ["alignment", "approach", "grip_lift"],
                )
                self.assertTrue(row["success"])
            plot_kwargs = plot.call_args.kwargs
            self.assertEqual(sorted(plot_kwargs["series_by_n"]), [5, 15, 30, 50])
            self.assertTrue(
                all(len(series) == 6 for series in plot_kwargs["series_by_n"].values())
            )

    def test_phase_report_selection_and_alignment_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            for phase_id in ("approach", "alignment", "grip_lift"):
                path = root / f"{phase_id}.json"
                path.write_text(
                    json.dumps(
                        {"episodes": [_report_episode(index) for index in range(4)]}
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            contract = {
                "mode": "chain",
                "handoff_mode": "continuous",
                "phases": [
                    {
                        "id": phase_id,
                        "prompt": phase_id,
                        "max_steps": 10,
                        "reference_report_path": str(path),
                        "verifier": {"kind": phase_id},
                    }
                    for phase_id, path in zip(
                        ("approach", "alignment", "grip_lift"),
                        paths,
                        strict=True,
                    )
                ],
            }
            selected = [1, 3]
            start = evaluator._load_start_report_episodes(
                paths[0],
                episode_indices=selected,
                limit=2,
            )
            loaded = evaluator._load_phase_contract_episodes(
                contract,
                episode_indices=selected,
                episodes=2,
            )

            evaluator._validate_phase_episode_alignment(
                start_report_episodes=start,
                phase_contract=loaded,
            )
            self.assertEqual(
                [episode["_report_episode_index"] for episode in start],
                selected,
            )

    def test_local_phase_validation_reports_align_for_all_50_episodes(self) -> None:
        report_paths = [
            REPO_ROOT
            / "_workspace/so101_lerobot"
            / f"grip_the_cube_v3_hardware_locked_photoreal_{phase}_phase_v1_validation"
            / "so101_lerobot_export_report.json"
            for phase in ("approach", "alignment", "grip_lift")
        ]
        if not all(path.exists() for path in report_paths):
            self.skipTest("local phase validation artifacts are not present")
        contract = {
            "mode": "chain",
            "handoff_mode": "continuous",
            "phases": [
                {
                    "id": phase,
                    "reference_report_path": str(path),
                }
                for phase, path in zip(
                    ("approach", "alignment", "grip_lift"),
                    report_paths,
                    strict=True,
                )
            ],
        }
        start = evaluator._load_start_report_episodes(report_paths[0], limit=50)
        loaded = evaluator._load_phase_contract_episodes(
            contract,
            episode_indices=None,
            episodes=50,
        )

        evaluator._validate_phase_episode_alignment(
            start_report_episodes=start,
            phase_contract=loaded,
        )
        selected_bins = [
            start[index]["grid_balance_bin"]
            for index in PERIODIC_INDICES
        ]
        self.assertEqual(selected_bins[:5], [9] * 5)
        self.assertEqual(selected_bins[5:], [10] * 5)

    def test_teacher_phase_endpoints_pass_geometric_verifiers(self) -> None:
        config = _resolved_config()
        cases = {
            case["phase_contract"]["phases"][0]["id"]: case
            for case in config["closed_loop"]["test_cases"]
            if case["id"].endswith("_periodic10")
            and case["phase_contract"]["mode"] == "primitive"
        }
        try:
            env = evaluator._make_eval_env(
                "grip_the_cube_v1",
                target_object_color="green",
                env_config=cases["approach"]["env_config"],
            )
        except ImportError as exc:
            self.skipTest(f"SO101 MuJoCo runtime unavailable: {exc}")
        try:
            for phase_id in ("approach", "alignment"):
                case = cases[phase_id]
                report_episode = evaluator._load_start_report_episodes(
                    Path(case["start_report_path"]),
                    episode_indices=[0],
                    limit=1,
                )[0]
                evaluator._restore_report_start_state(env, report_episode)
                evaluator._set_qpos(
                    env,
                    np.asarray(
                        report_episode["phase_end_observation_state"],
                        dtype=float,
                    ),
                )
                phase = {
                    **case["phase_contract"]["phases"][0],
                    "_reference_episode": report_episode,
                }
                result = evaluator._evaluate_phase_verifier(
                    env=env,
                    phase=phase,
                    hold_streak=0,
                )
                self.assertTrue(result["passed"], result)

            import pyarrow.dataset as arrow_dataset

            grip_case = cases["grip_lift"]
            grip_episode = evaluator._load_start_report_episodes(
                Path(grip_case["start_report_path"]),
                episode_indices=[0],
                limit=1,
            )[0]
            evaluator._restore_report_start_state(env, grip_episode)
            action_table = arrow_dataset.dataset(
                str(REPO_ROOT / grip_case["start_dataset"]["root"] / "data"),
                format="parquet",
            ).to_table(
                filter=arrow_dataset.field("episode_index") == 0,
                columns=["frame_index", "action"],
            )
            teacher_actions = sorted(
                zip(
                    action_table["frame_index"].to_pylist(),
                    action_table["action"].to_pylist(),
                    strict=True,
                ),
                key=lambda row: row[0],
            )
            grip_phase = {
                **grip_case["phase_contract"]["phases"][0],
                "_reference_episode": grip_episode,
            }
            hold_streak = 0
            for _frame_index, action in teacher_actions:
                _observation, _reward, _terminated, _truncated, info = env.step(
                    np.asarray(action, dtype=float)
                )
                if (
                    float(info.get("is_grasped", 0.0)) > 0.5
                    and float(info.get("lift_height", 0.0))
                    >= float(grip_phase["verifier"]["lift_height_m"])
                ):
                    hold_streak += 1
                else:
                    hold_streak = 0
            grip_result = evaluator._evaluate_phase_verifier(
                env=env,
                phase=grip_phase,
                hold_streak=hold_streak,
            )
            self.assertTrue(grip_result["passed"], grip_result)
        finally:
            env.close()

    def test_phase_report_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            approach_path = root / "approach.json"
            alignment_path = root / "alignment.json"
            approach_path.write_text(
                json.dumps({"episodes": [_report_episode(0)]}),
                encoding="utf-8",
            )
            alignment_path.write_text(
                json.dumps({"episodes": [_report_episode(0, seed=9999)]}),
                encoding="utf-8",
            )
            loaded = evaluator._load_phase_contract_episodes(
                {
                    "mode": "chain",
                    "handoff_mode": "continuous",
                    "phases": [
                        {
                            "id": "approach",
                            "reference_report_path": str(approach_path),
                        },
                        {
                            "id": "alignment",
                            "reference_report_path": str(alignment_path),
                        },
                    ],
                },
                episode_indices=[0],
                episodes=1,
            )
            start = evaluator._load_start_report_episodes(
                approach_path,
                episode_indices=[0],
                limit=1,
            )

            with self.assertRaisesRegex(ValueError, "not aligned"):
                evaluator._validate_phase_episode_alignment(
                    start_report_episodes=start,
                    phase_contract=loaded,
                )

    def test_valid_mask_proposal_requires_verifier_before_advance(self) -> None:
        self.assertEqual(
            evaluator._phase_transition_outcome(
                verifier_passed=False,
                final_phase=False,
            ),
            {"action": "continue", "reason": "valid_mask_verifier_rejected"},
        )
        self.assertEqual(
            evaluator._phase_transition_outcome(
                verifier_passed=True,
                final_phase=False,
            ),
            {"action": "advance", "reason": "env_success"},
        )
        self.assertEqual(
            evaluator._phase_transition_outcome(
                verifier_passed=True,
                final_phase=True,
            ),
            {"action": "complete", "reason": "env_success"},
        )

    def test_tensorboard_media_reads_phase_contract_trace_keys(self) -> None:
        metadata = dashboard._closed_loop_frame_metadata(
            {
                "phase": "alignment",
                "termination_reason": "valid_mask_verifier_rejected",
            },
            "observation.images.camera1",
        )

        self.assertEqual(metadata["phase"], "alignment")
        self.assertEqual(
            metadata["termination_reason"],
            "valid_mask_verifier_rejected",
        )
        self.assertEqual(
            dashboard._phase_border_color("alignment"),
            (90, 180, 255),
        )
        self.assertIn("verifier-rejected", dashboard._phase_label(metadata))


if __name__ == "__main__":
    unittest.main()
