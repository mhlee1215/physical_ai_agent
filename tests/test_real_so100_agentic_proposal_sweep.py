from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_agentic_proposal_sweep import (
    feedback_driven_prompts,
    memory_refine_prompts,
    memory_residual_prompts,
    memory_sample_prompts,
    memory_structured_prompts,
    policy_camera_feedback_prompts,
    run_proposal_sweep,
    score_execute_dry_gate,
)


class RealSO100AgenticProposalSweepTest(TestCase):
    def test_score_prefers_fewer_and_smaller_range_violations(self) -> None:
        score = score_execute_dry_gate(
            {
                "dry_plan": {
                    "ready_for_execution": False,
                    "blockers": ["blocked"],
                    "step_plans": [
                        {
                            "step_index": 0,
                            "joint_targets": [
                                {"joint": "shoulder_pan", "target_raw": 1500, "range_min": 1000, "range_max": 2000},
                                {"joint": "elbow_flex", "target_raw": 2200, "range_min": 1000, "range_max": 2000},
                                {"joint": "wrist_flex", "target_raw": 850, "range_min": 1000, "range_max": 2000},
                            ],
                        },
                    ],
                }
            }
        )

        self.assertFalse(score["ready_for_execution"])
        self.assertEqual(score["range_violation_count"], 2)
        self.assertEqual(score["total_range_excess_raw_ticks"], 350.0)
        self.assertEqual(score["max_range_excess_raw_ticks"], 200.0)
        self.assertEqual(score["violation_joint_counts"], {"elbow_flex": 1, "wrist_flex": 1})
        self.assertGreater(score["range_penalty_score"], score["total_range_excess_raw_ticks"])

    def test_feedback_prompts_use_dominant_joint_blockers(self) -> None:
        prompts = feedback_driven_prompts(
            {
                "status": "passed",
                "ranked_candidates": [
                    {
                        "score": {
                            "violation_joint_counts": {"shoulder_lift": 6, "elbow_flex": 4},
                            "violation_joint_excess_raw_ticks": {"shoulder_lift": 1000, "elbow_flex": 800},
                        }
                    }
                ],
            }
        )

        self.assertTrue(prompts)
        self.assertIn("shoulder low", prompts[0])
        self.assertIn("elbow bent", prompts[0])

    def test_projection_feedback_prompts_use_distortion_joints(self) -> None:
        prompts = feedback_driven_prompts(
            {
                "status": "passed",
                "ranked_candidates": [
                    {
                        "projection": {
                            "joint_distortion": {
                                "shoulder_lift": {"violation_count": 10, "total_raw_distortion": 5000},
                                "elbow_flex": {"violation_count": 10, "total_raw_distortion": 3000},
                                "wrist_flex": {"violation_count": 10, "total_raw_distortion": 4000},
                            }
                        }
                    }
                ],
            },
            projection_aware=True,
        )

        self.assertTrue(prompts)
        self.assertIn("smallest reachable side pre-grasp", prompts[0])
        self.assertIn("wrist neutral", prompts[0])

    def test_residual_distortion_prompts_focus_wrist_and_shoulder(self) -> None:
        prompts = feedback_driven_prompts(
            {
                "status": "passed",
                "ranked_candidates": [
                    {
                        "projection": {
                            "joint_distortion": {
                                "wrist_flex": {"violation_count": 10, "total_raw_distortion": 4700},
                                "shoulder_lift": {"violation_count": 10, "total_raw_distortion": 3800},
                                "elbow_flex": {"violation_count": 5, "total_raw_distortion": 1900},
                            }
                        }
                    }
                ],
            },
            residual_distortion=True,
        )

        self.assertTrue(prompts)
        self.assertIn("wrist straight", prompts[0])
        self.assertIn("shoulder not rising", prompts[0])

    def test_memory_refine_prompts_preserve_selected_prompt(self) -> None:
        selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
        prompts = memory_refine_prompts(
            {
                "next_agentic_layer_step": {
                    "type": "reuse_best_historical_prompt_family",
                    "selected_prompt": selected,
                },
                "best_candidate": {"prompt": "fallback prompt"},
            }
        )

        self.assertGreaterEqual(len(prompts), 4)
        self.assertEqual(prompts[0], selected)
        self.assertIn("Stop earlier", prompts[1])

    def test_memory_sample_prompts_repeat_selected_prompt(self) -> None:
        selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
        prompts = memory_sample_prompts(
            {
                "next_agentic_layer_step": {
                    "type": "continue_from_latest_best_prompt_family",
                    "selected_prompt": selected,
                },
                "best_candidate": {"prompt": "fallback prompt"},
            },
            sample_count=5,
        )

        self.assertEqual(prompts, [selected, selected, selected, selected, selected])

    def test_memory_residual_prompts_preserve_anchor_and_target_dominant_joints(self) -> None:
        selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
        prompts = memory_residual_prompts(
            {
                "next_agentic_layer_step": {
                    "type": "reuse_best_historical_prompt_family",
                    "selected_prompt": selected,
                },
                "best_candidate": {"prompt": "fallback prompt"},
                "ranked_candidates": [
                    {
                        "score": {
                            "joint_violation_counts": {"shoulder_lift": 10, "wrist_flex": 5, "gripper": 8},
                            "joint_excess_raw_ticks": {
                                "elbow_flex": 946.6906,
                                "shoulder_lift": 2892.371,
                                "wrist_flex": 1499.1736,
                                "gripper": 73.4454,
                            },
                        }
                    }
                ],
            }
        )

        self.assertGreaterEqual(len(prompts), 4)
        self.assertEqual(prompts[0], selected)
        self.assertIn("wrist neutral", prompts[1])
        self.assertIn("upper arm low", prompts[1])
        self.assertTrue(any("elbow" in prompt and "midrange" in prompt for prompt in prompts))
        self.assertTrue(any("gripper open" in prompt for prompt in prompts))

    def test_memory_residual_prompts_do_not_duplicate_existing_clause(self) -> None:
        selected = (
            "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder. "
            "Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low."
        )
        prompts = memory_residual_prompts(
            {
                "next_agentic_layer_step": {
                    "type": "continue_from_latest_best_prompt_family",
                    "selected_prompt": selected,
                },
                "ranked_candidates": [
                    {
                        "score": {
                            "joint_violation_counts": {"shoulder_lift": 2, "wrist_flex": 3},
                            "joint_excess_raw_ticks": {"shoulder_lift": 848.8153, "wrist_flex": 612.3239},
                        }
                    }
                ],
            }
        )

        repeated = "Preserve the same side pre-grasp"
        self.assertEqual(prompts[0].count(repeated), 1)
        self.assertTrue(all(prompt.count(repeated) <= 1 for prompt in prompts))

    def test_memory_structured_prompts_compact_anchor_and_joint_hints(self) -> None:
        selected = (
            "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, "
            "extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral."
        )
        prompts = memory_structured_prompts(
            {
                "next_agentic_layer_step": {
                    "type": "reuse_best_historical_prompt_family",
                    "selected_prompt": selected,
                },
                "ranked_candidates": [
                    {
                        "score": {
                            "joint_violation_counts": {"elbow_flex": 3, "shoulder_lift": 2, "wrist_flex": 3},
                            "joint_excess_raw_ticks": {
                                "elbow_flex": 946.6906,
                                "shoulder_lift": 848.8153,
                                "wrist_flex": 612.3239,
                            },
                        }
                    }
                ],
            }
        )

        self.assertGreaterEqual(len(prompts), 5)
        self.assertEqual(prompts[0], selected)
        self.assertIn("Goal: Set up a conservative side pre-grasp", prompts[1])
        self.assertIn("elbow midrange", prompts[1])
        self.assertIn("low upper arm", prompts[1])
        self.assertIn("neutral wrist", prompts[1])
        self.assertNotIn("Preserve the same side pre-grasp", prompts[1])

    def test_policy_camera_feedback_prompt_targets_smolvla_only(self) -> None:
        prompt = "Use camera 1 for coarse approach, then camera 0 for local jaw alignment."
        prompts = policy_camera_feedback_prompts(
            {
                "operation": "real_so100_policy_camera_pseudo_llm_feedback",
                "pseudo_llm_feedback": {
                    "does_not_prompt_operator": True,
                    "next_smolvla_prompt": prompt,
                },
            }
        )

        self.assertEqual(prompts, [prompt])
        self.assertEqual(
            policy_camera_feedback_prompts(
                {
                    "operation": "real_so100_policy_camera_pseudo_llm_feedback",
                    "pseudo_llm_feedback": {
                        "does_not_prompt_operator": False,
                        "next_smolvla_prompt": prompt,
                    },
                }
            ),
            [],
        )

    def test_sweep_records_no_actuation_and_selects_lowest_excess_prompt(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=_fake_dry),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=["bad prompt", "better prompt"],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=None,
                    prompt_profile="default",
                )
                self.assertTrue(Path(report["json_path"]).exists())
                self.assertTrue(Path(report["markdown_path"]).exists())

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["policy_camera_indexes"], ["0", "1"])
        self.assertEqual(report["observer_camera_indexes"], [])
        self.assertEqual(report["observer_camera_status"], "temporarily_unavailable")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["policy_actions_executed"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])
        self.assertEqual(report["best_candidate"]["candidate_index"], 2)
        self.assertEqual(report["best_candidate"]["prompt"], "better prompt")
        self.assertEqual(report["best_candidate"]["score"]["range_violation_count"], 1)

    def test_sweep_uses_feedback_prompts_when_profile_requests_them(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "ranked_candidates": [
                            {
                                "score": {
                                    "violation_joint_counts": {"shoulder_lift": 6, "elbow_flex": 4},
                                    "violation_joint_excess_raw_ticks": {"shoulder_lift": 1000, "elbow_flex": 800},
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="lower_pregrasp",
                )

        self.assertGreaterEqual(len(seen_prompts), 4)
        self.assertIn("shoulder low", seen_prompts[0])
        self.assertEqual(report["prompt_profile"], "lower_pregrasp")

    def test_sweep_uses_projection_aware_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "projection_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "ranked_candidates": [
                            {
                                "projection": {
                                    "joint_distortion": {
                                        "shoulder_lift": {"violation_count": 10, "total_raw_distortion": 5000},
                                        "elbow_flex": {"violation_count": 10, "total_raw_distortion": 3000},
                                        "wrist_flex": {"violation_count": 10, "total_raw_distortion": 4000},
                                    }
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="projection_aware",
                )

        self.assertGreaterEqual(len(seen_prompts), 4)
        self.assertIn("smallest reachable side pre-grasp", seen_prompts[0])
        self.assertEqual(report["prompt_profile"], "projection_aware")

    def test_sweep_uses_residual_distortion_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "residual_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "ranked_candidates": [
                            {
                                "projection": {
                                    "joint_distortion": {
                                        "wrist_flex": {"violation_count": 10, "total_raw_distortion": 4700},
                                        "shoulder_lift": {"violation_count": 10, "total_raw_distortion": 3800},
                                        "elbow_flex": {"violation_count": 5, "total_raw_distortion": 1900},
                                    }
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="residual_distortion",
                )

        self.assertGreaterEqual(len(seen_prompts), 4)
        self.assertIn("wrist straight", seen_prompts[0])
        self.assertEqual(report["prompt_profile"], "residual_distortion")

    def test_sweep_uses_memory_refine_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "memory_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "next_agentic_layer_step": {
                            "type": "reuse_best_historical_prompt_family",
                            "selected_prompt": selected,
                        },
                        "best_candidate": {"prompt": "fallback prompt"},
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="memory_refine",
                )

        self.assertGreaterEqual(len(seen_prompts), 4)
        self.assertEqual(seen_prompts[0], selected)
        self.assertEqual(report["prompt_profile"], "memory_refine")

    def test_sweep_uses_memory_sample_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "memory_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "next_agentic_layer_step": {
                            "type": "continue_from_latest_best_prompt_family",
                            "selected_prompt": selected,
                        },
                        "best_candidate": {"prompt": "fallback prompt"},
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="memory_sample",
                )

        self.assertEqual(seen_prompts, [selected, selected, selected, selected, selected])
        self.assertEqual(report["prompt_profile"], "memory_sample")

    def test_sweep_uses_memory_residual_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "memory_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            selected = "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder."
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "next_agentic_layer_step": {
                            "type": "reuse_best_historical_prompt_family",
                            "selected_prompt": selected,
                        },
                        "ranked_candidates": [
                            {
                                "score": {
                                    "joint_violation_counts": {"elbow_flex": 3, "shoulder_lift": 2, "wrist_flex": 3},
                                    "joint_excess_raw_ticks": {
                                        "elbow_flex": 946.6906,
                                        "shoulder_lift": 848.8153,
                                        "wrist_flex": 612.3239,
                                    },
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="memory_residual",
                )

        self.assertGreaterEqual(len(seen_prompts), 4)
        self.assertEqual(seen_prompts[0], selected)
        self.assertIn("wrist neutral", seen_prompts[1])
        self.assertTrue(any("elbow" in prompt and "midrange" in prompt for prompt in seen_prompts))
        self.assertEqual(report["prompt_profile"], "memory_residual")

    def test_sweep_uses_memory_structured_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "memory_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            selected = (
                "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, "
                "extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral."
            )
            feedback.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "next_agentic_layer_step": {
                            "type": "reuse_best_historical_prompt_family",
                            "selected_prompt": selected,
                        },
                        "ranked_candidates": [
                            {
                                "score": {
                                    "joint_violation_counts": {"elbow_flex": 3, "shoulder_lift": 2, "wrist_flex": 3},
                                    "joint_excess_raw_ticks": {
                                        "elbow_flex": 946.6906,
                                        "shoulder_lift": 848.8153,
                                        "wrist_flex": 612.3239,
                                    },
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="memory_structured",
                )

        self.assertGreaterEqual(len(seen_prompts), 5)
        self.assertIn("Goal:", seen_prompts[1])
        self.assertIn("Constraints:", seen_prompts[1])
        self.assertEqual(report["prompt_profile"], "memory_structured")

    def test_transition_execution_feedback_blocks_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "transition_execution_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            feedback.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_transition_execution_feedback",
                        "status": "passed",
                        "prompt_mutation_allowed": False,
                        "failure_modes": [
                            "execution_packet_not_ready",
                            "observer_or_live_readback_preflight_incomplete",
                        ],
                        "next_agentic_layer_step": {
                            "type": "rerun_observer_return_refresh_live_readonly_when_camera_3_available",
                            "reason": "Execution was blocked by observer/live-readback preflight.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference") as dry,
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk") as execute,
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="memory_residual",
                )

        dry.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["best_candidate"], None)
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])
        self.assertFalse(report["feedback_gate"]["prompt_mutation_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "preserve_transition_candidate_until_observer_live_readback_gate",
        )

    def test_sweep_uses_policy_camera_feedback_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("{}", encoding="utf-8")
            config = tmp / "config.json"
            stats = tmp / "stats.json"
            calibration = tmp / "calibration.json"
            feedback = tmp / "policy_camera_feedback.json"
            config.write_text("{}", encoding="utf-8")
            stats.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            prompt = "Use the wide context camera to approach the green figure before local wrist alignment."
            feedback.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_policy_camera_pseudo_llm_feedback",
                        "status": "passed",
                        "pseudo_llm_feedback": {
                            "does_not_prompt_operator": True,
                            "next_smolvla_prompt": prompt,
                        },
                    }
                ),
                encoding="utf-8",
            )
            seen_prompts = []

            def fake_dry_with_prompt(**kwargs):
                seen_prompts.append(kwargs["instruction"])
                return _fake_dry(**kwargs)

            with (
                patch("scripts.real_so100_agentic_proposal_sweep.run_dry_inference", side_effect=fake_dry_with_prompt),
                patch("scripts.real_so100_agentic_proposal_sweep.execute_action_chunk", side_effect=_fake_execute),
            ):
                report = run_proposal_sweep(
                    episode=episode,
                    frame_index=0,
                    output_dir=tmp / "sweep",
                    prompts=[],
                    model_id="lerobot/smolvla_base",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    action_steps=10,
                    metadata_config=config,
                    action_stats=stats,
                    calibration=calibration,
                    port="/dev/cu.fake",
                    action_semantics="absolute_joint_position",
                    gripper_semantics="higher_raw_opens",
                    command_units="lerobot_so100_position",
                    feedback_report=feedback,
                    prompt_profile="policy_camera_feedback",
                )

        self.assertEqual(seen_prompts, [prompt])
        self.assertEqual(report["prompt_profile"], "policy_camera_feedback")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["task_success_claim_allowed"])


def _fake_dry(**kwargs):
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    action_path = output_dir / "smolvla_action_chunk.json"
    action_path.write_text(json.dumps({"raw_action_chunk": [[0, 0, 0, 0, 0, 0] for _ in range(10)]}), encoding="utf-8")
    return {
        "status": "passed",
        "action_path": str(action_path),
        "send_action_called": False,
        "policy_actions_executed": False,
    }


def _fake_execute(**kwargs):
    output = Path(kwargs["output"])
    prompt_is_better = "candidate_02" in str(output)
    target_raw = 2100 if prompt_is_better else 2600
    report = {
        "status": "dry_run",
        "send_action_called": False,
        "policy_actions_executed": False,
        "dry_plan": {
            "ready_for_execution": False,
            "blockers": ["range"],
            "step_plans": [
                {
                    "step_index": 0,
                    "joint_targets": [
                        {"joint": "shoulder_pan", "target_raw": target_raw, "range_min": 1000, "range_max": 2000},
                    ],
                }
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
