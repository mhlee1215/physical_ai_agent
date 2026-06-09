from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_policy_camera_task_state_packet import build_policy_camera_task_state_packet


class RealSO100PolicyCameraTaskStatePacketTest(TestCase):
    def test_builds_structured_llm_vlm_packet_from_policy_cameras(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cam0 = tmp / "camera_0.jpg"
            cam1 = tmp / "camera_1.jpg"
            _write_scene(cam0, object_visible=True, jaw_visible=True, edge_clipped=False)
            _write_scene(cam1, object_visible=True, jaw_visible=False, edge_clipped=False)
            manifest = _write_manifest(tmp, cam0=cam0, cam1=cam1)
            lane = _write_lane(tmp)

            report = build_policy_camera_task_state_packet(
                development_lane=lane,
                observation_manifest=manifest,
                output=tmp / "packet.json",
                min_area_px=100,
                min_jaw_marker_area_px=100,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["camera_contract"]["policy_camera_indexes"], [0, 1])
        self.assertFalse(report["camera_contract"]["camera_3_is_policy_input"])
        self.assertTrue(report["object_state"]["visible"])
        self.assertTrue(report["object_state"]["usable_for_pregrasp"])
        self.assertEqual(report["jaw_context"]["status"], "ready")
        self.assertEqual(report["task_goal"]["target_relation"]["direction"], "right")
        self.assertEqual(report["task_goal"]["relation_frame"], "object_or_observer_image_frame")
        self.assertEqual(report["llm_vlm_input_packet"]["consumer"], "in_loop_agent_or_smolvla_prompt_builder")
        self.assertTrue(report["llm_vlm_input_packet"]["does_not_prompt_operator"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "ask_llm_vlm_for_no_actuation_best_prompt_evaluation_packet",
        )

    def test_visible_but_edge_clipped_object_requests_reframe_reasoning(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cam0 = tmp / "camera_0.jpg"
            cam1 = tmp / "camera_1.jpg"
            _write_scene(cam0, object_visible=True, jaw_visible=True, edge_clipped=True)
            _write_scene(cam1, object_visible=True, jaw_visible=False, edge_clipped=True)
            manifest = _write_manifest(tmp, cam0=cam0, cam1=cam1)
            lane = _write_lane(tmp)

            report = build_policy_camera_task_state_packet(
                development_lane=lane,
                observation_manifest=manifest,
                output=tmp / "packet.json",
                min_area_px=100,
                min_jaw_marker_area_px=100,
                edge_margin_px=8,
            )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["object_state"]["visible"])
        self.assertFalse(report["object_state"]["usable_for_pregrasp"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "ask_llm_vlm_for_no_actuation_reframe_or_approach_prompt_packet",
        )

    def test_blocks_when_source_lane_is_not_passed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cam0 = tmp / "camera_0.jpg"
            cam1 = tmp / "camera_1.jpg"
            _write_scene(cam0, object_visible=True, jaw_visible=True, edge_clipped=False)
            _write_scene(cam1, object_visible=True, jaw_visible=False, edge_clipped=False)
            manifest = _write_manifest(tmp, cam0=cam0, cam1=cam1)
            lane = _write_lane(tmp, status="blocked")

            report = build_policy_camera_task_state_packet(
                development_lane=lane,
                observation_manifest=manifest,
                output=tmp / "packet.json",
                min_area_px=100,
                min_jaw_marker_area_px=100,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["blockers"])
        self.assertFalse(report["physical_robot_motion"])


def _write_scene(path: Path, *, object_visible: bool, jaw_visible: bool, edge_clipped: bool) -> None:
    import cv2
    import numpy as np

    image = np.full((160, 220, 3), 235, dtype=np.uint8)
    if object_visible:
        if edge_clipped:
            cv2.rectangle(image, (-10, 35), (40, 115), (0, 190, 55), thickness=-1)
        else:
            cv2.rectangle(image, (40, 35), (95, 115), (0, 190, 55), thickness=-1)
    if jaw_visible:
        cv2.rectangle(image, (130, 90), (180, 135), (140, 60, 20), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_manifest(root: Path, *, cam0: Path, cam1: Path) -> Path:
    episode = root / "episode.jsonl"
    episode.write_text(
        json.dumps(
            {
                "frame_index": 4,
                "task": "Pick up the green Android figure and move it to the right.",
                "observation": {
                    "camera_roles": {"0": "wrist_cam", "1": "egocentric_wide_context"},
                    "images": {"0": str(cam0), "1": str(cam1)},
                    "state": {"shoulder_pan": 2000, "gripper": 1800},
                    "state_source": "test_readback",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ok": True,
                "episode_jsonl": str(episode),
                "task": "Pick up the green Android figure and move it to the right.",
                "camera_roles": {"0": "wrist_cam", "1": "egocentric_wide_context"},
                "send_action_called": False,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_lane(root: Path, *, status: str = "passed") -> Path:
    path = root / "lane.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_development_lane",
                "status": status,
                "camera_contract": {
                    "policy_camera_roles": {"0": "wrist_cam", "1": "egocentric_wide_context"},
                },
                "physical_robot_motion": False,
            }
        ),
        encoding="utf-8",
    )
    return path
