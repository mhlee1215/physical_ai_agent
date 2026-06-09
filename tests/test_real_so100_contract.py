from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.real_so100.contract import (
    CameraRoute,
    build_no_actuation_gate,
    classify_sim_camera_names,
    current_so100_contract,
    sim_policy_feature_keys,
    validate_policy_camera_routes,
    write_contract_artifact,
)


class RealSO100ContractTest(TestCase):
    def test_current_contract_keeps_camera_3_out_of_policy_inputs(self) -> None:
        contract = current_so100_contract()

        self.assertEqual(contract.policy_camera_indexes, (0, 1))
        self.assertEqual(contract.observer_camera_routes[0].index, 3)
        self.assertEqual(contract.default_action_chunk_steps, 10)
        self.assertEqual(
            contract.joint_order,
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
        )
        self.assertEqual(validate_policy_camera_routes(contract.policy_camera_routes), [])
        self.assertFalse(contract.observer_camera_routes[0].policy_input)

    def test_validation_rejects_observer_and_legacy_policy_cameras(self) -> None:
        invalid = [
            CameraRoute(
                index=3,
                name="camera_3",
                role="bad_policy_observer",
                policy_input=True,
                observer_input=True,
                feature_key="observation.images.camera_3",
                notes="invalid",
            ),
            CameraRoute(
                index=2,
                name="camera_2",
                role="legacy",
                policy_input=True,
                observer_input=False,
                feature_key="observation.images.camera_2",
                notes="invalid",
            ),
        ]

        errors = validate_policy_camera_routes(invalid)

        self.assertTrue(any("camera_3" in error for error in errors))
        self.assertTrue(any("camera index 3" in error for error in errors))
        self.assertTrue(any("legacy camera index 2" in error for error in errors))

    def test_sim_camera_roles_match_latest_policy_debug_split(self) -> None:
        policy, debug = classify_sim_camera_names(["top_down", "wrist_cam", "egocentric_cam"])

        self.assertEqual(policy, ["wrist_cam", "egocentric_cam"])
        self.assertEqual(debug, ["top_down"])
        self.assertEqual(
            sim_policy_feature_keys(),
            ["observation.images.wrist_cam", "observation.images.egocentric_cam"],
        )

    def test_no_actuation_gate_records_required_blockers(self) -> None:
        gate = build_no_actuation_gate("observer camera 3 unavailable")

        self.assertFalse(gate.physical_execution_allowed)
        self.assertFalse(gate.send_action_called)
        self.assertIn("observer_camera_3_evidence_available", gate.required_before_motion)
        self.assertIn("smolvla_postprocessor_or_unnormalization_verified", gate.required_before_motion)

    def test_contract_artifact_is_json_serializable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "so100_contract.json"

            write_contract_artifact(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["robot"], "SO-100 follower")
            self.assertEqual(payload["policy_camera_routes"][0]["index"], 0)
            self.assertEqual(payload["observer_camera_routes"][0]["index"], 3)
