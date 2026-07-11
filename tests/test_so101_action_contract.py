from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physical_ai_agent.agent_core import qwen_so101_closed_loop as loop
from physical_ai_agent.policies.lerobot_policy_runner import _visual_servo_head_path


class _FakeConfig:
    robot_state_feature = None
    image_features = {}


class _FakePolicy:
    config = _FakeConfig()

    def select_action(self, _batch):
        return [-2.0, 0.5, 3.0]


class _FakeRunner:
    processor_source = "saved_preprocessor_and_postprocessor"
    preprocessor = type("Pre", (), {"steps": []})()
    postprocessor = type("Post", (), {"steps": []})()

    def __init__(self):
        self.policy = _FakePolicy()

    def select_action_with_trace(self, _observation):
        return {
            "action": [-2.0, 0.5, 3.0],
            "raw_action": [-10.0, 0.0, 10.0],
            "postprocessed_action": [-2.0, 0.5, 3.0],
            "processor_source": self.processor_source,
            "preprocessor_steps": ["NormalizerProcessorStep"],
            "postprocessor_steps": ["UnnormalizerProcessorStep"],
        }

    def predict_visual_servo_with_trace(self, _observation):
        return {
            "camera1": {"dx_norm": -0.2, "dy_norm": 0.1, "edge_angle_error": 0.0, "visible": True},
            "camera2": {"dx_norm": 0.5, "dy_norm": -0.5, "edge_angle_error": 0.25, "visible": True},
            "stop_prob": 0.1,
            "delta_q": [0.05, -0.03, 0.0],
            "processor_source": self.processor_source,
        }


class SO101ActionContractTest(unittest.TestCase):
    def test_visual_servo_head_can_live_next_to_pretrained_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "000100"
            policy_path = checkpoint_dir / "pretrained_model"
            policy_path.mkdir(parents=True)
            head_path = checkpoint_dir / "visual_servo_head.pt"
            head_path.write_bytes(b"head")

            self.assertEqual(_visual_servo_head_path(policy_path), head_path)

    def test_processor_mode_uses_saved_processor_trace(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[0.0, 0.0, 0.0],
            action_base_qpos=None,
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="processor",
            dataset_action_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        )
        self.assertEqual(result["action"], [-2.0, 0.5, 3.0])
        self.assertEqual(result["processor_raw_action"], [-10.0, 0.0, 10.0])
        self.assertEqual(result["processor_source"], "saved_preprocessor_and_postprocessor")
        self.assertEqual(result["postprocessor_steps"], ["UnnormalizerProcessorStep"])

    def test_processor_dataset_clamp_clamps_only_env_action(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[0.0, 0.0, 0.0],
            action_base_qpos=None,
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="processor_dataset_clamp",
            dataset_action_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        )
        self.assertEqual(result["action"], [-1.0, 0.5, 1.0])
        self.assertEqual(result["unclamped_env_action"], [-2.0, 0.5, 3.0])
        self.assertEqual(result["processor_postprocessed_action"], [-2.0, 0.5, 3.0])

    def test_processor_gripper_snap_snaps_only_gripper_axis(self) -> None:
        class _SixDimRunner(_FakeRunner):
            def select_action_with_trace(self, _observation):
                return {
                    "action": [0.1, 0.2, 0.3, 0.4, 0.5, 2.1],
                    "raw_action": [0.1, 0.2, 0.3, 0.4, 0.5, 2.1],
                    "postprocessed_action": [0.1, 0.2, 0.3, 0.4, 0.5, 2.1],
                    "processor_source": self.processor_source,
                    "preprocessor_steps": ["NormalizerProcessorStep"],
                    "postprocessor_steps": ["UnnormalizerProcessorStep"],
                }

        result = loop._select_env_action_with_trace(
            policy_executor=_SixDimRunner(),
            policy=_FakePolicy(),
            obs=[0.0] * 6,
            action_base_qpos=None,
            camera_pixels={},
            instruction="move",
            action_dim=6,
            action_contract_mode="processor_gripper_snap",
            dataset_action_bounds={
                "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -0.17453],
                "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.74533],
            },
        )

        self.assertEqual(result["action"], [0.1, 0.2, 0.3, 0.4, 0.5, 1.74533])
        self.assertEqual(result["unclamped_env_action"], [0.1, 0.2, 0.3, 0.4, 0.5, 2.1])
        self.assertEqual(result["processor_postprocessed_action"], [0.1, 0.2, 0.3, 0.4, 0.5, 2.1])

    def test_processor_delta_q_adds_model_delta_to_observation_state(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0, 9.0, 9.0],
            action_base_qpos=[0.25, -0.5, 1.0],
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="processor_delta_q",
            dataset_action_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        )
        self.assertEqual(result["delta_q_action"], [-2.0, 0.5, 3.0])
        self.assertEqual(result["action"], [-1.75, 0.0, 4.0])
        self.assertEqual(result["unclamped_env_action"], [-2.0, 0.5, 3.0])

    def test_visual_servo_delta_q_uses_head_prediction_instead_of_action_chunk(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0, 9.0, 9.0],
            action_base_qpos=[0.25, -0.5, 1.0],
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="visual_servo_delta_q",
            dataset_action_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        )
        self.assertEqual(result["delta_q_action"], [0.05, -0.03, 0.0])
        self.assertEqual(result["action"], [0.3, -0.53, 1.0])
        self.assertEqual(result["visual_servo_prediction"]["camera2"]["dx_norm"], 0.5)

    def test_visual_servo_delta_q_preserves_gripper_joint(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0] * 6,
            action_base_qpos=[0.25, -0.5, 1.0, 0.2, -0.1, -0.4],
            camera_pixels={},
            instruction="move",
            action_dim=6,
            action_contract_mode="visual_servo_delta_q",
            dataset_action_bounds={},
        )

        self.assertEqual(result["delta_q_action"][5], 0.0)
        self.assertEqual(result["action"][5], -0.4)

    def test_gt_teacher_replay_delta_q_uses_teacher_delta_label(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0, 9.0, 9.0],
            action_base_qpos=[0.25, -0.5, 1.0],
            gt_teacher_delta_q=[0.05, -0.03, 0.0],
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="gt_teacher_replay_delta_q",
            dataset_action_bounds={},
        )

        self.assertEqual(result["delta_q_action"], [0.05, -0.03, 0.0])
        self.assertEqual(result["action"], [0.3, -0.53, 1.0])
        self.assertEqual(result["processor_source"], "ground_truth_teacher_delta_q")

    def test_gt_staged_waypoint_target_uses_absolute_waypoint(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0, 9.0, 9.0],
            action_base_qpos=[0.25, -0.5, 1.0],
            gt_waypoint_qpos=[0.1, 0.2, 0.3],
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="gt_staged_waypoint_target",
            dataset_action_bounds={},
        )

        self.assertEqual(result["action"], [0.1, 0.2, 0.3])
        self.assertEqual(result["processor_source"], "ground_truth_staged_waypoint")

    def test_gt_label_nn_staged_waypoint_target_uses_absolute_waypoint(self) -> None:
        result = loop._select_env_action_with_trace(
            policy_executor=_FakeRunner(),
            policy=_FakePolicy(),
            obs=[9.0, 9.0, 9.0],
            action_base_qpos=[0.25, -0.5, 1.0],
            gt_waypoint_qpos=[0.4, 0.5, 0.6],
            camera_pixels={},
            instruction="move",
            action_dim=3,
            action_contract_mode="gt_label_nn_staged_waypoint_target",
            dataset_action_bounds={},
        )

        self.assertEqual(result["action"], [0.4, 0.5, 0.6])
        self.assertEqual(result["processor_source"], "ground_truth_staged_waypoint")

    def test_visual_servo_stop_uses_selected_servo_camera(self) -> None:
        self.assertTrue(
            loop._visual_servo_should_stop(
                {
                    "visual_servo_prediction": {
                        "servo_camera": "camera1",
                        "camera1": {"dx_norm": 0.01, "dy_norm": -0.01, "edge_angle_error": 0.01},
                        "camera2": {"dx_norm": 0.9, "dy_norm": 0.9, "edge_angle_error": 0.9},
                        "stop_prob": 0.9,
                    }
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
