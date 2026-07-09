from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from physical_ai_agent.policies.so101_visual_servo_head import (
    SO101VisualServoHead,
    SO101VisualServoHeadConfig,
    extract_smolvla_camera_patch_features,
    visual_servo_loss,
)
from physical_ai_agent.so101_visual_servo import (
    VisualServoError,
    VisualServoGains,
    VisualServoStopThresholds,
    controller_error_for_camera,
    select_visual_servo_camera,
    should_stop_visual_servo,
    visual_servo_delta_q,
    visual_servo_delta_q_for_camera,
)
from scripts.build_so101_visual_servo_labels import extract_egocentric_visual_servo_label, extract_wrist_visual_servo_label
from scripts.build_so101_visual_servo_labels import build_visual_servo_labels


class SO101VisualServoTest(unittest.TestCase):
    def test_visual_servo_head_bounds_image_error_outputs(self) -> None:
        head = SO101VisualServoHead(SO101VisualServoHeadConfig(context_dim=4))
        pred = head(torch.full((2, 4), 1000.0))

        self.assertLessEqual(float(pred["camera1"].detach().abs().max()), 1.0)
        self.assertLessEqual(float(pred["camera2"].detach().abs().max()), 1.0)

    def test_extracts_wrist_center_error_from_green_cube_mask(self) -> None:
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        for x in range(70, 90):
            for y in range(20, 40):
                image.putpixel((x, y), (0, 220, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        label = extract_wrist_visual_servo_label({"bytes": buffer.getvalue()}, min_area=20)

        self.assertTrue(label["visible"])
        self.assertGreater(label["wrist_dx_norm"], 0.4)
        self.assertLess(label["wrist_dy_norm"], -0.2)
        self.assertEqual(label["target_area"], 400)

    def test_extracts_egocentric_error_from_gripper_proxy_to_green_cube(self) -> None:
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        for x in range(20, 30):
            for y in range(50, 60):
                image.putpixel((x, y), (220, 180, 20))
        for x in range(70, 80):
            for y in range(20, 30):
                image.putpixel((x, y), (0, 220, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        label = extract_egocentric_visual_servo_label({"bytes": buffer.getvalue()}, min_area=20)

        self.assertTrue(label["visible"])
        self.assertGreater(label["dx_norm"], 0.8)
        self.assertLess(label["dy_norm"], -0.4)
        self.assertIn("gripper_proxy_x", label)

    def test_controller_maps_image_error_to_tunable_delta_q(self) -> None:
        delta = visual_servo_delta_q(
            VisualServoError(wrist_dx_norm=0.5, wrist_dy_norm=-0.5, edge_angle_error=0.25),
            VisualServoGains(pan=0.1, lift=0.2, flex=0.3, wrist_roll=0.4, max_abs_delta=0.2),
        )

        np.testing.assert_allclose(delta, [0.05, -0.1, 0.0, 0.15, 0.1, 0.0], atol=1e-6)

    def test_policy_camera_controller_flips_vertical_error(self) -> None:
        error = VisualServoError(wrist_dx_norm=0.2, wrist_dy_norm=0.4, edge_angle_error=0.1)

        corrected = controller_error_for_camera(error, "camera2")

        self.assertEqual(corrected.wrist_dx_norm, 0.2)
        self.assertEqual(corrected.wrist_dy_norm, -0.4)
        self.assertEqual(controller_error_for_camera(error, "camera1").wrist_dy_norm, -0.4)
        self.assertEqual(controller_error_for_camera(error, "camera1").edge_angle_error, 0.0)
        self.assertEqual(controller_error_for_camera(error, "camera2").edge_angle_error, 0.1)

    def test_linear_controller_uses_camera_specific_teacher_fit(self) -> None:
        error = VisualServoError(wrist_dx_norm=0.4, wrist_dy_norm=-0.2, edge_angle_error=0.3)

        camera1 = visual_servo_delta_q_for_camera(error, "camera1")
        camera2 = visual_servo_delta_q_for_camera(error, "camera2")

        self.assertNotEqual(camera1, camera2)
        self.assertNotEqual(camera1[2], 0.0)
        self.assertNotEqual(camera2[2], 0.0)
        self.assertLessEqual(max(abs(value) for value in camera1), 0.08)
        self.assertEqual(camera1[5], 0.0)
        self.assertEqual(camera2[5], 0.0)

    def test_egocentric_pan_follows_image_x_sign(self) -> None:
        left = visual_servo_delta_q_for_camera(VisualServoError(-0.4, -0.8, 0.1), "camera1")
        right = visual_servo_delta_q_for_camera(VisualServoError(0.4, -0.8, 0.1), "camera1")

        self.assertLess(left[0], 0.0)
        self.assertGreater(right[0], 0.0)

    def test_linear_controller_can_condition_on_qpos(self) -> None:
        error = VisualServoError(0.1, -0.3, 0.2)

        without_state = visual_servo_delta_q_for_camera(error, "camera2")
        with_state = visual_servo_delta_q_for_camera(error, "camera2", qpos=[0.1, -0.2, 0.3, 0.4, -0.5, 0.6])

        self.assertNotEqual(without_state, with_state)

    def test_visual_servo_camera_selection_uses_egocentric_before_wrist_handoff(self) -> None:
        selection = select_visual_servo_camera(
            camera1_error=(0.8, 0.1, 0.0),
            camera2_error=(0.1, 0.1, 0.0),
            camera1_visible_prob=0.9,
            camera2_visible_prob=0.95,
        )

        self.assertEqual(selection["servo_camera"], "camera1")
        self.assertEqual(selection["reason"], "egocentric_approach")
        self.assertFalse(selection["camera1_near"])

    def test_visual_servo_camera_selection_hands_off_to_wrist_when_near(self) -> None:
        selection = select_visual_servo_camera(
            camera1_error=(0.08, -0.06, 0.0),
            camera2_error=(0.3, -0.2, 0.0),
            camera1_visible_prob=0.9,
            camera2_visible_prob=0.95,
        )

        self.assertEqual(selection["servo_camera"], "camera2")
        self.assertEqual(selection["reason"], "wrist_handoff")
        self.assertTrue(selection["camera1_near"])

    def test_stop_requires_probability_and_small_image_error(self) -> None:
        self.assertTrue(
            should_stop_visual_servo(
                VisualServoError(0.01, -0.02, 0.03, stop_prob=0.9),
                VisualServoStopThresholds(stop_prob=0.7, dx=0.1, dy=0.1, angle=0.1),
            )
        )
        self.assertFalse(
            should_stop_visual_servo(
                VisualServoError(0.3, -0.02, 0.03, stop_prob=0.9),
                VisualServoStopThresholds(stop_prob=0.7, dx=0.1, dy=0.1, angle=0.1),
            )
        )

    def test_builds_camera1_and_camera2_labels_in_one_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/chunk-000").mkdir(parents=True)
            row = {
                "episode_index": 0,
                "frame_index": 0,
                "index": 0,
                "observation.images.camera1": {"bytes": _green_square_png(10, 10)},
                "observation.images.camera2": {"bytes": _green_square_png(70, 20)},
                "action": np.zeros(6, dtype=np.float32),
            }
            pd.DataFrame([row]).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)

            report = build_visual_servo_labels(
                dataset_root=root,
                camera_keys=("observation.images.camera1", "observation.images.camera2"),
                action_key="action",
                min_area=20,
                stop_action_norm=1e-4,
            )
            table = pd.read_parquet(report["parquet_path"])

            self.assertIn("camera1_dx_norm", table.columns)
            self.assertIn("camera2_dx_norm", table.columns)
            self.assertLess(float(table["camera1_dx_norm"].iloc[0]), 0.0)
            self.assertGreater(float(table["camera2_dx_norm"].iloc[0]), 0.0)

    def test_visual_servo_head_loss_uses_camera_labels_and_stop_label(self) -> None:
        batch = {
            "observation.images.camera1": torch.zeros((2, 3, 256, 256)),
            "observation.images.camera2": torch.zeros((2, 3, 256, 256)),
            "observation.state": torch.zeros((2, 6)),
            "observation.language.tokens": torch.zeros((2, 4), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((2, 4), dtype=torch.bool),
            "visual_servo.camera1": torch.zeros((2, 3)),
            "visual_servo.camera1_visible": torch.tensor([True, False]),
            "visual_servo.camera2": torch.zeros((2, 3)),
            "visual_servo.camera2_visible": torch.tensor([True, True]),
            "visual_servo.stop_label": torch.tensor([0.0, 1.0]),
        }

        loss, metrics = visual_servo_loss(
            SO101VisualServoHead(SO101VisualServoHeadConfig(context_dim=4)),
            batch,
            weight=0.1,
            policy=_FakeSmolVLAPolicy(),
        )

        self.assertIsNotNone(loss)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIn("visual_servo_mse", metrics)
        self.assertIn("visual_servo_rmse", metrics)
        self.assertIn("visual_servo_camera1_rmse", metrics)
        self.assertIn("visual_servo_camera1_dx_mae", metrics)
        self.assertIn("visual_servo_camera1_dy_mae", metrics)
        self.assertIn("visual_servo_stop_accuracy", metrics)
        self.assertIn("visual_servo_camera1_visible_accuracy", metrics)

    def test_visual_servo_head_accepts_observation_history_axis(self) -> None:
        batch = {
            "observation.images.camera1": torch.zeros((2, 2, 3, 256, 256)),
            "observation.images.camera2": torch.zeros((2, 2, 3, 256, 256)),
            "observation.state": torch.zeros((2, 2, 6)),
            "observation.language.tokens": torch.zeros((2, 4), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((2, 4), dtype=torch.bool),
            "visual_servo.camera1": torch.zeros((2, 3)),
            "visual_servo.camera1_visible": torch.ones((2,), dtype=torch.bool),
            "visual_servo.camera2": torch.zeros((2, 3)),
            "visual_servo.camera2_visible": torch.ones((2,), dtype=torch.bool),
            "visual_servo.stop_label": torch.zeros((2,)),
        }

        loss, metrics = visual_servo_loss(
            SO101VisualServoHead(SO101VisualServoHeadConfig(context_dim=4)),
            batch,
            weight=0.1,
            policy=_FakeSmolVLAPolicy(),
        )

        self.assertIsNotNone(loss)
        self.assertIn("visual_servo_loss", metrics)

    def test_visual_servo_head_predicts_visibility(self) -> None:
        pred = SO101VisualServoHead(SO101VisualServoHeadConfig(context_dim=4))(torch.full((2, 4), 1000.0))

        self.assertEqual(tuple(pred["camera1"].shape), (2, 3))
        self.assertEqual(tuple(pred["camera1_visible_logit"].shape), (2,))
        self.assertEqual(tuple(pred["camera2_visible_logit"].shape), (2,))
        self.assertEqual(float(pred["camera1"].detach().abs().max()), 0.0)
        self.assertEqual(float(pred["camera2"].detach().abs().max()), 0.0)

    def test_visual_servo_features_include_instruction_context(self) -> None:
        batch = {
            "observation.images.camera1": torch.zeros((2, 3, 256, 256)),
            "observation.images.camera2": torch.zeros((2, 3, 256, 256)),
            "observation.state": torch.zeros((2, 6)),
            "observation.language.tokens": torch.zeros((2, 4), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((2, 4), dtype=torch.bool),
        }

        features = extract_smolvla_camera_patch_features(_FakeSmolVLAPolicy(), batch)

        self.assertEqual(set(features), {"camera1", "camera2", "context"})
        self.assertEqual(tuple(features["context"].shape), (2, 4))


def _green_square_png(x0: int, y0: int) -> bytes:
    image = Image.new("RGB", (100, 100), (128, 128, 128))
    for x in range(x0, x0 + 20):
        for y in range(y0, y0 + 20):
            image.putpixel((x, y), (0, 220, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeSmolVLAPolicy:
    name = "smolvla"

    def __init__(self) -> None:
        self.model = self
        self.vlm_with_expert = self

    def prepare_images(self, batch):
        batch_size = batch["observation.state"].shape[0]
        image = torch.zeros((batch_size, 3, 4, 4))
        return [image, image], [torch.ones((batch_size,), dtype=torch.bool), torch.ones((batch_size,), dtype=torch.bool)]

    def prepare_state(self, batch):
        return batch["observation.state"].float()

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        del images, img_masks, lang_tokens, lang_masks
        batch_size = state.shape[0]
        return torch.ones((batch_size, 3, 4)), torch.ones((batch_size, 3), dtype=torch.bool), torch.ones((batch_size, 3))

    def embed_image(self, image):
        batch_size = image.shape[0]
        return torch.ones((batch_size, 4, 4), device=image.device, dtype=image.dtype)


if __name__ == "__main__":
    unittest.main()
