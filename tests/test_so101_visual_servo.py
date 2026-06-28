from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from physical_ai_agent.policies.so101_visual_servo_head import SO101VisualServoHead, visual_servo_loss
from physical_ai_agent.so101_visual_servo import (
    VisualServoError,
    VisualServoGains,
    VisualServoStopThresholds,
    should_stop_visual_servo,
    visual_servo_delta_q,
)
from scripts.build_so101_visual_servo_labels import extract_wrist_visual_servo_label
from scripts.build_so101_visual_servo_labels import build_visual_servo_labels


class SO101VisualServoTest(unittest.TestCase):
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

    def test_controller_maps_image_error_to_tunable_delta_q(self) -> None:
        delta = visual_servo_delta_q(
            VisualServoError(wrist_dx_norm=0.5, wrist_dy_norm=-0.5, edge_angle_error=0.25),
            VisualServoGains(pan=0.1, lift=0.2, flex=0.3, wrist_roll=0.4, max_abs_delta=0.2),
        )

        np.testing.assert_allclose(delta, [0.05, -0.1, 0.0, 0.15, 0.1, 0.0], atol=1e-6)

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
            "visual_servo.camera1": torch.zeros((2, 3)),
            "visual_servo.camera1_visible": torch.tensor([True, False]),
            "visual_servo.camera2": torch.zeros((2, 3)),
            "visual_servo.camera2_visible": torch.tensor([True, True]),
            "visual_servo.stop_label": torch.tensor([0.0, 1.0]),
        }

        loss, metrics = visual_servo_loss(SO101VisualServoHead(), batch, weight=0.1)

        self.assertIsNotNone(loss)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIn("visual_servo_stop_accuracy", metrics)

    def test_visual_servo_head_accepts_observation_history_axis(self) -> None:
        batch = {
            "observation.images.camera1": torch.zeros((2, 2, 3, 256, 256)),
            "observation.images.camera2": torch.zeros((2, 2, 3, 256, 256)),
            "observation.state": torch.zeros((2, 2, 6)),
            "visual_servo.camera1": torch.zeros((2, 3)),
            "visual_servo.camera1_visible": torch.ones((2,), dtype=torch.bool),
            "visual_servo.camera2": torch.zeros((2, 3)),
            "visual_servo.camera2_visible": torch.ones((2,), dtype=torch.bool),
            "visual_servo.stop_label": torch.zeros((2,)),
        }

        loss, metrics = visual_servo_loss(SO101VisualServoHead(), batch, weight=0.1)

        self.assertIsNotNone(loss)
        self.assertIn("visual_servo_loss", metrics)


def _green_square_png(x0: int, y0: int) -> bytes:
    image = Image.new("RGB", (100, 100), (128, 128, 128))
    for x in range(x0, x0 + 20):
        for y in range(y0, y0 + 20):
            image.putpixel((x, y), (0, 220, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
