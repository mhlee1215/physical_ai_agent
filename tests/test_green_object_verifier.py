import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.perception.green_object_verifier import (
    image_paths_from_episode_record,
    verify_green_object_images,
)


class GreenObjectVerifierTest(TestCase):
    def test_detects_green_region_and_reports_primary_camera(self) -> None:
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "green.png"
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            image[20:60, 35:95] = (0, 220, 0)
            cv2.imwrite(str(image_path), image)

            result = verify_green_object_images({"2": image_path}, min_area_px=500)

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.object_visible)
        self.assertEqual(result.primary_camera, "2")
        self.assertEqual(result.visible_cameras, ["2"])
        self.assertEqual(result.detections[0].bbox_xyxy, [35, 20, 95, 60])

    def test_blocks_when_green_region_is_too_small(self) -> None:
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "small.png"
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            image[5:10, 5:10] = (0, 220, 0)
            cv2.imwrite(str(image_path), image)

            result = verify_green_object_images({"0": image_path}, min_area_px=500)

        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.object_visible)
        self.assertEqual(result.visible_cameras, [])

    def test_extracts_image_paths_from_episode_record(self) -> None:
        record = json.loads(
            """
            {
              "observation": {
                "images": {
                  "0": "/tmp/camera0.jpg",
                  "1": "",
                  "2": "/tmp/camera2.jpg"
                }
              }
            }
            """
        )

        self.assertEqual(
            image_paths_from_episode_record(record),
            {"0": Path("/tmp/camera0.jpg"), "2": Path("/tmp/camera2.jpg")},
        )
