from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_grasp_outcome import assess_grasp_outcome


class RealSO100GraspOutcomeTest(TestCase):
    def test_stationary_green_object_is_failed_grasp(self) -> None:
        with TemporaryDirectory() as tmpdir:
            before = Path(tmpdir) / "before.jpg"
            after = Path(tmpdir) / "after.jpg"
            _write_scene(before, object_shift_x=0, gripper_shift_x=0)
            _write_scene(after, object_shift_x=0, gripper_shift_x=-20)

            result = assess_grasp_outcome(before_image=before, after_image=after)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["grasp_outcome"], "grasp_failed_object_stationary")
        self.assertTrue(result["object_stationary"])

    def test_moved_green_object_is_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            before = Path(tmpdir) / "before.jpg"
            after = Path(tmpdir) / "after.jpg"
            _write_scene(before, object_shift_x=0, gripper_shift_x=0)
            _write_scene(after, object_shift_x=30, gripper_shift_x=-20)

            result = assess_grasp_outcome(before_image=before, after_image=after)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["grasp_outcome"], "object_moved_or_occluded_candidate")
        self.assertFalse(result["object_stationary"])


def _write_scene(path: Path, *, object_shift_x: int, gripper_shift_x: int) -> None:
    import cv2
    import numpy as np

    image = np.full((240, 320, 3), 235, dtype=np.uint8)
    x1 = 80 + object_shift_x
    x2 = 150 + object_shift_x
    cv2.rectangle(image, (x1, 120), (x2, 210), (0, 190, 55), thickness=-1)
    cv2.rectangle(image, (205 + gripper_shift_x, 135), (230 + gripper_shift_x, 220), (240, 240, 240), thickness=-1)
    cv2.imwrite(str(path), image)
