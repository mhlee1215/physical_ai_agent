from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_object_relocation import assess_object_relocation


class RealSO100ObjectRelocationTest(TestCase):
    def test_passes_when_green_object_moves_right_enough(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            before = tmp / "before.jpg"
            after = tmp / "after.jpg"
            _write_green_object(before, center_x=80)
            _write_green_object(after, center_x=150)

            result = assess_object_relocation(
                before_image=before,
                after_image=after,
                target_direction="right",
                min_delta_px=40,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["relocation_outcome"], "object_moved_right")
        self.assertTrue(result["task_success_candidate"])
        self.assertGreaterEqual(result["signed_goal_delta_px"], 40)

    def test_rejects_wrong_direction_motion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            before = tmp / "before.jpg"
            after = tmp / "after.jpg"
            _write_green_object(before, center_x=150)
            _write_green_object(after, center_x=80)

            result = assess_object_relocation(
                before_image=before,
                after_image=after,
                target_direction="right",
                min_delta_px=40,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["relocation_outcome"], "object_moved_wrong_direction")
        self.assertFalse(result["task_success_candidate"])
        self.assertLess(result["signed_goal_delta_px"], 0)

    def test_blocks_when_object_is_not_visible(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            before = tmp / "before.jpg"
            after = tmp / "after.jpg"
            _write_blank(before)
            _write_green_object(after, center_x=120)

            result = assess_object_relocation(before_image=before, after_image=after)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["relocation_outcome"], "object_not_visible")
        self.assertFalse(result["task_success_candidate"])


def _write_green_object(path: Path, *, center_x: int) -> None:
    import cv2
    import numpy as np

    image = np.full((180, 260, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (center_x - 25, 70), (center_x + 25, 130), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_blank(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((180, 260, 3), 235, dtype=np.uint8)
    cv2.imwrite(str(path), image)
