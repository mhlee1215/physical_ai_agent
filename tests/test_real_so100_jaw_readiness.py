from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_jaw_readiness import assess_jaw_readiness


class RealSO100JawReadinessTest(TestCase):
    def test_blocks_edge_clipped_object_even_when_jaw_marker_visible(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "edge.jpg"
            _write_scene(image, edge_clipped=True, include_jaw_marker=True)

            result = assess_jaw_readiness(image_path=image, min_object_area_px=100, min_jaw_marker_area_px=100)

        self.assertEqual(result["status"], "blocked")
        self.assertIn("green object touches image boundary", result["blockers"])
        self.assertIsNotNone(result["jaw_marker_candidate"])

    def test_ready_when_object_and_jaw_marker_are_visible(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "ready.jpg"
            _write_scene(image, edge_clipped=False, include_jaw_marker=True)

            result = assess_jaw_readiness(image_path=image, min_object_area_px=100, min_jaw_marker_area_px=100)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["blockers"], [])
        self.assertFalse(result["object_edge_clipped"])


def _write_scene(path: Path, *, edge_clipped: bool, include_jaw_marker: bool) -> None:
    import cv2
    import numpy as np

    image = np.full((160, 220, 3), 235, dtype=np.uint8)
    if edge_clipped:
        cv2.rectangle(image, (-10, 40), (35, 115), (0, 190, 55), thickness=-1)
    else:
        cv2.rectangle(image, (40, 40), (95, 115), (0, 190, 55), thickness=-1)
    if include_jaw_marker:
        cv2.rectangle(image, (130, 90), (180, 135), (140, 60, 20), thickness=-1)
    cv2.imwrite(str(path), image)
