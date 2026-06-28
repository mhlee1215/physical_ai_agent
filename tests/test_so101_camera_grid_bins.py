from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.build_so101_camera_grid_bins import build_bins


class SO101CameraGridBinsTest(unittest.TestCase):
    def test_assigns_visible_green_object_to_expected_4x4_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/chunk-000").mkdir(parents=True)
            rows = [
                _row(episode=0, x0=10, y0=80),
                _row(episode=1, x0=200, y0=140),
            ]
            pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)

            report = build_bins(
                dataset_root=root,
                camera_key="observation.images.camera1",
                grid_size=4,
                frame_index=0,
                min_area=20,
            )

            table = pd.read_parquet(report["parquet_path"])
            self.assertEqual(report["visible_episodes"], 2)
            self.assertEqual(table["grid_bin"].tolist(), [4, 11])


def _row(*, episode: int, x0: int, y0: int) -> dict:
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    for x in range(x0, x0 + 20):
        for y in range(y0, y0 + 20):
            image.putpixel((x, y), (0, 220, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {
        "episode_index": episode,
        "frame_index": 0,
        "observation.images.camera1": {"bytes": buffer.getvalue()},
    }


if __name__ == "__main__":
    unittest.main()
