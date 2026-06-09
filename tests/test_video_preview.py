from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.video_preview import write_preview_gif


class VideoPreviewTest(unittest.TestCase):
    def test_write_preview_gif_decodes_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "motion.mp4"
            gif = root / "preview.gif"
            _write_video(video)

            write_preview_gif(source=video, output=gif, max_width=40, max_frames=4)

            from PIL import Image

            with Image.open(gif) as image:
                self.assertEqual(image.format, "GIF")
                self.assertLessEqual(image.size[0], 40)
                self.assertGreaterEqual(getattr(image, "n_frames", 1), 1)


def _write_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 60))
    if not writer.isOpened():
        raise RuntimeError("failed to open test video writer")
    for index in range(5):
        frame = np.full((60, 80, 3), 240, dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 5, 20), (35 + index * 5, 45), (0, 190, 55), thickness=-1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    unittest.main()
