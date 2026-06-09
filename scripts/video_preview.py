from __future__ import annotations

from pathlib import Path


def write_preview_gif(
    *,
    source: Path,
    output: Path,
    max_width: int = 640,
    max_frames: int = 16,
    frame_duration_ms: int = 120,
) -> None:
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {source}")
    frames = []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    stride = max(1, frame_count // max_frames)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if width > max_width:
                target_height = max(1, int(height * (max_width / width)))
                rgb = cv2.resize(rgb, (max_width, target_height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(rgb))
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {source}")
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
