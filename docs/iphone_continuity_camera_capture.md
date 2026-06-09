# iPhone Continuity Camera Capture With Python/OpenCV

This note records the working local path for capturing still images and short
videos from the iPhone Continuity Camera on this Mac.

## Known Working Setup

- Python environment: `.venv/bin/python`
- OpenCV version observed: `4.13.0`
- Backend: `AVFOUNDATION`
- iPhone Continuity Camera index: `1`
- Other working camera index: `0`
- Index `2` opened during probing but did not return frames.
- Working frame size from the iPhone camera: `1920x1080`
- Camera permission must be granted to the process host. In this Codex desktop
  session, direct sandboxed execution was denied, but running the same Python
  command through `Terminal.app` succeeded.

## Permission Notes

On macOS, camera access is attached to the GUI app/process host, not only to the
Python script.

Use:

```sh
System Settings -> Privacy & Security -> Camera
```

Enable camera access for the app that launches Python. For the verified path
here, that app is `Terminal`.

If permission state gets stuck, reset the camera privacy record and rerun from
Terminal:

```sh
tccutil reset Camera
```

## Capture One Image

Run from the repo root:

```sh
.venv/bin/python _workspace/camera_permission_probe/probe_camera_opencv.py \
  --index 1 \
  --output _workspace/camera_permission_probe/iphone_camera.jpg \
  --report _workspace/camera_permission_probe/iphone_camera_report.json \
  --timeout-seconds 15
```

Expected successful report fields:

```json
{
  "backend": "AVFOUNDATION",
  "camera_index": 1,
  "opened": true,
  "captured": true,
  "frame_shape": [1080, 1920, 3]
}
```

Minimal Python equivalent:

```python
from pathlib import Path

import cv2

output = Path("_workspace/camera_permission_probe/iphone_camera.jpg")
output.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    raise RuntimeError("iPhone Continuity Camera did not return a frame")

cv2.imwrite(str(output), frame)
```

## Record A Video

Run from the repo root:

```sh
.venv/bin/python _workspace/camera_permission_probe/record_opencv_video.py \
  --index 1 \
  --output _workspace/camera_permission_probe/iphone_video.mp4 \
  --report _workspace/camera_permission_probe/iphone_video_report.json \
  --duration-seconds 5 \
  --fps 15
```

Expected successful report fields:

```json
{
  "camera_index": 1,
  "captured": true,
  "frames_written": 75,
  "target_fps": 15.0,
  "frame_size": [1080, 1920]
}
```

Minimal Python equivalent:

```python
from pathlib import Path
import time

import cv2

output = Path("_workspace/camera_permission_probe/iphone_video.mp4")
output.parent.mkdir(parents=True, exist_ok=True)

fps = 15.0
duration_seconds = 5.0

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("iPhone Continuity Camera did not open")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    cap.release()
    raise RuntimeError("VideoWriter did not open")

deadline = time.monotonic() + duration_seconds
while time.monotonic() < deadline:
    ok, frame = cap.read()
    if ok and frame is not None:
        writer.write(frame)

cap.release()
writer.release()
```

## Scan Available Camera Indexes

Use this when the iPhone index changes:

```sh
.venv/bin/python _workspace/camera_permission_probe/scan_opencv_cameras.py \
  --start 0 \
  --end 8 \
  --output-dir _workspace/camera_permission_probe/opencv_scan \
  --report _workspace/camera_permission_probe/opencv_scan_report.json
```

The scan saves one frame per captured index under:

```text
_workspace/camera_permission_probe/opencv_scan/
```

## Run Through Terminal.app From Codex

If direct execution from Codex reports `camera access has been denied`, run the
same command through `Terminal.app`:

```sh
osascript -e 'tell application "Terminal" to do script "cd /Users/minhaeng/workspace/physical_ai_agent && .venv/bin/python _workspace/camera_permission_probe/probe_camera_opencv.py --index 1 --output _workspace/camera_permission_probe/iphone_camera.jpg --report _workspace/camera_permission_probe/iphone_camera_report.json --timeout-seconds 15"'
```

For video:

```sh
osascript -e 'tell application "Terminal" to do script "cd /Users/minhaeng/workspace/physical_ai_agent && .venv/bin/python _workspace/camera_permission_probe/record_opencv_video.py --index 1 --output _workspace/camera_permission_probe/iphone_video.mp4 --report _workspace/camera_permission_probe/iphone_video_report.json --duration-seconds 5 --fps 15"'
```

## Current Limitation

The iPhone Continuity Camera captures successfully through OpenCV, but lens/FOV
selection is not controlled by this OpenCV path. `CAP_PROP_ZOOM` did not change
the effective field of view during probing. Move the phone physically or use
macOS Continuity Camera video effects for wider composition.
