#!/bin/sh
set -eu

ROOT="${ROOT:-_workspace/mycobot_ros_teacher_poc_mac}"
FRAMES="${FRAMES:-24}"
FPS="${FPS:-12}"
WIDTH="${WIDTH:-96}"
HEIGHT="${HEIGHT:-96}"
INPUT_TRACE="${INPUT_TRACE:-}"
OPEN_UI="${OPEN_UI:-0}"
RENDER_3D="${RENDER_3D:-0}"
REQUIRE_3D_RENDER="${REQUIRE_3D_RENDER:-0}"
MYCOBOT_MUJOCO_ROOT="${MYCOBOT_MUJOCO_ROOT:-_vendor/mycobot_mujoco}"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

case "$(uname -s)" in
  Darwin)
    RUNTIME_PLATFORM="macos"
    ;;
  *)
    RUNTIME_PLATFORM="non_macos"
    ;;
esac

set -- \
  --root "$ROOT" \
  --frames "$FRAMES" \
  --fps "$FPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --overwrite

if [ -n "$INPUT_TRACE" ]; then
  set -- "$@" --input-trace "$INPUT_TRACE"
fi

echo "runtime_platform=$RUNTIME_PLATFORM"
echo "python=$PYTHON_BIN"
echo "root=$ROOT"

"$PYTHON_BIN" scripts/export_mycobot_ros_teacher_poc.py "$@"

if [ "$RENDER_3D" = "1" ] || [ "$REQUIRE_3D_RENDER" = "1" ]; then
  set -- \
    --root "$ROOT" \
    --asset-root "$MYCOBOT_MUJOCO_ROOT" \
    --width 640 \
    --height 360
  if [ "$REQUIRE_3D_RENDER" = "1" ]; then
    "$PYTHON_BIN" scripts/render_mycobot_mujoco_teacher_poc.py "$@" --require-render
  else
    "$PYTHON_BIN" scripts/render_mycobot_mujoco_teacher_poc.py "$@"
  fi
fi

"$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_path = root / "report.json"
frames_path = root / "data" / "frames.jsonl"
info_path = root / "meta" / "info.json"
viewer_path = root / "viewer.html"
if not report_path.exists():
    raise SystemExit(f"missing report: {report_path}")
if not frames_path.exists():
    raise SystemExit(f"missing frames: {frames_path}")
if not info_path.exists():
    raise SystemExit(f"missing info: {info_path}")
if not viewer_path.exists():
    raise SystemExit(f"missing viewer: {viewer_path}")

report = json.loads(report_path.read_text(encoding="utf-8"))
frames = frames_path.read_text(encoding="utf-8").splitlines()
if report.get("status") != "passed":
    raise SystemExit(f"unexpected status: {report.get('status')}")
if len(frames) != int(report["frames"]):
    raise SystemExit(f"frame count mismatch: {len(frames)} != {report['frames']}")

first = json.loads(frames[0])
for key in ("top_image", "wrist_image"):
    image_path = root / first[key]
    if not image_path.exists():
        raise SystemExit(f"missing image: {image_path}")

print(f"mac_poc_report={report_path}")
print(f"mac_poc_viewer={viewer_path}")
render_report = root / "render" / "render_report.json"
if render_report.exists():
    render = json.loads(render_report.read_text(encoding="utf-8"))
    print(f"mac_poc_render_status={render.get('status')}")
    print(f"mac_poc_render_report={render_report}")
print(f"mac_poc_frames={len(frames)}")
print("mac_poc_status=passed")
PY

if [ "$OPEN_UI" = "1" ]; then
  if command -v open >/dev/null 2>&1; then
    open "$ROOT/viewer.html"
  else
    echo "open_ui=skipped_no_open_command"
  fi
fi
