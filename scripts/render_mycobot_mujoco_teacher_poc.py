#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    MYCOBOT_MODEL_JOINT_NAMES,
    build_mycobot_nexus_scene_model,
)

MODEL_RELATIVE_PATH = Path("xml/mycobot_280jn_mujoco.xml")


@dataclass(frozen=True)
class RenderAttempt:
    backend: str
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class RenderReport:
    operation: str
    status: str
    root: str
    asset_root: str
    model_path: str
    scene_path: str
    frames: int
    width: int
    height: int
    output_dir: str
    blocker_path: str
    attempts: list[RenderAttempt]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render real myCobot MuJoCo frames for the teacher-data POC using the "
            "official elephantrobotics/mycobot_mujoco model assets."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/mycobot_ros_teacher_poc_mac"))
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("MYCOBOT_MUJOCO_ROOT", "_vendor/mycobot_mujoco")),
        help="Path to a local clone of https://github.com/elephantrobotics/mycobot_mujoco.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--require-render", action="store_true")
    args = parser.parse_args()

    report = render_mycobot_mujoco_teacher_poc(
        root=args.root,
        asset_root=args.asset_root,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    if args.require_render and report.status != "passed":
        raise SystemExit(2)


def render_mycobot_mujoco_teacher_poc(
    *,
    root: Path,
    asset_root: Path,
    width: int,
    height: int,
    max_frames: int = 0,
) -> RenderReport:
    root = root.expanduser()
    asset_root = asset_root.expanduser()
    output_dir = root / "render" / "scene"
    render_dir = root / "render"
    output_dir.mkdir(parents=True, exist_ok=True)
    blocker_path = render_dir / "render_blocker.md"
    report_path = render_dir / "render_report.json"
    model_path = asset_root / MODEL_RELATIVE_PATH
    scene_path = render_dir / "mycobot_nexus_scene.xml"
    attempts: list[RenderAttempt] = []

    try:
        frames = _read_frames(root / "data" / "frames.jsonl")
        if max_frames > 0:
            frames = frames[:max_frames]
        if not model_path.exists():
            raise FileNotFoundError(
                f"missing myCobot MuJoCo model: {model_path}. "
                "Clone https://github.com/elephantrobotics/mycobot_mujoco and pass --asset-root."
            )
        build_mycobot_nexus_scene_model(model_path=model_path, scene_path=scene_path)
        frame_count = _render_frames(
            model_path=scene_path,
            frames=frames,
            output_dir=output_dir,
            width=width,
            height=height,
        )
        attempts.append(RenderAttempt("mujoco.Renderer", True))
        blocker_path.write_text("", encoding="utf-8")
        status = "passed"
    except Exception as exc:  # noqa: BLE001
        attempts.append(RenderAttempt("mujoco.Renderer", False, _short_error(exc)))
        frame_count = 0
        status = "blocked"
        _write_blocker(blocker_path, attempts, model_path)

    report = RenderReport(
        operation="render_mycobot_mujoco_teacher_poc",
        status=status,
        root=str(root),
        asset_root=str(asset_root),
        model_path=str(model_path),
        scene_path=str(scene_path),
        frames=frame_count,
        width=width,
        height=height,
        output_dir=str(output_dir),
        blocker_path=str(blocker_path),
        attempts=attempts,
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _read_frames(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing teacher frames: {path}")
    frames = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frames:
        raise ValueError(f"no teacher frames in {path}")
    return frames


def _render_frames(
    *,
    model_path: Path,
    frames: list[dict[str, Any]],
    output_dir: Path,
    width: int,
    height: int,
) -> int:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [-0.13, -0.10, 0.11]
        camera.distance = 1.12
        camera.azimuth = 138.0
        camera.elevation = -34.0
        qpos_indices = _joint_qpos_indices(mujoco, model)
        for frame in frames:
            values = _finite_vector(frame.get("observation_state", []))
            for value_index, qpos_index in enumerate(qpos_indices):
                if value_index < len(values):
                    data.qpos[qpos_index] = values[value_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            frame_index = int(frame["frame_index"])
            _write_bmp(output_dir / f"frame_{frame_index:06d}.bmp", rgb)
    finally:
        renderer.close()
    return len(frames)


def _joint_qpos_indices(mujoco: Any, model: Any) -> list[int]:
    indices: list[int] = []
    for name in MYCOBOT_MODEL_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def _finite_vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        number = float(value)
        out.append(number if math.isfinite(number) else 0.0)
    return out


def _write_bmp(path: Path, rgb: Any) -> None:
    height = int(rgb.shape[0])
    width = int(rgb.shape[1])
    row_stride = (width * 3 + 3) & ~3
    image_size = row_stride * height
    file_size = 14 + 40 + image_size
    with path.open("wb") as file:
        file.write(b"BM")
        file.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        file.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                width,
                height,
                1,
                24,
                0,
                image_size,
                2835,
                2835,
                0,
                0,
            )
        )
        padding = b"\x00" * (row_stride - width * 3)
        for y in range(height - 1, -1, -1):
            row = rgb[y, :, :3]
            file.write(row[:, ::-1].tobytes())
            file.write(padding)


def _write_blocker(path: Path, attempts: list[RenderAttempt], model_path: Path) -> None:
    lines = [
        "# myCobot MuJoCo Render Blocker",
        "",
        "The teacher-data POC was generated, but no real robot-arm render frames were produced.",
        "",
        "## Required local asset",
        "",
        "```bash",
        "git clone https://github.com/elephantrobotics/mycobot_mujoco.git _vendor/mycobot_mujoco",
        "```",
        "",
        f"Expected model path: `{model_path}`",
        "",
        "## Attempts",
        "",
    ]
    for attempt in attempts:
        status = "ok" if attempt.ok else "failed"
        lines.append(f"- `{attempt.backend}`: {status}")
        if attempt.error:
            lines.append(f"  - `{attempt.error}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]


if __name__ == "__main__":
    main()
