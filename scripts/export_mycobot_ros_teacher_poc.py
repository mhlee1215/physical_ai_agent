#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
    "gripper_controller",
]
TASK = "Use MoveIt/Gazebo teacher motion to reach the visible can with the myCobot gripper."
DEFAULT_SOURCE_NOTES = {
    "official_ros_repo": "https://github.com/elephantrobotics/mycobot_ros",
    "official_ros2_repo": "https://github.com/elephantrobotics/mycobot_ros2",
    "candidate_ros1_launch": (
        "roslaunch mycobot_280_gripper_moveit demo_gazebo.launch gazebo_gui:=false"
    ),
    "candidate_unofficial_table_launch": (
        "roslaunch mycobot_move_it_config demo_gazebo.launch gazebo_gui:=false"
    ),
    "official_ros1_moveit_doc": (
        "https://docs.elephantrobotics.com/docs/gitbook-en/12-ApplicationBaseROS/"
        "12.1-ROS1/12.1.5-Moveit/myCobot-280.html"
    ),
}


@dataclass(frozen=True)
class MyCobotTeacherFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    observation_state: list[float]
    action: list[float]
    task: str
    top_image: str
    wrist_image: str
    source: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small myCobot ROS/Gazebo/MoveIt teacher-data POC. The default path "
            "is offline and deterministic; pass --input-trace with JSONL records captured "
            "from ROS topics to convert a real Gazebo/MoveIt run."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/mycobot_ros_teacher_poc"))
    parser.add_argument("--input-trace", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--repo-id", default="physical-ai-agent/mycobot-ros-teacher-poc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_mycobot_ros_teacher_poc(
        root=args.root,
        input_trace=args.input_trace,
        episode_index=args.episode_index,
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def export_mycobot_ros_teacher_poc(
    *,
    root: Path,
    input_trace: Path | None,
    episode_index: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
    repo_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    root.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "images" / "top").mkdir(parents=True)
    (root / "images" / "wrist").mkdir(parents=True)
    (root / "meta").mkdir()

    raw_records = (
        load_trace(input_trace)
        if input_trace
        else synthetic_moveit_trace(frames=frames, fps=fps)
    )
    teacher_frames = build_teacher_frames(
        raw_records,
        root=root,
        episode_index=episode_index,
        width=width,
        height=height,
    )
    if not teacher_frames:
        raise ValueError("trace did not produce any teacher frames")

    frames_path = root / "data" / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as file:
        for frame in teacher_frames:
            file.write(json.dumps(asdict(frame), sort_keys=True) + "\n")

    episode = {
        "episode_index": episode_index,
        "length": len(teacher_frames),
        "from_frame": 0,
        "to_frame": len(teacher_frames) - 1,
        "task": TASK,
        "success": None,
        "success_label": "not_claimed_poc_trace_only",
    }
    episodes_path = root / "data" / "episodes.jsonl"
    episodes_path.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")

    info = {
        "repo_id": repo_id,
        "robot_type": "mycobot_280_ros_gazebo_moveit_poc",
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "action": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "observation.images.top": {"dtype": "image", "shape": [height, width, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [height, width, 3]},
            "task": {"dtype": "string"},
        },
        "joint_names": JOINT_NAMES,
        "source_notes": DEFAULT_SOURCE_NOTES,
        "poc_boundary": (
            "Offline POC adapter for ROS/Gazebo/MoveIt traces. It does not claim "
            "Gazebo task success until object pose/contact success checks are added."
        ),
    }
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "operation": "export_mycobot_ros_teacher_poc",
        "status": "passed",
        "root": str(root),
        "input_trace": str(input_trace) if input_trace else "synthetic_moveit_trace",
        "frames": len(teacher_frames),
        "episodes": 1,
        "fps": fps,
        "image_size": [width, height],
        "dataset_files": {
            "frames": str(frames_path),
            "episodes": str(episodes_path),
            "info": str(info_path),
        },
        "poc_boundary": info["poc_boundary"],
        "next_steps": [
            "Record /joint_states, MoveIt FollowJointTrajectory goals, and Gazebo camera topics.",
            "Add Gazebo model-state object pose and gripper/contact success oracle.",
            "Swap placeholder PPM images for decoded ROS image messages before training.",
        ],
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_trace(input_trace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_trace.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_trace}:{line_number}: invalid JSONL record") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{input_trace}:{line_number}: expected object record")
            records.append(payload)
    return records


def synthetic_moveit_trace(*, frames: int, fps: int) -> list[dict[str, Any]]:
    count = max(2, int(frames))
    records = []
    for index in range(count):
        phase = index / float(max(1, count - 1))
        smooth = 0.5 - 0.5 * math.cos(math.pi * phase)
        positions = [
            -0.42 + 0.58 * smooth,
            0.18 - 0.36 * smooth,
            -0.28 + 0.42 * smooth,
            -0.12 - 0.30 * smooth,
            0.20 * math.sin(math.pi * phase),
            0.06 + 0.18 * smooth,
            -0.78 + 0.93 * min(1.0, phase * 1.4),
        ]
        next_phase = min(1.0, (index + 1) / float(max(1, count - 1)))
        next_smooth = 0.5 - 0.5 * math.cos(math.pi * next_phase)
        command = [
            -0.42 + 0.58 * next_smooth,
            0.18 - 0.36 * next_smooth,
            -0.28 + 0.42 * next_smooth,
            -0.12 - 0.30 * next_smooth,
            0.20 * math.sin(math.pi * next_phase),
            0.06 + 0.18 * next_smooth,
            -0.78 + 0.93 * min(1.0, next_phase * 1.4),
        ]
        records.append(
            {
                "timestamp": index / float(fps),
                "joint_state": {"name": JOINT_NAMES, "position": positions},
                "planned_action": command,
                "moveit_goal": "reach_coke_can_on_table",
                "gazebo_world": "table.world",
            }
        )
    return records


def build_teacher_frames(
    records: list[dict[str, Any]],
    *,
    root: Path,
    episode_index: int,
    width: int,
    height: int,
) -> list[MyCobotTeacherFrame]:
    frames: list[MyCobotTeacherFrame] = []
    for frame_index, record in enumerate(records):
        state = extract_joint_vector(record)
        action = extract_action_vector(record, fallback=state)
        top_path = (
            root / "images" / "top" / f"episode_{episode_index:03d}_frame_{frame_index:04d}.ppm"
        )
        wrist_path = (
            root
            / "images"
            / "wrist"
            / f"episode_{episode_index:03d}_frame_{frame_index:04d}.ppm"
        )
        write_state_image(top_path, state, width=width, height=height, view="top")
        write_state_image(wrist_path, action, width=width, height=height, view="wrist")
        frames.append(
            MyCobotTeacherFrame(
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=float(record.get("timestamp", frame_index)),
                observation_state=state,
                action=action,
                task=str(record.get("task", TASK)),
                top_image=str(top_path.relative_to(root)),
                wrist_image=str(wrist_path.relative_to(root)),
                source={
                    "moveit_goal": record.get("moveit_goal"),
                    "gazebo_world": record.get("gazebo_world"),
                    "source_record_index": frame_index,
                },
            )
        )
    return frames


def extract_joint_vector(record: dict[str, Any]) -> list[float]:
    joint_state = record.get("joint_state") or record.get("joint_states") or {}
    if not isinstance(joint_state, dict):
        raise ValueError("joint_state must be an object")
    positions = joint_state.get("position")
    names = joint_state.get("name") or JOINT_NAMES
    if not isinstance(positions, list):
        raise ValueError("joint_state.position must be a list")
    if not isinstance(names, list):
        raise ValueError("joint_state.name must be a list when provided")
    by_name = {str(name): float(value) for name, value in zip(names, positions)}
    if all(name in by_name for name in JOINT_NAMES):
        return [float(by_name[name]) for name in JOINT_NAMES]
    if len(positions) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} joint positions")
    return [float(value) for value in positions[: len(JOINT_NAMES)]]


def extract_action_vector(record: dict[str, Any], *, fallback: list[float]) -> list[float]:
    action = (
        record.get("planned_action")
        or record.get("commanded_joint_positions")
        or record.get("trajectory_point")
        or fallback
    )
    if isinstance(action, dict):
        positions = action.get("positions") or action.get("position")
        names = action.get("joint_names") or action.get("name") or JOINT_NAMES
        if not isinstance(positions, list):
            raise ValueError("action dict must include positions")
        by_name = {
            str(name): float(value) for name, value in zip(names, positions)
        }
        if all(name in by_name for name in JOINT_NAMES):
            return [float(by_name[name]) for name in JOINT_NAMES]
        action = positions
    if not isinstance(action, list):
        raise ValueError("planned action must be a list or trajectory point object")
    if len(action) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} action values")
    return [float(value) for value in action[: len(JOINT_NAMES)]]


def write_state_image(
    path: Path, values: list[float], *, width: int, height: int, view: str
) -> None:
    width = max(16, int(width))
    height = max(16, int(height))
    pixels = bytearray()
    tint = 45 if view == "top" else 85
    for y in range(height):
        for x in range(width):
            idx = min(len(values) - 1, int(x / max(1, width) * len(values)))
            value = max(-1.0, min(1.0, float(values[idx])))
            bar_height = int((value + 1.0) * 0.5 * (height - 1))
            active = y >= height - 1 - bar_height
            r = 40 + int(120 * active) + tint
            g = 42 + int(90 * active) + (idx * 17) % 60
            b = 48 + int(70 * active) + (x + y) % 30
            pixels.extend((min(255, r), min(255, g), min(255, b)))
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(pixels)


if __name__ == "__main__":
    main()
