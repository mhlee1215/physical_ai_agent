#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    "joint7_to_joint6",
    "gripper_controller",
]
ROBOT_TYPE = "mycobot_280_pi_adaptive_gripper"
DEFAULT_TASK = "Pick up the object with the myCobot 280 Pi adaptive gripper."
CAMERA_KEYS = ["top", "wrist"]


@dataclass(frozen=True)
class DatasetFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    observation_state: list[float]
    action: list[float]
    top_image: str
    wrist_image: str
    object_position: list[float] | None
    contact_count: int
    success: bool
    task: str
    source: dict[str, Any]


@dataclass(frozen=True)
class OracleResult:
    success: bool
    success_label: str
    max_lift_m: float
    max_contact_count: int
    first_success_frame: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a myCobot 280 Pi adaptive-gripper ROS/Gazebo trace plus real camera "
            "frame manifest into a LeRobot-style dataset folder."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--repo-id", default="physical-ai-agent/mycobot-280pi-adaptive")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_mycobot_280_pi_adaptive_lerobot_dataset(
        root=args.root,
        input_trace=args.input_trace,
        camera_manifest=args.camera_manifest,
        episode_index=args.episode_index,
        fps=args.fps,
        repo_id=args.repo_id,
        task=args.task,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def export_mycobot_280_pi_adaptive_lerobot_dataset(
    *,
    root: Path,
    input_trace: Path,
    camera_manifest: Path,
    episode_index: int,
    fps: int,
    repo_id: str,
    task: str,
    overwrite: bool,
) -> dict[str, Any]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    for camera in CAMERA_KEYS:
        (root / "images" / camera).mkdir(parents=True)

    trace_records = load_jsonl(input_trace)
    camera_records = load_camera_manifest(camera_manifest)
    if len(trace_records) != len(camera_records):
        raise ValueError(
            "trace and camera manifest must have the same frame count: "
            f"{len(trace_records)} != {len(camera_records)}"
        )
    if len(trace_records) < 2:
        raise ValueError("dataset export requires at least two frames")

    frames = build_dataset_frames(
        trace_records,
        camera_records,
        root=root,
        episode_index=episode_index,
        task=task,
    )
    oracle = compute_object_contact_oracle(frames)

    frames_path = root / "data" / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(asdict(frame), sort_keys=True) + "\n")

    episodes_path = root / "data" / "episodes.jsonl"
    episode = {
        "episode_index": episode_index,
        "length": len(frames),
        "from_frame": 0,
        "to_frame": len(frames) - 1,
        "task": task,
        "success": oracle.success,
        "success_label": oracle.success_label,
    }
    episodes_path.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")

    info = build_info(
        repo_id=repo_id,
        fps=fps,
        frame_count=len(frames),
        source_trace=input_trace,
        camera_manifest=camera_manifest,
    )
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps({"task_index": 0, "task": task}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_path = root / "meta" / "stats.json"
    stats_path.write_text(json.dumps(compute_stats(frames), indent=2, sort_keys=True), encoding="utf-8")

    smolvla_smoke_path = root / "meta" / "smolvla_tiny_smoke_plan.json"
    smolvla_smoke = build_smolvla_tiny_smoke_plan(root=root, repo_id=repo_id)
    smolvla_smoke_path.write_text(json.dumps(smolvla_smoke, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "operation": "export_mycobot_280_pi_adaptive_lerobot_dataset",
        "status": "passed",
        "root": str(root),
        "robot_type": ROBOT_TYPE,
        "frames": len(frames),
        "episodes": 1,
        "real_camera_frames": True,
        "oracle": asdict(oracle),
        "dataset_files": {
            "frames": str(frames_path),
            "episodes": str(episodes_path),
            "info": str(info_path),
            "tasks": str(tasks_path),
            "stats": str(stats_path),
            "smolvla_tiny_smoke_plan": str(smolvla_smoke_path),
        },
        "claim_boundary": (
            "This export validates schema, real-frame file provenance, and an object pose/contact "
            "success oracle. It does not run SmolVLA training unless a separate LeRobot/SmolVLA "
            "environment executes the generated smoke plan."
        ),
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(payload)
    return records


def load_camera_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        frames = payload.get("frames")
    else:
        frames = payload
    if not isinstance(frames, list) or not all(isinstance(frame, dict) for frame in frames):
        raise ValueError("camera manifest must be a list or an object with a frames list")
    return frames


def build_dataset_frames(
    trace_records: list[dict[str, Any]],
    camera_records: list[dict[str, Any]],
    *,
    root: Path,
    episode_index: int,
    task: str,
) -> list[DatasetFrame]:
    frames: list[DatasetFrame] = []
    for frame_index, (trace, cameras) in enumerate(zip(trace_records, camera_records, strict=True)):
        state = extract_joint_vector(trace)
        action = extract_action_vector(trace, fallback=state)
        copied_images = copy_camera_images(
            cameras,
            root=root,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        object_position = extract_object_position(trace)
        contact_count = extract_contact_count(trace)
        success = bool(trace.get("success", False)) or bool(
            object_position is not None and object_position[2] >= initial_object_z(trace_records) + 0.025 and contact_count >= 1
        )
        frames.append(
            DatasetFrame(
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=float(trace.get("timestamp", cameras.get("timestamp", frame_index))),
                observation_state=state,
                action=action,
                top_image=copied_images["top"],
                wrist_image=copied_images["wrist"],
                object_position=object_position,
                contact_count=contact_count,
                success=success,
                task=str(trace.get("task", task)),
                source={
                    "trace_record_index": frame_index,
                    "camera_manifest_index": frame_index,
                    "object_pose_source": object_pose_source(trace),
                    "contact_source": contact_source(trace),
                },
            )
        )
    return frames


def copy_camera_images(
    camera_record: dict[str, Any],
    *,
    root: Path,
    episode_index: int,
    frame_index: int,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for camera in CAMERA_KEYS:
        raw_path = camera_record.get(camera) or camera_record.get(f"{camera}_image")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"camera manifest frame {frame_index} missing {camera} image path")
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            manifest_root = Path(str(camera_record.get("manifest_root", "."))).expanduser()
            source = manifest_root / source
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"missing real camera frame for {camera}: {source}")
        suffix = source.suffix.lower()
        if suffix not in {".ppm", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"unsupported camera frame extension for {source}")
        dest = root / "images" / camera / f"episode_{episode_index:03d}_frame_{frame_index:04d}{suffix}"
        shutil.copyfile(source, dest)
        copied[camera] = str(dest.relative_to(root))
    return copied


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
        return [by_name[name] for name in JOINT_NAMES]
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
        by_name = {str(name): float(value) for name, value in zip(names, positions)}
        if all(name in by_name for name in JOINT_NAMES):
            return [by_name[name] for name in JOINT_NAMES]
        action = positions
    if not isinstance(action, list):
        raise ValueError("planned action must be a list or trajectory point object")
    if len(action) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} action values")
    return [float(value) for value in action[: len(JOINT_NAMES)]]


def extract_object_position(record: dict[str, Any]) -> list[float] | None:
    pose = record.get("object_pose") or record.get("gazebo_object_pose") or record.get("object_state")
    if pose is None:
        return None
    if isinstance(pose, dict):
        position = pose.get("position") or pose.get("xyz")
        if isinstance(position, dict):
            return [float(position[axis]) for axis in ("x", "y", "z")]
        if isinstance(position, list) and len(position) >= 3:
            return [float(value) for value in position[:3]]
    if isinstance(pose, list) and len(pose) >= 3:
        return [float(value) for value in pose[:3]]
    raise ValueError("object pose must include a 3D position")


def extract_contact_count(record: dict[str, Any]) -> int:
    contacts = record.get("contacts") or record.get("gripper_object_contacts") or record.get("contact_count")
    if contacts is None:
        return 0
    if isinstance(contacts, int):
        return contacts
    if isinstance(contacts, float):
        return int(contacts)
    if isinstance(contacts, list):
        return len(contacts)
    if isinstance(contacts, dict):
        if "count" in contacts:
            return int(contacts["count"])
        return sum(1 for value in contacts.values() if bool(value))
    raise ValueError("contacts must be count, list, or object")


def compute_object_contact_oracle(frames: list[DatasetFrame]) -> OracleResult:
    positions = [frame.object_position for frame in frames if frame.object_position is not None]
    if not positions:
        raise ValueError("object pose/contact oracle requires object_position on at least one frame")
    initial_z = positions[0][2]
    max_lift = max(position[2] - initial_z for position in positions)
    max_contacts = max(frame.contact_count for frame in frames)
    first_success = next((frame.frame_index for frame in frames if frame.success), None)
    success = first_success is not None or (max_lift >= 0.025 and max_contacts >= 1)
    label = "object_contact_lift_success" if success else "object_contact_lift_not_success"
    return OracleResult(
        success=success,
        success_label=label,
        max_lift_m=float(max_lift),
        max_contact_count=int(max_contacts),
        first_success_frame=first_success,
    )


def initial_object_z(records: list[dict[str, Any]]) -> float:
    for record in records:
        position = extract_object_position(record)
        if position is not None:
            return position[2]
    return 0.0


def object_pose_source(record: dict[str, Any]) -> str:
    for key in ("object_pose", "gazebo_object_pose", "object_state"):
        if key in record:
            return key
    return "missing"


def contact_source(record: dict[str, Any]) -> str:
    for key in ("contacts", "gripper_object_contacts", "contact_count"):
        if key in record:
            return key
    return "missing"


def build_info(
    *,
    repo_id: str,
    fps: int,
    frame_count: int,
    source_trace: Path,
    camera_manifest: Path,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "robot_type": ROBOT_TYPE,
        "fps": fps,
        "total_frames": frame_count,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "action": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "observation.images.top": {"dtype": "image", "shape": [None, None, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [None, None, 3]},
            "object_position": {"dtype": "float32", "shape": [3]},
            "contact_count": {"dtype": "int64", "shape": [1]},
            "task": {"dtype": "string"},
        },
        "joint_names": JOINT_NAMES,
        "source_trace": str(source_trace),
        "camera_manifest": str(camera_manifest),
        "profile": "280-pi-adaptive-gripper",
        "dataset_quality_gates": [
            "external camera frame files are required and copied",
            "object pose/contact success oracle is computed",
            "SmolVLA tiny smoke plan is emitted but not executed locally",
        ],
    }


def compute_stats(frames: list[DatasetFrame]) -> dict[str, Any]:
    state_dim = len(frames[0].observation_state)
    mins = [min(frame.observation_state[index] for frame in frames) for index in range(state_dim)]
    maxs = [max(frame.observation_state[index] for frame in frames) for index in range(state_dim)]
    return {
        "observation.state": {"min": mins, "max": maxs},
        "contact_count": {
            "min": min(frame.contact_count for frame in frames),
            "max": max(frame.contact_count for frame in frames),
        },
    }


def build_smolvla_tiny_smoke_plan(*, root: Path, repo_id: str) -> dict[str, Any]:
    return {
        "status": "blocked_until_lerobot_smolvla_env_available",
        "dataset_root": str(root),
        "repo_id": repo_id,
        "intent": "tiny train/eval smoke over the exported myCobot 280 Pi adaptive dataset",
        "minimum_command_shape": [
            "python",
            "-m",
            "lerobot.scripts.train",
            "--dataset.repo_id",
            repo_id,
            "--policy.type",
            "smolvla",
            "--steps",
            "1",
        ],
    }


if __name__ == "__main__":
    main()
