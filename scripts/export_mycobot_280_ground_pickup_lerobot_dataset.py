#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SmolVLA-loadable planning/intermediate export from a myCobot 280 "
            "ground-pickup teacher dataset. Native LeRobot parquet writing remains a "
            "separate runtime step."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_plan(
        source_root=args.source_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


def export_plan(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("teacher_attachment_enabled") is not False:
        raise ValueError("source manifest must have teacher_attachment_enabled=false")
    if manifest.get("object_teleport_during_pickup_lift") is not False:
        raise ValueError("source manifest must have object_teleport_during_pickup_lift=false")
    if list(manifest.get("joint_names", [])) != JOINT_NAMES:
        raise ValueError("source manifest joint_names do not match myCobot 280 training contract")

    episode_summaries = [item for item in manifest.get("episode_summaries", []) if isinstance(item, dict)]
    frame_count = 0
    rendered_frame_count = 0
    missing_episode_paths = []
    for summary in episode_summaries:
        frame_count += int(summary.get("frames", 0))
        rendered_frame_count += int(summary.get("rendered_frames", 0))
        episode_path = source_root / str(summary.get("path", ""))
        if not episode_path.exists():
            missing_episode_paths.append(str(episode_path))
    if missing_episode_paths:
        raise FileNotFoundError(f"missing episode files: {missing_episode_paths[:3]}")

    if output_root.exists() and overwrite and not dry_run:
        shutil.rmtree(output_root)
    if output_root.exists() and not overwrite and not dry_run and any(output_root.iterdir()):
        raise FileExistsError(f"{output_root} exists and is non-empty; pass --overwrite")

    features = {
        "observation.images.camera1": {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channels"],
            "source": "teacher render resized by native conversion/training input pipeline",
        },
        "observation.state": {"dtype": "float32", "shape": [7], "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": [7], "names": JOINT_NAMES},
    }
    exported_frames = []
    if not dry_run:
        exported_frames = _write_intermediate_dataset(
            source_root=source_root,
            output_root=output_root,
            manifest=manifest,
            repo_id=repo_id,
            features=features,
        )

    report = {
        "operation": "export_mycobot_280_ground_pickup_lerobot_dataset",
        "status": "passed",
        "dry_run": bool(dry_run),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "repo_id": repo_id,
        "source_format": manifest.get("format"),
        "source_dataset_id": manifest.get("dataset_id"),
        "episodes": manifest.get("episodes"),
        "passed_episodes": manifest.get("passed_episodes"),
        "frames": frame_count,
        "rendered_frames": rendered_frame_count,
        "exported_frames": len(exported_frames),
        "features": features,
        "dataset_files": {
            "frames": str(output_root / "data" / "frames.jsonl"),
            "episodes": str(output_root / "data" / "episodes.jsonl"),
            "info": str(output_root / "meta" / "info.json"),
            "tasks": str(output_root / "meta" / "tasks.jsonl"),
            "stats": str(output_root / "meta" / "stats.json"),
        }
        if not dry_run
        else {},
        "source_quality": {
            "generation_mode": manifest.get("generation_mode"),
            "randomization_enabled": manifest.get("randomization_enabled"),
            "teacher_attachment_enabled": manifest.get("teacher_attachment_enabled"),
            "object_teleport_during_pickup_lift": manifest.get("object_teleport_during_pickup_lift"),
            "success_criteria": manifest.get("success_criteria"),
            "aggregate_metrics": manifest.get("aggregate_metrics"),
        },
        "state_mapping": {
            "source_observation_state": "first 7 entries are robot joint/gripper state; trailing entries are cube position metadata",
            "exported_observation_state": JOINT_NAMES,
            "action": JOINT_NAMES,
            "gripper_semantics": "lower command closes in the current ground-pickup POC",
        },
        "native_lerobot_next_step": {
            "script": "scripts/convert_mycobot_280_pi_adaptive_jsonl_to_lerobot.py",
            "note": "Native LeRobotDataset writing requires the LeRobot runtime and is intentionally not installed by this planner.",
        },
        "claim_boundary": "SmolVLA-loadable conversion plan/intermediate export only; no policy training or native LeRobot parquet writing is claimed.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mycobot280_ground_pickup_lerobot_plan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _write_intermediate_dataset(
    *,
    source_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    repo_id: str,
    features: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "camera1").mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    next_frame_index = 0
    for episode_summary in manifest.get("episode_summaries", []):
        if not isinstance(episode_summary, dict):
            continue
        episode_index = int(episode_summary["episode_index"])
        source_episode = source_root / str(episode_summary["path"])
        episode_rows = _load_jsonl(source_episode)
        first = next_frame_index
        for row in episode_rows:
            frame = _convert_row(
                row=row,
                source_root=source_root,
                output_root=output_root,
                global_frame_index=next_frame_index,
            )
            frames.append(frame)
            next_frame_index += 1
        episodes.append(
            {
                "episode_index": episode_index,
                "from_frame": first,
                "to_frame": next_frame_index - 1,
                "length": next_frame_index - first,
                "task": manifest.get("task"),
                "success": bool(episode_summary.get("success", False)),
                "source_episode": str(source_episode),
            }
        )
    (output_root / "data" / "frames.jsonl").write_text(
        "".join(json.dumps(frame, sort_keys=True) + "\n" for frame in frames),
        encoding="utf-8",
    )
    (output_root / "data" / "episodes.jsonl").write_text(
        "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    info = {
        "repo_id": repo_id,
        "robot_type": "mycobot_280_pi_adaptive_gripper",
        "fps": manifest.get("fps"),
        "features": features,
        "source_format": manifest.get("format"),
        "source_dataset_id": manifest.get("dataset_id"),
        "claim_boundary": "Intermediate SmolVLA-loadable JSONL export; native LeRobotDataset parquet writing is a separate runtime step.",
    }
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": manifest.get("task")}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "meta" / "stats.json").write_text(json.dumps(_stats(frames), indent=2, sort_keys=True), encoding="utf-8")
    return frames


def _convert_row(*, row: dict[str, Any], source_root: Path, output_root: Path, global_frame_index: int) -> dict[str, Any]:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    images = observation.get("images") if isinstance(observation.get("images"), dict) else {}
    render = str(images.get("render", ""))
    if not render:
        raise ValueError(f"source row {global_frame_index} has no rendered image")
    source_image = source_root / render
    if not source_image.exists():
        raise FileNotFoundError(f"missing source image: {source_image}")
    image_name = f"frame_{global_frame_index:06d}{source_image.suffix.lower()}"
    image_target = output_root / "images" / "camera1" / image_name
    shutil.copy2(source_image, image_target)
    source_state = _float_list(observation.get("state"), "observation.state")
    action = _float_list(row.get("action"), "action")
    if len(source_state) < len(JOINT_NAMES):
        raise ValueError(f"state length {len(source_state)} < {len(JOINT_NAMES)}")
    if len(action) != len(JOINT_NAMES):
        raise ValueError(f"action length {len(action)} != {len(JOINT_NAMES)}")
    state = source_state[: len(JOINT_NAMES)]
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    ground = info.get("ground_pickup") if isinstance(info.get("ground_pickup"), dict) else {}
    return {
        "episode_index": int(row.get("episode_index", 0)),
        "frame_index": int(row.get("frame_index", global_frame_index)),
        "timestamp": float(row.get("timestamp", 0.0)),
        "task": str(row.get("task", "")),
        "observation.state": state,
        "observation.images.camera1": str(image_target.relative_to(output_root)),
        "action": action,
        "metadata": {
            "phase": row.get("phase"),
            "cube_lift_m": ground.get("cube_lift_m"),
            "cube_position_m": source_state[len(JOINT_NAMES) :],
            "pad_cube_contacted_pads": ground.get("pad_cube_contacted_pads"),
            "max_pad_cube_penetration_m": _nested_float(ground, ["pad_cube_contact_depth", "max_penetration_m"]),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(payload)
    return rows


def _float_list(value: Any, label: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [float(item) for item in value]


def _nested_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": len(frames),
        "episodes": len({int(frame["episode_index"]) for frame in frames}),
        "state_dim": len(frames[0]["observation.state"]) if frames else 0,
        "action_dim": len(frames[0]["action"]) if frames else 0,
        "joint_names": JOINT_NAMES,
    }


if __name__ == "__main__":
    main()
