#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import JOINT_NAMES, ROBOT_TYPE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the stricter myCobot 280 Pi adaptive JSONL export into a native "
            "LeRobotDataset when the LeRobot runtime is available."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="physical-ai-agent/mycobot-280pi-adaptive")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-lerobot",
        action="store_true",
        help="Fail instead of writing a blocked report when lerobot is unavailable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = convert_mycobot_280_pi_adaptive_jsonl_to_lerobot(
        source_root=args.source_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        require_lerobot=args.require_lerobot,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"passed", "blocked"} else 1)


def convert_mycobot_280_pi_adaptive_jsonl_to_lerobot(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    fps: int | None,
    use_videos: bool,
    overwrite: bool,
    require_lerobot: bool = False,
    lerobot_dataset_cls: Any | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    frames = _load_jsonl(source_root / "data" / "frames.jsonl")
    episodes = _load_jsonl(source_root / "data" / "episodes.jsonl")
    info = _load_json(source_root / "meta" / "info.json")
    if not frames:
        raise ValueError("source dataset has no frames")
    if not episodes:
        raise ValueError("source dataset has no episodes")

    resolved_fps = int(fps if fps is not None else info.get("fps", 12))
    image_shape = _first_image_shape(source_root, frames[0])
    features = _lerobot_features(height=image_shape[0], width=image_shape[1], use_videos=use_videos)

    if lerobot_dataset_cls is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as exc:  # noqa: BLE001
            report = _blocked_report(
                source_root=source_root,
                output_root=output_root,
                repo_id=repo_id,
                fps=resolved_fps,
                features=features,
                reason=f"lerobot import failed: {exc}",
            )
            _write_report(output_root, report, overwrite=overwrite)
            if require_lerobot:
                raise RuntimeError(report["blocker"]) from exc
            return report
        lerobot_dataset_cls = LeRobotDataset

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    dataset = lerobot_dataset_cls.create(
        repo_id=repo_id,
        fps=resolved_fps,
        features=features,
        root=output_root,
        robot_type=ROBOT_TYPE,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    frames_by_episode: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        frames_by_episode.setdefault(int(frame["episode_index"]), []).append(frame)

    exported_frames = 0
    exported_episodes = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_frames = sorted(frames_by_episode.get(episode_index, []), key=lambda item: int(item["frame_index"]))
        if not episode_frames:
            raise ValueError(f"episode {episode_index} has no frames")
        for frame in episode_frames:
            dataset.add_frame(_native_frame(source_root, frame))
            exported_frames += 1
        dataset.save_episode()
        exported_episodes += 1
    if hasattr(dataset, "finalize"):
        dataset.finalize()

    report = {
        "operation": "convert_mycobot_280_pi_adaptive_jsonl_to_lerobot",
        "status": "passed",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "repo_id": repo_id,
        "robot_type": ROBOT_TYPE,
        "fps": resolved_fps,
        "use_videos": use_videos,
        "exported_episodes": exported_episodes,
        "exported_frames": exported_frames,
        "features": features,
        "source_episodes": episodes,
        "claim_boundary": (
            "This proves conversion through the native LeRobotDataset API. It does not prove "
            "SmolVLA train/eval success unless a separate training command runs on this output."
        ),
    }
    _write_report(output_root, report, overwrite=False)
    return report


def _blocked_report(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    fps: int,
    features: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "operation": "convert_mycobot_280_pi_adaptive_jsonl_to_lerobot",
        "status": "blocked",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "repo_id": repo_id,
        "robot_type": ROBOT_TYPE,
        "fps": fps,
        "features": features,
        "blocker": reason,
        "next_step": "Install LeRobot in the execution environment, then rerun with --require-lerobot.",
        "claim_boundary": "No native LeRobotDataset was written because the LeRobot runtime is unavailable.",
    }


def _write_report(output_root: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mycobot_280_pi_lerobot_convert_report.json"
    if report_path.exists() and not overwrite:
        report_path.unlink()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _native_frame(source_root: Path, frame: dict[str, Any]) -> dict[str, Any]:
    object_position = frame.get("object_position") or [0.0, 0.0, 0.0]
    return {
        "observation.images.camera1": _read_image_hwc_uint8(source_root / str(frame["top_image"])),
        "observation.images.camera2": _read_image_hwc_uint8(source_root / str(frame["wrist_image"])),
        "observation.state": _array(frame["observation_state"], dtype="float32"),
        "action": _array(frame["action"], dtype="float32"),
        "object_position": _array(object_position, dtype="float32"),
        "contact_count": _array([int(frame.get("contact_count", 0))], dtype="int64"),
        "task": str(frame.get("task", "")),
    }


def _lerobot_features(*, height: int, width: int, use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    image_feature = {
        "dtype": image_dtype,
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.camera1": dict(image_feature),
        "observation.images.camera2": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "object_position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "contact_count": {"dtype": "int64", "shape": (1,), "names": ["count"]},
    }


def _first_image_shape(source_root: Path, frame: dict[str, Any]) -> tuple[int, int]:
    image = _read_image_hwc_uint8(source_root / str(frame["top_image"]))
    shape = _shape_of(image)
    if len(shape) != 3:
        raise ValueError("decoded image must have HWC shape")
    return int(shape[0]), int(shape[1])


def _read_image_hwc_uint8(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing image: {path}")
    if path.suffix.lower() == ".ppm":
        return _read_ppm(path)
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Pillow is required to decode non-PPM image {path}") from exc
    with Image.open(path) as image:
        return _array(image.convert("RGB"), dtype="uint8")


def _read_ppm(path: Path) -> Any:
    data = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4 and index < len(data):
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])
    if len(tokens) != 4 or tokens[0] != b"P6":
        raise ValueError(f"unsupported PPM header in {path}")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if max_value != 255:
        raise ValueError(f"unsupported PPM max value {max_value} in {path}")
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    pixels = data[index:]
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"PPM pixel byte count mismatch in {path}: {len(pixels)} != {expected}")
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return [
            [list(pixels[row * width * 3 + col * 3 : row * width * 3 + col * 3 + 3]) for col in range(width)]
            for row in range(height)
        ]
    return np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3)).copy()


def _array(value: Any, *, dtype: str) -> Any:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        if dtype.startswith("float"):
            return [float(item) for item in value]
        if dtype.startswith("int"):
            return [int(item) for item in value]
        return value
    return np.asarray(value, dtype=getattr(np, dtype))


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    dims: list[int] = []
    current = value
    while isinstance(current, list):
        dims.append(len(current))
        current = current[0] if current else []
    return tuple(dims)

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(payload)
    return records


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    main()
