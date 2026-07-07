#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
PHOTO_CAMERA_CONTRACT = {
    "observation.images.camera1": "photoreal egocentric_cam",
    "observation.images.camera2": "photoreal wrist_cam",
    "observation.images.camera3": "photoreal wrist_cam duplicate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an SO101 LeRobot parquet dataset and replace embedded image bytes "
            "with Blender photoreal renders while preserving state/action/timestamps."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-keys", default=",".join(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--duplicate-camera3-from-camera2", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    report = build_photoreal_lerobot_dataset(
        source_dataset_root=args.source_dataset_root,
        rendered_dir=args.rendered_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        camera_keys=tuple(item.strip() for item in args.camera_keys.split(",") if item.strip()),
        duplicate_camera3_from_camera2=args.duplicate_camera3_from_camera2,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_photoreal_lerobot_dataset(
    *,
    source_dataset_root: Path,
    rendered_dir: Path,
    output_root: Path,
    repo_id: str,
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
    duplicate_camera3_from_camera2: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dataset_root = source_dataset_root.resolve()
    rendered_dir = rendered_dir.resolve()
    output_root = output_root.resolve()
    if not (source_dataset_root / "data").exists():
        raise FileNotFoundError(f"missing source LeRobot data directory: {source_dataset_root / 'data'}")
    if not rendered_dir.exists():
        raise FileNotFoundError(f"missing rendered frame directory: {rendered_dir}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)

    rendered_index = _rendered_index(rendered_dir)
    missing: list[str] = []
    replaced_frames = 0
    replaced_images = 0

    shutil.copytree(source_dataset_root, output_root, ignore=shutil.ignore_patterns("photoreal_preview"))
    data_files = sorted((output_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing parquet files under {output_root / 'data'}")

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        columns = table.to_pydict()
        row_count = table.num_rows
        replacement_columns: dict[str, list[dict[str, bytes | str | None]]] = {key: [] for key in camera_keys}
        for row_index in range(row_count):
            episode = int(columns["episode_index"][row_index])
            frame = int(columns["frame_index"][row_index])
            for camera_key in camera_keys:
                render_path = rendered_index.get((episode, frame, camera_key))
                if render_path is None and duplicate_camera3_from_camera2 and camera_key == "observation.images.camera3":
                    render_path = rendered_index.get((episode, frame, "observation.images.camera2"))
                if render_path is None:
                    missing.append(f"episode={episode} frame={frame} camera={camera_key}")
                    replacement_columns[camera_key].append(columns[camera_key][row_index])
                    continue
                replacement_columns[camera_key].append(
                    {
                        "bytes": _rgb_png_bytes(render_path),
                        "path": f"images/{camera_key.replace('.', '_')}/episode_{episode:04d}_frame_{frame:04d}.png",
                    }
                )
                replaced_images += 1
            replaced_frames += 1
        if missing:
            continue
        updated = table
        for camera_key, values in replacement_columns.items():
            field_index = updated.schema.get_field_index(camera_key)
            if field_index < 0:
                raise ValueError(f"source parquet does not contain camera column: {camera_key}")
            array = pa.array(values, type=updated.schema.field(field_index).type)
            updated = updated.set_column(field_index, camera_key, array)
        pq.write_table(updated, parquet_path)

    if missing:
        shutil.rmtree(output_root)
        sample = ", ".join(missing[:12])
        suffix = f" ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise ValueError(f"missing rendered frames: {sample}{suffix}")

    _write_manifest(
        output_root=output_root,
        source_dataset_root=source_dataset_root,
        rendered_dir=rendered_dir,
        repo_id=repo_id,
        camera_keys=camera_keys,
        replaced_frames=replaced_frames,
        replaced_images=replaced_images,
        duplicate_camera3_from_camera2=duplicate_camera3_from_camera2,
    )
    return _read_manifest(output_root)


def _rendered_index(rendered_dir: Path) -> dict[tuple[int, int, str], Path]:
    index: dict[tuple[int, int, str], Path] = {}
    patterns = (
        re.compile(r"episode_(\d+)_frame_(\d+)_(camera\d)\.png$"),
        re.compile(r"episode_(\d+)_frame_(\d+)_(observation_images_camera\d)\.png$"),
    )
    for path in sorted(rendered_dir.rglob("*.png")):
        for pattern in patterns:
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            camera = match.group(3).replace("observation_images_", "")
            index[(int(match.group(1)), int(match.group(2)), f"observation.images.{camera}")] = path
            break
    return index


def _rgb_png_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue()


def _write_manifest(
    *,
    output_root: Path,
    source_dataset_root: Path,
    rendered_dir: Path,
    repo_id: str,
    camera_keys: tuple[str, ...],
    replaced_frames: int,
    replaced_images: int,
    duplicate_camera3_from_camera2: bool,
) -> None:
    info = _read_json(output_root / "meta" / "info.json")
    total_episodes = int(info.get("total_episodes") or 0)
    total_frames = int(info.get("total_frames") or replaced_frames)
    manifest = {
        "format": "so101_photoreal_lerobot_v1",
        "repo_id": repo_id,
        "source_dataset_root": str(source_dataset_root),
        "source_dataset_name": source_dataset_root.name,
        "rendered_dir": str(rendered_dir),
        "episodes": total_episodes,
        "frames": total_frames,
        "fps": info.get("fps"),
        "camera_keys": list(camera_keys),
        "camera_contract": PHOTO_CAMERA_CONTRACT,
        "duplicate_camera3_from_camera2": duplicate_camera3_from_camera2,
        "replaced_frames": replaced_frames,
        "replaced_images": replaced_images,
        "training_ready": replaced_images == total_frames * len(camera_keys),
        "note": "LeRobot parquet root with photoreal image bytes and original state/action/timestamp columns.",
    }
    (output_root / "photoreal_lerobot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_manifest(output_root: Path) -> dict[str, Any]:
    return json.loads((output_root / "photoreal_lerobot_manifest.json").read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
