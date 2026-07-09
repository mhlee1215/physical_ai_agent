#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sidecar mapping SO101 episodes to camera image object-position grid bins."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--camera-key", default="observation.images.camera1")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args()

    report = build_bins(
        dataset_root=args.dataset_root,
        camera_key=args.camera_key,
        grid_size=args.grid_size,
        frame_index=args.frame_index,
        min_area=args.min_area,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_bins(
    *,
    dataset_root: Path,
    camera_key: str,
    grid_size: int,
    frame_index: int,
    min_area: int,
) -> dict:
    rows = []
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise SystemExit(f"no parquet data files under {dataset_root / 'data'}")
    for path in data_files:
        frame = pd.read_parquet(path, columns=["episode_index", "frame_index", camera_key])
        rows.append(frame[frame["frame_index"] == frame_index])
    first = pd.concat(rows, ignore_index=True).sort_values("episode_index")

    records = []
    counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    invisible = 0
    for _, row in first.iterrows():
        centroid = _green_centroid(row[camera_key], min_area=min_area)
        if centroid is None:
            invisible += 1
            records.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(frame_index),
                    "camera_key": camera_key,
                    "visible": False,
                    "centroid_x": None,
                    "centroid_y": None,
                    "area": 0,
                    "grid_x": None,
                    "grid_y": None,
                    "grid_bin": -1,
                }
            )
            continue
        x, y, area = centroid
        grid_x = min(grid_size - 1, max(0, int(x * grid_size)))
        grid_y = min(grid_size - 1, max(0, int(y * grid_size)))
        grid_bin = grid_y * grid_size + grid_x
        counts[grid_y, grid_x] += 1
        records.append(
            {
                "episode_index": int(row["episode_index"]),
                "frame_index": int(frame_index),
                "camera_key": camera_key,
                "visible": True,
                "centroid_x": float(x),
                "centroid_y": float(y),
                "area": int(area),
                "grid_x": int(grid_x),
                "grid_y": int(grid_y),
                "grid_bin": int(grid_bin),
            }
        )

    out_dir = dataset_root / "meta" / "camera_grid_bins"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{camera_key.replace('.', '_')}_{grid_size}x{grid_size}_frame{frame_index}"
    parquet_path = out_dir / f"{stem}.parquet"
    report_path = out_dir / f"{stem}.json"
    table = pd.DataFrame(records)
    table.to_parquet(parquet_path, index=False)

    visible_counts = counts[counts > 0]
    report = {
        "dataset_root": str(dataset_root),
        "camera_key": camera_key,
        "grid_size": grid_size,
        "frame_index": frame_index,
        "min_area": min_area,
        "episodes": int(len(records)),
        "visible_episodes": int(len(records) - invisible),
        "invisible_episodes": int(invisible),
        "occupied_bins": int((counts > 0).sum()),
        "empty_bins": int((counts == 0).sum()),
        "min_occupied_bin_count": int(visible_counts.min()) if len(visible_counts) else 0,
        "max_occupied_bin_count": int(visible_counts.max()) if len(visible_counts) else 0,
        "bin_counts_yx": counts.tolist(),
        "parquet_path": str(parquet_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _green_centroid(image_value: object, *, min_area: int) -> tuple[float, float, int] | None:
    if isinstance(image_value, dict):
        blob = image_value.get("bytes")
    else:
        blob = bytes(image_value)  # type: ignore[arg-type]
    if not blob:
        return None
    image = np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mask = (green > 80) & (green > red + 25) & (green > blue + 20)
    ys, xs = np.where(mask)
    if len(xs) < min_area:
        return None
    height, width = image.shape[:2]
    return float(xs.mean() / max(1, width - 1)), float(ys.mean() / max(1, height - 1)), int(len(xs))


if __name__ == "__main__":
    main()
