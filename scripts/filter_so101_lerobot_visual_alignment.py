#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PHASES_BEFORE_CLOSE = (
    "move_to_cube",
    "roll_align_with_cube_edge",
    "gripper_descend",
    "settle_aligned",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter SO101 LeRobot episodes by camera-space cube/jaw alignment."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--selection-source-root",
        type=Path,
        default=None,
        help="Score episode indices from this aligned source while copying frames from --source-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-key", default="observation.images.camera2")
    parser.add_argument("--max-angle-deg", type=float, default=20.0)
    parser.add_argument(
        "--edge-mode",
        choices=["legacy-any-axis", "top-contact"],
        default="legacy-any-axis",
        help=(
            "legacy-any-axis compares the jaw against the whole green mask rect axes. "
            "top-contact compares against the top-face edge nearest the gripper in camera space."
        ),
    )
    parser.add_argument("--max-kept", type=int, default=None)
    parser.add_argument("--sort-by-error", action="store_true")
    parser.add_argument(
        "--stable-close-check",
        action="store_true",
        help="Also require contact-edge alignment to remain bounded during close at 25%%, 50%%, and 75%% of the close phase.",
    )
    parser.add_argument("--close-25-max-angle-deg", type=float, default=5.0)
    parser.add_argument("--close-50-max-angle-deg", type=float, default=20.0)
    parser.add_argument("--close-75-max-angle-deg", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = filter_dataset(
        source_root=args.source_root,
        selection_source_root=args.selection_source_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        camera_key=args.camera_key,
        max_angle_deg=args.max_angle_deg,
        edge_mode=args.edge_mode,
        max_kept=args.max_kept,
        sort_by_error=args.sort_by_error,
        stable_close_check=args.stable_close_check,
        close_25_max_angle_deg=args.close_25_max_angle_deg,
        close_50_max_angle_deg=args.close_50_max_angle_deg,
        close_75_max_angle_deg=args.close_75_max_angle_deg,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def filter_dataset(
    *,
    source_root: Path,
    selection_source_root: Path | None = None,
    output_root: Path,
    repo_id: str,
    camera_key: str,
    max_angle_deg: float,
    edge_mode: str = "legacy-any-axis",
    max_kept: int | None = None,
    sort_by_error: bool = False,
    stable_close_check: bool = False,
    close_25_max_angle_deg: float = 5.0,
    close_50_max_angle_deg: float = 20.0,
    close_75_max_angle_deg: float = 25.0,
    overwrite: bool,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        from export_so101_teacher_rollouts_lerobot import _lerobot_features, audit_lerobot_dataset
        from merge_so101_lerobot_shards import _frame_from_source, _image_size_from_report
    except ModuleNotFoundError:  # pragma: no cover
        from scripts.export_so101_teacher_rollouts_lerobot import _lerobot_features, audit_lerobot_dataset
        from scripts.merge_so101_lerobot_shards import _frame_from_source, _image_size_from_report

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    source_report_path = source_root / "so101_lerobot_export_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    selection_source_root = selection_source_root or source_root
    selection_report = json.loads(
        (selection_source_root / "so101_lerobot_export_report.json").read_text(encoding="utf-8")
    )
    _validate_selection_source(source_report=source_report, selection_report=selection_report)
    include_camera3 = bool(source_report.get("camera3_duplicate", {}).get("enabled", True))
    width, height = _image_size_from_report(source_report)
    features = _lerobot_features(
        height=height,
        width=width,
        use_videos=False,
        include_camera3_duplicate=include_camera3,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(source_report["fps"]),
        features=features,
        root=output_root,
        robot_type="so101",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    source = LeRobotDataset(str(source_report["repo_id"]), root=source_root)
    frame_table = _load_frame_table(selection_source_root)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    source_episodes = source_report.get("episodes") or []
    for episode_index, episode in enumerate(selection_report.get("episodes") or []):
        score = _episode_alignment_score(
            frame_table=frame_table,
            episode=episode,
            episode_index=episode_index,
            camera_key=camera_key,
            edge_mode=edge_mode,
        )
        row = {
            "source_episode_index": int(episode_index),
            "image_alignment_error_deg": score.get("image_alignment_error_deg"),
            "cube_edge_angle_deg": score.get("cube_edge_angle_deg"),
            "cube_top_contact_edge_angle_deg": score.get("cube_top_contact_edge_angle_deg"),
            "jaw_angle_deg": score.get("jaw_angle_deg"),
            "camera_key": camera_key,
            "edge_mode": edge_mode,
            "threshold_deg": float(max_angle_deg),
            "reason": score.get("reason"),
            "grid_balance_bin": episode.get("grid_balance_bin"),
            "seed": episode.get("seed"),
            "trajectory_variant": episode.get("trajectory_variant"),
        }
        stable_scores: dict[str, Any] = {}
        stable_ok = True
        if stable_close_check:
            stable_scores = _episode_stable_close_scores(
                frame_table=frame_table,
                episode=episode,
                episode_index=episode_index,
                camera_key=camera_key,
                edge_mode=edge_mode,
            )
            row.update(stable_scores)
            stable_limits = {
                "close_25_image_alignment_error_deg": float(close_25_max_angle_deg),
                "close_50_image_alignment_error_deg": float(close_50_max_angle_deg),
                "close_75_image_alignment_error_deg": float(close_75_max_angle_deg),
            }
            for key, limit in stable_limits.items():
                value = stable_scores.get(key)
                if value is None or float(value) > limit:
                    stable_ok = False
                    break
            if stable_ok:
                row["stable_close_total_error_deg"] = float(
                    sum(float(stable_scores[key]) for key in stable_limits)
                    + float(score.get("image_alignment_error_deg") or 0.0)
                )
        if (
            score.get("image_alignment_error_deg") is not None
            and float(score["image_alignment_error_deg"]) <= float(max_angle_deg)
            and stable_ok
        ):
            kept_episode = dict(source_episodes[episode_index])
            kept_episode["source_episode_index"] = int(episode_index)
            kept_episode["image_space_alignment"] = row
            selected.append((kept_episode, row))
        else:
            rejected.append(row)

    if sort_by_error:
        selected.sort(
            key=lambda item: float(
                item[1].get("stable_close_total_error_deg")
                or item[1].get("image_alignment_error_deg")
                or math.inf
            )
        )
    if max_kept is not None:
        selected = selected[: int(max_kept)]

    kept: list[dict[str, Any]] = []
    for kept_episode, _row in selected:
        source_episode_index = int(kept_episode["source_episode_index"])
        source_episode = source.meta.episodes[source_episode_index]
        start = int(source_episode["dataset_from_index"])
        end = int(source_episode["dataset_to_index"])
        for index in range(start, end):
            dataset.add_frame(_frame_from_source(source[index], include_camera3=include_camera3))
        dataset.save_episode()
        kept.append(kept_episode)

    dataset.finalize()
    audit = audit_lerobot_dataset(
        root=output_root,
        repo_id=repo_id,
        features=features,
        action_space_low=np.asarray(source_report["audit"]["action_space_low"], dtype=np.float32),
        action_space_high=np.asarray(source_report["audit"]["action_space_high"], dtype=np.float32),
    )
    output_report = dict(source_report)
    output_report.update(
        {
            "operation": "filter_so101_lerobot_visual_alignment",
            "root": str(output_root),
            "repo_id": repo_id,
            "source_root": str(source_root),
            "selection_source_root": str(selection_source_root),
            "requested_episodes": len(kept),
            "exported_episodes": len(kept),
            "episodes": kept,
            "visual_alignment_filter": {
                "camera_key": camera_key,
                "edge_mode": edge_mode,
                "max_angle_deg": float(max_angle_deg),
                "kept_episodes": len(kept),
                "rejected_episodes": len(rejected),
                "source_episodes": len(source_report.get("episodes") or []),
                "max_kept": max_kept,
                "sort_by_error": bool(sort_by_error),
                "stable_close_check": bool(stable_close_check),
                "close_25_max_angle_deg": float(close_25_max_angle_deg),
                "close_50_max_angle_deg": float(close_50_max_angle_deg),
                "close_75_max_angle_deg": float(close_75_max_angle_deg),
                "rejected": rejected,
            },
            "audit": audit,
        }
    )
    report_path = output_root / "so101_lerobot_export_report.json"
    output_report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(output_report, indent=2, sort_keys=True), encoding="utf-8")
    merge_report = {
        "operation": "filter_so101_lerobot_visual_alignment",
        "source_root": str(source_root),
        "selection_source_root": str(selection_source_root),
        "output_root": str(output_root),
        "repo_id": repo_id,
        "camera_key": camera_key,
        "edge_mode": edge_mode,
        "max_angle_deg": float(max_angle_deg),
        "max_kept": max_kept,
        "sort_by_error": bool(sort_by_error),
        "stable_close_check": bool(stable_close_check),
        "close_25_max_angle_deg": float(close_25_max_angle_deg),
        "close_50_max_angle_deg": float(close_50_max_angle_deg),
        "close_75_max_angle_deg": float(close_75_max_angle_deg),
        "kept_episodes": len(kept),
        "rejected_episodes": len(rejected),
        "audit": audit,
    }
    (output_root / "so101_lerobot_visual_alignment_filter_report.json").write_text(
        json.dumps(merge_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "so101_lerobot_merge_report.json").write_text(
        json.dumps(merge_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_filtered_photoreal_manifest(
        source_root=source_root,
        output_root=output_root,
        repo_id=repo_id,
        selection_source_root=selection_source_root,
        episodes=len(kept),
        frames=int(audit["dataset_len"]),
    )
    return merge_report


def _validate_selection_source(
    *, source_report: dict[str, Any], selection_report: dict[str, Any]
) -> None:
    source_episodes = source_report.get("episodes") or []
    selection_episodes = selection_report.get("episodes") or []
    if len(source_episodes) != len(selection_episodes):
        raise ValueError(
            "selection source episode count does not match copied source: "
            f"{len(selection_episodes)} != {len(source_episodes)}"
        )
    mismatches: list[dict[str, Any]] = []
    for index, (source_episode, selection_episode) in enumerate(
        zip(source_episodes, selection_episodes, strict=True)
    ):
        for key in ("seed", "frames"):
            if source_episode.get(key) != selection_episode.get(key):
                mismatches.append(
                    {
                        "episode_index": index,
                        "field": key,
                        "source": source_episode.get(key),
                        "selection": selection_episode.get(key),
                    }
                )
        if len(mismatches) >= 8:
            break
    if mismatches:
        raise ValueError(
            "selection source is not episode-aligned with copied source: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _write_filtered_photoreal_manifest(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    selection_source_root: Path,
    episodes: int,
    frames: int,
) -> None:
    source_manifest_path = source_root / "photoreal_lerobot_manifest.json"
    if not source_manifest_path.is_file():
        return
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    camera_keys = list(source_manifest.get("camera_keys") or [])
    manifest = {
        **source_manifest,
        "repo_id": repo_id,
        "source_dataset_root": str(source_root.resolve()),
        "source_dataset_name": source_root.name,
        "selection_source_root": str(selection_source_root.resolve()),
        "operation": "episode_subset",
        "episodes": int(episodes),
        "frames": int(frames),
        "replaced_frames": int(frames),
        "replaced_images": int(frames) * len(camera_keys),
        "training_ready": bool(camera_keys) and frames > 0,
    }
    (output_root / "photoreal_lerobot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_frame_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _episode_alignment_score(
    *,
    frame_table: pd.DataFrame,
    episode: dict[str, Any],
    episode_index: int,
    camera_key: str,
    edge_mode: str = "legacy-any-axis",
) -> dict[str, Any]:
    phase_counts = episode.get("phase_counts") or {}
    pre_close_frame = sum(int(phase_counts.get(name, 0)) for name in PHASES_BEFORE_CLOSE) - 1
    rows = frame_table[frame_table["episode_index"] == int(episode_index)]
    if rows.empty:
        return {"reason": "missing_episode_rows"}
    row = rows.iloc[max(0, min(int(pre_close_frame), len(rows) - 1))]
    image = _decode_image(row[camera_key])
    return _image_alignment_score(image, edge_mode=edge_mode)


def _episode_stable_close_scores(
    *,
    frame_table: pd.DataFrame,
    episode: dict[str, Any],
    episode_index: int,
    camera_key: str,
    edge_mode: str,
) -> dict[str, Any]:
    trace_scores = _episode_stable_close_scores_from_trace(episode)
    if trace_scores:
        return trace_scores

    phase_counts = episode.get("phase_counts") or {}
    close_start = sum(
        int(phase_counts.get(name, 0))
        for name in (*PHASES_BEFORE_CLOSE,)
    )
    close_steps = int(phase_counts.get("close", 0))
    frame_offsets = {
        "close_25": max(0, int(close_steps * 0.25) - 1),
        "close_50": max(0, int(close_steps * 0.50) - 1),
        "close_75": max(0, int(close_steps * 0.75) - 1),
    }
    rows = frame_table[frame_table["episode_index"] == int(episode_index)]
    result: dict[str, Any] = {}
    for name, offset in frame_offsets.items():
        if rows.empty:
            result[f"{name}_reason"] = "missing_episode_rows"
            result[f"{name}_image_alignment_error_deg"] = None
            continue
        frame_index = max(0, min(int(close_start + offset), len(rows) - 1))
        row = rows.iloc[frame_index]
        score = _image_alignment_score(_decode_image(row[camera_key]), edge_mode=edge_mode)
        result[f"{name}_frame_index"] = int(frame_index)
        result[f"{name}_reason"] = score.get("reason")
        result[f"{name}_image_alignment_error_deg"] = score.get("image_alignment_error_deg")
        result[f"{name}_cube_top_contact_edge_angle_deg"] = score.get("cube_top_contact_edge_angle_deg")
        result[f"{name}_jaw_angle_deg"] = score.get("jaw_angle_deg")
    return result


def _episode_stable_close_scores_from_trace(episode: dict[str, Any]) -> dict[str, Any]:
    trace = episode.get("camera2_top_contact_close_alignment_trace")
    if not isinstance(trace, list):
        return {}
    wanted = {0.25: "close_25", 0.50: "close_50", 0.75: "close_75"}
    result: dict[str, Any] = {}
    for item in trace:
        if not isinstance(item, dict) or "checkpoint_fraction" not in item:
            continue
        fraction = float(item["checkpoint_fraction"])
        name = wanted.get(round(fraction, 2))
        if not name:
            continue
        actual = item.get("actual_after_step") or {}
        if not isinstance(actual, dict):
            actual = {}
        result[f"{name}_frame_index"] = item.get("close_index")
        result[f"{name}_reason"] = actual.get("reason", item.get("reason"))
        result[f"{name}_image_alignment_error_deg"] = actual.get("image_alignment_error_deg")
        result[f"{name}_cube_top_contact_edge_angle_deg"] = actual.get("cube_top_contact_edge_angle_deg")
        result[f"{name}_jaw_angle_deg"] = actual.get("jaw_angle_deg")
        result[f"{name}_source"] = "camera2_top_contact_close_alignment_trace"
    if all(f"{name}_image_alignment_error_deg" in result for name in wanted.values()):
        return result
    return {}


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        return np.array(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return array


def _image_alignment_score(image: np.ndarray, *, edge_mode: str = "legacy-any-axis") -> dict[str, Any]:
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)).astype(
        "uint8"
    )
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 55)).astype(
        "uint8"
    )
    kernel = np.ones((3, 3), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    cube_angle = _rect_edge_angle(green)
    jaw_angle = _jaw_axis_angle(yellow)
    if cube_angle is None:
        return {"reason": "missing_cube_mask"}
    if jaw_angle is None:
        return {"reason": "missing_jaw_mask", "cube_edge_angle_deg": cube_angle}
    if edge_mode == "top-contact":
        contact = _top_contact_edge_angle(image=image, green=green, yellow=yellow)
        if contact is None:
            return {
                "reason": "missing_top_contact_edge",
                "cube_edge_angle_deg": cube_angle,
                "jaw_angle_deg": jaw_angle,
                "green_pixels": int(green.sum()),
                "yellow_pixels": int(yellow.sum()),
            }
        error = _angle_diff(contact["cube_top_contact_edge_angle_deg"], jaw_angle)
        return {
            "reason": "ok",
            "image_alignment_error_deg": float(error),
            "cube_edge_angle_deg": float(cube_angle),
            "cube_top_contact_edge_angle_deg": float(contact["cube_top_contact_edge_angle_deg"]),
            "jaw_angle_deg": float(jaw_angle),
            "contact_edge_distance_px": float(contact["contact_edge_distance_px"]),
            "green_pixels": int(green.sum()),
            "top_face_pixels": int(contact["top_face_pixels"]),
            "yellow_pixels": int(yellow.sum()),
        }
    # A square cube has two valid visible edge axes. Compare against both.
    error = min(_angle_diff(cube_angle, jaw_angle), _angle_diff((cube_angle + 90.0) % 180.0, jaw_angle))
    return {
        "reason": "ok",
        "image_alignment_error_deg": float(error),
        "cube_edge_angle_deg": float(cube_angle),
        "jaw_angle_deg": float(jaw_angle),
        "green_pixels": int(green.sum()),
        "yellow_pixels": int(yellow.sum()),
    }


def _top_contact_edge_angle(*, image: np.ndarray, green: np.ndarray, yellow: np.ndarray) -> dict[str, float] | None:
    import cv2

    if int(green.sum()) < 40 or int(yellow.sum()) < 40:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    green_values = value[green.astype(bool)]
    if len(green_values) < 40:
        return None
    # In the MuJoCo cube material, the top face is the brighter green region.
    # Use an adaptive cutoff so lighting changes do not hard-code one RGB value.
    threshold = max(50.0, float(np.percentile(green_values, 58.0)))
    top = ((green > 0) & (value >= threshold)).astype("uint8")
    kernel = np.ones((3, 3), np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 80.0]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True).reshape(-1, 2)
    if len(polygon) < 4:
        polygon = cv2.convexHull(contour).reshape(-1, 2)
    if len(polygon) < 3:
        return None

    ys, xs = np.nonzero(yellow)
    if len(xs) < 40:
        return None
    distance_to_top = cv2.distanceTransform((1 - top).astype("uint8"), cv2.DIST_L2, 3)
    nearest = int(np.argmin(distance_to_top[ys, xs]))
    contact_point = np.asarray([xs[nearest], ys[nearest]], dtype=float)

    best: tuple[float, float] | None = None
    for index in range(len(polygon)):
        start = polygon[index].astype(float)
        end = polygon[(index + 1) % len(polygon)].astype(float)
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq < 25.0:
            continue
        t = float(np.clip(np.dot(contact_point - start, segment) / length_sq, 0.0, 1.0))
        closest = start + t * segment
        distance = float(np.linalg.norm(contact_point - closest))
        angle = _angle_mod_180(math.degrees(math.atan2(float(segment[1]), float(segment[0]))))
        if best is None or distance < best[0]:
            best = (distance, angle)
    if best is None:
        return None
    return {
        "cube_top_contact_edge_angle_deg": float(best[1]),
        "contact_edge_distance_px": float(best[0]),
        "top_face_pixels": float(top.sum()),
    }


def _rect_edge_angle(mask: np.ndarray) -> float | None:
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    angle = float(rect[2])
    if width < height:
        angle += 90.0
    return _angle_mod_180(angle)


def _jaw_axis_angle(mask: np.ndarray) -> float | None:
    import cv2

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    angles: list[float] = []
    components: list[tuple[int, int]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 80:
            components.append((area, label))
    for _area, label in sorted(components, reverse=True)[:4]:
        angle = _pca_angle(labels == label)
        if angle is not None:
            angles.append(angle)
    if not angles:
        return None
    cos2 = float(np.mean([math.cos(math.radians(2.0 * angle)) for angle in angles]))
    sin2 = float(np.mean([math.sin(math.radians(2.0 * angle)) for angle in angles]))
    return _angle_mod_180(0.5 * math.degrees(math.atan2(sin2, cos2)))


def _pca_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    points = np.column_stack([xs.astype(float), ys.astype(float)])
    points -= points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points, full_matrices=False)
    return _angle_mod_180(math.degrees(math.atan2(float(vh[0, 1]), float(vh[0, 0]))))


def _angle_mod_180(angle: float) -> float:
    return float((angle + 180.0) % 180.0)


def _angle_diff(left: float, right: float) -> float:
    delta = abs(_angle_mod_180(left) - _angle_mod_180(right)) % 180.0
    return float(min(delta, 180.0 - delta))


if __name__ == "__main__":
    main()
