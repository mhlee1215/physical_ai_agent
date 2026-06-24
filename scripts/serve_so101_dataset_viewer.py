#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # myCobot JSONL datasets do not require pyarrow.
    pq = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


DATASET_CONTRACT = Path("configs/so101/training_datasets/dataset_contract.json")
SKILL_DATASET_CONTRACT = Path("configs/so101/training_datasets/skill_dataset_contract.json")
DATASETS: dict[str, Path] = {}
OFFICIAL_DATASET_SPLITS: list[str] = []
ARCHIVED_DATASET_SPLITS: list[str] = []
TEMP_DATASET_PATTERNS = [
    "smoke_*",
    "*diverse*",
    "*shape*",
    "*fixed_jaw*preview*",
]
CAMERA_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MYCOBOT_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
    "gripper_controller",
]
SO101_PLATFORM = "so101"
SO101_PLATFORM_LABEL = "SO101"
MYCOBOT_PLATFORM = "mycobot"
MYCOBOT_PLATFORM_LABEL = "MyCobot"


def _contract_dataset_roots(repo_root: Path) -> dict[str, Path]:
    contract_path = repo_root / DATASET_CONTRACT
    if not contract_path.exists():
        return {}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for dataset_name, dataset in contract.get("datasets", {}).items():
        for split_name, split in (("train", dataset.get("train")), ("validation", dataset.get("validation"))):
            if not isinstance(split, dict):
                continue
            suffix = "val" if split_name == "validation" else "train"
            roots[f"{dataset_name}_{suffix}"] = Path(split["root"])
    return roots


def _skill_dataset_roots(repo_root: Path) -> dict[str, Path]:
    contract_path = repo_root / SKILL_DATASET_CONTRACT
    if not contract_path.exists():
        return {}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for dataset_name, dataset in contract.get("datasets", {}).items():
        for split_name, split in (("train", dataset.get("train")), ("validation", dataset.get("validation"))):
            if not isinstance(split, dict):
                continue
            suffix = "val" if split_name == "validation" else "train"
            roots[f"{dataset_name}_{suffix}"] = Path(split["root"])
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a lightweight robot dataset browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_index_html())
                return
            if parsed.path == "/api/datasets":
                self._send_json(_datasets_payload(repo_root))
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                split = _query_str(query, "split", "picklift_train")
                episode = int(_query_str(query, "episode", "0"))
                frame = int(_query_str(query, "frame", "0"))
                self._send_json(_frame_payload(repo_root, split, episode, frame))
                return
            self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[dataset-viewer] {self.address_string()} {fmt % args}", flush=True)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dataset-viewer] serving http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


def _query_str(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def _datasets_payload(repo_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    official_roots = _official_dataset_roots(repo_root)
    official_items = [
        _dataset_catalog_item(repo_root, split, root, category="official")
        for split, root in official_roots.items()
    ]
    skill_roots = _skill_dataset_roots(repo_root)
    skill_items = [
        _dataset_catalog_item(repo_root, split, root, category="skill")
        for split, root in skill_roots.items()
    ]
    archived_items = [_dataset_catalog_item(repo_root, split, DATASETS[split], category="archived") for split in ARCHIVED_DATASET_SPLITS]
    archived_visible_items = [item for item in archived_items if item["status"] == "available"]
    temporary_items = [
        _dataset_catalog_item(repo_root, split, path, category="temporary")
        for split, path in _discover_temporary_datasets(repo_root).items()
    ]
    mycobot_items = [
        _mycobot_dataset_catalog_item(repo_root, split, path)
        for split, path in _discover_mycobot_datasets(repo_root).items()
    ]
    for item in [*official_items, *skill_items, *archived_items, *temporary_items, *mycobot_items]:
        if item["status"] == "available":
            payload[item["name"]] = item["summary"]
    return {
        "datasets": payload,
        "dataset_groups": [
            {
                "id": "official",
                "title": "Official / current training",
                "description": "Datasets currently used by the active training/evaluation run.",
                "items": official_items,
            },
            {
                "id": "skill",
                "title": "Skill primitives / additive",
                "description": "Agentic primitive datasets generated without replacing the official full-task datasets.",
                "items": skill_items,
            },
            {
                "id": "temporary",
                "title": "Temporary / recently generated",
                "description": "Smoke or experimental datasets generated while testing new object/grasp variants.",
                "items": temporary_items,
            },
            {
                "id": "mycobot",
                "title": "myCobot teacher POC",
                "description": "myCobot 320 adaptive-gripper JSONL teacher datasets generated from MuJoCo gates.",
                "items": mycobot_items,
            },
            {
                "id": "archived",
                "title": "Archived official",
                "description": "Older stable datasets kept for comparison.",
                "items": archived_visible_items,
            },
        ],
        "camera_view_note": "train and validation use camera1, camera2, camera3 at the stored dataset resolution.",
    }


def _dataset_catalog_item(repo_root: Path, split: str, root: Path, *, category: str) -> dict[str, Any]:
    resolved = _resolve_dataset_path(repo_root, root)
    base = {
        "name": split,
        "root": str(resolved),
        "category": category,
        "platform": SO101_PLATFORM,
        "platform_label": SO101_PLATFORM_LABEL,
    }
    try:
        dataset = _dataset(repo_root, split)
    except Exception as exc:  # noqa: BLE001 - dashboard should show incomplete datasets instead of failing.
        return {
            **base,
            "status": "incomplete" if resolved.exists() else "missing",
            "detail": str(exc),
            "size_bytes": _dir_size(resolved),
            "size_human": _format_bytes(_dir_size(resolved)),
        }
    summary = _dataset_summary(split, dataset)
    if int(summary.get("episodes") or 0) <= 0 or int(summary.get("frames") or 0) <= 0:
        return {
            **base,
            "status": "incomplete",
            "detail": "dataset has metadata but no completed episodes/frames",
            "summary": summary,
            **summary,
        }
    return {
        **base,
        "status": "available",
        "detail": "ready",
        "summary": summary,
        **summary,
    }


def _dataset_summary(split: str, dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_format": "lerobot_parquet",
        "platform": SO101_PLATFORM,
        "platform_label": SO101_PLATFORM_LABEL,
        "root": str(dataset["root"]),
        "name": split,
        "episodes": dataset["info"]["total_episodes"],
        "frames": dataset["info"]["total_frames"],
        "fps": dataset["info"].get("fps"),
        "size_bytes": dataset["size_bytes"],
        "size_human": _format_bytes(dataset["size_bytes"]),
        "data_bytes": dataset["data_bytes"],
        "data_human": _format_bytes(dataset["data_bytes"]),
        "image_bytes": dataset["image_bytes"],
        "image_human": _format_bytes(dataset["image_bytes"]),
        "features": dataset["camera_keys"],
        "image_shapes": {
            key: dataset["info"]["features"][key]["shape"] for key in dataset["camera_keys"]
        },
        "episode_lengths": dataset["episode_lengths"],
    }


def _discover_temporary_datasets(repo_root: Path) -> dict[str, Path]:
    env_roots = _parse_dataset_env("SO101_TEMP_DATASETS")
    discovered: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for name, path in env_roots.items():
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        discovered[name] = path
        seen_paths.add(resolved)
    if os.environ.get("SO101_SHOW_TEMP_DATASETS", "").strip() not in {"1", "true", "yes"}:
        return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))

    roots = [
        repo_root / "_workspace" / "so101_lerobot",
        Path("/workspace/physical-ai/so101_lerobot"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for pattern in TEMP_DATASET_PATTERNS:
            for path in root.glob(pattern):
                if not path.is_dir():
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                if path.name in {root.name for root in DATASETS.values()}:
                    continue
                split = _unique_split_name("tmp_" + _slug(path.name), discovered)
                discovered[split] = path
                seen_paths.add(resolved)
    return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))


def _discover_mycobot_datasets(repo_root: Path) -> dict[str, Path]:
    discovered = _parse_dataset_env("MYCOBOT_TEMP_DATASETS")
    seen_paths = {path.resolve() for path in discovered.values()}
    root = repo_root / "_workspace" / "mycobot_teacher_datasets"
    if root.exists():
        for manifest in root.glob("*/manifest.json"):
            path = manifest.parent
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            discovered[_unique_split_name("mycobot_" + _slug(path.name), discovered)] = path
            seen_paths.add(resolved)
    return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))


def _mycobot_dataset_catalog_item(repo_root: Path, split: str, root: Path) -> dict[str, Any]:
    resolved = _resolve_dataset_path(repo_root, root)
    base = {
        "name": split,
        "root": str(resolved),
        "category": "temporary",
        "platform": MYCOBOT_PLATFORM,
        "platform_label": MYCOBOT_PLATFORM_LABEL,
    }
    try:
        dataset = _mycobot_dataset(resolved)
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "incomplete" if resolved.exists() else "missing",
            "detail": str(exc),
            "size_bytes": _dir_size(resolved),
            "size_human": _format_bytes(_dir_size(resolved)),
        }
    summary = _mycobot_dataset_summary(split, dataset)
    return {
        **base,
        "status": "available" if summary["episodes"] > 0 and summary["frames"] > 0 else "incomplete",
        "detail": "previewable teacher dataset; not LeRobot/SmolVLA training-ready yet",
        "summary": summary,
        **summary,
    }


def _official_dataset_roots(repo_root: Path) -> dict[str, Path]:
    env_roots = _parse_dataset_env("SO101_OFFICIAL_DATASETS")
    if env_roots:
        return {split: path.resolve() for split, path in env_roots.items()}
    return {split: _resolve_dataset_path(repo_root, root) for split, root in _contract_dataset_roots(repo_root).items()}


def _parse_dataset_env(name: str) -> dict[str, Path]:
    value = os.environ.get(name, "").strip()
    if not value:
        return {}
    rows = {}
    for index, item in enumerate(value.split(",")):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            split, path = item.split("=", 1)
            rows[_slug(split)] = Path(path).expanduser()
        else:
            path = Path(item).expanduser()
            rows[_unique_split_name("tmp_" + _slug(path.name or f"dataset_{index}"), rows)] = path
    return rows


def _unique_split_name(name: str, existing: dict[str, Path]) -> str:
    candidate = name
    suffix = 2
    while candidate in DATASETS or candidate in existing:
        candidate = f"{name}_{suffix}"
        suffix += 1
    return candidate


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "dataset"


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _frame_payload(repo_root: Path, split: str, episode: int, frame: int) -> dict[str, Any]:
    mycobot_roots = _discover_mycobot_datasets(repo_root)
    if split in mycobot_roots:
        return _mycobot_frame_payload(_resolve_dataset_path(repo_root, mycobot_roots[split]), split, episode, frame)
    dataset = _dataset(repo_root, split)
    episodes = dataset["episodes"]
    if episode < 0 or episode >= len(episodes):
        raise ValueError(f"episode out of range: {episode}")
    meta = episodes[episode]
    frame = max(0, min(frame, int(meta["length"]) - 1))
    row_index = int(meta["dataset_from_index"]) + frame
    table = dataset["table"]
    row = table.slice(row_index, 1).to_pydict()
    images = {}
    for camera_key in dataset["camera_keys"]:
        image_struct = row[camera_key][0]
        images[camera_key] = "data:image/png;base64," + base64.b64encode(image_struct["bytes"]).decode("ascii")
    state = [float(v) for v in row["observation.state"][0]]
    action = [float(v) for v in row["action"][0]]
    return {
        "split": split,
        "episode": episode,
        "frame": frame,
        "episode_length": int(meta["length"]),
        "row_index": row_index,
        "timestamp": float(row["timestamp"][0]),
        "task": meta["tasks"][0] if meta.get("tasks") else "",
        "images": images,
        "state": dict(zip(JOINT_NAMES, state, strict=True)),
        "action": dict(zip(JOINT_NAMES, action, strict=True)),
    }


def _mycobot_dataset(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "mycobot_jsonl_v1":
        raise ValueError(f"unsupported myCobot dataset format: {manifest.get('format')}")
    summaries = manifest.get("episode_summaries") or []
    return {
        "root": root,
        "manifest": manifest,
        "episode_lengths": [int(row["frames"]) for row in summaries],
        "size_bytes": _dir_size(root),
    }


def _mycobot_dataset_summary(split: str, dataset: dict[str, Any]) -> dict[str, Any]:
    manifest = dataset["manifest"]
    episode_summaries = manifest.get("episode_summaries") or []
    rendered_frames = sum(int(row.get("rendered_frames") or 0) for row in episode_summaries)
    return {
        "type": "mycobot_jsonl",
        "dataset_format": "mycobot_jsonl_v1",
        "platform": MYCOBOT_PLATFORM,
        "platform_label": MYCOBOT_PLATFORM_LABEL,
        "root": str(dataset["root"]),
        "name": split,
        "episodes": int(manifest.get("episodes") or 0),
        "frames": int(manifest.get("frames") or 0),
        "fps": manifest.get("fps"),
        "size_bytes": dataset["size_bytes"],
        "size_human": _format_bytes(dataset["size_bytes"]),
        "data_bytes": _dir_size(dataset["root"] / "episodes"),
        "data_human": _format_bytes(_dir_size(dataset["root"] / "episodes")),
        "image_bytes": _dir_size(dataset["root"] / "frames"),
        "image_human": _format_bytes(_dir_size(dataset["root"] / "frames")),
        "features": ["render"],
        "image_shapes": {"render": [240, 320, 3]},
        "episode_lengths": dataset["episode_lengths"],
        "rendered_frames": rendered_frames,
        "failed_episodes": manifest.get("failed_episodes") or [],
        "robot_model": manifest.get("robot_model") or "mycobot_320",
        "gripper": manifest.get("gripper") or "adaptive",
        "gate": manifest.get("gate") or "gate8",
        "training_ready": False,
    }


def _mycobot_frame_payload(root: Path, split: str, episode: int, frame: int) -> dict[str, Any]:
    dataset = _mycobot_dataset(root)
    lengths = dataset["episode_lengths"]
    if episode < 0 or episode >= len(lengths):
        raise ValueError(f"episode out of range: {episode}")
    frame = max(0, min(frame, lengths[episode] - 1))
    episode_path = root / "episodes" / f"episode_{episode:04d}.jsonl"
    row = _jsonl_row(episode_path, frame)
    image_path = ""
    images = {}
    render_path = row.get("observation", {}).get("images", {}).get("render")
    if not render_path:
        render_path = _nearest_mycobot_render_path(root, episode, frame, lengths[episode])
    if render_path:
        image_path = str(render_path)
        image_bytes = (root / render_path).read_bytes()
        images["render"] = "data:image/bmp;base64," + base64.b64encode(image_bytes).decode("ascii")
    state_values = [float(value) for value in row.get("observation", {}).get("state", [])]
    action_values = [float(value) for value in row.get("action", [])]
    return {
        "split": split,
        "episode": episode,
        "frame": frame,
        "episode_length": lengths[episode],
        "row_index": frame,
        "timestamp": float(row.get("timestamp") or 0.0),
        "task": row.get("task", ""),
        "prompt": row.get("prompt") or row.get("task", ""),
        "phase": row.get("phase", ""),
        "images": images,
        "image_path": image_path,
        "state": dict(zip(MYCOBOT_JOINT_NAMES, state_values, strict=False)),
        "action": dict(zip(MYCOBOT_JOINT_NAMES, action_values, strict=False)),
        "info": row.get("info", {}),
    }


def _jsonl_row(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            if row_index == index:
                return json.loads(line)
    raise ValueError(f"frame out of range: {index}")


def _nearest_mycobot_render_path(root: Path, episode: int, frame: int, length: int) -> str:
    frame_dir = root / "frames" / f"episode_{episode:04d}"
    for offset in range(length):
        for candidate in (frame - offset, frame + offset):
            if candidate < 0 or candidate >= length:
                continue
            image = frame_dir / f"frame_{candidate:04d}.bmp"
            if image.exists():
                return str(image.relative_to(root))
    return ""


@lru_cache(maxsize=4)
def _dataset(repo_root: Path, split: str) -> dict[str, Any]:
    if pq is None:
        raise RuntimeError("pyarrow is required for SO101 LeRobot parquet datasets")
    roots = _dataset_roots(repo_root)
    if split not in roots:
        raise ValueError(f"unknown split: {split}")
    root = roots[split]
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_keys = [key for key in CAMERA_KEYS if key in info["features"]]
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    table = pq.read_table([str(path) for path in data_files])
    episodes_table = pq.read_table([str(path) for path in episode_files])
    episodes = _rows(episodes_table.to_pydict())
    return {
        "root": root,
        "info": info,
        "camera_keys": camera_keys,
        "table": table,
        "episodes": episodes,
        "episode_lengths": [int(row["length"]) for row in episodes],
        "size_bytes": _dir_size(root),
        "data_bytes": sum(path.stat().st_size for path in data_files),
        "image_bytes": _dir_size(root / "images"),
    }


def _dataset_roots(repo_root: Path) -> dict[str, Path]:
    roots = {split: _resolve_dataset_path(repo_root, root) for split, root in DATASETS.items()}
    roots.update(_official_dataset_roots(repo_root))
    roots.update({split: _resolve_dataset_path(repo_root, root) for split, root in _skill_dataset_roots(repo_root).items()})
    roots.update({split: path.resolve() for split, path in _discover_temporary_datasets(repo_root).items()})
    return roots


def _resolve_dataset_path(repo_root: Path, root: Path) -> Path:
    path = root if root.is_absolute() else repo_root / root
    return path.resolve()


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Experiment Manager</title>
  <style>
    :root { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #16181d; background: #f6f7f9; }
    body { margin: 0; }
    header { background: #fff; border-bottom: 1px solid #d9dee8; padding: 14px 18px; }
    h1 { margin: 0; font-size: 20px; }
    main { padding: 16px 18px; display: grid; gap: 14px; }
    section { background: #fff; border: 1px solid #d9dee8; border-radius: 8px; padding: 12px; }
    .controls { display: grid; grid-template-columns: 150px 220px 1fr 1fr auto auto auto 100px; gap: 10px; align-items: end; }
    label { display: grid; gap: 4px; font-size: 12px; color: #596273; }
    select, input, button { font: inherit; border: 1px solid #cfd6e3; border-radius: 6px; padding: 7px 9px; background: #fff; }
    button { cursor: pointer; font-weight: 650; }
    .cameras { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    figure { margin: 0; }
    figcaption { font-size: 12px; color: #596273; margin-bottom: 5px; }
    img { width: 100%; image-rendering: pixelated; border: 1px solid #d9dee8; border-radius: 6px; background: #111; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: right; padding: 6px 5px; border-bottom: 1px solid #edf0f5; font-variant-numeric: tabular-nums; }
    th:first-child, td:first-child { text-align: left; }
    .meta { color: #596273; font-size: 13px; }
    @media (max-width: 900px) { .controls, .cameras { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>Robot Experiment Manager</h1></header>
  <main>
    <section>
      <div class="controls">
        <label>Platform<select id="platform"></select></label>
        <label>Dataset<select id="split"></select></label>
        <label>Episode<input id="episode" type="range" min="0" max="0" value="0"></label>
        <label>Frame<input id="frame" type="range" min="0" max="0" value="0"></label>
        <button id="play">Play</button>
        <label>FPS<select id="fps"><option value="30" selected>30</option><option value="24">24</option><option value="12">12</option><option value="6">6</option></select></label>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      </div>
      <p id="meta" class="meta"></p>
    </section>
    <section class="cameras" id="cameras"></section>
    <section>
      <table>
        <thead><tr><th>Joint</th><th>State</th><th>Action</th></tr></thead>
        <tbody id="jointRows"></tbody>
      </table>
    </section>
  </main>
  <script>
    let datasets = {};
    const platform = document.getElementById("platform");
    const split = document.getElementById("split");
    const episode = document.getElementById("episode");
    const frame = document.getElementById("frame");
    const play = document.getElementById("play");
    const fps = document.getElementById("fps");
    const meta = document.getElementById("meta");
    const cameras = document.getElementById("cameras");
    const jointRows = document.getElementById("jointRows");
    const fmt = value => Number(value).toFixed(4);
    let timer = null;
    let loading = false;
    const knownPlatforms = [
      { id: "so101", label: "SO101" },
      { id: "mycobot", label: "MyCobot" },
    ];

    async function init() {
      const payload = await fetch("/api/datasets").then(r => r.json());
      datasets = payload.datasets;
      syncPlatformOptions();
      syncDatasetOptions();
      syncEpisodeRange();
      await loadFrame();
    }

    function syncPlatformOptions() {
      const counts = {};
      Object.values(datasets).forEach(data => {
        const id = data.platform || "so101";
        counts[id] = (counts[id] || 0) + 1;
      });
      platform.innerHTML = knownPlatforms.map(({ id, label }) => {
        const count = counts[id] || 0;
        return `<option value="${id}">${label}${count ? ` (${count})` : ""}</option>`;
      }).join("");
      const firstAvailable = knownPlatforms.find(({ id }) => counts[id]);
      platform.value = firstAvailable ? firstAvailable.id : knownPlatforms[0].id;
    }

    function syncDatasetOptions() {
      const selected = split.value;
      const names = Object.keys(datasets).filter(name => (datasets[name].platform || "so101") === platform.value);
      split.innerHTML = names.map(name => `<option value="${name}">${name}</option>`).join("");
      if (names.includes(selected)) split.value = selected;
      if (!names.length) {
        meta.textContent = "No datasets available for this platform.";
        cameras.innerHTML = "";
        jointRows.innerHTML = "";
      }
    }

    function syncEpisodeRange() {
      const data = datasets[split.value];
      if (!data) return;
      if (data.fps) fps.value = String(data.fps);
      episode.max = String(data.episodes - 1);
      episode.value = String(Math.min(Number(episode.value), data.episodes - 1));
      syncFrameRange();
    }

    function syncFrameRange() {
      const data = datasets[split.value];
      if (!data) return;
      const length = data.episode_lengths[Number(episode.value)];
      frame.max = String(length - 1);
      frame.value = String(Math.min(Number(frame.value), length - 1));
    }

    async function loadFrame() {
      if (loading) return;
      if (!datasets[split.value]) return;
      loading = true;
      syncFrameRange();
      const url = `/api/frame?split=${split.value}&episode=${episode.value}&frame=${frame.value}`;
      const row = await fetch(url).then(r => r.json()).finally(() => { loading = false; });
      const data = datasets[row.split];
      const format = data.dataset_format || data.type || "";
      const readiness = data.training_ready === false ? "preview teacher dataset" : "training dataset";
      meta.textContent = `${data.platform_label || "Robot"} | ${row.split} | ${format} | ${readiness} | episode ${row.episode}/${data.episodes - 1} | frame ${row.frame}/${row.episode_length - 1} | row ${row.row_index} | t=${row.timestamp.toFixed(3)}s | ${row.task}`;
      cameras.innerHTML = Object.entries(row.images).map(([name, src]) => `
        <figure>
          <figcaption>${name}</figcaption>
          <img src="${src}" alt="${name}">
        </figure>
      `).join("");
      jointRows.innerHTML = Object.keys(row.state).map(joint => `
        <tr><td>${joint}</td><td>${fmt(row.state[joint])}</td><td>${fmt(row.action[joint])}</td></tr>
      `).join("");
    }

    platform.addEventListener("change", () => { syncDatasetOptions(); syncEpisodeRange(); loadFrame(); });
    split.addEventListener("change", () => { syncEpisodeRange(); loadFrame(); });
    episode.addEventListener("input", () => { frame.value = "0"; loadFrame(); });
    frame.addEventListener("input", loadFrame);
    play.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        play.textContent = "Play";
        return;
      }
      play.textContent = "Pause";
      timer = setInterval(() => {
        if (Number(frame.value) >= Number(frame.max)) frame.value = "0";
        else frame.value = String(Number(frame.value) + 1);
        loadFrame();
      }, Math.max(16, 1000 / Number(fps.value || 12)));
    });
    document.getElementById("prev").addEventListener("click", () => { frame.value = String(Math.max(0, Number(frame.value) - 1)); loadFrame(); });
    document.getElementById("next").addEventListener("click", () => { frame.value = String(Math.min(Number(frame.max), Number(frame.value) + 1)); loadFrame(); });
    init();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
