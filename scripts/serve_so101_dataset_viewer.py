#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

from physical_ai_agent.so101_dataset_registry import (
    DATASET_RECIPE_DIR,
    registered_dataset_roots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


DATASET_CONTRACT = Path("configs/so101/training_datasets/dataset_contract.json")
SKILL_DATASET_CONTRACT = Path("configs/so101/training_datasets/skill_dataset_contract.json")
TRAINING_CONFIGS = [
    Path("configs/so101/training_datasets/qwen_edge_primitives.json"),
    Path("configs/so101/training/qwen_edge_primitives.json"),
    Path("configs/so101/training/pick_photoreal.json"),
    Path("configs/so101/training/grip_the_cube_v2.json"),
]
DATASET_GENERATION_CONFIG_DIR = DATASET_RECIPE_DIR
INTERACTIVE_RUN_ROOT = Path("_workspace/so101_interactive_sim/runs")
DEFAULT_VALID_MASK_CHECKPOINT = Path("_workspace/so101_valid_mask_head/qwen_edge_primitives/valid_mask_head.pt")
LOOP_ANALYZER_ROUTE = "/loop-analyzer"
LOOP_ANALYZER_MEDIA_JOBS: dict[str, dict[str, Any]] = {}
LOOP_ANALYZER_MEDIA_JOB_LOCK = threading.Lock()
DATASETS_PAYLOAD_CACHE_SECONDS = 5.0
DATASETS_PAYLOAD_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
DATASETS_PAYLOAD_LOCK = threading.Lock()
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
SO101_CAMERA_CONTRACT = {
    "observation.images.camera1": "egocentric_cam",
    "observation.images.camera2": "wrist_cam",
    "observation.images.camera3": "wrist_cam duplicate",
}
PHOTO_REAL_PREVIEW_ROOT = Path("docs/research/2026_07_04/so101_photoreal_render_pipeline")
PHOTO_REAL_PREVIEW_DIRS = {
    "pick_cube_train50_ego_wrist_256_seed98200": PHOTO_REAL_PREVIEW_ROOT / "so101_pick_cube_train5episodes",
}
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
            roots[f"{dataset_name}_{suffix}"] = _resolve_contract_dataset_root(
                repo_root,
                dataset_name=str(dataset_name),
                split_name=split_name,
                configured_root=Path(split["root"]),
            )
    return roots


def _skill_dataset_roots(repo_root: Path) -> dict[str, Path]:
    contract_path = repo_root / SKILL_DATASET_CONTRACT
    if not contract_path.exists():
        return {}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for dataset_name, dataset in contract.get("datasets", {}).items():
        for split_name, split in (
            ("train", dataset.get("train")),
            ("validation", dataset.get("validation")),
            ("loop_validation", dataset.get("loop_validation")),
        ):
            if not isinstance(split, dict):
                continue
            suffix = {
                "train": "train",
                "validation": "val",
                "loop_validation": "loop_val",
            }[split_name]
            roots[f"{dataset_name}_{suffix}"] = _resolve_contract_dataset_root(
                repo_root,
                dataset_name=str(dataset_name),
                split_name=split_name,
                configured_root=Path(split["root"]),
            )
    return roots


def _resolve_contract_dataset_root(
    repo_root: Path,
    *,
    dataset_name: str,
    split_name: str,
    configured_root: Path,
) -> Path:
    configured = _resolve_dataset_path(repo_root, configured_root)
    if configured.exists():
        return configured

    # Older official datasets were moved into the durable Hugging Face upload
    # staging tree. Keep the contract name/split authoritative when locating
    # that preserved copy instead of silently dropping it from the catalog.
    fallback_roots = (
        repo_root / "_workspace" / "hf_upload" / "so101-nexus-sim-dataset" / "datasets",
        repo_root / "_workspace" / "hf_datasets" / "mhlee1215__so101-nexus-sim-dataset" / "datasets",
    )
    for fallback_root in fallback_roots:
        candidate = fallback_root / dataset_name / split_name
        if candidate.exists():
            return candidate.resolve()
    return configured


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a lightweight SO101 LeRobot dataset browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    server = ReusableThreadingHTTPServer((args.host, args.port), make_handler(repo_root))
    print(f"[dataset-viewer] serving http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


def make_handler(repo_root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_index_html())
                return
            if parsed.path in {f"{LOOP_ANALYZER_ROUTE}/", f"{LOOP_ANALYZER_ROUTE}/index.html"}:
                self._send_html(_loop_analyzer_index_html(repo_root))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/api/loop-tests":
                self._send_json(_loop_analyzer_loop_tests_payload(repo_root))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/api/loop-test":
                query = parse_qs(parsed.query)
                loop_test_id = _query_str(query, "id", "")
                self._send_json(_loop_analyzer_loop_test_detail_payload(repo_root, loop_test_id))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/api/generate-media-status":
                query = parse_qs(parsed.query)
                loop_test_id = _query_str(query, "loop", "") or None
                self._send_json(_loop_analyzer_generate_media_status(repo_root, loop_test_id=loop_test_id))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/artifact":
                query = parse_qs(parsed.query)
                self._send_loop_analyzer_artifact(repo_root, Path(_query_str(query, "path", "")))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/vendor/chart.umd.min.js":
                self._send_file(repo_root / "third_party" / "chartjs" / "chart.umd.min.js", content_type="application/javascript; charset=utf-8")
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
            if parsed.path == "/api/loop-tests":
                self._send_json(_loop_tests_payload(repo_root))
                return
            if parsed.path == "/api/loop-frame":
                query = parse_qs(parsed.query)
                export_id = _query_str(query, "export", "")
                loop_id = _query_str(query, "loop", "")
                episode = int(_query_str(query, "episode", "0"))
                step = int(_query_str(query, "step", "0"))
                self._send_json(_loop_frame_payload(repo_root, export_id, loop_id, episode, step))
                return
            if parsed.path == "/api/simulator/config":
                self._send_json(_simulator_config_payload(repo_root))
                return
            if parsed.path == "/api/training/runs":
                self._send_json(_training_runs_payload(repo_root))
                return
            if parsed.path == "/api/training/run":
                query = parse_qs(parsed.query)
                training_id = _query_str(query, "id", "")
                if not training_id:
                    self._send_json({"error": "missing id"})
                    return
                self._send_json(_training_run_detail_payload(repo_root, training_id))
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/simulator/run":
                self._send_json(_run_interactive_simulator(repo_root, self._read_json_body()))
                return
            if parsed.path == f"{LOOP_ANALYZER_ROUTE}/api/generate-media":
                query = parse_qs(parsed.query)
                loop_test_id = _query_str(query, "loop", "") or None
                self._send_json(_loop_analyzer_start_generate_media(repo_root, loop_test_id=loop_test_id))
                return
            self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[dataset-viewer] {self.address_string()} {fmt % args}", flush=True)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

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

        def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_loop_analyzer_artifact(self, repo_root: Path, requested_path: Path) -> None:
            export_dir = _loop_analyzer_default_export_dir(repo_root)
            if export_dir is None:
                self.send_error(404)
                return
            try:
                artifact_path = requested_path.resolve()
                artifact_path.relative_to(export_dir)
            except (OSError, ValueError):
                self.send_error(403)
                return
            self._send_file(artifact_path)

    return Handler


def _query_str(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def _datasets_payload(repo_root: Path) -> dict[str, Any]:
    cache_key = str(repo_root.resolve())
    now = time.monotonic()
    cached = DATASETS_PAYLOAD_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < DATASETS_PAYLOAD_CACHE_SECONDS:
        return cached[1]
    with DATASETS_PAYLOAD_LOCK:
        now = time.monotonic()
        cached = DATASETS_PAYLOAD_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < DATASETS_PAYLOAD_CACHE_SECONDS:
            return cached[1]
        payload = _build_datasets_payload(repo_root)
        DATASETS_PAYLOAD_CACHE[cache_key] = (now, payload)
        return payload


def _build_datasets_payload(repo_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    official_roots = _official_dataset_roots(repo_root)
    skill_roots = _skill_dataset_roots(repo_root)
    recipe_roots = _generation_recipe_dataset_roots(repo_root)
    recipe_paths = {_resolve_dataset_path(repo_root, root) for root in recipe_roots.values()}
    official_items = [
        _dataset_catalog_item(repo_root, split, root, category="official")
        for split, root in official_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    skill_items = [
        _dataset_catalog_item(repo_root, split, root, category="skill")
        for split, root in skill_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    photoreal_items = [
        _so101_photoreal_dataset_catalog_item(repo_root, split, path)
        for split, path in _discover_so101_photoreal_datasets(repo_root).items()
        if _resolve_dataset_path(repo_root, path) not in recipe_paths
    ]
    photoreal_items.extend(
        _dataset_catalog_item(repo_root, split, path, category="photoreal")
        for split, path in _discover_so101_photoreal_lerobot_datasets(repo_root).items()
        if _resolve_dataset_path(repo_root, path) not in recipe_paths
    )
    generated_items: list[dict[str, Any]] = []
    official_paths = {_resolve_dataset_path(repo_root, root) for root in official_roots.values()}
    skill_paths = {_resolve_dataset_path(repo_root, root) for root in skill_roots.values()}
    for split, root in recipe_roots.items():
        resolved = _resolve_dataset_path(repo_root, root)
        if (resolved / "photoreal_lerobot_manifest.json").is_file():
            category = "photoreal"
        else:
            category = (
                "official"
                if resolved in official_paths
                else "skill"
                if resolved in skill_paths
                else "generated"
            )
        item = _dataset_catalog_item(repo_root, split, root, category=category)
        if category == "official":
            official_items.append(item)
        elif category == "skill":
            skill_items.append(item)
        elif category == "photoreal":
            photoreal_items.append(item)
        else:
            generated_items.append(item)
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
    for item in [
        *official_items,
        *skill_items,
        *generated_items,
        *archived_items,
        *temporary_items,
        *photoreal_items,
        *mycobot_items,
    ]:
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
                "id": "generated",
                "title": "Generated / recipe-backed",
                "description": "Completed datasets declared by reproducible dataset-generation recipes.",
                "items": generated_items,
            },
            {
                "id": "temporary",
                "title": "Temporary / recently generated",
                "description": "Smoke or experimental datasets generated while testing new object/grasp variants.",
                "items": temporary_items,
            },
            {
                "id": "photoreal",
                "title": "Photoreal datasets",
                "description": "SO101 datasets whose stored image frames are photoreal renders.",
                "items": photoreal_items,
            },
            {
                "id": "mycobot",
                "title": "myCobot teacher POC",
                "description": "myCobot preview teacher datasets. These are not LeRobot/SmolVLA training-ready until exported.",
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


def _training_manager_module() -> Any:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import serve_so101_training_manager as training_manager  # type: ignore[import-not-found]

    return training_manager


def _training_runs_payload(repo_root: Path) -> dict[str, Any]:
    manager = _training_manager_module()
    return manager._runs_payload(repo_root)  # noqa: SLF001


def _training_run_detail_payload(repo_root: Path, training_id: str) -> dict[str, Any]:
    manager = _training_manager_module()
    try:
        return manager._run_detail(repo_root, training_id)  # noqa: SLF001
    except Exception as exc:
        return {"error": str(exc)}


def _loop_analyzer_module() -> Any:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import serve_loop_test_analyzer as analyzer  # type: ignore[import-not-found]

    return analyzer


def _loop_analyzer_export_dirs(repo_root: Path) -> list[Path]:
    root = repo_root / "_workspace" / "so101_training" / "runs"
    manifests = sorted(root.glob("**/loop_test_analyzer_export/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.parent for path in manifests]


def _loop_analyzer_default_export_dir(repo_root: Path) -> Path | None:
    dirs = _loop_analyzer_export_dirs(repo_root)
    return dirs[0] if dirs else None


def _loop_analyzer_index_html(repo_root: Path) -> str:
    export_dir = _loop_analyzer_default_export_dir(repo_root)
    analyzer = _loop_analyzer_module()
    if export_dir is None:
        return """<!doctype html><html><body><h1>Loop Test Analyzer</h1><p>No loop_test_analyzer_export/manifest.json found.</p></body></html>"""
    html = analyzer._index_html()  # noqa: SLF001
    replacements = {
        'src="/vendor/chart.umd.min.js"': f'src="{LOOP_ANALYZER_ROUTE}/vendor/chart.umd.min.js"',
        'fetch("/api/loop-tests")': f'fetch("{LOOP_ANALYZER_ROUTE}/api/loop-tests")',
        'fetch(`/api/loop-test?id=${encodeURIComponent(id)}`)': f'fetch(`{LOOP_ANALYZER_ROUTE}/api/loop-test?id=${{encodeURIComponent(id)}}`)',
        'fetch(`/api/generate-media?loop=${encodeURIComponent(state.active)}`, { method: "POST" })': f'fetch(`{LOOP_ANALYZER_ROUTE}/api/generate-media?loop=${{encodeURIComponent(state.active)}}`, {{ method: "POST" }})',
        'fetch(`/api/generate-media-status${loopQuery}`)': f'fetch(`{LOOP_ANALYZER_ROUTE}/api/generate-media-status${{loopQuery}}`)',
        'return `/artifact?path=${encodeURIComponent(path)}`;': f'return `{LOOP_ANALYZER_ROUTE}/artifact?path=${{encodeURIComponent(path)}}`;',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    banner = (
        f'<div style="padding:8px 12px;background:#ecfeff;border-bottom:1px solid #99f6e4;'
        f'font:12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#115e59;">'
        f'Embedded in Experiment Manager · export_dir={export_dir}</div>'
    )
    html = html.replace('<div class="app">', banner + '<div class="app">', 1)
    return html


def _loop_analyzer_loop_tests_payload(repo_root: Path) -> dict[str, Any]:
    export_dir = _loop_analyzer_default_export_dir(repo_root)
    if export_dir is None:
        return {"export_dir": None, "schema_version": None, "summary": {}, "loop_tests": []}
    analyzer = _loop_analyzer_module()
    return analyzer._loop_tests_payload(export_dir)  # noqa: SLF001


def _loop_analyzer_loop_test_detail_payload(repo_root: Path, loop_test_id: str) -> dict[str, Any]:
    export_dir = _loop_analyzer_default_export_dir(repo_root)
    if export_dir is None:
        return {"error": "No loop analyzer export found."}
    analyzer = _loop_analyzer_module()
    return analyzer._loop_test_detail(export_dir, loop_test_id)  # noqa: SLF001


def _loop_analyzer_generate_media_status(repo_root: Path, *, loop_test_id: str | None) -> dict[str, Any]:
    export_dir = _loop_analyzer_default_export_dir(repo_root)
    if export_dir is None:
        return {"status": "idle", "error": "No loop analyzer export found."}
    analyzer = _loop_analyzer_module()
    key = str(export_dir)
    with LOOP_ANALYZER_MEDIA_JOB_LOCK:
        payload = LOOP_ANALYZER_MEDIA_JOBS.get(key)
        if payload is None:
            payload = analyzer._load_media_job_status(export_dir)  # noqa: SLF001
            LOOP_ANALYZER_MEDIA_JOBS[key] = payload
        return analyzer._status_for_loop(export_dir, dict(payload), loop_test_id=loop_test_id)  # noqa: SLF001


def _loop_analyzer_start_generate_media(repo_root: Path, *, loop_test_id: str | None) -> dict[str, Any]:
    if not loop_test_id:
        return {"status": "failed", "error": "loop query parameter is required"}
    export_dir = _loop_analyzer_default_export_dir(repo_root)
    if export_dir is None:
        return {"status": "failed", "error": "No loop analyzer export found."}
    analyzer = _loop_analyzer_module()
    if not analyzer._loop_test_exists(export_dir, loop_test_id):  # noqa: SLF001
        return {"status": "failed", "error": f"loop test not found: {loop_test_id}"}
    key = str(export_dir)
    with LOOP_ANALYZER_MEDIA_JOB_LOCK:
        existing = LOOP_ANALYZER_MEDIA_JOBS.get(key)
        if existing and existing.get("status") == "running":
            return analyzer._with_media_progress(export_dir, dict(existing), loop_test_id=loop_test_id)  # noqa: SLF001
        generation_root = analyzer._media_generation_repo_root(repo_root, export_dir)  # noqa: SLF001
        command = analyzer._media_generation_command(generation_root, export_dir, loop_test_id=loop_test_id)  # noqa: SLF001
        payload = {
            "status": "running",
            "loop_test_id": loop_test_id,
            "started_at": time.time(),
            "finished_at": None,
            "command": command,
            "repo_root": str(generation_root),
            "run_dir": str(export_dir.parent),
            "output_dir": str(export_dir),
            "server_pid": os.getpid(),
            "progress": analyzer._media_artifact_progress(export_dir, loop_test_id=loop_test_id),  # noqa: SLF001
            "stdout": "",
            "stderr": "",
        }
        LOOP_ANALYZER_MEDIA_JOBS[key] = payload
        analyzer._save_media_job_status(export_dir, payload)  # noqa: SLF001

    def run_job() -> None:
        env = os.environ.copy()
        src_path = str(generation_root / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        try:
            completed = subprocess.run(command, cwd=generation_root, env=env, text=True, capture_output=True, check=False)
            result = {
                "status": "succeeded" if completed.returncode == 0 else "failed",
                "finished_at": time.time(),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "progress": analyzer._media_artifact_progress(export_dir, loop_test_id=loop_test_id),  # noqa: SLF001
            }
        except OSError as exc:
            result = {
                "status": "failed",
                "finished_at": time.time(),
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
                "progress": analyzer._media_artifact_progress(export_dir, loop_test_id=loop_test_id),  # noqa: SLF001
            }
        with LOOP_ANALYZER_MEDIA_JOB_LOCK:
            current = LOOP_ANALYZER_MEDIA_JOBS.get(key, {})
            current.update(result)
            LOOP_ANALYZER_MEDIA_JOBS[key] = current
            analyzer._save_media_job_status(export_dir, current)  # noqa: SLF001

    threading.Thread(target=run_job, name="experiment-manager-loop-media", daemon=True).start()
    return analyzer._with_media_progress(export_dir, dict(payload), loop_test_id=loop_test_id)  # noqa: SLF001


def _dataset_catalog_item(repo_root: Path, split: str, root: Path, *, category: str) -> dict[str, Any]:
    resolved = _resolve_dataset_path(repo_root, root)
    platform = _dataset_platform(split, resolved)
    base = {
        "name": split,
        "root": str(resolved),
        "category": category,
        "platform": platform,
        "platform_label": _platform_label(platform),
    }
    try:
        dataset = _dataset_metadata(repo_root, split)
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
    platform = _dataset_platform(split, Path(dataset["root"]))
    photoreal_preview = _photoreal_preview_summary(Path(dataset["root"]))
    return {
        "dataset_format": "lerobot_parquet",
        "root": str(dataset["root"]),
        "name": split,
        "platform": platform,
        "platform_label": _platform_label(platform),
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
        "camera_contract": SO101_CAMERA_CONTRACT,
        "photoreal_preview": photoreal_preview,
    }


def _dataset_platform(split: str, root: Path) -> str:
    text = f"{split} {root}".lower()
    if "mycobot" in text or "my_cobot" in text or "my-cobot" in text:
        return "mycobot"
    return "so101"


def _platform_label(platform: str) -> str:
    return {"mycobot": "MyCobot", "so101": "SO101"}.get(platform, platform)


def _photoreal_preview_summary(dataset_root: Path) -> dict[str, Any]:
    preview_dir = _photoreal_preview_dir(dataset_root)
    if preview_dir is None:
        return {"available": False}
    frames: dict[int, list[int]] = {}
    for path in sorted(preview_dir.glob("episode_*_frame_*.png")):
        match = re.search(r"episode_(\d+)_frame_(\d+)\.png$", path.name)
        if not match:
            continue
        episode = int(match.group(1))
        frame = int(match.group(2))
        frames.setdefault(episode, []).append(frame)
    return {
        "available": bool(frames),
        "path": str(preview_dir),
        "contact_sheet": str(preview_dir / "contact_sheet.png") if (preview_dir / "contact_sheet.png").exists() else None,
        "episodes": sorted(frames),
        "frames_by_episode": {str(episode): sorted(values) for episode, values in frames.items()},
        "note": "Photoreal sidecar preview; original LeRobot camera images remain canonical policy inputs.",
    }


def _photoreal_frame_images(dataset_root: Path, *, episode: int, frame: int) -> dict[str, str]:
    preview_dir = _photoreal_preview_dir(dataset_root)
    if preview_dir is None:
        return {}
    image_path = preview_dir / f"episode_{episode:04d}_frame_{frame:04d}.png"
    if not image_path.exists():
        return {}
    return {"photoreal_sidecar": _file_data_uri(image_path)}


def _photoreal_preview_dir(dataset_root: Path) -> Path | None:
    relative = PHOTO_REAL_PREVIEW_DIRS.get(dataset_root.name)
    if relative is None:
        return None
    path = relative if relative.is_absolute() else REPO_ROOT / relative
    return path if path.exists() else None


def _file_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


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


def _discover_so101_photoreal_datasets(repo_root: Path) -> dict[str, Path]:
    discovered = _parse_dataset_env("SO101_PHOTOREAL_DATASETS")
    roots = [
        repo_root / "_workspace" / "so101_photoreal_datasets",
        REPO_ROOT / "_workspace" / "so101_photoreal_datasets",
    ]
    seen = {path.resolve() for path in discovered.values()}
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/manifest.json")):
            dataset_root = manifest_path.parent
            resolved = dataset_root.resolve()
            if resolved in seen:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("format") != "so101_photoreal_jsonl_v1":
                continue
            split = _unique_split_name("photoreal_" + _slug(dataset_root.name), discovered)
            discovered[split] = dataset_root
            seen.add(resolved)
    return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))


def _discover_so101_photoreal_lerobot_datasets(repo_root: Path) -> dict[str, Path]:
    discovered = _parse_dataset_env("SO101_PHOTOREAL_LEROBOT_DATASETS")
    roots = [
        repo_root / "_workspace" / "so101_photoreal_lerobot",
        REPO_ROOT / "_workspace" / "so101_photoreal_lerobot",
    ]
    seen = {path.resolve() for path in discovered.values()}
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/photoreal_lerobot_manifest.json")):
            dataset_root = manifest_path.parent
            resolved = dataset_root.resolve()
            if resolved in seen:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("format") != "so101_photoreal_lerobot_v1":
                continue
            split = _unique_split_name("photoreal_lerobot_" + _slug(dataset_root.name), discovered)
            discovered[split] = dataset_root
            seen.add(resolved)
    return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))


def _so101_photoreal_dataset_catalog_item(repo_root: Path, split: str, root: Path) -> dict[str, Any]:
    resolved = _resolve_dataset_path(repo_root, root)
    base = {
        "name": split,
        "root": str(resolved),
        "category": "photoreal",
        "platform": "so101",
        "platform_label": "SO101",
        "dataset_format": "so101_photoreal_jsonl_v1",
    }
    try:
        dataset = _so101_photoreal_dataset(resolved)
    except Exception as exc:  # noqa: BLE001 - dashboard should show incomplete datasets instead of failing.
        return {
            **base,
            "status": "missing",
            "detail": str(exc),
            "summary": {
                **base,
                "episodes": 0,
                "frames": 0,
                "size_bytes": _dir_size(resolved),
                "size_human": _format_bytes(_dir_size(resolved)),
            },
        }
    summary = _so101_photoreal_dataset_summary(split, dataset)
    return {
        **base,
        "status": "available" if summary["episodes"] > 0 and summary["frames"] > 0 else "incomplete",
        "detail": "photoreal image dataset",
        "summary": summary,
        **summary,
    }


def _so101_photoreal_dataset(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "so101_photoreal_jsonl_v1":
        raise ValueError(f"unsupported SO101 photoreal dataset format: {manifest.get('format')}")
    summaries = manifest.get("episode_summaries") or []
    return {
        "root": root,
        "manifest": manifest,
        "episode_lengths": [int(row.get("frames") or 0) for row in summaries],
        "size_bytes": _dir_size(root),
    }


def _so101_photoreal_dataset_summary(split: str, dataset: dict[str, Any]) -> dict[str, Any]:
    manifest = dataset["manifest"]
    image_shape = manifest.get("image_shape") or [480, 640, 3]
    image_features = [feature for feature in manifest.get("features", []) if str(feature).startswith("observation.images.")]
    return {
        "type": "so101_photoreal_jsonl",
        "dataset_format": "so101_photoreal_jsonl_v1",
        "platform": "so101",
        "platform_label": "SO101",
        "root": str(dataset["root"]),
        "name": split,
        "episodes": int(manifest.get("episodes") or len(dataset["episode_lengths"])),
        "frames": int(manifest.get("frames") or sum(dataset["episode_lengths"])),
        "fps": manifest.get("fps"),
        "size_bytes": dataset["size_bytes"],
        "size_human": _format_bytes(dataset["size_bytes"]),
        "data_bytes": _dir_size(dataset["root"] / "episodes"),
        "data_human": _format_bytes(_dir_size(dataset["root"] / "episodes")),
        "image_bytes": _dir_size(dataset["root"] / "images"),
        "image_human": _format_bytes(_dir_size(dataset["root"] / "images")),
        "features": manifest.get("features") or ["observation.images.camera1"],
        "image_shapes": {feature: image_shape for feature in image_features},
        "episode_lengths": dataset["episode_lengths"],
        "camera_contract": manifest.get("camera_contract") or {},
        "source_dataset_root": manifest.get("source_dataset_root"),
        "source_dataset_name": manifest.get("source_dataset_name"),
        "training_ready": bool(manifest.get("training_ready")),
        "note": manifest.get("note"),
    }


def _generation_recipe_dataset_roots(repo_root: Path) -> dict[str, Path]:
    return registered_dataset_roots(repo_root, existing_only=True)


def _discover_mycobot_datasets(repo_root: Path) -> dict[str, Path]:
    discovered = _parse_dataset_env("MYCOBOT_TEMP_DATASETS")
    seen_paths = {path.resolve() for path in discovered.values()}
    roots = [
        repo_root / "_workspace" / "mycobot_teacher_datasets",
        Path("/private/tmp/physical_ai_agent_mycobot_ros_poc/_workspace/mycobot_teacher_datasets"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in root.glob("*/manifest.json"):
            dataset_root = manifest_path.parent
            resolved = dataset_root.resolve()
            if resolved in seen_paths:
                continue
            split = _unique_split_name("mycobot_" + _slug(dataset_root.name), discovered)
            discovered[split] = dataset_root
            seen_paths.add(resolved)
    return dict(sorted(discovered.items(), key=lambda item: _safe_mtime(item[1]), reverse=True))


def _mycobot_dataset_catalog_item(repo_root: Path, split: str, root: Path) -> dict[str, Any]:
    resolved = _resolve_dataset_path(repo_root, root)
    base = {
        "name": split,
        "root": str(resolved),
        "category": "temporary",
        "platform": "mycobot",
        "platform_label": "MyCobot",
        "dataset_format": "mycobot_jsonl_v1",
    }
    try:
        dataset = _mycobot_dataset(resolved)
    except Exception as exc:  # noqa: BLE001 - preview datasets should not crash the dashboard.
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
    roots: dict[str, Path] = {}
    roots.update(_training_config_dataset_roots(repo_root))
    roots.update(_contract_dataset_roots(repo_root))
    return {split: _resolve_dataset_path(repo_root, root) for split, root in roots.items()}


def _training_config_dataset_roots(repo_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for relative_path in TRAINING_CONFIGS:
        path = repo_root / relative_path
        if not path.exists():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        name = _slug(str(config.get("name") or path.stem))
        dataset_config = config.get("dataset") if isinstance(config.get("dataset"), dict) else config
        train = dataset_config.get("train_dataset")
        train_datasets = dataset_config.get("train_datasets")
        validation = dataset_config.get("validation_dataset")
        loop_validation = dataset_config.get("loop_validation_dataset")
        if isinstance(train, dict) and train.get("root"):
            roots[f"{name}_train"] = Path(train["root"])
        if isinstance(train_datasets, list):
            for index, dataset in enumerate(train_datasets):
                if not isinstance(dataset, dict) or not dataset.get("root"):
                    continue
                split_name = _slug(str(dataset.get("name") or f"{name}_train_{index}"))
                roots[split_name] = Path(dataset["root"])
        if isinstance(validation, dict) and validation.get("root"):
            roots[f"{name}_val"] = Path(validation["root"])
        if isinstance(loop_validation, dict) and loop_validation.get("root"):
            roots[f"{name}_loop_val"] = Path(loop_validation["root"])
        if isinstance(validation, dict):
            sources = validation.get("hf_resolved_sources") or validation.get("hf_merge_sources")
            if isinstance(sources, list):
                for index, dataset in enumerate(sources):
                    if not isinstance(dataset, dict):
                        continue
                    root = dataset.get("root")
                    if not root:
                        hf_path = dataset.get("hf_path_in_repo")
                        if hf_path:
                            root = Path("_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset") / str(hf_path)
                    if not root:
                        continue
                    split_name = _slug(str(dataset.get("name") or f"{name}_val_{index}"))
                    roots[split_name] = Path(root)
    return roots


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
    photoreal_roots = _discover_so101_photoreal_datasets(repo_root)
    if split in photoreal_roots:
        return _so101_photoreal_frame_payload(_resolve_dataset_path(repo_root, photoreal_roots[split]), split, episode, frame)
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
    task_index = int(row["task_index"][0]) if "task_index" in row else None
    prompt = dataset["tasks"].get(task_index)
    if prompt is None:
        prompt = meta["tasks"][0] if meta.get("tasks") else ""
    photoreal_images = _photoreal_frame_images(Path(dataset["root"]), episode=episode, frame=frame)
    return {
        "split": split,
        "episode": episode,
        "frame": frame,
        "episode_length": int(meta["length"]),
        "row_index": row_index,
        "timestamp": float(row["timestamp"][0]),
        "task": prompt,
        "prompt": prompt,
        "task_index": task_index,
        "images": images,
        "camera_contract": SO101_CAMERA_CONTRACT,
        "photoreal_images": photoreal_images,
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
        "episode_lengths": [int(row.get("frames") or 0) for row in summaries],
        "size_bytes": _dir_size(root),
    }


def _so101_photoreal_frame_payload(root: Path, split: str, episode: int, frame: int) -> dict[str, Any]:
    dataset = _so101_photoreal_dataset(root)
    lengths = dataset["episode_lengths"]
    if episode < 0 or episode >= len(lengths):
        raise ValueError(f"episode out of range: {episode}")
    frame = max(0, min(frame, lengths[episode] - 1))
    episode_path = root / "episodes" / f"episode_{episode:04d}.jsonl"
    row = _jsonl_row(episode_path, frame)
    images = {}
    image_paths = {}
    for name, image_rel in (row.get("observation", {}).get("images", {}) or {}).items():
        if not image_rel:
            continue
        image_file = root / image_rel
        mime = dataset["manifest"].get("image_mime_type") or "image/png"
        key = name if str(name).startswith("observation.images.") else f"observation.images.{name}"
        image_paths[key] = str(image_rel)
        images[key] = f"data:{mime};base64," + base64.b64encode(image_file.read_bytes()).decode("ascii")
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
        "task_index": row.get("task_index"),
        "source_episode_index": row.get("source_episode_index"),
        "source_frame_index": row.get("source_frame_index"),
        "source_seed": row.get("source_seed"),
        "images": images,
        "image_paths": image_paths,
        "camera_contract": dataset["manifest"].get("camera_contract") or {},
        "state": _named_values(dataset["manifest"].get("joint_names") or JOINT_NAMES, state_values),
        "action": _named_values(dataset["manifest"].get("action_names") or JOINT_NAMES, action_values),
        "info": {
            "format": dataset["manifest"].get("format"),
            "source_dataset_root": dataset["manifest"].get("source_dataset_root"),
            "source_dataset_name": dataset["manifest"].get("source_dataset_name"),
            "training_ready": bool(dataset["manifest"].get("training_ready")),
        },
    }


def _mycobot_dataset_summary(split: str, dataset: dict[str, Any]) -> dict[str, Any]:
    manifest = dataset["manifest"]
    episode_summaries = manifest.get("episode_summaries") or []
    rendered_frames = sum(int(row.get("rendered_frames") or 0) for row in episode_summaries)
    robot = manifest.get("robot") if isinstance(manifest.get("robot"), dict) else {}
    viewer = manifest.get("viewer") if isinstance(manifest.get("viewer"), dict) else {}
    image_shape = viewer.get("image_shape") or [240, 320, 3]
    return {
        "type": "mycobot_jsonl",
        "dataset_format": "mycobot_jsonl_v1",
        "platform": "mycobot",
        "platform_label": "MyCobot",
        "root": str(dataset["root"]),
        "name": split,
        "episodes": int(manifest.get("episodes") or len(episode_summaries)),
        "frames": int(manifest.get("frames") or sum(dataset["episode_lengths"])),
        "fps": manifest.get("fps"),
        "size_bytes": dataset["size_bytes"],
        "size_human": _format_bytes(dataset["size_bytes"]),
        "data_bytes": _dir_size(dataset["root"] / "episodes"),
        "data_human": _format_bytes(_dir_size(dataset["root"] / "episodes")),
        "image_bytes": _dir_size(dataset["root"] / "frames"),
        "image_human": _format_bytes(_dir_size(dataset["root"] / "frames")),
        "features": ["render"],
        "image_shapes": {"render": image_shape},
        "episode_lengths": dataset["episode_lengths"],
        "rendered_frames": rendered_frames,
        "failed_episodes": manifest.get("failed_episodes") or [],
        "robot_model": robot.get("model") or manifest.get("robot_model") or "mycobot_320",
        "gripper": robot.get("gripper") or manifest.get("gripper") or "adaptive",
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
    images = {}
    image_path = ""
    render_path = row.get("observation", {}).get("images", {}).get("render")
    if not render_path:
        render_path = _nearest_mycobot_render_path(root, episode, frame, lengths[episode])
    if render_path:
        image_path = str(render_path)
        image_bytes = (root / render_path).read_bytes()
        mime = dataset["manifest"].get("image_mime_type") or "image/bmp"
        images["render"] = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
    state_values = [float(value) for value in row.get("observation", {}).get("state", [])]
    action_values = [float(value) for value in row.get("action", [])]
    state_names = list(dataset["manifest"].get("joint_names") or [])
    action_names = list(dataset["manifest"].get("action_names") or [])
    return {
        "split": split,
        "episode": episode,
        "frame": frame,
        "episode_length": lengths[episode],
        "row_index": frame,
        "timestamp": float(row.get("timestamp") or 0.0),
        "task": row.get("task", ""),
        "prompt": row.get("prompt") or row.get("task", ""),
        "task_index": None,
        "phase": row.get("phase", ""),
        "images": images,
        "image_path": image_path,
        "state": _named_values(state_names or MYCOBOT_JOINT_NAMES, state_values),
        "action": _named_values(action_names or MYCOBOT_JOINT_NAMES, action_values),
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


def _named_values(names: list[str], values: list[float]) -> dict[str, float]:
    rows = {}
    for index, value in enumerate(values):
        name = names[index] if index < len(names) else f"value_{index}"
        rows[str(name)] = value
    return rows


@lru_cache(maxsize=64)
def _dataset_metadata(repo_root: Path, split: str) -> dict[str, Any]:
    roots = _dataset_roots(repo_root)
    if split not in roots:
        raise ValueError(f"unknown split: {split}")
    root = roots[split]
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_keys = [key for key in CAMERA_KEYS if key in info["features"]]
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    episodes_table = pq.read_table([str(path) for path in episode_files])
    episodes = _rows(episodes_table.to_pydict())
    return {
        "root": root,
        "info": info,
        "camera_keys": camera_keys,
        "data_files": data_files,
        "episodes": episodes,
        "episode_lengths": [int(row["length"]) for row in episodes],
        "size_bytes": _dir_size(root),
        "data_bytes": sum(path.stat().st_size for path in data_files),
        "image_bytes": _dir_size(root / "images"),
    }


@lru_cache(maxsize=2)
def _dataset(repo_root: Path, split: str) -> dict[str, Any]:
    metadata = _dataset_metadata(repo_root, split)
    root = metadata["root"]
    tasks_file = root / "meta" / "tasks.parquet"
    return {
        **metadata,
        "table": pq.read_table([str(path) for path in metadata["data_files"]]),
        "tasks": _tasks_by_index(tasks_file) if tasks_file.exists() else {},
    }


def _dataset_roots(repo_root: Path) -> dict[str, Path]:
    roots = {split: _resolve_dataset_path(repo_root, root) for split, root in DATASETS.items()}
    roots.update(_official_dataset_roots(repo_root))
    roots.update({split: _resolve_dataset_path(repo_root, root) for split, root in _skill_dataset_roots(repo_root).items()})
    roots.update({split: _resolve_dataset_path(repo_root, root) for split, root in _generation_recipe_dataset_roots(repo_root).items()})
    roots.update({split: path.resolve() for split, path in _discover_temporary_datasets(repo_root).items()})
    roots.update({split: path.resolve() for split, path in _discover_so101_photoreal_lerobot_datasets(repo_root).items()})
    return roots


def _resolve_dataset_path(repo_root: Path, root: Path) -> Path:
    path = root if root.is_absolute() else repo_root / root
    return path.resolve()


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


def _tasks_by_index(tasks_file: Path) -> dict[int, str]:
    rows = _rows(pq.read_table(str(tasks_file)).to_pydict())
    tasks: dict[int, str] = {}
    for row in rows:
        if "task_index" in row and "task" in row:
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


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


def _loop_tests_payload(repo_root: Path) -> dict[str, Any]:
    test_cases = _official_closed_loop_test_cases(repo_root)
    loop_tests = []
    for test_case in test_cases:
        report_path = _latest_test_case_report(repo_root, test_case)
        if report_path is None:
            loop_tests.append(_unrun_test_case_item(test_case))
        else:
            item = _raw_loop_test_item(report_path)
            item.update(
                {
                    "test_case_id": test_case["id"],
                    "description": test_case.get("description"),
                    "scenario": test_case.get("task_prompt") or item.get("scenario"),
                    "configured_episodes": test_case.get("episodes"),
                    "configured_steps": test_case.get("steps"),
                    "configured_seed": test_case.get("seed"),
                    "start_contract": test_case.get("start_contract"),
                    "plan_json": test_case.get("plan_json"),
                    "precondition_plan_json": test_case.get("precondition_plan_json"),
                }
            )
            loop_tests.append(item)
    return {
        "exports": [
            {
                "id": "official_qwen_edge_test_cases",
                "root": str(repo_root),
                "status": "available",
                "summary": {
                    "loop_tests": len(loop_tests),
                    "source_configs": [str(repo_root / path) for path in TRAINING_CONFIGS if (repo_root / path).exists()],
                },
                "loop_tests": loop_tests,
            }
        ],
        "source": "official_closed_loop_test_cases",
    }


def _official_closed_loop_test_cases(repo_root: Path) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for relative_path in TRAINING_CONFIGS:
        config_path = repo_root / relative_path
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        closed_loop = config.get("training_config", {}).get("closed_loop") if isinstance(config, dict) else None
        if not isinstance(closed_loop, dict):
            closed_loop = config.get("closed_loop") if isinstance(config, dict) else None
        test_cases = closed_loop.get("test_cases") if isinstance(closed_loop, dict) else None
        if not isinstance(test_cases, list) and isinstance(closed_loop, dict):
            test_cases = closed_loop.get("suites")
        if not isinstance(test_cases, list):
            continue
        for test_case in test_cases:
            if isinstance(test_case, dict) and test_case.get("id"):
                merged[str(test_case["id"])] = test_case
    return list(merged.values())


def _latest_test_case_report(repo_root: Path, test_case: dict[str, Any]) -> Path | None:
    test_case_id = str(test_case["id"])
    configured_seed = test_case.get("seed")
    pattern = f"qwen_chain_{test_case_id}_seed*"
    reports = []
    for run_dir in _loop_run_roots(repo_root):
        for report in (run_dir / "closed_loop_evals").glob(f"{pattern}/qwen_closed_loop_eval_report.json"):
            if _report_matches_test_case_contract(report, test_case, configured_seed=configured_seed):
                reports.append(report)
    if not reports:
        return None
    return sorted(reports, key=lambda path: (_checkpoint_to_int(_checkpoint_from_loop_id(path.parent.name)) or -1, _safe_mtime(path)))[-1]


def _report_matches_test_case_contract(report_path: Path, test_case: dict[str, Any], *, configured_seed: Any) -> bool:
    if configured_seed is not None and f"_seed{int(configured_seed)}_" not in report_path.parent.name:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if configured_seed is not None and int(report.get("seed", -1)) != int(configured_seed):
        return False
    expected_start = test_case.get("start_contract")
    if expected_start and report.get("start_contract") != expected_start:
        return False
    expected_color = test_case.get("env_object_color")
    env_config = report.get("env_config") if isinstance(report, dict) else {}
    if expected_color and isinstance(env_config, dict) and env_config.get("object_color") not in {None, expected_color}:
        return False
    return True


def _unrun_test_case_item(test_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "loop_test_id": "",
        "test_case_id": test_case.get("id"),
        "checkpoint": None,
        "training_step": None,
        "scenario": test_case.get("task_prompt"),
        "env_id": None,
        "validation_loss": None,
        "success_rate": None,
        "episodes_completed": 0,
        "status": "not_run",
        "description": test_case.get("description"),
        "configured_episodes": test_case.get("episodes"),
        "configured_steps": test_case.get("steps"),
        "configured_seed": test_case.get("seed"),
        "start_contract": test_case.get("start_contract"),
        "plan_json": test_case.get("plan_json"),
        "precondition_plan_json": test_case.get("precondition_plan_json"),
    }


def _raw_loop_test_item(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    loop_dir = report_path.parent
    checkpoint = _checkpoint_from_loop_id(loop_dir.name)
    step = _checkpoint_to_int(checkpoint)
    return {
        "loop_test_id": loop_dir.name,
        "checkpoint": checkpoint,
        "training_step": step,
        "scenario": report.get("scenario") or report.get("eval_skill_mode") or (report.get("plan") or {}).get("task"),
        "env_id": report.get("env_id"),
        "validation_loss": report.get("validation_loss"),
        "success_rate": report.get("success_rate"),
        "episodes_completed": report.get("episodes_completed") or len(report.get("episodes") or []),
        "status": report.get("status"),
        "seed": report.get("seed"),
        "start_contract": report.get("start_contract"),
    }


def _loop_frame_payload(repo_root: Path, export_id: str, loop_id: str, episode: int, step: int) -> dict[str, Any]:
    official_test_case = None
    if export_id in {"official_qwen_edge_test_cases", "official_qwen_edge_suites"}:
        report_path = _report_path_by_loop_id(repo_root, loop_id)
        if report_path is None:
            raise ValueError(f"unknown loop test: {loop_id}")
        loop_dir = report_path.parent
        run_dir = loop_dir.parent.parent
        official_test_case = _official_test_case_for_report(repo_root, report_path)
    else:
        run_dir = _loop_run_by_id(repo_root, export_id)
        loop_dir = run_dir / "closed_loop_evals" / loop_id
        report_path = loop_dir / "qwen_closed_loop_eval_report.json"
    if not report_path.exists():
        raise ValueError(f"unknown loop test: {loop_id}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes", [])
    if episode < 0 or episode >= len(episodes):
        raise ValueError(f"episode out of range: {episode}")
    episode_manifest = episodes[episode]
    timeline_path = _resolve_repo_path(repo_root, loop_dir, Path(episode_manifest["trace_path"]))
    policy_steps = _read_jsonl(timeline_path)
    policy_steps = _ensure_loop_media_for_viewer(
        repo_root=repo_root,
        loop_dir=loop_dir,
        report=report,
        episode_manifest=episode_manifest,
        episode_index=episode,
        records=policy_steps,
    )
    step = max(0, min(step, len(policy_steps) - 1)) if policy_steps else 0
    row = policy_steps[step] if policy_steps else {}
    start_row = policy_steps[0] if policy_steps else {}
    media = row.get("media") or {}
    policy_input_images, robot_images = _inline_loop_images(repo_root, loop_dir, row)
    start_policy_input_images, start_robot_images = _inline_loop_images(repo_root, loop_dir, start_row)
    start_video = _first_inline_video(repo_root, loop_dir, start_row, episode_manifest)
    checkpoint = _checkpoint_from_loop_id(loop_id)
    configured_scenario = official_test_case.get("task_prompt") if isinstance(official_test_case, dict) else None
    return {
        "export": {"id": export_id, "root": str(run_dir), "kind": "closed_loop_evals"},
        "loop_test": {
            "id": loop_id,
            "checkpoint": checkpoint,
            "training_step": _checkpoint_to_int(checkpoint),
            "scenario": configured_scenario
            or report.get("scenario")
            or report.get("eval_skill_mode")
            or (report.get("plan") or {}).get("task"),
            "env_id": report.get("env_id"),
            "validation_loss": report.get("validation_loss"),
            "success_rate": report.get("success_rate"),
            "status": report.get("status"),
            "seed": report.get("seed"),
            "start_contract": report.get("start_contract"),
            "episodes_completed": report.get("episodes_completed"),
            "camera_contract": report.get("camera_contract", {}),
        },
        "plan": report.get("plan", {}),
        "qwen_prompts": report.get("qwen_prompts", {}),
        "episode": {
            "index": episode,
            "count": len(episodes),
            "seed": episode_manifest.get("seed") or report.get("seed"),
            "final_success": episode_manifest.get("final_success"),
            "total_reward": episode_manifest.get("total_reward"),
            "start_contract": episode_manifest.get("start_contract"),
            "start_contract_state": episode_manifest.get("start_contract_state", {}),
            "steps": episode_manifest.get("steps"),
            "reset_info": episode_manifest.get("reset_info", {}),
            "final_info": episode_manifest.get("final_info", {}),
            "iterations": episode_manifest.get("iterations", []),
        },
        "step": {
            "index": step,
            "count": len(policy_steps),
            "global_step": row.get("global_step"),
            "primitive_step": row.get("primitive_step"),
            "tool_call": row.get("fn"),
            "primitive_id": row.get("primitive_id"),
            "tool_parameters": {
                "object": ((report.get("plan") or {}).get("calls") or [{}])[0].get("object"),
                "primitive_id": row.get("primitive_id"),
                "prompt": row.get("prompt"),
            },
            "policy_input_prompt": (row.get("policy_input") or {}).get("prompt") or row.get("prompt"),
            "policy_input_mapping": (row.get("policy_input") or {}).get("image_feature_mapping", {}) or row.get("image_feature_mapping", {}),
            "policy_output": {"action": row.get("action"), "action_chunk": row.get("policy_rollout_config")},
            "robot": {"reward": row.get("reward"), "info": row.get("info")},
            "media_available": bool(policy_input_images or robot_images or start_video),
            "media_reason": None if media else "no inline media saved in this raw loop test",
        },
        "planner": {},
        "first_tool": {},
        "images": {
            "policy_inputs": policy_input_images,
            "robot_frames": robot_images,
        },
        "start_images": {
            "policy_inputs": start_policy_input_images,
            "robot_frames": start_robot_images,
        },
        "start_video": start_video,
    }


def _official_test_case_for_report(repo_root: Path, report_path: Path) -> dict[str, Any] | None:
    for test_case in _official_closed_loop_test_cases(repo_root):
        try:
            configured_seed = test_case.get("seed")
            if not _report_matches_test_case_contract(report_path, test_case, configured_seed=configured_seed):
                continue
            return test_case
        except Exception:
            continue
    return None


def _inline_loop_images(repo_root: Path, loop_dir: Path, row: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, str | None]]:
    media = row.get("media") or {}
    policy_input_images = {
        name: _data_uri_for_file(repo_root, loop_dir, path)
        for name, path in (media.get("policy_input_images") or {}).items()
    }
    robot_images = {}
    if media.get("robot_frame"):
        robot_images.setdefault("top_down", _data_uri_for_file(repo_root, loop_dir, media["robot_frame"]))
    return policy_input_images, robot_images


def _ensure_loop_media_for_viewer(
    *,
    repo_root: Path,
    loop_dir: Path,
    report: dict[str, Any],
    episode_manifest: dict[str, Any],
    episode_index: int,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return records
    if any((record.get("media") or {}).get("policy_input_images") for record in records):
        return records
    if not any((record.get("media") or {}).get("render_mode") == "deferred" for record in records):
        return records
    cache_dir = loop_dir / ".dataset_viewer_media" / f"episode_{episode_index:03d}"
    cached_trace = cache_dir / "generated_trace.jsonl"
    if cached_trace.exists():
        return _read_jsonl(cached_trace)
    try:
        import build_loop_test_analyzer_export as analyzer_export

        generated = analyzer_export._generate_media_for_records(  # noqa: SLF001
            records=records,
            report=report,
            episode={**episode_manifest, "episode": episode_index},
            episode_dir=cache_dir,
            width=128,
            height=128,
            fps=12,
            every_n_steps=1,
        )
    except Exception:
        return records
    if generated is records or not any((record.get("media") or {}).get("policy_input_images") for record in generated):
        return records
    cached_trace.parent.mkdir(parents=True, exist_ok=True)
    cached_trace.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in generated) + "\n",
        encoding="utf-8",
    )
    return generated


def _loop_run_roots(repo_root: Path) -> list[Path]:
    env_roots = [Path(item).expanduser() for item in os.environ.get("SO101_LOOP_TEST_RUNS", "").split(",") if item.strip()]
    candidates = [_resolve_dataset_path(repo_root, path) for path in env_roots]
    search_root = repo_root / "_workspace" / "so101_training" / "runs"
    if search_root.exists():
        candidates.extend(path.parent.parent.parent for path in search_root.glob("**/closed_loop_evals/*/qwen_closed_loop_eval_report.json"))
    interactive_root = repo_root / INTERACTIVE_RUN_ROOT
    if interactive_root.exists():
        candidates.extend(path.parent.parent.parent for path in interactive_root.glob("**/closed_loop_evals/*/qwen_closed_loop_eval_report.json"))
    unique: dict[Path, None] = {}
    for path in candidates:
        if (path / "closed_loop_evals").exists():
            unique[path.resolve()] = None
    return sorted(unique, key=lambda path: _safe_mtime(path / "closed_loop_evals"), reverse=True)


def _loop_run_by_id(repo_root: Path, export_id: str) -> Path:
    runs = _loop_run_roots(repo_root)
    if not export_id and runs:
        return runs[0]
    for run_dir in runs:
        candidate = _slug(str(run_dir.relative_to(repo_root)) if run_dir.is_relative_to(repo_root) else str(run_dir))
        if candidate == export_id:
            return run_dir
    raise ValueError(f"unknown loop run: {export_id}")


def _report_path_by_loop_id(repo_root: Path, loop_id: str) -> Path | None:
    if not loop_id:
        return None
    reports = [
        run_dir / "closed_loop_evals" / loop_id / "qwen_closed_loop_eval_report.json"
        for run_dir in _loop_run_roots(repo_root)
    ]
    existing = [path for path in reports if path.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: (_checkpoint_to_int(_checkpoint_from_loop_id(path.parent.name)) or -1, _safe_mtime(path)))[-1]


def _resolve_repo_path(repo_root: Path, base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return base_dir / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _data_uri_for_file(repo_root: Path, export_dir: Path, path_value: str | Path) -> str | None:
    path = _resolve_repo_path(repo_root, export_dir, Path(path_value))
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _first_inline_video(
    repo_root: Path,
    loop_dir: Path,
    row: dict[str, Any],
    episode_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    media = row.get("media") or {}
    for key in ("iteration_video_gif", "iteration_video_mp4"):
        if media.get(key):
            path = _resolve_repo_path(repo_root, loop_dir, Path(media[key]))
            return {"name": path.name, "src": _data_uri_for_file(repo_root, loop_dir, path)}
    media_root_value = episode_manifest.get("media_root")
    if media_root_value:
        media_root = _resolve_repo_path(repo_root, loop_dir, Path(media_root_value))
        for pattern in ("*.gif", "*.mp4"):
            videos = sorted((media_root / "videos").glob(pattern))
            if videos:
                return {"name": videos[0].name, "src": _data_uri_for_file(repo_root, loop_dir, videos[0])}
    return None


def _checkpoint_from_loop_id(loop_id: str) -> str | None:
    match = re.search(r"_(\d{4,})$", loop_id)
    return match.group(1) if match else None


def _checkpoint_to_int(checkpoint: str | None) -> int | None:
    if checkpoint and checkpoint.isdigit():
        return int(checkpoint)
    return None


def _simulator_config_payload(repo_root: Path) -> dict[str, Any]:
    test_cases = _official_closed_loop_test_cases(repo_root)
    presets = []
    for test_case in test_cases:
        plan_path = _resolve_repo_path(repo_root, repo_root, Path(str(test_case.get("plan_json") or "")))
        start_report = _start_report_path_for_test_case(repo_root, test_case)
        prompt = str(test_case.get("task_prompt") or "")
        if plan_path.exists():
            try:
                plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
                prompt = str((plan_payload.get("plan") or {}).get("task") or prompt)
            except Exception:
                pass
        presets.append(
            {
                "id": test_case["id"],
                "label": test_case["id"],
                "description": test_case.get("description"),
                "prompt": prompt,
                "task_prompt": test_case.get("task_prompt"),
                "plan_json": str(test_case.get("plan_json") or ""),
                "start_contract": test_case.get("start_contract") or test_case["id"],
                "start_report_path": str(start_report) if start_report else "",
                "seed": int(test_case.get("seed") or 98100),
                "episodes": int(test_case.get("episodes") or 1),
                "env_object_color": test_case.get("env_object_color") or "green",
                "qwen_object": test_case.get("qwen_object") or "green cube",
            }
        )
    return {
        "presets": presets,
        "training_runs": _discover_training_runs(repo_root),
        "valid_mask_checkpoint": str((repo_root / DEFAULT_VALID_MASK_CHECKPOINT).resolve()),
        "default_output_root": str((repo_root / INTERACTIVE_RUN_ROOT).resolve()),
        "defaults": {
            "device": "auto",
            "episodes": 1,
            "policy_n_action_steps": 15,
            "policy_num_steps": 10,
            "artifact_width": 192,
            "artifact_height": 192,
            "artifact_fps": 12,
        },
    }


def _discover_training_runs(repo_root: Path) -> list[dict[str, Any]]:
    run_root = repo_root / "_workspace" / "so101_training" / "runs"
    runs_by_dir: dict[Path, dict[str, Any]] = {}
    for summary_path in sorted(run_root.glob("**/training_run_summary.json")) if run_root.exists() else []:
        summary = _read_json_file(summary_path)
        if not isinstance(summary, dict):
            continue
        run_dir = Path(str(summary.get("run_dir") or summary_path.parent)).resolve()
        runs_by_dir[run_dir] = _training_run_item(repo_root, run_dir, summary=summary, summary_path=summary_path)

    registry = _read_json_file(repo_root / "_workspace" / "so101_training" / "training_runs_index.json")
    if isinstance(registry, dict) and isinstance(registry.get("runs"), list):
        for row in registry["runs"]:
            if not isinstance(row, dict) or not row.get("run_dir"):
                continue
            run_dir = Path(str(row["run_dir"])).resolve()
            runs_by_dir.setdefault(run_dir, _training_run_item(repo_root, run_dir, registry_row=row))

    if run_root.exists():
        for checkpoints_root in run_root.glob("**/model/checkpoints"):
            run_dir = checkpoints_root.parent.parent
            runs_by_dir.setdefault(run_dir.resolve(), _training_run_item(repo_root, run_dir.resolve()))

    runs = [run for run in runs_by_dir.values() if run["checkpoints"]]
    runs.sort(key=lambda item: (item.get("mtime") or 0, item["training_id"]), reverse=True)
    for run in runs:
        run.pop("mtime", None)
    return runs[:50]


def _training_run_item(
    repo_root: Path,
    run_dir: Path,
    *,
    summary: dict[str, Any] | None = None,
    summary_path: Path | None = None,
    registry_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    registry_row = registry_row or {}
    training_id = str(
        summary.get("training_id")
        or registry_row.get("training_id")
        or _slug(run_dir.name)
    )
    dataset_config = summary.get("dataset_config") if isinstance(summary.get("dataset_config"), dict) else {}
    dataset_config_name = (
        dataset_config.get("name")
        or registry_row.get("dataset_config_name")
        or _infer_dataset_config_name(run_dir)
    )
    checkpoints = _policy_checkpoints_for_run(run_dir)
    mtime = max([item.get("mtime", 0.0) for item in checkpoints] or [_safe_mtime(run_dir)])
    for item in checkpoints:
        item.pop("mtime", None)
    label = f"{training_id}"
    if dataset_config_name:
        label = f"{label} · {dataset_config_name}"
    return {
        "training_id": training_id,
        "label": label,
        "task": dataset_config.get("task"),
        "dataset_config_name": dataset_config_name,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path) if summary_path else registry_row.get("training_run_summary_path"),
        "started_at_utc": summary.get("started_at_utc") or registry_row.get("started_at_utc"),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "mtime": mtime,
    }


def _policy_checkpoints_for_run(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints_root = run_dir / "model" / "checkpoints"
    if not checkpoints_root.exists():
        checkpoints_root = run_dir / "checkpoints"
    checkpoints = []
    if not checkpoints_root.exists():
        return checkpoints
    for config_path in checkpoints_root.glob("*/pretrained_model/config.json"):
        policy_dir = config_path.parent
        if not (policy_dir / "model.safetensors").exists():
            continue
        checkpoint = policy_dir.parent.name
        checkpoints.append(
            {
                "path": str(policy_dir),
                "label": checkpoint,
                "checkpoint": checkpoint,
                "mtime": _safe_mtime(policy_dir / "model.safetensors"),
            }
        )
    return sorted(checkpoints, key=lambda item: (item["checkpoint"], item["mtime"]), reverse=True)


def _infer_dataset_config_name(run_dir: Path) -> str | None:
    train_config_paths = sorted((run_dir / "model" / "checkpoints").glob("*/pretrained_model/train_config.json"))
    if not train_config_paths:
        return None
    payload = _read_json_file(train_config_paths[-1])
    if isinstance(payload, dict):
        dataset = payload.get("dataset")
        if isinstance(dataset, dict) and dataset.get("repo_id"):
            return str(dataset["repo_id"])
        if payload.get("dataset_config"):
            return str(payload["dataset_config"])
    return None


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _start_report_path_for_test_case(repo_root: Path, test_case: dict[str, Any]) -> Path | None:
    start_dataset = test_case.get("start_dataset")
    if not isinstance(start_dataset, dict) or not start_dataset.get("root"):
        return None
    root = _resolve_dataset_path(repo_root, Path(str(start_dataset["root"])))
    report_path = root / "so101_lerobot_export_report.json"
    return report_path if report_path.exists() else None


def _run_interactive_simulator(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = _simulator_config_payload(repo_root)
    presets = {item["id"]: item for item in config["presets"]}
    preset_id = str(payload.get("preset_id") or "")
    if preset_id not in presets:
        raise ValueError(f"unknown simulator preset: {preset_id}")
    preset = presets[preset_id]
    policy_path = Path(str(payload.get("policy_path") or _default_policy_path(config)))
    if not policy_path.is_absolute():
        policy_path = repo_root / policy_path
    if not policy_path.exists():
        raise ValueError(f"policy path does not exist: {policy_path}")
    valid_mask_checkpoint = Path(str(payload.get("valid_mask_checkpoint") or config["valid_mask_checkpoint"]))
    if not valid_mask_checkpoint.is_absolute():
        valid_mask_checkpoint = repo_root / valid_mask_checkpoint
    if not valid_mask_checkpoint.exists():
        raise ValueError(f"valid mask checkpoint does not exist: {valid_mask_checkpoint}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / INTERACTIVE_RUN_ROOT / f"{preset_id}_{timestamp}"
    seed = int(payload.get("seed") or preset["seed"])
    loop_id = f"interactive_{preset_id}_seed{seed}_{timestamp}"
    output_dir = run_root / "closed_loop_evals" / loop_id
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_json = _interactive_plan_json(repo_root, output_dir, preset, str(payload.get("prompt") or preset["prompt"]))
    start_report_path = Path(str(payload.get("start_report_path") or preset.get("start_report_path") or ""))
    if start_report_path and not start_report_path.is_absolute():
        start_report_path = repo_root / start_report_path
    if start_report_path and not start_report_path.exists():
        raise ValueError(f"start report path does not exist: {start_report_path}")

    episodes = max(1, min(10, int(payload.get("episodes") or 1)))
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_so101_qwen_closed_loop_eval.py"),
        "--qwen-plan-json",
        str(plan_json),
        "--policy-path",
        str(policy_path),
        "--valid-mask-checkpoint",
        str(valid_mask_checkpoint),
        "--output-dir",
        str(output_dir),
        "--env-id",
        "MuJoCoPickLift-v1",
        "--env-object-color",
        str(preset.get("env_object_color") or "green"),
        "--object",
        str(preset.get("qwen_object") or "green cube"),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--start-contract",
        str(preset["start_contract"]),
        "--device",
        str(payload.get("device") or "auto"),
	        "--policy-n-action-steps",
	        str(int(payload.get("policy_n_action_steps") or 15)),
	        "--policy-num-steps",
	        str(int(payload.get("policy_num_steps") or 10)),
        "--record-loop-artifacts",
        "--render-loop-media",
        "--artifact-width",
        str(int(payload.get("artifact_width") or 192)),
        "--artifact-height",
        str(int(payload.get("artifact_height") or 192)),
        "--artifact-fps",
        str(int(payload.get("artifact_fps") or 12)),
    ]
    max_steps_per_primitive = payload.get("max_steps_per_primitive")
    if max_steps_per_primitive is not None:
        command.extend(["--max-steps-per-primitive", str(int(max_steps_per_primitive))])
    if start_report_path:
        command.extend(["--start-report-path", str(start_report_path)])
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = output_dir / "interactive_simulator.log"
    started = time.time()
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(payload.get("timeout_s") or 900),
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    report_path = output_dir / "qwen_closed_loop_eval_report.json"
    status = "passed" if result.returncode == 0 and report_path.exists() else "failed"
    response = {
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "loop_id": loop_id,
        "export_id": _slug(str(run_root.relative_to(repo_root))),
        "report_path": str(report_path),
        "log_path": str(log_path),
        "command": command,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-80:]),
    }
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        continuation_start_report = _write_interactive_continuation_start_report(
            report_path=report_path,
            output_dir=output_dir,
            preset=preset,
        )
        response.update(
            {
                "success_rate": report.get("success_rate"),
                "episodes_completed": report.get("episodes_completed"),
                "report_status": report.get("status"),
                "continuation_start_report_path": str(continuation_start_report) if continuation_start_report else None,
            }
        )
    return response


def _write_interactive_continuation_start_report(
    *,
    report_path: Path,
    output_dir: Path,
    preset: dict[str, Any],
) -> Path | None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    if not episodes:
        return None
    episode = episodes[0]
    trace_path = Path(str(episode.get("trace_path") or ""))
    if not trace_path.is_absolute():
        trace_path = output_dir / trace_path.name
    if not trace_path.exists():
        return None
    rows = _read_jsonl(trace_path)
    if not rows:
        return None
    last = rows[-1]
    q_start = last.get("observation")
    if not isinstance(q_start, list) or not q_start:
        return None
    continuation_report = {
        "operation": "so101_interactive_continuation_start",
        "source_report_path": str(report_path),
        "source_trace_path": str(trace_path),
        "source_last_global_step": last.get("global_step"),
        "episodes": [
            {
                "episode_index": 0,
                "seed": episode.get("seed") or report.get("seed"),
                "task": (report.get("plan") or {}).get("task") or preset.get("prompt"),
                "object_color": preset.get("env_object_color") or "green",
                "object_shape": "cube",
                "q_start": [float(value) for value in q_start],
                "sim_snapshot": last.get("sim_snapshot"),
                "source": "interactive_rollout_last_frame",
            }
        ],
    }
    output_path = output_dir / "continuation_start_report.json"
    output_path.write_text(json.dumps(continuation_report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _default_policy_path(config: dict[str, Any]) -> str:
    runs = config.get("training_runs") or []
    if not runs:
        return ""
    checkpoints = runs[0].get("checkpoints") or []
    if not checkpoints:
        return ""
    return str(checkpoints[0]["path"])


def _interactive_plan_json(repo_root: Path, output_dir: Path, preset: dict[str, Any], prompt: str) -> Path:
    source_path = _resolve_repo_path(repo_root, repo_root, Path(str(preset["plan_json"])))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        raise ValueError(f"invalid plan json: {source_path}")
    plan["task"] = prompt
    calls = plan.get("calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                call["prompt"] = prompt
    output_path = output_dir / "interactive_plan.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Experiment Manager</title>
  <style>
    :root {
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172033;
      background: #eef2f7;
      --bg: #eef2f7;
      --surface: #ffffff;
      --surface-muted: #f8fafc;
      --line: #d6deeb;
      --line-strong: #aebdd0;
      --text-soft: #64748b;
      --ink: #111827;
      --data: #2563eb;
      --data-soft: #eaf1ff;
      --train: #7c3aed;
      --train-soft: #f2ecff;
      --loop: #0f766e;
      --loop-soft: #e5f7f3;
      --sim: #c2410c;
      --sim-soft: #fff1e8;
      --ok: #15803d;
      --bad: #b91c1c;
      --shadow: 0 16px 44px rgba(15, 23, 42, 0.11);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 8% 0%, rgba(37,99,235,0.18) 0, transparent 24rem),
        radial-gradient(circle at 90% 8%, rgba(194,65,12,0.12) 0, transparent 24rem),
        linear-gradient(180deg, #f8fbff 0, var(--bg) 42%);
    }
    header {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 46%, #0f766e 100%);
      color: white;
      padding: 22px 24px 20px;
      box-shadow: var(--shadow);
    }
    h1 { margin: 0; font-size: 30px; letter-spacing: 0; }
    .subtitle { margin: 7px 0 0; color: rgba(255,255,255,0.78); font-size: 13px; }
    main { padding: 18px; display: grid; gap: 16px; max-width: 1480px; margin: 0 auto; }
    section {
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 15px;
      box-shadow: 0 1px 2px rgba(15,23,42,0.04);
      min-width: 0;
      overflow: hidden;
    }
    .app-tabs {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      align-items: stretch;
      padding: 10px;
      background: rgba(255,255,255,0.88);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    .tab-button {
      border: 1px solid var(--line);
      color: #24324a;
      background: #fff;
      border-radius: 10px;
      padding: 10px 13px;
      min-height: 44px;
      transition: background 140ms ease, color 140ms ease, transform 140ms ease, border-color 140ms ease;
    }
    .tab-button:hover { transform: translateY(-1px); border-color: var(--line-strong); }
    .tab-button.active { color: white; border-color: transparent; box-shadow: 0 10px 22px rgba(15,23,42,0.15); }
    #tabDataViewer.active { background: linear-gradient(135deg, var(--data), #1d4ed8); }
    #tabTrainingManager.active { background: linear-gradient(135deg, var(--train), #5b21b6); }
    #tabLoopAnalyzer.active { background: linear-gradient(135deg, var(--loop), #115e59); }
    #tabSimulator.active { background: linear-gradient(135deg, var(--sim), #9a3412); }
    .panel { display: grid; gap: 14px; }
    .panel[hidden] { display: none; }
    #datasetPanel section { border-left: 4px solid var(--data); }
    #trainingPanel section { border-left: 4px solid var(--train); }
    #loopPanel section { border-left: 4px solid var(--loop); }
    #simPanel section { border-left: 4px solid var(--sim); }
    #dataToolbar { border-left: 4px solid var(--data); }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 11px;
      align-items: end;
    }
    .viewer-kind { grid-template-columns: minmax(200px, 300px) minmax(260px, 1fr); }
    .loop-controls { grid-template-columns: minmax(240px, 1.2fr) minmax(240px, 1.2fr) minmax(140px, 0.7fr) minmax(140px, 0.7fr) minmax(88px, auto) minmax(88px, auto); }
	    .sim-controls { grid-template-columns: minmax(220px, 1fr) minmax(300px, 1.25fr) minmax(220px, 1fr) minmax(120px, 0.45fr); }
	    .sim-start-controls { grid-template-columns: minmax(240px, 0.45fr) minmax(360px, 1fr); align-items: stretch; }
	    .sim-policy-controls { grid-template-columns: minmax(300px, 1.1fr) minmax(260px, 1fr) minmax(120px, 0.4fr); }
	    .sim-run-controls { grid-template-columns: repeat(3, minmax(115px, 160px)) repeat(2, minmax(120px, auto)); justify-content: start; }
	    .sim-play-controls { grid-template-columns: minmax(90px, auto) minmax(260px, 1fr) minmax(220px, 0.8fr); align-items: center; margin-bottom: 12px; }
    .manager-grid { display: grid; grid-template-columns: minmax(300px, 410px) minmax(0, 1fr); gap: 14px; align-items: start; }
    .run-list { display: grid; gap: 9px; max-height: 70vh; overflow: auto; padding-right: 2px; }
    .run-item {
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: linear-gradient(180deg, #fff, #fbfdff);
      padding: 11px;
      display: grid;
      gap: 5px;
      white-space: normal;
      min-width: 0;
    }
    .run-item * { min-width: 0; overflow-wrap: anywhere; }
    .run-item.active { border-color: var(--train); background: var(--train-soft); box-shadow: inset 4px 0 0 var(--train); }
    .run-id { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; font-weight: 800; overflow-wrap: anywhere; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin: 10px 0; }
    .metric-card { border: 1px solid var(--line); border-radius: 12px; padding: 11px; background: linear-gradient(180deg, #fff, var(--surface-muted)); min-width: 0; }
    .metric-label { color: var(--text-soft); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 17px; font-weight: 850; margin-top: 5px; overflow-wrap: anywhere; line-height: 1.25; }
    .processing { display: inline-flex; align-items: center; gap: 8px; margin-top: 8px; padding: 9px 11px; border: 1px solid #fed7aa; border-radius: 9px; background: var(--sim-soft); color: #9a3412; font-size: 13px; font-weight: 700; }
    .processing[hidden] { display: none; }
    .spinner { width: 14px; height: 14px; border: 2px solid #fdba74; border-top-color: var(--sim); border-radius: 999px; animation: spin 0.8s linear infinite; flex: 0 0 auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
    label { display: grid; gap: 5px; font-size: 12px; color: var(--text-soft); font-weight: 750; min-width: 0; }
    select, input, textarea, button { font: inherit; border: 1px solid #cbd6e4; border-radius: 9px; padding: 8px 10px; background: #fff; color: var(--ink); min-width: 0; }
    input[type="range"] { padding-left: 0; padding-right: 0; }
    textarea { width: 100%; resize: vertical; }
    select:focus, input:focus, textarea:focus { outline: 3px solid rgba(37,99,235,0.16); border-color: var(--data); }
    button { cursor: pointer; font-weight: 750; white-space: nowrap; }
    button:not(.tab-button):not(.zoom-btn) { background: #f8fafc; transition: background 120ms ease, border-color 120ms ease, transform 120ms ease; }
    button:not(.tab-button):not(.zoom-btn):hover { background: #eef4ff; border-color: #9fb7ee; transform: translateY(-1px); }
    .cameras { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .photoreal-cameras { grid-template-columns: minmax(260px, 520px); }
    .quick-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .quick-strip:empty { display: none; }
    .quick-strip button { min-height: 42px; border-color: #b9ccf5; background: var(--data-soft); color: #1e3a8a; }
    .rollout-row { display: grid; grid-template-columns: repeat(3, minmax(210px, 1fr)); gap: 12px; align-items: start; overflow-x: auto; padding-bottom: 2px; }
    .thumb-row { display: grid; grid-template-columns: repeat(3, minmax(110px, 160px)); gap: 9px; align-items: start; overflow-x: auto; padding-bottom: 2px; }
    .thumb-row img { max-height: 118px; object-fit: cover; }
    .image-card { position: relative; background: #0f172a; border-radius: 12px; overflow: hidden; border: 1px solid #cfd8e6; box-shadow: 0 8px 20px rgba(15,23,42,0.08); }
    .image-card figcaption { padding: 8px 10px; color: #dbeafe; background: linear-gradient(90deg, #0f172a, #253149); margin: 0; font-weight: 750; }
    .image-card img { border: 0; border-radius: 0; display: block; }
    .zoom-btn { position: absolute; top: 7px; right: 7px; padding: 4px 8px; border: 1px solid rgba(255,255,255,0.7); background: rgba(17,24,39,0.78); color: white; border-radius: 7px; font-size: 11px; }
    .top-metrics { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; min-width: 0; }
    .chip { border: 1px solid #d5deea; background: #fff; color: #263244; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 700; max-width: 100%; overflow-wrap: anywhere; }
    .chip strong { color: var(--ink); }
    #datasetPanel .chip, #dataToolbar .chip { background: var(--data-soft); border-color: #c7d8ff; }
    #trainingPanel .chip { background: var(--train-soft); border-color: #ddd0ff; }
    #loopPanel .chip { background: var(--loop-soft); border-color: #bfe8df; }
    #simPanel .chip { background: var(--sim-soft); border-color: #fed7aa; }
    .zoom-modal { position: fixed; inset: 0; z-index: 10; background: rgba(17,24,39,0.88); display: grid; place-items: center; padding: 22px; }
    .zoom-modal[hidden] { display: none; }
    .zoom-modal img { width: min(96vw, 1100px); height: auto; max-height: 88vh; object-fit: contain; image-rendering: pixelated; }
    .zoom-modal button { position: fixed; top: 16px; right: 16px; color: white; background: rgba(17,24,39,0.85); border-color: rgba(255,255,255,0.6); }
    iframe { width: 100%; min-height: 78vh; border: 1px solid var(--line); border-radius: 12px; background: #fff; display: block; }
    figure { margin: 0; }
    figcaption { font-size: 12px; color: #596273; margin-bottom: 5px; }
    img { width: 100%; image-rendering: pixelated; border: 1px solid #d9dee8; border-radius: 8px; background: #111; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: right; padding: 8px 7px; border-bottom: 1px solid #edf0f5; font-variant-numeric: tabular-nums; }
    th { color: #475569; background: #f8fafc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
    th:first-child, td:first-child { text-align: left; }
    .meta { color: var(--text-soft); font-size: 13px; overflow-wrap: anywhere; }
    .prompt { display: grid; gap: 7px; }
    .prompt-label { color: var(--text-soft); font-size: 12px; font-weight: 850; text-transform: uppercase; letter-spacing: 0.05em; }
    .prompt-text { margin: 0; color: var(--ink); font-size: 16px; font-weight: 760; line-height: 1.42; overflow-wrap: anywhere; }
    .loop-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 0.8fr); gap: 12px; align-items: start; }
    .kv { display: grid; grid-template-columns: minmax(130px, 180px) minmax(0, 1fr); gap: 7px 12px; font-size: 13px; align-items: start; }
    .kv div { overflow-wrap: anywhere; min-width: 0; }
    .kv div:nth-child(odd) { color: var(--text-soft); font-weight: 750; }
    .json { white-space: pre-wrap; overflow-wrap: anywhere; background: #0f172a; border: 1px solid #1f2a44; border-radius: 10px; padding: 11px; font-size: 12px; color: #e5edf8; max-height: 520px; overflow: auto; }
    details { border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: #fff; margin-top: 8px; }
    summary { cursor: pointer; font-weight: 800; color: #334155; }
    video { width: 100%; border: 1px solid #d9dee8; border-radius: 8px; background: #111; }
    .empty { color: #7b8496; font-size: 13px; }
    @media (max-width: 1100px) {
      .app-tabs { grid-template-columns: repeat(2, minmax(160px, 1fr)); position: static; }
	      .loop-controls, .sim-controls, .sim-start-controls, .sim-policy-controls { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
      .rollout-row { grid-template-columns: repeat(3, minmax(180px, 240px)); }
    }
    @media (max-width: 760px) {
      main { padding: 12px; }
      header { padding: 18px 16px; }
      h1 { font-size: 25px; }
	      .app-tabs, .controls, .cameras, .loop-grid, .viewer-kind, .manager-grid, .metric-grid, .loop-controls, .sim-controls, .sim-start-controls, .sim-policy-controls, .sim-run-controls { grid-template-columns: 1fr; }
      .quick-strip { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .rollout-row, .thumb-row { grid-template-columns: minmax(220px, 1fr); }
      .kv { grid-template-columns: 1fr; }
    }
	  </style>
	</head>
	<body>
	  <header>
	    <h1>Experiment Manager</h1>
	    <p class="subtitle">Unified workspace for robot policy datasets, training runs, loop-test analysis, and interactive rollouts.</p>
	  </header>
	  <main>
	    <section class="app-tabs" aria-label="Analyzer tabs">
	      <button id="tabDataViewer" class="tab-button active" type="button">Data Viewer</button>
	      <button id="tabTrainingManager" class="tab-button" type="button">Training Manager</button>
	      <button id="tabLoopAnalyzer" class="tab-button" type="button">Loop Test Analyzer</button>
	      <button id="tabSimulator" class="tab-button" type="button">Interactive Simulator</button>
	    </section>
	    <section id="dataToolbar">
	      <div class="controls viewer-kind">
	        <label>Platform
	          <select id="platformKind">
	            <option value="so101" selected>SO101</option>
	            <option value="mycobot">MyCobot</option>
	          </select>
	        </label>
	        <label>Data type
	          <select id="viewKind">
	            <option value="train" selected>Train datasets</option>
	            <option value="valid">Validation datasets</option>
	            <option value="photoreal">Photoreal datasets</option>
	            <option value="preview">Preview datasets</option>
	            <option value="closed_loop">closed loop test case</option>
	          </select>
	        </label>
	        <p id="kindMeta" class="meta"></p>
	      </div>
	    </section>
	    <div id="datasetPanel" class="panel">
	    <section>
	      <div class="prompt-label">Dataset playback</div>
	      <div class="controls">
	        <label>Dataset<select id="split"></select></label>
	        <label>Episode<input id="episode" type="range" min="0" max="0" value="0"></label>
        <label>Frame<input id="frame" type="range" min="0" max="0" value="0"></label>
        <button id="play">Play</button>
        <label>FPS<select id="fps"><option value="30" selected>30</option><option value="24">24</option><option value="12">12</option><option value="6">6</option></select></label>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      </div>
      <p id="meta" class="meta"></p>
      <div id="photorealShortcuts" class="quick-strip"></div>
    </section>
    <section class="prompt">
      <div class="prompt-label">Prompt</div>
      <p id="promptText" class="prompt-text"></p>
    </section>
    <section class="cameras" id="cameras"></section>
    <section id="photorealPanel" hidden>
      <div class="prompt-label">Photoreal sidecar</div>
      <p id="photorealMeta" class="meta"></p>
      <div class="cameras photoreal-cameras" id="photorealCameras"></div>
    </section>
    <section>
      <div class="prompt-label">Motor state and action</div>
      <table>
	        <thead><tr><th>Joint</th><th>State</th><th>Action</th></tr></thead>
	        <tbody id="jointRows"></tbody>
	      </table>
	    </section>
	    </div>
	    <div id="trainingPanel" class="panel" hidden>
	    <section>
	      <div class="top-metrics">
	        <span class="chip">source: <strong>_workspace/so101_training</strong></span>
	        <span id="trainingActiveChip" class="chip">active: <strong>loading</strong></span>
	        <button id="trainingReload">Reload</button>
	      </div>
	    </section>
	    <section class="manager-grid">
	      <div>
	        <div class="prompt-label">Training runs</div>
	        <div id="trainingRuns" class="run-list"></div>
	      </div>
	      <div>
	        <div class="prompt-label">Training detail</div>
	        <div id="trainingDetail" class="prompt-text">Select a training run.</div>
	      </div>
	    </section>
	    </div>
	    <div id="loopPanel" class="panel" hidden>
	    <section>
	      <div class="top-metrics">
	        <span class="chip">mode: <strong>full analyzer</strong></span>
	        <span class="chip">route: <strong>/loop-analyzer/</strong></span>
	        <button id="loopAnalyzerReload" type="button">Reload analyzer</button>
	      </div>
	      <p class="meta">Full Loop Test Analyzer is mounted inside Experiment Manager with its original filters, episode selector, diagnostics, media generation, synced camera playback, and raw payload panels.</p>
	      <iframe id="loopAnalyzerFrame" title="Loop Test Analyzer" src="about:blank"></iframe>
	      <div hidden>
	        <div id="loopPolicyCameras"></div>
	        <div id="loopStartCameras">Episode start images</div>
	      </div>
	    </section>
	    </div>
	    <div id="simPanel" class="panel" hidden>
	    <section>
	      <div class="prompt-label">Start preset and prompt</div>
	      <div class="controls sim-start-controls">
	        <label>Start preset<select id="simPreset"></select></label>
	        <label>Policy prompt<textarea id="simPrompt" rows="3"></textarea></label>
	      </div>
	      <p id="simPresetMeta" class="meta"></p>
	    </section>
	    <section>
	      <div class="prompt-label">Training task/run and checkpoint</div>
	      <div class="controls sim-policy-controls">
	        <label>Training task/run<select id="simTrainingRun"></select></label>
	        <label>Checkpoint<select id="simCheckpoint"></select></label>
	        <label>Device<select id="simDevice"><option value="auto" selected>auto</option><option value="mps">mps</option><option value="cpu">cpu</option><option value="cuda">cuda</option></select></label>
	      </div>
	    </section>
	    <section>
	      <div class="prompt-label">Rollout execution</div>
	      <div class="controls sim-run-controls">
	        <label>Episodes<input id="simEpisodes" type="number" min="1" max="10" value="1"></label>
	        <label>Seed<input id="simSeed" type="number" value="98100"></label>
	        <label>Frames / chunks<input id="simPolicyNumSteps" type="number" min="1" max="50" value="10"></label>
	        <button id="simRun">Run model</button>
	        <button id="simContinue" disabled>Continue</button>
	      </div>
	      <div id="simProcessing" class="processing" hidden>
	        <span class="spinner" aria-hidden="true"></span>
	        <span id="simProcessingText">Starting...</span>
	      </div>
	      <p id="simStatus" class="meta"></p>
	    </section>
	    <section>
	      <div class="prompt-label">Latest simulator result</div>
	      <div id="simResult" class="kv"></div>
	    </section>
	    <section>
	      <div class="prompt-label">Replay generated rollout</div>
	      <p class="meta">실행이 끝나면 생성된 rollout의 첫 step을 바로 표시합니다. Zoom으로 카메라 입력을 크게 확인할 수 있습니다.</p>
	      <div class="controls sim-play-controls">
	        <button id="simPlay" disabled>Play</button>
	        <label>Frame<input id="simTimeline" type="range" min="0" max="0" value="0" disabled></label>
	        <span id="simFrameMeta" class="meta">No rollout loaded.</span>
	      </div>
	      <div id="simPreview" class="sim-preview">
	        <p class="empty">Run model을 누르면 rollout preview가 여기에 표시됩니다.</p>
	      </div>
	    </section>
	    </div>
	    <div id="zoomModal" class="zoom-modal" hidden>
	      <button id="zoomClose">Close</button>
	      <img id="zoomImage" alt="zoomed frame">
	    </div>
	  </main>
	  <script>
	    let datasets = {};
	    let datasetNamesByKind = { train: [], valid: [], photoreal: [], preview: [], closed_loop: [] };
	    let datasetPlatformByName = {};
	    let datasetCategoryByName = {};
	    let loopAnalyzerLoaded = false;
	    let simulatorConfig = { presets: [], training_runs: [], defaults: {} };
	    const split = document.getElementById("split");
	    const episode = document.getElementById("episode");
	    const frame = document.getElementById("frame");
    const play = document.getElementById("play");
    const fps = document.getElementById("fps");
    const meta = document.getElementById("meta");
	    const promptText = document.getElementById("promptText");
	    const cameras = document.getElementById("cameras");
	    const photorealShortcuts = document.getElementById("photorealShortcuts");
	    const photorealPanel = document.getElementById("photorealPanel");
	    const photorealMeta = document.getElementById("photorealMeta");
	    const photorealCameras = document.getElementById("photorealCameras");
	    const jointRows = document.getElementById("jointRows");
	    const tabDataViewer = document.getElementById("tabDataViewer");
	    const tabTrainingManager = document.getElementById("tabTrainingManager");
	    const tabLoopAnalyzer = document.getElementById("tabLoopAnalyzer");
	    const tabSimulator = document.getElementById("tabSimulator");
	    const dataToolbar = document.getElementById("dataToolbar");
	    const platformKind = document.getElementById("platformKind");
	    const viewKind = document.getElementById("viewKind");
	    const kindMeta = document.getElementById("kindMeta");
	    const datasetPanel = document.getElementById("datasetPanel");
	    function loopPlaybackTick() {}
	    const trainingPanel = document.getElementById("trainingPanel");
	    const trainingRuns = document.getElementById("trainingRuns");
	    const trainingDetail = document.getElementById("trainingDetail");
	    const trainingActiveChip = document.getElementById("trainingActiveChip");
	    const trainingReload = document.getElementById("trainingReload");
	    const loopPanel = document.getElementById("loopPanel");
	    const simPanel = document.getElementById("simPanel");
	    const loopAnalyzerFrame = document.getElementById("loopAnalyzerFrame");
	    const loopAnalyzerReload = document.getElementById("loopAnalyzerReload");
	    const zoomModal = document.getElementById("zoomModal");
	    const zoomImage = document.getElementById("zoomImage");
	    const zoomClose = document.getElementById("zoomClose");
	    const simPreset = document.getElementById("simPreset");
	    const simTrainingRun = document.getElementById("simTrainingRun");
	    const simCheckpoint = document.getElementById("simCheckpoint");
	    const simDevice = document.getElementById("simDevice");
	    const simPrompt = document.getElementById("simPrompt");
	    const simPresetMeta = document.getElementById("simPresetMeta");
	    const simEpisodes = document.getElementById("simEpisodes");
	    const simSeed = document.getElementById("simSeed");
	    const simPolicyNumSteps = document.getElementById("simPolicyNumSteps");
	    const simRun = document.getElementById("simRun");
	    const simContinue = document.getElementById("simContinue");
	    const simPlay = document.getElementById("simPlay");
	    const simTimeline = document.getElementById("simTimeline");
	    const simFrameMeta = document.getElementById("simFrameMeta");
	    const simProcessing = document.getElementById("simProcessing");
	    const simProcessingText = document.getElementById("simProcessingText");
	    const simStatus = document.getElementById("simStatus");
	    const simResult = document.getElementById("simResult");
	    const simPreview = document.getElementById("simPreview");
	    const fmt = value => Number(value).toFixed(4);
	    let timer = null;
	    let loading = false;
	    let simProcessingTimer = null;
	    let simTimelineRows = [];
	    let simTimelineTimer = null;
	    let simContinuationStartReportPath = null;
	    let trainingRunRows = [];
	    let selectedTrainingId = null;

    async function init() {
      const payload = await fetch("/api/datasets").then(r => r.json());
      datasets = payload.datasets;
      datasetPlatformByName = {};
      datasetCategoryByName = {};
      const orderedNames = [];
      for (const group of payload.dataset_groups || []) {
        for (const item of group.items || []) {
          if (item.name) {
            datasetPlatformByName[item.name] = item.platform || item.summary?.platform || "so101";
            datasetCategoryByName[item.name] = item.category || group.id || "";
          }
          if (item.status === "available" && datasets[item.name] && !orderedNames.includes(item.name)) {
            orderedNames.push(item.name);
          }
        }
      }
      for (const name of Object.keys(datasets)) {
        datasetPlatformByName[name] = datasetPlatformByName[name] || datasets[name]?.platform || "so101";
        datasetCategoryByName[name] = datasetCategoryByName[name] || datasets[name]?.category || "";
        if (!orderedNames.includes(name)) orderedNames.push(name);
      }
	      datasetNamesByKind = {
	        train: orderedNames.filter(name => isTrainDataset(name)),
	        valid: orderedNames.filter(name => name.endsWith("_val") || name.endsWith("_valid") || name.includes("_validation")),
	        photoreal: orderedNames.filter(name => isPhotorealDataset(name)),
	        closed_loop: orderedNames.filter(name => datasetCategoryByName[name] === "closed_loop"),
	        preview: orderedNames.filter(name => isPreviewDataset(name)),
	      };
	      if (!datasetNamesByKind.train.length) datasetNamesByKind.train = orderedNames;
	      if (!datasetNamesByKind.valid.length) datasetNamesByKind.valid = orderedNames.filter(name => !isTrainDataset(name));
	      if (!datasetNamesByKind.preview.length) datasetNamesByKind.preview = orderedNames.filter(name => !isTrainDataset(name) && !name.endsWith("_val") && !name.endsWith("_valid") && !name.includes("_validation") && !isPhotorealDataset(name) && datasetCategoryByName[name] !== "closed_loop");
	      syncViewKind();
	    }

	    function syncAppTab(tabName) {
	      const showData = tabName === "viewer";
	      const showTraining = tabName === "training";
	      const showLoop = tabName === "loop";
	      const showSim = tabName === "simulator";
	      tabDataViewer.classList.toggle("active", showData);
	      tabTrainingManager.classList.toggle("active", showTraining);
	      tabLoopAnalyzer.classList.toggle("active", showLoop);
	      tabSimulator.classList.toggle("active", showSim);
	      dataToolbar.hidden = !showData;
	      datasetPanel.hidden = !showData;
	      trainingPanel.hidden = !showTraining;
	      loopPanel.hidden = !showLoop;
	      simPanel.hidden = !showSim;
	      if (showSim) {
	        kindMeta.textContent = "";
	        if (!simulatorConfig.presets.length) initSimulator();
	        return;
	      }
	      if (showTraining) {
	        loadTrainingRuns();
	        return;
	      }
	      if (showLoop) {
	        initLoopAnalyzer();
	        return;
	      }
	      syncViewKind();
	    }

	    function syncViewKind() {
	      datasetPanel.hidden = false;
	      loopPanel.hidden = true;
	      trainingPanel.hidden = true;
	      simPanel.hidden = true;
	      const platform = platformKind.value || "so101";
	      const selectedNames = namesForKindAndPlatform(viewKind.value, platform);
	      if (!selectedNames.length) {
	        const fallbackKind = ["photoreal", "closed_loop", "preview", "train", "valid"].find(kind => namesForKindAndPlatform(kind, platform).length);
	        if (fallbackKind && fallbackKind !== viewKind.value) {
	          viewKind.value = fallbackKind;
	        }
	      }
	      const names = namesForKindAndPlatform(viewKind.value, platform);
	      const previous = split.value;
	      split.innerHTML = names.map(name => `<option value="${name}">${name}</option>`).join("");
	      if (names.includes(previous)) split.value = previous;
	      else if (names.length) split.value = names[0];
      kindMeta.textContent = `${platformLabel(platform)} ${viewKindLabel(viewKind.value)} datasets (${names.length})`;
      if (split.value) {
        syncEpisodeRange();
        renderPhotorealShortcuts();
        loadFrame();
      } else {
        meta.textContent = "No datasets are available for this platform and data type.";
        promptText.textContent = "";
        cameras.innerHTML = "";
        photorealShortcuts.innerHTML = "";
        photorealPanel.hidden = true;
        photorealCameras.innerHTML = "";
        jointRows.innerHTML = "";
      }
	    }

	    function namesForKindAndPlatform(kind, platform) {
	      return (datasetNamesByKind[kind] || []).filter(name => (datasetPlatformByName[name] || "so101") === platform);
	    }

	    function isTrainDataset(name) {
	      if (name.endsWith("_val") || name.endsWith("_valid") || name.includes("_validation") || name.includes("_loop_")) return false;
	      const category = datasetCategoryByName[name] || "";
	      return /_train[0-9]*$/.test(name) || category === "official" || category === "skill" || category === "generated";
	    }

	    function isPreviewDataset(name) {
	      const platform = datasetPlatformByName[name] || "so101";
	      const category = datasetCategoryByName[name] || "";
	      return platform !== "so101" || category === "temporary" || category === "mycobot";
	    }

	    function isPhotorealDataset(name) {
	      const category = datasetCategoryByName[name] || "";
	      return category === "photoreal" || name.startsWith("photoreal_");
	    }

	    function platformLabel(platform) {
	      return platform === "mycobot" ? "MyCobot" : "SO101";
	    }

	    function viewKindLabel(kind) {
	      if (kind === "valid") return "validation";
	      if (kind === "photoreal") return "photoreal";
	      if (kind === "closed_loop") return "closed loop";
	      if (kind === "preview") return "preview";
	      return "train";
	    }

	    async function loadTrainingRuns() {
	      trainingActiveChip.innerHTML = `active: <strong>loading</strong>`;
	      const payload = await fetch("/api/training/runs").then(r => r.json());
	      trainingRunRows = payload.runs || [];
	      trainingActiveChip.innerHTML = `active: <strong>${payload.active_training_id || "none"}</strong>`;
	      trainingRuns.innerHTML = trainingRunRows.map(row => `
	        <button class="run-item ${row.training_id === selectedTrainingId ? "active" : ""}" type="button" data-training-id="${escapeAttr(row.training_id)}">
	          <div class="run-id">${escapeHtml(row.training_id)} ${row.active ? '<span class="chip">active</span>' : ''}</div>
	          <div class="meta">${escapeHtml(row.dataset_config_name || "")}</div>
	          <div class="meta">train ${fmtMaybe(row.latest_train_loss)} · val ${fmtMaybe(row.latest_val_loss)} · ckpt ${row.checkpoint_count || 0}</div>
	        </button>
	      `).join("") || `<p class="empty">No training runs found.</p>`;
	      if (!selectedTrainingId && payload.active_training_id) selectedTrainingId = payload.active_training_id;
	      if (!selectedTrainingId && trainingRunRows.length) selectedTrainingId = trainingRunRows[0].training_id;
	      if (selectedTrainingId) await loadTrainingDetail(selectedTrainingId);
	    }

	    async function loadTrainingDetail(trainingId) {
	      selectedTrainingId = trainingId;
	      trainingRuns.querySelectorAll(".run-item").forEach(button => {
	        button.classList.toggle("active", button.dataset.trainingId === selectedTrainingId);
	      });
	      trainingDetail.innerHTML = "Loading...";
	      const payload = await fetch(`/api/training/run?id=${encodeURIComponent(trainingId)}`).then(r => r.json());
	      if (payload.error) {
	        trainingDetail.textContent = payload.error;
	        return;
	      }
	      const summary = payload.summary || {};
	      const metrics = payload.metrics || {};
	      const trainRows = metrics.training || [];
	      const valRows = metrics.validation || [];
	      const loopRows = metrics.closed_loop || [];
	      const latestLoop = loopRows[loopRows.length - 1] || {};
	      trainingDetail.innerHTML = `
	        <div class="top-metrics">
	          <span class="chip">id: <strong>${escapeHtml(payload.training_id)}</strong></span>
	          <span class="chip">active: <strong>${payload.status?.active ? "true" : "false"}</strong></span>
	          <span class="chip">checkpoints: <strong>${(metrics.checkpoints || []).length}</strong></span>
	        </div>
	        <p class="meta">${escapeHtml(payload.paths?.run_dir || "")}</p>
	        <p class="meta">${linkHtml(summary.tensorboard_url, "TensorBoard")} ${linkHtml(summary.mobile_tensorboard_url, "Mobile TensorBoard")}</p>
	        <div class="metric-grid">
	          ${metricCard("dataset", summary.dataset_config?.name || "n/a")}
	          ${metricCard("train loss", fmtMaybe(lastMetric(trainRows, "loss")))}
	          ${metricCard("val loss", fmtMaybe(lastMetric(valRows, "loss")))}
	          ${metricCard("closed loop", latestLoop.test_id ? `${latestLoop.test_id}: ${fmtMaybe(latestLoop.success_rate)}` : "n/a")}
	          ${metricCard("train rows", trainRows.length)}
	          ${metricCard("val rows", valRows.length)}
	          ${metricCard("loop rows", loopRows.length)}
	          ${metricCard("started", summary.started_at_utc || "n/a")}
	        </div>
	        <details open><summary>Run identity</summary><pre class="json">${escapeHtml(JSON.stringify(trainingIdentity(payload), null, 2))}</pre></details>
	        <details><summary>Dataset config</summary><pre class="json">${escapeHtml(JSON.stringify(summary.dataset_config || {}, null, 2))}</pre></details>
	        <details><summary>Training command</summary><pre class="json">${escapeHtml((summary.train_cmd || []).join(" \\n"))}</pre></details>
	        <details><summary>Metrics</summary><pre class="json">${escapeHtml(JSON.stringify(metrics, null, 2))}</pre></details>
	        <details><summary>Train log tail</summary><pre class="json">${escapeHtml(payload.logs?.train_tail || "")}</pre></details>
	      `;
	    }

	    function trainingIdentity(payload) {
	      const summary = payload.summary || {};
	      return {
	        training_id: payload.training_id,
	        active: payload.status?.active || false,
	        run_dir: payload.paths?.run_dir,
	        summary_path: payload.summary_path,
	        started_at_utc: summary.started_at_utc,
	        written_at_utc: summary.written_at_utc,
	        tensorboard_url: summary.tensorboard_url,
	        mobile_tensorboard_url: summary.mobile_tensorboard_url,
	      };
	    }

	    function metricCard(label, value) {
	      return `<div class="metric-card"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(String(value))}</div></div>`;
	    }

	    function lastMetric(rows, key) {
	      for (let index = rows.length - 1; index >= 0; index--) {
	        if (typeof rows[index][key] === "number") return rows[index][key];
	      }
	      return null;
	    }

	    function fmtMaybe(value) {
	      return typeof value === "number" ? value.toFixed(5) : "n/a";
	    }

	    function escapeHtml(value) {
	      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
	    }

	    function escapeAttr(value) {
	      return escapeHtml(value).replace(/`/g, "&#096;");
	    }

	    function linkHtml(url, text) {
	      return url ? `<a href="${escapeAttr(url)}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>` : "";
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

	    function renderPhotorealShortcuts() {
	      const data = datasets[split.value] || {};
	      const preview = data.photoreal_preview || {};
	      const framesByEpisode = preview.frames_by_episode || {};
	      const buttons = [];
	      for (const [episodeIndex, frames] of Object.entries(framesByEpisode)) {
	        if (!Array.isArray(frames)) continue;
	        frames.forEach((frameIndex, index) => {
	          const label = index === 0 ? "start" : (index === frames.length - 1 ? "grip" : `f${frameIndex}`);
	          buttons.push(`<button type="button" data-episode="${episodeIndex}" data-frame="${frameIndex}">ep${episodeIndex} ${label}</button>`);
	        });
	      }
	      photorealShortcuts.innerHTML = buttons.join("");
	    }

	    async function loadFrame() {
      if (loading) return;
      loading = true;
      syncFrameRange();
      const url = `/api/frame?split=${split.value}&episode=${episode.value}&frame=${frame.value}`;
      const row = await fetch(url).then(r => r.json()).finally(() => { loading = false; });
      meta.textContent = `${row.split} | episode ${row.episode}/${datasets[row.split].episodes - 1} | frame ${row.frame}/${row.episode_length - 1} | row ${row.row_index} | task_index ${row.task_index ?? "n/a"} | t=${row.timestamp.toFixed(3)}s`;
      promptText.textContent = row.prompt || row.task || "(no prompt stored)";
      cameras.innerHTML = cameraFigures(row.images);
      renderPhotorealPanel(row);
      jointRows.innerHTML = Object.keys(row.state).map(joint => `
        <tr><td>${joint}</td><td>${fmt(row.state[joint])}</td><td>${fmt(row.action[joint])}</td></tr>
	      `).join("");
	    }

	    function renderPhotorealPanel(row) {
	      const images = row.photoreal_images || {};
	      const entries = Object.entries(images).filter(([, src]) => src);
	      if (!entries.length) {
	        photorealPanel.hidden = true;
	        photorealMeta.textContent = "";
	        photorealCameras.innerHTML = "";
	        return;
	      }
	      const contract = row.camera_contract || {};
	      photorealPanel.hidden = false;
	      photorealMeta.textContent = `sidecar frame for ep ${row.episode}, frame ${row.frame}; policy cameras: camera1=${contract["observation.images.camera1"] || "n/a"}, camera2=${contract["observation.images.camera2"] || "n/a"}`;
	      photorealCameras.innerHTML = cameraFigures(images);
	    }

	    function initLoopAnalyzer(force = false) {
	      if (!loopAnalyzerFrame) return;
	      if (force || !loopAnalyzerLoaded || loopAnalyzerFrame.src === "about:blank") {
	        loopAnalyzerFrame.src = "/loop-analyzer/";
	        loopAnalyzerLoaded = true;
	      }
	    }

	    function cameraFiguresWithOptions(images, options) {
	      const entries = Object.entries(images || {}).filter(([, src]) => src);
	      if (!entries.length) return `<p class="empty">No images saved for this step.</p>`;
	      return entries.map(([name, src]) => `
	        <figure class="image-card ${options.thumbnail ? "thumb" : ""}">
	          <figcaption>${name}</figcaption>
	          <button class="zoom-btn" type="button" data-zoom-src="${src}" data-zoom-name="${name}">Zoom</button>
	          <img src="${src}" alt="${name}">
	        </figure>
	      `).join("");
	    }

	    function cameraFigures(images, options = {}) {
	      return cameraFiguresWithOptions(images, options);
	    }

	    function chips(values) {
	      return Object.entries(values)
	        .filter(([, value]) => value !== undefined && value !== null && value !== "")
	        .map(([key, value]) => `<span class="chip">${key}: <strong>${value}</strong></span>`)
	        .join("");
	    }

	    function kvRows(values) {
	      return Object.entries(values).map(([key, value]) => `<div>${key}</div><div>${value ?? "n/a"}</div>`).join("");
	    }

	    function jsonDetails(title, value) {
	      return `<details><summary>${escapeHtml(title)}</summary><pre class="json">${escapeHtml(JSON.stringify(value, null, 2))}</pre></details>`;
	    }

	    function clipText(value, maxLength = 2000) {
	      const text = String(value ?? "");
	      if (text.length <= maxLength) return text;
	      return `${text.slice(0, maxLength)}\n... clipped ${text.length - maxLength} chars`;
	    }

	    function openZoom(src, name) {
	      if (!src) return;
	      zoomImage.src = src;
	      zoomImage.alt = name || "zoomed frame";
	      zoomModal.hidden = false;
	    }

	    function closeZoom() {
	      zoomModal.hidden = true;
	      zoomImage.removeAttribute("src");
	    }

	    async function initSimulator() {
	      simulatorConfig = await fetch("/api/simulator/config").then(r => r.json());
	      simPreset.innerHTML = (simulatorConfig.presets || []).map(item => `<option value="${item.id}">${item.label}</option>`).join("");
	      simTrainingRun.innerHTML = (simulatorConfig.training_runs || []).map(item => {
	        const count = item.checkpoint_count || (item.checkpoints || []).length;
	        return `<option value="${item.training_id}">${item.label} (${count})</option>`;
	      }).join("");
	      syncSimulatorCheckpoints();
	      syncSimulatorPreset();
	    }

	    function selectedSimPreset() {
	      return (simulatorConfig.presets || []).find(item => item.id === simPreset.value) || (simulatorConfig.presets || [])[0];
	    }

	    function syncSimulatorPreset() {
	      const preset = selectedSimPreset();
	      if (!preset) {
	        simPresetMeta.textContent = "No simulator presets configured.";
	        return;
	      }
	      simPrompt.value = preset.prompt || preset.task_prompt || "";
	      simSeed.value = String(preset.seed || 98100);
	      simEpisodes.value = "1";
	      simContinuationStartReportPath = null;
	      if (simContinue) simContinue.disabled = true;
	      simPresetMeta.textContent = `${preset.id} | start ${preset.start_contract} | start report ${preset.start_report_path || "missing"} | plan ${preset.plan_json}`;
	    }

	    function selectedTrainingRun() {
	      return (simulatorConfig.training_runs || []).find(item => item.training_id === simTrainingRun.value) || (simulatorConfig.training_runs || [])[0];
	    }

	    function syncSimulatorCheckpoints() {
	      const run = selectedTrainingRun();
	      const checkpoints = run?.checkpoints || [];
	      simCheckpoint.innerHTML = checkpoints.map(item => `<option value="${item.path}">${item.label}</option>`).join("");
	      if (checkpoints.length) simCheckpoint.value = checkpoints[0].path;
	      simContinuationStartReportPath = null;
	      if (simContinue) simContinue.disabled = true;
	      if (run) {
	        const count = checkpoints.length;
	        simStatus.textContent = `Selected training run: ${run.training_id} | checkpoints ${count} | ${run.run_dir}`;
	      } else {
	        simStatus.textContent = "No training runs with checkpoints were found.";
	      }
	    }

	    async function runSimulator(options = {}) {
	      const preset = selectedSimPreset();
	      if (!preset) return;
	      const continueFromLast = Boolean(options.continueFromLast);
	      const startedAt = Date.now();
	      if (simProcessingTimer) clearInterval(simProcessingTimer);
	      if (simTimelineTimer) {
	        clearInterval(simTimelineTimer);
	        simTimelineTimer = null;
	        simPlay.textContent = "Play";
	      }
	      simRun.disabled = true;
	      simContinue.disabled = true;
	      simRun.textContent = "Running...";
	      simProcessing.hidden = false;
	      simProcessingText.textContent = "Loading policy and running rollout... 0s";
	      simProcessingTimer = setInterval(() => {
	        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
	        simProcessingText.textContent = `Loading policy and running rollout... ${elapsed}s`;
	      }, 1000);
	      simStatus.textContent = continueFromLast
	        ? "Continuing from previous rollout state."
	        : "Model load + closed-loop rollout is in progress.";
	      simResult.innerHTML = "";
	      if (!continueFromLast) {
	        simTimelineRows = [];
	        simTimeline.disabled = true;
	        simTimeline.value = "0";
	        simTimeline.max = "0";
	        simPlay.disabled = true;
	        simFrameMeta.textContent = "No rollout loaded.";
	        simPreview.innerHTML = `<p class="empty">Running rollout. Preview will appear when the first report is ready.</p>`;
	      } else {
	        simPreview.insertAdjacentHTML("afterbegin", `<p class="empty">Continuing rollout. Existing timeline will be preserved and new frames appended.</p>`);
	      }
	      try {
	        const frameBudget = Number(simPolicyNumSteps.value || 10);
	        const payload = {
	          preset_id: preset.id,
	          prompt: simPrompt.value,
	          policy_path: simCheckpoint.value,
	          valid_mask_checkpoint: simulatorConfig.valid_mask_checkpoint,
	          episodes: Number(simEpisodes.value || 1),
	          seed: Number(simSeed.value || preset.seed || 98100),
	          device: simDevice.value,
	          policy_n_action_steps: 15,
	          policy_num_steps: frameBudget,
	          max_steps_per_primitive: frameBudget,
	          artifact_width: 192,
	          artifact_height: 192,
	          artifact_fps: 12,
	        };
	        if (continueFromLast && simContinuationStartReportPath) {
	          payload.start_report_path = simContinuationStartReportPath;
	        }
	        const result = await fetch("/api/simulator/run", {
	          method: "POST",
	          headers: {"Content-Type": "application/json"},
	          body: JSON.stringify(payload),
	        }).then(async r => {
	          const body = await r.json();
	          if (!r.ok) throw new Error(body.message || JSON.stringify(body));
	          return body;
	        });
	        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
	        simProcessingText.textContent = `Finished in ${elapsed}s`;
	        simStatus.textContent = `${result.status} | elapsed ${result.elapsed_s}s | success ${result.success_rate ?? "n/a"}`;
	        simResult.innerHTML = `
	          <div class="top-metrics">${chips({
	            status: result.status,
	            returncode: result.returncode,
	            success_rate: result.success_rate ?? "n/a",
	            episodes_completed: result.episodes_completed ?? "n/a",
	          })}</div>
	          <div class="kv" style="margin-top:10px">${kvRows({
	            run_root: result.run_root,
	            loop_id: result.loop_id,
	            report_path: result.report_path,
	            log_path: result.log_path,
	            continuation_start_report_path: result.continuation_start_report_path,
	          })}</div>
	          ${jsonDetails("stdout tail", clipText(result.stdout_tail || ""))}
	          ${jsonDetails("command", result.command || [])}
	        `;
	        simContinuationStartReportPath = result.continuation_start_report_path || null;
	        await loadSimulatorPreview(result, { append: continueFromLast });
	        loopAnalyzerLoaded = false;
	      } catch (error) {
	        simProcessingText.textContent = "Failed";
	        simStatus.textContent = `failed: ${error.message || error}`;
	        simResult.innerHTML = `${jsonDetails("error", error.message || String(error))}`;
	        simPreview.innerHTML = `<p class="empty">Rollout preview is unavailable because the run failed.</p>`;
	      } finally {
	        if (simProcessingTimer) clearInterval(simProcessingTimer);
	        simProcessingTimer = null;
	        simRun.disabled = false;
	        simContinue.disabled = !simContinuationStartReportPath;
	        simRun.textContent = "Run model";
	      }
	    }

	    async function loadSimulatorPreview(result, options = {}) {
	      if (!result?.export_id || !result?.loop_id) {
	        simPreview.innerHTML = `<p class="empty">No replay ids returned by simulator.</p>`;
	        return;
	      }
	      const append = Boolean(options.append);
	      const segmentIndex = append
	        ? Math.max(1, ...simTimelineRows.map(row => Number(row.timelineSegmentIndex || 0))) + 1
	        : 1;
	      const priorLength = append ? simTimelineRows.length : 0;
	      const url = `/api/loop-frame?export=${encodeURIComponent(result.export_id)}&loop=${encodeURIComponent(result.loop_id)}&episode=0&step=0`;
	      try {
	        const firstRow = await fetch(url).then(async r => {
	          const body = await r.json();
	          if (!r.ok) throw new Error(body.message || JSON.stringify(body));
	          return body;
	        });
	        const frameCount = Math.max(1, Number(firstRow.step?.count || 1));
	        const newRows = [firstRow];
	        for (let index = 1; index < frameCount; index++) {
	          const frameUrl = `/api/loop-frame?export=${encodeURIComponent(result.export_id)}&loop=${encodeURIComponent(result.loop_id)}&episode=0&step=${index}`;
	          const row = await fetch(frameUrl).then(async r => {
	            const body = await r.json();
	            if (!r.ok) throw new Error(body.message || JSON.stringify(body));
	            return body;
	          });
	          newRows.push(row);
	        }
	        for (let index = 0; index < newRows.length; index++) {
	          newRows[index].timelineSegmentIndex = segmentIndex;
	          newRows[index].timelineSegmentLocalIndex = index;
	          newRows[index].timelineSegmentLength = newRows.length;
	          newRows[index].timelineGlobalIndex = priorLength + index;
	          newRows[index].timelineLoopId = result.loop_id;
	        }
	        simTimelineRows = append ? [...simTimelineRows, ...newRows] : newRows;
	        simTimeline.min = "0";
	        simTimeline.max = String(Math.max(0, simTimelineRows.length - 1));
	        simTimeline.value = String(priorLength);
	        simTimeline.disabled = simTimelineRows.length <= 1;
	        simPlay.disabled = simTimelineRows.length <= 1;
	        renderSimulatorFrame(priorLength);
	      } catch (error) {
	        simPreview.innerHTML = `<p class="empty">Preview load failed: ${escapeHtml(error.message || error)}</p>`;
	      }
	    }

	    function renderSimulatorFrame(index) {
	      if (!simTimelineRows.length) {
	        simFrameMeta.textContent = "No rollout loaded.";
	        simPreview.innerHTML = `<p class="empty">No rollout frames loaded.</p>`;
	        return;
	      }
	      const boundedIndex = Math.max(0, Math.min(Number(index), simTimelineRows.length - 1));
	      const row = simTimelineRows[boundedIndex];
	      simTimeline.value = String(boundedIndex);
	      simFrameMeta.textContent = `frame ${boundedIndex + 1}/${simTimelineRows.length} · segment ${row.timelineSegmentIndex || 1} frame ${(row.timelineSegmentLocalIndex || 0) + 1}/${row.timelineSegmentLength || row.step?.count || 1}`;
	        const prompt = row.step?.policy_input_prompt || row.loop_test?.scenario || row.plan?.task || "(prompt not recorded)";
	        const images = {
	          ...(row.images?.policy_inputs || {}),
	          ...(row.images?.robot_frames || {}),
	        };
	        simPreview.innerHTML = `
	          <div class="top-metrics">
		            ${chips({
		              segment: row.timelineSegmentIndex || 1,
		              segment_frame: `${(row.timelineSegmentLocalIndex || 0) + 1}/${row.timelineSegmentLength || row.step?.count || 1}`,
			              loop: row.loop_test?.id,
			              episode: `${row.episode?.index ?? 0}/${Math.max(0, (row.episode?.count ?? 1) - 1)}`,
			              step: `${boundedIndex}/${Math.max(0, simTimelineRows.length - 1)}`,
			              seed: row.episode?.seed,
		              success: row.episode?.final_success,
	              reward: typeof row.episode?.total_reward === "number" ? row.episode.total_reward.toFixed(4) : row.episode?.total_reward,
	            })}
	          </div>
		          <div class="prompt" style="margin-top:12px">
		            <div class="prompt-label">Model inference prompt for this frame</div>
		            <p class="prompt-text">${escapeHtml(prompt)}</p>
		          </div>
		          <div class="prompt" style="margin-top:12px">
		            <div class="prompt-label">Model input cameras for this frame</div>
		            <p class="meta">${escapeHtml(Object.keys(row.images?.policy_inputs || {}).join(", ") || "no policy input images recorded")}</p>
		          </div>
		          <div class="rollout-row" style="margin-top:12px">${cameraFigures(images)}</div>
		          <details open>
		            <summary>Step and camera contract</summary>
		            <div class="kv" style="margin-top:8px">${kvRows({
		              timeline_index: `${boundedIndex + 1}/${simTimelineRows.length}`,
		              segment_loop_id: row.timelineLoopId || row.loop_test?.id,
		              tool_call: row.step?.tool_call,
	              primitive_id: row.step?.primitive_id,
	              policy_mapping: JSON.stringify(row.step?.policy_input_mapping || {}),
	              camera_contract: JSON.stringify(row.loop_test?.camera_contract || {}),
	              report_status: row.loop_test?.status,
	            })}</div>
		          </details>
		        `;
	    }

	    tabDataViewer.addEventListener("click", () => syncAppTab("viewer"));
	    tabTrainingManager.addEventListener("click", () => syncAppTab("training"));
	    tabLoopAnalyzer.addEventListener("click", () => syncAppTab("loop"));
	    tabSimulator.addEventListener("click", () => syncAppTab("simulator"));
	    platformKind.addEventListener("change", syncViewKind);
	    viewKind.addEventListener("change", syncViewKind);
	    trainingReload.addEventListener("click", loadTrainingRuns);
	    trainingRuns.addEventListener("click", event => {
	      const button = event.target.closest?.(".run-item");
	      if (button?.dataset?.trainingId) loadTrainingDetail(button.dataset.trainingId);
	    });
	    simPreset.addEventListener("change", syncSimulatorPreset);
	    simTrainingRun.addEventListener("change", syncSimulatorCheckpoints);
	    simRun.addEventListener("click", () => runSimulator({ continueFromLast: false }));
	    simContinue.addEventListener("click", () => runSimulator({ continueFromLast: true }));
	    simTimeline.addEventListener("input", () => renderSimulatorFrame(Number(simTimeline.value || 0)));
	    simPlay.addEventListener("click", () => {
	      if (simTimelineTimer) {
	        clearInterval(simTimelineTimer);
	        simTimelineTimer = null;
	        simPlay.textContent = "Play";
	        return;
	      }
	      if (!simTimelineRows.length) return;
	      simPlay.textContent = "Pause";
	      simTimelineTimer = setInterval(() => {
	        const current = Number(simTimeline.value || 0);
	        const next = current >= simTimelineRows.length - 1 ? 0 : current + 1;
	        renderSimulatorFrame(next);
	      }, 1000 / 12);
	    });
	    document.addEventListener("click", event => {
	      const button = event.target.closest?.(".zoom-btn");
	      if (button) openZoom(button.dataset.zoomSrc, button.dataset.zoomName);
	    });
	    zoomClose.addEventListener("click", closeZoom);
	    zoomModal.addEventListener("click", event => {
	      if (event.target === zoomModal) closeZoom();
	    });
	    photorealShortcuts.addEventListener("click", event => {
	      const button = event.target.closest?.("button[data-episode][data-frame]");
	      if (!button) return;
	      episode.value = button.dataset.episode;
	      syncFrameRange();
	      frame.value = button.dataset.frame;
	      loadFrame();
	    });
	    split.addEventListener("change", () => { syncEpisodeRange(); renderPhotorealShortcuts(); loadFrame(); });
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
	    loopAnalyzerReload.addEventListener("click", () => initLoopAnalyzer(true));
	    init();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
