#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import ipaddress
import mimetypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

from physical_ai_agent.so101_closed_loop_contract import (
    contract_path_for_start_report,
    load_executable_loop_test_contract,
)
from physical_ai_agent.so101_dataset_registry import (
    DATASET_RECIPE_DIR,
    DatasetRegistryEntry,
    registered_dataset_roots,
    scan_dataset_registry,
)
from physical_ai_agent.so101_trainable_dataset_selection import (
    DatasetRole,
    dataset_role_counts,
    dataset_role_selection_path,
    load_dataset_role_selection,
    selected_catalog_names,
    selected_dataset_roots,
    update_dataset_role_selection,
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
DATASETS_PROGRESS: dict[str, dict[str, Any]] = {}
DATASETS_PROGRESS_LOCK = threading.Lock()
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
DATASET_DELETE_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


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
            if parsed.path == "/vendor/tabulator.min.css":
                self._send_file(repo_root / "third_party" / "tabulator" / "tabulator.min.css", content_type="text/css; charset=utf-8")
                return
            if parsed.path == "/vendor/tabulator.min.js":
                self._send_file(repo_root / "third_party" / "tabulator" / "tabulator.min.js", content_type="application/javascript; charset=utf-8")
                return
            if parsed.path == "/api/datasets":
                self._send_json(_datasets_payload(repo_root))
                return
            if parsed.path == "/api/datasets/catalog":
                self._send_json(_dataset_catalog_page_payload(repo_root, parse_qs(parsed.query)))
                return
            if parsed.path == "/api/datasets/catalog/item":
                query = parse_qs(parsed.query)
                try:
                    self._send_json(
                        _dataset_catalog_named_item_payload(
                            repo_root,
                            _query_str(query, "name", ""),
                        )
                    )
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, status=404)
                return
            if parsed.path == "/api/datasets/progress":
                self._send_json(_datasets_progress_payload(repo_root))
                return
            if parsed.path in {
                "/api/datasets/role-selection",
                "/api/datasets/trainable-selection",
            }:
                self._send_json(_dataset_role_selection_payload(repo_root))
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
            if parsed.path == "/api/datasets/delete":
                body = self._read_json_body()
                try:
                    if not _trusted_dataset_delete_request(self.client_address[0], self.headers):
                        raise PermissionError("dataset deletion is only available from localhost or the same private network")
                    if self.headers.get("X-Dataset-Delete-Confirmation", "") != str(body.get("name") or ""):
                        raise PermissionError("missing dataset deletion confirmation header")
                    self._send_json(_delete_dataset(repo_root, body))
                except PermissionError as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=403)
                except (FileNotFoundError, ValueError) as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=400)
                return
            if parsed.path == "/api/datasets/bulk-delete":
                body = self._read_json_body()
                try:
                    if not _trusted_dataset_delete_request(self.client_address[0], self.headers):
                        raise PermissionError("dataset deletion is only available from localhost or the same private network")
                    confirmation = str(body.get("confirmation") or "")
                    if self.headers.get("X-Dataset-Delete-Confirmation", "") != confirmation:
                        raise PermissionError("missing bulk dataset deletion confirmation header")
                    self._send_json(_delete_datasets(repo_root, body))
                except PermissionError as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=403)
                except (FileNotFoundError, ValueError) as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=400)
                return
            if parsed.path in {
                "/api/datasets/role-selection",
                "/api/datasets/trainable-selection",
            }:
                body = self._read_json_body()
                try:
                    if not _trusted_dataset_delete_request(self.client_address[0], self.headers):
                        raise PermissionError("dataset role selection can only be changed from localhost or the same private network")
                    if parsed.path.endswith("/trainable-selection"):
                        body.setdefault("role", "training")
                    self._send_json(_update_dataset_role_selection(repo_root, body))
                except PermissionError as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=403)
                except (FileNotFoundError, ValueError) as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=400)
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

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
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


def _trusted_dataset_delete_request(client_ip: str, headers: Any) -> bool:
    if headers.get("CF-Connecting-IP") or headers.get("X-Forwarded-For") or headers.get("Forwarded"):
        return False
    try:
        address = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    if not _is_local_or_private_address(address):
        return False

    origin = str(headers.get("Origin") or "").strip()
    if not origin:
        return True
    origin_host = urlparse(origin).hostname
    if origin_host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        origin_address = ipaddress.ip_address(origin_host or "")
    except ValueError:
        return False
    return _is_local_or_private_address(origin_address)


def _is_local_or_private_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_loopback or any(
        address.version == network.version and address in network
        for network in DATASET_DELETE_PRIVATE_NETWORKS
    )


def _delete_dataset(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    confirmation = str(payload.get("confirm_name") or "").strip()
    if not name:
        raise ValueError("missing dataset name")
    if confirmation != name:
        raise ValueError("dataset name confirmation does not match")

    roots = _dataset_roots(repo_root)
    if name not in roots:
        raise ValueError(f"unknown dataset: {name}")
    dataset_root = roots[name].resolve()
    workspace_root = (repo_root / "_workspace").resolve()
    try:
        dataset_root.relative_to(workspace_root)
    except ValueError as exc:
        raise PermissionError(f"refusing to delete a dataset outside {workspace_root}") from exc
    if dataset_root == workspace_root:
        raise PermissionError("refusing to delete the workspace root")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")

    affected_names = sorted(
        split_name
        for split_name, root in roots.items()
        if root.resolve() == dataset_root
    )
    size_bytes = _dir_size(dataset_root)
    shutil.rmtree(dataset_root)
    update_dataset_role_selection(repo_root, remove_roots=[dataset_root])
    _clear_dataset_caches(repo_root)
    return {
        "status": "deleted",
        "name": name,
        "root": str(dataset_root),
        "affected_names": affected_names,
        "size_bytes": size_bytes,
        "size_human": _format_bytes(size_bytes),
    }


def _delete_datasets(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    names = _unique_dataset_names(payload.get("names"))
    confirmation = str(payload.get("confirmation") or "").strip()
    expected_confirmation = f"DELETE {len(names)} DATASETS"
    if confirmation != expected_confirmation:
        raise ValueError(
            f"bulk deletion confirmation must exactly match: {expected_confirmation}"
        )

    roots = _dataset_roots(repo_root)
    workspace_root = (repo_root / "_workspace").resolve()
    targets: dict[Path, dict[str, Any]] = {}
    for name in names:
        if name not in roots:
            raise ValueError(f"unknown dataset: {name}")
        dataset_root = roots[name].resolve()
        try:
            dataset_root.relative_to(workspace_root)
        except ValueError as exc:
            raise PermissionError(
                f"refusing to delete a dataset outside {workspace_root}"
            ) from exc
        if dataset_root == workspace_root:
            raise PermissionError("refusing to delete the workspace root")
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
        if not dataset_root.is_dir():
            raise ValueError(f"dataset root is not a directory: {dataset_root}")
        targets.setdefault(
            dataset_root,
            {
                "root": dataset_root,
                "requested_names": [],
                "affected_names": sorted(
                    split_name
                    for split_name, root in roots.items()
                    if root.resolve() == dataset_root
                ),
                "size_bytes": _dir_size(dataset_root),
            },
        )["requested_names"].append(name)

    total_size = sum(int(target["size_bytes"]) for target in targets.values())
    for dataset_root in targets:
        shutil.rmtree(dataset_root)
    update_dataset_role_selection(repo_root, remove_roots=targets)
    _clear_dataset_caches(repo_root)
    affected_names = sorted(
        {
            affected_name
            for target in targets.values()
            for affected_name in target["affected_names"]
        }
    )
    return {
        "status": "deleted",
        "requested_names": names,
        "deleted_roots": [str(root) for root in sorted(targets, key=str)],
        "affected_names": affected_names,
        "size_bytes": total_size,
        "size_human": _format_bytes(total_size),
    }


def _dataset_role_selection_payload(repo_root: Path) -> dict[str, Any]:
    selection = load_dataset_role_selection(repo_root)
    counts = dataset_role_counts(repo_root)
    return {
        "status": "ok",
        "path": str(dataset_role_selection_path(repo_root)),
        "schema_version": selection.schema_version,
        "updated_at": selection.updated_at,
        "count": len(selection.datasets),
        "counts": counts,
        "datasets": [entry.model_dump(mode="json") for entry in selection.datasets],
        "marked_names": {
            role: sorted(selected_catalog_names(repo_root, role))
            for role in ("training", "validation", "loop_test")
        },
        "marked_roots_by_role": {
            role: [
                str(root)
                for root in sorted(selected_dataset_roots(repo_root, role), key=str)
            ]
            for role in ("training", "validation", "loop_test")
        },
        # Compatibility for clients that only understood the original train set.
        "marked_roots": [
            str(root)
            for root in sorted(selected_dataset_roots(repo_root, "training"), key=str)
        ],
    }


def _trainable_dataset_selection_payload(repo_root: Path) -> dict[str, Any]:
    return _dataset_role_selection_payload(repo_root)


def _update_dataset_role_selection(
    repo_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"mark", "remove"}:
        raise ValueError("dataset role selection action must be 'mark' or 'remove'")
    role = _dataset_selection_role(payload.get("role"))
    names = _unique_dataset_names(payload.get("names"))
    candidates = {candidate["name"]: candidate for candidate in _dataset_catalog_candidates(repo_root)}
    missing = [name for name in names if name not in candidates]
    if missing:
        raise ValueError(f"unknown datasets: {', '.join(missing)}")

    registry_by_root = _training_registry_entries_by_root(repo_root)
    additions: list[dict[str, Any]] = []
    if action == "mark":
        rejected: list[str] = []
        for name in names:
            candidate = candidates[name]
            eligible, reason, registry_entry = _dataset_role_eligibility(
                candidate,
                role,
                registry_by_root,
            )
            if not eligible:
                rejected.append(f"{name}: {reason}")
                continue
            additions.append(
                _dataset_role_selection_entry(
                    candidate,
                    role,
                    registry_entry,
                )
            )
        if rejected:
            raise ValueError(
                f"datasets cannot be used for role '{role}': " + "; ".join(rejected)
            )

    selection = update_dataset_role_selection(
        repo_root,
        additions=additions,
        removals=(
            ({"role": role, "catalog_name": name} for name in names)
            if action == "remove"
            else ()
        ),
    )
    _clear_dataset_caches(repo_root)
    counts = dataset_role_counts(repo_root)
    return {
        "status": "updated",
        "action": action,
        "role": role,
        "requested_names": names,
        "count": len(selection.datasets),
        "counts": counts,
        "datasets": [entry.model_dump(mode="json") for entry in selection.datasets],
        "marked_names": {
            selected_role: sorted(selected_catalog_names(repo_root, selected_role))
            for selected_role in ("training", "validation", "loop_test")
        },
    }


def _update_trainable_dataset_selection(
    repo_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    compatible = dict(payload)
    compatible.setdefault("role", "training")
    return _update_dataset_role_selection(repo_root, compatible)


def _unique_dataset_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("dataset names must be a non-empty list")
    names = list(dict.fromkeys(str(name).strip() for name in value if str(name).strip()))
    if not names:
        raise ValueError("dataset names must be a non-empty list")
    return names


def _training_registry_entries_by_root(
    repo_root: Path,
) -> dict[Path, list[DatasetRegistryEntry]]:
    entries: dict[Path, list[DatasetRegistryEntry]] = {}
    for entry in scan_dataset_registry(repo_root, inspect_artifacts=True).entries:
        entries.setdefault(Path(entry.absolute_root).resolve(), []).append(entry)
    return entries


def _dataset_training_eligibility(
    candidate: dict[str, Any],
    registry_by_root: dict[Path, list[DatasetRegistryEntry]],
) -> tuple[bool, str, DatasetRegistryEntry | None]:
    return _dataset_role_eligibility(candidate, "training", registry_by_root)


def _dataset_role_eligibility(
    candidate: dict[str, Any],
    role: DatasetRole,
    registry_by_root: dict[Path, list[DatasetRegistryEntry]],
) -> tuple[bool, str, DatasetRegistryEntry | None]:
    if candidate.get("platform") != "so101":
        return False, "only SO101 LeRobot datasets are supported", None
    if candidate.get("loader") != "lerobot":
        return False, "dataset is not a LeRobot parquet export", None

    if role == "loop_test":
        if candidate.get("split_key") != "closed_loop":
            return False, "only closed-loop catalog entries can be marked", None
        test_case = candidate.get("loop_test_case")
        if not isinstance(test_case, dict):
            return False, "no executable loop-test contract is registered", None
        report_value = test_case.get("start_report_path")
        if report_value:
            report_path = _resolve_dataset_path(
                Path(candidate.get("repo_root") or REPO_ROOT),
                Path(str(report_value)),
            )
            if not report_path.is_file():
                return False, f"loop-test start report is missing: {report_path}", None
        return True, "executable loop-test contract is registered", None

    expected_split_key = "train" if role == "training" else "valid"
    if candidate.get("split_key") != expected_split_key:
        return False, f"only {expected_split_key} splits can be marked", None
    registry_splits = {"train"} if role == "training" else {"validation", "val"}
    entries = [
        entry
        for entry in registry_by_root.get(Path(candidate["root"]).resolve(), [])
        if entry.split in registry_splits
    ]
    if not entries:
        return False, "dataset is not registered by a generation recipe", None
    ready = next((entry for entry in entries if entry.training_ready), None)
    if ready is None:
        errors = sorted({error for entry in entries for error in entry.readiness_errors})
        return False, "; ".join(errors) or "dataset completion gate has not passed", None
    return True, "registry completion gate passed", ready


def _dataset_role_selection_entry(
    candidate: dict[str, Any],
    role: DatasetRole,
    registry_entry: DatasetRegistryEntry | None,
) -> dict[str, Any]:
    test_case = candidate.get("loop_test_case") if role == "loop_test" else None
    start_dataset = test_case.get("start_dataset") if isinstance(test_case, dict) else None
    repo_id = (
        registry_entry.repo_id
        if registry_entry is not None and registry_entry.repo_id
        else start_dataset.get("repo_id")
        if isinstance(start_dataset, dict) and start_dataset.get("repo_id")
        else "physical-ai-agent/" + str(candidate["name"]).replace("_", "-")
    )
    expected_episodes = (
        registry_entry.episodes
        if registry_entry is not None
        else start_dataset.get("expected_episodes")
        if isinstance(start_dataset, dict)
        else test_case.get("episodes")
        if isinstance(test_case, dict)
        else None
    )
    expected_frames = (
        registry_entry.frames
        if registry_entry is not None
        else start_dataset.get("expected_frames")
        if isinstance(start_dataset, dict)
        else None
    )
    return {
        "role": role,
        "catalog_name": str(candidate["name"]),
        "root": str(candidate["root"]),
        "repo_id": repo_id,
        "expected_episodes": expected_episodes,
        "expected_frames": expected_frames,
        "grid_bin_sidecar": registry_entry.grid_sidecar if registry_entry else None,
        "loop_test_case": copy.deepcopy(test_case),
    }


def _trainable_selection_entry(
    candidate: dict[str, Any],
    registry_entry: DatasetRegistryEntry,
) -> dict[str, Any]:
    return _dataset_role_selection_entry(candidate, "training", registry_entry)


def _dataset_selection_role(value: Any) -> DatasetRole:
    role = str(value or "").strip()
    if role not in {"training", "validation", "loop_test"}:
        raise ValueError(
            "dataset role must be one of: training, validation, loop_test"
        )
    return role  # type: ignore[return-value]


def _clear_dataset_caches(repo_root: Path) -> None:
    _dataset_metadata.cache_clear()
    _dataset.cache_clear()
    with DATASETS_PAYLOAD_LOCK:
        DATASETS_PAYLOAD_CACHE.pop(str(repo_root.resolve()), None)


def _datasets_payload(repo_root: Path) -> dict[str, Any]:
    cache_key = str(repo_root.resolve())
    now = time.monotonic()
    cached = DATASETS_PAYLOAD_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < DATASETS_PAYLOAD_CACHE_SECONDS:
        _set_datasets_progress(repo_root, status="complete", percent=100, message="Catalog ready")
        return cached[1]
    with DATASETS_PAYLOAD_LOCK:
        now = time.monotonic()
        cached = DATASETS_PAYLOAD_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < DATASETS_PAYLOAD_CACHE_SECONDS:
            _set_datasets_progress(repo_root, status="complete", percent=100, message="Catalog ready")
            return cached[1]
        _set_datasets_progress(
            repo_root,
            status="loading",
            percent=1,
            completed=0,
            total=0,
            message="Discovering dataset sources",
        )

        def report_progress(completed: int, total: int, message: str) -> None:
            fraction = completed / max(total, 1)
            _set_datasets_progress(
                repo_root,
                status="loading",
                percent=min(98, 5 + round(fraction * 92)),
                completed=completed,
                total=total,
                message=message,
            )

        try:
            payload = _build_datasets_payload(repo_root, progress=report_progress)
        except Exception as exc:
            _set_datasets_progress(
                repo_root,
                status="error",
                message=f"Catalog loading failed: {exc}",
            )
            raise
        DATASETS_PAYLOAD_CACHE[cache_key] = (time.monotonic(), payload)
        _set_datasets_progress(
            repo_root,
            status="complete",
            percent=100,
            message="Catalog ready",
        )
        return payload


def _dataset_catalog_candidates(repo_root: Path) -> list[dict[str, Any]]:
    official_roots = _official_dataset_roots(repo_root)
    skill_roots = _skill_dataset_roots(repo_root)
    recipe_roots = _generation_recipe_dataset_roots(repo_root)
    photoreal_roots = _discover_so101_photoreal_datasets(repo_root)
    photoreal_lerobot_roots = _discover_so101_photoreal_lerobot_datasets(repo_root)
    closed_loop_views = _generation_closed_loop_views(repo_root)
    temporary_roots = _discover_temporary_datasets(repo_root)
    mycobot_roots = _discover_mycobot_datasets(repo_root)
    loop_test_cases = _official_closed_loop_test_cases(repo_root)
    recipe_paths = {_resolve_dataset_path(repo_root, root) for root in recipe_roots.values()}
    official_paths = {_resolve_dataset_path(repo_root, root) for root in official_roots.values()}
    skill_paths = {_resolve_dataset_path(repo_root, root) for root in skill_roots.values()}
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def add(
        name: str,
        root: Path,
        *,
        category: str,
        loader: str = "lerobot",
        loop_report: Path | None = None,
    ) -> None:
        if name in seen_names:
            return
        resolved = _resolve_dataset_path(repo_root, root)
        if not resolved.exists():
            return
        seen_names.add(name)
        platform = _dataset_platform(name, resolved)
        creation = _dataset_creation_metadata(resolved)
        split_key = _dataset_catalog_split_key(name, category)
        render_key = _dataset_catalog_render_key(name, resolved, category, loader)
        identity = _dataset_catalog_identity(name)
        loop_test_case = (
            _closed_loop_test_case_for_candidate(
                repo_root,
                dataset_root=resolved,
                start_report=loop_report,
                test_cases=loop_test_cases,
            )
            if split_key == "closed_loop"
            else None
        )
        candidates.append(
            {
                "name": name,
                "repo_root": repo_root,
                "root": resolved,
                "category": category,
                "loader": loader,
                "platform": platform,
                "platform_label": _platform_label(platform),
                "split_key": split_key,
                "split_label": _dataset_catalog_split_label(split_key),
                "render_key": render_key,
                "render_label": _dataset_catalog_render_label(render_key),
                "loop_report": loop_report.resolve() if loop_report else None,
                "loop_test_case": loop_test_case,
                **identity,
                **creation,
            }
        )

    for name, root in official_roots.items():
        if _resolve_dataset_path(repo_root, root) not in recipe_paths:
            add(name, root, category="official")
    for name, root in skill_roots.items():
        if _resolve_dataset_path(repo_root, root) not in recipe_paths:
            add(name, root, category="skill")
    for name, root in recipe_roots.items():
        resolved = _resolve_dataset_path(repo_root, root)
        category = (
            "photoreal"
            if _is_photoreal_lerobot_dataset(resolved)
            else "official"
            if resolved in official_paths
            else "skill"
            if resolved in skill_paths
            else "generated"
        )
        add(name, root, category=category)
    for name, view in closed_loop_views.items():
        add(
            name,
            Path(view["root"]),
            category="closed_loop",
            loop_report=Path(view["report"]),
        )
    for name in ARCHIVED_DATASET_SPLITS:
        if name in DATASETS:
            add(name, DATASETS[name], category="archived")
    for name, root in temporary_roots.items():
        add(name, root, category="temporary")
    for name, root in photoreal_roots.items():
        if _resolve_dataset_path(repo_root, root) not in recipe_paths:
            add(name, root, category="photoreal", loader="photoreal_jsonl")
    for name, root in photoreal_lerobot_roots.items():
        if _resolve_dataset_path(repo_root, root) not in recipe_paths:
            add(name, root, category="photoreal")
    for name, root in mycobot_roots.items():
        add(name, root, category="mycobot", loader="mycobot")
    return candidates


def _dataset_catalog_split_key(name: str, category: str) -> str:
    if category == "closed_loop" or name.endswith("_loop_val") or "_loop_validation" in name:
        return "closed_loop"
    if name.endswith("_val") or name.endswith("_valid") or "_validation" in name:
        return "valid"
    return "train"


def _dataset_catalog_split_label(key: str) -> str:
    return {"valid": "Validation", "closed_loop": "Closed loop"}.get(key, "Train")


_CATALOG_NAME_SUFFIXES = (
    "_loop_validation",
    "_loop_test",
    "_loop_val",
    "_source_validation",
    "_source_train",
    "_validation",
    "_valid",
    "_val",
    "_train",
)
_CATALOG_RENDER_TOKENS = {"photoreal", "simulation"}
_CATALOG_FAMILY_PREFIXES = ("grip_the_cube",)
_PHOTOREAL_LEROBOT_FORMAT = "so101_photoreal_lerobot_v1"


def _dataset_catalog_identity(name: str) -> dict[str, str]:
    """Split a catalog id into a stable task family and display version."""

    canonical = name.strip().lower()
    for suffix in _CATALOG_NAME_SUFFIXES:
        if canonical.endswith(suffix):
            canonical = canonical[: -len(suffix)]
            break

    match = re.search(r"(?:^|_)v(\d+(?:_\d+)*)(?=_|$)", canonical)
    if match is None:
        family_key = canonical or name.strip().lower()
        return {
            "family_key": family_key,
            "family_label": family_key.replace("_", " "),
            "version_key": "",
            "version_label": "Unversioned",
        }

    family_key = canonical[: match.start()].rstrip("_")
    tail_tokens = [token for token in canonical[match.end() :].strip("_").split("_") if token]
    tail_tokens = [token for token in tail_tokens if token not in _CATALOG_RENDER_TOKENS]
    tail_tokens = [token for token in tail_tokens if not re.fullmatch(r"train\d*", token)]

    for known_family in _CATALOG_FAMILY_PREFIXES:
        if family_key == known_family:
            break
        if family_key.startswith(known_family + "_"):
            prefix_variant = family_key[len(known_family) + 1 :]
            tail_tokens[:0] = [token for token in prefix_variant.split("_") if token]
            family_key = known_family
            break

    version_key = f"v{match.group(1).replace('_', '.')}"
    if tail_tokens:
        version_key += "_" + "_".join(tail_tokens)
    return {
        "family_key": family_key,
        "family_label": family_key.replace("_", " "),
        "version_key": version_key,
        "version_label": version_key,
    }


def _dataset_catalog_version_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    version = str(
        candidate.get("version_key")
        or _dataset_catalog_identity(str(candidate.get("name") or ""))["version_key"]
    )
    match = re.match(r"^v(\d+(?:\.\d+)*)(?:_(.*))?$", version)
    if match is None:
        return (0, (), version)
    return (
        1,
        tuple(int(part) for part in match.group(1).split(".")),
        match.group(2) or "",
    )


def _dataset_catalog_render_key(name: str, root: Path, category: str, loader: str) -> str:
    if category == "photoreal" or loader == "photoreal_jsonl" or name.startswith("photoreal_"):
        return "photoreal"
    if loader == "mycobot" and "real" in name.lower():
        return "real"
    for metadata_path in (root / "manifest.json", root / "meta" / "info.json"):
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset_format = str(payload.get("dataset_format") or payload.get("format") or "").lower()
        if "real_camera" in dataset_format or "hardware" in dataset_format:
            return "real"
    return "simulation"


def _is_photoreal_lerobot_dataset(root: Path) -> bool:
    manifest_path = root / "photoreal_lerobot_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("format") == _PHOTOREAL_LEROBOT_FORMAT


def _dataset_catalog_render_label(key: str) -> str:
    return {"photoreal": "Photoreal", "real": "Real camera"}.get(key, "Standard sim")


def _dataset_catalog_page_payload(
    repo_root: Path,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    try:
        requested_page = max(1, int(_query_str(query, "page", "1")))
        page_size = max(1, min(50, int(_query_str(query, "size", "10"))))
    except ValueError as exc:
        raise ValueError("catalog page and size must be integers") from exc

    candidates = _dataset_catalog_candidates(repo_root)
    marked_names_by_role = {
        role: selected_catalog_names(repo_root, role)
        for role in ("training", "validation", "loop_test")
    }
    marked_names = set().union(*marked_names_by_role.values())
    filters = {
        "platform_label": _query_str(query, "platform", "").strip().casefold(),
        "split_label": _query_str(query, "status", "").strip().casefold(),
        "render_label": _query_str(query, "type", "").strip().casefold(),
        "family_label": _query_str(query, "family", "").strip().casefold(),
        "version_label": _query_str(query, "version", "").strip().casefold(),
        "name": _query_str(query, "name", "").strip().casefold(),
        "marked_only": _query_str(query, "marked", "").strip().casefold()
        in {"1", "true", "yes", "on"},
    }

    def matches(candidate: dict[str, Any]) -> bool:
        if filters["marked_only"] and str(candidate["name"]) not in marked_names:
            return False
        identity = _dataset_catalog_identity(str(candidate["name"]))
        for field in ("platform_label", "split_label", "render_label"):
            expected = filters[field]
            if expected and str(candidate[field]).casefold() != expected:
                return False
        family_label = str(candidate.get("family_label") or identity["family_label"])
        if filters["family_label"] and filters["family_label"] not in family_label.casefold():
            return False
        version_label = str(candidate.get("version_label") or identity["version_label"])
        if filters["version_label"] and filters["version_label"] not in version_label.casefold():
            return False
        return not filters["name"] or filters["name"] in str(candidate["name"]).casefold()

    filtered = [candidate for candidate in candidates if matches(candidate)]
    sort_field = _query_str(query, "sort", "createdEpoch")
    sort_direction = _query_str(query, "dir", "desc").lower()
    sort_key_by_field = {
        "createdEpoch": lambda row: float(row.get("created_at_epoch") or 0),
        "name": lambda row: str(row.get("name") or "").casefold(),
        "familyLabel": lambda row: str(
            row.get("family_label") or _dataset_catalog_identity(str(row["name"]))["family_label"]
        ).casefold(),
        "versionLabel": _dataset_catalog_version_sort_key,
        "platformLabel": lambda row: str(row.get("platform_label") or "").casefold(),
        "splitLabel": lambda row: str(row.get("split_label") or "").casefold(),
        "renderLabel": lambda row: str(row.get("render_label") or "").casefold(),
    }
    sort_key = sort_key_by_field.get(sort_field, sort_key_by_field["createdEpoch"])
    filtered.sort(key=sort_key, reverse=sort_direction != "asc")

    total = len(filtered)
    last_page = max(1, (total + page_size - 1) // page_size)
    page = min(requested_page, last_page)
    page_candidates = filtered[(page - 1) * page_size : page * page_size]
    registry_by_root = _training_registry_entries_by_root(repo_root)
    selection_counts = dataset_role_counts(repo_root)
    _set_datasets_progress(
        repo_root,
        status="loading",
        percent=5,
        completed=0,
        total=len(page_candidates),
        message=f"Loading catalog page {page}",
    )
    rows = []
    for index, candidate in enumerate(page_candidates, start=1):
        item = _load_dataset_catalog_candidate(repo_root, candidate)
        rows.append(
            _dataset_catalog_remote_row(
                candidate,
                item,
                registry_by_root=registry_by_root,
                marked_names_by_role=marked_names_by_role,
            )
        )
        _set_datasets_progress(
            repo_root,
            status="loading",
            percent=min(98, 5 + round(index / max(len(page_candidates), 1) * 92)),
            completed=index,
            total=len(page_candidates),
            message=f"Loading {candidate['name']}",
        )
    _set_datasets_progress(
        repo_root,
        status="complete",
        percent=100,
        completed=len(page_candidates),
        total=len(page_candidates),
        message=f"Catalog page {page} ready",
    )
    return {
        "last_page": last_page,
        "last_row": total,
        "data": rows,
        "page": page,
        "size": page_size,
        "total": total,
        "marked_only": bool(filters["marked_only"]),
        "platform_count": len({candidate["platform"] for candidate in filtered}),
        "selection_counts": selection_counts,
        "trainable_set_count": selection_counts["training"],
    }


def _dataset_catalog_named_item_payload(repo_root: Path, name: str) -> dict[str, Any]:
    requested_name = name.strip()
    if not requested_name:
        raise ValueError("dataset name is required")
    candidate = next(
        (
            item
            for item in _dataset_catalog_candidates(repo_root)
            if str(item.get("name") or "") == requested_name
        ),
        None,
    )
    if candidate is None:
        raise FileNotFoundError(f"dataset is not registered: {requested_name}")
    marked_names_by_role = {
        role: selected_catalog_names(repo_root, role)
        for role in ("training", "validation", "loop_test")
    }
    return {
        "data": _dataset_catalog_remote_row(
            candidate,
            _load_dataset_catalog_candidate(repo_root, candidate),
            registry_by_root=_training_registry_entries_by_root(repo_root),
            marked_names_by_role=marked_names_by_role,
        )
    }


def _load_dataset_catalog_candidate(repo_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    loader = candidate["loader"]
    if loader == "photoreal_jsonl":
        return _so101_photoreal_dataset_catalog_item(repo_root, candidate["name"], candidate["root"])
    if loader == "mycobot":
        return _mycobot_dataset_catalog_item(repo_root, candidate["name"], candidate["root"])
    return _dataset_catalog_item(
        repo_root,
        candidate["name"],
        candidate["root"],
        category=candidate["category"],
    )


def _dataset_catalog_remote_row(
    candidate: dict[str, Any],
    item: dict[str, Any],
    *,
    registry_by_root: dict[Path, list[DatasetRegistryEntry]],
    marked_names_by_role: dict[str, set[str]] | None = None,
    marked_trainable_roots: set[Path] | None = None,
) -> dict[str, Any]:
    summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
    identity = _dataset_catalog_identity(str(candidate["name"]))
    role_eligibility = {}
    for role in ("training", "validation", "loop_test"):
        eligible, reason, _registry_entry = _dataset_role_eligibility(
            candidate,
            role,  # type: ignore[arg-type]
            registry_by_root,
        )
        role_eligibility[role] = {"eligible": eligible, "reason": reason}
    resolved_root = Path(candidate["root"]).resolve()
    marked_names_by_role = marked_names_by_role or {
        "training": set(),
        "validation": set(),
        "loop_test": set(),
    }
    marked_roles = [
        role
        for role in ("training", "validation", "loop_test")
        if str(candidate["name"]) in marked_names_by_role.get(role, set())
    ]
    if (
        marked_trainable_roots is not None
        and resolved_root in marked_trainable_roots
        and "training" not in marked_roles
    ):
        marked_roles.insert(0, "training")
    training_eligibility = role_eligibility["training"]
    return {
        "name": candidate["name"],
        "familyKey": candidate.get("family_key") or identity["family_key"],
        "familyLabel": candidate.get("family_label") or identity["family_label"],
        "versionKey": candidate.get("version_key") or identity["version_key"],
        "versionLabel": candidate.get("version_label") or identity["version_label"],
        "category": candidate["category"],
        "platform": candidate["platform"],
        "platformLabel": candidate["platform_label"],
        "splitKey": candidate["split_key"],
        "splitLabel": candidate["split_label"],
        "renderKey": candidate["render_key"],
        "renderLabel": candidate["render_label"],
        "createdAt": item.get("created_at") or candidate.get("created_at"),
        "createdEpoch": float(item.get("created_at_epoch") or candidate.get("created_at_epoch") or 0),
        "episodes": int(summary.get("episodes") or item.get("episodes") or 0),
        "availability": item.get("status") or "incomplete",
        "detail": item.get("detail") or "",
        "roleEligibility": role_eligibility,
        "trainingEligible": training_eligibility["eligible"],
        "trainingEligibilityReason": training_eligibility["reason"],
        "markedRoles": marked_roles,
        "markedTraining": "training" in marked_roles,
        "markedValidation": "validation" in marked_roles,
        "markedLoopTest": "loop_test" in marked_roles,
        "markedTrainable": "training" in marked_roles,
        "summary": summary,
    }


def _datasets_progress_payload(repo_root: Path) -> dict[str, Any]:
    cache_key = str(repo_root.resolve())
    with DATASETS_PROGRESS_LOCK:
        progress = DATASETS_PROGRESS.get(cache_key)
        if progress is None:
            return {
                "status": "idle",
                "percent": 0,
                "completed": 0,
                "total": 0,
                "message": "Waiting to load catalog",
            }
        return dict(progress)


def _set_datasets_progress(
    repo_root: Path,
    *,
    status: str,
    percent: int | None = None,
    completed: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    cache_key = str(repo_root.resolve())
    with DATASETS_PROGRESS_LOCK:
        current = DATASETS_PROGRESS.get(
            cache_key,
            {
                "status": "idle",
                "percent": 0,
                "completed": 0,
                "total": 0,
                "message": "Waiting to load catalog",
            },
        )
        updated = dict(current)
        updated["status"] = status
        if percent is not None:
            updated["percent"] = max(0, min(100, int(percent)))
        if completed is not None:
            updated["completed"] = max(0, int(completed))
        if total is not None:
            updated["total"] = max(0, int(total))
        if message is not None:
            updated["message"] = message
        updated["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        DATASETS_PROGRESS[cache_key] = updated


def _build_datasets_payload(
    repo_root: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    official_roots = _official_dataset_roots(repo_root)
    skill_roots = _skill_dataset_roots(repo_root)
    recipe_roots = _generation_recipe_dataset_roots(repo_root)
    photoreal_roots = _discover_so101_photoreal_datasets(repo_root)
    photoreal_lerobot_roots = _discover_so101_photoreal_lerobot_datasets(repo_root)
    closed_loop_views = _generation_closed_loop_views(repo_root)
    temporary_roots = _discover_temporary_datasets(repo_root)
    mycobot_roots = _discover_mycobot_datasets(repo_root)
    recipe_paths = {_resolve_dataset_path(repo_root, root) for root in recipe_roots.values()}
    official_candidates = [
        (split, root)
        for split, root in official_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    skill_candidates = [
        (split, root)
        for split, root in skill_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    photoreal_candidates = [
        (split, root)
        for split, root in photoreal_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    photoreal_lerobot_candidates = [
        (split, root)
        for split, root in photoreal_lerobot_roots.items()
        if _resolve_dataset_path(repo_root, root) not in recipe_paths
    ]
    archived_candidates = [(split, DATASETS[split]) for split in ARCHIVED_DATASET_SPLITS]
    total_items = sum(
        (
            len(official_candidates),
            len(skill_candidates),
            len(photoreal_candidates),
            len(photoreal_lerobot_candidates),
            len(recipe_roots),
            len(closed_loop_views),
            len(archived_candidates),
            len(temporary_roots),
            len(mycobot_roots),
        )
    )
    completed_items = 0

    def record_item(item: dict[str, Any], split: str) -> dict[str, Any]:
        nonlocal completed_items
        completed_items += 1
        if progress is not None:
            progress(
                completed_items,
                total_items,
                f"Reading {split}",
            )
        return item

    if progress is not None:
        progress(0, total_items, "Reading dataset metadata")

    official_items = [
        record_item(
            _dataset_catalog_item(repo_root, split, root, category="official"),
            split,
        )
        for split, root in official_candidates
    ]
    skill_items = [
        record_item(
            _dataset_catalog_item(repo_root, split, root, category="skill"),
            split,
        )
        for split, root in skill_candidates
    ]
    photoreal_items = [
        record_item(
            _so101_photoreal_dataset_catalog_item(repo_root, split, root),
            split,
        )
        for split, root in photoreal_candidates
    ]
    photoreal_items.extend(
        record_item(
            _dataset_catalog_item(repo_root, split, root, category="photoreal"),
            split,
        )
        for split, root in photoreal_lerobot_candidates
    )
    generated_items: list[dict[str, Any]] = []
    closed_loop_items: list[dict[str, Any]] = []
    official_paths = {_resolve_dataset_path(repo_root, root) for root in official_roots.values()}
    skill_paths = {_resolve_dataset_path(repo_root, root) for root in skill_roots.values()}
    for split, root in recipe_roots.items():
        resolved = _resolve_dataset_path(repo_root, root)
        if _is_photoreal_lerobot_dataset(resolved):
            category = "photoreal"
        else:
            category = (
                "official"
                if resolved in official_paths
                else "skill"
                if resolved in skill_paths
                else "generated"
            )
        item = record_item(
            _dataset_catalog_item(repo_root, split, root, category=category),
            split,
        )
        if category == "official":
            official_items.append(item)
        elif category == "skill":
            skill_items.append(item)
        elif category == "photoreal":
            photoreal_items.append(item)
        else:
            generated_items.append(item)
    for split, view in closed_loop_views.items():
        closed_loop_items.append(
            record_item(
                _dataset_catalog_item(
                    repo_root,
                    split,
                    Path(view["root"]),
                    category="closed_loop",
                ),
                split,
            )
        )
    archived_items = [
        record_item(
            _dataset_catalog_item(repo_root, split, root, category="archived"),
            split,
        )
        for split, root in archived_candidates
    ]
    archived_visible_items = [item for item in archived_items if item["status"] == "available"]
    temporary_items = [
        record_item(
            _dataset_catalog_item(repo_root, split, root, category="temporary"),
            split,
        )
        for split, root in temporary_roots.items()
    ]
    mycobot_items = [
        record_item(_mycobot_dataset_catalog_item(repo_root, split, root), split)
        for split, root in mycobot_roots.items()
    ]
    if progress is not None:
        progress(total_items, total_items, "Finalizing catalog")
    for item in [
        *official_items,
        *skill_items,
        *generated_items,
        *closed_loop_items,
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
                "id": "closed_loop",
                "title": "Closed-loop test cases",
                "description": "Validation-derived episode starts used by closed-loop evaluation.",
                "items": closed_loop_items,
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
        **_dataset_creation_metadata(resolved),
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
        **_dataset_creation_metadata(Path(dataset["root"])),
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
        "camera_contract": _camera_contract_for_keys(dataset["camera_keys"]),
        "photoreal_preview": photoreal_preview,
    }


def _camera_contract_for_keys(camera_keys: list[str]) -> dict[str, str]:
    return {
        key: SO101_CAMERA_CONTRACT.get(key, key.rsplit(".", 1)[-1])
        for key in camera_keys
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
            if not _is_photoreal_lerobot_dataset(dataset_root):
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
        **_dataset_creation_metadata(resolved),
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
        **_dataset_creation_metadata(Path(dataset["root"])),
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


def _generation_closed_loop_views(repo_root: Path) -> dict[str, dict[str, Path]]:
    views: dict[str, dict[str, Path]] = {}
    registry = scan_dataset_registry(repo_root, inspect_artifacts=True)
    for entry in registry.entries:
        if entry.status != "available" or not entry.closed_loop_start:
            continue
        report = _resolve_dataset_path(repo_root, Path(entry.closed_loop_start))
        if not report.is_file():
            continue
        name = f"{entry.dataset_id}_loop_test"
        if name in views:
            name = f"{entry.dataset_id}_{entry.split}_loop_test"
        views[name] = {
            "root": Path(entry.absolute_root),
            "report": report,
        }
    return views


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
        **_dataset_creation_metadata(resolved),
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


def _dataset_creation_metadata(root: Path) -> dict[str, Any]:
    explicit_paths = (
        root / "meta" / "info.json",
        root / "manifest.json",
        root / "photoreal_lerobot_manifest.json",
        root / "so101_lerobot_export_report.json",
        root / "so101_lerobot_merge_report.json",
    )
    for path in explicit_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for field in ("created_at", "generated_at", "completed_at", "finished_at"):
            timestamp = _datetime_timestamp(payload.get(field))
            if timestamp is not None:
                return _creation_metadata_payload(
                    timestamp,
                    source=f"{path.relative_to(root)}:{field}",
                )

    try:
        root_stat = root.stat()
    except OSError:
        return {"created_at": None, "created_at_epoch": None, "created_at_source": None}

    birthtime = getattr(root_stat, "st_birthtime", None)
    if isinstance(birthtime, (int, float)) and birthtime > 0:
        return _creation_metadata_payload(float(birthtime), source="filesystem_birthtime")

    for path in (root / "meta" / "info.json", root / "manifest.json"):
        try:
            return _creation_metadata_payload(
                path.stat().st_mtime,
                source=f"{path.relative_to(root)}:mtime",
            )
        except OSError:
            continue
    return _creation_metadata_payload(root_stat.st_mtime, source="filesystem_mtime")


def _datetime_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _creation_metadata_payload(timestamp: float, *, source: str) -> dict[str, Any]:
    return {
        "created_at": datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "created_at_epoch": timestamp,
        "created_at_source": source,
    }


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
        "camera_contract": _camera_contract_for_keys(dataset["camera_keys"]),
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
        **_dataset_creation_metadata(Path(dataset["root"])),
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
    closed_loop_view = _generation_closed_loop_views(repo_root).get(split)
    if closed_loop_view is not None:
        selection = json.loads(Path(closed_loop_view["report"]).read_text(encoding="utf-8"))
        source_indices = [
            int(row["source_validation_episode_index"])
            for row in selection.get("episodes", [])
        ]
        if not source_indices:
            raise ValueError(f"closed-loop view has no selected episodes: {split}")
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(f"closed-loop view repeats source episodes: {split}")
        if min(source_indices) < 0 or max(source_indices) >= len(episodes):
            raise ValueError(f"closed-loop view source episode is out of range: {split}")
        episodes = [episodes[index] for index in source_indices]
        info = dict(info)
        info["total_episodes"] = len(episodes)
        info["total_frames"] = sum(int(row["length"]) for row in episodes)
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
    roots.update(
        {
            split: Path(view["root"]).resolve()
            for split, view in _generation_closed_loop_views(repo_root).items()
        }
    )
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
    for config_path in _training_config_paths(repo_root):
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
                normalized = copy.deepcopy(test_case)
                renderer = closed_loop.get("observation_renderer")
                if isinstance(renderer, dict):
                    normalized["_observation_renderer_contract"] = copy.deepcopy(
                        renderer
                    )
                merged[str(test_case["id"])] = normalized
    return list(merged.values())


def _training_config_paths(repo_root: Path) -> list[Path]:
    paths = {
        (repo_root / relative_path).resolve()
        for relative_path in TRAINING_CONFIGS
    }
    paths.update((repo_root / "configs/so101/training").glob("*.json"))
    return sorted(path.resolve() for path in paths if path.is_file())


def _closed_loop_test_case_for_candidate(
    repo_root: Path,
    *,
    dataset_root: Path,
    start_report: Path | None,
    test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    dataset_root = dataset_root.resolve()
    resolved_report = start_report.resolve() if start_report else None
    if resolved_report is not None:
        contract_path = contract_path_for_start_report(resolved_report)
        if contract_path.is_file():
            contract = load_executable_loop_test_contract(
                contract_path,
                repo_root=repo_root,
                expected_start_report=resolved_report,
            )
            test_case = copy.deepcopy(contract["test_case"])
            test_case["_observation_renderer_contract"] = copy.deepcopy(
                contract["observation_renderer"]
            )
            test_case["_contract_path"] = str(contract_path)
            return test_case
    for test_case in test_cases or _official_closed_loop_test_cases(repo_root):
        configured_report = test_case.get("start_report_path")
        if configured_report and resolved_report is not None:
            candidate_report = _resolve_dataset_path(repo_root, Path(str(configured_report)))
            if candidate_report == resolved_report:
                return copy.deepcopy(test_case)
        start_dataset = test_case.get("start_dataset")
        if isinstance(start_dataset, dict) and start_dataset.get("root"):
            configured_root = _resolve_dataset_path(
                repo_root,
                Path(str(start_dataset["root"])),
            )
            if configured_root == dataset_root:
                return copy.deepcopy(test_case)
    return None


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
  <link rel="stylesheet" href="/vendor/tabulator.min.css">
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
    .dataset-catalog-header { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .catalog-loading {
      position: absolute;
      inset: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 10px;
      align-content: center;
      justify-content: center;
      align-items: center;
      padding: 18px max(18px, calc((100% - 440px) / 2));
      border: 0;
      border-radius: 10px;
      background: rgba(239, 246, 255, 0.94);
      backdrop-filter: blur(2px);
      color: #1e3a8a;
    }
    .catalog-loading[hidden] { display: none; }
    .catalog-loading.error { border-color: #fecaca; background: #fff1f2; color: #991b1b; }
    .catalog-loading-spinner {
      width: 18px;
      height: 18px;
      border: 3px solid #bfdbfe;
      border-top-color: #2563eb;
      border-radius: 50%;
      animation: catalog-spin 700ms linear infinite;
    }
    .catalog-loading.error .catalog-loading-spinner { display: none; }
    .catalog-loading-copy { display: grid; gap: 6px; min-width: 0; }
    .catalog-loading-line { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; }
    .catalog-loading-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .catalog-loading-percent { min-width: 42px; text-align: right; font-variant-numeric: tabular-nums; }
    .catalog-loading progress {
      width: 100%;
      height: 8px;
      border: 0;
      border-radius: 999px;
      overflow: hidden;
      background: #dbeafe;
      accent-color: #2563eb;
    }
    .catalog-loading progress::-webkit-progress-bar { background: #dbeafe; border-radius: 999px; }
    .catalog-loading progress::-webkit-progress-value { background: #2563eb; border-radius: 999px; transition: width 180ms ease; }
    @keyframes catalog-spin { to { transform: rotate(360deg); } }
    .dataset-catalog-shell {
      position: relative;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
    }
    .dataset-catalog-scroll { overflow-x: auto; }
    #datasetCatalogGrid { min-width: 1140px; }
    #datasetCatalogGrid.tabulator {
      border: 0;
      background: #fff;
      color: var(--ink);
      font-size: 12px;
    }
    #datasetCatalogGrid .tabulator-header {
      border-bottom: 1px solid var(--line);
      background: #edf3ff;
      color: #334155;
      font-weight: 800;
    }
    #datasetCatalogGrid .tabulator-header .tabulator-col { background: #edf3ff; }
    #datasetCatalogGrid .tabulator-header .tabulator-col:hover { background: #e2eaff; }
    #datasetCatalogGrid .tabulator-header-filter input,
    #datasetCatalogGrid .tabulator-header-filter select {
      min-height: 30px;
      padding: 4px 7px;
      border: 1px solid #bdc9da;
      border-radius: 6px;
      background: #fff;
      color: #24324a;
      font: inherit;
    }
    #datasetCatalogGrid .tabulator-row {
      cursor: pointer;
      border-bottom: 1px solid #e2e8f0;
      background: #fff;
      transition: background 120ms ease, box-shadow 120ms ease;
    }
    #datasetCatalogGrid .tabulator-row:nth-child(even) { background: #fbfdff; }
    #datasetCatalogGrid .tabulator-row:hover { background: #f4f7ff; }
    #datasetCatalogGrid .tabulator-row.dataset-selected {
      background: var(--data-soft);
      box-shadow: inset 4px 0 0 var(--data);
    }
    #datasetCatalogGrid .tabulator-cell { padding: 8px 9px; border-right-color: #edf1f7; }
    #datasetCatalogGrid .tabulator-footer { display: none; }
    #datasetCatalogGrid .tabulator-placeholder { min-height: 120px; color: var(--text-soft); }
    .catalog-pager {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 48px;
      padding: 7px 10px;
      border-top: 1px solid var(--line);
      background: #f8fafc;
    }
    .catalog-page-button,
    .playback-icon-button {
      display: inline-grid;
      place-items: center;
      width: 36px;
      min-width: 36px;
      height: 34px;
      min-height: 34px;
      padding: 0;
      font-size: 16px;
      line-height: 1;
    }
    .catalog-page-button:disabled { cursor: not-allowed; opacity: 0.42; }
    .catalog-page-status { min-width: 150px; color: #475569; font-size: 12px; text-align: center; font-variant-numeric: tabular-nums; }
    .dataset-name-cell { color: var(--ink); overflow-wrap: anywhere; }
    .dataset-family-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .dataset-family { display: block; font-weight: 820; }
    .dataset-id { display: block; margin-top: 2px; color: #64748b; font-size: 10px; font-weight: 560; }
    .trainable-set-badge {
      display: inline-flex;
      align-items: center;
      flex: 0 0 auto;
      min-height: 20px;
      padding: 2px 6px;
      border: 1px solid #86efac;
      border-radius: 999px;
      background: #ecfdf5;
      color: #047857;
      font-size: 9px;
      font-weight: 850;
      white-space: nowrap;
    }
    .trainable-set-badge.validation-role {
      border-color: #93c5fd;
      background: #eff6ff;
      color: #1d4ed8;
    }
    .trainable-set-badge.loop-test-role {
      border-color: #5eead4;
      background: #f0fdfa;
      color: #0f766e;
    }
    .dataset-select-checkbox { width: 16px; height: 16px; margin: 0; accent-color: #2563eb; cursor: pointer; }
    .dataset-bulk-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      padding: 8px 10px;
      border: 1px solid #d7e0ed;
      border-radius: 9px;
      background: #f8fafc;
    }
    .dataset-bulk-toolbar select { min-height: 34px; min-width: 220px; padding: 5px 8px; }
    .dataset-bulk-toolbar button { min-height: 34px; padding: 5px 11px; }
    .dataset-filter-toggle {
      border-color: #bfdbfe;
      background: #eff6ff;
      color: #1e3a8a;
      white-space: nowrap;
    }
    .dataset-filter-toggle[aria-pressed="true"] {
      border-color: #2563eb;
      background: #2563eb;
      color: #fff;
      box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.14);
    }
    .dataset-bulk-count { color: #334155; font-size: 12px; font-weight: 800; }
    .dataset-bulk-spacer { flex: 1 1 auto; }
    .dataset-cache-note { color: #64748b; font-size: 11px; }
    .dataset-version-cell { color: #1d4ed8; font-weight: 780; white-space: nowrap; }
    .dataset-created-cell { color: #475569; white-space: nowrap; }
    .dataset-badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid #d6deeb;
      border-radius: 999px;
      background: #f8fafc;
      color: #334155;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .dataset-badge.train { background: #eef2ff; border-color: #c7d2fe; color: #3730a3; }
    .dataset-badge.valid { background: #ecfdf5; border-color: #a7f3d0; color: #047857; }
    .dataset-badge.closed_loop { background: var(--loop-soft); border-color: #99ddd0; color: #115e59; }
    .dataset-badge.simulation { background: #f8fafc; border-color: #cbd5e1; color: #475569; }
    .dataset-badge.photoreal { background: #fff7ed; border-color: #fed7aa; color: #c2410c; }
    .dataset-badge.real { background: #fdf2f8; border-color: #fbcfe8; color: #be185d; }
    .selected-dataset-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 8px 0 12px;
      color: var(--text-soft);
      font-size: 13px;
    }
    .selected-dataset-name { color: var(--ink); font-weight: 850; overflow-wrap: anywhere; }
    .playback-controls { grid-template-columns: minmax(210px, 1fr) minmax(210px, 1fr) auto; gap: 9px; }
    .playback-actions { display: flex; align-items: end; gap: 6px; }
    .playback-actions .fps-control { width: 68px; }
    .playback-actions .fps-control select { min-height: 34px; padding: 5px 7px; }
    .playback-icon-button.playing { border-color: var(--data); background: var(--data-soft); color: #1d4ed8; }
    .range-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .range-value {
      min-width: 34px;
      padding: 2px 7px;
      border-radius: 6px;
      background: var(--data-soft);
      color: #1d4ed8;
      text-align: center;
      font-variant-numeric: tabular-nums;
      font-weight: 850;
    }
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
      color: inherit;
      text-decoration: none;
      cursor: pointer;
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
    button:not(.tab-button):not(.zoom-btn):not(.dataset-sort-button) { background: #f8fafc; transition: background 120ms ease, border-color 120ms ease, transform 120ms ease; }
    button:not(.tab-button):not(.zoom-btn):not(.dataset-sort-button):hover { background: #eef4ff; border-color: #9fb7ee; transform: translateY(-1px); }
    button.dataset-delete-button {
      min-height: 30px;
      padding: 5px 9px;
      border-color: #fecaca;
      background: #fff5f5;
      color: var(--bad);
      font-size: 11px;
    }
    button.dataset-delete-button:hover:not(:disabled) {
      border-color: #f87171;
      background: #fee2e2;
      transform: none;
    }
    button.dataset-delete-button:disabled {
      cursor: not-allowed;
      border-color: #e2e8f0;
      background: #f8fafc;
      color: #94a3b8;
      opacity: 0.78;
    }
    .dataset-action-status {
      margin: 10px 0 0;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-muted);
      color: #334155;
      font-size: 12px;
      font-weight: 750;
    }
    .dataset-action-status[hidden] { display: none; }
    .dataset-action-status.success { border-color: #86efac; background: #f0fdf4; color: var(--ok); }
    .dataset-action-status.error { border-color: #fca5a5; background: #fef2f2; color: var(--bad); }
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
	      .app-tabs, .controls, .cameras, .loop-grid, .manager-grid, .metric-grid, .loop-controls, .sim-controls, .sim-start-controls, .sim-policy-controls, .sim-run-controls { grid-template-columns: 1fr; }
      .playback-controls { grid-template-columns: 1fr; }
      .playback-actions { justify-content: flex-start; }
      .catalog-loading { padding: 16px; }
      .dataset-catalog-header { align-items: start; flex-direction: column; }
      #datasetCatalogGrid { min-width: 980px; }
      #datasetCatalogGrid.tabulator { font-size: 11px; }
      #datasetCatalogGrid .tabulator-cell { padding: 7px 5px; }
      #datasetCatalogGrid .tabulator-col-title { font-size: 10px; }
      .dataset-badge { padding: 2px 5px; font-size: 9px; }
      .dataset-bulk-toolbar { align-items: stretch; }
      .dataset-bulk-toolbar select { min-width: 0; flex: 1 1 190px; }
      .dataset-bulk-spacer { display: none; }
      button.dataset-delete-button { padding: 4px 5px; font-size: 10px; }
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
	      <div class="dataset-catalog-header">
	        <div>
	          <div class="prompt-label">Dataset catalog</div>
	          <p id="catalogMeta" class="meta"></p>
	        </div>
	        <span class="chip">Select one row to inspect its episodes and frames</span>
	      </div>
	      <div class="dataset-bulk-toolbar" role="group" aria-label="Dataset bulk actions">
	        <span id="catalogSelectedCount" class="dataset-bulk-count">0 selected</span>
	        <select id="catalogBulkAction" aria-label="Dataset bulk action">
	          <option value="">Choose action</option>
	          <option value="mark_training">Mark as training set</option>
	          <option value="remove_training">Remove from training set</option>
	          <option value="mark_validation">Mark as validation set</option>
	          <option value="remove_validation">Remove from validation set</option>
	          <option value="mark_loop_test">Mark as loop test set</option>
	          <option value="remove_loop_test">Remove from loop test set</option>
	          <option value="delete">Delete selected datasets</option>
	        </select>
	        <button id="catalogApplyBulkAction" type="button" disabled>Apply</button>
	        <button id="catalogMarkedOnly" class="dataset-filter-toggle" type="button" aria-pressed="false">Marked only</button>
	        <span class="dataset-bulk-spacer"></span>
	        <span id="catalogTrainingCount" class="dataset-bulk-count">Training 0</span>
	        <span id="catalogValidationCount" class="dataset-bulk-count">Validation 0</span>
	        <span id="catalogLoopTestCount" class="dataset-bulk-count">Loop test 0</span>
	        <span id="catalogCacheNote" class="dataset-cache-note">0 cached pages</span>
	      </div>
	      <div class="dataset-catalog-shell">
	        <div id="catalogLoading" class="catalog-loading" role="status" aria-live="polite" hidden>
	          <span class="catalog-loading-spinner" aria-hidden="true"></span>
	          <div class="catalog-loading-copy">
	            <div class="catalog-loading-line">
	              <span id="catalogLoadingText" class="catalog-loading-text">Preparing dataset catalog</span>
	              <strong id="catalogLoadingPercent" class="catalog-loading-percent">0%</strong>
	            </div>
	            <progress id="catalogLoadingBar" max="100" value="0" aria-label="Dataset catalog loading progress"></progress>
	          </div>
	        </div>
	        <div class="dataset-catalog-scroll">
	          <div id="datasetCatalogGrid" aria-label="Available datasets"></div>
	        </div>
	        <div class="catalog-pager" aria-label="Dataset catalog pages">
	          <button id="catalogPreviousPage" class="catalog-page-button" type="button" title="Previous page" aria-label="Previous page" disabled>&#8592;</button>
	          <span id="catalogPageStatus" class="catalog-page-status">Page 1 of 1</span>
	          <button id="catalogNextPage" class="catalog-page-button" type="button" title="Next page" aria-label="Next page" disabled>&#8594;</button>
	        </div>
	      </div>
	      <p id="datasetActionStatus" class="dataset-action-status" role="status" hidden></p>
	    </section>
	    <div id="datasetPanel" class="panel">
	    <section>
	      <div class="prompt-label">Dataset playback</div>
	      <select id="split" hidden aria-hidden="true"></select>
	      <div id="selectedDatasetMeta" class="selected-dataset-bar"></div>
      <div class="controls playback-controls">
	        <label>
	          <span class="range-heading">Episode <output id="episodeValue" class="range-value">0</output></span>
	          <input id="episode" type="range" min="0" max="0" value="0">
	        </label>
        <label>
          <span class="range-heading">Frame <output id="frameValue" class="range-value">0</output></span>
          <input id="frame" type="range" min="0" max="0" value="0">
        </label>
        <div class="playback-actions" role="group" aria-label="Playback controls">
          <button id="prev" class="playback-icon-button" type="button" title="Previous frame" aria-label="Previous frame">&#9198;</button>
          <button id="play" class="playback-icon-button" type="button" title="Play" aria-label="Play"><span aria-hidden="true">&#9654;</span></button>
          <button id="next" class="playback-icon-button" type="button" title="Next frame" aria-label="Next frame">&#9197;</button>
          <label class="fps-control">FPS<select id="fps"><option value="30" selected>30</option><option value="24">24</option><option value="12">12</option><option value="6">6</option></select></label>
        </div>
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
	  <script src="/vendor/tabulator.min.js"></script>
	  <script>
	    let datasets = {};
	    let datasetOrder = [];
	    let datasetPlatformByName = {};
	    let datasetPlatformLabelByName = {};
	    let datasetCategoryByName = {};
	    let datasetSplitByName = {};
	    let datasetRenderTypeByName = {};
	    let loopAnalyzerLoaded = false;
	    let simulatorConfig = { presets: [], training_runs: [], defaults: {} };
	    const split = document.getElementById("split");
	    const episode = document.getElementById("episode");
	    const frame = document.getElementById("frame");
	    const episodeValue = document.getElementById("episodeValue");
	    const frameValue = document.getElementById("frameValue");
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
	    const catalogMeta = document.getElementById("catalogMeta");
	    const catalogLoading = document.getElementById("catalogLoading");
	    const catalogLoadingText = document.getElementById("catalogLoadingText");
	    const catalogLoadingPercent = document.getElementById("catalogLoadingPercent");
	    const catalogLoadingBar = document.getElementById("catalogLoadingBar");
	    const datasetCatalogGrid = document.getElementById("datasetCatalogGrid");
	    const catalogPreviousPage = document.getElementById("catalogPreviousPage");
	    const catalogNextPage = document.getElementById("catalogNextPage");
	    const catalogPageStatus = document.getElementById("catalogPageStatus");
	    const catalogSelectedCount = document.getElementById("catalogSelectedCount");
	    const catalogBulkAction = document.getElementById("catalogBulkAction");
	    const catalogApplyBulkAction = document.getElementById("catalogApplyBulkAction");
	    const catalogMarkedOnlyButton = document.getElementById("catalogMarkedOnly");
	    const catalogTrainingCount = document.getElementById("catalogTrainingCount");
	    const catalogValidationCount = document.getElementById("catalogValidationCount");
	    const catalogLoopTestCount = document.getElementById("catalogLoopTestCount");
	    const catalogCacheNote = document.getElementById("catalogCacheNote");
	    const datasetActionStatus = document.getElementById("datasetActionStatus");
	    const selectedDatasetMeta = document.getElementById("selectedDatasetMeta");
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
	    let frameLoadGeneration = 0;
	    let simProcessingTimer = null;
	    let simTimelineRows = [];
	    let simTimelineTimer = null;
	    let simContinuationStartReportPath = null;
	    let trainingRunRows = [];
	    let catalogProgressTimer = null;
	    let catalogLoadingHideTimer = null;
	    let catalogProgressRequestInFlight = false;
	    let catalogLoadGeneration = 0;
	    let datasetCatalogTable = null;

	    function boundedUrlInteger(params, key, fallback = 0) {
	      const value = Number(params.get(key));
	      return Number.isInteger(value) && value >= 0 ? value : fallback;
	    }

	    function initialViewStateFromUrl() {
	      const params = new URLSearchParams(window.location.search);
	      const requestedView = String(params.get("view") || params.get("tab") || "viewer").toLowerCase();
	      const viewAliases = {
	        data: "viewer",
	        viewer: "viewer",
	        training: "training",
	        loop: "loop",
	        analyzer: "loop",
	        simulator: "simulator",
	      };
	      const allowedSortFields = new Set([
	        "createdEpoch", "name", "familyLabel", "versionLabel",
	        "platformLabel", "splitLabel", "renderLabel",
	      ]);
	      const requestedSort = String(params.get("sort") || "createdEpoch");
	      const requestedFps = String(params.get("fps") || "");
	      return {
	        view: viewAliases[requestedView] || "viewer",
	        dataset: String(params.get("dataset") || ""),
	        episode: boundedUrlInteger(params, "episode", 0),
	        frame: boundedUrlInteger(params, "frame", 0),
	        fps: new Set(["6", "12", "24", "30"]).has(requestedFps) ? requestedFps : "",
	        catalogPage: Math.max(1, boundedUrlInteger(params, "page", 1)),
	        markedOnly: new Set(["1", "true", "yes", "on"]).has(
	          String(params.get("marked") || "").toLowerCase(),
	        ),
	        catalogFilters: {
	          platformLabel: String(params.get("platform") || ""),
	          splitLabel: String(params.get("status") || ""),
	          renderLabel: String(params.get("type") || ""),
	          familyLabel: String(params.get("family") || ""),
	          versionLabel: String(params.get("version") || ""),
	        },
	        catalogSort: allowedSortFields.has(requestedSort) ? requestedSort : "createdEpoch",
	        catalogSortDirection: String(params.get("dir") || "desc").toLowerCase() === "asc" ? "asc" : "desc",
	        trainingId: String(params.get("run") || ""),
	      };
	    }

	    function catalogQueryFromViewState(state) {
	      const query = new URLSearchParams({
	        page: String(state.catalogPage),
	        size: "10",
	        sort: state.catalogSort,
	        dir: state.catalogSortDirection,
	      });
	      const urlKeyByField = {
	        platformLabel: "platform",
	        splitLabel: "status",
	        renderLabel: "type",
	        familyLabel: "family",
	        versionLabel: "version",
	      };
	      Object.entries(state.catalogFilters).forEach(([field, value]) => {
	        if (value) query.set(urlKeyByField[field], value);
	      });
	      if (state.markedOnly) query.set("marked", "1");
	      return query;
	    }

	    function initialCatalogHeaderFilters() {
	      return Object.entries(initialViewState.catalogFilters)
	        .filter(([, value]) => Boolean(value))
	        .map(([field, value]) => ({field, value}));
	    }

	    const initialViewState = initialViewStateFromUrl();
	    let currentAppTab = initialViewState.view;
	    let viewStateRestoring = true;
	    let initialViewRestoreStarted = false;
	    let selectedTrainingId = initialViewState.trainingId || null;
	    let catalogUrlQuery = catalogQueryFromViewState(initialViewState);
	    let catalogCurrentPage = initialViewState.catalogPage;
	    let catalogLastPage = 1;
	    let catalogTotalRows = 0;
	    let catalogMarkedOnly = initialViewState.markedOnly;
	    let catalogSelectionCounts = {training: 0, validation: 0, loop_test: 0};
	    const catalogSelectedNames = new Set();
	    const datasetCatalogPageCache = new Map();
	    const catalogPageCacheLimit = 20;

	    function syncCurrentViewUrl() {
	      if (viewStateRestoring) return;
	      const params = new URLSearchParams();
	      params.set("view", currentAppTab);
	      if (currentAppTab === "viewer") {
	        if (split.value) {
	          params.set("dataset", split.value);
	          params.set("episode", String(Math.max(0, Number(episode.value || 0))));
	          params.set("frame", String(Math.max(0, Number(frame.value || 0))));
	        }
	        params.set("page", String(Math.max(1, catalogCurrentPage)));
	        if (catalogMarkedOnly) params.set("marked", "1");
	        for (const key of ["platform", "status", "type", "family", "version", "sort", "dir"]) {
	          const value = catalogUrlQuery.get(key);
	          if (value) params.set(key, value);
	        }
	        if (fps.value) params.set("fps", fps.value);
	      } else if (currentAppTab === "training" && selectedTrainingId) {
	        params.set("run", selectedTrainingId);
	      }
	      const query = params.toString();
	      const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
	      if (`${window.location.pathname}${window.location.search}` !== nextUrl) {
	        window.history.replaceState({experimentManagerView: true}, "", nextUrl);
	      }
	    }

	    function trainingRunUrl(trainingId) {
	      const params = new URLSearchParams({view: "training", run: String(trainingId)});
	      return `${window.location.pathname}?${params.toString()}`;
	    }

	    function renderCatalogLoading(progress = {}) {
	      const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
	      const completed = Math.max(0, Number(progress.completed || 0));
	      const total = Math.max(0, Number(progress.total || 0));
	      const count = total > 0 ? ` · ${completed.toLocaleString()} / ${total.toLocaleString()}` : "";
	      catalogLoading.hidden = false;
	      catalogLoading.classList.toggle("error", progress.status === "error");
	      catalogLoadingText.textContent = `${progress.message || "Loading dataset catalog"}${count}`;
	      catalogLoadingPercent.textContent = `${Math.round(percent)}%`;
	      catalogLoadingBar.value = percent;
	    }

	    async function pollCatalogProgress(generation) {
	      if (generation !== catalogLoadGeneration || catalogProgressRequestInFlight) return;
	      catalogProgressRequestInFlight = true;
	      try {
	        const response = await fetch("/api/datasets/progress", { cache: "no-store" });
	        if (!response.ok) return;
	        const progress = await response.json();
	        if (generation === catalogLoadGeneration) renderCatalogLoading(progress);
	      } catch (_) {
	        // The main catalog request reports actionable failures.
	      } finally {
	        catalogProgressRequestInFlight = false;
	      }
	    }

	    function startCatalogLoading(message = "Loading dataset catalog") {
	      catalogLoadGeneration += 1;
	      const generation = catalogLoadGeneration;
	      if (catalogLoadingHideTimer) clearTimeout(catalogLoadingHideTimer);
	      if (catalogProgressTimer) clearInterval(catalogProgressTimer);
	      catalogProgressRequestInFlight = false;
	      renderCatalogLoading({ status: "loading", percent: 0, message });
	      catalogProgressTimer = setInterval(() => pollCatalogProgress(generation), 400);
	    }

	    function finishCatalogLoading() {
	      catalogLoadGeneration += 1;
	      if (catalogProgressTimer) clearInterval(catalogProgressTimer);
	      catalogProgressTimer = null;
	      renderCatalogLoading({ status: "complete", percent: 100, message: "Catalog ready" });
	      catalogLoadingHideTimer = setTimeout(() => {
	        catalogLoading.hidden = true;
	      }, 350);
	    }

	    function failCatalogLoading(error) {
	      catalogLoadGeneration += 1;
	      if (catalogProgressTimer) clearInterval(catalogProgressTimer);
	      catalogProgressTimer = null;
	      renderCatalogLoading({
	        status: "error",
	        percent: Number(catalogLoadingBar.value || 0),
	        message: `Catalog loading failed: ${error?.message || error}`,
	      });
	    }

    async function init(options = {}) {
      if (!options.preserveStatus) setDatasetActionStatus("");
      if (!options.preserveLoadedDatasets) {
        datasets = {};
        datasetOrder = [];
        datasetPlatformByName = {};
        datasetPlatformLabelByName = {};
        datasetCategoryByName = {};
        datasetSplitByName = {};
        datasetRenderTypeByName = {};
        split.innerHTML = "";
      }
      renderDatasetCatalog();
      if (datasetCatalogTable && options.reload) {
        await datasetCatalogTable.setData();
      }
    }

	    function syncAppTab(tabName, options = {}) {
	      currentAppTab = tabName;
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
	        if (!simulatorConfig.presets.length) initSimulator();
	        if (options.updateUrl !== false) syncCurrentViewUrl();
	        return;
	      }
	      if (showTraining) {
	        loadTrainingRuns();
	        if (options.updateUrl !== false) syncCurrentViewUrl();
	        return;
	      }
	      if (showLoop) {
	        initLoopAnalyzer();
	        if (options.updateUrl !== false) syncCurrentViewUrl();
	        return;
	      }
	      syncDataViewer();
	      if (options.updateUrl !== false) syncCurrentViewUrl();
	    }

	    function syncDataViewer() {
	      datasetPanel.hidden = false;
	      loopPanel.hidden = true;
	      trainingPanel.hidden = true;
	      simPanel.hidden = true;
	      renderDatasetCatalog();
      if (split.value) {
        syncEpisodeRange();
        renderPhotorealShortcuts();
        loadFrame();
      } else {
        meta.textContent = "No datasets are available in the catalog.";
        promptText.textContent = "";
        cameras.innerHTML = "";
        photorealShortcuts.innerHTML = "";
        photorealPanel.hidden = true;
        photorealCameras.innerHTML = "";
        jointRows.innerHTML = "";
        selectedDatasetMeta.innerHTML = "";
	      }
	    }

	    async function ensureDeepLinkedDataset(name) {
	      if (!name || datasets[name]) return Boolean(datasets[name]);
	      const response = await fetch(`/api/datasets/catalog/item?name=${encodeURIComponent(name)}`, {
	        cache: "no-store",
	      });
	      const payload = await response.json();
	      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
	      if (!payload.data) throw new Error(`dataset catalog item is missing: ${name}`);
	      registerCatalogRow(payload.data);
	      return Boolean(datasets[name]);
	    }

	    async function restoreInitialViewState() {
	      if (initialViewRestoreStarted) return;
	      initialViewRestoreStarted = true;
	      try {
	        if (initialViewState.fps) fps.value = initialViewState.fps;
	        if (initialViewState.dataset) {
	          await ensureDeepLinkedDataset(initialViewState.dataset);
	          selectDataset(initialViewState.dataset, {
	            episode: initialViewState.episode,
	            frame: initialViewState.frame,
	            fps: initialViewState.fps,
	          });
	        }
	      } catch (error) {
	        setDatasetActionStatus(`Deep link could not load ${initialViewState.dataset}: ${error.message || error}`, "error");
	        const firstReady = datasetCatalogTable?.getRows()
	          .map(row => row.getData())
	          .find(row => row.availability === "available" && datasets[row.name]);
	        if (firstReady) selectDataset(firstReady.name);
	      } finally {
	        viewStateRestoring = false;
	        syncCurrentViewUrl();
	      }
	    }

	    function formatDatasetCreatedAt(value) {
	      const date = new Date(value || "");
	      if (!Number.isFinite(date.getTime())) return "Unknown";
	      const pad = number => String(number).padStart(2, "0");
	      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
	    }


	    function syncDatasetCatalogSelection() {
	      if (!datasetCatalogTable) return;
	      datasetCatalogTable.getRows().forEach(row => {
	        const selected = row.getData().name === split.value;
	        const element = row.getElement();
	        element.classList.toggle("dataset-selected", selected);
	        element.setAttribute("aria-selected", String(selected));
	      });
	    }

	    function catalogFiltersFromParams(params) {
	      const values = {};
	      const visit = filter => {
	        if (Array.isArray(filter)) {
	          filter.forEach(visit);
	          return;
	        }
	        if (filter?.field && filter.value !== undefined && filter.value !== null && String(filter.value) !== "") {
	          values[filter.field] = String(filter.value);
	        }
	      };
	      visit(params?.filter || params?.filters || []);
	      return values;
	    }

	    function catalogQueryFromParams(params = {}) {
	      const query = new URLSearchParams({
	        page: String(params.page || 1),
	        size: String(params.size || 10),
	      });
	      const filters = catalogFiltersFromParams(params);
	      if (filters.platformLabel) query.set("platform", filters.platformLabel);
	      if (filters.splitLabel) query.set("status", filters.splitLabel);
	      if (filters.renderLabel) query.set("type", filters.renderLabel);
	      if (filters.familyLabel) query.set("family", filters.familyLabel);
	      if (filters.versionLabel) query.set("version", filters.versionLabel);
	      if (filters.name) query.set("name", filters.name);
	      if (catalogMarkedOnly) query.set("marked", "1");
	      const sorters = params.sorters || params.sort || [];
	      if (Array.isArray(sorters) && sorters.length) {
	        query.set("sort", sorters[0].field || "createdEpoch");
	        query.set("dir", sorters[0].dir || "desc");
	      }
	      return query;
	    }

	    function cloneCatalogPayload(payload) {
	      return JSON.parse(JSON.stringify(payload));
	    }

	    function catalogCacheKey(url, query) {
	      return `${url}?${query.toString()}`;
	    }

	    function getCachedCatalogPage(key) {
	      const payload = datasetCatalogPageCache.get(key);
	      if (!payload) return null;
	      datasetCatalogPageCache.delete(key);
	      datasetCatalogPageCache.set(key, payload);
	      return cloneCatalogPayload(payload);
	    }

	    function cacheCatalogPage(key, payload) {
	      datasetCatalogPageCache.delete(key);
	      datasetCatalogPageCache.set(key, cloneCatalogPayload(payload));
	      while (datasetCatalogPageCache.size > catalogPageCacheLimit) {
	        datasetCatalogPageCache.delete(datasetCatalogPageCache.keys().next().value);
	      }
	      updateBulkActionUi();
	    }

	    function clearCatalogPageCache() {
	      datasetCatalogPageCache.clear();
	      updateBulkActionUi();
	    }

	    function hideCatalogLoadingImmediately() {
	      catalogLoadGeneration += 1;
	      if (catalogProgressTimer) clearInterval(catalogProgressTimer);
	      if (catalogLoadingHideTimer) clearTimeout(catalogLoadingHideTimer);
	      catalogProgressTimer = null;
	      catalogLoadingHideTimer = null;
	      catalogLoading.hidden = true;
	    }

	    function currentCatalogPageNames() {
	      if (!datasetCatalogTable) return [];
	      return datasetCatalogTable.getRows().map(row => row.getData().name).filter(Boolean);
	    }

	    function updateBulkActionUi() {
	      const currentNames = currentCatalogPageNames();
	      const selectedOnPage = currentNames.filter(name => catalogSelectedNames.has(name));
	      const pageCheckbox = datasetCatalogGrid.querySelector(".dataset-select-page-checkbox");
	      if (pageCheckbox) {
	        pageCheckbox.checked = currentNames.length > 0 && selectedOnPage.length === currentNames.length;
	        pageCheckbox.indeterminate = selectedOnPage.length > 0 && selectedOnPage.length < currentNames.length;
	      }
	      datasetCatalogGrid.querySelectorAll(".dataset-row-select-checkbox").forEach(checkbox => {
	        checkbox.checked = catalogSelectedNames.has(checkbox.dataset.datasetName || "");
	      });
	      catalogSelectedCount.textContent = `${catalogSelectedNames.size.toLocaleString()} selected`;
	      catalogTrainingCount.textContent = `Training ${Number(catalogSelectionCounts.training || 0).toLocaleString()}`;
	      catalogValidationCount.textContent = `Validation ${Number(catalogSelectionCounts.validation || 0).toLocaleString()}`;
	      catalogLoopTestCount.textContent = `Loop test ${Number(catalogSelectionCounts.loop_test || 0).toLocaleString()}`;
	      catalogCacheNote.textContent = `${datasetCatalogPageCache.size.toLocaleString()} cached pages`;
	      catalogMarkedOnlyButton.setAttribute("aria-pressed", String(catalogMarkedOnly));
	      catalogMarkedOnlyButton.title = catalogMarkedOnly
	        ? "Show all datasets"
	        : "Show datasets marked for training, validation, or loop testing";
	      catalogApplyBulkAction.disabled = (
	        catalogSelectedNames.size === 0
	        || !catalogBulkAction.value
	        || !isPrivateViewerHost()
	      );
	      catalogApplyBulkAction.title = isPrivateViewerHost()
	        ? "Apply the selected action"
	        : "Dataset changes are available from localhost or the same Wi-Fi only";
	    }

	    function toggleCatalogSelection(name, checked) {
	      if (checked) catalogSelectedNames.add(name);
	      else catalogSelectedNames.delete(name);
	      updateBulkActionUi();
	    }

	    function selectCurrentCatalogPage(checked) {
	      currentCatalogPageNames().forEach(name => {
	        if (checked) catalogSelectedNames.add(name);
	        else catalogSelectedNames.delete(name);
	      });
	      updateBulkActionUi();
	    }

	    function clearCatalogSelection() {
	      catalogSelectedNames.clear();
	      catalogBulkAction.value = "";
	      updateBulkActionUi();
	    }

	    function registerCatalogRow(row) {
	      const summary = row.summary || {};
	      row.deleteEnabled = row.availability === "available" && canDeleteDataset(summary);
	      row.deleteTitle = row.deleteEnabled
	        ? `Delete ${row.name}`
	        : "Deletion is only available for ready _workspace datasets over localhost or the same Wi-Fi";
	      if (row.availability !== "available" || !summary.episodes) return;
	      datasets[row.name] = summary;
	      datasetPlatformByName[row.name] = row.platform || summary.platform || "so101";
	      datasetPlatformLabelByName[row.name] = row.platformLabel || summary.platform_label || platformLabel(datasetPlatformByName[row.name]);
	      datasetCategoryByName[row.name] = row.category || "";
	      datasetSplitByName[row.name] = row.splitKey || splitForDataset(row.name);
	      datasetRenderTypeByName[row.name] = row.renderKey || renderTypeForDataset(row.name);
	      if (!datasetOrder.includes(row.name)) datasetOrder.push(row.name);
	      if (![...split.options].some(option => option.value === row.name)) {
	        split.add(new Option(row.name, row.name));
	      }
	    }

	    function updateCatalogPager() {
	      const first = catalogTotalRows ? (catalogCurrentPage - 1) * 10 + 1 : 0;
	      const last = Math.min(catalogCurrentPage * 10, catalogTotalRows);
	      catalogPreviousPage.disabled = catalogCurrentPage <= 1;
	      catalogNextPage.disabled = catalogCurrentPage >= catalogLastPage || catalogTotalRows === 0;
	      catalogPageStatus.textContent = `Page ${catalogCurrentPage} of ${catalogLastPage} · ${first}-${last} of ${catalogTotalRows}`;
	    }

	    function applyDatasetCatalogPayload(payload, requestedPage, {fromCache = false} = {}) {
	      const hadSelection = Boolean(split.value && datasets[split.value]);
	      (payload.data || []).forEach(registerCatalogRow);
	      catalogCurrentPage = Number(payload.page || requestedPage);
	      catalogUrlQuery.set("page", String(catalogCurrentPage));
	      catalogLastPage = Number(payload.last_page || 1);
	      catalogTotalRows = Number(payload.total || payload.last_row || 0);
	      catalogSelectionCounts = {
	        training: Number(payload.selection_counts?.training ?? payload.trainable_set_count ?? 0),
	        validation: Number(payload.selection_counts?.validation || 0),
	        loop_test: Number(payload.selection_counts?.loop_test || 0),
	      };
	      catalogMeta.textContent = (
	        `${catalogTotalRows.toLocaleString()} matching datasets · server-loaded 10 per page · newest first`
	        + (catalogMarkedOnly ? " · marked only" : "")
	        + (fromCache ? " · restored from page cache" : "")
	      );
	      updateCatalogPager();
	      if (!hadSelection && !initialViewState.dataset) {
	        const firstReady = (payload.data || []).find(row => row.availability === "available" && datasets[row.name]);
	        if (firstReady) selectDataset(firstReady.name);
	      }
	      updateBulkActionUi();
	      syncCurrentViewUrl();
	      return payload;
	    }

	    async function requestDatasetCatalogPage(url, _config, params) {
	      const requestedPage = Number(params?.page || 1);
	      const query = catalogQueryFromParams(params);
	      catalogUrlQuery = new URLSearchParams(query);
	      const cacheKey = catalogCacheKey(url, query);
	      const cachedPayload = getCachedCatalogPage(cacheKey);
	      if (cachedPayload) {
	        hideCatalogLoadingImmediately();
	        return applyDatasetCatalogPayload(cachedPayload, requestedPage, {fromCache: true});
	      }

	      startCatalogLoading(`Loading catalog page ${requestedPage}`);
	      try {
	        const response = await fetch(`${url}?${query.toString()}`, { cache: "no-store" });
	        if (!response.ok) throw new Error(`HTTP ${response.status}`);
	        const payload = await response.json();
	        cacheCatalogPage(cacheKey, payload);
	        applyDatasetCatalogPayload(payload, requestedPage);
	        finishCatalogLoading();
	        return payload;
	      } catch (error) {
	        failCatalogLoading(error);
	        throw error;
	      }
	    }

	    function catalogSelectHeaderFilter(values) {
	      return (cell, _onRendered, success) => {
	        const select = document.createElement("select");
	        select.setAttribute("aria-label", `${cell.getColumn().getDefinition().title} filter`);
	        select.add(new Option("All", ""));
	        values.forEach(value => select.add(new Option(value, value)));
	        select.value = String(cell.getValue() || "");
	        select.addEventListener("change", () => success(select.value));
	        return select;
	      };
	    }

	    function renderDatasetCatalog() {
	      if (datasetCatalogTable) {
	        syncDatasetCatalogSelection();
	        renderSelectedDatasetMeta();
	        return;
	      }
	      catalogMeta.textContent = "Loading the first 10 datasets...";
	      datasetCatalogTable = new Tabulator(datasetCatalogGrid, {
	          index: "name",
	          layout: "fitColumns",
	          ajaxURL: "/api/datasets/catalog",
	          ajaxRequestFunc: requestDatasetCatalogPage,
	          pagination: true,
	          paginationMode: "remote",
	          paginationSize: 10,
	          paginationInitialPage: initialViewState.catalogPage,
	          filterMode: "remote",
	          sortMode: "remote",
	          initialHeaderFilter: initialCatalogHeaderFilters(),
	          initialSort: [{column: initialViewState.catalogSort, dir: initialViewState.catalogSortDirection}],
	          placeholder: "No datasets match the current filters.",
	          headerFilterLiveFilterDelay: 300,
	          rowFormatter: row => {
	            const selected = row.getData().name === split.value;
	            row.getElement().classList.toggle("dataset-selected", selected);
	            row.getElement().setAttribute("aria-selected", String(selected));
	          },
	          columns: [
	            {
	              title: "",
	              field: "name",
	              width: 42,
	              minWidth: 42,
	              headerSort: false,
	              hozAlign: "center",
	              headerHozAlign: "center",
	              titleFormatter: () => {
	                const checkbox = document.createElement("input");
	                checkbox.type = "checkbox";
	                checkbox.className = "dataset-select-checkbox dataset-select-page-checkbox";
	                checkbox.setAttribute("aria-label", "Select all datasets on this page");
	                checkbox.addEventListener("click", event => event.stopPropagation());
	                checkbox.addEventListener("change", () => selectCurrentCatalogPage(checkbox.checked));
	                return checkbox;
	              },
	              formatter: cell => {
	                const row = cell.getRow().getData();
	                const checkbox = document.createElement("input");
	                checkbox.type = "checkbox";
	                checkbox.className = "dataset-select-checkbox dataset-row-select-checkbox";
	                checkbox.dataset.datasetName = row.name;
	                checkbox.checked = catalogSelectedNames.has(row.name);
	                const eligibleRoles = Object.entries(row.roleEligibility || {})
	                  .filter(([, contract]) => contract?.eligible)
	                  .map(([role]) => role.replace("_", " "));
	                checkbox.title = eligibleRoles.length
	                  ? `${row.name} can be used for: ${eligibleRoles.join(", ")}`
	                  : `${row.name}: not eligible for a dataset role`;
	                checkbox.setAttribute("aria-label", `Select ${row.name}`);
	                checkbox.addEventListener("click", event => event.stopPropagation());
	                checkbox.addEventListener("change", () => toggleCatalogSelection(row.name, checkbox.checked));
	                return checkbox;
	              },
	            },
	            { title: "Platform", field: "platformLabel", minWidth: 90, headerFilter: catalogSelectHeaderFilter(["SO101", "MyCobot"]), headerFilterFunc: "=" },
	            {
	              title: "Dataset",
	              field: "familyLabel",
	              minWidth: 250,
	              widthGrow: 3,
	              headerFilter: "input",
	              cssClass: "dataset-name-cell",
	              formatter: cell => {
	                const row = cell.getRow().getData();
	                const markedRoles = new Set(row.markedRoles || []);
	                const badges = [
	                  markedRoles.has("training")
	                    ? '<span class="trainable-set-badge" title="Included in the marked training set">Training</span>'
	                    : "",
	                  markedRoles.has("validation")
	                    ? '<span class="trainable-set-badge validation-role" title="Included in the marked validation set">Validation</span>'
	                    : "",
	                  markedRoles.has("loop_test")
	                    ? '<span class="trainable-set-badge loop-test-role" title="Included in the marked loop test set">Loop test</span>'
	                    : "",
	                ].join("");
	                return `<span class="dataset-family-line"><span class="dataset-family">${escapeHtml(cell.getValue())}</span>${badges}</span><span class="dataset-id" title="${escapeAttr(row.name)}">${escapeHtml(row.name)}</span>`;
	              },
	            },
	            {
	              title: "Version",
	              field: "versionLabel",
	              minWidth: 132,
	              headerFilter: "input",
	              cssClass: "dataset-version-cell",
	            },
	            {
	              title: "Status",
	              field: "splitLabel",
	              minWidth: 112,
	              headerFilter: catalogSelectHeaderFilter(["Train", "Validation", "Closed loop"]),
	              headerFilterFunc: "=",
	              formatter: cell => `<span class="dataset-badge ${escapeAttr(cell.getRow().getData().splitKey)}">${escapeHtml(cell.getValue())}</span>`,
	            },
	            {
	              title: "Type",
	              field: "renderLabel",
	              minWidth: 125,
	              headerFilter: catalogSelectHeaderFilter(["Standard sim", "Photoreal", "Real camera"]),
	              headerFilterFunc: "=",
	              formatter: cell => `<span class="dataset-badge ${escapeAttr(cell.getRow().getData().renderKey)}">${escapeHtml(cell.getValue())}</span>`,
	            },
	            {
	              title: "Created",
	              field: "createdEpoch",
	              minWidth: 155,
	              sorter: "number",
	              cssClass: "dataset-created-cell",
	              formatter: cell => {
	                const createdAt = cell.getRow().getData().createdAt;
	                return `<time datetime="${escapeAttr(createdAt)}" title="${escapeAttr(createdAt || "Creation date unavailable")}">${escapeHtml(formatDatasetCreatedAt(createdAt))}</time>`;
	              },
	            },
	            { title: "Episodes", field: "episodes", headerSort: false, width: 96, hozAlign: "right", headerHozAlign: "right" },
	            {
	              title: "Action",
	              field: "name",
	              width: 92,
	              headerSort: false,
	              hozAlign: "center",
	              formatter: cell => {
	                const row = cell.getRow().getData();
	                return `<button class="dataset-delete-button" type="button" title="${escapeAttr(row.deleteTitle)}" aria-label="${escapeAttr(row.deleteTitle)}" ${row.deleteEnabled ? "" : "disabled"}>Delete</button>`;
	              },
	              cellClick: (event, cell) => {
	                event.stopPropagation();
	                const row = cell.getRow().getData();
	                if (row.deleteEnabled) deleteDataset(row.name);
	              },
	            },
	          ],
	      });
	      datasetCatalogTable.on("rowClick", (event, row) => {
	        if (event.target.closest(".dataset-select-checkbox, .dataset-delete-button")) return;
	        if (row.getData().availability === "available") selectDataset(row.getData().name);
	      });
	      datasetCatalogTable.on("dataLoaded", () => {
	        syncDatasetCatalogSelection();
	        updateBulkActionUi();
	        restoreInitialViewState();
	      });
	      datasetCatalogTable.on("pageLoaded", page => {
	        catalogCurrentPage = Number(page || 1);
	        catalogUrlQuery.set("page", String(catalogCurrentPage));
	        updateCatalogPager();
	        updateBulkActionUi();
	        syncCurrentViewUrl();
	      });
	    }

	    function renderSelectedDatasetMeta() {
	      const name = split.value;
	      const data = datasets[name];
	      if (!name || !data) {
	        selectedDatasetMeta.innerHTML = "";
	        return;
	      }
	      selectedDatasetMeta.innerHTML = `
	        <span>Selected</span>
	        <span class="selected-dataset-name">${escapeHtml(name)}</span>
	        <span class="chip">${escapeHtml(datasetPlatformLabelByName[name] || platformLabel(datasetPlatformByName[name]))}</span>
	        <span class="chip">Split: ${escapeHtml(splitLabel(datasetSplitByName[name]))}</span>
	        <span class="chip">Render: ${escapeHtml(renderTypeLabel(datasetRenderTypeByName[name]))}</span>
	        <span class="chip">${Number(data.episodes || 0).toLocaleString()} episodes</span>
	        <span class="chip">${Number(data.frames || 0).toLocaleString()} frames</span>
	        <span class="chip">Created: ${escapeHtml(formatDatasetCreatedAt(data.created_at))}</span>
	        <span class="chip">${escapeHtml(data.size_human || "")}</span>
	      `;
	    }

	    function setPlaybackRunning(running) {
	      play.innerHTML = running ? '<span aria-hidden="true">&#10074;&#10074;</span>' : '<span aria-hidden="true">&#9654;</span>';
	      play.classList.toggle("playing", running);
	      play.title = running ? "Pause" : "Play";
	      play.setAttribute("aria-label", running ? "Pause" : "Play");
	    }

	    function selectDataset(name, options = {}) {
	      if (!datasets[name]) return;
	      if (timer) {
	        clearInterval(timer);
	        timer = null;
	        setPlaybackRunning(false);
	      }
	      split.value = name;
	      syncDatasetCatalogSelection();
	      renderSelectedDatasetMeta();
	      syncEpisodeRange();
	      episode.value = String(Math.min(
	        Number(episode.max),
	        Math.max(0, Number(options.episode ?? 0)),
	      ));
	      episodeValue.value = episode.value;
	      syncFrameRange();
	      frame.value = String(Math.min(
	        Number(frame.max),
	        Math.max(0, Number(options.frame ?? 0)),
	      ));
	      frameValue.value = frame.value;
	      if (options.fps) fps.value = String(options.fps);
	      renderPhotorealShortcuts();
	      loadFrame();
	      syncCurrentViewUrl();
	    }

	    function isPrivateViewerHost() {
	      const host = window.location.hostname;
	      if (host === "localhost" || host === "127.0.0.1" || host === "::1") return true;
	      if (/^10[.]/.test(host) || /^192[.]168[.]/.test(host)) return true;
	      const match = host.match(/^172[.]([0-9]+)[.]/);
	      return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
	    }

	    function canDeleteDataset(data) {
	      const root = String(data?.root || "").replaceAll("\\\\", "/");
	      return isPrivateViewerHost() && root.includes("/_workspace/");
	    }

	    function setDatasetActionStatus(message, tone = "") {
	      datasetActionStatus.textContent = message || "";
	      datasetActionStatus.className = `dataset-action-status ${tone}`.trim();
	      datasetActionStatus.hidden = !message;
	    }

	    function removeCatalogEntries(names) {
	      for (const name of names) {
	        catalogSelectedNames.delete(name);
	        delete datasets[name];
	        datasetOrder = datasetOrder.filter(candidate => candidate !== name);
	        [...split.options].filter(option => option.value === name).forEach(option => option.remove());
	      }
	    }

	    async function reloadCatalogAfterMutation() {
	      clearCatalogPageCache();
	      await init({preserveStatus: true, preserveLoadedDatasets: true, reload: true});
	      updateBulkActionUi();
	    }

	    async function applyCatalogBulkAction() {
	      const action = catalogBulkAction.value;
	      const names = [...catalogSelectedNames].sort((left, right) => left.localeCompare(right));
	      if (!action || !names.length) return;
	      if (!isPrivateViewerHost()) {
	        setDatasetActionStatus("Dataset changes are available from localhost or the same Wi-Fi only.", "error");
	        return;
	      }

	      catalogApplyBulkAction.disabled = true;
	      try {
	        const roleActions = {
	          mark_training: {apiAction: "mark", role: "training", label: "training"},
	          remove_training: {apiAction: "remove", role: "training", label: "training"},
	          mark_validation: {apiAction: "mark", role: "validation", label: "validation"},
	          remove_validation: {apiAction: "remove", role: "validation", label: "validation"},
	          mark_loop_test: {apiAction: "mark", role: "loop_test", label: "loop test"},
	          remove_loop_test: {apiAction: "remove", role: "loop_test", label: "loop test"},
	        };
	        const roleAction = roleActions[action];
	        if (roleAction) {
	          const {apiAction, role, label} = roleAction;
	          setDatasetActionStatus(
	            `${apiAction === "mark" ? "Adding" : "Removing"} ${names.length} dataset${names.length === 1 ? "" : "s"} ${apiAction === "mark" ? "to" : "from"} the ${label} set...`,
	          );
	          const response = await fetch("/api/datasets/role-selection", {
	            method: "POST",
	            headers: {"Content-Type": "application/json"},
	            body: JSON.stringify({action: apiAction, role, names}),
	          });
	          const result = await response.json();
	          if (!response.ok) throw new Error(result.message || `${label} set update failed`);
	          catalogSelectionCounts = {
	            training: Number(result.counts?.training || 0),
	            validation: Number(result.counts?.validation || 0),
	            loop_test: Number(result.counts?.loop_test || 0),
	          };
	          clearCatalogSelection();
	          await reloadCatalogAfterMutation();
	          setDatasetActionStatus(
	            `${names.length} dataset${names.length === 1 ? "" : "s"} ${apiAction === "mark" ? "added to" : "removed from"} the ${label} set.`,
	            "success",
	          );
	          return;
	        }

	        if (action === "delete") {
	          const preview = names.slice(0, 8).join("\\n");
	          const remainder = names.length > 8 ? `\\n...and ${names.length - 8} more` : "";
	          if (!window.confirm(
	            `Delete ${names.length} selected dataset${names.length === 1 ? "" : "s"} permanently?\\n\\n${preview}${remainder}\\n\\nThis cannot be undone.`,
	          )) return;
	          const required = `DELETE ${names.length} DATASETS`;
	          const confirmation = window.prompt(
	            `One more confirmation is required.\\n\\nType exactly:\\n${required}`,
	            "",
	          );
	          if (confirmation === null) return;
	          if (confirmation.trim() !== required) {
	            setDatasetActionStatus("Bulk deletion cancelled: the confirmation text did not match.", "error");
	            return;
	          }
	          setDatasetActionStatus(`Deleting ${names.length} selected datasets...`);
	          const response = await fetch("/api/datasets/bulk-delete", {
	            method: "POST",
	            headers: {
	              "Content-Type": "application/json",
	              "X-Dataset-Delete-Confirmation": required,
	            },
	            body: JSON.stringify({names, confirmation: required}),
	          });
	          const result = await response.json();
	          if (!response.ok) throw new Error(result.message || "bulk dataset deletion failed");
	          removeCatalogEntries(result.affected_names || names);
	          clearCatalogSelection();
	          await reloadCatalogAfterMutation();
	          setDatasetActionStatus(
	            `Deleted ${result.deleted_roots?.length || names.length} dataset director${(result.deleted_roots?.length || names.length) === 1 ? "y" : "ies"} (${result.size_human || "size unavailable"}).`,
	            "success",
	          );
	        }
	      } catch (error) {
	        setDatasetActionStatus(`Bulk action failed: ${error.message || error}`, "error");
	      } finally {
	        updateBulkActionUi();
	      }
	    }

	    async function deleteDataset(name) {
	      const data = datasets[name];
	      if (!data || !canDeleteDataset(data)) {
	        setDatasetActionStatus("This dataset cannot be deleted from the current address.", "error");
	        return;
	      }
	      const aliases = datasetOrder.filter(candidate => datasets[candidate]?.root === data.root);
	      const aliasNote = aliases.length > 1
	        ? `\\n\\nThis physical directory is also used by:\\n${aliases.filter(candidate => candidate !== name).join("\\n")}`
	        : "";
	      const approved = window.confirm(
	        `Delete this dataset permanently?\\n\\n${name}\\n${data.root}${aliasNote}\\n\\nThis cannot be undone.`,
	      );
	      if (!approved) return;
	      const confirmation = window.prompt(
	        `One more confirmation is required.\\n\\nType the exact dataset name to delete:\\n${name}`,
	        "",
	      );
	      if (confirmation === null) return;
	      if (confirmation.trim() !== name) {
	        setDatasetActionStatus("Deletion cancelled: the dataset name did not match.", "error");
	        return;
	      }
	      if (timer) {
	        clearInterval(timer);
	        timer = null;
	        setPlaybackRunning(false);
	      }
	      setDatasetActionStatus(`Deleting ${name}...`);
	      datasetCatalogGrid.querySelectorAll(".dataset-delete-button").forEach(button => { button.disabled = true; });
	      try {
	        const response = await fetch("/api/datasets/delete", {
	          method: "POST",
	          headers: {
	            "Content-Type": "application/json",
	            "X-Dataset-Delete-Confirmation": name,
	          },
	          body: JSON.stringify({name, confirm_name: confirmation.trim()}),
	        });
	        const result = await response.json();
	        if (!response.ok) throw new Error(result.message || "dataset deletion failed");
	        removeCatalogEntries(result.affected_names || [name]);
	        await reloadCatalogAfterMutation();
	        const affected = (result.affected_names || []).join(", ");
	        setDatasetActionStatus(
	          `Deleted ${result.name} (${result.size_human || "size unavailable"}). Removed catalog entries: ${affected || result.name}.`,
	          "success",
	        );
	      } catch (error) {
	        renderDatasetCatalog();
	        setDatasetActionStatus(`Delete failed: ${error.message || error}`, "error");
	      }
	    }

	    function splitForDataset(name) {
	      const category = datasetCategoryByName[name] || "";
	      if (category === "closed_loop" || name.endsWith("_loop_val") || name.includes("_loop_validation")) return "closed_loop";
	      if (name.endsWith("_val") || name.endsWith("_valid") || name.includes("_validation")) return "valid";
	      return "train";
	    }

	    function renderTypeForDataset(name) {
	      if (isPhotorealDataset(name)) return "photoreal";
	      const format = String(datasets[name]?.dataset_format || "").toLowerCase();
	      if (format.includes("real_camera") || format.includes("hardware")) return "real";
	      return "simulation";
	    }

	    function isPhotorealDataset(name) {
	      const category = datasetCategoryByName[name] || "";
	      return category === "photoreal" || name.startsWith("photoreal_");
	    }

	    function platformLabel(platform) {
	      return platform === "mycobot" ? "MyCobot" : "SO101";
	    }

	    function splitLabel(kind) {
	      if (kind === "valid") return "Validation";
	      if (kind === "closed_loop") return "Closed loop";
	      return "Train";
	    }

	    function renderTypeLabel(kind) {
	      if (kind === "photoreal") return "Photoreal";
	      if (kind === "real") return "Real camera";
	      return "Standard sim";
	    }

	    async function loadTrainingRuns() {
	      trainingActiveChip.innerHTML = `active: <strong>loading</strong>`;
	      const payload = await fetch("/api/training/runs").then(r => r.json());
	      trainingRunRows = payload.runs || [];
	      if (selectedTrainingId && !trainingRunRows.some(row => row.training_id === selectedTrainingId)) {
	        selectedTrainingId = null;
	      }
	      if (!selectedTrainingId && payload.active_training_id) selectedTrainingId = payload.active_training_id;
	      if (!selectedTrainingId && trainingRunRows.length) selectedTrainingId = trainingRunRows[0].training_id;
	      trainingActiveChip.innerHTML = `active: <strong>${payload.active_training_id || "none"}</strong>`;
	      trainingRuns.innerHTML = trainingRunRows.map(row => `
	        <a class="run-item ${row.training_id === selectedTrainingId ? "active" : ""}" href="${escapeAttr(trainingRunUrl(row.training_id))}" data-training-id="${escapeAttr(row.training_id)}">
	          <div class="run-id">${escapeHtml(row.training_id)} ${row.active ? '<span class="chip">active</span>' : ''}</div>
	          <div class="meta">${escapeHtml(row.dataset_config_name || "")}</div>
	          <div class="meta">train ${fmtMaybe(row.latest_train_loss)} · val ${fmtMaybe(row.latest_val_loss)} · ckpt ${row.checkpoint_count || 0}</div>
	        </a>
	      `).join("") || `<p class="empty">No training runs found.</p>`;
	      if (selectedTrainingId) await loadTrainingDetail(selectedTrainingId);
	    }

	    async function loadTrainingDetail(trainingId) {
	      selectedTrainingId = trainingId;
	      syncCurrentViewUrl();
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
      episodeValue.value = episode.value;
      syncFrameRange();
    }

    function syncFrameRange() {
      const data = datasets[split.value];
      if (!data) return;
      const length = data.episode_lengths[Number(episode.value)];
      frame.max = String(length - 1);
      frame.value = String(Math.min(Number(frame.value), length - 1));
      frameValue.value = frame.value;
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
	      const generation = ++frameLoadGeneration;
	      syncFrameRange();
	      const requestedSplit = split.value;
	      const requestedEpisode = episode.value;
	      const requestedFrame = frame.value;
	      const url = `/api/frame?split=${encodeURIComponent(requestedSplit)}&episode=${requestedEpisode}&frame=${requestedFrame}`;
	      const row = await fetch(url).then(r => r.json());
	      if (generation !== frameLoadGeneration) return;
	      episode.value = String(row.episode);
	      episodeValue.value = episode.value;
	      frame.value = String(row.frame);
	      frameValue.value = frame.value;
	      meta.textContent = `${row.split} | episode ${row.episode}/${datasets[row.split].episodes - 1} | frame ${row.frame}/${row.episode_length - 1} | row ${row.row_index} | task_index ${row.task_index ?? "n/a"} | t=${row.timestamp.toFixed(3)}s`;
      promptText.textContent = row.prompt || row.task || "(no prompt stored)";
      cameras.innerHTML = cameraFigures(row.images);
      renderPhotorealPanel(row);
	      jointRows.innerHTML = Object.keys(row.state).map(joint => `
	        <tr><td>${joint}</td><td>${fmt(row.state[joint])}</td><td>${fmt(row.action[joint])}</td></tr>
	      `).join("");
	      syncCurrentViewUrl();
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
	    trainingReload.addEventListener("click", loadTrainingRuns);
	    trainingRuns.addEventListener("click", event => {
	      const button = event.target.closest?.(".run-item");
	      if (!button?.dataset?.trainingId) return;
	      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
	      event.preventDefault();
	      loadTrainingDetail(button.dataset.trainingId);
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
	      episodeValue.value = episode.value;
	      syncFrameRange();
	      frame.value = button.dataset.frame;
	      frameValue.value = frame.value;
	      loadFrame();
	    });
	    episode.addEventListener("input", () => {
	      episodeValue.value = episode.value;
	      frame.value = "0";
	      frameValue.value = "0";
	      loadFrame();
	    });
	    frame.addEventListener("input", () => {
	      frameValue.value = frame.value;
	      loadFrame();
	    });
	    fps.addEventListener("change", syncCurrentViewUrl);
    play.addEventListener("click", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        setPlaybackRunning(false);
        return;
      }
      setPlaybackRunning(true);
      timer = setInterval(() => {
        if (Number(frame.value) >= Number(frame.max)) frame.value = "0";
        else frame.value = String(Number(frame.value) + 1);
        frameValue.value = frame.value;
        loadFrame();
      }, Math.max(16, 1000 / Number(fps.value || 12)));
	    });
	    document.getElementById("prev").addEventListener("click", () => {
	      frame.value = String(Math.max(0, Number(frame.value) - 1));
	      frameValue.value = frame.value;
	      loadFrame();
	    });
	    document.getElementById("next").addEventListener("click", () => {
	      frame.value = String(Math.min(Number(frame.max), Number(frame.value) + 1));
	      frameValue.value = frame.value;
	      loadFrame();
	    });
	    catalogPreviousPage.addEventListener("click", () => {
	      if (datasetCatalogTable && catalogCurrentPage > 1) datasetCatalogTable.setPage(catalogCurrentPage - 1);
	    });
	    catalogNextPage.addEventListener("click", () => {
	      if (datasetCatalogTable && catalogCurrentPage < catalogLastPage) datasetCatalogTable.setPage(catalogCurrentPage + 1);
	    });
	    catalogBulkAction.addEventListener("change", updateBulkActionUi);
	    catalogApplyBulkAction.addEventListener("click", applyCatalogBulkAction);
	    catalogMarkedOnlyButton.addEventListener("click", async () => {
	      catalogMarkedOnly = !catalogMarkedOnly;
	      updateBulkActionUi();
	      if (!datasetCatalogTable) return;
	      try {
	        if (catalogCurrentPage === 1) await datasetCatalogTable.setData();
	        else await datasetCatalogTable.setPage(1);
	      } catch (error) {
	        setDatasetActionStatus(`Dataset filter failed: ${error?.message || error}`, "error");
	      }
	    });
	    loopAnalyzerReload.addEventListener("click", () => initLoopAnalyzer(true));
	    window.addEventListener("popstate", () => window.location.reload());
	    init();
	    syncAppTab(initialViewState.view, {updateUrl: false});
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
