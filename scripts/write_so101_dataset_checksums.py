#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ROOTS = {
    "pick_train": Path("_workspace/so101_lerobot/pick_train50_top_wrist_256_seed98200"),
    "pick_val": Path("_workspace/so101_lerobot/pick_val24_top_wrist_256_seed98100"),
    "pick_place_train": Path("_workspace/so101_lerobot/pick_place_train50_top_wrist_256_seed99000"),
    "pick_place_val": Path("_workspace/so101_lerobot/pick_place_val24_top_wrist_256_seed102000"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write compact checksums for SO101 LeRobot datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/so101/training_datasets/checksums.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "note": "Raw dataset files are local artifacts and are intentionally excluded from git.",
        "datasets": {
            name: _dataset_entry(repo_root, name, root)
            for name, root in DATASET_ROOTS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _dataset_entry(repo_root: Path, name: str, root: Path) -> dict[str, Any]:
    resolved = (repo_root / root).resolve()
    report_path = resolved / "so101_lerobot_export_report.json"
    audit_path = resolved / "so101_lerobot_audit.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing export report for {name}: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = report.get("audit") or {}
    return {
        "root": str(root),
        "repo_id": report.get("repo_id"),
        "task": report.get("task"),
        "episodes": int(report.get("exported_episodes", 0)),
        "frames": int(audit.get("dataset_len", 0)),
        "size_bytes": _dir_size(resolved),
        "file_count": sum(1 for path in resolved.rglob("*") if path.is_file()),
        "directory_sha256": _directory_sha256(resolved),
        "export_report_sha256": _file_sha256(report_path),
        "audit_sha256": _file_sha256(audit_path) if audit_path.exists() else None,
        "camera_contract": {
            "observation.images.camera1": (report.get("feature_mapping") or {}).get("observation.images.camera1"),
            "observation.images.camera2": (report.get("feature_mapping") or {}).get("observation.images.camera2"),
            "observation.images.camera3": (report.get("feature_mapping") or {}).get("observation.images.camera3"),
        },
        "sample_shapes": {
            "observation.images.camera1": (audit.get("sample_shapes") or {}).get("observation.images.camera1"),
            "observation.images.camera2": (audit.get("sample_shapes") or {}).get("observation.images.camera2"),
            "observation.images.camera3": (audit.get("sample_shapes") or {}).get("observation.images.camera3"),
        },
    }


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        file_digest = _file_sha256(path)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


if __name__ == "__main__":
    main()
