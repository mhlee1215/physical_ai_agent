#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_prestage_evidence(
    *,
    pre_stage_pack: Path,
    runbook_manifest: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    pack = _load_json(pre_stage_pack)
    movement = _load_json(Path(pack["movement_report_manifest"]))
    gate_report = _load_json(Path(pack["gate_report_manifest"])) if pack.get("gate_report_manifest") else {}
    runbook = _load_json(runbook_manifest) if runbook_manifest is not None else {}

    checks: list[dict[str, Any]] = []
    checks.append(_check_path("pre_stage_pack", pre_stage_pack))
    checks.append(_check_path("movement_report_html", Path(pack["movement_report_html"])))
    checks.append(_check_path("movement_report_manifest", Path(pack["movement_report_manifest"])))
    checks.append(_check_path("gate_report_html", Path(pack["gate_report_html"]) if pack.get("gate_report_html") else None))
    checks.append(_check_path("gate_report_manifest", Path(pack["gate_report_manifest"]) if pack.get("gate_report_manifest") else None))
    checks.append(_check_path("next_action_gate", Path(pack["next_action_gate"])))
    checks.append(_check_path("grasp_outcome", Path(pack["grasp_outcome"])))
    if runbook_manifest is not None:
        checks.append(_check_path("runbook_manifest", runbook_manifest))
        checks.append(_check_path("runbook_markdown", Path(runbook.get("output_markdown", ""))))

    motion_videos = _motion_videos_from_movement(movement)
    video_paths = [Path(item["path"]) for item in motion_videos]
    for index, item in enumerate(motion_videos):
        path = Path(item["path"])
        checks.append(_check_path(f"motion_video_{index}", path))
        checks.append(
            _check_bool(
                f"motion_video_{index}_has_probe_metadata",
                bool(item.get("actual_codec"))
                and item.get("actual_frame_count") is not None
                and "first_frame_readable" in item
                and isinstance(item.get("browser_preview_recommended"), bool),
                {
                    "report": item.get("report"),
                    "path": item.get("path"),
                    "actual_codec": item.get("actual_codec"),
                    "actual_frame_count": item.get("actual_frame_count"),
                    "first_frame_readable": item.get("first_frame_readable"),
                    "browser_preview_recommended": item.get("browser_preview_recommended"),
                },
            )
        )
        checks.append(
            _check_bool(
                f"motion_video_{index}_first_frame_readable",
                item.get("first_frame_readable") is True,
                {
                    "report": item.get("report"),
                    "path": item.get("path"),
                    "first_frame_readable": item.get("first_frame_readable"),
                },
            )
        )
        checks.append(
            _check_bool(
                f"motion_video_{index}_frame_count_positive",
                _positive_int(item.get("actual_frame_count")),
                {
                    "report": item.get("report"),
                    "path": item.get("path"),
                    "actual_frame_count": item.get("actual_frame_count"),
                },
            )
        )
    preview_paths = [Path(path) for path in movement.get("video_previews", [])]
    for index, path in enumerate(preview_paths):
        checks.append(_check_path(f"motion_video_preview_{index}", path))
    checks.append(
        _check_bool(
            "movement_video_count_matches_files",
            int(movement.get("video_count", 0)) == len(video_paths),
            {
                "manifest_video_count": movement.get("video_count", 0),
                "motion_video_files": [str(path) for path in video_paths],
            },
        )
    )
    checks.append(
        _check_bool(
            "movement_preview_count_matches_files",
            int(movement.get("video_preview_count", 0)) == len(preview_paths),
            {
                "manifest_video_preview_count": movement.get("video_preview_count", 0),
                "motion_video_preview_files": [str(path) for path in preview_paths],
            },
        )
    )
    checks.append(
        _check_bool(
            "movement_has_preview_for_each_video",
            int(movement.get("video_count", 0)) == int(movement.get("video_preview_count", 0)),
            {
                "video_count": movement.get("video_count", 0),
                "video_preview_count": movement.get("video_preview_count", 0),
            },
        )
    )
    checks.append(
        _check_bool(
            "pack_video_count_matches_movement",
            pack.get("video_count") == movement.get("video_count"),
            {"pack_video_count": pack.get("video_count"), "movement_video_count": movement.get("video_count")},
        )
    )
    checks.append(
        _check_bool(
            "purpose_mentions_prestage_not_benchmark",
            "pre-stage" in str(pack.get("purpose", "")) and "not benchmark success" in str(pack.get("purpose", "")),
            {"purpose": pack.get("purpose")},
        )
    )
    if gate_report:
        checks.append(
            _check_bool(
                "gate_status_matches_pack",
                gate_report.get("current_gate_status") == pack.get("current_gate_status"),
                {
                    "pack_gate_status": pack.get("current_gate_status"),
                    "gate_report_status": gate_report.get("current_gate_status"),
                },
            )
        )
        for camera, overlay in sorted(gate_report.get("overlays", {}).items()):
            checks.append(_check_path(f"gate_overlay_camera_{camera}", Path(overlay)))
    if runbook_manifest is not None:
        blocked = pack.get("current_gate_status") == "blocked"
        checks.append(
            _check_bool(
                "blocked_runbook_has_no_physical_command",
                not blocked or runbook.get("contains_physical_command") is False,
                {
                    "pack_gate_status": pack.get("current_gate_status"),
                    "contains_physical_command": runbook.get("contains_physical_command"),
                },
            )
        )
        checks.append(
            _check_bool(
                "runbook_links_same_gate_report",
                runbook.get("gate_report_html") == pack.get("gate_report_html"),
                {"runbook_gate_report_html": runbook.get("gate_report_html"), "pack_gate_report_html": pack.get("gate_report_html")},
            )
        )

    failed = [item for item in checks if item["status"] != "passed"]
    result = {
        "status": "passed" if not failed else "failed",
        "operation": "real_so100_prestage_evidence_audit",
        "pre_stage_pack": str(pre_stage_pack),
        "runbook_manifest": str(runbook_manifest) if runbook_manifest else None,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "purpose": "verify video-backed SO-100 agentic-layer pre-stage evidence before continuing physical actions",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result["manifest_path"] = str(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _motion_videos_from_movement(movement: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for report_path in movement.get("reports", []):
        report = _load_json(Path(report_path))
        video = report.get("motion_video")
        if video and video.get("path"):
            videos.append({"report": report_path, **video})
    return videos


def _positive_int(value: Any) -> bool:
    try:
        return int(value) >= 1
    except (TypeError, ValueError):
        return False


def _check_path(name: str, path: Path | None) -> dict[str, Any]:
    exists = bool(path and path.exists())
    return {
        "name": name,
        "status": "passed" if exists else "failed",
        "path": str(path) if path else None,
    }


def _check_bool(name: str, condition: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if condition else "failed",
        "details": details,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SO-100 agentic pre-stage evidence artifacts.")
    parser.add_argument("--pre-stage-pack", type=Path, required=True)
    parser.add_argument("--runbook-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_prestage_evidence(
                pre_stage_pack=args.pre_stage_pack,
                runbook_manifest=args.runbook_manifest,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
