#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_smolvla_pipeline import (
    SO101DatasetManifest,
    load_dataset_manifest,
    write_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate SO101 SmolVLA dataset manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--dataset-id", required=True)
    create.add_argument("--split", choices=["train", "validation", "test"], required=True)
    create.add_argument("--episodes", type=int, required=True)
    create.add_argument("--frames", type=int, required=True)
    create.add_argument("--expected-frames-per-episode", type=int)
    create.add_argument("--source-episode-count", type=int, required=True)
    create.add_argument("--target-expansion-factor", type=float, default=2.0)
    create.add_argument("--includes-recovery-or-off-nominal-states", action="store_true")
    create.add_argument("--teacher-uses-privileged-state", action=argparse.BooleanOptionalAction, default=True)
    create.add_argument("--sticky-grasp-allowed", action="store_true")
    create.add_argument("--generator", default="")
    create.add_argument("--notes", default="")

    from_report = subparsers.add_parser("from-export-report")
    from_report.add_argument("--report", type=Path, required=True)
    from_report.add_argument("--output", type=Path, required=True)
    from_report.add_argument("--dataset-id", required=True)
    from_report.add_argument("--split", choices=["train", "validation", "test"], required=True)
    from_report.add_argument("--source-episode-count", type=int, required=True)
    from_report.add_argument("--target-expansion-factor", type=float, default=2.0)
    from_report.add_argument("--includes-recovery-or-off-nominal-states", action="store_true")
    from_report.add_argument("--teacher-uses-privileged-state", action=argparse.BooleanOptionalAction, default=True)
    from_report.add_argument("--generator", default="")
    from_report.add_argument("--notes", default="")

    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)

    args = parser.parse_args()
    if args.command == "create":
        manifest = SO101DatasetManifest(
            dataset_id=args.dataset_id,
            split=args.split,
            episodes=args.episodes,
            frames=args.frames,
            source_episode_count=args.source_episode_count,
            target_expansion_factor=args.target_expansion_factor,
            expected_frames_per_episode=args.expected_frames_per_episode,
            includes_recovery_or_off_nominal_states=args.includes_recovery_or_off_nominal_states,
            teacher_uses_privileged_state=args.teacher_uses_privileged_state,
            sticky_grasp_allowed=args.sticky_grasp_allowed,
            generator=args.generator,
            notes=args.notes,
        )
        return _write_and_exit(args.output, manifest)
    if args.command == "from-export-report":
        report = json.loads(args.report.read_text(encoding="utf-8"))
        manifest = SO101DatasetManifest(
            dataset_id=args.dataset_id,
            split=args.split,
            episodes=int(report.get("exported_episodes") or report.get("requested_episodes") or 0),
            frames=_frames_from_report(report),
            source_episode_count=args.source_episode_count,
            target_expansion_factor=args.target_expansion_factor,
            includes_recovery_or_off_nominal_states=args.includes_recovery_or_off_nominal_states,
            teacher_uses_privileged_state=args.teacher_uses_privileged_state,
            sticky_grasp_allowed=bool(report.get("config", {}).get("allow_sticky_grasp", False)),
            generator=args.generator or str(args.report),
            notes=args.notes,
        )
        return _write_and_exit(args.output, manifest)
    if args.command == "validate":
        manifest = load_dataset_manifest(args.manifest)
        errors = manifest.validate()
        print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1)


def _write_and_exit(path: Path, manifest: SO101DatasetManifest) -> None:
    write_dataset_manifest(path, manifest)
    print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
    if manifest.validate():
        raise SystemExit(1)


def _frames_from_report(report: dict[str, Any]) -> int:
    episodes = report.get("episodes") or []
    if isinstance(episodes, list):
        return int(sum(int(row.get("frames") or 0) for row in episodes if isinstance(row, dict)))
    return int(report.get("frames") or report.get("exported_frames") or 0)


if __name__ == "__main__":
    main()
