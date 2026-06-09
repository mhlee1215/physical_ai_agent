#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import (
    extract_policy_postprocessor_action_stats,
    inspect_smolvla_action_metadata,
    load_action_stats,
    load_smolvla_config,
)


DEFAULT_LOCAL_CONFIG = Path(
    "/Users/minhaeng/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json"
)


def inspect_metadata(
    *,
    config_path: Path,
    output_dir: Path,
    model_id: str,
    stats_path: Path | None = None,
    action_semantics: str | None = None,
    gripper_semantics: str | None = None,
    command_units: str | None = None,
    confirm_so100_joint_order: bool = False,
    policy_postprocessor_model: str | None = None,
    action_stats_key: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    config = load_smolvla_config(config_path)
    extracted_stats_path = None
    if stats_path is None and policy_postprocessor_model:
        extracted_stats_path = output_dir / "policy_postprocessor_action_stats.json"
        stats = extract_policy_postprocessor_action_stats(
            model_id_or_path=policy_postprocessor_model,
            output=extracted_stats_path,
            action_stats_key=action_stats_key,
            local_files_only=local_files_only,
        )
    else:
        stats = load_action_stats(stats_path)
    joint_order = SO100_JOINT_ORDER if confirm_so100_joint_order else None
    metadata = inspect_smolvla_action_metadata(
        config=config,
        model_id=model_id,
        stats=stats,
        action_semantics=action_semantics,
        joint_order=joint_order,
        gripper_semantics=gripper_semantics,
        command_units=command_units,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "passed" if not metadata.blockers else "blocked",
        "model_id": model_id,
        "config_path": str(config_path),
        "stats_path": str(stats_path) if stats_path else None,
        "extracted_stats_path": str(extracted_stats_path) if extracted_stats_path else None,
        "metadata": asdict(metadata),
        "required_next_steps": _required_next_steps(asdict(metadata)),
    }
    json_path = output_dir / "smolvla_action_metadata_report.json"
    md_path = output_dir / "smolvla_action_metadata_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    return report


def _required_next_steps(metadata: dict[str, Any]) -> list[str]:
    steps = []
    if metadata.get("output_is_normalized") and not metadata.get("action_stats_available"):
        steps.append("Find or provide authoritative action mean/std stats for this SmolVLA checkpoint.")
    if not metadata.get("action_semantics"):
        steps.append("Confirm whether action values are absolute joint positions or joint deltas.")
    if metadata.get("joint_order") != SO100_JOINT_ORDER:
        steps.append(f"Confirm follower joint order exactly as {SO100_JOINT_ORDER}.")
    if not metadata.get("gripper_semantics"):
        steps.append("Confirm whether larger raw gripper values open or close the follower gripper.")
    if metadata.get("command_units") not in {"feetech_raw_ticks", "lerobot_so100_position"}:
        steps.append(
            "Confirm the postprocessed action units and conversion to Feetech raw Goal_Position ticks."
        )
    return steps


def _render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# SmolVLA Action Metadata Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Model id: `{report['model_id']}`",
        f"- Config path: `{report['config_path']}`",
        f"- Stats path: `{report['stats_path']}`",
        f"- Extracted stats path: `{report['extracted_stats_path']}`",
        f"- Action dim: `{metadata['action_dim']}`",
        f"- Action normalization: `{metadata['action_normalization']}`",
        f"- Action stats available: `{metadata['action_stats_available']}`",
        f"- Stats source: `{metadata['stats_source']}`",
        f"- Selected action stats key: `{metadata['selected_action_stats_key']}`",
        f"- Available action stats keys: `{metadata['available_action_stats_keys']}`",
        f"- Chunk size: `{metadata['chunk_size']}`",
        f"- n_action_steps: `{metadata['n_action_steps']}`",
        f"- Action semantics: `{metadata['action_semantics']}`",
        f"- Joint order: `{metadata['joint_order']}`",
        f"- Gripper semantics: `{metadata['gripper_semantics']}`",
        f"- Command units: `{metadata['command_units']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = metadata.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] or ["- None"])
    lines.extend(["", "## Required Next Steps", ""])
    lines.extend([f"- {step}" for step in report["required_next_steps"]] or ["- None"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SmolVLA action metadata before SO-100 execution.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LOCAL_CONFIG)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/real_so100/smolvla_action_metadata"))
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--action-semantics", choices=["absolute_joint_position", "joint_delta"])
    parser.add_argument("--gripper-semantics", choices=["higher_raw_opens", "higher_raw_closes"])
    parser.add_argument("--command-units", choices=["feetech_raw_ticks", "lerobot_so100_position"])
    parser.add_argument("--confirm-so100-joint-order", action="store_true")
    parser.add_argument("--policy-postprocessor-model")
    parser.add_argument("--action-stats-key")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_metadata(
                config_path=args.config,
                output_dir=args.output_dir,
                model_id=args.model_id,
                stats_path=args.stats,
                action_semantics=args.action_semantics,
                gripper_semantics=args.gripper_semantics,
                command_units=args.command_units,
                confirm_so100_joint_order=args.confirm_so100_joint_order,
                policy_postprocessor_model=args.policy_postprocessor_model,
                action_stats_key=args.action_stats_key,
                local_files_only=args.local_files_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
