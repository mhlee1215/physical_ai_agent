#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "name",
    "robot",
    "scenario",
    "task_prompt",
    "object_suite",
    "feature_contract",
    "source_dataset",
    "lerobot_conversion",
    "training_smoke",
    "closed_loop_stub",
}

JOINT_NAME_ALIASES = {
    "joint6output_to_joint6": "joint7_to_joint6",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a config-first myCobot 280 SmolVLA readiness dataset contract."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-present", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_config(
        config_path=args.config,
        dataset_root_override=args.dataset_root,
        require_present=args.require_present,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"passed", "blocked"} and not (args.require_present and report["status"] != "passed") else 1)


def validate_config(
    *,
    config_path: Path,
    dataset_root_override: Path | None = None,
    require_present: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL.difference(config))
    if missing:
        errors.append(f"missing top-level fields: {missing}")

    robot = _dict(config.get("robot"))
    feature = _dict(config.get("feature_contract"))
    source = _dict(config.get("source_dataset"))
    conversion = _dict(config.get("lerobot_conversion"))
    smoke = _dict(config.get("training_smoke"))
    closed_loop = _dict(config.get("closed_loop_stub"))

    joint_names = _str_list(robot.get("joint_names"))
    state_names = _str_list(feature.get("state_names"))
    action_names = _str_list(feature.get("action_names"))
    state_dim = int(robot.get("state_dim", -1)) if isinstance(robot.get("state_dim"), int) else -1
    action_dim = int(robot.get("action_dim", -1)) if isinstance(robot.get("action_dim"), int) else -1

    if state_dim != 7 or action_dim != 7:
        errors.append(f"myCobot 280 smoke contract must be 7D state/action, got state={state_dim}, action={action_dim}")
    if len(joint_names) != 7:
        errors.append(f"robot.joint_names must have 7 entries, got {len(joint_names)}")
    if state_names != joint_names:
        errors.append("feature_contract.state_names must match robot.joint_names")
    if action_names != joint_names:
        errors.append("feature_contract.action_names must match robot.joint_names")
    if feature.get("base_checkpoint") != "lerobot/smolvla_base":
        warnings.append("base checkpoint is not lerobot/smolvla_base")
    if not _dict(feature.get("camera_contract")):
        errors.append("feature_contract.camera_contract must declare at least one camera")
    if not str(config.get("task_prompt", "")).strip():
        errors.append("task_prompt is required")
    if source.get("expected_teacher_attachment_enabled") is not False:
        errors.append("source_dataset.expected_teacher_attachment_enabled must be false")
    if source.get("expected_object_teleport_during_pickup_lift") is not False:
        errors.append("source_dataset.expected_object_teleport_during_pickup_lift must be false")
    if "converter_script" not in conversion:
        errors.append("lerobot_conversion.converter_script is required")
    if "output_dir" not in smoke:
        errors.append("training_smoke.output_dir is required")
    if "script" not in closed_loop:
        errors.append("closed_loop_stub.script is required")

    dataset_root = (dataset_root_override or Path(str(source.get("root", "")))).resolve()
    dataset_report = _validate_source_dataset(config=config, dataset_root=dataset_root)
    if dataset_report["status"] == "failed":
        errors.extend(dataset_report["errors"])
    elif dataset_report["status"] == "blocked":
        warnings.append(dataset_report["blocker"])

    status = "failed" if errors else dataset_report["status"]
    if status == "blocked" and require_present:
        status = "failed"

    return {
        "operation": "validate_mycobot280_training_dataset",
        "status": status,
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "errors": errors,
        "warnings": warnings,
        "config_summary": {
            "name": config.get("name"),
            "robot": robot.get("name"),
            "scenario": config.get("scenario"),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "joint_names": joint_names,
            "camera_keys": sorted(_dict(feature.get("camera_contract")).keys()),
            "conversion_output_root": conversion.get("output_root"),
            "training_output_dir": smoke.get("output_dir"),
            "closed_loop_script": closed_loop.get("script"),
        },
        "dataset_report": dataset_report,
        "claim_boundary": "Readiness validation only; this does not train or evaluate a policy.",
    }


def _validate_source_dataset(*, config: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    source = _dict(config.get("source_dataset"))
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "blocked",
            "blocker": f"source dataset manifest is missing: {manifest_path}",
            "next_step": "Run the configured generation_command, then rerun with --require-present.",
            "generation_command": source.get("generation_command"),
            "errors": [],
        }
    manifest = _load_json(manifest_path)
    errors: list[str] = []

    expected_episodes = int(source.get("expected_episodes", 0))
    required_success = _dict(source.get("required_success_criteria"))
    aggregate = _dict(manifest.get("aggregate_metrics"))

    _expect_equal(errors, "format", manifest.get("format"), source.get("format"))
    _expect_equal(errors, "generation_mode", manifest.get("generation_mode"), source.get("expected_generation_mode"))
    _expect_equal(errors, "randomization_enabled", manifest.get("randomization_enabled"), source.get("expected_randomization_enabled"))
    _expect_equal(errors, "teacher_attachment_enabled", manifest.get("teacher_attachment_enabled"), source.get("expected_teacher_attachment_enabled"))
    _expect_equal(errors, "object_teleport_during_pickup_lift", manifest.get("object_teleport_during_pickup_lift"), source.get("expected_object_teleport_during_pickup_lift"))
    if int(manifest.get("episodes", -1)) != expected_episodes:
        errors.append(f"manifest episodes {manifest.get('episodes')} != expected {expected_episodes}")
    if int(manifest.get("passed_episodes", -1)) != expected_episodes:
        errors.append(f"manifest passed_episodes {manifest.get('passed_episodes')} != expected {expected_episodes}")
    if manifest.get("failed_episodes") not in ([], None):
        errors.append(f"manifest failed_episodes must be empty, got {manifest.get('failed_episodes')}")
    if int(manifest.get("frames", 0)) < int(source.get("expected_min_frames", 0)):
        errors.append(f"manifest frames {manifest.get('frames')} below expected_min_frames {source.get('expected_min_frames')}")
    if source.get("expected_all_frames_rendered") is True:
        rendered = sum(int(item.get("rendered_frames", 0)) for item in manifest.get("episode_summaries", []) if isinstance(item, dict))
        if rendered != int(manifest.get("frames", -1)):
            errors.append(f"rendered frame count {rendered} must equal manifest frames {manifest.get('frames')}")
    if _str_list(manifest.get("joint_names")) != _str_list(_dict(config.get("robot")).get("joint_names")):
        errors.append("manifest joint_names do not match config robot.joint_names")
    if _str_list(manifest.get("action_names")) != _str_list(_dict(config.get("robot")).get("joint_names")):
        errors.append("manifest action_names do not match config robot.joint_names")

    if float(aggregate.get("min_final_cube_lift_m", 0.0)) < float(required_success.get("final_cube_lift_m", 0.0)):
        errors.append("aggregate min_final_cube_lift_m is below required final_cube_lift_m")
    if int(aggregate.get("min_lift_best_sustained_two_pad_steps", 0)) < int(required_success.get("lift_best_sustained_two_pad_steps", 0)):
        errors.append("aggregate min_lift_best_sustained_two_pad_steps is below required threshold")
    if int(aggregate.get("min_post_lift_hold_sustained_two_pad_steps", 0)) < int(required_success.get("post_lift_hold_best_sustained_two_pad_steps", 0)):
        errors.append("aggregate min_post_lift_hold_sustained_two_pad_steps is below required threshold")
    if float(aggregate.get("min_post_lift_hold_cube_lift_m", 0.0)) < float(required_success.get("post_lift_hold_min_cube_lift_m", 0.0)):
        errors.append("aggregate min_post_lift_hold_cube_lift_m is below required threshold")
    if float(aggregate.get("max_pad_cube_penetration_m", 999.0)) > float(required_success.get("max_pad_cube_penetration_m", 0.0)):
        errors.append("aggregate max_pad_cube_penetration_m exceeds threshold")

    episode_paths = []
    for summary in manifest.get("episode_summaries", []):
        if isinstance(summary, dict) and summary.get("path"):
            episode_paths.append(dataset_root / str(summary["path"]))
    missing_episodes = [str(path) for path in episode_paths if not path.exists()]
    if missing_episodes:
        errors.append(f"missing episode files: {missing_episodes[:3]}")

    row_report = {"rows_checked": 0, "image_paths_checked": 0, "source_state_dims": [], "action_dims": []}
    if not missing_episodes:
        row_report = _validate_episode_rows(dataset_root=dataset_root, episode_paths=episode_paths, config=config)
        errors.extend(row_report.get("errors", []))
        if int(row_report.get("rows_checked", -1)) != int(manifest.get("frames", -2)):
            errors.append(
                f"episode row count {row_report.get('rows_checked')} != manifest frames {manifest.get('frames')}"
            )

    return {
        "status": "failed" if errors else "passed",
        "manifest_path": str(manifest_path),
        "episodes": manifest.get("episodes"),
        "passed_episodes": manifest.get("passed_episodes"),
        "frames": manifest.get("frames"),
        "aggregate_metrics": aggregate,
        "episode_files_checked": len(episode_paths),
        "row_report": row_report,
        "errors": errors,
    }


def _validate_episode_rows(*, dataset_root: Path, episode_paths: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    joint_names = _str_list(_dict(config.get("robot")).get("joint_names"))
    expected_task = str(config.get("task_prompt", ""))
    errors: list[str] = []
    rows_checked = 0
    image_paths_checked = 0
    source_state_dims: set[int] = set()
    action_dims: set[int] = set()
    tasks_seen: set[str] = set()
    info_joint_name_aliases: set[str] = set()

    for episode_path in episode_paths:
        for line_number, line in enumerate(episode_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows_checked += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{episode_path}:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{episode_path}:{line_number}: row must be a JSON object")
                continue

            task = str(row.get("task", ""))
            tasks_seen.add(task)
            if task != expected_task:
                errors.append(f"{episode_path}:{line_number}: task {task!r} != expected {expected_task!r}")

            observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
            state = observation.get("state")
            if not isinstance(state, list):
                errors.append(f"{episode_path}:{line_number}: observation.state must be a list")
            else:
                source_state_dims.add(len(state))
                if len(state) < len(joint_names):
                    errors.append(f"{episode_path}:{line_number}: observation.state length {len(state)} < robot dim {len(joint_names)}")

            action = row.get("action")
            if not isinstance(action, list):
                errors.append(f"{episode_path}:{line_number}: action must be a list")
            else:
                action_dims.add(len(action))
                if len(action) != len(joint_names):
                    errors.append(f"{episode_path}:{line_number}: action length {len(action)} != robot dim {len(joint_names)}")

            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            info_joint_names = _str_list(info.get("joint_names"))
            if info_joint_names:
                normalized_info_joint_names = [_normalize_joint_name(name) for name in info_joint_names]
                for raw, normalized in zip(info_joint_names, normalized_info_joint_names):
                    if raw != normalized:
                        info_joint_name_aliases.add(f"{raw}->{normalized}")
                if normalized_info_joint_names != joint_names:
                    errors.append(f"{episode_path}:{line_number}: info.joint_names do not match robot.joint_names after alias normalization")

            images = observation.get("images") if isinstance(observation.get("images"), dict) else {}
            render = images.get("render")
            if not isinstance(render, str) or not render:
                errors.append(f"{episode_path}:{line_number}: observation.images.render is required")
            else:
                image_paths_checked += 1
                if not (dataset_root / render).exists():
                    errors.append(f"{episode_path}:{line_number}: missing rendered image {render}")

    return {
        "rows_checked": rows_checked,
        "image_paths_checked": image_paths_checked,
        "source_state_dims": sorted(source_state_dims),
        "exported_state_dim": len(joint_names),
        "action_dims": sorted(action_dims),
        "tasks_seen": sorted(tasks_seen),
        "info_joint_name_aliases": sorted(info_joint_name_aliases),
        "state_mapping": "source observation.state may include robot 7D plus cube-position metadata; converter exports the first 7 robot-control entries",
        "errors": errors,
    }


def _normalize_joint_name(name: str) -> str:
    return JOINT_NAME_ALIASES.get(name, name)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _expect_equal(errors: list[str], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"manifest {field} {actual!r} != expected {expected!r}")


if __name__ == "__main__":
    main()
