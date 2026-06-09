#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_relocation_verifier_packet(
    *,
    vla_prompt_packet: Path,
    execution_report: Path | None = None,
    before_image: Path | None = None,
    after_image: Path | None = None,
    output: Path | None = None,
    relocation_output: Path | None = None,
    min_delta_px: float | None = None,
    min_object_area_px: int = 800,
    color_preset: str = "green",
    python_executable: str = ".venv/bin/python",
) -> dict[str, Any]:
    packet = _load_json(vla_prompt_packet)
    success = packet.get("success_verifier", {})
    frame = success.get("frame", {})
    observer_camera_index = str(frame.get("primary_camera_index") or "3")
    report_images = _images_from_execution_report(execution_report) if execution_report else {}
    if before_image is None and report_images.get("before_image"):
        before_image = Path(str(report_images["before_image"]))
    if after_image is None and report_images.get("after_image"):
        after_image = Path(str(report_images["after_image"]))
    report_camera_index = report_images.get("camera_index")
    observer_camera_matches_report = (
        report_camera_index is None or str(report_camera_index) == observer_camera_index
    )
    source_images_exist = _path_exists(before_image) and _path_exists(after_image)
    target_direction = str(success.get("target_direction") or "right")
    threshold = float(min_delta_px if min_delta_px is not None else success.get("min_delta_px_default", 40.0))
    result_output = relocation_output or _default_relocation_output(output=output, target_direction=target_direction)
    has_images = before_image is not None and after_image is not None and source_images_exist
    can_run = bool(has_images and observer_camera_matches_report)
    command_template = _command(
        python_executable=python_executable,
        before="${before_observer_image}",
        after="${after_observer_image}",
        target_direction=target_direction,
        min_delta_px=threshold,
        min_object_area_px=min_object_area_px,
        color_preset=color_preset,
        output="${relocation_verifier_output}",
    )
    manifest = {
        "status": _status(
            has_images=has_images,
            source_images_exist=source_images_exist,
            observer_camera_matches_report=observer_camera_matches_report,
            execution_report=execution_report,
        ),
        "operation": "real_so100_relocation_verifier_packet",
        "purpose": "standardize post-action task verification from the SmolVLA prompt packet",
        "vla_prompt_packet": str(vla_prompt_packet),
        "source_execution_report": str(execution_report) if execution_report else None,
        "source_execution_report_status": report_images.get("report_status"),
        "source_execution_send_action_called": report_images.get("send_action_called"),
        "source_execution_camera_index": str(report_camera_index) if report_camera_index is not None else None,
        "observer_camera_matches_report": observer_camera_matches_report,
        "source_images_exist": source_images_exist,
        "verifier_target": success.get("type") or "object_relocation_image_space",
        "target_object": success.get("target_object"),
        "target_direction": target_direction,
        "success_predicate": success.get("success_predicate"),
        "min_delta_px": threshold,
        "min_object_area_px": min_object_area_px,
        "color_preset": color_preset,
        "verifier_frame": frame,
        "observer_camera_index": observer_camera_index,
        "policy_inputs_are_not_verifier_frame": _policy_inputs_are_not_verifier_frame(packet),
        "before_after_required": True,
        "before_image": str(before_image) if before_image else None,
        "after_image": str(after_image) if after_image else None,
        "relocation_output": str(result_output),
        "can_run": can_run,
        "cannot_run_without_before_after_images": not has_images,
        "task_success_claim_allowed_without_this": False,
        "vla_prompt_target": (packet.get("vla_prompt", {}) or {}).get("target"),
        "does_not_prompt_operator": (packet.get("agentic_layer_contract", {}) or {}).get(
            "does_not_prompt_operator", True
        ),
        "coordinate_guardrails": {
            "do_not_translate_goal_to_fixed_robot_direction": success.get(
                "do_not_translate_goal_to_fixed_robot_direction", True
            ),
            "not_equivalent_to": (packet.get("coordinate_semantics", {}) or {}).get("not_equivalent_to", []),
        },
        "command_template": command_template,
        "command": (
            _command(
                python_executable=python_executable,
                before=str(before_image),
                after=str(after_image),
                target_direction=target_direction,
                min_delta_px=threshold,
                min_object_area_px=min_object_area_px,
                color_preset=color_preset,
                output=str(result_output),
            )
            if can_run
            else None
        ),
        "notes": [
            "Run this only after a physical attempt has before/after observer-camera images.",
            "The verifier checks object displacement in image space; it does not assume a robot-arm direction.",
            "A transport task cannot be counted as successful without this task-level verifier and grasp evidence.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest["manifest_path"] = str(output)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _images_from_execution_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    report = _load_json(path)
    visual = report.get("visual_check", {}) or {}
    before = visual.get("before", {}) or {}
    after = visual.get("after", {}) or {}
    return {
        "report_status": report.get("status"),
        "send_action_called": report.get("send_action_called"),
        "camera_index": report.get("camera_index") or before.get("camera_index") or after.get("camera_index"),
        "before_image": before.get("image_path") or after.get("before_path"),
        "after_image": after.get("image_path"),
    }


def _status(
    *,
    has_images: bool,
    source_images_exist: bool,
    observer_camera_matches_report: bool,
    execution_report: Path | None,
) -> str:
    if not has_images:
        if execution_report is not None and not source_images_exist:
            return "source_images_missing"
        return "waiting_for_before_after_images"
    if not observer_camera_matches_report:
        return "observer_camera_mismatch"
    return "ready"


def _path_exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def _command(
    *,
    python_executable: str,
    before: str,
    after: str,
    target_direction: str,
    min_delta_px: float,
    min_object_area_px: int,
    color_preset: str,
    output: str,
) -> list[str]:
    return [
        "PYTHONPATH=src:.",
        python_executable,
        "-B",
        "scripts/real_so100_object_relocation.py",
        "--before",
        before,
        "--after",
        after,
        "--target-direction",
        target_direction,
        "--min-delta-px",
        _format_number(min_delta_px),
        "--min-object-area-px",
        str(min_object_area_px),
        "--color-preset",
        color_preset,
        "--output",
        output,
    ]


def _default_relocation_output(*, output: Path | None, target_direction: str) -> Path:
    if output is None:
        return Path(f"_workspace/real_so100/reports/object_relocation_{target_direction}.json")
    return output.with_name(f"{output.stem}_result.json")


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _policy_inputs_are_not_verifier_frame(packet: dict[str, Any]) -> bool:
    prompt = packet.get("vla_prompt", {}) or {}
    policy = {str(item) for item in prompt.get("policy_camera_indexes") or []}
    observer = {str(item) for item in prompt.get("observer_camera_indexes_excluded_from_policy") or []}
    frame = packet.get("success_verifier", {}).get("frame", {}) or {}
    primary = str(frame.get("primary_camera_index") or "3")
    return primary in observer and primary not in policy


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the real SO-100 post-action relocation verifier packet from a SmolVLA prompt packet."
    )
    parser.add_argument("--vla-prompt-packet", type=Path, required=True)
    parser.add_argument("--execution-report", type=Path)
    parser.add_argument("--before-image", type=Path)
    parser.add_argument("--after-image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--relocation-output", type=Path)
    parser.add_argument("--min-delta-px", type=float)
    parser.add_argument("--min-object-area-px", type=int, default=800)
    parser.add_argument("--color-preset", choices=["green"], default="green")
    parser.add_argument("--python-executable", default=".venv/bin/python")
    args = parser.parse_args()
    print(
        json.dumps(
            build_relocation_verifier_packet(
                vla_prompt_packet=args.vla_prompt_packet,
                execution_report=args.execution_report,
                before_image=args.before_image,
                after_image=args.after_image,
                output=args.output,
                relocation_output=args.relocation_output,
                min_delta_px=args.min_delta_px,
                min_object_area_px=args.min_object_area_px,
                color_preset=args.color_preset,
                python_executable=args.python_executable,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
