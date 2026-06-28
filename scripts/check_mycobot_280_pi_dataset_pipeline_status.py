#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.check_mycobot_280_pi_gate8_readiness import check_readiness
from scripts.run_mycobot_280_pi_smolvla_tiny_smoke import audit_native_lerobot_dataset_root
from scripts.verify_mycobot_280_pi_capture_contract import verify_mycobot_280_pi_capture_contract


@dataclass(frozen=True)
class PipelineStage:
    name: str
    status: str
    evidence: dict[str, Any]
    next_command: str
    claim_boundary: str


@dataclass(frozen=True)
class PipelineStatusReport:
    status: str
    stages: list[PipelineStage]
    first_blocked_stage: str | None
    output_path: str | None
    claim_boundary: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize myCobot 280 Pi adaptive progress from Gate 7/8 assets through "
            "capture, JSONL export, native LeRobotDataset, and tiny SmolVLA smoke readiness."
        )
    )
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--input-trace", type=Path)
    parser.add_argument("--camera-manifest", type=Path)
    parser.add_argument("--jsonl-dataset-root", type=Path, default=Path("_workspace/mycobot_280pi_adaptive_dataset"))
    parser.add_argument(
        "--native-dataset-root",
        type=Path,
        default=Path("_workspace/mycobot_280pi_adaptive_lerobot_native"),
    )
    parser.add_argument(
        "--smolvla-smoke-report",
        type=Path,
        default=Path("_workspace/mycobot_280pi_adaptive_lerobot_native/smolvla_tiny_smoke.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = check_mycobot_280_pi_dataset_pipeline_status(
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        input_trace=args.input_trace,
        camera_manifest=args.camera_manifest,
        jsonl_dataset_root=args.jsonl_dataset_root,
        native_dataset_root=args.native_dataset_root,
        smolvla_smoke_report=args.smolvla_smoke_report,
        output=args.output,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    print(payload)
    raise SystemExit(0 if report.status in {"passed", "blocked"} else 1)


def check_mycobot_280_pi_dataset_pipeline_status(
    *,
    asset_root: Path,
    official_gripper_root: Path,
    input_trace: Path | None,
    camera_manifest: Path | None,
    jsonl_dataset_root: Path,
    native_dataset_root: Path,
    smolvla_smoke_report: Path,
    output: Path | None = None,
) -> PipelineStatusReport:
    stages = [
        _profile_and_gate8_stage(asset_root=asset_root, official_gripper_root=official_gripper_root),
        _capture_contract_stage(input_trace=input_trace, camera_manifest=camera_manifest),
        _jsonl_dataset_stage(jsonl_dataset_root=jsonl_dataset_root, input_trace=input_trace, camera_manifest=camera_manifest),
        _native_lerobot_stage(jsonl_dataset_root=jsonl_dataset_root, native_dataset_root=native_dataset_root),
        _smolvla_smoke_stage(native_dataset_root=native_dataset_root, smolvla_smoke_report=smolvla_smoke_report),
    ]
    first_blocked = next((stage.name for stage in stages if stage.status != "passed"), None)
    report = PipelineStatusReport(
        status="passed" if first_blocked is None else "blocked",
        stages=stages,
        first_blocked_stage=first_blocked,
        output_path=str(output) if output is not None else None,
        claim_boundary=(
            "This report summarizes executable gate readiness only. It does not prove real 280 "
            "physics, real camera capture quality, native LeRobot conversion, or SmolVLA policy "
            "quality unless the corresponding stage status is passed with its own artifacts."
        ),
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _profile_and_gate8_stage(*, asset_root: Path, official_gripper_root: Path) -> PipelineStage:
    readiness = check_readiness(asset_root=asset_root, official_gripper_root=official_gripper_root)
    return PipelineStage(
        name="profile_and_gate8_assets",
        status="passed" if readiness.status == "passed" else "blocked",
        evidence=asdict(readiness),
        next_command=(
            "PYTHONPATH=src:. python3 scripts/check_mycobot_280_pi_gate8_readiness.py "
            f"--asset-root {asset_root} --official-gripper-root {official_gripper_root} "
            "--output _workspace/mycobot_280_pi_gate8_readiness/report.json"
        ),
        claim_boundary="Passed means runtime/assets are available; Gate 7/8 physics still must run.",
    )


def _capture_contract_stage(*, input_trace: Path | None, camera_manifest: Path | None) -> PipelineStage:
    if input_trace is None or camera_manifest is None:
        return PipelineStage(
            name="capture_contract",
            status="blocked",
            evidence={"reason": "input trace or camera manifest not provided"},
            next_command=(
                "PYTHONPATH=src:. python3 scripts/verify_mycobot_280_pi_capture_contract.py "
                "--input-trace path/to/ros_gazebo_trace.jsonl "
                "--camera-manifest path/to/camera_manifest.json "
                "--output-dir _workspace/mycobot_280pi_capture_contract_verify"
            ),
            claim_boundary="Capture contract requires real trace rows and existing top/wrist camera frames.",
        )
    report = verify_mycobot_280_pi_capture_contract(
        input_trace=input_trace,
        camera_manifest=camera_manifest,
        output_dir=Path("_workspace/mycobot_280pi_capture_contract_verify"),
    )
    return PipelineStage(
        name="capture_contract",
        status="passed" if report.status == "passed" else "blocked",
        evidence=asdict(report),
        next_command=(
            "Fix capture rows until every frame has timestamp, 280 joint/action, object pose, "
            "contact evidence, and existing top/wrist images."
        ),
        claim_boundary=report.claim_boundary,
    )


def _jsonl_dataset_stage(
    *,
    jsonl_dataset_root: Path,
    input_trace: Path | None,
    camera_manifest: Path | None,
) -> PipelineStage:
    required = [
        jsonl_dataset_root / "data" / "frames.jsonl",
        jsonl_dataset_root / "data" / "episodes.jsonl",
        jsonl_dataset_root / "meta" / "info.json",
        jsonl_dataset_root / "meta" / "stats.json",
        jsonl_dataset_root / "meta" / "smolvla_tiny_smoke_plan.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return PipelineStage(
        name="jsonl_dataset_export",
        status="passed" if not missing else "blocked",
        evidence={"root": str(jsonl_dataset_root), "missing": missing},
        next_command=(
            "PYTHONPATH=src:. python3 scripts/export_mycobot_280_pi_adaptive_lerobot_dataset.py "
            f"--root {jsonl_dataset_root} "
            f"--input-trace {input_trace or 'path/to/ros_gazebo_trace.jsonl'} "
            f"--camera-manifest {camera_manifest or 'path/to/camera_manifest.json'} "
            "--repo-id physical-ai-agent/mycobot-280pi-adaptive --overwrite"
        ),
        claim_boundary="Passed means JSONL export files exist; native LeRobot parquet/video conversion is separate.",
    )


def _native_lerobot_stage(*, jsonl_dataset_root: Path, native_dataset_root: Path) -> PipelineStage:
    audit = audit_native_lerobot_dataset_root(native_dataset_root)
    return PipelineStage(
        name="native_lerobot_dataset",
        status="passed" if audit["status"] == "passed" else "blocked",
        evidence=audit,
        next_command=(
            "PYTHONPATH=src:. python3 scripts/convert_mycobot_280_pi_adaptive_jsonl_to_lerobot.py "
            f"--source-root {jsonl_dataset_root} --output-root {native_dataset_root} "
            "--repo-id physical-ai-agent/mycobot-280pi-adaptive --require-lerobot --overwrite"
        ),
        claim_boundary="Passed means native LeRobotDataset layout exists; it does not prove SmolVLA training.",
    )


def _smolvla_smoke_stage(*, native_dataset_root: Path, smolvla_smoke_report: Path) -> PipelineStage:
    smoke: dict[str, Any] = {}
    errors: list[str] = []
    if smolvla_smoke_report.exists():
        try:
            smoke = json.loads(smolvla_smoke_report.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid smoke report: {exc}")
    else:
        errors.append(f"missing {smolvla_smoke_report}")
    status = "passed" if smoke.get("status") == "passed" and not errors else "blocked"
    return PipelineStage(
        name="smolvla_tiny_smoke",
        status=status,
        evidence={"report_path": str(smolvla_smoke_report), "report": smoke, "errors": errors},
        next_command=(
            "PYTHONPATH=src:. python3 scripts/run_mycobot_280_pi_smolvla_tiny_smoke.py "
            f"--dataset-root {native_dataset_root} "
            "--dataset-repo-id physical-ai-agent/mycobot-280pi-adaptive "
            "--policy-path lerobot/smolvla_base "
            f"--output-path {smolvla_smoke_report} "
            "--max-batches 1 --require-runtime"
        ),
        claim_boundary="Passed means a one-batch supervised-loss SmolVLA smoke ran; not closed-loop robot success.",
    )


if __name__ == "__main__":
    main()
