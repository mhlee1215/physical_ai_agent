#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_mycobot280_training_dataset import validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a config-first myCobot 280 SmolVLA fine-tuning readiness dry-run report."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("_workspace/mycobot280_training/ground_pickup_tiny_smoke/dry_run.json"))
    parser.add_argument("--runtime-platform", choices=["auto", "macos", "linux"], default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_dry_run_report(
        config_path=args.config,
        dataset_root_override=args.dataset_root,
        runtime_platform=args.runtime_platform,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"ready", "blocked"} else 1)


def build_dry_run_report(
    *,
    config_path: Path,
    dataset_root_override: Path | None = None,
    runtime_platform: str = "auto",
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = validate_config(
        config_path=config_path,
        dataset_root_override=dataset_root_override,
        require_present=False,
    )
    source = config["source_dataset"]
    conversion = config["lerobot_conversion"]
    smoke = config["training_smoke"]
    feature = config["feature_contract"]
    robot = config["robot"]
    closed_loop = config["closed_loop_stub"]
    dataset_root = Path(str(dataset_root_override or source["root"]))
    native_root = Path(str(conversion["output_root"]))
    output_dir = Path(str(smoke["output_dir"]))

    status = "ready" if validation["status"] == "passed" else "blocked"
    blocker = None
    if validation["status"] == "blocked":
        blocker = "source teacher dataset is not present yet"
    elif validation["status"] == "failed":
        blocker = "source teacher dataset config or manifest failed validation"
        status = "blocked"

    report = {
        "operation": "plan_mycobot280_smolvla_training",
        "status": status,
        "config_path": str(config_path),
        "runtime_platform": runtime_platform,
        "blocker": blocker,
        "validation": validation,
        "resolved_contract": {
            "robot": robot["name"],
            "scenario": config["scenario"],
            "task_prompt": config["task_prompt"],
            "base_checkpoint": feature["base_checkpoint"],
            "state_key": feature["state_key"],
            "action_key": feature["action_key"],
            "state_dim": robot["state_dim"],
            "action_dim": robot["action_dim"],
            "joint_names": robot["joint_names"],
            "camera_contract": feature["camera_contract"],
            "dataset_root": str(dataset_root),
            "native_lerobot_root": str(native_root),
            "training_output_dir": str(output_dir),
            "tensorboard_dir": smoke["tensorboard_dir"],
            "checkpoint_dir": smoke["checkpoint_dir"],
        },
        "commands": {
            "validate_source_dataset": [
                "PYTHONPATH=src:.",
                "python3",
                "scripts/validate_mycobot280_training_dataset.py",
                "--config",
                str(config_path),
            ],
            "generate_source_dataset": source.get("generation_command"),
            "plan_lerobot_conversion": [
                "PYTHONPATH=src:.",
                "python3",
                str(conversion["converter_script"]),
                "--source-root",
                str(dataset_root),
                "--output-root",
                str(native_root),
                "--repo-id",
                str(conversion["repo_id"]),
                "--dry-run",
            ],
            "native_lerobot_conversion_when_runtime_available": [
                "PYTHONPATH=src:.",
                "python3",
                str(conversion["native_converter_script"]),
                "--source-root",
                str(native_root),
                "--output-root",
                str(native_root) + "_native",
                "--repo-id",
                str(conversion["repo_id"]),
                "--require-lerobot",
            ],
            "tiny_smolvla_smoke_when_runtime_available": [
                "PYTHONPATH=src:.",
                "python3",
                "scripts/run_mycobot_280_pi_smolvla_tiny_smoke.py",
                "--dataset-root",
                str(native_root) + "_native",
                "--dataset-repo-id",
                str(conversion["repo_id"]),
                "--policy-path",
                str(smoke["policy_path"]),
                "--output-path",
                str(output_dir / "tiny_smoke.json"),
                "--batch-size",
                str(smoke["batch_size"]),
                "--max-batches",
                str(smoke["max_batches"]),
                "--device",
                str(smoke["device"]),
                "--require-runtime",
            ],
            "closed_loop_eval_stub": [
                "PYTHONPATH=src:.",
                "python3",
                str(closed_loop["script"]),
                "--policy-path",
                str(output_dir / "checkpoints" / "latest" / "pretrained_model"),
                "--config",
                str(config_path),
                "--output-dir",
                str(closed_loop["output_dir"]),
            ],
        },
        "artifact_plan": {
            "validation_report": str(output_dir / "dataset_validation.json"),
            "conversion_report": str(native_root / "mycobot280_ground_pickup_lerobot_plan.json"),
            "training_log": str(output_dir / "train.log"),
            "tensorboard_dir": smoke["tensorboard_dir"],
            "checkpoint_dir": smoke["checkpoint_dir"],
            "closed_loop_report": str(Path(str(closed_loop["output_dir"])) / "eval_report.json"),
        },
        "dependency_policy": (
            "Do not silently install or upgrade Torch, LeRobot, SmolVLA, MuJoCo, LIBERO, "
            "or system packages. If a command blocks on missing runtime, request approval "
            "before installing/downloading."
        ),
        "claim_boundary": "Dry-run readiness only; no policy-performance claim.",
    }
    return report


if __name__ == "__main__":
    main()
