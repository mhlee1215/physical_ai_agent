#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


DEFAULT_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_REPO_ID = "physical-ai-agent/mycobot-280pi-adaptive"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run, or explicitly block, a tiny SmolVLA supervised-loss smoke over the "
            "native myCobot 280 Pi adaptive LeRobotDataset."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Fail instead of writing a blocked report when LeRobot/SmolVLA runtime imports are unavailable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_mycobot_280_pi_smolvla_tiny_smoke(
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        policy_path=args.policy_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        local_files_only=args.local_files_only,
        require_runtime=args.require_runtime,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"passed", "blocked"} else 1)


def run_mycobot_280_pi_smolvla_tiny_smoke(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    policy_path: str,
    output_path: Path,
    batch_size: int,
    max_batches: int,
    device: str,
    local_files_only: bool,
    require_runtime: bool = False,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    dataset_audit = audit_native_lerobot_dataset_root(dataset_root)
    if dataset_audit["status"] != "passed":
        report = _blocked_report(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            policy_path=policy_path,
            output_path=output_path,
            blocker="native LeRobotDataset root is incomplete",
            dataset_audit=dataset_audit,
        )
        _write_report(output_path, report)
        if require_runtime:
            raise RuntimeError(report["blocker"])
        return report

    if evaluator is None:
        try:
            from scripts.evaluate_smolvla_supervised_loss import evaluate_supervised_loss
        except Exception as exc:  # noqa: BLE001
            report = _blocked_report(
                dataset_root=dataset_root,
                dataset_repo_id=dataset_repo_id,
                policy_path=policy_path,
                output_path=output_path,
                blocker=f"SmolVLA supervised-loss runtime import failed: {exc}",
                dataset_audit=dataset_audit,
            )
            _write_report(output_path, report)
            if require_runtime:
                raise RuntimeError(report["blocker"]) from exc
            return report
        evaluator = evaluate_supervised_loss

    loss_output = output_path.with_name(output_path.stem + "_supervised_loss.json")
    try:
        loss_report = evaluator(
            policy_path=policy_path,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            output_path=loss_output,
            batch_size=int(batch_size),
            num_workers=0,
            max_batches=int(max_batches),
            device=device,
            local_files_only=local_files_only,
            torch_seed=1000,
        )
    except Exception as exc:  # noqa: BLE001
        report = _blocked_report(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            policy_path=policy_path,
            output_path=output_path,
            blocker=f"SmolVLA supervised-loss smoke failed: {exc}",
            dataset_audit=dataset_audit,
        )
        _write_report(output_path, report)
        if require_runtime:
            raise
        return report

    report = {
        "operation": "run_mycobot_280_pi_smolvla_tiny_smoke",
        "status": "passed",
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "policy_path": policy_path,
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "device": device,
        "local_files_only": bool(local_files_only),
        "dataset_audit": dataset_audit,
        "loss_report": loss_report,
        "claim_boundary": (
            "This is a tiny supervised-loss smoke over an existing native LeRobotDataset. "
            "It does not prove closed-loop robot task success."
        ),
    }
    _write_report(output_path, report)
    return report


def audit_native_lerobot_dataset_root(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    converter_report = dataset_root / "mycobot_280_pi_lerobot_convert_report.json"
    errors: list[str] = []
    info: dict[str, Any] = {}
    if not info_path.exists():
        errors.append(f"missing {info_path}")
    else:
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid {info_path}: {exc}")
    if not data_files:
        errors.append("missing data/chunk-*/file-*.parquet")
    if not episode_files:
        errors.append("missing meta/episodes/chunk-*/file-*.parquet")
    if not tasks_parquet.exists():
        errors.append("missing meta/tasks.parquet")
    return {
        "status": "passed" if not errors else "blocked",
        "dataset_root": str(dataset_root),
        "info_path": str(info_path),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "data_files": [str(path) for path in data_files],
        "episode_files": [str(path) for path in episode_files],
        "tasks_parquet": str(tasks_parquet),
        "converter_report": str(converter_report) if converter_report.exists() else None,
        "errors": errors,
    }


def _blocked_report(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    policy_path: str,
    output_path: Path,
    blocker: str,
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operation": "run_mycobot_280_pi_smolvla_tiny_smoke",
        "status": "blocked",
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "policy_path": policy_path,
        "output_path": str(output_path),
        "blocker": blocker,
        "dataset_audit": dataset_audit,
        "install_command": "sh scripts/install/local_install.sh --checkpoint 05-06",
        "approval_required": True,
        "native_conversion_command": (
            "PYTHONPATH=src:. python3 scripts/convert_mycobot_280_pi_adaptive_jsonl_to_lerobot.py "
            "--source-root _workspace/mycobot280_lerobot/ground_pickup_tiny_smoke "
            "--output-root _workspace/mycobot280_lerobot/ground_pickup_tiny_smoke_native "
            "--repo-id physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke "
            "--require-lerobot"
        ),
        "next_step": (
            "After approval, install the LeRobot/SmolVLA runtime, create the native 280 "
            "LeRobotDataset, then rerun this script with --require-runtime."
        ),
        "claim_boundary": "No SmolVLA smoke was completed.",
    }


def _write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
