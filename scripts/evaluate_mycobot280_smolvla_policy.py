#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_mycobot280_training_dataset import validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run a myCobot 280 SmolVLA closed-loop simulation evaluation."
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-policy", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_eval_report(
        policy_path=args.policy_path,
        config_path=args.config,
        output_dir=args.output_dir,
        episodes=args.episodes,
        dry_run=args.dry_run,
        require_policy=args.require_policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"planned", "blocked"} else 1)


def build_eval_report(
    *,
    policy_path: Path,
    config_path: Path,
    output_dir: Path,
    episodes: int | None,
    dry_run: bool,
    require_policy: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = validate_config(config_path=config_path, require_present=False)
    closed_loop = config["closed_loop_stub"]
    requested_episodes = int(episodes if episodes is not None else closed_loop.get("episodes", 3))
    policy_exists = policy_path.exists()
    status = "planned" if dry_run or policy_exists else "blocked"
    if require_policy and not policy_exists:
        status = "blocked"
    return {
        "operation": "evaluate_mycobot280_smolvla_policy",
        "status": status,
        "policy_path": str(policy_path),
        "policy_exists": policy_exists,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "episodes": requested_episodes,
        "validation_status": validation["status"],
        "blocker": None if status == "planned" else "policy checkpoint does not exist yet; run tiny smoke/fine-tune first",
        "scenario": config["scenario"],
        "task_prompt": config["task_prompt"],
        "metrics": closed_loop["metrics"],
        "comparison_rows": closed_loop["future_comparison_rows"],
        "planned_execution": {
            "mode": "policy_only_closed_loop_sim",
            "robot": config["robot"]["name"],
            "state_dim": config["robot"]["state_dim"],
            "action_dim": config["robot"]["action_dim"],
            "success_source": "myCobot 280 ground-pickup contact/lift/hold verifier",
        },
        "claim_boundary": "Closed-loop evaluation stub only; no policy rollout was executed by this dry-run/blocker report.",
    }


if __name__ == "__main__":
    main()
