#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    evidence: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether LIBERO/LeRobot artifacts are ready for in-episode intervention metrics."
    )
    parser.add_argument("--lerobot-eval-source", type=Path)
    parser.add_argument("--libero-env-source", type=Path)
    parser.add_argument("--eval-info", type=Path, action="append", default=[])
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    checks: list[CheckResult] = []
    checks.extend(check_lerobot_eval_source(args.lerobot_eval_source))
    checks.extend(check_libero_env_source(args.libero_env_source))
    checks.extend(check_eval_infos(args.eval_info))

    payload = {
        "lerobot_eval_source": str(args.lerobot_eval_source) if args.lerobot_eval_source else None,
        "libero_env_source": str(args.libero_env_source) if args.libero_env_source else None,
        "eval_infos": [str(path) for path in args.eval_info],
        "checks": [check.__dict__ for check in checks],
        "summary": summarize(checks),
        "next_experiment_contract": next_experiment_contract(),
    }
    args.output_md.write_text(render_markdown(payload, checks), encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={args.output_md}")
    print(f"summary={args.output_json}")


def check_lerobot_eval_source(path: Path | None) -> list[CheckResult]:
    if path is None:
        return [CheckResult("lerobot_eval_source_present", "missing", "No --lerobot-eval-source provided.")]
    text = path.read_text(encoding="utf-8")
    return [
        CheckResult(
            "lerobot_rollout_function",
            "pass" if "def rollout(" in text else "fail",
            f"{path}: contains def rollout(" if "def rollout(" in text else f"{path}: def rollout( not found",
        ),
        CheckResult(
            "rollout_records_actions",
            "pass" if "all_actions.append" in text and "ACTION: torch.stack(all_actions" in text else "fail",
            "rollout returns stacked ACTION tensor"
            if "all_actions.append" in text and "ACTION: torch.stack(all_actions" in text
            else "stacked ACTION tensor not found",
        ),
        CheckResult(
            "rollout_records_success_done",
            "pass" if '"success": torch.stack(all_successes' in text and '"done": torch.stack(all_dones' in text else "fail",
            "rollout returns success and done sequences"
            if '"success": torch.stack(all_successes' in text and '"done": torch.stack(all_dones' in text
            else "success/done sequence tensors not found",
        ),
        CheckResult(
            "online_hook_location",
            "pass" if "observation, reward, terminated, truncated, info = env.step(action_numpy)" in text else "fail",
            "hook can be inserted immediately before/after env.step(action_numpy)"
            if "observation, reward, terminated, truncated, info = env.step(action_numpy)" in text
            else "env.step(action_numpy) line not found",
        ),
        CheckResult(
            "default_eval_info_action_steps",
            "fail" if '"per_episode":' in text and '"episode_ix"' in text and '"action_steps"' not in text else "unknown",
            "default eval_info per_episode lacks action_steps"
            if '"per_episode":' in text and '"episode_ix"' in text and '"action_steps"' not in text
            else "could not determine default eval_info action-step fields",
        ),
    ]


def check_libero_env_source(path: Path | None) -> list[CheckResult]:
    if path is None:
        return [CheckResult("libero_env_source_present", "missing", "No --libero-env-source provided.")]
    text = path.read_text(encoding="utf-8")
    return [
        CheckResult(
            "libero_step_exposes_success",
            "pass" if "is_success = self._env.check_success()" in text and '"is_success": is_success' in text else "fail",
            "LiberoEnv.step exposes check_success via info['is_success']"
            if "is_success = self._env.check_success()" in text and '"is_success": is_success' in text
            else "LiberoEnv.step success info not found",
        ),
        CheckResult(
            "libero_step_auto_resets_on_terminal",
            "warn" if "if terminated:" in text and "self.reset()" in text else "pass",
            "LiberoEnv.step auto-resets after terminated; custom intervention loop must account for this"
            if "if terminated:" in text and "self.reset()" in text
            else "no auto-reset detected in provided source",
        ),
    ]


def check_eval_infos(paths: list[Path]) -> list[CheckResult]:
    if not paths:
        return [CheckResult("eval_info_present", "missing", "No --eval-info files provided.")]

    checks: list[CheckResult] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        prefix = f"eval_info:{path.name}"
        checks.append(
            CheckResult(
                f"{prefix}:eval_seconds",
                "pass" if _eval_seconds(data) > 0 else "fail",
                f"overall.eval_s={_eval_seconds(data):.3f}",
            )
        )
        checks.append(
            CheckResult(
                f"{prefix}:successes",
                "pass" if count_success_entries(data) > 0 else "fail",
                f"success entries={count_success_entries(data)}",
            )
        )
        action_step_fields = find_action_step_fields(data)
        checks.append(
            CheckResult(
                f"{prefix}:action_step_counts",
                "pass" if action_step_fields else "fail",
                "action step fields=" + ",".join(action_step_fields)
                if action_step_fields
                else "no per-episode action-step count fields found",
            )
        )
    return checks


def summarize(checks: list[CheckResult]) -> dict[str, int]:
    return {
        "pass": sum(1 for check in checks if check.status == "pass"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "fail": sum(1 for check in checks if check.status == "fail"),
        "missing": sum(1 for check in checks if check.status == "missing"),
        "unknown": sum(1 for check in checks if check.status == "unknown"),
    }


def render_markdown(payload: dict[str, Any], checks: list[CheckResult]) -> str:
    lines = [
        "# LIBERO In-Episode Intervention Readiness Report",
        "",
        f"- lerobot_eval_source: `{payload['lerobot_eval_source']}`",
        f"- libero_env_source: `{payload['libero_env_source']}`",
        f"- eval_infos: `{', '.join(payload['eval_infos'])}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.evidence} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Pass | Warn | Fail | Missing | Unknown |",
            "| ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {payload['summary']['pass']} | {payload['summary']['warn']} | "
                f"{payload['summary']['fail']} | {payload['summary']['missing']} | {payload['summary']['unknown']} |"
            ),
            "",
            "## Next Experiment Contract",
            "",
        ]
    )
    for item in payload["next_experiment_contract"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def next_experiment_contract() -> list[str]:
    return [
        "Use a custom rollout path or patch LeRobot rollout to emit per-episode action_step_count.",
        "Insert the online verifier immediately before or after env.step(action_numpy).",
        "Log verifier_triggered, trigger_step, intervention_type, action_steps, eval_seconds, and final benchmark success.",
        "Treat LiberoEnv auto-reset after terminal success/failure as a boundary; in-episode interventions must occur before terminated=True.",
        "Compare policy-only, blind retry, horizon-switch retry, and in-episode intervention under the same task/seed/action budget.",
    ]


def _eval_seconds(data: dict[str, Any]) -> float:
    try:
        return float(data.get("overall", {}).get("eval_s", 0.0))
    except (TypeError, ValueError):
        return 0.0


def count_success_entries(data: dict[str, Any]) -> int:
    total = 0
    for item in data.get("per_task", []):
        if isinstance(item, dict):
            metrics = item.get("metrics", {})
            if isinstance(metrics, dict):
                total += len(metrics.get("successes", []))
    return total


def find_action_step_fields(data: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if ("action" in lowered and "step" in lowered) or lowered in {"num_frames", "frame_index"}:
                fields.append(path)
            fields.extend(find_action_step_fields(value, path))
    elif isinstance(data, list):
        for index, value in enumerate(data[:3]):
            fields.extend(find_action_step_fields(value, f"{prefix}[{index}]"))
    return sorted(set(fields))


if __name__ == "__main__":
    main()
