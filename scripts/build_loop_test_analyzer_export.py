#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "loop-test-analyzer-v0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized loop-test analyzer export.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-source", action="store_true")
    args = parser.parse_args()

    manifest = build_export(args.run_dir, args.output_dir, copy_source=args.copy_source)
    print(json.dumps({"manifest_path": str(args.output_dir / "manifest.json"), **manifest["summary"]}, indent=2))


def build_export(run_dir: Path, output_dir: Path, *, copy_source: bool = False) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validations = _validation_by_checkpoint(run_dir)
    loop_tests = []
    for report_path in _closed_loop_report_paths(run_dir):
        report = _read_json(report_path)
        loop_test = _build_loop_test(run_dir, output_dir, report_path, report, validations, copy_source=copy_source)
        loop_tests.append(loop_test)
    loop_tests.sort(key=lambda row: (row.get("training_step") or -1, row.get("checkpoint") or ""))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "loop_tests": loop_tests,
        "summary": {
            "loop_tests": len(loop_tests),
            "checkpoints": [row["checkpoint"] for row in loop_tests],
            "successes": sum(1 for row in loop_tests if (row.get("success_rate") or 0) > 0),
            "latest_checkpoint": loop_tests[-1]["checkpoint"] if loop_tests else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _build_loop_test(
    run_dir: Path,
    output_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    validations: dict[str, dict[str, Any]],
    *,
    copy_source: bool,
) -> dict[str, Any]:
    checkpoint = _checkpoint_from_report_path(report_path)
    step = _checkpoint_to_step(checkpoint)
    loop_test_id = f"qwen_chain_{checkpoint}"
    loop_dir = output_dir / "loop_tests" / loop_test_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    source_dir = loop_dir / "source"
    if copy_source:
        source_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, source_dir / report_path.name)

    episode_rows = []
    for episode in report.get("episodes") or []:
        episode_rows.append(
            _write_episode_timeline(
                loop_dir=loop_dir,
                source_dir=source_dir,
                episode=episode,
                report=report,
                checkpoint=checkpoint,
                step=step,
                copy_source=copy_source,
            )
        )

    validation = validations.get(checkpoint, {})
    loop_manifest = {
        "schema_version": SCHEMA_VERSION,
        "loop_test_id": loop_test_id,
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "pick_up_cube",
        "policy_type": "qwen_chain",
        "policy_label": "Qwen chain + SmolVLA",
        "checkpoint": checkpoint,
        "training_step": step,
        "validation_loss": validation.get("loss"),
        "success_rate": report.get("success_rate"),
        "status": report.get("status"),
        "status_meaning": "evaluator_completed" if report.get("status") == "passed" else report.get("status"),
        "episodes_requested": report.get("episodes_requested"),
        "episodes_completed": report.get("episodes_completed"),
        "seed": report.get("seed"),
        "qwen_plan": report.get("plan"),
        "report_path": str(report_path),
        "source_report_path": str(source_dir / report_path.name) if copy_source else str(report_path),
        "episodes": episode_rows,
    }
    (loop_dir / "loop_test_manifest.json").write_text(
        json.dumps(loop_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "loop_test_id": loop_test_id,
        "manifest_path": str(loop_dir / "loop_test_manifest.json"),
        "checkpoint": checkpoint,
        "training_step": step,
        "scenario": loop_manifest["scenario"],
        "policy_type": loop_manifest["policy_type"],
        "policy_label": loop_manifest["policy_label"],
        "validation_loss": loop_manifest["validation_loss"],
        "success_rate": loop_manifest["success_rate"],
        "status": loop_manifest["status"],
        "episodes_completed": loop_manifest["episodes_completed"],
    }


def _write_episode_timeline(
    *,
    loop_dir: Path,
    source_dir: Path,
    episode: dict[str, Any],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    copy_source: bool,
) -> dict[str, Any]:
    episode_index = int(episode.get("episode") or 0)
    episode_dir = loop_dir / "episodes" / f"episode_{episode_index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(episode.get("trace_path") or "")
    if copy_source and trace_path.exists():
        shutil.copy2(trace_path, source_dir / trace_path.name)
    records = _read_jsonl(trace_path)
    timeline = _timeline_rows(records, report, checkpoint, step, episode_index)
    timeline_path = episode_dir / "timeline.jsonl"
    timeline_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in timeline) + "\n", encoding="utf-8")
    episode_manifest = {
        "episode_index": episode_index,
        "final_success": episode.get("final_success"),
        "total_reward": episode.get("total_reward"),
        "steps": episode.get("steps"),
        "reset_info": episode.get("reset_info"),
        "final_info": episode.get("final_info"),
        "timeline_path": str(timeline_path),
        "source_trace_path": str(source_dir / trace_path.name) if copy_source and trace_path.exists() else str(trace_path),
        "media_root": str(episode_dir / "media"),
        "iterations": _iteration_summary(timeline),
    }
    (episode_dir / "episode_manifest.json").write_text(
        json.dumps(episode_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return episode_manifest


def _timeline_rows(
    records: list[dict[str, Any]],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
) -> list[dict[str, Any]]:
    plan = report.get("plan") or {}
    rows: list[dict[str, Any]] = [
        {
            "type": "planner_call",
            "iteration": 0,
            "checkpoint": checkpoint,
            "training_step": step,
            "episode": episode_index,
            "policy": {"type": "qwen_chain", "model": plan.get("model"), "thinking_mode": plan.get("thinking_mode")},
            "policy_input": {"task": plan.get("task")},
            "policy_output": {"tool_calls": plan.get("calls") or []},
            "robot": None,
            "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
        }
    ]
    current_key: tuple[str | None, str | None] | None = None
    iteration = 0
    primitive_step_counts: dict[tuple[str | None, str | None], int] = defaultdict(int)
    for record in records:
        key = (record.get("primitive_id"), record.get("fn"))
        if key != current_key:
            if current_key is not None:
                rows.append(_tool_end_row(iteration, current_key, record, checkpoint, step, episode_index))
            iteration += 1
            current_key = key
            rows.append(_tool_start_row(iteration, record, checkpoint, step, episode_index))
        primitive_step_counts[key] += 1
        rows.append(_policy_step_row(iteration, record, checkpoint, step, episode_index))
    if current_key is not None and records:
        rows.append(_tool_end_row(iteration, current_key, records[-1], checkpoint, step, episode_index))
    rows.append(
        {
            "type": "episode_end",
            "iteration": iteration + 1,
            "checkpoint": checkpoint,
            "training_step": step,
            "episode": episode_index,
            "policy": None,
            "policy_input": None,
            "policy_output": None,
            "robot": {"final_info": (report.get("episodes") or [{}])[episode_index].get("final_info")},
            "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
        }
    )
    return rows


def _tool_start_row(iteration: int, record: dict[str, Any], checkpoint: str, step: int | None, episode_index: int) -> dict[str, Any]:
    return {
        "type": "tool_call_start",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "tool_call": record.get("fn"),
        "primitive_id": record.get("primitive_id"),
        "policy": {"type": "smolvla", "policy_path": record.get("policy_path")},
        "policy_input": {"prompt": record.get("prompt")},
        "policy_output": None,
        "robot": None,
        "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
    }


def _policy_step_row(iteration: int, record: dict[str, Any], checkpoint: str, step: int | None, episode_index: int) -> dict[str, Any]:
    return {
        "type": "policy_step",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "primitive_step": record.get("primitive_step"),
        "tool_call": record.get("fn"),
        "primitive_id": record.get("primitive_id"),
        "policy": {"type": "smolvla", "policy_path": record.get("policy_path")},
        "policy_input": {
            "prompt": record.get("prompt"),
            "observation": record.get("observation"),
            "image_feature_mapping": record.get("image_feature_mapping"),
            "images": {},
        },
        "policy_output": {"action": record.get("action"), "action_chunk": None},
        "robot": {
            "reward": record.get("reward"),
            "info": record.get("info"),
            "terminated": record.get("terminated"),
            "truncated": record.get("truncated"),
        },
        "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
        "source": {"record": record},
    }


def _tool_end_row(
    iteration: int,
    key: tuple[str | None, str | None],
    record: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
) -> dict[str, Any]:
    primitive_id, fn = key
    return {
        "type": "tool_call_end",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "tool_call": fn,
        "primitive_id": primitive_id,
        "policy": None,
        "policy_input": None,
        "policy_output": None,
        "robot": {"last_info": record.get("info"), "last_reward": record.get("reward")},
        "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
    }


def _iteration_summary(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in timeline:
        if row.get("type") == "tool_call_start":
            rows.append(
                {
                    "iteration": row.get("iteration"),
                    "tool_call": row.get("tool_call"),
                    "primitive_id": row.get("primitive_id"),
                    "start_global_step": row.get("global_step"),
                }
            )
    return rows


def _closed_loop_report_paths(run_dir: Path) -> list[Path]:
    root = run_dir / "closed_loop_evals"
    if not root.exists():
        return []
    return sorted(root.glob("*/qwen_closed_loop_eval_report.json"))


def _validation_by_checkpoint(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl")
    return {str(row["checkpoint"]): row for row in rows if row.get("checkpoint")}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_from_report_path(path: Path) -> str:
    match = path.parent.name.rsplit("_", 1)
    if len(match) == 2 and match[1].isdigit():
        return match[1]
    return path.parent.name


def _checkpoint_to_step(checkpoint: str) -> int | None:
    try:
        return int(checkpoint)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
