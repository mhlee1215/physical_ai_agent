#!/usr/bin/env python3
"""Merge LeRobot eval_info.json files from split evaluation lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_infos", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_values(per_task: list[dict[str, Any]], key: str) -> list[float | bool]:
    values: list[float | bool] = []
    for item in per_task:
        values.extend(item.get("metrics", {}).get(key, []))
    return values


def mean(values: list[float | bool]) -> float:
    if not values:
        return float("nan")
    return sum(float(value) for value in values) / len(values)


def summarize_task_group(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    sum_rewards = metric_values(tasks, "sum_rewards")
    max_rewards = metric_values(tasks, "max_rewards")
    successes = metric_values(tasks, "successes")
    video_paths: list[str] = []
    for item in tasks:
        video_paths.extend(item.get("metrics", {}).get("video_paths", []))
    return {
        "avg_sum_reward": mean(sum_rewards),
        "avg_max_reward": mean(max_rewards),
        "pc_success": 100.0 * mean(successes),
        "n_episodes": len(successes),
        "video_paths": video_paths,
    }


def merge_eval_infos(paths: list[Path]) -> dict[str, Any]:
    all_tasks: list[dict[str, Any]] = []
    eval_seconds = 0.0
    for path in paths:
        data = load_json(path)
        all_tasks.extend(data.get("per_task", []))
        eval_seconds += float(data.get("overall", {}).get("eval_s", 0.0))

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in all_tasks:
        groups.setdefault(str(item.get("task_group")), []).append(item)

    per_group = {group: summarize_task_group(tasks) for group, tasks in sorted(groups.items())}
    overall = summarize_task_group(all_tasks)
    overall["eval_s"] = eval_seconds
    overall["eval_ep_s"] = eval_seconds / overall["n_episodes"] if overall["n_episodes"] else float("nan")

    return {
        "per_task": all_tasks,
        "per_group": per_group,
        "overall": overall,
        "merged_from": [str(path) for path in paths],
    }


def main() -> None:
    args = parse_args()
    merged = merge_eval_infos(args.eval_infos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
