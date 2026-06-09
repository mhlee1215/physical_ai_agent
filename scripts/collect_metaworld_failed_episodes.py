#!/usr/bin/env python3
"""Collect failed Meta-World Very Hard episodes into a review pack."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


TASK_LINE_RE = re.compile(
    r"Building vec env \| group=(?P<group>[^ ]+) \| task_id=(?P<task_id>\d+) \| task=(?P<task>[^ ]+)"
)


def repo_rel(path: Path) -> str:
    return str(path).replace(str(Path.cwd()) + "/", "")


def task_names(root: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in (root / "eval.log").read_text(errors="replace").splitlines():
        match = TASK_LINE_RE.search(line)
        if not match or match.group("group") != "very_hard":
            continue
        names[int(match.group("task_id"))] = match.group("task")
    return names


def failure_kind(sum_reward: float, max_reward: float) -> str:
    if max_reward <= 0.05 and sum_reward <= 1.0:
        return "zero_progress"
    if max_reward < 0.5:
        return "minimal_progress"
    if max_reward < 3.0:
        return "partial_shaped_reward"
    return "near_success_or_contact_failure"


def safe_name(task_name: str) -> str:
    return task_name.replace("-v3", "").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/"
            "metaworld_public_full_mt50_10ep_nas15_20260608T1635Z"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("_workspace/runpod_results/metaworld_very_hard_failed_pack_nas15"),
    )
    args = parser.parse_args()

    eval_info = json.loads((args.root / "eval" / "eval_info.json").read_text())
    names = task_names(args.root)
    videos_dir = args.out / "failed_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for task in eval_info["per_task"]:
        if task["task_group"] != "very_hard":
            continue
        task_id = int(task["task_id"])
        task_name = names.get(task_id, f"very_hard_{task_id}")
        metrics = task["metrics"]
        for episode, (success, sum_reward, max_reward) in enumerate(
            zip(metrics["successes"], metrics["sum_rewards"], metrics["max_rewards"])
        ):
            if success:
                continue
            source = args.root / "eval" / "videos" / f"very_hard_{task_id}" / f"eval_episode_{episode}.mp4"
            filename = (
                f"task{task_id}_{safe_name(task_name)}_ep{episode:02d}_"
                f"sum{float(sum_reward):.1f}_max{float(max_reward):.2f}.mp4"
            )
            target = videos_dir / filename
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(Path(os.path.relpath(source, target.parent)), target)
            rows.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode": episode,
                    "sum_reward": float(sum_reward),
                    "max_reward": float(max_reward),
                    "failure_kind": failure_kind(float(sum_reward), float(max_reward)),
                    "source_video": repo_rel(source),
                    "pack_video": repo_rel(target),
                    "exists": source.exists(),
                }
            )

    rows.sort(key=lambda r: (int(r["task_id"]), int(r["episode"])))

    manifest_json = args.out / "failed_manifest.json"
    manifest_csv = args.out / "failed_manifest.csv"
    manifest_md = args.out / "README.md"
    manifest_json.write_text(json.dumps(rows, indent=2))

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_kind: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for row in rows:
        by_kind[str(row["failure_kind"])] = by_kind.get(str(row["failure_kind"]), 0) + 1
        by_task[str(row["task_name"])] = by_task.get(str(row["task_name"]), 0) + 1

    lines = [
        "# Meta-World Very Hard Failed Episode Pack - n_action_steps=15",
        "",
        f"Source root: `{repo_rel(args.root)}`",
        "",
        f"Failed episodes collected: `{len(rows)}`",
        "",
        "## Counts By Task",
        "",
        "| Task | Failed episodes |",
        "| --- | ---: |",
    ]
    for task_name, count in sorted(by_task.items()):
        lines.append(f"| `{task_name}` | {count} |")
    lines.extend(["", "## Counts By Failure Kind", "", "| Failure kind | Count |", "| --- | ---: |"])
    for kind, count in sorted(by_kind.items()):
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(
        [
            "",
            "## Failure Kind Heuristic",
            "",
            "- `zero_progress`: no meaningful reward; likely no useful approach/contact.",
            "- `minimal_progress`: tiny shaped reward; likely brief approach/contact but no task progress.",
            "- `partial_shaped_reward`: sustained shaped reward but no success event.",
            "- `near_success_or_contact_failure`: high shaped reward but still no success event.",
            "",
            "## Files",
            "",
            "- `failed_manifest.json`: structured manifest.",
            "- `failed_manifest.csv`: spreadsheet-friendly manifest.",
            "- `failed_videos/`: symlinks to local failed episode videos.",
            "- `thumbnails/`: optional QuickLook thumbnails, generated separately.",
            "",
        ]
    )
    manifest_md.write_text("\n".join(lines))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
