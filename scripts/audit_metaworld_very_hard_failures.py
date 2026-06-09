#!/usr/bin/env python3
"""Build a Meta-World Very Hard failure audit report from local eval outputs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class RunSpec:
    label: str
    short: str
    root: Path
    note: str


TASK_LINE_RE = re.compile(
    r"Building vec env \| group=(?P<group>[^ ]+) \| task_id=(?P<task_id>\d+) \| task=(?P<task>[^ ]+)"
)


def repo_rel(path: Path) -> str:
    return str(path).replace(str(Path.cwd()) + "/", "")


def load_task_names(root: Path) -> dict[tuple[str, int], str]:
    log_path = root / "eval.log"
    names: dict[tuple[str, int], str] = {}
    if not log_path.exists():
        return names
    for line in log_path.read_text(errors="replace").splitlines():
        match = TASK_LINE_RE.search(line)
        if not match:
            continue
        names[(match.group("group"), int(match.group("task_id")))] = match.group("task")
    return names


def load_eval(root: Path) -> dict:
    path = root / "eval" / "eval_info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def local_video_path(root: Path, task_id: int, episode: int) -> Path:
    return root / "eval" / "videos" / f"very_hard_{task_id}" / f"eval_episode_{episode}.mp4"


def row_for_task(run: RunSpec, task: dict, task_name: str) -> dict:
    metrics = task["metrics"]
    successes = list(metrics["successes"])
    sum_rewards = list(metrics["sum_rewards"])
    max_rewards = list(metrics["max_rewards"])
    success_count = sum(bool(v) for v in successes)
    failed = [idx for idx, ok in enumerate(successes) if not ok]
    succeeded = [idx for idx, ok in enumerate(successes) if ok]
    return {
        "run": run.short,
        "task_id": task["task_id"],
        "task_name": task_name,
        "success_count": success_count,
        "n": len(successes),
        "success_pct": 100.0 * success_count / len(successes),
        "failed": failed,
        "succeeded": succeeded,
        "avg_sum_reward": mean(sum_rewards),
        "avg_max_reward": mean(max_rewards),
        "failed_avg_sum_reward": mean([sum_rewards[i] for i in failed]) if failed else None,
        "success_avg_sum_reward": mean([sum_rewards[i] for i in succeeded]) if succeeded else None,
    }


def collect(run: RunSpec) -> tuple[dict, list[dict]]:
    eval_info = load_eval(run.root)
    task_names = load_task_names(run.root)
    rows = []
    for task in eval_info["per_task"]:
        if task["task_group"] != "very_hard":
            continue
        task_id = int(task["task_id"])
        task_name = task_names.get(("very_hard", task_id), f"very_hard_{task_id}")
        rows.append(row_for_task(run, task, task_name))
    rows.sort(key=lambda r: r["task_id"])
    return eval_info, rows


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def fmt_reward(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def build_report(runs: list[RunSpec], out_path: Path) -> None:
    collected = {run.short: collect(run) for run in runs}
    canonical = collected["nas15"][1]
    task_names = {row["task_id"]: row["task_name"] for row in canonical}

    lines: list[str] = []
    lines.append("# Meta-World Very Hard Failure Audit - 2026-06-08")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This audit uses already-downloaded local Meta-World MT50 evaluation artifacts. "
        "No new RunPod evaluation was started."
    )
    lines.append("")
    lines.append("Very Hard task mapping from the LeRobot eval logs:")
    lines.append("")
    lines.append(
        md_table(
            ["Task id", "Task name"],
            [[str(task_id), task_names[task_id]] for task_id in sorted(task_names)],
        )
    )
    lines.append("")
    lines.append("Local result roots:")
    lines.append("")
    for run in runs:
        lines.append(f"- `{run.short}`: `{repo_rel(run.root)}`")
    lines.append("")
    lines.append("Failed-only review pack:")
    lines.append("")
    lines.append("```text")
    lines.append("_workspace/runpod_results/metaworld_very_hard_failed_pack_nas15/")
    lines.append("```")
    lines.append("")
    lines.append("The pack contains failed-video symlinks, a CSV/JSON manifest, and QuickLook thumbnails.")
    lines.append("")

    lines.append("## Split-Level Result")
    lines.append("")
    split_rows = []
    for run in runs:
        eval_info, rows = collected[run.short]
        very = eval_info["per_group"]["very_hard"]
        split_avg = mean([eval_info["per_group"][k]["pc_success"] for k in ["easy", "medium", "hard", "very_hard"]])
        split_rows.append(
            [
                run.label,
                fmt_pct(eval_info["overall"]["pc_success"]),
                fmt_pct(split_avg),
                fmt_pct(very["pc_success"]),
                f"{sum(r['success_count'] for r in rows)}/{sum(r['n'] for r in rows)}",
                run.note,
            ]
        )
    lines.append(
        md_table(
            ["Run", "Weighted overall %", "Paper-style avg %", "Very Hard %", "Very Hard successes", "Note"],
            split_rows,
        )
    )
    lines.append("")
    lines.append(
        "The paper reference reports Very Hard as `60.00%`, but does not give per-task "
        "Very Hard numbers in the table used here. Therefore per-task deltas below are "
        "diagnostic only, not paper-number comparisons."
    )
    lines.append("")

    lines.append("## Very Hard Per-Task Matrix")
    lines.append("")
    matrix_rows = []
    for task_id in sorted(task_names):
        row = [str(task_id), task_names[task_id]]
        for run in runs:
            r = next(item for item in collected[run.short][1] if item["task_id"] == task_id)
            row.append(f"{r['success_count']}/{r['n']} ({fmt_pct(r['success_pct'])}%)")
        matrix_rows.append(row)
    lines.append(
        md_table(
            ["Task id", "Task"] + [run.label for run in runs],
            matrix_rows,
        )
    )
    lines.append("")

    lines.append("## Baseline Failure Detail (`n_action_steps=15`)")
    lines.append("")
    lines.append(
        "`n_action_steps=15` is the current best paper-style split-average baseline, "
        "so this section audits its failed Very Hard episodes."
    )
    lines.append("")
    detail_rows = []
    for r in canonical:
        detail_rows.append(
            [
                str(r["task_id"]),
                r["task_name"],
                f"{r['success_count']}/{r['n']}",
                ", ".join(map(str, r["succeeded"])) or "-",
                ", ".join(map(str, r["failed"])) or "-",
                fmt_reward(r["avg_sum_reward"]),
                fmt_reward(r["success_avg_sum_reward"]),
                fmt_reward(r["failed_avg_sum_reward"]),
            ]
        )
    lines.append(
        md_table(
            [
                "Task id",
                "Task",
                "Successes",
                "Success eps",
                "Failed eps",
                "Avg sum reward",
                "Success avg reward",
                "Failed avg reward",
            ],
            detail_rows,
        )
    )
    lines.append("")

    lines.append("## Failure Type Breakdown (`n_action_steps=15`)")
    lines.append("")
    lines.append(
        "Failure kind is a heuristic based on `sum_reward` and `max_reward`; it is useful "
        "for triage, not a substitute for state/action trace instrumentation."
    )
    lines.append("")
    breakdown: dict[int, dict[str, int]] = {}
    for r in canonical:
        counts = {"zero_progress": 0, "minimal_progress": 0, "partial_shaped_reward": 0, "near_success_or_contact_failure": 0}
        metrics_failed = []
        # Re-open the task metrics to avoid carrying all episode-level details in row_for_task.
        task = next(
            task
            for task in collected["nas15"][0]["per_task"]
            if task["task_group"] == "very_hard" and int(task["task_id"]) == int(r["task_id"])
        )
        for episode, success in enumerate(task["metrics"]["successes"]):
            if success:
                continue
            sum_reward = float(task["metrics"]["sum_rewards"][episode])
            max_reward = float(task["metrics"]["max_rewards"][episode])
            if max_reward <= 0.05 and sum_reward <= 1.0:
                kind = "zero_progress"
            elif max_reward < 0.5:
                kind = "minimal_progress"
            elif max_reward < 3.0:
                kind = "partial_shaped_reward"
            else:
                kind = "near_success_or_contact_failure"
            counts[kind] += 1
            metrics_failed.append((episode, sum_reward, max_reward, kind))
        breakdown[int(r["task_id"])] = counts
    lines.append(
        md_table(
            ["Task", "zero progress", "minimal progress", "partial shaped reward", "near success/contact failure"],
            [
                [
                    canonical_row["task_name"],
                    str(breakdown[int(canonical_row["task_id"])]["zero_progress"]),
                    str(breakdown[int(canonical_row["task_id"])]["minimal_progress"]),
                    str(breakdown[int(canonical_row["task_id"])]["partial_shaped_reward"]),
                    str(breakdown[int(canonical_row["task_id"])]["near_success_or_contact_failure"]),
                ]
                for canonical_row in canonical
            ],
        )
    )
    lines.append("")

    lines.append("## Why This Misses The Paper Number")
    lines.append("")
    lines.append(
        "The paper's Very Hard reference is `60.00%`, which corresponds to `30/50` "
        "successes at this 10-episode-per-task scale. The current paper-style baseline "
        "has `19/50`, so the gap is `11` successes."
    )
    lines.append("")
    lines.append(
        "The missing successes are concentrated in three tasks, not spread evenly. "
        "`stick-push-v3` already reaches `10/10`, so it is not the immediate blocker. "
        "`stick-pull-v3` contributes `9` failures, `shelf-place-v3` contributes `8`, "
        "and `pick-place-wall-v3` contributes `8`."
    )
    lines.append("")
    lines.append(
        "The failure signatures point to at least two different causes. "
        "`shelf-place-v3` and `pick-place-wall-v3` mostly show zero-progress failures, "
        "which suggests approach/grasp/contact is failing before the policy gets into "
        "the final placement phase. `stick-pull-v3` has several high shaped-reward "
        "failures without success, which suggests contact or partial manipulation can "
        "happen but the success condition is not reached reliably."
    )
    lines.append("")
    lines.append(
        "This makes a single global action-horizon fix unlikely to close the gap by "
        "itself. The horizon sweep helped the aggregate, but the per-task matrix shows "
        "different tasks prefer different settings: `stick-push-v3` peaks at `15`, while "
        "`shelf-place-v3` improves under CUDA-pinned `10/20`. The remaining parity gap is "
        "therefore more likely task/protocol/reset/contact specific than simply "
        "`n_action_steps` being wrong."
    )
    lines.append("")

    lines.append("## Failure Video Index (`n_action_steps=15`)")
    lines.append("")
    for r in canonical:
        lines.append(f"### Task {r['task_id']}: `{r['task_name']}`")
        lines.append("")
        if not r["failed"]:
            lines.append("No failed episodes.")
            lines.append("")
            continue
        video_rows = []
        for ep in r["failed"]:
            path = local_video_path(runs[1].root, r["task_id"], ep)
            video_rows.append([str(ep), repo_rel(path), "yes" if path.exists() else "missing"])
        lines.append(md_table(["Episode", "Local video path", "Exists"], video_rows))
        lines.append("")

    lines.append("## Visual Thumbnail Index")
    lines.append("")
    lines.append(
        "These QuickLook thumbnails are lightweight visual aids generated from local "
        "`n_action_steps=15` videos. They are not a metric source; use the JSON tables "
        "above for success/failure labels."
    )
    lines.append("")
    thumb_root = Path("_workspace/runpod_results/metaworld_very_hard_audit_thumbnails")
    thumb_rows = [
        ["shelf-place-v3", "failure", "episode 0", thumb_root / "shelf_fail_ep0.png"],
        ["shelf-place-v3", "success", "episode 3", thumb_root / "shelf_success_ep3.png"],
        ["stick-pull-v3", "failure", "episode 0", thumb_root / "stick_pull_fail_ep0.png"],
        ["stick-pull-v3", "success", "episode 4", thumb_root / "stick_pull_success_ep4.png"],
        ["pick-place-wall-v3", "failure", "episode 0", thumb_root / "pick_wall_fail_ep0.png"],
        ["pick-place-wall-v3", "success", "episode 1", thumb_root / "pick_wall_success_ep1.png"],
    ]
    lines.append(
        md_table(
            ["Task", "Label", "Episode", "Thumbnail path", "Exists"],
            [[task, label, ep, repo_rel(path), "yes" if path.exists() else "missing"] for task, label, ep, path in thumb_rows],
        )
    )
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    lines.append(
        "1. The remaining paper gap is not evenly distributed. Under the current "
        "paper-style baseline (`n_action_steps=15`), `stick-push-v3` is already solved "
        "at `10/10`, while `stick-pull-v3`, `shelf-place-v3`, and "
        "`pick-place-wall-v3` account for most failed episodes."
    )
    lines.append(
        "2. `stick-pull-v3` is the most severe persistent failure. It scores only "
        "`1/10` in the best paper-style baseline and `0/10` in three of the four full "
        "MT50 runs inspected here."
    )
    lines.append(
        "3. `stick-push-v3` is horizon-sensitive rather than persistently weak: it moves "
        "from `1/10` at the default horizon to `10/10` at `n_action_steps=15`, then "
        "drops to `7/10` for CUDA-pinned `10` and `20`."
    )
    lines.append(
        "4. `shelf-place-v3` improves strongly under the CUDA-pinned runs (`5/10`), but "
        "the current paper-style baseline remains `2/10`. This suggests the task is "
        "not impossible, but the best paper-comparison setting is not the best setting "
        "for this task."
    )
    lines.append(
        "5. `pick-place-wall-v3` remains low across horizons (`1/10` to `3/10`). In "
        "`n_action_steps=15`, failed episodes have near-zero average reward, which points "
        "to early approach/contact failure rather than only final placement precision."
    )
    lines.append("")

    lines.append("## Recommended Next Debugging Steps")
    lines.append("")
    lines.append(
        "1. Focus visual inspection on `stick-pull-v3`, `pick-place-wall-v3`, and "
        "`shelf-place-v3` first. These explain the clearest deficit in the current "
        "`n_action_steps=15` baseline."
    )
    lines.append(
        "2. For `stick-pull-v3`, compare the single successful episode `4` against "
        "failed episodes `0`, `1`, and `2`. For `pick-place-wall-v3`, compare successful "
        "episodes `1` and `7` against failed episodes with near-zero reward."
    )
    lines.append(
        "3. Instrument action/state traces for those three tasks only, rather than "
        "rerunning full MT50. The likely next question is whether failures are caused "
        "by grasp/contact, object pose reset distribution, or long-horizon target "
        "approach."
    )
    lines.append(
        "4. Keep `n_action_steps=15` as the paper-style baseline for agentic-wrapper comparison. "
        "Use CUDA `n_action_steps=10` only when weighted overall or runtime matters more than "
        "clean paper-style parity."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/metaworld_very_hard_failure_audit_2026_06_08.md"),
    )
    args = parser.parse_args()

    base = Path("_workspace/runpod_results")
    runs = [
        RunSpec(
            "Default",
            "default",
            base / "metaworld_public_full_mt50_10ep_20260608T0650Z" / "metaworld_public_full_mt50_10ep_20260608T0650Z",
            "public resolver, checkpoint/default action horizon",
        ),
        RunSpec(
            "n_action_steps=15",
            "nas15",
            base / "metaworld_public_full_mt50_10ep_nas15_20260608T1635Z" / "metaworld_public_full_mt50_10ep_nas15_20260608T1635Z",
            "best current paper-style split average",
        ),
        RunSpec(
            "n_action_steps=10 CUDA",
            "nas10",
            base / "metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z" / "metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z",
            "best current weighted overall, torch 2.5.1+cu124 caveat",
        ),
        RunSpec(
            "n_action_steps=20 CUDA",
            "nas20",
            base / "metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z" / "metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z",
            "torch 2.5.1+cu124 caveat",
        ),
    ]
    build_report(runs, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
