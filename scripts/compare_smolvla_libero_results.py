#!/usr/bin/env python3
"""Create SmolVLA LIBERO comparison tables from LeRobot eval_info.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTIONX_REFERENCE = {
    "Goal": 91.0,
    "Object": 94.0,
    "Spatial": 93.0,
    "Long": 77.0,
    "Avg": 88.8,
}

HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE = {
    "Goal": 83.0,
    "Object": 91.0,
    "Spatial": 73.0,
    "Long": 43.0,
    "Avg": 72.75,
}

HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE = {
    "Goal": 92.0,
    "Object": 96.0,
    "Spatial": 90.0,
    "Long": 71.0,
    "Avg": 87.25,
}

GROUP_TO_LABEL = {
    "libero_goal": "Goal",
    "libero_object": "Object",
    "libero_spatial": "Spatial",
    "libero_10": "Long",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_info", type=Path, help="Path to eval_info.json")
    parser.add_argument("--run-name", default="Internal run")
    parser.add_argument("--policy", default="")
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_eval_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_row(data: dict[str, Any]) -> dict[str, float]:
    per_group = data.get("per_group", {})
    row: dict[str, float] = {}
    for group, label in GROUP_TO_LABEL.items():
        row[label] = float(per_group.get(group, {}).get("pc_success", float("nan")))
    row["Avg"] = float(data.get("overall", {}).get("pc_success", float("nan")))
    return row


def per_task_counts(data: dict[str, Any]) -> dict[str, dict[int, tuple[int, int]]]:
    counts: dict[str, dict[int, tuple[int, int]]] = {}
    for item in data.get("per_task", []):
        group = item.get("task_group")
        task_id = int(item.get("task_id"))
        successes = item.get("metrics", {}).get("successes", [])
        passed = sum(1 for value in successes if bool(value))
        counts.setdefault(group, {})[task_id] = (passed, len(successes))
    return counts


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def delta_row(result: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: result[key] - reference[key] for key in reference}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(args: argparse.Namespace, data: dict[str, Any]) -> str:
    result = result_row(data)
    actionx_delta = delta_row(result, ACTIONX_REFERENCE)
    hf_repro_delta = delta_row(result, HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE)
    hf_paper_delta = delta_row(result, HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE)
    n_episodes = int(data.get("overall", {}).get("n_episodes", 0))

    headers = ["Source", "Goal", "Object", "Spatial", "Long", "Avg", "Episodes"]
    rows = [
        [
            args.run_name,
            fmt(result["Goal"]),
            fmt(result["Object"]),
            fmt(result["Spatial"]),
            fmt(result["Long"]),
            fmt(result["Avg"]),
            str(n_episodes),
        ],
        [
            "ActionX Table 1, SmolVLA",
            fmt(ACTIONX_REFERENCE["Goal"]),
            fmt(ACTIONX_REFERENCE["Object"]),
            fmt(ACTIONX_REFERENCE["Spatial"]),
            fmt(ACTIONX_REFERENCE["Long"]),
            fmt(ACTIONX_REFERENCE["Avg"]),
            "400",
        ],
        [
            "Delta vs ActionX",
            fmt(actionx_delta["Goal"]),
            fmt(actionx_delta["Object"]),
            fmt(actionx_delta["Spatial"]),
            fmt(actionx_delta["Long"]),
            fmt(actionx_delta["Avg"]),
            "",
        ],
        [
            "HF issue #2354 repro",
            fmt(HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE["Goal"]),
            fmt(HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE["Object"]),
            fmt(HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE["Spatial"]),
            fmt(HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE["Long"]),
            fmt(HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE["Avg"]),
            "400",
        ],
        [
            "Delta vs HF repro",
            fmt(hf_repro_delta["Goal"]),
            fmt(hf_repro_delta["Object"]),
            fmt(hf_repro_delta["Spatial"]),
            fmt(hf_repro_delta["Long"]),
            fmt(hf_repro_delta["Avg"]),
            "",
        ],
        [
            "HF issue #2354 paper",
            fmt(HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE["Goal"]),
            fmt(HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE["Object"]),
            fmt(HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE["Spatial"]),
            fmt(HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE["Long"]),
            fmt(HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE["Avg"]),
            "400",
        ],
        [
            "Delta vs HF paper",
            fmt(hf_paper_delta["Goal"]),
            fmt(hf_paper_delta["Object"]),
            fmt(hf_paper_delta["Spatial"]),
            fmt(hf_paper_delta["Long"]),
            fmt(hf_paper_delta["Avg"]),
            "",
        ],
    ]

    task_counts = per_task_counts(data)
    task_headers = ["Suite"] + [f"Task {idx}" for idx in range(10)]
    task_rows: list[list[str]] = []
    for group in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        group_counts = task_counts.get(group, {})
        row = [GROUP_TO_LABEL[group]]
        for idx in range(10):
            passed, total = group_counts.get(idx, (0, 0))
            row.append(f"{passed}/{total}" if total else "")
        task_rows.append(row)

    seconds = data.get("overall", {}).get("eval_s")
    ep_s = data.get("overall", {}).get("eval_ep_s")
    speed_note = ""
    if seconds is not None and ep_s is not None:
        speed_note = f"- eval_s: `{fmt(float(seconds))}`\n- eval_ep_s: `{fmt(float(ep_s))}`\n"

    policy_line = f"- policy: `{args.policy}`\n" if args.policy else ""
    return "\n".join(
        [
            "# SmolVLA LIBERO Comparison",
            "",
            f"- run: `{args.run_name}`",
            policy_line.rstrip(),
            f"- eval_info: `{args.eval_info}`",
            f"- episodes: `{n_episodes}`",
            speed_note.rstrip(),
            "",
            "## Side-By-Side",
            "",
            markdown_table(headers, rows),
            "",
            "## Per-Task Success Counts",
            "",
            markdown_table(task_headers, task_rows),
            "",
            "## References",
            "",
            "- ActionX Table 1: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full",
            "- LeRobot public checkpoint reproduction issue: https://github.com/huggingface/lerobot/issues/2354",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    data = load_eval_info(args.eval_info)
    result = result_row(data)
    output = {
        "run_name": args.run_name,
        "policy": args.policy,
        "result": result,
        "delta_vs_actionx": delta_row(result, ACTIONX_REFERENCE),
        "delta_vs_hf_repro": delta_row(result, HF_PUBLIC_CHECKPOINT_REPRO_REFERENCE),
        "delta_vs_hf_paper": delta_row(result, HF_PUBLIC_CHECKPOINT_PAPER_REFERENCE),
        "n_episodes": data.get("overall", {}).get("n_episodes"),
        "per_task_counts": {
            group: {str(task): list(count) for task, count in tasks.items()}
            for group, tasks in per_task_counts(data).items()
        },
    }

    md = build_markdown(args, data)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")
    else:
        print(md)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
