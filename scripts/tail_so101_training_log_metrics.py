#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path
from typing import Any


FIELD_PATTERNS = {
    "step": re.compile(r"\b(?:step|steps)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?[KMG]?)\b", re.IGNORECASE),
    "loss": re.compile(r"\bloss\s*[=:]\s*([0-9.eE+-]+)\b"),
    "grad_norm": re.compile(r"\b(?:grad(?:_norm)?|grdn)\s*[=:]\s*([0-9.eE+-]+)\b"),
    "lr": re.compile(r"\blr\s*[=:]\s*([0-9.eE+-]+)\b"),
    "epoch": re.compile(r"\b(?:epoch|epch)\s*[=:]\s*([0-9.eE+-]+)\b", re.IGNORECASE),
    "samples": re.compile(r"\b(?:samples|smpl)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?[KMG]?)\b", re.IGNORECASE),
    "update_s": re.compile(r"\b(?:update_s|updt_s)\s*[=:]\s*([0-9.eE+-]+)\b"),
    "data_s": re.compile(r"\bdata_s\s*[=:]\s*([0-9.eE+-]+)\b"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail a LeRobot train log into dashboard JSONL metrics.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--start-at-end", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_steps = _read_seen_steps(args.output)
    offset = args.log.stat().st_size if args.start_at_end and args.log.exists() else 0
    while True:
        if not args.log.exists():
            time.sleep(args.poll_s)
            continue
        with args.log.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                row = _parse_line(line, step_offset=args.step_offset)
                if not row:
                    continue
                step = row.get("step")
                if step is None or int(step) in seen_steps:
                    continue
                row["source"] = "live_stdout"
                _append_jsonl(args.output, row)
                seen_steps.add(int(step))
            offset = handle.tell()
        time.sleep(args.poll_s)


def _read_seen_steps(path: Path) -> set[int]:
    steps: set[int] = set()
    if not path.exists():
        return steps
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("step") is not None:
            steps.add(int(row["step"]))
    return steps


def _parse_line(line: str, step_offset: int = 0) -> dict[str, Any] | None:
    row = _parse_mapping(line) or {}
    progress_step = _parse_latest_progress_step(line)
    for field, pattern in FIELD_PATTERNS.items():
        if field in row:
            continue
        match = pattern.search(line)
        if match:
            row[field] = match.group(1)
    if progress_step is not None:
        row["step"] = progress_step + step_offset
    if "step" not in row or "loss" not in row:
        return None
    return _normalize_row(row)


def _parse_latest_progress_step(line: str) -> int | None:
    matches = re.findall(r"\|\s*([0-9]+)\s*/\s*[0-9]+\s*\[", line)
    if not matches:
        return None
    return int(matches[-1])


def _parse_mapping(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    text = line[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ("step", "samples"):
        if row.get(key) is not None:
            normalized[key] = _parse_count(row[key])
    for key in ("loss", "grad_norm", "epoch", "update_s", "data_s"):
        if row.get(key) is not None:
            normalized[key] = round(float(row[key]), 6)
    if row.get("lr") is not None:
        normalized["lr"] = str(row["lr"])
    return normalized


def _parse_count(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    multiplier = 1
    if text[-1:].upper() in {"K", "M", "G"}:
        suffix = text[-1:].upper()
        text = text[:-1]
        multiplier = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[suffix]
    return int(float(text) * multiplier)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
