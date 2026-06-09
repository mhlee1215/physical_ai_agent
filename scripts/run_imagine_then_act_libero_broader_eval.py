#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = "lerobot/smolvla_libero"
DEFAULT_SUITE = "libero_goal"
DEFAULT_TASK_IDS = "0-9"
DEFAULT_SEED = 1201
DEFAULT_POLICY_NUM_STEPS = 10
DEFAULT_POLICY_N_ACTION_STEPS = 15
DEFAULT_METHODS = ("policy_only", "ita_baseline_fallback")
RESULTS_JSONL = "results.jsonl"
SUMMARY_JSON = "summary.json"
MONITOR_JSONL = "monitor.jsonl"


@dataclass(frozen=True)
class EvalConfig:
    suite: str
    task_ids: tuple[int, ...]
    seed: int
    methods: tuple[str, ...]
    target: str
    output_dir: str
    python_bin: str
    policy_path: str
    policy_num_steps: int
    policy_n_action_steps: int
    num_candidates: int
    chunk_steps: int
    dry_run: bool
    monitor_interval: float
    early_stop_zero_at_half: bool


def parse_task_ids(raw_value: str) -> tuple[int, ...]:
    task_ids: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid descending task range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    if not task_ids:
        raise ValueError("task ids must include at least one id")
    seen: set[int] = set()
    deduped = []
    for task_id in task_ids:
        if task_id not in seen:
            seen.add(task_id)
            deduped.append(task_id)
    return tuple(deduped)


def parse_methods(raw_value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not methods:
        raise ValueError("methods must include at least one method")
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unsupported methods: {', '.join(unknown)}")
    return methods


def default_output_dir(suite: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(REPO_ROOT / "_workspace" / "runpod_results" / f"ita_broader_{suite}_seed{seed}_{timestamp}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable argv-based RunPod/local orchestration for LIBERO broader policy-only + ITA baseline fallback eval."
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-ids", default=DEFAULT_TASK_IDS, help="Comma/range task ids, for example 0-9 or 0,2,4.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--target", choices=("local", "runpod"), default="runpod")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-num-steps", type=int, default=DEFAULT_POLICY_NUM_STEPS)
    parser.add_argument("--policy-n-action-steps", type=int, default=DEFAULT_POLICY_N_ACTION_STEPS)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--chunk-steps", type=int, default=DEFAULT_POLICY_N_ACTION_STEPS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitor-interval", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-zero-at-half",
        action="store_true",
        help="Stop the current broader evaluation if half the planned tasks finish with zero environment success.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> EvalConfig:
    task_ids = parse_task_ids(args.task_ids)
    methods = parse_methods(args.methods)
    output_dir = args.output_dir or default_output_dir(args.suite, args.seed)
    if args.policy_num_steps <= 0:
        raise ValueError("policy-num-steps must be > 0")
    if args.policy_n_action_steps <= 0:
        raise ValueError("policy-n-action-steps must be > 0")
    if args.num_candidates <= 0:
        raise ValueError("num-candidates must be > 0")
    if args.chunk_steps <= 0:
        raise ValueError("chunk-steps must be > 0")
    if args.monitor_interval < 0:
        raise ValueError("monitor-interval must be >= 0")
    return EvalConfig(
        suite=args.suite,
        task_ids=task_ids,
        seed=args.seed,
        methods=methods,
        target=args.target,
        output_dir=output_dir,
        python_bin=args.python_bin,
        policy_path=args.policy_path,
        policy_num_steps=args.policy_num_steps,
        policy_n_action_steps=args.policy_n_action_steps,
        num_candidates=args.num_candidates,
        chunk_steps=args.chunk_steps,
        dry_run=bool(args.dry_run),
        monitor_interval=args.monitor_interval,
        early_stop_zero_at_half=bool(args.early_stop_zero_at_half),
    )


def candidate_seed_csv(seed: int, num_candidates: int) -> str:
    return ",".join(str(seed + index) for index in range(num_candidates))


def build_entrypoint_argv(config: EvalConfig, method: str, task_id: int, run_output_dir: Path) -> list[str]:
    mode = "runpod-libero" if config.target == "runpod" else "libero"
    argv = [
        config.python_bin,
        "-B",
        "scripts/run_imagine_then_act.py",
        "--mode",
        mode,
        "--target",
        config.target,
        "--eval-method",
        method,
        "--policy-path",
        config.policy_path,
        "--env-type",
        "libero",
        "--task-suite",
        config.suite,
        "--task-id",
        str(task_id),
        "--num-candidates",
        str(config.num_candidates),
        "--candidate-seeds",
        candidate_seed_csv(config.seed, config.num_candidates),
        "--imagination-backend",
        "sim-rollout",
        "--judge-backend",
        "heuristic",
        "--post-check-backend",
        "heuristic",
        "--retry-budget",
        "1",
        "--output-dir",
        str(run_output_dir),
        "--episode-seed",
        str(config.seed),
        "--chunk-steps",
        str(config.chunk_steps),
        "--action-dim",
        "7",
        "--policy-num-steps",
        str(config.policy_num_steps),
        "--policy-n-action-steps",
        str(config.policy_n_action_steps),
        "--selector-strategy",
        "baseline_fallback",
        "--json",
    ]
    if config.dry_run:
        argv.append("--dry-run")
    return argv


def run_output_dir(config: EvalConfig, method: str, task_id: int) -> Path:
    return Path(config.output_dir) / method / f"task_{task_id:02d}_seed_{config.seed}"


def nvidia_smi_snapshot() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or completed.stdout.strip()}
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"utilization_gpu_pct": parts[0], "memory_used_mib": parts[1], "memory_total_mib": parts[2]})
    return {"gpus": rows}


def file_growth_snapshot(run_dir: Path) -> dict[str, int]:
    paths = [
        run_dir / "report.json",
        run_dir / "trace.jsonl",
        run_dir / "benchmark_trace.jsonl",
        run_dir / "benchmark_result.json",
        run_dir / "eval_logs" / "eval_info.json",
    ]
    return {str(path): path.stat().st_size for path in paths if path.exists()}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_argv(argv: list[str], cwd: Path, run_dir: Path, monitor_interval: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "orchestrator_stdout.json"
    stderr_path = run_dir / "orchestrator_stderr.log"
    monitor_path = run_dir / MONITOR_JSONL
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(argv, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, text=True)
        if monitor_interval > 0:
            while process.poll() is None:
                append_jsonl(
                    monitor_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "process_alive": True,
                        "pid": process.pid,
                        "nvidia_smi": nvidia_smi_snapshot(),
                        "file_sizes": file_growth_snapshot(run_dir),
                    },
                )
                time.sleep(monitor_interval)
        exit_code = process.wait()
    elapsed_s = round(time.monotonic() - started, 3)
    append_jsonl(
        monitor_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_alive": False,
            "exit_code": exit_code,
            "file_sizes": file_growth_snapshot(run_dir),
        },
    )
    return {
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "monitor_path": str(monitor_path),
    }


def load_entrypoint_payload(stdout_path: Path) -> dict[str, Any]:
    if not stdout_path.exists():
        return {}
    text = stdout_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": "stdout was not valid JSON", "stdout_tail": text[-1000:]}


def load_report(report_path: str | None, run_dir: Path) -> dict[str, Any]:
    path = Path(report_path) if report_path else run_dir / "report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "path": str(path)}


def result_record(config: EvalConfig, method: str, task_id: int, argv: list[str], run_dir: Path, process_result: dict[str, Any]) -> dict[str, Any]:
    payload = load_entrypoint_payload(Path(process_result["stdout_path"]))
    report = load_report(payload.get("report_path"), run_dir)
    benchmark_result = report.get("benchmark_result", {}) if isinstance(report, dict) else {}
    pc_success = benchmark_result.get("pc_success")
    benchmark_success = benchmark_result.get("success")
    status = "passed" if process_result["exit_code"] == 0 else "failed"
    if report.get("status") == "blocked":
        status = "blocked"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "task_result",
        "status": status,
        "method": method,
        "suite": config.suite,
        "task_id": task_id,
        "seed": config.seed,
        "dry_run": config.dry_run,
        "entrypoint": "scripts/run_imagine_then_act.py",
        "command_argv": argv,
        "used_shell": False,
        "output_dir": str(run_dir),
        "exit_code": process_result["exit_code"],
        "elapsed_s": process_result["elapsed_s"],
        "stdout_path": process_result["stdout_path"],
        "stderr_path": process_result["stderr_path"],
        "monitor_path": process_result["monitor_path"],
        "report_path": payload.get("report_path") or str(run_dir / "report.json"),
        "benchmark_success": benchmark_success,
        "pc_success": pc_success,
        "selected_candidate_applied": benchmark_result.get("selected_candidate_applied", False),
        "eval_method": report.get("eval_method", method) if isinstance(report, dict) else method,
    }


def is_successful_record(record: dict[str, Any]) -> bool:
    pc_success = record.get("pc_success")
    if isinstance(pc_success, (int, float)) and pc_success > 0:
        return True
    return bool(record.get("benchmark_success"))


def summarize_results(records: list[dict[str, Any]], config: EvalConfig, stop_reason: str | None = None) -> dict[str, Any]:
    task_results = [record for record in records if record.get("event") == "task_result"]
    by_method: dict[str, dict[str, Any]] = {}
    for method in config.methods:
        method_records = [record for record in task_results if record.get("method") == method]
        success_count = sum(1 for record in method_records if is_successful_record(record))
        by_method[method] = {
            "planned": len(config.task_ids),
            "completed": len(method_records),
            "success_count": success_count,
            "success_rate": success_count / len(method_records) if method_records else None,
            "statuses": [record.get("status") for record in method_records],
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "bug_suspect_zero_success_at_half" if stop_reason else "completed",
        "stop_reason": stop_reason,
        "config": asdict(config),
        "method_order": list(config.methods),
        "task_ids": list(config.task_ids),
        "result_count": len(task_results),
        "by_method": by_method,
        "results_jsonl": str(Path(config.output_dir) / RESULTS_JSONL),
    }


def should_early_stop(records: list[dict[str, Any]], method: str, planned_count: int) -> bool:
    method_records = [record for record in records if record.get("event") == "task_result" and record.get("method") == method]
    half = max(1, math.ceil(planned_count / 2))
    if len(method_records) < half:
        return False
    return not any(is_successful_record(record) for record in method_records)


def run_eval(config: EvalConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_JSONL
    records: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for method in config.methods:
        for task_id in config.task_ids:
            run_dir = run_output_dir(config, method, task_id)
            argv = build_entrypoint_argv(config, method, task_id, run_dir)
            process_result = run_argv(argv, REPO_ROOT, run_dir, config.monitor_interval)
            record = result_record(config, method, task_id, argv, run_dir, process_result)
            records.append(record)
            append_jsonl(results_path, record)
            if config.early_stop_zero_at_half and should_early_stop(records, method, len(config.task_ids)):
                stop_reason = f"{method}: zero environment success after half of planned tasks"
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "early_stop",
                    "status": "bug_suspect_zero_success_at_half",
                    "method": method,
                    "reason": stop_reason,
                }
                records.append(event)
                append_jsonl(results_path, event)
                break
        if stop_reason:
            break
    summary = summarize_results(records, config, stop_reason=stop_reason)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        config = build_config(build_parser().parse_args(argv))
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    summary = run_eval(config)
    if "--json" in (argv or sys.argv[1:]):
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"status={summary['status']}")
        print(f"output_dir={config.output_dir}")
        print(f"summary={Path(config.output_dir) / SUMMARY_JSON}")
        print(f"results={Path(config.output_dir) / RESULTS_JSONL}")
    return 0 if summary["status"] in {"completed", "bug_suspect_zero_success_at_half"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
