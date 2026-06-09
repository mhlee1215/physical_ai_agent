#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_MAPPING = '{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'
DEFAULT_TASK_IDS = "0-9"
DEFAULT_SEED = 1201
DEFAULT_POLICY_NUM_STEPS = 10
DEFAULT_POLICY_N_ACTION_STEPS = 15
DEFAULT_METHODS = ("policy_only", "ita_baseline_fallback")


@dataclass(frozen=True)
class EvalConfig:
    suite: str
    task_ids: tuple[int, ...]
    seed: int
    episodes_per_task: int
    methods: tuple[str, ...]
    output_dir: str
    python_bin: str
    policy_path: str
    policy_num_steps: int
    policy_n_action_steps: int
    num_candidates: int
    commit_steps: int
    monitor_gpu: bool
    monitor_interval: float
    early_stop_zero_at_half: bool
    dry_run: bool


def parse_task_ids(raw_value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in (item.strip() for item in raw_value.split(",") if item.strip()):
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid task range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("task ids must not be empty")
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


def parse_methods(raw_value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not methods:
        raise ValueError("methods must not be empty")
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    return methods


def preset_defaults(preset: str) -> tuple[str, int]:
    if preset == "smoke":
        return "6", 1
    if preset == "breadth":
        return DEFAULT_TASK_IDS, 1
    if preset == "full":
        return DEFAULT_TASK_IDS, 5
    raise ValueError(f"unknown preset: {preset}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RunPod LIBERO Goal parity evaluator for direct SmolVLA policy-only "
            "and ITA baseline_fallback control conditions."
        )
    )
    parser.add_argument("--preset", choices=("smoke", "breadth", "full"), default="breadth")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default=None, help="Task ids, e.g. 6, 0-9, or 0,2,4.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python-bin", default="/root/physical-ai/envs/lerobot_py312/bin/python")
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--policy-num-steps", type=int, default=DEFAULT_POLICY_NUM_STEPS)
    parser.add_argument("--policy-n-action-steps", type=int, default=DEFAULT_POLICY_N_ACTION_STEPS)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--commit-steps", type=int, default=DEFAULT_POLICY_N_ACTION_STEPS)
    parser.add_argument("--monitor-gpu", action="store_true")
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    parser.add_argument("--early-stop-zero-at-half", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> EvalConfig:
    preset_task_ids, preset_episodes = preset_defaults(args.preset)
    task_ids = parse_task_ids(args.task_ids or preset_task_ids)
    episodes = args.episodes_per_task if args.episodes_per_task is not None else preset_episodes
    if episodes <= 0:
        raise ValueError("episodes-per-task must be > 0")
    if args.monitor_interval <= 0:
        raise ValueError("monitor-interval must be > 0")
    if args.policy_num_steps <= 0 or args.policy_n_action_steps <= 0:
        raise ValueError("policy horizon flags must be > 0")
    if args.num_candidates <= 0 or args.commit_steps <= 0:
        raise ValueError("num-candidates and commit-steps must be > 0")
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = str(REPO_ROOT / "_workspace" / "runpod_results" / f"libero_goal_parity_{args.preset}_{stamp}")
    return EvalConfig(
        suite=args.suite,
        task_ids=task_ids,
        seed=args.seed,
        episodes_per_task=episodes,
        methods=parse_methods(args.methods),
        output_dir=output_dir,
        python_bin=args.python_bin,
        policy_path=args.policy_path,
        policy_num_steps=args.policy_num_steps,
        policy_n_action_steps=args.policy_n_action_steps,
        num_candidates=args.num_candidates,
        commit_steps=args.commit_steps,
        monitor_gpu=bool(args.monitor_gpu),
        monitor_interval=float(args.monitor_interval),
        early_stop_zero_at_half=bool(args.early_stop_zero_at_half),
        dry_run=bool(args.dry_run),
    )


def method_output_dir(config: EvalConfig, method: str) -> Path:
    return Path(config.output_dir) / method


def trace_name(method: str) -> str:
    return "benchmark_trace.jsonl" if method == "ita_baseline_fallback" else "in_episode_trace.jsonl"


def build_runner_argv(config: EvalConfig, method: str, output_dir: Path) -> list[str]:
    argv = [
        config.python_bin,
        "-B",
        "scripts/run_libero_in_episode_smolvla_instrumented.py",
        "--trace-path",
        str(output_dir / trace_name(method)),
        "--trigger-mode",
        "semantic_no_progress",
        "--intervention-mode",
        "none",
        "--semantic-min-step",
        "220",
        "--semantic-window",
        "20",
        "--semantic-progress-threshold",
        "0.002",
        "--output_dir=" + str(output_dir / "eval_logs"),
        f"--policy.path={config.policy_path}",
        "--env.type=libero",
        f"--env.task={config.suite}",
        "--env.task_ids=[" + ",".join(str(task_id) for task_id in config.task_ids) + "]",
        f"--env.camera_name_mapping={DEFAULT_CAMERA_MAPPING}",
        f"--eval.n_episodes={config.episodes_per_task}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--env.max_parallel_tasks=1",
        "--policy.empty_cameras=0",
        f"--seed={config.seed}",
        f"--policy.num_steps={config.policy_num_steps}",
        f"--policy.n_action_steps={config.policy_n_action_steps}",
    ]
    if method == "ita_baseline_fallback":
        candidate_seeds = ",".join(str(config.seed + index) for index in range(config.num_candidates))
        argv.extend(
            [
                "--ita-enable",
                "--ita-candidate-seeds",
                candidate_seeds,
                "--ita-num-candidates",
                str(config.num_candidates),
                "--ita-commit-steps",
                str(config.commit_steps),
                "--ita-selector-strategy",
                "baseline_fallback",
            ]
        )
    return argv


def nvidia_smi_snapshot() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or completed.stdout.strip()}
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"utilization_gpu_pct": parts[0], "memory_used_mib": parts[1], "memory_total_mib": parts[2]})
    return {"gpus": rows}


def trace_success_count(trace_path: Path) -> int:
    if not trace_path.exists():
        return 0
    count = 0
    with trace_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("success") is True:
                count += 1
            if record.get("event") == "rollout_summary" and record.get("success") is True:
                count += 1
    return count


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_method(config: EvalConfig, method: str) -> dict[str, Any]:
    output_dir = method_output_dir(config, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_runner_argv(config, method, output_dir)
    command_path = output_dir / "exact_command.txt"
    command_path.write_text(" ".join(shlex_quote(item) for item in argv) + "\n", encoding="utf-8")
    if config.dry_run:
        return {
            "method": method,
            "status": "dry_run",
            "exit_code": None,
            "output_dir": str(output_dir),
            "command": argv,
        }

    env = os.environ.copy()
    hf_home = env.get("HF_HOME", "/workspace/physical-ai/hf_home")
    env.update(
        {
            "LIBERO_CONFIG_PATH": env.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")),
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "HF_HOME": hf_home,
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE", f"{hf_home}/transformers"),
            "HF_HUB_CACHE": env.get("HF_HUB_CACHE", f"{hf_home}/hub"),
            "PYTHONPATH": env.get("PYTHONPATH", "src"),
        }
    )
    log_path = output_dir / "benchmark.log"
    monitor_path = output_dir / "monitor.jsonl"
    trace_path = output_dir / trace_name(method)
    planned_max_steps = len(config.task_ids) * config.episodes_per_task * 300
    half_steps = max(1, math.ceil(planned_max_steps / 2))
    start = time.monotonic()
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        last_trace_size = -1
        stagnant = 0
        early_stopped = False
        while process.poll() is None:
            trace_size = trace_path.stat().st_size if trace_path.exists() else 0
            trace_lines = line_count(trace_path)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "pid": process.pid,
                "process_alive": True,
                "trace_lines": trace_lines,
                "trace_bytes": trace_size,
                "nvidia_smi": nvidia_smi_snapshot() if config.monitor_gpu else None,
            }
            append_jsonl(monitor_path, payload)
            if config.early_stop_zero_at_half and trace_lines >= half_steps and trace_success_count(trace_path) == 0:
                process.terminate()
                early_stopped = True
                append_jsonl(
                    monitor_path,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "method": method,
                        "event": "early_stop_zero_at_half",
                        "trace_lines": trace_lines,
                        "planned_half_steps": half_steps,
                    },
                )
                break
            if trace_size == last_trace_size:
                stagnant += 1
            else:
                stagnant = 0
                last_trace_size = trace_size
            time.sleep(config.monitor_interval)
        if early_stopped:
            process.wait(timeout=30)
        exit_code = process.wait()
    elapsed_s = round(time.monotonic() - start, 3)
    result = load_eval_result(output_dir)
    status = "early_stopped_zero_at_half" if early_stopped else ("completed" if exit_code == 0 and result else "failed")
    return {
        "method": method,
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "output_dir": str(output_dir),
        "command_path": str(command_path),
        "log_path": str(log_path),
        "monitor_path": str(monitor_path),
        "trace_path": str(trace_path),
        "result": result,
    }


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(errors="ignore") as handle:
        return sum(1 for _ in handle)


def load_eval_result(output_dir: Path) -> dict[str, Any] | None:
    eval_info = output_dir / "eval_logs" / "eval_info.json"
    if not eval_info.exists():
        return None
    data = json.loads(eval_info.read_text(encoding="utf-8"))
    rows = []
    for task in data.get("per_task") or []:
        successes = [bool(item) for item in (task.get("metrics", {}).get("successes") or [])]
        rows.append(
            {
                "task_id": int(task.get("task_id")),
                "successes": successes,
                "success_count": sum(1 for item in successes if item),
                "episode_count": len(successes),
            }
        )
    return {
        "pc_success": data.get("overall", {}).get("pc_success"),
        "rows": sorted(rows, key=lambda item: item["task_id"]),
        "eval_info_path": str(eval_info),
        "video_paths": data.get("overall", {}).get("video_paths", []),
    }


def summarize(records: list[dict[str, Any]], config: EvalConfig) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for record in records:
        result = record.get("result") or {}
        rows = result.get("rows") or []
        success_count = sum(int(row["success_count"]) for row in rows)
        episode_count = sum(int(row["episode_count"]) for row in rows)
        by_method[record["method"]] = {
            "status": record.get("status"),
            "exit_code": record.get("exit_code"),
            "pc_success": result.get("pc_success"),
            "success_count": success_count,
            "episode_count": episode_count,
            "success_rate": success_count / episode_count if episode_count else None,
            "rows": rows,
            "output_dir": record.get("output_dir"),
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "records": records,
        "by_method": by_method,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        config = build_config(build_parser().parse_args(argv))
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = []
    for method in config.methods:
        record = run_method(config, method)
        records.append(record)
        append_jsonl(output_dir / "results.jsonl", record)
        if record["status"] == "early_stopped_zero_at_half":
            break
    summary = summarize(records, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "--json" in (argv or sys.argv[1:]):
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"summary={output_dir / 'summary.json'}")
        for method, payload in summary["by_method"].items():
            print(
                f"{method}: status={payload['status']} success={payload['success_count']}/"
                f"{payload['episode_count']} pc_success={payload['pc_success']}"
            )
    statuses = {record["status"] for record in records}
    return 0 if statuses <= {"completed", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
