from __future__ import annotations

import json
import math
import os
import random
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physical_ai_agent.imagine_then_act.interfaces import (
    ActionCandidate,
    BenchmarkResult,
    ExecutionContract,
    ImaginedCandidate,
    JudgedCandidate,
    PostCheckResult,
    RunArtifacts,
    RunConfig,
    RunReport,
    SelectionDecision,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_OUTPUT_ROOT = REPO_ROOT / "_workspace" / "imagine_then_act"
DEFAULT_RUNPOD_OUTPUT_ROOT = "/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results"
RUNPOD_WORKSPACE_ROOT = Path("/workspace/physical-ai/physical_ai_agent")
RUNPOD_BACKEND_OVERRIDE_ENV = "PHYSICAL_AI_ALLOW_RUNPOD_BACKEND"
LIBERO_MODES = {"libero", "runpod-libero"}
CANONICAL_LIBERO_CAMERA_NAME_MAPPING = '{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
CANONICAL_TRIGGER_MODE = "semantic_no_progress"
CANONICAL_INTERVENTION_MODE = "none"
CANONICAL_POLICY_EMPTY_CAMERAS = 0
CANONICAL_POLICY_PATH = "lerobot/smolvla_libero"
CANONICAL_TASK_SUITE = "libero_goal"
CANONICAL_TASK_ID = 6
BASELINE_CANDIDATE_ID = "candidate_00_policy_only"
DEFAULT_SELECTOR_STRATEGY = "baseline_fallback"
DEBUG_MIN_ACTION_NORM_SELECTOR = "debug_min_action_norm"


def parse_candidate_seeds(raw_value: str | None, num_candidates: int, episode_seed: int) -> tuple[int, ...]:
    if raw_value is None:
        return tuple(episode_seed + index for index in range(num_candidates))
    cleaned = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not cleaned:
        raise ValueError("candidate-seeds must include at least one integer")
    seeds = tuple(int(part) for part in cleaned)
    if len(seeds) != num_candidates:
        raise ValueError(
            f"candidate-seeds count ({len(seeds)}) must match num-candidates ({num_candidates})"
        )
    return seeds


def default_env_type(mode: str) -> str:
    if mode in LIBERO_MODES:
        return "libero"
    return "mock"


def default_task_suite(mode: str) -> str:
    if mode in LIBERO_MODES:
        return "libero_goal"
    return "mock_pick_and_place"


def default_output_dir(mode: str, target: str, task_suite: str, task_id: int | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_suffix = f"{task_suite}_task{task_id}" if task_id is not None else task_suite
    return DEFAULT_LOCAL_OUTPUT_ROOT / f"{target}_{mode}_{task_suffix}_{timestamp}"


def build_run_config(args: Any) -> RunConfig:
    env_type = args.env_type or default_env_type(args.mode)
    task_suite = args.task_suite or default_task_suite(args.mode)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        mode=args.mode,
        target=args.target,
        task_suite=task_suite,
        task_id=args.task_id,
    )
    candidate_seeds = parse_candidate_seeds(
        raw_value=args.candidate_seeds,
        num_candidates=args.num_candidates,
        episode_seed=args.episode_seed,
    )
    config = RunConfig(
        mode=args.mode,
        target=args.target,
        policy_path=args.policy_path,
        env_type=env_type,
        task_suite=task_suite,
        task_id=args.task_id,
        num_candidates=args.num_candidates,
        candidate_seeds=candidate_seeds,
        imagination_backend=args.imagination_backend,
        judge_backend=args.judge_backend,
        post_check_backend=args.post_check_backend,
        retry_budget=args.retry_budget,
        output_dir=str(output_dir),
        dry_run=bool(args.dry_run or args.mode in {"smoke", "local-dry-run"}),
        episode_seed=args.episode_seed,
        chunk_steps=args.chunk_steps,
        action_dim=args.action_dim,
        instruction=args.instruction,
        selector_strategy=args.selector_strategy,
    )
    errors = validate_run_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return config


def validate_run_config(config: RunConfig) -> list[str]:
    errors: list[str] = []
    if config.num_candidates <= 0:
        errors.append("num-candidates must be > 0")
    if config.retry_budget < 0:
        errors.append("retry-budget must be >= 0")
    if config.chunk_steps <= 0:
        errors.append("chunk-steps must be > 0")
    if config.action_dim <= 0:
        errors.append("action-dim must be > 0")
    if config.selector_strategy not in {DEFAULT_SELECTOR_STRATEGY, DEBUG_MIN_ACTION_NORM_SELECTOR}:
        errors.append("selector-strategy must be baseline_fallback or debug_min_action_norm")
    if config.mode == "runpod-libero" and config.target != "runpod":
        errors.append("mode runpod-libero requires --target runpod")
    if config.target == "runpod" and config.mode == "smoke":
        errors.append("mode smoke is reserved for local deterministic validation")
    if config.mode == "local-dry-run" and config.target != "local":
        errors.append("mode local-dry-run requires --target local")
    if config.mode in LIBERO_MODES and config.env_type != "libero":
        errors.append("libero modes require --env-type libero")
    if config.mode in LIBERO_MODES and config.task_id is None:
        errors.append("libero modes require --task-id")
    if len(config.candidate_seeds) != config.num_candidates:
        errors.append("candidate seed count must match num-candidates")
    return errors


def prepare_run_artifacts(config: RunConfig) -> RunArtifacts:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        output_dir=str(output_dir),
        config_path=str(output_dir / "config.json"),
        execution_contract_path=str(output_dir / "execution_contract.json"),
        trace_path=str(output_dir / "trace.jsonl"),
        report_path=str(output_dir / "report.json"),
        summary_path=str(output_dir / "summary.md"),
        command_path=str(output_dir / "replay_command.sh"),
        blocker_path=str(output_dir / "blocker.md"),
        benchmark_command_path=str(output_dir / "benchmark_command.sh"),
        benchmark_log_path=str(output_dir / "benchmark.log"),
        benchmark_trace_path=str(output_dir / "benchmark_trace.jsonl"),
        benchmark_eval_info_path=str(output_dir / "eval_logs" / "eval_info.json"),
        benchmark_result_path=str(output_dir / "benchmark_result.json"),
    )


def build_execution_contract(config: RunConfig) -> ExecutionContract:
    output_dir = Path(config.output_dir)
    entrypoint = "scripts/run_imagine_then_act.py"
    python_bin = "python3"
    current_tokens = [
        python_bin,
        "-B",
        entrypoint,
        "--mode",
        config.mode,
        "--target",
        config.target,
        "--policy-path",
        config.policy_path,
        "--env-type",
        config.env_type,
        "--task-suite",
        config.task_suite,
        "--num-candidates",
        str(config.num_candidates),
        "--candidate-seeds",
        ",".join(str(seed) for seed in config.candidate_seeds),
        "--imagination-backend",
        config.imagination_backend,
        "--judge-backend",
        config.judge_backend,
        "--post-check-backend",
        config.post_check_backend,
        "--retry-budget",
        str(config.retry_budget),
        "--output-dir",
        str(output_dir),
        "--episode-seed",
        str(config.episode_seed),
        "--chunk-steps",
        str(config.chunk_steps),
        "--action-dim",
        str(config.action_dim),
        "--instruction",
        config.instruction,
        "--selector-strategy",
        config.selector_strategy,
    ]
    if config.task_id is not None:
        current_tokens.extend(["--task-id", str(config.task_id)])
    if config.dry_run:
        current_tokens.append("--dry-run")

    benchmark_tokens = current_tokens.copy()
    benchmark_command: str | None = None
    backend_command: str | None = None
    environment_exports: dict[str, str] = {}
    remote_output_dir: str | None = None
    notes = [
        "Use one entrypoint for local smoke, local dry-run, and future RunPod benchmark execution.",
        "Pre-execution imagined outcome selection and post-execution progress checks stay separate.",
        "Judge backends are selectors only; final benchmark success must come from the environment.",
    ]
    requires_linux = config.mode in LIBERO_MODES or config.target == "runpod"

    if config.target == "runpod":
        remote_output_dir = (
            f"{DEFAULT_RUNPOD_OUTPUT_ROOT}/imagine_then_act_{config.task_suite}_task"
            f"{config.task_id if config.task_id is not None else 'none'}"
        )
        benchmark_tokens = [
            "/root/physical-ai/envs/lerobot_py312/bin/python",
            "-B",
            entrypoint,
            "--mode",
            "runpod-libero" if config.mode in LIBERO_MODES else config.mode,
            "--target",
            "runpod",
            "--policy-path",
            config.policy_path,
            "--env-type",
            "libero" if config.mode in LIBERO_MODES else config.env_type,
            "--task-suite",
            config.task_suite,
            "--num-candidates",
            str(config.num_candidates),
            "--candidate-seeds",
            ",".join(str(seed) for seed in config.candidate_seeds),
            "--imagination-backend",
            config.imagination_backend,
            "--judge-backend",
            config.judge_backend,
            "--post-check-backend",
            config.post_check_backend,
            "--retry-budget",
            str(config.retry_budget),
            "--output-dir",
            remote_output_dir,
            "--episode-seed",
            str(config.episode_seed),
            "--chunk-steps",
            str(config.chunk_steps),
            "--action-dim",
            str(config.action_dim),
            "--instruction",
            config.instruction,
            "--selector-strategy",
            config.selector_strategy,
        ]
        if config.task_id is not None:
            benchmark_tokens.extend(["--task-id", str(config.task_id)])
        environment_exports = {
            "LIBERO_CONFIG_PATH": "$HOME/.libero",
            "MUJOCO_GL": "egl",
            "HF_HOME": "/workspace/physical-ai/hf_home",
            "TRANSFORMERS_CACHE": "/workspace/physical-ai/hf_home/transformers",
            "HF_HUB_CACHE": "/workspace/physical-ai/hf_home/hub",
        }
        notes.append("RunPod execution should be performed on a committed revision and stopped after fetching results.")
        benchmark_command = shell_command(benchmark_tokens, environment_exports, working_dir="/workspace/physical-ai/physical_ai_agent")
        if config.mode in LIBERO_MODES:
            backend_tokens = build_backend_command_tokens(
                config=config,
                trace_path=f"{remote_output_dir}/benchmark_trace.jsonl",
                eval_logs_dir=f"{remote_output_dir}/eval_logs",
                python_bin="/root/physical-ai/envs/lerobot_py312/bin/python",
                script_path="scripts/run_libero_in_episode_smolvla_instrumented.py",
            )
            backend_command = shell_command(
                backend_tokens,
                environment_exports,
                working_dir="/workspace/physical-ai/physical_ai_agent",
            )
    elif config.mode in LIBERO_MODES:
        notes.append("Local LIBERO parity is tracked as a Linux or RunPod follow-up, not a Mac benchmark claim.")

    return ExecutionContract(
        entrypoint=entrypoint,
        working_dir=str(REPO_ROOT),
        python_bin=python_bin,
        current_command=shell_command(current_tokens, {"PYTHONPATH": "src"}, working_dir=str(REPO_ROOT)),
        benchmark_command=benchmark_command,
        backend_command=backend_command,
        environment_exports=environment_exports,
        local_output_dir=str(output_dir),
        remote_output_dir=remote_output_dir,
        requires_linux=requires_linux,
        notes=notes,
    )


def shell_command(tokens: list[str], env: dict[str, str], working_dir: str) -> str:
    env_prefix = " ".join(f"{key}={value}" for key, value in env.items())
    command = shlex.join(tokens)
    if env_prefix:
        return f"cd {shlex.quote(working_dir)} && {env_prefix} {command}"
    return f"cd {shlex.quote(working_dir)} && {command}"


def write_config_snapshot(config: RunConfig, artifacts: RunArtifacts) -> None:
    write_json(Path(artifacts.config_path), asdict(config))


def write_execution_contract(contract: ExecutionContract, artifacts: RunArtifacts) -> None:
    write_json(Path(artifacts.execution_contract_path), asdict(contract))
    command_lines = ["#!/bin/sh", "set -eu", "", contract.current_command]
    if contract.benchmark_command:
        command_lines.extend(["", "# Future entrypoint benchmark command", contract.benchmark_command])
    if contract.backend_command:
        command_lines.extend(["", "# Canonical backend command invoked by the entrypoint", contract.backend_command])
    Path(artifacts.command_path).write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def build_backend_command_tokens(
    *,
    config: RunConfig,
    trace_path: str,
    eval_logs_dir: str,
    python_bin: str,
    script_path: str | None = None,
    selected_candidate_id: str | None = None,
) -> list[str]:
    runner_path = script_path or str(REPO_ROOT / "scripts" / "run_libero_in_episode_smolvla_instrumented.py")
    tokens = [
        python_bin,
        "-B",
        runner_path,
        "--trace-path",
        trace_path,
        "--trigger-mode",
        CANONICAL_TRIGGER_MODE,
        "--intervention-mode",
        CANONICAL_INTERVENTION_MODE,
        "--semantic-min-step",
        "220",
        "--semantic-window",
        "20",
        "--semantic-progress-threshold",
        "0.002",
        f"--output_dir={eval_logs_dir}",
        f"--policy.path={config.policy_path}",
        "--env.type=libero",
        f"--env.task={config.task_suite}",
        f"--env.task_ids=[{config.task_id}]",
        f"--env.camera_name_mapping={CANONICAL_LIBERO_CAMERA_NAME_MAPPING}",
        "--eval.n_episodes=1",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--env.max_parallel_tasks=1",
        f"--policy.empty_cameras={CANONICAL_POLICY_EMPTY_CAMERAS}",
        f"--seed={config.episode_seed}",
        "--ita-enable",
        "--ita-candidate-seeds",
        ",".join(str(seed) for seed in config.candidate_seeds),
        "--ita-num-candidates",
        str(config.num_candidates),
        "--ita-commit-steps",
        str(config.chunk_steps),
        "--ita-selector-strategy",
        config.selector_strategy,
    ]
    if selected_candidate_id:
        tokens.extend(["--ita-selected-candidate-id", selected_candidate_id])
    return tokens


def build_real_backend_command(
    config: RunConfig,
    artifacts: RunArtifacts,
    selected_candidate_id: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    ensure_canonical_libero_backend_config(config)
    eval_logs_dir = Path(artifacts.benchmark_eval_info_path).parent
    command = build_backend_command_tokens(
        config=config,
        trace_path=artifacts.benchmark_trace_path,
        eval_logs_dir=str(eval_logs_dir),
        python_bin=sys.executable,
        selected_candidate_id=selected_candidate_id,
    )
    env = {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
    }
    if config.target == "runpod":
        env["LIBERO_CONFIG_PATH"] = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
        env["HF_HOME"] = os.environ.get("HF_HOME", "/workspace/physical-ai/hf_home")
        env["TRANSFORMERS_CACHE"] = os.environ.get(
            "TRANSFORMERS_CACHE",
            f"{env['HF_HOME']}/transformers",
        )
        env["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", f"{env['HF_HOME']}/hub")
    elif "LIBERO_CONFIG_PATH" in os.environ:
        env["LIBERO_CONFIG_PATH"] = os.environ["LIBERO_CONFIG_PATH"]
    return command, env


def execute_real_backend(
    config: RunConfig,
    artifacts: RunArtifacts,
) -> BenchmarkResult:
    command, env_updates = build_real_backend_command(config, artifacts)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        env_updates["PYTHONPATH"] = f"{env_updates['PYTHONPATH']}{os.pathsep}{existing_pythonpath}"
    environment.update(env_updates)

    command_text = shell_command(command, env_updates, working_dir=str(REPO_ROOT))
    Path(artifacts.benchmark_command_path).write_text(
        "\n".join(["#!/bin/sh", "set -eu", "", command_text]) + "\n",
        encoding="utf-8",
    )

    log_path = Path(artifacts.benchmark_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return load_benchmark_result(
        config=config,
        artifacts=artifacts,
        command_text=command_text,
        exit_code=completed.returncode,
    )


def should_execute_real_backend(config: RunConfig, blockers: list[str]) -> bool:
    return config.mode in LIBERO_MODES and not config.dry_run and not blockers


def load_benchmark_result(
    *,
    config: RunConfig,
    artifacts: RunArtifacts,
    command_text: str,
    exit_code: int,
) -> BenchmarkResult:
    eval_info_path = Path(artifacts.benchmark_eval_info_path)
    trace_path = Path(artifacts.benchmark_trace_path)
    summary = read_rollout_summary(trace_path)
    action_steps = int(summary.get("action_step_count", 0)) if summary else None
    trace_success = bool(summary.get("success")) if summary and "success" in summary else None
    ita_trace = read_ita_application_summary(trace_path)

    if eval_info_path.exists():
        payload = json.loads(eval_info_path.read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        pc_success = float(overall.get("pc_success", 0.0))
        eval_seconds = float(overall.get("eval_s", 0.0))
        success = trace_success if trace_success is not None else pc_success > 0.0
        rationale = (
            "Real LIBERO backend executed via the instrumented SmolVLA baseline path. "
            "Final success came from benchmark artifacts; selected_candidate_applied is read from rollout trace action-source records."
        )
        result = BenchmarkResult(
            available=True,
            success=bool(success),
            source="eval_info.json",
            rationale=rationale,
            command=command_text,
            log_path=str(Path(artifacts.benchmark_log_path)),
            trace_path=str(trace_path),
            eval_info_path=str(eval_info_path),
            pc_success=pc_success,
            eval_seconds=eval_seconds,
            action_steps=action_steps,
            seed=config.episode_seed,
            exit_code=exit_code,
            selected_candidate_applied=ita_trace["selected_candidate_applied"],
            selected_candidate_id=ita_trace["selected_candidate_id"],
            selected_action_shape=ita_trace["selected_action_shape"],
            committed_action_steps=ita_trace["committed_action_steps"],
            candidate_generation_source=ita_trace["candidate_generation_source"],
            baseline_candidate_available=ita_trace["baseline_candidate_available"],
            baseline_candidate_selected=ita_trace["baseline_candidate_selected"],
            selector_strategy=ita_trace["selector_strategy"],
            selector_confidence=ita_trace["selector_confidence"],
            selector_fallback_used=ita_trace["selector_fallback_used"],
            method_claim_ready=ita_trace["method_claim_ready"],
        )
    else:
        result = BenchmarkResult(
            available=False,
            success=None,
            source="command_failed",
            rationale=(
                "Real LIBERO backend was invoked but did not produce eval_info.json. "
                "Inspect the benchmark log for the environment failure."
            ),
            command=command_text,
            log_path=str(Path(artifacts.benchmark_log_path)),
            trace_path=str(trace_path),
            eval_info_path=str(eval_info_path),
            pc_success=None,
            eval_seconds=None,
            action_steps=action_steps,
            seed=config.episode_seed,
            exit_code=exit_code,
            selected_candidate_applied=ita_trace["selected_candidate_applied"],
            selected_candidate_id=ita_trace["selected_candidate_id"],
            selected_action_shape=ita_trace["selected_action_shape"],
            committed_action_steps=ita_trace["committed_action_steps"],
            candidate_generation_source=ita_trace["candidate_generation_source"],
            baseline_candidate_available=ita_trace["baseline_candidate_available"],
            baseline_candidate_selected=ita_trace["baseline_candidate_selected"],
            selector_strategy=ita_trace["selector_strategy"],
            selector_confidence=ita_trace["selector_confidence"],
            selector_fallback_used=ita_trace["selector_fallback_used"],
            method_claim_ready=ita_trace["method_claim_ready"],
        )

    write_json(Path(artifacts.benchmark_result_path), asdict(result))
    return result


def read_rollout_summary(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "rollout_summary":
            return record
    return {}


def read_ita_application_summary(trace_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selected_candidate_applied": False,
        "selected_candidate_id": None,
        "selected_action_shape": None,
        "committed_action_steps": 0,
        "candidate_generation_source": None,
        "baseline_candidate_available": False,
        "baseline_candidate_selected": False,
        "selector_strategy": None,
        "selector_confidence": None,
        "selector_fallback_used": False,
        "method_claim_ready": False,
    }
    if not trace_path.exists():
        return result
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "rollout_summary":
            result["selected_candidate_applied"] = bool(record.get("selected_candidate_applied", False))
            result["selected_candidate_id"] = record.get("ita_selected_candidate_id")
            result["selected_action_shape"] = record.get("ita_selected_action_shape")
            result["committed_action_steps"] = int(record.get("ita_committed_action_steps", 0) or 0)
            result["candidate_generation_source"] = record.get("ita_candidate_generation_source")
            result["baseline_candidate_available"] = bool(record.get("baseline_candidate_available", False))
            result["baseline_candidate_selected"] = bool(record.get("baseline_candidate_selected", False))
            result["selector_strategy"] = record.get("selector_strategy")
            result["selector_confidence"] = record.get("selector_confidence")
            result["selector_fallback_used"] = bool(record.get("selector_fallback_used", False))
            result["method_claim_ready"] = bool(record.get("method_claim_ready", False))
            continue
        ita = record.get("ita", {}) if isinstance(record, dict) else {}
        if ita.get("selected_candidate_applied") is True or ita.get("action_source") == "ita_selected_candidate":
            result["selected_candidate_applied"] = True
            result["selected_candidate_id"] = ita.get("selected_candidate_id") or result["selected_candidate_id"]
            result["selected_action_shape"] = ita.get("selected_action_shape") or result["selected_action_shape"]
            result["candidate_generation_source"] = (
                ita.get("candidate_generation_source") or result["candidate_generation_source"]
            )
            result["committed_action_steps"] = max(
                result["committed_action_steps"],
                int(ita.get("committed_action_steps_count", 0) or 0),
            )
            result["baseline_candidate_available"] = bool(
                ita.get("baseline_candidate_available", result["baseline_candidate_available"])
            )
            result["baseline_candidate_selected"] = bool(
                ita.get("baseline_candidate_selected", result["baseline_candidate_selected"])
            )
            result["selector_strategy"] = ita.get("selector_strategy") or result["selector_strategy"]
            if ita.get("selector_confidence") is not None:
                result["selector_confidence"] = ita.get("selector_confidence")
            result["selector_fallback_used"] = bool(
                ita.get("selector_fallback_used", result["selector_fallback_used"])
            )
            result["method_claim_ready"] = bool(ita.get("method_claim_ready", result["method_claim_ready"]))
    return result


def ensure_canonical_libero_backend_config(config: RunConfig) -> None:
    mismatches: list[str] = []
    if config.policy_path != CANONICAL_POLICY_PATH:
        mismatches.append(f"policy-path must be {CANONICAL_POLICY_PATH}")
    if config.env_type != "libero":
        mismatches.append("env-type must be libero")
    if config.task_suite != CANONICAL_TASK_SUITE:
        mismatches.append(f"task-suite must be {CANONICAL_TASK_SUITE}")
    if config.task_id != CANONICAL_TASK_ID:
        mismatches.append(f"task-id must be {CANONICAL_TASK_ID}")
    if mismatches:
        raise ValueError("Real LIBERO backend currently supports only the canonical focused baseline: " + "; ".join(mismatches))


def trace_event(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "payload": payload,
    }


def generate_candidate_chunks(config: RunConfig) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    baseline_rng = random.Random(config.episode_seed)
    baseline_chunk = []
    for _step in range(config.chunk_steps):
        baseline_chunk.append([round(baseline_rng.uniform(-1.0, 1.0), 4) for _dim in range(config.action_dim)])
    candidates.append(
        ActionCandidate(
            candidate_id=BASELINE_CANDIDATE_ID,
            seed=config.episode_seed,
            action_chunk=baseline_chunk,
            summary=summarize_action_chunk(baseline_chunk),
            source="policy_only_baseline_placeholder",
            is_baseline=True,
        )
    )
    for index, seed in enumerate(config.candidate_seeds):
        rng = random.Random(seed)
        chunk = []
        for _step in range(config.chunk_steps):
            chunk.append([round(rng.uniform(-1.0, 1.0), 4) for _dim in range(config.action_dim)])
        candidates.append(
            ActionCandidate(
                candidate_id=f"candidate_{index + 1:02d}",
                seed=seed,
                action_chunk=chunk,
                summary=summarize_action_chunk(chunk),
            )
        )
    return candidates


def summarize_action_chunk(chunk: list[list[float]]) -> dict[str, float]:
    flattened = [abs(value) for row in chunk for value in row]
    return {
        "mean_abs_action": round(sum(flattened) / max(1, len(flattened)), 6),
        "first_axis_mean": round(sum(row[0] for row in chunk) / max(1, len(chunk)), 6),
        "last_axis_mean": round(sum(row[-1] for row in chunk) / max(1, len(chunk)), 6),
        "chunk_l2_proxy": round(math.sqrt(sum(value * value for row in chunk for value in row)), 6),
    }


def imagine_candidates(config: RunConfig, candidates: list[ActionCandidate]) -> list[ImaginedCandidate]:
    imagined: list[ImaginedCandidate] = []
    for candidate in candidates:
        mean_abs_action = candidate.summary["mean_abs_action"]
        directional_bias = candidate.summary["first_axis_mean"] - abs(candidate.summary["last_axis_mean"]) * 0.25
        progress = 0.0
        alignment = 0.0
        success_proxy = 0.0
        if config.imagination_backend == "none":
            rationale = "No imagined rollout; preserve candidate metadata only."
        elif config.imagination_backend == "sim-rollout":
            progress = clamp01(0.45 + directional_bias * 0.35)
            alignment = clamp01(0.50 + (0.75 - mean_abs_action) * 0.30)
            success_proxy = clamp01(progress * 0.65 + alignment * 0.35)
            rationale = "Oracle-style placeholder rollout estimated progress from deterministic chunk statistics."
        else:
            progress = clamp01(0.35 + directional_bias * 0.20)
            alignment = clamp01(0.55 + (0.60 - mean_abs_action) * 0.15)
            success_proxy = clamp01(progress * 0.40 + alignment * 0.60)
            rationale = "Learned-placeholder imagination used a deterministic surrogate, not a trained predictor."
        imagined.append(
            ImaginedCandidate(
                candidate_id=candidate.candidate_id,
                backend=config.imagination_backend,
                predicted_progress=round(progress, 6),
                predicted_alignment=round(alignment, 6),
                predicted_success_proxy=round(success_proxy, 6),
                rationale=rationale,
            )
        )
    return imagined


def judge_candidates(
    config: RunConfig,
    candidates: list[ActionCandidate],
    imagined_candidates: list[ImaginedCandidate],
) -> list[JudgedCandidate]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    scored: list[tuple[str, float, str]] = []
    for imagined in imagined_candidates:
        candidate = candidate_by_id[imagined.candidate_id]
        stability_bonus = clamp01(1.0 - candidate.summary["mean_abs_action"])
        if config.judge_backend == "heuristic":
            score = imagined.predicted_progress * 0.55 + imagined.predicted_alignment * 0.35 + stability_bonus * 0.10
            rationale = "Heuristic judge weighted imagined progress, alignment, and chunk stability."
        elif config.judge_backend == "vlm-placeholder":
            score = imagined.predicted_alignment * 0.55 + imagined.predicted_progress * 0.30 + stability_bonus * 0.15
            rationale = "VLM-placeholder judge emphasized subgoal-image alignment but did not produce final success."
        else:
            score = imagined.predicted_success_proxy
            rationale = "Oracle-state placeholder ranked candidates by privileged imagined success proxy."
        scored.append((imagined.candidate_id, round(score, 6), rationale))
    ranked_ids = [candidate_id for candidate_id, _score, _rationale in sorted(scored, key=lambda item: (-item[1], item[0]))]
    judged: list[JudgedCandidate] = []
    for candidate_id, score, rationale in scored:
        judged.append(
            JudgedCandidate(
                candidate_id=candidate_id,
                backend=config.judge_backend,
                score=score,
                rank=ranked_ids.index(candidate_id) + 1,
                rationale=rationale,
            )
        )
    return sorted(judged, key=lambda item: (item.rank, item.candidate_id))


def select_candidate(
    judged_candidates: list[JudgedCandidate],
    *,
    candidates: list[ActionCandidate] | None = None,
    selector_strategy: str = DEFAULT_SELECTOR_STRATEGY,
) -> SelectionDecision:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates or []}
    baseline_available = BASELINE_CANDIDATE_ID in {item.candidate_id for item in judged_candidates}
    if selector_strategy == DEFAULT_SELECTOR_STRATEGY and baseline_available:
        best = next(item for item in judged_candidates if item.candidate_id == BASELINE_CANDIDATE_ID)
        return SelectionDecision(
            candidate_id=best.candidate_id,
            score=best.score,
            rank=best.rank,
            rationale=(
                "Selected policy-only baseline candidate because no non-debug method selector is configured. "
                "This preserves baseline behavior for first acceptance."
            ),
            selector_strategy=selector_strategy,
            confidence=1.0,
            fallback_used=True,
            baseline_candidate_available=True,
            baseline_candidate_selected=True,
            method_claim_ready=False,
        )
    if selector_strategy == DEBUG_MIN_ACTION_NORM_SELECTOR and candidate_by_id:
        selected_candidate = min(
            candidate_by_id.values(),
            key=lambda item: (item.summary.get("mean_abs_action", 0.0), item.candidate_id),
        )
        best = next(item for item in judged_candidates if item.candidate_id == selected_candidate.candidate_id)
        return SelectionDecision(
            candidate_id=best.candidate_id,
            score=selected_candidate.summary.get("mean_abs_action", best.score),
            rank=best.rank,
            rationale=(
                "Debug-only selector chose the minimum mean absolute action norm. "
                "This is not a method selector and must not support performance claims."
            ),
            selector_strategy=selector_strategy,
            confidence=0.0,
            fallback_used=False,
            baseline_candidate_available=baseline_available,
            baseline_candidate_selected=best.candidate_id == BASELINE_CANDIDATE_ID,
            method_claim_ready=False,
        )
    best = min(judged_candidates, key=lambda item: (item.rank, item.candidate_id))
    return SelectionDecision(
        candidate_id=best.candidate_id,
        score=best.score,
        rank=best.rank,
        rationale=f"Selected rank {best.rank} candidate by judge score {best.score:.6f}.",
        selector_strategy=selector_strategy,
        confidence=0.5,
        fallback_used=False,
        baseline_candidate_available=baseline_available,
        baseline_candidate_selected=best.candidate_id == BASELINE_CANDIDATE_ID,
        method_claim_ready=False,
    )


def run_post_check(
    config: RunConfig,
    selected_candidate: SelectionDecision,
    judged_candidates: list[JudgedCandidate],
) -> PostCheckResult:
    if config.post_check_backend == "none":
        return PostCheckResult(
            backend=config.post_check_backend,
            passed=True,
            score=selected_candidate.score,
            rationale="Post-check skipped by configuration.",
        )
    margin = selected_candidate.score - min(item.score for item in judged_candidates)
    if config.post_check_backend == "heuristic":
        passed = selected_candidate.score >= 0.45 and margin >= 0.02
        rationale = "Heuristic post-check verified minimum imagined quality and ranking margin."
    elif config.post_check_backend == "vlm-placeholder":
        passed = selected_candidate.score >= 0.40
        rationale = "VLM-placeholder post-check accepted the selection without claiming benchmark success."
    else:
        passed = selected_candidate.score >= 0.50
        rationale = "Oracle-state placeholder post-check accepted the candidate under a privileged proxy."
    return PostCheckResult(
        backend=config.post_check_backend,
        passed=passed,
        score=round(margin, 6),
        rationale=rationale,
    )


def evaluate_execution_readiness(config: RunConfig) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []
    if config.dry_run:
        notes.append("Dry-run mode validated the selection pipeline contract without benchmark execution.")
        notes.append("Baseline policy-only candidate is included and selected by default unless a debug selector is requested.")
        return "dry_run", blockers, notes
    if config.mode in LIBERO_MODES:
        if config.target == "local":
            blockers.append("Comparable LIBERO execution remains a Linux or RunPod path, not a Mac-local benchmark claim.")
            notes.append("Non-dry-run LIBERO backend is available only for Linux or RunPod execution.")
            return "blocked_local_libero", blockers, notes
        if not is_runpod_execution_context():
            blockers.append(
                "Non-dry-run RunPod execution must run inside the RunPod workspace "
                f"({RUNPOD_WORKSPACE_ROOT}) or set {RUNPOD_BACKEND_OVERRIDE_ENV}=1 for an explicit debug override."
            )
            notes.append("--target runpod shapes the command contract; it does not remotely execute from this Mac.")
            return "blocked_runpod_runtime", blockers, notes
        notes.append("Real backend adapter reuses the instrumented SmolVLA runner and reads final success from benchmark artifacts.")
        notes.append(
            "Canonical focused baseline settings stay fixed at policy lerobot/smolvla_libero, libero_goal task 6, "
            "camera1/camera2 mapping, one episode, batch size 1, async false, max parallel tasks 1, and policy.empty_cameras=0."
        )
        notes.append(
            "ITA candidate chunks are sampled from the real policy action path and the selected action is committed through env.step."
        )
        notes.append(
            "Baseline-preserving selector defaults to candidate_00_policy_only; debug_min_action_norm is not a method selector."
        )
        return "benchmark_backend_ready", blockers, notes
    notes.append("Non-LIBERO local modes are interface-contract runs, not benchmark evaluations.")
    return "contract_only", blockers, notes


def is_runpod_execution_context() -> bool:
    if os.environ.get(RUNPOD_BACKEND_OVERRIDE_ENV) == "1":
        return True
    try:
        return REPO_ROOT.resolve().is_relative_to(RUNPOD_WORKSPACE_ROOT)
    except AttributeError:
        return str(REPO_ROOT.resolve()).startswith(str(RUNPOD_WORKSPACE_ROOT))


def build_run_report(
    config: RunConfig,
    artifacts: RunArtifacts,
    contract: ExecutionContract,
    selected_candidate: SelectionDecision,
    post_check: PostCheckResult,
    trace_events: list[dict[str, Any]],
    blockers: list[str],
    notes: list[str],
    benchmark_result: BenchmarkResult | None = None,
) -> RunReport:
    if benchmark_result is None:
        benchmark_result = BenchmarkResult(
            available=False,
            success=None,
            source="not_run",
            rationale=(
                "Benchmark execution was not run in this invocation; final success must "
                "come from the environment once the real backend is executed."
            ),
            command=None,
            log_path=None,
            trace_path=None,
            eval_info_path=None,
            pc_success=None,
            eval_seconds=None,
            action_steps=None,
            seed=None,
            exit_code=None,
            selected_candidate_applied=False,
            selected_candidate_id=None,
            selected_action_shape=None,
            committed_action_steps=0,
            candidate_generation_source=None,
            baseline_candidate_available=selected_candidate.baseline_candidate_available,
            baseline_candidate_selected=selected_candidate.baseline_candidate_selected,
            selector_strategy=selected_candidate.selector_strategy,
            selector_confidence=selected_candidate.confidence,
            selector_fallback_used=selected_candidate.fallback_used,
            method_claim_ready=False,
        )
    execution_readiness = "blocked" if blockers else "passed"
    if config.dry_run:
        execution_readiness = "dry_run"
    elif benchmark_result.available:
        execution_readiness = "benchmark_executed"
    elif config.mode in LIBERO_MODES and config.target == "runpod":
        execution_readiness = "backend_ready"
    elif not blockers:
        execution_readiness = "contract_only"

    report_status = "blocked" if blockers else "passed"
    if config.dry_run:
        report_status = "passed"
    elif benchmark_result.exit_code not in (None, 0) and not benchmark_result.available:
        report_status = "blocked"

    report_selected_candidate_id = benchmark_result.selected_candidate_id or selected_candidate.candidate_id
    if benchmark_result.candidate_generation_source:
        report_stage_candidate_generation = benchmark_result.candidate_generation_source
    elif config.mode in LIBERO_MODES:
        report_stage_candidate_generation = "backend_policy_generation_pending"
    else:
        report_stage_candidate_generation = "deterministic_seeded_chunk_generator"
    return RunReport(
        status=report_status,
        mode=config.mode,
        target=config.target,
        env_type=config.env_type,
        task_suite=config.task_suite,
        task_id=config.task_id,
        policy_path=config.policy_path,
        dry_run=config.dry_run,
        candidate_count=config.num_candidates + 1,
        selected_candidate_id=report_selected_candidate_id,
        selected_score=selected_candidate.score,
        baseline_candidate_available=benchmark_result.baseline_candidate_available
        or selected_candidate.baseline_candidate_available,
        baseline_candidate_selected=benchmark_result.baseline_candidate_selected
        or selected_candidate.baseline_candidate_selected,
        selector_strategy=benchmark_result.selector_strategy or selected_candidate.selector_strategy,
        selector_confidence=(
            benchmark_result.selector_confidence
            if benchmark_result.selector_confidence is not None
            else selected_candidate.confidence
        ),
        selector_fallback_used=benchmark_result.selector_fallback_used or selected_candidate.fallback_used,
        method_claim_ready=benchmark_result.method_claim_ready and not bool(blockers),
        benchmark_success_available=benchmark_result.available,
        benchmark_success=benchmark_result.success,
        execution_readiness=execution_readiness,
        blockers=blockers,
        notes=notes,
        stage_backends={
            "candidate_generation": report_stage_candidate_generation,
            "imagination": config.imagination_backend,
            "judge": config.judge_backend,
            "post_check": config.post_check_backend,
        },
        artifacts={
            "output_dir": artifacts.output_dir,
            "config": artifacts.config_path,
            "execution_contract": artifacts.execution_contract_path,
            "trace": artifacts.trace_path,
            "report": artifacts.report_path,
            "summary": artifacts.summary_path,
            "command": artifacts.command_path,
            "blocker": artifacts.blocker_path,
            "benchmark_command": artifacts.benchmark_command_path,
            "benchmark_log": artifacts.benchmark_log_path,
            "benchmark_trace": artifacts.benchmark_trace_path,
            "benchmark_eval_info": artifacts.benchmark_eval_info_path,
            "benchmark_result": artifacts.benchmark_result_path,
        },
        current_command=contract.current_command,
        benchmark_command=contract.benchmark_command,
        backend_command=contract.backend_command,
        trace_event_count=len(trace_events),
        post_check_passed=post_check.passed,
        post_check_score=post_check.score,
        post_check_rationale=post_check.rationale,
        benchmark_result=benchmark_result,
    )


def write_run_outputs(
    artifacts: RunArtifacts,
    trace_events: list[dict[str, Any]],
    report: RunReport,
) -> None:
    write_jsonl(Path(artifacts.trace_path), trace_events)
    write_json(Path(artifacts.report_path), asdict(report))
    Path(artifacts.summary_path).write_text(render_markdown(report), encoding="utf-8")
    if report.blockers:
        Path(artifacts.blocker_path).write_text(
            "\n".join(
                [
                    "# Imagine-Then-Act Blockers",
                    "",
                    *[f"- {blocker}" for blocker in report.blockers],
                    "",
                    "Run the deterministic local dry-run first, then hand off RunPod execution to the researcher lane.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def render_markdown(report: RunReport) -> str:
    lines = [
        "# Imagine-Then-Act Chunk Selection",
        "",
        f"- status: `{report.status}`",
        f"- mode: `{report.mode}`",
        f"- target: `{report.target}`",
        f"- env_type: `{report.env_type}`",
        f"- task_suite: `{report.task_suite}`",
        f"- task_id: `{report.task_id}`",
        f"- dry_run: `{str(report.dry_run).lower()}`",
        f"- policy_path: `{report.policy_path}`",
        f"- selected_candidate: `{report.selected_candidate_id}`",
        f"- selected_score: `{report.selected_score}`",
        f"- baseline_candidate_available: `{str(report.baseline_candidate_available).lower()}`",
        f"- baseline_candidate_selected: `{str(report.baseline_candidate_selected).lower()}`",
        f"- selector_strategy: `{report.selector_strategy}`",
        f"- selector_confidence: `{report.selector_confidence}`",
        f"- selector_fallback_used: `{str(report.selector_fallback_used).lower()}`",
        f"- method_claim_ready: `{str(report.method_claim_ready).lower()}`",
        f"- post_check_passed: `{report.post_check_passed}`",
        f"- post_check_score: `{report.post_check_score}`",
        "",
        "## Stage Contract",
        "",
        f"- candidate_generation: `{report.stage_backends['candidate_generation']}`",
        f"- imagination_backend: `{report.stage_backends['imagination']}`",
        f"- judge_backend: `{report.stage_backends['judge']}`",
        f"- post_check_backend: `{report.stage_backends['post_check']}`",
        "",
        "## Claim Boundary",
        "",
        "- Imagined outcome selection happens before execution and is not the final task success signal.",
        f"- Post-check rationale: {report.post_check_rationale}",
        "- Final benchmark success must come from the environment, not from the judge backend.",
        "- First method claims require policy-only baseline parity; selected_candidate_applied alone is not sufficient.",
        f"- Benchmark result source: `{report.benchmark_result.source}`",
        f"- Benchmark result rationale: {report.benchmark_result.rationale}",
        f"- Benchmark success available: `{str(report.benchmark_success_available).lower()}`",
        f"- Benchmark success: `{report.benchmark_success}`",
        f"- Benchmark pc_success: `{report.benchmark_result.pc_success}`",
        f"- Benchmark eval_seconds: `{report.benchmark_result.eval_seconds}`",
        f"- Benchmark action_steps: `{report.benchmark_result.action_steps}`",
        f"- Selected candidate applied to benchmark actions: `{str(report.benchmark_result.selected_candidate_applied).lower()}`",
        f"- Benchmark selected_candidate_id: `{report.benchmark_result.selected_candidate_id}`",
        f"- Benchmark selected_action_shape: `{report.benchmark_result.selected_action_shape}`",
        f"- Benchmark committed_action_steps: `{report.benchmark_result.committed_action_steps}`",
        f"- Benchmark candidate_generation_source: `{report.benchmark_result.candidate_generation_source}`",
        f"- Benchmark baseline_candidate_available: `{str(report.benchmark_result.baseline_candidate_available).lower()}`",
        f"- Benchmark baseline_candidate_selected: `{str(report.benchmark_result.baseline_candidate_selected).lower()}`",
        f"- Benchmark selector_strategy: `{report.benchmark_result.selector_strategy}`",
        f"- Benchmark selector_confidence: `{report.benchmark_result.selector_confidence}`",
        f"- Benchmark selector_fallback_used: `{str(report.benchmark_result.selector_fallback_used).lower()}`",
        f"- Benchmark method_claim_ready: `{str(report.benchmark_result.method_claim_ready).lower()}`",
        "",
        "## Commands",
        "",
        f"- current_command: `{report.current_command}`",
        f"- benchmark_command: `{report.benchmark_command or 'not-generated'}`",
        f"- backend_command: `{report.backend_command or 'not-generated'}`",
        f"- backend_command_artifact: `{report.artifacts['benchmark_command']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report.notes)
    if report.blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
