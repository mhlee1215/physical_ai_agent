#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "eval" / "risk1bc_baseline_difficulty_manifest.json"
RESULTS_JSONL = "results.jsonl"
PLAN_JSON = "risk1bc_difficulty_plan.json"
SUMMARY_JSON = "summary.json"


@dataclass(frozen=True)
class DifficultyEvalConfig:
    manifest: str
    categories: tuple[str, ...]
    output_dir: str
    python_bin: str
    vlm_python_bin: str
    model_id: str
    renderer_backend: str
    context_backend: str
    risk_backend: str
    num_candidates: int
    chunk_steps: int
    action_dim: int
    policy_path: str
    policy_num_steps: int
    policy_n_action_steps: int
    execute: bool
    json_output: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_model_slug(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    return "".join(char.lower() if char.isalnum() else "_" for char in model_name).strip("_")


def parse_categories(raw_value: str) -> tuple[str, ...]:
    categories = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not categories:
        raise ValueError("categories must include at least one value")
    return categories


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty rows list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        if "baseline_category" not in row:
            raise ValueError(f"manifest row {index} missing baseline_category")
        if "seed" not in row:
            raise ValueError(f"manifest row {index} missing seed")
        if "suite" not in row:
            raise ValueError(f"manifest row {index} missing suite")
    return manifest


def selected_rows(manifest: dict[str, Any], categories: tuple[str, ...]) -> list[dict[str, Any]]:
    category_set = set(categories)
    rows = [row for row in manifest["rows"] if row.get("baseline_category") in category_set]
    runnable_rows = [row for row in rows if row.get("task_id") is not None]
    if not runnable_rows:
        raise ValueError("no runnable rows matched; rows with task_id=null are pool pointers only")
    return runnable_rows


def default_output_dir() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(REPO_ROOT / "_workspace" / "runpod_results" / f"risk1bc_difficulty_split_{stamp}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or execute a baseline-difficulty split plan for Risk1-B/C. "
            "The script uses argv lists only; shell quoting is not part of the execution path."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--categories",
        default="baseline_fail_hard",
        help=(
            "Comma-separated baseline categories to include. Default is all baseline_fail_hard rows; "
            "ambiguous/control rows must be requested explicitly."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--vlm-python-bin", default=sys.executable)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--renderer-backend", choices=("egl", "osmesa", "auto"), default="osmesa")
    parser.add_argument("--context-backend", choices=("mock", "libero", "libero-shallow"), default="libero-shallow")
    parser.add_argument("--risk-backend", choices=("mock", "libero-contract", "direct-libero"), default="libero-contract")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--policy-num-steps", type=int, default=None)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Run the generated command chain instead of only writing the plan.")
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace, manifest: dict[str, Any]) -> DifficultyEvalConfig:
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be >= 2")
    if args.chunk_steps <= 0 or args.action_dim <= 0:
        raise ValueError("chunk-steps and action-dim must be > 0")
    policy_path = args.policy_path or manifest.get("policy_path") or "lerobot/smolvla_libero"
    policy_num_steps = args.policy_num_steps if args.policy_num_steps is not None else int(manifest.get("policy_num_steps", 10))
    policy_n_action_steps = (
        args.policy_n_action_steps
        if args.policy_n_action_steps is not None
        else int(manifest.get("policy_n_action_steps", 15))
    )
    return DifficultyEvalConfig(
        manifest=str(Path(args.manifest)),
        categories=parse_categories(args.categories),
        output_dir=args.output_dir or default_output_dir(),
        python_bin=args.python_bin,
        vlm_python_bin=args.vlm_python_bin,
        model_id=args.model_id,
        renderer_backend=args.renderer_backend,
        context_backend=args.context_backend,
        risk_backend=args.risk_backend,
        num_candidates=args.num_candidates,
        chunk_steps=args.chunk_steps,
        action_dim=args.action_dim,
        policy_path=policy_path,
        policy_num_steps=policy_num_steps,
        policy_n_action_steps=policy_n_action_steps,
        execute=bool(args.execute),
        json_output=bool(args.json),
    )


def row_output_dir(config: DifficultyEvalConfig, row: dict[str, Any]) -> Path:
    task_id = int(row["task_id"])
    seed = int(row["seed"])
    category = str(row["baseline_category"])
    return Path(config.output_dir) / category / f"{row['suite']}_task{task_id}_seed{seed}"


def context_paths(row_dir: Path, task_id: int, seed: int) -> tuple[Path, Path]:
    context_dir = row_dir / "risk1b_context"
    return (
        context_dir / f"contact_sheet_task{task_id}_seed{seed}.png",
        context_dir / f"context_task{task_id}_seed{seed}.json",
    )


def subgoals_path(row_dir: Path, model_id: str, suite: str, task_id: int, seed: int) -> Path:
    return row_dir / f"risk1b_subgoals_{safe_model_slug(model_id)}_{suite}_task{task_id}_seed{seed}.json"


def build_context_argv(config: DifficultyEvalConfig, row: dict[str, Any], row_dir: Path) -> list[str]:
    task_id = int(row["task_id"])
    seed = int(row["seed"])
    return [
        config.python_bin,
        "-B",
        "scripts/capture_risk1b_context.py",
        "--backend",
        config.context_backend,
        "--suite",
        str(row["suite"]),
        "--task-id",
        str(task_id),
        "--seed",
        str(seed),
        "--policy-path",
        config.policy_path,
        "--policy-num-steps",
        str(config.policy_num_steps),
        "--policy-n-action-steps",
        str(config.policy_n_action_steps),
        "--renderer-backend",
        config.renderer_backend,
        "--output-dir",
        str(row_dir / "risk1b_context"),
        "--json",
    ]


def build_generation_argv(config: DifficultyEvalConfig, row: dict[str, Any], row_dir: Path) -> list[str]:
    task_id = int(row["task_id"])
    seed = int(row["seed"])
    contact_sheet, context_json = context_paths(row_dir, task_id, seed)
    return [
        config.vlm_python_bin,
        "-B",
        "scripts/generate_risk1b_vlm_subgoals.py",
        "--backend",
        "transformers",
        "--model-id",
        config.model_id,
        "--suite",
        str(row["suite"]),
        "--task-id",
        str(task_id),
        "--seed",
        str(seed),
        "--num-subgoals",
        str(config.num_candidates),
        "--task-description",
        "Complete the LIBERO goal task from the current observation.",
        "--context-image",
        str(contact_sheet),
        "--context-json",
        str(context_json),
        "--output-dir",
        str(row_dir),
        "--json",
    ]


def build_probe_argv(config: DifficultyEvalConfig, row: dict[str, Any], row_dir: Path) -> list[str]:
    task_id = int(row["task_id"])
    seed = int(row["seed"])
    return [
        config.python_bin,
        "-B",
        "scripts/run_imagine_then_act_risk_probes.py",
        "--preset",
        "runpod-libero-smoke",
        "--backend",
        config.risk_backend,
        "--suite",
        str(row["suite"]),
        "--task-ids",
        str(task_id),
        "--seed",
        str(seed),
        "--num-candidates",
        str(config.num_candidates),
        "--chunk-steps",
        str(config.chunk_steps),
        "--action-dim",
        str(config.action_dim),
        "--policy-path",
        config.policy_path,
        "--policy-num-steps",
        str(config.policy_num_steps),
        "--policy-n-action-steps",
        str(config.policy_n_action_steps),
        "--renderer-backend",
        config.renderer_backend,
        "--risk1b-vlm-strategy-variants",
        "--risk1b-generator-backend",
        "json",
        "--risk1b-model",
        config.model_id,
        "--risk1b-candidate-prompts-json",
        str(subgoals_path(row_dir, config.model_id, str(row["suite"]), task_id, seed)),
        "--risk1c-sim-selector",
        "--risk1c-selector-modes",
        "c1",
        "--output-dir",
        str(row_dir / "risk1bc_probe"),
        "--json",
    ]


def build_row_plan(config: DifficultyEvalConfig, row: dict[str, Any]) -> dict[str, Any]:
    row_dir = row_output_dir(config, row)
    commands = {
        "context_capture": build_context_argv(config, row, row_dir),
        "vlm_generation": build_generation_argv(config, row, row_dir),
        "risk1bc_probe": build_probe_argv(config, row, row_dir),
    }
    return {
        "row_id": row.get("row_id"),
        "suite": row.get("suite"),
        "task_id": row.get("task_id"),
        "seed": row.get("seed"),
        "baseline_category": row.get("baseline_category"),
        "baseline_evidence": row.get("evidence", {}),
        "baseline_success": row.get("baseline_success"),
        "baseline_pc_success": row.get("baseline_pc_success"),
        "recommended_use": row.get("recommended_use"),
        "output_dir": str(row_dir),
        "used_shell": False,
        "entrypoints": {
            "context_capture": "scripts/capture_risk1b_context.py",
            "vlm_generation": "scripts/generate_risk1b_vlm_subgoals.py",
            "risk1bc_probe": "scripts/run_imagine_then_act_risk_probes.py",
        },
        "commands": commands,
        "expected_artifacts": {
            "context_json": str(context_paths(row_dir, int(row["task_id"]), int(row["seed"]))[1]),
            "context_contact_sheet": str(context_paths(row_dir, int(row["task_id"]), int(row["seed"]))[0]),
            "risk1b_subgoals_json": str(
                subgoals_path(row_dir, config.model_id, str(row["suite"]), int(row["task_id"]), int(row["seed"]))
            ),
            "risk1bc_summary": str(row_dir / "risk1bc_probe" / "summary.json"),
            "risk1c_selector": str(row_dir / "risk1bc_probe" / "risk1c_sim_selector.json"),
        },
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_command(argv: list[str], cwd: Path, run_dir: Path, label: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"{label}_stdout.json"
    stderr_path = run_dir / f"{label}_stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(argv, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, text=True, check=False)
    return {
        "label": label,
        "exit_code": completed.returncode,
        "elapsed_s": round(time.monotonic() - started, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def nested_find(payload: Any, names: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names:
                return value
        for value in payload.values():
            found = nested_find(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = nested_find(value, names)
            if found is not None:
                return found
    return None


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "path": str(path)}


def extract_risk1b_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    risk1b = summary.get("risk1b") if isinstance(summary.get("risk1b"), dict) else {}
    diversity = summary.get("diversity_metrics") or summary.get("diversity") or risk1b.get("diversity_metrics") or {}
    return {
        "risk1b_verdict": nested_find(summary, {"risk1b_verdict", "risk_1_candidate_diversity"}),
        "risk1b_provenance": nested_find(summary, {"provenance", "candidate_provenance"}),
        "risk1b_diversity_metrics": diversity if isinstance(diversity, dict) else {},
        "risk1b_selected_vs_policy_l2_semantics": (
            "Risk1-B selected_vs_policy_l2 is a diversity/proxy-selected metric and does not reflect "
            "Risk1-C C1 selected candidate. Inspect risk1c_selected_vs_policy_l2 for selector distance."
        ),
    }


def extract_risk1c_metrics(selector: dict[str, Any]) -> dict[str, Any]:
    if not selector:
        return {}
    c1 = select_c1_selector_payload(selector)
    selected_candidate_id = nested_find(c1, {"selected_candidate_id", "selected_id"})
    selected_vs_policy_l2 = nested_find(c1, {"selected_vs_policy_l2"})
    return {
        "selected_candidate_id": selected_candidate_id,
        "selected_vs_policy_l2": selected_vs_policy_l2,
        "risk1c_selected_candidate_id": selected_candidate_id,
        "risk1c_selected_vs_policy_l2": selected_vs_policy_l2,
        "risk1c_mode": nested_find(c1, {"mode"}) or "c1",
        "score_source": nested_find(c1, {"score_source"}),
        "score_spread": nested_find(c1, {"score_spread"}),
        "per_candidate_details": nested_find(c1, {"per_candidate_details", "candidate_scores", "score_details"}),
    }


def select_c1_selector_payload(selector: dict[str, Any]) -> dict[str, Any]:
    c1 = selector.get("c1")
    if isinstance(c1, dict):
        return c1
    modes = selector.get("modes")
    if isinstance(modes, dict):
        c1_mode = modes.get("c1_non_oracle_proxy")
        if isinstance(c1_mode, dict):
            return c1_mode
    return selector


def extract_env_success(row_dir: Path) -> dict[str, Any]:
    candidate_paths = [
        row_dir / "risk1bc_probe" / "libero_eval_logs" / "eval_info.json",
        row_dir / "risk1bc_probe" / "eval_logs" / "eval_info.json",
        row_dir / "risk1bc_probe" / "libero_eval" / "eval_info.json",
    ]
    for path in candidate_paths:
        payload = load_optional_json(path)
        if payload:
            return {
                "env_success_source": str(path),
                "pc_success": nested_find(payload, {"pc_success"}),
                "success_rate": nested_find(payload, {"success_rate"}),
                "success": nested_find(payload, {"success", "is_success"}),
            }
    return {
        "env_success_source": None,
        "pc_success": None,
        "success_rate": None,
        "success": None,
    }


def collect_row_result(plan: dict[str, Any], process_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    row_dir = Path(plan["output_dir"])
    summary = load_optional_json(Path(plan["expected_artifacts"]["risk1bc_summary"]))
    selector = load_optional_json(Path(plan["expected_artifacts"]["risk1c_selector"]))
    return {
        "timestamp": now_iso(),
        "event": "risk1bc_difficulty_row",
        "row_id": plan["row_id"],
        "suite": plan["suite"],
        "task_id": plan["task_id"],
        "seed": plan["seed"],
        "baseline_category": plan["baseline_category"],
        "baseline_evidence": plan["baseline_evidence"],
        "baseline_success": plan["baseline_success"],
        "baseline_pc_success": plan["baseline_pc_success"],
        "output_dir": plan["output_dir"],
        "process_results": process_results or [],
        **extract_risk1b_metrics(summary),
        **extract_risk1c_metrics(selector),
        **extract_env_success(row_dir),
    }


def write_plan(config: DifficultyEvalConfig, plans: list[dict[str, Any]]) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PLAN_JSON
    payload = {
        "timestamp": now_iso(),
        "status": "plan_ready",
        "config": asdict(config),
        "row_count": len(plans),
        "claim_boundary": (
            "Shallow OSMesa Risk1-B/C runs are diagnostic/plumbing evidence only unless separate EGL benchmark "
            "environment success evidence is produced. Baseline categories come from explicit manifest provenance."
        ),
        "plans": plans,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summarize(config: DifficultyEvalConfig, records: list[dict[str, Any]], plan_path: Path) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for record in records:
        category = str(record.get("baseline_category"))
        bucket = by_category.setdefault(category, {"rows": 0, "env_success_observed": 0, "risk1b_actual_rows": 0})
        bucket["rows"] += 1
        pc_success = record.get("pc_success")
        success_rate = record.get("success_rate")
        if (isinstance(pc_success, (int, float)) and pc_success > 0) or (
            isinstance(success_rate, (int, float)) and success_rate > 0
        ) or record.get("success") is True:
            bucket["env_success_observed"] += 1
        provenance = record.get("risk1b_provenance")
        if provenance in {"policy_generated", "actual_policy_generated"}:
            bucket["risk1b_actual_rows"] += 1
    return {
        "timestamp": now_iso(),
        "status": "completed" if config.execute else "dry_run_plan_ready",
        "config": asdict(config),
        "plan_path": str(plan_path),
        "results_jsonl": str(Path(config.output_dir) / RESULTS_JSONL),
        "row_count": len(records),
        "by_baseline_category": by_category,
        "claim_boundary": (
            "Do not judge Risk1-B/C selector usefulness from a single easy task. Easy/control rows and hard/failing "
            "rows must be reported separately; env success remains separate from diversity and selector proxy metrics."
        ),
    }


def run_plan(config: DifficultyEvalConfig, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    results_path = Path(config.output_dir) / RESULTS_JSONL
    for plan in plans:
        process_results: list[dict[str, Any]] = []
        for label in ("context_capture", "vlm_generation", "risk1bc_probe"):
            result = run_command(plan["commands"][label], REPO_ROOT, Path(plan["output_dir"]), label)
            process_results.append(result)
            if result["exit_code"] != 0:
                break
        record = collect_row_result(plan, process_results)
        records.append(record)
        append_jsonl(results_path, record)
    return records


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        manifest = load_manifest(Path(args.manifest))
        config = build_config(args, manifest)
        rows = selected_rows(manifest, config.categories)
        plans = [build_row_plan(config, row) for row in rows]
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    plan_path = write_plan(config, plans)
    if config.execute:
        records = run_plan(config, plans)
    else:
        records = [collect_row_result(plan) for plan in plans]
        results_path = Path(config.output_dir) / RESULTS_JSONL
        for record in records:
            append_jsonl(results_path, record)
    summary = summarize(config, records, plan_path)
    summary_path = Path(config.output_dir) / SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"status={summary['status']}")
        print(f"output_dir={config.output_dir}")
        print(f"plan={plan_path}")
        print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
