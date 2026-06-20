#!/usr/bin/env python3
"""Run Risk1-B alternative-goal full Meta-World MT50 evaluation.

This is the Meta-World analogue of the LIBERO-10 zero-fallback production lane:
generate Qwen candidate prompts per task, gate schema/provenance/fallback status,
then run full policy rollouts using those candidate prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "eval" / "risk1b_metaworld_mt50_manifest.json"
DEFAULT_RENAME_MAP = '{"observation.image":"observation.images.camera1"}'


@dataclass(frozen=True)
class Config:
    manifest: Path
    output_dir: Path
    python_bin: str
    vlm_python_bin: str
    model_id: str
    num_candidates: int
    episodes_per_task: int
    candidate_seeds: str
    commit_steps: int
    policy_path: str
    policy_n_action_steps: int
    policy_empty_cameras: int
    rename_map: str
    renderer_backend: str
    context_root: Path | None
    repair_attempts: int
    fallback_on_validation_error: str
    execute: bool
    stop_on_invalid_qwen: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--vlm-python-bin", default=sys.executable)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--candidate-seeds", default="1201,1202,1203,1204,1205")
    parser.add_argument("--commit-steps", type=int, default=20)
    parser.add_argument("--policy-path", default="lerobot/smolvla_metaworld")
    parser.add_argument("--policy-n-action-steps", type=int, default=20)
    parser.add_argument("--policy-empty-cameras", type=int, default=0)
    parser.add_argument("--rename-map", default=DEFAULT_RENAME_MAP)
    parser.add_argument("--renderer-backend", default="osmesa")
    parser.add_argument(
        "--context-root",
        type=Path,
        default=None,
        help=(
            "Directory containing actual Meta-World context JSON/PNG artifacts. "
            "Required with --execute; expected per-row JSON names include "
            "context_<task-name>_seed<seed>.json or <row_id>/context.json."
        ),
    )
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument(
        "--fallback-on-validation-error",
        choices=("none", "deterministic_locked_fields"),
        default="none",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-stop-on-invalid-qwen", dest="stop_on_invalid_qwen", action="store_false")
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        manifest=args.manifest,
        output_dir=args.output_dir,
        python_bin=args.python_bin,
        vlm_python_bin=args.vlm_python_bin,
        model_id=args.model_id,
        num_candidates=args.num_candidates,
        episodes_per_task=args.episodes_per_task,
        candidate_seeds=args.candidate_seeds,
        commit_steps=args.commit_steps,
        policy_path=args.policy_path,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_empty_cameras=args.policy_empty_cameras,
        rename_map=args.rename_map,
        renderer_backend=args.renderer_backend,
        context_root=args.context_root,
        repair_attempts=args.repair_attempts,
        fallback_on_validation_error=args.fallback_on_validation_error,
        execute=bool(args.execute),
        stop_on_invalid_qwen=bool(args.stop_on_invalid_qwen),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no rows: {path}")
    return payload


def safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def run_command(argv: list[str], *, cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(argv, cwd=cwd, env=env, stdout=out, stderr=err, check=False)
    return int(proc.returncode)


def qwen_json_path(row_dir: Path, model_id: str, task_name: str, seed: int) -> Path:
    model_slug = safe_slug(model_id.split("/")[-1])
    return row_dir / f"risk1b_subgoals_{model_slug}_metaworld_task_{safe_slug(task_name)}_seed{seed}.json"


def resolve_context_artifacts(config: Config, row: dict[str, Any]) -> dict[str, Path | None]:
    if config.context_root is None:
        return {"json": None, "image": None}
    task_name = str(row["task_name"])
    seed = int(row.get("seed", 0))
    row_id = str(row["row_id"])
    candidates = [
        config.context_root / row_id / "context.json",
        config.context_root / row_id / f"context_{safe_slug(task_name)}_seed{seed}.json",
        config.context_root / f"context_{safe_slug(task_name)}_seed{seed}.json",
        config.context_root / f"context_{task_name}_seed{seed}.json",
    ]
    context_json = next((path for path in candidates if path.exists()), None)
    image_candidates: list[Path] = []
    if context_json is not None:
        row_dir = context_json.parent
        image_candidates.extend(
            [
                row_dir / "contact_sheet.png",
                row_dir / f"contact_sheet_{safe_slug(task_name)}_seed{seed}.png",
                row_dir / f"contact_sheet_{task_name}_seed{seed}.png",
            ]
        )
        try:
            payload = json.loads(context_json.read_text(encoding="utf-8"))
            for key in ("contact_sheet", "context_image", "image_path"):
                value = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(value, str):
                    image_candidates.append((context_json.parent / value).resolve() if not Path(value).is_absolute() else Path(value))
        except Exception:
            pass
    context_image = next((path for path in image_candidates if path.exists()), None)
    return {"json": context_json, "image": context_image}


def build_generation_argv(
    config: Config,
    row: dict[str, Any],
    row_dir: Path,
    *,
    context_json: Path | None = None,
    context_image: Path | None = None,
) -> list[str]:
    task_name = str(row["task_name"])
    seed = int(row.get("seed", 0))
    argv = [
        config.vlm_python_bin,
        "-B",
        "scripts/generate_risk1b_vlm_subgoals.py",
        "--backend",
        "transformers",
        "--model-id",
        config.model_id,
        "--suite",
        "metaworld",
        "--task-id",
        str(row.get("task_id", 0)),
        "--seed",
        str(seed),
        "--num-subgoals",
        str(config.num_candidates),
        "--task-description",
        str(row.get("task_description") or f"Complete the Meta-World task {task_name}."),
        "--output-dir",
        str(row_dir),
        "--output-path",
        str(qwen_json_path(row_dir, config.model_id, task_name, seed)),
        "--repair-attempts",
        str(config.repair_attempts),
        "--fallback-on-validation-error",
        config.fallback_on_validation_error,
    ]
    if context_json is not None:
        argv.extend(["--context-json", str(context_json)])
    if context_image is not None:
        argv.extend(["--context-image", str(context_image)])
    argv.append("--json")
    return argv


def load_qwen_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "path": str(path), "missing": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload.get("schema_validation") if isinstance(payload.get("schema_validation"), dict) else {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    fallback = provenance.get("fallback") if isinstance(provenance, dict) else None
    gate = {
        "ok": bool(validation.get("valid") is True and fallback in (None, False)),
        "schema_valid": validation.get("valid"),
        "schema_errors": validation.get("errors") or [],
        "fallback": fallback,
        "provenance": provenance.get("source") or payload.get("candidate_prompt_provenance"),
        "generation_attempt_count": provenance.get("generation_attempt_count", payload.get("generation_attempt_count", 1)),
        "path": str(path),
    }
    return gate


def build_rollout_argv(config: Config, row: dict[str, Any], row_dir: Path, qwen_path: Path) -> list[str]:
    task_name = str(row["task_name"])
    seed = int(row.get("seed", 0))
    return [
        config.python_bin,
        "-B",
        "scripts/run_libero_in_episode_smolvla_instrumented.py",
        "--trace-path",
        str(row_dir / "benchmark_trace.jsonl"),
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
        "--ita-enable",
        "--ita-num-candidates",
        str(config.num_candidates),
        "--ita-candidate-seeds",
        config.candidate_seeds,
        "--ita-commit-steps",
        str(config.commit_steps),
        "--ita-candidate-prompts-json",
        str(qwen_path),
        "--ita-selector-strategy",
        "progress_proxy_or_baseline",
        f"--output_dir={row_dir / 'eval_logs'}",
        f"--policy.path={config.policy_path}",
        "--env.type=metaworld",
        f"--env.task={task_name}",
        f"--eval.n_episodes={config.episodes_per_task}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        f"--policy.empty_cameras={config.policy_empty_cameras}",
        f"--policy.n_action_steps={config.policy_n_action_steps}",
        f"--rename_map={config.rename_map}",
        f"--seed={seed}",
    ]


def extract_success(row_dir: Path) -> dict[str, Any]:
    eval_info = row_dir / "eval_logs" / "eval_info.json"
    if not eval_info.exists():
        return {"status": "missing_eval_info", "eval_info_path": str(eval_info)}
    payload = json.loads(eval_info.read_text(encoding="utf-8"))
    successes: list[bool] = []
    if isinstance(payload.get("overall"), dict):
        overall = payload["overall"]
        if "pc_success" in overall and "n_episodes" in overall:
            n = int(overall.get("n_episodes") or 0)
            pc = float(overall.get("pc_success") or 0.0)
            return {
                "status": "completed",
                "eval_info_path": str(eval_info),
                "n_episodes": n,
                "success_count": round(n * pc / 100.0),
                "pc_success": pc,
            }
    for task in payload.get("per_task", []) if isinstance(payload.get("per_task"), list) else []:
        metrics = task.get("metrics") if isinstance(task, dict) else {}
        values = metrics.get("successes") if isinstance(metrics, dict) else []
        successes.extend(bool(value) for value in values)
    if successes:
        success_count = sum(1 for value in successes if value)
        return {
            "status": "completed",
            "eval_info_path": str(eval_info),
            "n_episodes": len(successes),
            "success_count": success_count,
            "pc_success": 100.0 * success_count / len(successes),
        }
    return {"status": "unparsed_eval_info", "eval_info_path": str(eval_info)}


def aggregate(rows: list[dict[str, Any]], config: Config, manifest: dict[str, Any]) -> dict[str, Any]:
    total_episodes = sum(int(row.get("n_episodes") or 0) for row in rows)
    total_success = sum(int(row.get("success_count") or 0) for row in rows)
    by_group: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = str(row.get("task_group", "unknown"))
        bucket = by_group.setdefault(group, {"rows": 0, "n_episodes": 0, "success_count": 0})
        bucket["rows"] += 1
        bucket["n_episodes"] += int(row.get("n_episodes") or 0)
        bucket["success_count"] += int(row.get("success_count") or 0)
    for bucket in by_group.values():
        n = int(bucket["n_episodes"])
        bucket["pc_success"] = 100.0 * int(bucket["success_count"]) / n if n else None
    fallback_rows = [row["row_id"] for row in rows if row.get("qwen_json_gate", {}).get("fallback") not in (None, False)]
    invalid_rows = [row["row_id"] for row in rows if row.get("qwen_json_gate", {}).get("schema_valid") is not True]
    repaired_rows = [
        row["row_id"]
        for row in rows
        if int(row.get("qwen_json_gate", {}).get("generation_attempt_count") or 1) > 1
    ]
    planned_rows = [row["row_id"] for row in rows if row.get("status") == "planned"]
    failed_rows = [
        row["row_id"]
        for row in rows
        if row.get("status") not in {"completed", "planned"} or row.get("qwen_json_gate", {}).get("ok") is False
    ]
    return {
        "experiment_id": config.output_dir.name,
        "status": "completed" if total_episodes and not fallback_rows and not invalid_rows else "warn_or_blocked",
        "manifest": str(config.manifest),
        "manifest_name": manifest.get("name"),
        "suite": "metaworld_mt50",
        "policy_path": config.policy_path,
        "qwen_model": config.model_id,
        "candidate_policy": "alternative goal-conditioned strategy variants; fallback rows are incomplete evidence",
        "total_success": total_success,
        "total_episodes": total_episodes,
        "pc_success": 100.0 * total_success / total_episodes if total_episodes else None,
        "rows_expected": len(manifest["rows"]),
        "rows_completed": sum(1 for row in rows if row.get("status") == "completed"),
        "qwen_valid_rows": sum(1 for row in rows if row.get("qwen_json_gate", {}).get("schema_valid") is True),
        "qwen_repaired_rows": repaired_rows,
        "fallback_rows": fallback_rows,
        "failed_rows": failed_rows,
        "planned_rows": planned_rows,
        "per_group": by_group,
        "rows": rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    manifest = load_manifest(config.manifest)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "MUJOCO_GL": config.renderer_backend,
        "PYOPENGL_PLATFORM": config.renderer_backend,
    }
    rows_out: list[dict[str, Any]] = []
    plan = {
        "config": {key: str(value) for key, value in config.__dict__.items()},
        "rows": manifest["rows"],
    }
    (config.output_dir / "risk1b_metaworld_mt50_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for index, row in enumerate(manifest["rows"]):
        row_id = str(row["row_id"])
        row_dir = config.output_dir / f"{index:02d}_{row_id}"
        row_dir.mkdir(parents=True, exist_ok=True)
        qwen_path = qwen_json_path(row_dir, config.model_id, str(row["task_name"]), int(row.get("seed", 0)))
        context = resolve_context_artifacts(config, row)
        row_record = {
            "row_id": row_id,
            "task_group": row.get("task_group"),
            "task_name": row.get("task_name"),
            "seed": row.get("seed", 0),
            "row_dir": str(row_dir),
            "context_json": str(context["json"]) if context["json"] else None,
            "context_image": str(context["image"]) if context["image"] else None,
        }
        commands = {
            "generation": build_generation_argv(
                config,
                row,
                row_dir,
                context_json=context["json"],
                context_image=context["image"],
            ),
            "rollout": build_rollout_argv(config, row, row_dir, qwen_path),
        }
        (row_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
        if not config.execute:
            row_record["status"] = "planned"
            rows_out.append(row_record)
            continue
        if context["json"] is None:
            row_record["status"] = "context_missing"
            row_record["blocker"] = "METAWORLD_ACTUAL_CONTEXT_MISSING"
            rows_out.append(row_record)
            if config.stop_on_invalid_qwen:
                break
            continue

        gen_rc = run_command(
            commands["generation"],
            cwd=REPO_ROOT,
            env=env,
            stdout=row_dir / "vlm_generation_stdout.json",
            stderr=row_dir / "vlm_generation_stderr.log",
        )
        row_record["generation_exit_code"] = gen_rc
        gate = load_qwen_gate(qwen_path)
        row_record["qwen_json_gate"] = gate
        (row_dir / "qwen_json_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        if gen_rc != 0 or gate.get("ok") is not True:
            row_record["status"] = "qwen_invalid"
            rows_out.append(row_record)
            if config.stop_on_invalid_qwen:
                break
            continue

        roll_rc = run_command(
            commands["rollout"],
            cwd=REPO_ROOT,
            env=env,
            stdout=row_dir / "full_rollout_stdout.log",
            stderr=row_dir / "full_rollout_stderr.log",
        )
        row_record["rollout_exit_code"] = roll_rc
        row_record.update(extract_success(row_dir))
        if roll_rc != 0 and row_record.get("status") == "completed":
            row_record["status"] = "rollout_nonzero"
        rows_out.append(row_record)
        summary = aggregate(rows_out, config, manifest)
        (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (config.output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row_record, sort_keys=True) + "\n")

    summary = aggregate(rows_out, config, manifest)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
