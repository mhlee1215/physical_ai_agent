from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_ROBOCASA_TASK = "CloseFridge"
DEFAULT_ROBOCASA_POLICY = "lerobot/smolvla_robocasa"
DEFAULT_ROBOCASA_EPISODES = 20
DEFAULT_ROBOCASA_BENCHMARK_GROUP = "atomic_seen"
ROBOCASA_CAMERA_RENAME_MAP = {
    "observation.images.robot0_agentview_left": "observation.images.camera1",
    "observation.images.robot0_eye_in_hand": "observation.images.camera2",
    "observation.images.robot0_agentview_right": "observation.images.camera3",
}


@dataclass(frozen=True)
class RoboCasaProbeResult:
    status: str
    task: str
    steps_requested: int
    steps_executed: int
    language: str | None
    success: bool | None
    reward_sum: float | None
    blocker: str | None
    trace_path: str
    metrics_path: str
    summary_path: str


@dataclass(frozen=True)
class Checkpoint25Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    duration_s: float
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(
    output_dir: Path,
    task: str = DEFAULT_ROBOCASA_TASK,
    steps: int = 1,
    require_robocasa: bool = False,
    probe_reset_step: bool = False,
    seed: int = 0,
) -> Checkpoint25Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = _probe_robocasa(
        output_dir=output_dir / "robocasa_probe",
        task=task,
        steps=steps,
        probe_reset_step=probe_reset_step,
        seed=seed,
    )
    install_path = output_dir / "robocasa_install_and_eval.md"
    comparison_path = output_dir / "robocasa365_reference_table.md"
    blocker_path = output_dir / "robocasa_blocker.md"
    report_path = output_dir / "checkpoint_report.json"

    _write_install_and_eval_plan(install_path, task=task)
    _write_reference_table(comparison_path)
    _write_robocasa_blocker(blocker_path, probe)

    probe_ok = probe.status == "passed"
    checks = {
        "cp25_robocasa_checkpoint_registered": True,
        "cp25_install_eval_plan_saved": install_path.exists() and install_path.stat().st_size > 0,
        "cp25_reference_table_saved": comparison_path.exists() and comparison_path.stat().st_size > 0,
        "cp25_probe_trace_or_blocker_saved": Path(probe.trace_path).exists()
        or blocker_path.stat().st_size > 0,
        "cp25_probe_metrics_or_blocker_saved": Path(probe.metrics_path).exists()
        or blocker_path.stat().st_size > 0,
    }
    if require_robocasa:
        checks["cp25_require_robocasa_import"] = probe.blocker is None
    if require_robocasa and probe_reset_step:
        checks["cp25_require_reset_step_rollout"] = probe_ok

    metrics = {
        "benchmark_family": "RoboCasa / RoboCasa365",
        "task": task,
        "policy_reference": DEFAULT_ROBOCASA_POLICY,
        "reference_episodes_per_task": DEFAULT_ROBOCASA_EPISODES,
        "reference_benchmark_group": DEFAULT_ROBOCASA_BENCHMARK_GROUP,
        "probe_reset_step": probe_reset_step,
        "probe_status": probe.status,
        "probe_steps_requested": probe.steps_requested,
        "probe_steps_executed": probe.steps_executed,
        "probe_success": probe.success,
        "probe_reward_sum": probe.reward_sum,
        "probe_language": probe.language,
        "probe_blocker": probe.blocker,
        "require_robocasa": require_robocasa,
        "seed": seed,
        "camera_rename_map": ROBOCASA_CAMERA_RENAME_MAP,
    }
    artifacts = {
        "robocasa_trace": probe.trace_path,
        "robocasa_metrics": probe.metrics_path,
        "robocasa_summary": probe.summary_path,
        "robocasa_blocker": str(blocker_path),
        "robocasa_install_and_eval": str(install_path),
        "robocasa365_reference_table": str(comparison_path),
        "checkpoint_report": str(report_path),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = Checkpoint25Report(
        checkpoint="checkpoint_25_robocasa365_eval_probe",
        status=status,
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        duration_s=round(perf_counter() - started_at, 4),
        checks=checks,
        metrics=metrics,
        artifacts=artifacts,
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _probe_robocasa(
    output_dir: Path,
    task: str,
    steps: int,
    probe_reset_step: bool,
    seed: int,
) -> RoboCasaProbeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "episodes.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    for path in (trace_path, metrics_path, summary_path):
        if path.exists():
            path.unlink()

    imports = _probe_imports()
    if imports["robocasa"] is not True:
        blocker = f"robocasa import failed: {imports['robocasa']}"
        return _write_blocked_probe(
            task=task,
            steps=steps,
            blocker=blocker,
            trace_path=trace_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            imports=imports,
        )
    if imports["robosuite"] is not True:
        blocker = f"robosuite import failed: {imports['robosuite']}"
        return _write_blocked_probe(
            task=task,
            steps=steps,
            blocker=blocker,
            trace_path=trace_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            imports=imports,
        )
    if not probe_reset_step:
        return _write_import_only_probe(
            task=task,
            steps=steps,
            trace_path=trace_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            imports=imports,
        )

    try:
        env_utils = importlib.import_module("robocasa.utils.env_utils")
        create_env = getattr(env_utils, "create_env")
        env = create_env(
            env_name=task,
            render_onscreen=False,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - depends on external package/assets.
        blocker = f"RoboCasa create_env failed for {task}: {type(exc).__name__}: {exc}"
        return _write_blocked_probe(
            task=task,
            steps=steps,
            blocker=blocker,
            trace_path=trace_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            imports=imports,
        )

    records: list[dict[str, Any]] = []
    language = None
    reward_sum = 0.0
    success: bool | None = None
    steps_executed = 0
    try:
        obs = env.reset()
        if hasattr(env, "get_ep_meta"):
            meta = env.get_ep_meta()
            if isinstance(meta, dict):
                language = meta.get("lang")
        action_shape = tuple(getattr(env.action_spec[0], "shape", ()))
        import numpy as np  # External only after RoboCasa is present.

        for step_idx in range(max(steps, 0)):
            action = np.zeros(action_shape, dtype=float)
            obs, reward, done, info = env.step(action)
            reward_float = float(reward)
            reward_sum += reward_float
            success = _extract_success(info)
            record = {
                "step": step_idx,
                "task": task,
                "reward": reward_float,
                "done": bool(done),
                "success": success,
                "obs_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
                "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
            }
            records.append(record)
            steps_executed += 1
            if done:
                break
    except Exception as exc:  # pragma: no cover - depends on external package/assets.
        blocker = f"RoboCasa reset/step failed for {task}: {type(exc).__name__}: {exc}"
        return _write_blocked_probe(
            task=task,
            steps=steps,
            blocker=blocker,
            trace_path=trace_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            imports=imports,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    with trace_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    metrics = {
        "status": "passed",
        "task": task,
        "steps_requested": steps,
        "steps_executed": steps_executed,
        "language": language,
        "success": success,
        "reward_sum": reward_sum,
        "imports": imports,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary(summary_path, metrics, blocker=None)
    return RoboCasaProbeResult(
        status="passed",
        task=task,
        steps_requested=steps,
        steps_executed=steps_executed,
        language=language,
        success=success,
        reward_sum=reward_sum,
        blocker=None,
        trace_path=str(trace_path),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
    )


def _probe_imports() -> dict[str, bool | str]:
    results: dict[str, bool | str] = {}
    for module_name in ("robocasa", "robosuite"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            results[module_name] = f"{type(exc).__name__}: {exc}"
        else:
            results[module_name] = True
    return results


def _extract_success(info: Any) -> bool | None:
    if not isinstance(info, dict):
        return None
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return bool(info[key])
    return None


def _write_import_only_probe(
    task: str,
    steps: int,
    trace_path: Path,
    metrics_path: Path,
    summary_path: Path,
    imports: dict[str, bool | str],
) -> RoboCasaProbeResult:
    metrics = {
        "status": "import_only",
        "task": task,
        "steps_requested": steps,
        "steps_executed": 0,
        "language": None,
        "success": None,
        "reward_sum": None,
        "imports": imports,
        "note": "RoboCasa and robosuite imported. Add --probe-reset-step to execute env.reset()/env.step().",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    trace_path.write_text("", encoding="utf-8")
    _write_summary(summary_path, metrics, blocker=None)
    return RoboCasaProbeResult(
        status="import_only",
        task=task,
        steps_requested=steps,
        steps_executed=0,
        language=None,
        success=None,
        reward_sum=None,
        blocker=None,
        trace_path=str(trace_path),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
    )


def _write_blocked_probe(
    task: str,
    steps: int,
    blocker: str,
    trace_path: Path,
    metrics_path: Path,
    summary_path: Path,
    imports: dict[str, bool | str],
) -> RoboCasaProbeResult:
    metrics = {
        "status": "blocked",
        "task": task,
        "steps_requested": steps,
        "steps_executed": 0,
        "language": None,
        "success": None,
        "reward_sum": None,
        "imports": imports,
        "blocker": blocker,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    trace_path.write_text("", encoding="utf-8")
    _write_summary(summary_path, metrics, blocker=blocker)
    return RoboCasaProbeResult(
        status="blocked",
        task=task,
        steps_requested=steps,
        steps_executed=0,
        language=None,
        success=None,
        reward_sum=None,
        blocker=blocker,
        trace_path=str(trace_path),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
    )


def _write_summary(path: Path, metrics: dict[str, Any], blocker: str | None) -> None:
    lines = [
        "# RoboCasa CP25 Probe Summary",
        "",
        f"- status: `{metrics['status']}`",
        f"- task: `{metrics['task']}`",
        f"- steps requested: `{metrics['steps_requested']}`",
        f"- steps executed: `{metrics['steps_executed']}`",
    ]
    if metrics.get("language"):
        lines.append(f"- instruction: `{metrics['language']}`")
    if metrics.get("success") is not None:
        lines.append(f"- success: `{metrics['success']}`")
    if metrics.get("reward_sum") is not None:
        lines.append(f"- reward sum: `{metrics['reward_sum']}`")
    if blocker:
        lines.extend(["", "## Blocker", "", blocker])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_robocasa_blocker(path: Path, probe: RoboCasaProbeResult) -> None:
    if probe.blocker is None:
        path.write_text(
            "\n".join(
                [
                    "# RoboCasa CP25 Blocker",
                    "",
                    "No blocker recorded for the latest CP25 probe.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        "\n".join(
            [
                "# RoboCasa CP25 Blocker",
                "",
                probe.blocker,
                "",
                "Use `robocasa_install_and_eval.md` for the official install and evaluation path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_install_and_eval_plan(path: Path, task: str) -> None:
    rename_map = json.dumps(ROBOCASA_CAMERA_RENAME_MAP, sort_keys=True)
    lines = [
        "# RoboCasa / RoboCasa365 Install And Evaluation",
        "",
        "## Purpose",
        "",
        "CP25 is the non-LIBERO household-manipulation benchmark lane. It starts with",
        "a dependency plus reset/step smoke, then scales to RoboCasa365 policy evaluation.",
        "",
        "## Official Smoke API",
        "",
        "RoboCasa basic usage creates environments through",
        "`robocasa.utils.env_utils.create_env()`, calls `env.reset()`, reads",
        "`env.get_ep_meta()['lang']`, and advances the simulator with",
        "`env.step(action)`.",
        "",
        "## Minimal Repo Gate",
        "",
        "```bash",
        "sh scripts/checkpoint_25.sh --probe-reset-step --require-robocasa --task "
        + task,
        "```",
        "",
        "Without installed dependencies, omit `--require-robocasa` to write blocker",
        "artifacts without failing the checkpoint:",
        "",
        "```bash",
        "sh scripts/checkpoint_25.sh",
        "```",
        "",
        "## RunPod Install Sketch",
        "",
        "```bash",
        "cd /workspace/physical-ai",
        "git clone https://github.com/robocasa/robocasa.git robocasa",
        "git clone https://github.com/ARISE-Initiative/robosuite.git robosuite",
        "PY=/root/physical-ai/envs/lerobot_py312/bin/python",
        "$PY -m pip install -e robocasa --no-deps",
        "$PY -m pip install -e robosuite",
        "$PY -m pip install numpy numba scipy mujoco pygame Pillow opencv-python "
        "pyyaml pynput tqdm termcolor imageio h5py lxml hidapi tianshou gymnasium",
        "$PY -m robocasa.scripts.setup_macros",
        "$PY -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw",
        "```",
        "",
        "## SmolVLA RoboCasa365 Single-Task Evaluation",
        "",
        "```bash",
        "lerobot-eval \\",
        f"  --policy.path={DEFAULT_ROBOCASA_POLICY} \\",
        "  --env.type=robocasa \\",
        f"  --env.task={task} \\",
        "  --eval.batch_size=1 \\",
        f"  --eval.n_episodes={DEFAULT_ROBOCASA_EPISODES} \\",
        "  --eval.use_async_envs=false \\",
        "  --policy.device=cuda \\",
        f"  '--rename_map={rename_map}'",
        "```",
        "",
        "## RoboCasa365 Benchmark-Group Evaluation",
        "",
        "```bash",
        "lerobot-eval \\",
        f"  --policy.path={DEFAULT_ROBOCASA_POLICY} \\",
        "  --env.type=robocasa \\",
        f"  --env.task={DEFAULT_ROBOCASA_BENCHMARK_GROUP} \\",
        "  --eval.batch_size=1 \\",
        f"  --eval.n_episodes={DEFAULT_ROBOCASA_EPISODES} \\",
        "  --eval.use_async_envs=false \\",
        "  --policy.device=cuda \\",
        f"  '--rename_map={rename_map}'",
        "```",
        "",
        "Paper-comparable reporting should use 20 episodes per task and split-level",
        "task success against Atomic-Seen, Composite-Seen, and Composite-Unseen.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reference_table(path: Path) -> None:
    lines = [
        "# RoboCasa365 Reference Table",
        "",
        "Official leaderboard snapshot checked 2026-06-07. RoboCasa365 leaderboard",
        "was updated 2026-05-23 and reports a 50-task multi-task benchmark over",
        "Atomic-Seen, Composite-Seen, and Composite-Unseen splits.",
        "",
        "| Policy | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| RLDX-1 | 33.2 | 63.0% | 27.5% | 5.4% |",
        "| GR00T N1.5 | 23.9 | 50.7% | 14.8% | 2.7% |",
        "| GR00T N1.6 | 21.9 | 51.1% | 9.4% | 1.7% |",
        "| GigaWorld-Policy 0.1 | 20.7 | 44.4% | 11.8% | 2.9% |",
        "| pi0.5 | 16.9 | 39.6% | 7.1% | 1.2% |",
        "| pi0 | 14.8 | 34.6% | 6.1% | 1.1% |",
        "| Diffusion Policy | 6.1 | 15.7% | 0.2% | 1.3% |",
        "| Our SmolVLA RoboCasa365 full benchmark | pending | pending | pending | pending |",
        "",
        "## Current Partial Run",
        "",
        "| Run | Policy | Task | Episodes | Horizon | Success | Eval seconds | Artifact |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        "| RunPod 2026-06-07 | `lerobot/smolvla_robocasa` | `CloseFridge` | 20 | default LeRobot/RoboCasa `1000` | 0/20, 0.0% | 821.637 | `_workspace/runpod_results/robocasa_smolvla_closefridge_20ep_default_horizon_20260607T063345Z/eval_info.json` |",
        "| RunPod 2026-06-07 | `lerobot/smolvla_robocasa` | `CloseFridge`, `OpenCabinet`, `OpenDrawer` | 60 | default LeRobot/RoboCasa `1000` | 0/60, 0.0% | 2381.632 | `_workspace/runpod_results/robocasa_smolvla_3task_20ep_default_horizon_20260607T073804Z/eval_info.json` |",
        "",
        "## Current Scale-Up Blocker",
        "",
        "| Attempt | Intended tasks | Completed before blocker | Blocker | Artifact |",
        "| --- | --- | --- | --- | --- |",
        "| RunPod 2026-06-07 5-task subset | `CloseFridge`, `OpenCabinet`, `OpenDrawer`, `TurnOnMicrowave`, `TurnOffStove` | `CloseFridge`, `OpenCabinet`, `OpenDrawer` | `TurnOnMicrowave` failed with `ValueError: a cannot be empty unless no samples are taken` in `rng.choice(valid_categories)`, consistent with lightweight asset category coverage | `_workspace/runpod_results/robocasa_smolvla_5task_20ep_default_horizon_20260607T065632Z/eval.log` |",
        "",
        "Comparison rule: do not compare CP25 import/reset smoke to this table. The",
        "`CloseFridge` and 3-task subset rows use the LeRobot RoboCasa policy",
        "evaluation path with 20 episodes per task. They are useful evidence that",
        "the SmolVLA/RoboCasa pipeline runs, but they are not the same scale as the",
        "RoboCasa365 50-task leaderboard. Only the full `atomic_seen`,",
        "`composite_seen`, and `composite_unseen` runs should be compared directly",
        "against leaderboard split or overall values.",
        "",
        "Sources:",
        "",
        "- RoboCasa365 leaderboard: https://robocasa.ai/leaderboard.html",
        "- LeRobot RoboCasa365 docs: https://huggingface.co/docs/lerobot/main/robocasa",
        "- RoboCasa basic usage: https://robocasa.ai/docs/use_cases/basic_usage.html",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 25 RoboCasa / RoboCasa365 evaluation probe.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/checkpoints/checkpoint_25_robocasa"),
    )
    parser.add_argument("--task", default=DEFAULT_ROBOCASA_TASK)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--probe-reset-step",
        action="store_true",
        help="Create a RoboCasa env, reset it, and execute zero-action step(s).",
    )
    parser.add_argument(
        "--require-robocasa",
        action="store_true",
        help="Fail if RoboCasa/robosuite import or requested reset-step probe is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = run_checkpoint(
        output_dir=args.output_dir,
        task=args.task,
        steps=args.steps,
        require_robocasa=args.require_robocasa,
        probe_reset_step=args.probe_reset_step,
        seed=args.seed,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
