#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shlex
from pathlib import Path
from typing import Any


CONFIG = Path("configs/so101/training_datasets/qwen_edge_primitives.json")
LOCAL_STANDARD_DOC = Path("docs/so101_local_training_standard.md")
QWEN_MOCK_RESPONSE = Path("configs/agent/qwen3_so101_tool_planner_mock_response.json")
PLAN_NAME = "primitive training with qwen validation v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan one SmolVLA training run over all three SO101 Qwen edge primitive datasets, "
            "then evaluate scenario=pick_up_cube with execution_policy=qwen_edge_chain."
        )
    )
    parser.add_argument("--run-root", type=Path, default=Path("_workspace/so101_qwen_edge_training"))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--runtime-platform", choices=["auto", "macos", "linux"], default="auto")
    parser.add_argument("--base-train-config", default="<base_smolvla_train_config.json>")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--closed-loop-every-epochs", type=int, default=1)
    parser.add_argument("--closed-loop-smoke-episodes", type=int, default=1)
    parser.add_argument("--closed-loop-smoke-steps", type=int, default=90)
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--qwen-model", default="qwen3-vl-8b-instruct-mlx")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-max-steps-per-primitive", type=int, default=90)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_plan(args)
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(CONFIG)
    runtime = _runtime(args.runtime_platform)
    train_device = "mps" if runtime == "macos" else "cuda"
    accelerator = train_device
    eval_device = args.eval_device or train_device
    run_dir = args.run_root / "qwen_edge_primitives"
    checkpoint = run_dir / "model" / "checkpoints" / "last" / "pretrained_model"
    steps_per_epoch = int(config["training"]["steps_per_epoch"])
    save_freq = steps_per_epoch * max(1, int(args.save_every_epochs))
    validation_interval_steps = save_freq
    max_monitored_checkpoints = max(1, (int(args.steps) + save_freq - 1) // save_freq)

    return {
        "operation": "plan_so101_qwen_edge_primitive_training",
        "name": PLAN_NAME,
        "scenario": "pick_up_cube",
        "execution_policy": "qwen_edge_chain",
        "training_policy": "single_smolvla_checkpoint_trained_on_three_primitive_datasets",
        "runtime_platform": runtime,
        "local_training_standard_doc": str(LOCAL_STANDARD_DOC),
        "dataset_config": str(CONFIG),
        "primitive_datasets": [
            source["name"] for source in config["train_dataset"]["hf_merge_sources"]
        ],
        "validation_datasets": [
            source["name"] for source in config["validation_dataset"]["hf_merge_sources"]
        ],
        "notes": [
            f"Canonical name: {PLAN_NAME}.",
            "This is one training run over the three primitive datasets together.",
            "The output is one SmolVLA checkpoint, not three separate primitive checkpoints.",
            "Closed-loop uses Qwen to switch primitive prompts while routing every primitive to the same checkpoint.",
            "Dataset composition is declared through hf_merge_sources; do not run a separate manual pre-merge step.",
            "On local macOS, start training outside the Codex sandbox with runtime_platform=macos so the launcher selects MPS.",
        ],
        "dataset_merge_policy": "launcher_managed_hf_merge_sources_for_train_and_validation",
        "train_command": _train_command(
            python=args.python,
            config_path=CONFIG,
            run_dir=run_dir,
            base_train_config=args.base_train_config,
            steps=args.steps,
            validation_interval_steps=validation_interval_steps,
            save_freq=save_freq,
            closed_loop_every_epochs=args.closed_loop_every_epochs,
            closed_loop_smoke_episodes=args.closed_loop_smoke_episodes,
            closed_loop_smoke_steps=args.closed_loop_smoke_steps,
            max_monitored_checkpoints=max_monitored_checkpoints,
            runtime=runtime,
        ),
        "expected_checkpoint": str(checkpoint),
        "closed_loop_eval": {
            "scenario": "pick_up_cube",
            "execution_policy": "qwen_edge_chain",
            "command": _qwen_eval_command(
                python=args.python,
                checkpoint=checkpoint,
                qwen_base_url=args.qwen_base_url,
                qwen_model=args.qwen_model,
                episodes=args.eval_episodes,
                max_steps_per_primitive=args.eval_max_steps_per_primitive,
                device=eval_device,
                output_dir=args.run_root / "closed_loop_pick_up_cube_qwen_edge_chain",
            ),
        },
    }


def _train_command(
    *,
    python: str,
    config_path: Path,
    run_dir: Path,
    base_train_config: str,
    steps: int,
    validation_interval_steps: int,
    save_freq: int,
    closed_loop_every_epochs: int,
    closed_loop_smoke_episodes: int,
    closed_loop_smoke_steps: int,
    max_monitored_checkpoints: int,
    runtime: str,
) -> str:
    argv = [
        "PYTHONPATH=src",
        python,
        "-B",
        "scripts/start_so101_training.py",
        "start",
        "--dataset-config",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--runtime-platform",
        runtime,
        "--closed-loop-policy",
        "best_or_periodic",
        "--closed-loop-runner",
        "qwen_chain",
        "--closed-loop-every-epochs",
        str(int(closed_loop_every_epochs)),
        "--closed-loop-episodes",
        str(int(closed_loop_smoke_episodes)),
        "--closed-loop-steps",
        str(int(closed_loop_smoke_steps)),
        "--closed-loop-eval-skill-mode",
        "picklift",
        "--closed-loop-task-prompt",
        "pick and lift the green cube",
        "--policy-n-action-steps",
        "15",
        "--policy-num-steps",
        "10",
        "--qwen-response-json",
        str(QWEN_MOCK_RESPONSE),
        "--max-monitored-checkpoints",
        str(int(max_monitored_checkpoints)),
        "--validation-interval-steps",
        str(int(validation_interval_steps)),
        "--",
        f"--config_path={base_train_config}",
        f"--steps={int(steps)}",
        f"--save_freq={int(save_freq)}",
    ]
    if runtime == "macos":
        argv.append("--num_workers=0")
    return _shell(argv)


def _qwen_eval_command(
    *,
    python: str,
    checkpoint: Path,
    qwen_base_url: str,
    qwen_model: str,
    episodes: int,
    max_steps_per_primitive: int,
    device: str,
    output_dir: Path,
) -> str:
    argv = [
        "PYTHONPATH=src",
        python,
        "-B",
        "scripts/run_so101_qwen_closed_loop_eval.py",
        "--task",
        "pick and lift the green cube",
        "--object",
        "green cube",
        "--qwen-base-url",
        qwen_base_url,
        "--qwen-model",
        qwen_model,
        "--policy-path",
        str(checkpoint),
        "--episodes",
        str(int(episodes)),
        "--max-steps-per-primitive",
        str(int(max_steps_per_primitive)),
        "--policy-n-action-steps",
        "15",
        "--policy-num-steps",
        "10",
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--require-pass",
    ]
    return _shell(argv)


def _runtime(value: str) -> str:
    if value != "auto":
        return value
    return "macos" if platform.system().lower() == "darwin" else "linux"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


if __name__ == "__main__":
    main()
