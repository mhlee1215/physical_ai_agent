from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physical_ai_agent.evaluation.lerobot_eval import LeRobotEvalConfig


@dataclass(frozen=True)
class AgenticLayerDebugSpec:
    name: str
    benchmark: str
    runnable: bool
    verifier: str
    retry_budget: int
    notes: tuple[str, ...]
    expected_artifacts: tuple[str, ...]


class AgenticLayer(ABC):
    name: str

    @abstractmethod
    def apply(self, config: LeRobotEvalConfig) -> LeRobotEvalConfig:
        """Return the config this layer wants the base evaluator to execute."""

    @abstractmethod
    def debug_spec(self, benchmark: str) -> AgenticLayerDebugSpec:
        """Return a serializable debugging contract for this layer."""


@dataclass(frozen=True)
class BaselineLayer(AgenticLayer):
    name: str = "baseline"

    def apply(self, config: LeRobotEvalConfig) -> LeRobotEvalConfig:
        return config

    def debug_spec(self, benchmark: str) -> AgenticLayerDebugSpec:
        return AgenticLayerDebugSpec(
            name=self.name,
            benchmark=benchmark,
            runnable=True,
            verifier="benchmark_success_flag",
            retry_budget=0,
            notes=("Policy-only SmolVLA evaluation. No planner, verifier, or retry wrapper is applied.",),
            expected_artifacts=(
                "debug_artifacts/eval_manifest.json",
                "debug_artifacts/command_argv.json",
                "debug_artifacts/agentic_layer.json",
                "debug_artifacts/events.jsonl",
                "run_command.sh",
                "lerobot_eval.log",
                "eval_logs/eval_info.json",
            ),
        )


@dataclass(frozen=True)
class EpisodeRetryLayer(AgenticLayer):
    retry_budget: int = 1
    verifier: str = "benchmark_success_flag"
    name: str = "episode_retry"

    def apply(self, config: LeRobotEvalConfig) -> LeRobotEvalConfig:
        return config

    def debug_spec(self, benchmark: str) -> AgenticLayerDebugSpec:
        runnable = benchmark == "libero"
        notes = (
            "First pass runs the same SmolVLA policy-only evaluator.",
            "Retry planning is based on failed benchmark success flags from eval_info.json.",
            "This is an episode-level retry-budget wrapper, not an in-episode controller.",
        )
        if not runnable:
            notes = notes + ("Meta-World retry aggregation is not wired yet; this layer is metadata-only there.",)
        return AgenticLayerDebugSpec(
            name=self.name,
            benchmark=benchmark,
            runnable=runnable,
            verifier=self.verifier,
            retry_budget=self.retry_budget,
            notes=notes,
            expected_artifacts=(
                "debug_artifacts/eval_manifest.json",
                "debug_artifacts/command_argv.json",
                "debug_artifacts/agentic_layer.json",
                "debug_artifacts/events.jsonl",
                "run_command.sh",
                "lerobot_eval.log",
                "eval_logs/eval_info.json",
                "agentic/retry_plan.json",
                "agentic/agentic_retry_metrics.json",
                "agentic/agentic_retry_trace.jsonl",
            ),
        )


def build_agentic_layer(name: str, *, retry_budget: int = 1) -> AgenticLayer:
    normalized = name.replace("-", "_")
    if normalized == "baseline":
        return BaselineLayer()
    if normalized == "episode_retry":
        return EpisodeRetryLayer(retry_budget=retry_budget)
    choices = "baseline, episode_retry"
    raise ValueError(f"unsupported agentic layer: {name}; expected one of {choices}")


def write_debug_artifacts(
    *,
    output_root: Path,
    config: LeRobotEvalConfig,
    layer: AgenticLayer,
    command: str,
) -> dict[str, str]:
    debug_dir = output_root / "debug_artifacts"
    debug_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "benchmark": config.benchmark,
        "policy_path": config.policy_path,
        "env_task": config.env_task,
        "env_task_ids": config.env_task_ids,
        "n_episodes": config.n_episodes,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "mujoco_gl": config.mujoco_gl,
        "agentic_layer": layer.name,
        "output_root": str(output_root),
    }
    artifacts: dict[str, Any] = {
        "eval_manifest": manifest,
        "command_argv": config.build_argv(),
        "agentic_layer": asdict(layer.debug_spec(config.benchmark)),
        "run_command": command,
    }

    paths = {
        "eval_manifest": debug_dir / "eval_manifest.json",
        "command_argv": debug_dir / "command_argv.json",
        "agentic_layer": debug_dir / "agentic_layer.json",
        "events": debug_dir / "events.jsonl",
    }
    for key, value in artifacts.items():
        if key == "run_command":
            continue
        paths[key].write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    event = {
        "event": "eval_command_prepared",
        "benchmark": config.benchmark,
        "agentic_layer": layer.name,
        "output_root": str(output_root),
    }
    paths["events"].write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}
