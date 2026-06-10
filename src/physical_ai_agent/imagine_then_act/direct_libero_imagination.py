from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_ai_agent.imagine_then_act.risk_probes import (
    RiskProbeConfig,
    RiskProbeReport,
    run_risk_probes,
)


@dataclass(frozen=True)
class DirectLiberoProbeConfig:
    suite: str = "libero_goal"
    task_id: int = 6
    seed: int = 1201
    num_candidates: int = 5
    chunk_steps: int = 15
    action_dim: int = 7
    max_steps: int = 15
    output_dir: str = "_workspace/imagine_then_act/direct_libero_probe"
    camera_name: str = "agentview"
    image_width: int = 128
    image_height: int = 128
    backend: str = "direct-libero"


def build_risk_probe_config(config: DirectLiberoProbeConfig) -> RiskProbeConfig:
    return RiskProbeConfig(
        preset="runpod-libero-double-sim-smoke" if config.backend != "mock" else "local-dry-run",
        backend=config.backend,
        suite=config.suite,
        task_ids=(config.task_id,),
        seed=config.seed,
        num_candidates=config.num_candidates,
        chunk_steps=config.chunk_steps,
        action_dim=config.action_dim,
        output_dir=config.output_dir,
        actual_max_steps=config.max_steps,
        direct_libero_double_sim=True,
        direct_camera_name=config.camera_name,
        direct_image_width=config.image_width,
        direct_image_height=config.image_height,
    )


def run_direct_libero_probe(config: DirectLiberoProbeConfig) -> RiskProbeReport:
    return run_risk_probes(build_risk_probe_config(config))


def direct_probe_payload(report: RiskProbeReport) -> dict[str, Any]:
    return {
        "status": report.status,
        "risk_verdicts": report.risk_verdicts,
        "summary_path": report.artifacts["summary"],
        "events_path": report.artifacts["events"],
        "html_report": report.artifacts["html_report"],
        "direct_libero_double_sim_evidence": report.artifacts.get("direct_libero_double_sim_evidence"),
        "blockers": report.blockers,
    }
