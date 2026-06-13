#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from physical_ai_agent.imagine_then_act.risk_probes import (
    RiskProbeConfig,
    compute_diversity_metrics,
    generate_mock_candidates,
    run_risk_probes,
    sample_policy_action_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect SmolVLA stochastic sampling support and optionally sample same-observation candidate chunks. "
            "Risk1 PASS is not claimed by this probe."
        )
    )
    parser.add_argument("--mode", choices=("inspect", "mock", "libero-contract"), default="inspect")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--renderer-backend", default="egl")
    parser.add_argument("--actual-timeout-sec", type=int, default=1800)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = inspect_smolvla_sampling_api()
    payload: dict[str, Any] = {
        "probe": "smolvla_sampling_api",
        "mode": args.mode,
        "sampling_api_available": api["sampling_api_available"],
        "api": api,
        "risk1_claim": "not_claimed",
    }
    if args.mode == "mock":
        payload["mock_sampling"] = run_mock_sampling_probe(args)
    elif args.mode == "libero-contract":
        payload["libero_contract_sampling"] = run_libero_contract_sampling_probe(args)
    output_path = args.output_dir / "smolvla_sampling_probe.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"output_path": str(output_path), **payload}, indent=2, sort_keys=True))
    else:
        print(f"smolvla_sampling_probe={output_path}")
        print(f"sampling_api_available={payload['sampling_api_available']}")


def inspect_smolvla_sampling_api() -> dict[str, Any]:
    result: dict[str, Any] = {
        "sampling_api_available": "unclear",
        "import_error": None,
        "config_fields": {},
        "method_signatures": {},
        "supports": {
            "noise_argument": False,
            "temperature": False,
            "generator": False,
            "num_samples": False,
            "dropout_or_train_toggle": False,
            "action_noise_option": False,
        },
        "notes": [],
    }
    try:
        modeling = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
        configuration = importlib.import_module("lerobot.policies.smolvla.configuration_smolvla")
    except Exception as exc:  # noqa: BLE001 - diagnostic must be import-guarded.
        result["import_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["notes"].append("LeRobot SmolVLA import failed in this environment.")
        return result

    policy_cls = getattr(modeling, "SmolVLAPolicy", None)
    model_cls = getattr(modeling, "VLAFlowMatching", None)
    config_cls = getattr(configuration, "SmolVLAConfig", None)
    for method_name in ("predict_action_chunk", "select_action", "_get_action_chunk"):
        method = getattr(policy_cls, method_name, None)
        if method is not None:
            result["method_signatures"][f"SmolVLAPolicy.{method_name}"] = str(inspect.signature(method))
    for method_name in ("sample_actions", "sample_noise"):
        method = getattr(model_cls, method_name, None)
        if method is not None:
            result["method_signatures"][f"VLAFlowMatching.{method_name}"] = str(inspect.signature(method))
    if config_cls is not None:
        annotations = getattr(config_cls, "__annotations__", {})
        for field_name in (
            "num_steps",
            "chunk_size",
            "n_action_steps",
            "rtc_config",
            "compile_model",
        ):
            result["config_fields"][field_name] = {
                "present": hasattr(config_cls, field_name) or field_name in annotations,
                "default": getattr(config_cls, field_name, None),
            }
    signature_blob = "\n".join(result["method_signatures"].values())
    result["supports"]["noise_argument"] = "noise" in signature_blob
    result["supports"]["temperature"] = "temperature" in signature_blob
    result["supports"]["generator"] = "generator" in signature_blob
    result["supports"]["num_samples"] = "num_samples" in signature_blob or "num_candidates" in signature_blob
    result["supports"]["dropout_or_train_toggle"] = False
    result["supports"]["action_noise_option"] = result["supports"]["noise_argument"]
    if result["supports"]["noise_argument"]:
        result["sampling_api_available"] = "yes"
        result["notes"].append("SmolVLA exposes predict_action_chunk/select_action noise=...; use explicit noise tensors for seeded candidates.")
    else:
        result["sampling_api_available"] = "no"
        result["notes"].append("No explicit stochastic sampling argument was detected on SmolVLA policy methods.")
    return result


def run_mock_sampling_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = RiskProbeConfig(
        preset="smolvla-sampling-mock",
        backend="mock",
        suite=args.suite,
        task_ids=(args.task_id,),
        seed=args.seed,
        num_candidates=args.num_candidates,
        chunk_steps=min(args.chunk_steps, 3),
        action_dim=min(args.action_dim, 3),
        output_dir=str(args.output_dir),
    )
    candidates, metadata = sample_policy_action_candidates(
        policy=_FakeNoiseAwarePolicy(config.chunk_steps, config.action_dim),
        observation={"state": _FakeTensor([[0.0]])},
        postprocessor=lambda action: action,
        env_postprocessor=lambda transition: transition,
        action_key="action",
        config=config,
        torch_module=_FakeTorch,
        numpy_module=None,
    )
    metrics = compute_diversity_metrics(config, candidates)
    return {
        "candidate_generation": metadata,
        "diversity": asdict(metrics),
        "candidates": [candidate.sampling_metadata for candidate in candidates],
    }


def run_libero_contract_sampling_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = RiskProbeConfig(
        preset="runpod-libero-sampling-probe",
        backend="libero-contract",
        suite=args.suite,
        task_ids=(args.task_id,),
        seed=args.seed,
        num_candidates=args.num_candidates,
        chunk_steps=args.chunk_steps,
        action_dim=args.action_dim,
        output_dir=str(args.output_dir),
        policy_path=args.policy_path,
        actual_max_steps=args.chunk_steps,
        actual_timeout_sec=args.actual_timeout_sec,
        renderer_backend=args.renderer_backend,
    )
    report = run_risk_probes(config)
    return {
        "status": report.status,
        "risk_verdicts": report.risk_verdicts,
        "diversity": asdict(report.diversity),
        "candidate_generation": report.actual_evidence.get("candidate_generation", {}),
        "candidate_count": len(report.candidates),
        "candidate_artifact": report.artifacts.get("candidate_action_chunks"),
        "libero_adapter_evidence": report.artifacts.get("libero_adapter_evidence"),
    }


class _FakeTensor(list):
    @property
    def shape(self) -> tuple[int, ...]:
        if self and isinstance(self[0], list):
            return (len(self), len(self[0]))
        return (len(self),)

    @property
    def device(self) -> str:
        return "cpu"


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def inference_mode():
        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        return _Context()

    @staticmethod
    def manual_seed(seed):  # noqa: ANN001
        import random

        random.seed(seed)

    @staticmethod
    def normal(**kwargs):  # noqa: ANN003
        import random

        size = kwargs["size"]
        return [
            [
                [round(random.gauss(kwargs.get("mean", 0.0), kwargs.get("std", 1.0)), 6) for _dim in range(size[2])]
                for _step in range(size[1])
            ]
            for _batch in range(size[0])
        ]


class _FakePolicyConfig:
    def __init__(self, chunk_size: int, action_dim: int) -> None:
        self.chunk_size = chunk_size
        self.max_action_dim = action_dim


class _FakeNoiseAwarePolicy:
    name = "fake_smolvla"

    def __init__(self, chunk_size: int, action_dim: int) -> None:
        self.config = _FakePolicyConfig(chunk_size, action_dim)

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, _observation: object, noise: Any = None) -> Any:
        if noise is None:
            return [[[0.0 for _dim in range(self.config.max_action_dim)] for _step in range(self.config.chunk_size)]]
        return noise


if __name__ == "__main__":
    main()
