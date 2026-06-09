from __future__ import annotations

import html
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class RiskProbeConfig:
    preset: str
    backend: str
    suite: str
    task_ids: tuple[int, ...]
    seed: int
    num_candidates: int
    chunk_steps: int
    action_dim: int
    output_dir: str
    policy_path: str = "lerobot/smolvla_libero"
    camera_mapping: str = '{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'
    policy_num_steps: int = 10
    policy_n_action_steps: int = 15
    actual_max_steps: int = 15
    image_frequency: int = 1
    diversity_warn_threshold: float = 0.05
    diversity_fail_threshold: float = 0.001
    clone_state_l2_threshold: float = 1e-9
    clone_image_mse_threshold: float = 1e-9


@dataclass(frozen=True)
class ActionChunkCandidate:
    candidate_id: str
    source: str
    action_chunk: list[list[float]]
    privileged_success_proxy: float
    is_policy_only: bool = False


@dataclass(frozen=True)
class DiversityMetrics:
    verdict: str
    min_pairwise_l2: float
    mean_pairwise_l2: float
    max_pairwise_l2: float
    endpoint_spread_l2: float
    mean_per_dim_variance: float
    gripper_command_variance: float
    candidate_count: int
    rationale: str


@dataclass(frozen=True)
class CloneFidelityMetrics:
    verdict: str
    state_l2: float
    image_mse: float
    image_mae: float
    success_proxy_delta: float
    deterministic_replay_mismatch: bool
    rationale: str


@dataclass(frozen=True)
class OracleUpperBoundMetrics:
    verdict: str
    policy_only_score: float
    random_chunk_score: float
    oracle_selector_score: float
    selected_candidate_id: str
    oracle_beats_policy: bool
    oracle_beats_random: bool
    rationale: str


@dataclass(frozen=True)
class RiskProbeReport:
    status: str
    preset: str
    backend: str
    suite: str
    task_ids: list[int]
    seed: int
    risk_verdicts: dict[str, str]
    diversity: DiversityMetrics
    clone_fidelity: CloneFidelityMetrics
    oracle_upper_bound: OracleUpperBoundMetrics
    candidates: list[dict[str, Any]]
    artifacts: dict[str, str]
    blockers: list[str]
    actual_evidence: dict[str, Any]


def run_risk_probes(config: RiskProbeConfig) -> RiskProbeReport:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_mock_candidates(config)
    outcomes = {candidate.candidate_id: simulate_mock_env(candidate.action_chunk) for candidate in candidates}
    diversity = compute_diversity_metrics(config, candidates)
    clone_fidelity = compute_clone_fidelity_metrics(config, candidates[-1])
    oracle_upper_bound = compute_oracle_upper_bound_metrics(candidates, outcomes)
    blockers: list[str] = []
    actual_evidence: dict[str, Any] = {"mode": "mock", "available": False}
    if config.backend != "mock":
        actual_evidence = run_libero_actual_adapter(config, candidates, output_dir)
        blockers.extend(actual_evidence.get("blockers", []))
        if actual_evidence.get("outcomes"):
            outcomes = actual_evidence["outcomes"]
        if actual_evidence.get("clone_fidelity"):
            clone_fidelity = CloneFidelityMetrics(**actual_evidence["clone_fidelity"])
        if actual_evidence.get("oracle_upper_bound"):
            oracle_upper_bound = OracleUpperBoundMetrics(**actual_evidence["oracle_upper_bound"])
    artifacts = write_visual_artifacts(output_dir, candidates, outcomes, diversity, clone_fidelity, oracle_upper_bound)
    if actual_evidence.get("artifact_path"):
        artifacts["libero_adapter_evidence"] = actual_evidence["artifact_path"]
    risk_verdicts = {
        "risk_1_candidate_diversity": diversity.verdict,
        "risk_2_clone_fidelity": clone_fidelity.verdict if clone_fidelity.verdict != PASS or not blockers else PASS,
        "risk_5_oracle_selector_upper_bound": oracle_upper_bound.verdict if oracle_upper_bound.verdict != PASS or not blockers else PASS,
    }
    if blockers:
        if clone_fidelity.verdict == PASS:
            risk_verdicts["risk_2_clone_fidelity"] = BLOCKED
        if oracle_upper_bound.verdict == PASS:
            risk_verdicts["risk_5_oracle_selector_upper_bound"] = BLOCKED
    status = aggregate_status(risk_verdicts.values())
    report = RiskProbeReport(
        status=status,
        preset=config.preset,
        backend=config.backend,
        suite=config.suite,
        task_ids=list(config.task_ids),
        seed=config.seed,
        risk_verdicts=risk_verdicts,
        diversity=diversity,
        clone_fidelity=clone_fidelity,
        oracle_upper_bound=oracle_upper_bound,
        candidates=[
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "privileged_success_proxy": candidate.privileged_success_proxy,
                "is_policy_only": candidate.is_policy_only,
            }
            for candidate in candidates
        ],
        artifacts=artifacts,
        blockers=blockers,
        actual_evidence=actual_evidence,
    )
    write_report_bundle(output_dir, config, report)
    return report


def aggregate_status(verdicts: Any) -> str:
    verdict_set = set(verdicts)
    if FAIL in verdict_set:
        return FAIL
    if BLOCKED in verdict_set:
        return BLOCKED
    if WARN in verdict_set:
        return WARN
    return PASS


def generate_mock_candidates(config: RiskProbeConfig) -> list[ActionChunkCandidate]:
    rng = random.Random(config.seed)
    candidates: list[ActionChunkCandidate] = []
    policy_chunk = [[0.04 for _dim in range(config.action_dim)] for _step in range(config.chunk_steps)]
    candidates.append(
        ActionChunkCandidate(
            candidate_id="candidate_00_policy_only",
            source="mock_policy_only",
            action_chunk=policy_chunk,
            privileged_success_proxy=simulate_mock_env(policy_chunk)["success_proxy"],
            is_policy_only=True,
        )
    )
    for index in range(1, config.num_candidates):
        if index == config.num_candidates - 1:
            chunk = [[0.0 for _dim in range(config.action_dim)] for _step in range(config.chunk_steps)]
            for step in range(config.chunk_steps):
                chunk[step][0] = round(1.2 / config.chunk_steps, 4)
                if config.action_dim > 1:
                    chunk[step][1] = round(0.8 / config.chunk_steps, 4)
                if config.action_dim > 2:
                    chunk[step][2] = round(0.5 / config.chunk_steps, 4)
                chunk[step][-1] = 0.35
            source = "mock_oracle_good_chunk"
        else:
            chunk = [
                [round(rng.uniform(-0.2, 0.2), 4) for _dim in range(config.action_dim)]
                for _step in range(config.chunk_steps)
            ]
            source = "mock_seeded_random_chunk"
        candidates.append(
            ActionChunkCandidate(
                candidate_id=f"candidate_{index:02d}",
                source=source,
                action_chunk=chunk,
                privileged_success_proxy=simulate_mock_env(chunk)["success_proxy"],
            )
        )
    return candidates


def flatten_chunk(chunk: list[list[float]]) -> list[float]:
    return [value for row in chunk for value in row]


def l2(values_a: list[float], values_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(values_a, values_b)))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def compute_diversity_metrics(config: RiskProbeConfig, candidates: list[ActionChunkCandidate]) -> DiversityMetrics:
    flat_chunks = [flatten_chunk(candidate.action_chunk) for candidate in candidates]
    pairwise = []
    for left in range(len(flat_chunks)):
        for right in range(left + 1, len(flat_chunks)):
            pairwise.append(l2(flat_chunks[left], flat_chunks[right]))
    endpoints = [candidate.action_chunk[-1] for candidate in candidates]
    endpoint_distances = []
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            endpoint_distances.append(l2(endpoints[left], endpoints[right]))
    per_dim_variances = []
    for dim in range(config.action_dim):
        dim_values = [row[dim] for candidate in candidates for row in candidate.action_chunk]
        per_dim_variances.append(variance(dim_values))
    gripper_values = [row[-1] for candidate in candidates for row in candidate.action_chunk]
    min_pairwise = min(pairwise) if pairwise else 0.0
    mean_variance = mean(per_dim_variances)
    if min_pairwise <= config.diversity_fail_threshold:
        verdict = FAIL
        rationale = "Candidate chunks are identical or nearly identical."
    elif min_pairwise <= config.diversity_warn_threshold or mean_variance <= config.diversity_fail_threshold:
        verdict = WARN
        rationale = "Candidate chunks have limited spread; increase sampling diversity before method claims."
    else:
        verdict = PASS
        rationale = "Candidate chunks show non-trivial action spread in the mock probe."
    return DiversityMetrics(
        verdict=verdict,
        min_pairwise_l2=round(min_pairwise, 6),
        mean_pairwise_l2=round(mean(pairwise), 6),
        max_pairwise_l2=round(max(pairwise) if pairwise else 0.0, 6),
        endpoint_spread_l2=round(max(endpoint_distances) if endpoint_distances else 0.0, 6),
        mean_per_dim_variance=round(mean_variance, 6),
        gripper_command_variance=round(variance(gripper_values), 6),
        candidate_count=len(candidates),
        rationale=rationale,
    )


def simulate_mock_env(action_chunk: list[list[float]]) -> dict[str, Any]:
    state = [0.0, 0.0, 0.0]
    for row in action_chunk:
        for index in range(min(3, len(row))):
            state[index] += row[index]
    target = [1.2, 0.8, 0.5]
    distance = l2(state, target)
    success_proxy = max(0.0, 1.0 - distance / 2.0)
    image = render_mock_image_matrix(state)
    return {
        "state": state,
        "success_proxy": success_proxy,
        "image": image,
    }


def render_mock_image_matrix(state: list[float], size: int = 16) -> list[list[int]]:
    center_x = max(0, min(size - 1, int(round(size / 2 + state[0] * 3))))
    center_y = max(0, min(size - 1, int(round(size / 2 - state[1] * 3))))
    image: list[list[int]] = []
    for y in range(size):
        row = []
        for x in range(size):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            value = max(0, int(255 - distance * 55))
            row.append(value)
        image.append(row)
    return image


def compute_clone_fidelity_metrics(config: RiskProbeConfig, candidate: ActionChunkCandidate) -> CloneFidelityMetrics:
    committed = simulate_mock_env(candidate.action_chunk)
    clone = simulate_mock_env(json.loads(json.dumps(candidate.action_chunk)))
    state_l2 = l2(committed["state"], clone["state"])
    image_mse, image_mae = image_errors(committed["image"], clone["image"])
    success_delta = abs(committed["success_proxy"] - clone["success_proxy"])
    mismatch = (
        state_l2 > config.clone_state_l2_threshold
        or image_mse > config.clone_image_mse_threshold
        or success_delta > config.clone_state_l2_threshold
    )
    verdict = FAIL if mismatch else PASS
    rationale = (
        "Clone and committed deterministic mock rollout matched."
        if verdict == PASS
        else "Clone and committed deterministic mock rollout diverged."
    )
    return CloneFidelityMetrics(
        verdict=verdict,
        state_l2=round(state_l2, 12),
        image_mse=round(image_mse, 12),
        image_mae=round(image_mae, 12),
        success_proxy_delta=round(success_delta, 12),
        deterministic_replay_mismatch=mismatch,
        rationale=rationale,
    )


def image_errors(image_a: list[list[int]], image_b: list[list[int]]) -> tuple[float, float]:
    diffs = []
    for row_a, row_b in zip(image_a, image_b):
        for value_a, value_b in zip(row_a, row_b):
            diffs.append(float(value_a - value_b))
    mse = mean([diff * diff for diff in diffs])
    mae = mean([abs(diff) for diff in diffs])
    return mse, mae


def compute_oracle_upper_bound_metrics(
    candidates: list[ActionChunkCandidate],
    outcomes: dict[str, dict[str, Any]],
) -> OracleUpperBoundMetrics:
    policy = next(candidate for candidate in candidates if candidate.is_policy_only)
    random_candidates = [candidate for candidate in candidates if not candidate.is_policy_only and "random" in candidate.source]
    random_best = max(random_candidates, key=lambda item: outcomes[item.candidate_id]["success_proxy"]) if random_candidates else policy
    oracle = max(candidates, key=lambda item: outcomes[item.candidate_id]["success_proxy"])
    policy_score = outcomes[policy.candidate_id]["success_proxy"]
    random_score = outcomes[random_best.candidate_id]["success_proxy"]
    oracle_score = outcomes[oracle.candidate_id]["success_proxy"]
    beats_policy = oracle_score > policy_score
    beats_random = oracle_score >= random_score
    verdict = PASS if beats_policy and beats_random else WARN
    rationale = (
        "Privileged oracle selector finds an upper-bound candidate in the mock fixture."
        if verdict == PASS
        else "Oracle selector did not improve over baseline/random in the mock fixture."
    )
    return OracleUpperBoundMetrics(
        verdict=verdict,
        policy_only_score=round(policy_score, 6),
        random_chunk_score=round(random_score, 6),
        oracle_selector_score=round(oracle_score, 6),
        selected_candidate_id=oracle.candidate_id,
        oracle_beats_policy=beats_policy,
        oracle_beats_random=beats_random,
        rationale=rationale,
    )


def run_libero_actual_adapter(
    config: RiskProbeConfig,
    candidates: list[ActionChunkCandidate],
    output_dir: Path,
) -> dict[str, Any]:
    evidence_path = output_dir / "libero_adapter_evidence.json"
    try:
        from lerobot.scripts import lerobot_eval
    except Exception as exc:  # noqa: BLE001 - import guard keeps local tests dependency-free.
        evidence = {
            "mode": "libero_actual_adapter",
            "available": False,
            "blockers": [
                "LIBERO actual adapter blocked before env rollout: could not import lerobot.scripts.lerobot_eval "
                f"({type(exc).__name__}: {str(exc)[:300]}). Run inside the prepared RunPod LeRobot/LIBERO environment."
            ],
            "import_error": {"type": type(exc).__name__, "message": str(exc)[:500]},
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence["artifact_path"] = str(evidence_path)
        return evidence

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv[:]
    old_rollout = getattr(lerobot_eval, "rollout", None)
    try:
        lerobot_eval.rollout = build_libero_risk_probe_rollout(
            evidence_path=evidence_path,
            candidates=candidates,
            seed=config.seed,
            max_steps=config.actual_max_steps,
        )
        sys.argv = ["lerobot-eval", *build_lerobot_eval_argv(config, output_dir)]
        lerobot_eval.main()
    except Exception as exc:  # noqa: BLE001 - adapter should report actionable failure.
        evidence = {
            "mode": "libero_actual_adapter",
            "available": False,
            "blockers": [
                "LIBERO actual adapter failed during env rollout: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            ],
            "exception": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            "argv": sys.argv[1:],
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        sys.argv = old_argv
        if old_rollout is not None:
            lerobot_eval.rollout = old_rollout
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    else:
        evidence = {
            "mode": "libero_actual_adapter",
            "available": False,
            "blockers": ["LIBERO actual adapter did not produce evidence JSON."],
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence["artifact_path"] = str(evidence_path)
    return evidence


def build_lerobot_eval_argv(config: RiskProbeConfig, output_dir: Path) -> list[str]:
    task_id = config.task_ids[0] if config.task_ids else 0
    return [
        f"--output_dir={output_dir / 'libero_eval_logs'}",
        f"--policy.path={config.policy_path}",
        "--env.type=libero",
        f"--env.task={config.suite}",
        f"--env.task_ids=[{task_id}]",
        f"--env.camera_name_mapping={config.camera_mapping}",
        "--eval.n_episodes=1",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--env.max_parallel_tasks=1",
        "--policy.empty_cameras=0",
        f"--seed={config.seed}",
        f"--policy.num_steps={config.policy_num_steps}",
        f"--policy.n_action_steps={config.policy_n_action_steps}",
    ]


def build_libero_risk_probe_rollout(
    *,
    evidence_path: Path,
    candidates: list[ActionChunkCandidate],
    seed: int,
    max_steps: int,
):
    def rollout(
        env,
        policy,
        env_preprocessor,
        env_postprocessor,
        preprocessor,
        postprocessor,
        seeds=None,
        return_observations=False,
        render_callback=None,
    ) -> dict:
        del env_preprocessor, env_postprocessor, preprocessor, postprocessor, return_observations
        import numpy as np
        import torch

        from lerobot.utils.constants import ACTION

        if hasattr(policy, "reset"):
            policy.reset()
        rollout_seed = seeds if seeds is not None else [seed]
        introspection = inspect_actual_env(env)
        selected = select_actual_probe_candidates(candidates)
        candidate_results: dict[str, dict[str, Any]] = {}
        for candidate in selected:
            committed = apply_candidate_to_env(env, candidate, rollout_seed, max_steps, np)
            clone = apply_candidate_to_env(env, candidate, rollout_seed, max_steps, np)
            state_l2 = l2(committed["state_vector"], clone["state_vector"]) if committed["state_vector"] and clone["state_vector"] else 0.0
            image_mse, image_mae = image_errors_from_vectors(committed["image_vector"], clone["image_vector"])
            success_delta = abs(committed["success_proxy"] - clone["success_proxy"])
            candidate_results[candidate.candidate_id] = {
                "committed": committed,
                "clone_or_imagination": clone,
                "state_l2": round(state_l2, 12),
                "image_mse": round(image_mse, 12),
                "image_mae": round(image_mae, 12),
                "success_proxy_delta": round(success_delta, 12),
            }
            if render_callback is not None:
                try:
                    render_callback(env)
                except Exception:  # noqa: BLE001
                    pass

        clone_candidate = selected[-1]
        clone_record = candidate_results[clone_candidate.candidate_id]
        exact_clone_available = bool(introspection.get("exact_state_clone_available"))
        clone_verdict = PASS if exact_clone_available and clone_record["state_l2"] == 0.0 and clone_record["image_mse"] == 0.0 else BLOCKED
        clone_rationale = (
            "Actual LIBERO exact state clone/replay matched for the selected candidate."
            if clone_verdict == PASS
            else "Actual LIBERO exact simulator state clone API was not confirmed; deterministic seed replay evidence was recorded but is not a clone-fidelity pass."
        )
        outcomes = {
            candidate_id: {
                "success_proxy": record["committed"]["success_proxy"],
                "state": record["committed"]["state_vector"][:3],
                "image": vector_to_image_matrix(record["committed"]["image_vector"]),
            }
            for candidate_id, record in candidate_results.items()
        }
        oracle_metrics = compute_actual_oracle_or_proxy_metrics(selected, outcomes, introspection)
        evidence = {
            "mode": "libero_actual_adapter",
            "available": True,
            "blockers": [] if clone_verdict == PASS else [clone_rationale],
            "introspection": introspection,
            "candidate_results": candidate_results,
            "outcomes": outcomes,
            "clone_fidelity": asdict(
                CloneFidelityMetrics(
                    verdict=clone_verdict,
                    state_l2=clone_record["state_l2"],
                    image_mse=clone_record["image_mse"],
                    image_mae=clone_record["image_mae"],
                    success_proxy_delta=clone_record["success_proxy_delta"],
                    deterministic_replay_mismatch=bool(
                        clone_record["state_l2"] > 0.0 or clone_record["image_mse"] > 0.0
                    ),
                    rationale=clone_rationale,
                )
            ),
            "oracle_upper_bound": asdict(oracle_metrics),
            "oracle_mode": "privileged" if introspection.get("privileged_state_available") else "proxy_only",
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reward = np.zeros((getattr(env, "num_envs", 1),), dtype=np.float32)
        success = np.zeros((getattr(env, "num_envs", 1),), dtype=bool)
        done = np.ones((getattr(env, "num_envs", 1),), dtype=bool)
        action_dim = len(candidates[0].action_chunk[0]) if candidates and candidates[0].action_chunk else 1
        action = np.zeros((getattr(env, "num_envs", 1), 1, action_dim), dtype=np.float32)
        return {
            ACTION: torch.from_numpy(action),
            "reward": torch.from_numpy(np.expand_dims(reward, axis=1)),
            "success": torch.from_numpy(np.expand_dims(success, axis=1)),
            "done": torch.from_numpy(np.expand_dims(done, axis=1)),
        }

    return rollout


def select_actual_probe_candidates(candidates: list[ActionChunkCandidate]) -> list[ActionChunkCandidate]:
    policy = [candidate for candidate in candidates if candidate.is_policy_only][:1]
    random_candidates = [candidate for candidate in candidates if "random" in candidate.source][:1]
    oracle = [max(candidates, key=lambda item: item.privileged_success_proxy)]
    result: list[ActionChunkCandidate] = []
    for candidate in [*policy, *random_candidates, *oracle]:
        if candidate.candidate_id not in {item.candidate_id for item in result}:
            result.append(candidate)
    return result or candidates[:1]


def apply_candidate_to_env(env: Any, candidate: ActionChunkCandidate, seed_value: Any, max_steps: int, np_module: Any) -> dict[str, Any]:
    observation, reset_info = env.reset(seed=seed_value)
    info: Any = reset_info
    reward: Any = 0.0
    terminated: Any = False
    truncated: Any = False
    for row in candidate.action_chunk[:max_steps]:
        action = np_module.asarray([row], dtype=getattr(np_module, "float32", None))
        observation, reward, terminated, truncated, info = env.step(action)
        if is_done(terminated) or is_done(truncated):
            break
    state_vector, state_source = extract_state_vector(env, observation, info)
    image_vector, image_source = extract_image_vector(observation)
    success_proxy, success_source = extract_success_proxy(info, reward, state_vector)
    return {
        "candidate_id": candidate.candidate_id,
        "state_vector": state_vector,
        "state_source": state_source,
        "image_vector": image_vector,
        "image_source": image_source,
        "success_proxy": success_proxy,
        "success_proxy_source": success_source,
        "info_summary": summarize_value(info),
    }


def is_done(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return any(bool(item) for item in value)
    return bool(value)


def inspect_actual_env(env: Any) -> dict[str, Any]:
    call_probe = {}
    for name in ("get_sim_state", "get_state", "get_object_state", "set_sim_state", "set_state", "task_description"):
        try:
            call_probe[name] = summarize_value(env.call(name))
        except Exception as exc:  # noqa: BLE001
            call_probe[name] = {"error": type(exc).__name__, "message": str(exc)[:300]}
    attr_names = sorted(name for name in dir(env) if any(part in name.lower() for part in ("sim", "state", "env", "object")))[:150]
    internal = inspect_internal_sim(env)
    exact_clone_available = bool(internal.get("sim_get_state_available") and internal.get("sim_set_state_available"))
    privileged_state_available = exact_clone_available or any(
        not isinstance(call_probe.get(name), dict) or "error" not in call_probe.get(name, {})
        for name in ("get_sim_state", "get_state", "get_object_state")
    )
    return {
        "env_type": f"{type(env).__module__}.{type(env).__name__}",
        "num_envs": getattr(env, "num_envs", None),
        "call_probe": call_probe,
        "interesting_attrs": attr_names,
        "internal_sim": internal,
        "exact_state_clone_available": exact_clone_available,
        "privileged_state_available": privileged_state_available,
    }


def inspect_internal_sim(env: Any) -> dict[str, Any]:
    try:
        inner = env.envs[0]
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "message": str(exc)[:300]}
    candidates = []
    for root in (inner, getattr(inner, "_env", None)):
        if root is None:
            continue
        for path in ("sim", "env.sim", "env.env.sim", "env.env.env.sim"):
            current = root
            try:
                for part in path.split("."):
                    current = getattr(current, part)
                candidates.append((path, current))
            except Exception:  # noqa: BLE001
                continue
    if not candidates:
        return {
            "inner_type": f"{type(inner).__module__}.{type(inner).__name__}",
            "sim_get_state_available": False,
            "sim_set_state_available": False,
        }
    path, sim = candidates[0]
    return {
        "inner_type": f"{type(inner).__module__}.{type(inner).__name__}",
        "sim_path": path,
        "sim_type": f"{type(sim).__module__}.{type(sim).__name__}",
        "sim_get_state_available": hasattr(sim, "get_state"),
        "sim_set_state_available": hasattr(sim, "set_state"),
    }


def extract_state_vector(env: Any, observation: Any, info: Any) -> tuple[list[float], str]:
    for name in ("get_sim_state", "get_state", "get_object_state"):
        try:
            value = env.call(name)
            vector = numeric_vector(value, limit=256)
            if vector:
                return vector, f"env.call({name})"
        except Exception:  # noqa: BLE001
            continue
    vector = numeric_vector(info, limit=256)
    if vector:
        return vector, "info_numeric"
    vector = numeric_vector(observation, limit=256, skip_image_like=True)
    return vector, "observation_numeric" if vector else "unavailable"


def extract_image_vector(observation: Any) -> tuple[list[float], str]:
    if isinstance(observation, dict):
        for key, value in observation.items():
            if "image" in str(key).lower():
                vector = numeric_vector(value, limit=4096)
                if vector:
                    return vector, str(key)
    vector = numeric_vector(observation, limit=4096)
    return vector, "observation_any" if vector else "unavailable"


def extract_success_proxy(info: Any, reward: Any, state_vector: list[float]) -> tuple[float, str]:
    for key in ("is_success", "success", "task_success"):
        value = find_key(info, key)
        if value is not None:
            numbers = numeric_vector(value, limit=10)
            if numbers:
                return max(0.0, min(1.0, max(numbers))), f"info.{key}"
    reward_values = numeric_vector(reward, limit=10)
    if reward_values:
        return max(0.0, min(1.0, max(reward_values))), "reward_clamped"
    if state_vector:
        return max(0.0, min(1.0, 1.0 / (1.0 + math.sqrt(sum(value * value for value in state_vector[:8]))))), "state_norm_proxy"
    return 0.0, "unavailable"


def find_key(value: Any, wanted: str) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == wanted:
                return item
            found = find_key(item, wanted)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = find_key(item, wanted)
            if found is not None:
                return found
    return None


def summarize_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return repr(value)[:300]
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(key): summarize_value(item, depth + 1) for key, item in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [summarize_value(item, depth + 1) for item in list(value)[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "module": type(value).__module__, "repr": repr(value)[:300]}


def numeric_vector(value: Any, limit: int, skip_image_like: bool = False) -> list[float]:
    result: list[float] = []

    def visit(item: Any, path: str = "") -> None:
        if len(result) >= limit:
            return
        if hasattr(item, "detach"):
            try:
                item = item.detach()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(item, "cpu"):
            try:
                item = item.cpu()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(item, "tolist"):
            try:
                item = item.tolist()
            except Exception:  # noqa: BLE001
                pass
        if isinstance(item, dict):
            for key, child in item.items():
                if skip_image_like and "image" in str(key).lower():
                    continue
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, path)
            return
        if isinstance(item, bool):
            result.append(float(item))
        elif isinstance(item, (int, float)):
            result.append(float(item))

    visit(value)
    return result[:limit]


def image_errors_from_vectors(vector_a: list[float], vector_b: list[float]) -> tuple[float, float]:
    if not vector_a or not vector_b:
        return 0.0, 0.0
    size = min(len(vector_a), len(vector_b))
    diffs = [vector_a[index] - vector_b[index] for index in range(size)]
    return mean([diff * diff for diff in diffs]), mean([abs(diff) for diff in diffs])


def vector_to_image_matrix(vector: list[float], size: int = 16) -> list[list[int]]:
    if not vector:
        return [[0 for _x in range(size)] for _y in range(size)]
    values = vector[: size * size]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    padded = values + [0.0] * (size * size - len(values))
    return [
        [int(255 * ((padded[y * size + x] - min_value) / span)) for x in range(size)]
        for y in range(size)
    ]


def compute_actual_oracle_or_proxy_metrics(
    candidates: list[ActionChunkCandidate],
    outcomes: dict[str, dict[str, Any]],
    introspection: dict[str, Any],
) -> OracleUpperBoundMetrics:
    metrics = compute_oracle_upper_bound_metrics(candidates, outcomes)
    if not introspection.get("privileged_state_available"):
        return OracleUpperBoundMetrics(
            verdict=WARN,
            policy_only_score=metrics.policy_only_score,
            random_chunk_score=metrics.random_chunk_score,
            oracle_selector_score=metrics.oracle_selector_score,
            selected_candidate_id=metrics.selected_candidate_id,
            oracle_beats_policy=metrics.oracle_beats_policy,
            oracle_beats_random=metrics.oracle_beats_random,
            rationale=(
                "proxy_only: privileged oracle state was unavailable, so candidates were ranked by available env info/obs proxy only. "
                "This is not benchmark success."
            ),
        )
    return OracleUpperBoundMetrics(
        verdict=metrics.verdict,
        policy_only_score=metrics.policy_only_score,
        random_chunk_score=metrics.random_chunk_score,
        oracle_selector_score=metrics.oracle_selector_score,
        selected_candidate_id=metrics.selected_candidate_id,
        oracle_beats_policy=metrics.oracle_beats_policy,
        oracle_beats_random=metrics.oracle_beats_random,
        rationale="oracle_available: privileged env state/proxy was available for upper-bound candidate ranking; this is not benchmark success.",
    )


def write_visual_artifacts(
    output_dir: Path,
    candidates: list[ActionChunkCandidate],
    outcomes: dict[str, dict[str, Any]],
    diversity: DiversityMetrics,
    clone_fidelity: CloneFidelityMetrics,
    oracle: OracleUpperBoundMetrics,
) -> dict[str, str]:
    artifacts = {
        "candidate_action_heatmap": str(output_dir / "candidate_action_heatmap.svg"),
        "oracle_scores": str(output_dir / "oracle_scores.svg"),
        "clone_image_diff": str(output_dir / "clone_image_diff.svg"),
        "html_report": str(output_dir / "risk_probe_report.html"),
        "summary": str(output_dir / "summary.json"),
        "events": str(output_dir / "events.jsonl"),
    }
    (output_dir / "candidate_action_heatmap.svg").write_text(render_action_heatmap_svg(candidates), encoding="utf-8")
    (output_dir / "oracle_scores.svg").write_text(render_score_bar_svg(candidates, outcomes, oracle), encoding="utf-8")
    (output_dir / "clone_image_diff.svg").write_text(render_clone_diff_svg(clone_fidelity), encoding="utf-8")
    return artifacts


def write_report_bundle(output_dir: Path, config: RiskProbeConfig, report: RiskProbeReport) -> None:
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.jsonl"
    html_path = output_dir / "risk_probe_report.html"
    summary_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events = [
        {"event": "config", "payload": asdict(config)},
        {"event": "risk_1_candidate_diversity", "payload": asdict(report.diversity)},
        {"event": "risk_2_clone_fidelity", "payload": asdict(report.clone_fidelity)},
        {"event": "risk_5_oracle_selector_upper_bound", "payload": asdict(report.oracle_upper_bound)},
        {"event": "actual_evidence", "payload": report.actual_evidence},
        {"event": "summary", "payload": {"status": report.status, "risk_verdicts": report.risk_verdicts}},
    ]
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    html_path.write_text(render_html_report(report), encoding="utf-8")


def render_action_heatmap_svg(candidates: list[ActionChunkCandidate]) -> str:
    cell = 14
    label_width = 155
    rows = []
    max_steps = max(len(candidate.action_chunk) for candidate in candidates)
    action_dim = len(candidates[0].action_chunk[0]) if candidates and candidates[0].action_chunk else 1
    width = label_width + max_steps * action_dim * cell + 20
    height = 30 + len(candidates) * cell + 20
    for row_index, candidate in enumerate(candidates):
        y = 30 + row_index * cell
        rows.append(f'<text x="8" y="{y + 11}" font-size="10">{html.escape(candidate.candidate_id)}</text>')
        for step, action in enumerate(candidate.action_chunk):
            for dim, value in enumerate(action):
                x = label_width + (step * action_dim + dim) * cell
                color = value_to_color(value)
                rows.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}"/>')
    return svg_document(width, height, "<text x='8' y='18' font-size='13'>Candidate action heatmap</text>" + "".join(rows))


def value_to_color(value: float) -> str:
    clamped = max(-0.35, min(0.35, value))
    if clamped >= 0:
        red = 255
        green = int(255 * (1 - clamped / 0.35))
        blue = green
    else:
        blue = 255
        red = int(255 * (1 + clamped / 0.35))
        green = red
    return f"rgb({red},{green},{blue})"


def render_score_bar_svg(
    candidates: list[ActionChunkCandidate],
    outcomes: dict[str, dict[str, Any]],
    oracle: OracleUpperBoundMetrics,
) -> str:
    width = 560
    height = 35 + len(candidates) * 32
    rows = ["<text x='8' y='18' font-size='13'>Privileged success proxy by candidate</text>"]
    for index, candidate in enumerate(candidates):
        score = float(outcomes[candidate.candidate_id]["success_proxy"])
        y = 34 + index * 32
        bar_width = int(score * 320)
        fill = "#277da1" if candidate.candidate_id != oracle.selected_candidate_id else "#43aa8b"
        rows.append(f'<text x="8" y="{y + 15}" font-size="10">{html.escape(candidate.candidate_id)}</text>')
        rows.append(f'<rect x="160" y="{y}" width="{bar_width}" height="18" fill="{fill}"/>')
        rows.append(f'<text x="{170 + bar_width}" y="{y + 14}" font-size="10">{score:.3f}</text>')
    return svg_document(width, height, "".join(rows))


def render_clone_diff_svg(clone_fidelity: CloneFidelityMetrics) -> str:
    width = 420
    height = 90
    color = "#43aa8b" if clone_fidelity.verdict == PASS else "#f94144"
    body = (
        "<text x='8' y='18' font-size='13'>Clone-vs-commit diff</text>"
        f"<rect x='8' y='34' width='36' height='36' fill='{color}'/>"
        f"<text x='58' y='50' font-size='11'>state_l2={clone_fidelity.state_l2}</text>"
        f"<text x='58' y='66' font-size='11'>image_mse={clone_fidelity.image_mse}, image_mae={clone_fidelity.image_mae}</text>"
    )
    return svg_document(width, height, body)


def svg_document(width: int, height: int, body: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>{body}</svg>\n"
    )


def render_html_report(report: RiskProbeReport) -> str:
    artifact = {key: Path(value).name for key, value in report.artifacts.items()}
    risk_rows = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td class='{html.escape(verdict.lower())}'>{html.escape(verdict)}</td></tr>"
        for name, verdict in report.risk_verdicts.items()
    )
    blockers = "".join(f"<li>{html.escape(blocker)}</li>" for blocker in report.blockers) or "<li>None</li>"
    actual_link = ""
    if "libero_adapter_evidence" in artifact:
        actual_link = (
            "<h2>Actual Adapter Evidence</h2>"
            f"<p><a href=\"{html.escape(artifact['libero_adapter_evidence'])}\">libero_adapter_evidence.json</a></p>"
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Imagine-Then-Act Risk Probe Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccd5df; padding: 8px 10px; text-align: left; }}
    .pass {{ color: #157347; font-weight: 700; }}
    .warn {{ color: #9a6700; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
    .blocked {{ color: #6f42c1; font-weight: 700; }}
    img {{ display: block; margin: 12px 0 24px; max-width: 100%; border: 1px solid #d8dee6; }}
    code {{ background: #f4f6f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Imagine-Then-Act Risk Probe Report</h1>
  <p>Status: <strong class="{html.escape(report.status.lower())}">{html.escape(report.status)}</strong></p>
  <p>Preset: <code>{html.escape(report.preset)}</code>, backend: <code>{html.escape(report.backend)}</code>, seed: <code>{report.seed}</code></p>
  <h2>Risk Verdicts</h2>
  <table>{risk_rows}</table>
  <h2>Risk 1 Candidate Diversity</h2>
  <p>{html.escape(report.diversity.rationale)}</p>
  <img src="{html.escape(artifact['candidate_action_heatmap'])}" alt="candidate action heatmap">
  <h2>Risk 2 Clone Fidelity</h2>
  <p>{html.escape(report.clone_fidelity.rationale)}</p>
  <img src="{html.escape(artifact['clone_image_diff'])}" alt="clone versus commit image diff">
  <h2>Risk 5 Oracle Selector Upper-Bound</h2>
  <p>{html.escape(report.oracle_upper_bound.rationale)}</p>
  <img src="{html.escape(artifact['oracle_scores'])}" alt="oracle selector candidate scores">
  <h2>Blockers</h2>
  <ul>{blockers}</ul>
  {actual_link}
</body>
</html>
"""
