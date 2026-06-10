from __future__ import annotations

import copy
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
    direct_libero_double_sim: bool = False
    direct_camera_name: str = "agentview"
    direct_image_width: int = 128
    direct_image_height: int = 128
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
    visual_candidates = candidates
    diversity = compute_diversity_metrics(config, candidates)
    clone_fidelity = compute_clone_fidelity_metrics(config, candidates[-1])
    oracle_upper_bound = compute_oracle_upper_bound_metrics(candidates, outcomes)
    blockers: list[str] = []
    actual_evidence: dict[str, Any] = {"mode": "mock", "available": False}
    if config.backend == "direct-libero":
        actual_evidence = run_direct_libero_double_sim_probe(config, candidates, output_dir)
        blockers.extend(actual_evidence.get("blockers", []))
        diversity = mark_diversity_as_synthetic_contract(diversity)
        if actual_evidence.get("outcomes"):
            outcomes = actual_evidence["outcomes"]
            evaluated_ids = set(outcomes)
            visual_candidates = [candidate for candidate in candidates if candidate.candidate_id in evaluated_ids]
        if actual_evidence.get("clone_fidelity"):
            clone_fidelity = CloneFidelityMetrics(**actual_evidence["clone_fidelity"])
        oracle_upper_bound = compute_actual_oracle_or_proxy_metrics(visual_candidates or candidates, outcomes, {"privileged_state_available": True})
    elif config.backend != "mock":
        actual_evidence = run_libero_actual_adapter(config, candidates, output_dir)
        blockers.extend(actual_evidence.get("blockers", []))
        diversity = mark_diversity_as_synthetic_contract(diversity)
        if actual_evidence.get("outcomes"):
            outcomes = actual_evidence["outcomes"]
            evaluated_ids = set(outcomes)
            visual_candidates = [candidate for candidate in candidates if candidate.candidate_id in evaluated_ids]
        if actual_evidence.get("clone_fidelity"):
            clone_fidelity = CloneFidelityMetrics(**actual_evidence["clone_fidelity"])
        if actual_evidence.get("oracle_upper_bound"):
            oracle_upper_bound = OracleUpperBoundMetrics(**actual_evidence["oracle_upper_bound"])
    artifacts = write_visual_artifacts(output_dir, visual_candidates, outcomes, diversity, clone_fidelity, oracle_upper_bound)
    if actual_evidence.get("artifact_path"):
        key = "direct_libero_double_sim_evidence" if config.backend == "direct-libero" else "libero_adapter_evidence"
        artifacts[key] = actual_evidence["artifact_path"]
    direct_evidence = actual_evidence.get("direct_libero_double_sim", {})
    if isinstance(direct_evidence, dict) and direct_evidence.get("artifact_path"):
        artifacts["direct_libero_double_sim_evidence"] = direct_evidence["artifact_path"]
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
                "evaluated_in_actual_adapter": candidate.candidate_id in outcomes if config.backend != "mock" else True,
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


def mark_diversity_as_synthetic_contract(metrics: DiversityMetrics) -> DiversityMetrics:
    return DiversityMetrics(
        verdict=WARN,
        min_pairwise_l2=metrics.min_pairwise_l2,
        mean_pairwise_l2=metrics.mean_pairwise_l2,
        max_pairwise_l2=metrics.max_pairwise_l2,
        endpoint_spread_l2=metrics.endpoint_spread_l2,
        mean_per_dim_variance=metrics.mean_per_dim_variance,
        gripper_command_variance=metrics.gripper_command_variance,
        candidate_count=metrics.candidate_count,
        rationale=(
            "synthetic candidate diversity smoke only: runpod-libero actual adapter applies configured candidates, "
            "but does not yet prove SmolVLA policy-generated candidate chunk diversity."
        ),
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
    import_compat = apply_torch_transformers_import_compatibility_patch()
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
            "import_compat": import_compat,
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence["artifact_path"] = str(evidence_path)
        return evidence

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv[:]
    old_rollout = getattr(lerobot_eval, "rollout", None)
    try:
        lerobot_eval.rollout = build_libero_risk_probe_rollout(
            config=config,
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
            "import_compat": import_compat,
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


def apply_torch_transformers_import_compatibility_patch() -> dict[str, Any]:
    """Patch narrow torch/Transformers import drift seen on RunPod cu124.

    LeRobot 0.5.2 currently imports Transformers 5.x utilities, while the
    known-good RunPod driver path keeps torch at 2.5.1+cu124. Transformers 5.x
    probes ``torch.float8_e8m0fnu`` during lazy AutoProcessor imports; torch
    2.5 exposes other float8 dtypes but not that newer alias. The risk probe
    only needs import/runtime plumbing here, so provide the alias before
    importing LeRobot instead of upgrading torch into the CUDA 13 failure path.
    """

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency on local tests.
        return {
            "torch_imported": False,
            "patched": False,
            "patch": "torch.float8_e8m0fnu",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

    if hasattr(torch, "float8_e8m0fnu"):
        return {
            "torch_imported": True,
            "patched": False,
            "patch": "torch.float8_e8m0fnu",
            "reason": "already_present",
            "torch_version": getattr(torch, "__version__", "unknown"),
        }
    if hasattr(torch, "float8_e5m2"):
        setattr(torch, "float8_e8m0fnu", getattr(torch, "float8_e5m2"))
        return {
            "torch_imported": True,
            "patched": True,
            "patch": "torch.float8_e8m0fnu",
            "source_attr": "torch.float8_e5m2",
            "torch_version": getattr(torch, "__version__", "unknown"),
        }
    return {
        "torch_imported": True,
        "patched": False,
        "patch": "torch.float8_e8m0fnu",
        "reason": "source_attr_missing",
        "torch_version": getattr(torch, "__version__", "unknown"),
    }


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


def run_direct_libero_double_sim_probe(
    config: RiskProbeConfig,
    candidates: list[ActionChunkCandidate],
    output_dir: Path,
) -> dict[str, Any]:
    evidence_path = output_dir / "direct_libero_double_sim_evidence.json"
    try:
        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:  # noqa: BLE001 - local tests must not require LIBERO.
        evidence = {
            "enabled": True,
            "available": False,
            "blockers": [
                "direct LIBERO double-sim blocked before env creation: "
                f"{type(exc).__name__}: {str(exc)[:400]}"
            ],
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence["artifact_path"] = str(evidence_path)
        return evidence

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        bench = benchmark_dict[config.suite]()
        task_id = config.task_ids[0] if config.task_ids else 0
        task = bench.get_task(task_id)
        bddl_file = Path(str(task.bddl_file))
        if not bddl_file.is_absolute():
            bddl_root = Path(get_libero_path("bddl_files"))
            direct = bddl_root / bddl_file
            if direct.exists():
                bddl_file = direct
            else:
                matches = sorted(bddl_root.rglob(bddl_file.name))
                if not matches:
                    raise FileNotFoundError(f"Could not resolve LIBERO BDDL file: {bddl_file}")
                bddl_file = matches[0]
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=config.direct_image_height,
            camera_widths=config.direct_image_width,
        )
        env.seed(config.seed + task_id * 1000)
        init_states = bench.get_task_init_states(task_id)
        init_state = init_states[0] if len(init_states) > 0 else None
        evidence = run_direct_env_snapshot_replay(
            env=env,
            init_state=init_state,
            candidates=candidates,
            output_dir=output_dir,
            camera_name=config.direct_camera_name,
            max_steps=config.actual_max_steps,
            np_module=np,
        )
        evidence.update(
            {
                "enabled": True,
                "available": True,
                "suite": config.suite,
                "task_id": task_id,
                "task_name": getattr(task, "name", f"task_{task_id}"),
                "bddl_file": str(bddl_file),
                "sync_scope": "episode_start_init_state_only",
                "mid_episode_sync": "future_work",
            }
        )
    except Exception as exc:  # noqa: BLE001 - evidence should guide the RunPod rerun.
        evidence = {
            "enabled": True,
            "available": False,
            "blockers": [
                "direct LIBERO double-sim failed during env rollout: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            ],
            "exception": {"type": type(exc).__name__, "message": str(exc)[:1000]},
        }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence["artifact_path"] = str(evidence_path)
    return evidence


def run_direct_env_snapshot_replay(
    *,
    env: Any,
    init_state: Any,
    candidates: list[ActionChunkCandidate],
    output_dir: Path,
    camera_name: str,
    max_steps: int,
    np_module: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_actual_probe_candidates(candidates)
    handle = find_sim_clone_handle(env)
    candidate_results: dict[str, dict[str, Any]] = {}
    image_artifacts: dict[str, str] = {}
    for candidate in selected:
        start_observation = reset_direct_env_to_init_state(env, init_state)
        start_state = capture_sim_state(handle)
        committed = apply_candidate_to_direct_env(
            env,
            candidate,
            max_steps,
            np_module,
            initial_observation=start_observation,
        )
        restore_result = restore_sim_state(handle, start_state)
        replay_start_observation = start_observation if restore_result["restored"] else reset_direct_env_to_init_state(env, init_state)
        replay = apply_candidate_to_direct_env(
            env,
            candidate,
            max_steps,
            np_module,
            initial_observation=replay_start_observation,
        )
        state_l2 = l2(committed["state_vector"], replay["state_vector"]) if committed["state_vector"] and replay["state_vector"] else 0.0
        image_mse, image_mae = image_errors_from_vectors(committed["image_vector"], replay["image_vector"])
        artifact_path = write_future_image_artifact(output_dir, candidate.candidate_id, committed)
        image_artifacts[candidate.candidate_id] = str(artifact_path)
        candidate_results[candidate.candidate_id] = {
            "committed": committed,
            "replay": replay,
            "snapshot_restore": restore_result,
            "state_l2": round(state_l2, 12),
            "image_mse": round(image_mse, 12),
            "image_mae": round(image_mae, 12),
            "future_image_artifact": str(artifact_path),
            "camera_name": camera_name,
        }

    chosen = selected[-1]
    chosen_record = candidate_results[chosen.candidate_id]
    snapshot_restored = bool(chosen_record["snapshot_restore"].get("restored"))
    state_available = bool(chosen_record["committed"]["state_vector"] and chosen_record["replay"]["state_vector"])
    image_available = bool(chosen_record["committed"]["image_vector"] and chosen_record["replay"]["image_vector"])
    final_state_match = chosen_record["state_l2"] == 0.0
    final_image_match = chosen_record["image_mse"] == 0.0
    image_exists = Path(chosen_record["future_image_artifact"]).exists()
    verdict = PASS if snapshot_restored and state_available and image_available and final_state_match and final_image_match and image_exists else BLOCKED
    rationale = (
        "episode-start double-sim PASS: direct LIBERO env used sim snapshot restore and replayed the same action chunk with matching state/image. "
        "This does not prove mid-episode LeRobot env synchronization."
        if verdict == PASS
        else "direct LIBERO double-sim could not prove snapshot-restore replay with matching state/image."
    )
    outcomes = {
        candidate_id: {
            "success_proxy": record["committed"]["success_proxy"],
            "state": record["committed"]["state_vector"][:3],
            "image": vector_to_image_matrix(record["committed"]["image_vector"]),
        }
        for candidate_id, record in candidate_results.items()
    }
    clone_metrics = CloneFidelityMetrics(
        verdict=verdict,
        state_l2=chosen_record["state_l2"],
        image_mse=chosen_record["image_mse"],
        image_mae=chosen_record["image_mae"],
        success_proxy_delta=abs(chosen_record["committed"]["success_proxy"] - chosen_record["replay"]["success_proxy"]),
        deterministic_replay_mismatch=not (final_state_match and final_image_match),
        rationale=rationale,
    )
    return {
        "available": True,
        "blockers": [] if verdict == PASS else [rationale],
        "candidate_results": candidate_results,
        "outcomes": outcomes,
        "image_artifacts": image_artifacts,
        "clone_restore_evidence": {
            "selected_candidate_id": chosen.candidate_id,
            "snapshot_restored": snapshot_restored,
            "snapshot_source": chosen_record["snapshot_restore"].get("path", "unavailable"),
            "restore_error": chosen_record["snapshot_restore"].get("reason"),
            "future_image_artifact": chosen_record["future_image_artifact"],
            "state_available": state_available,
            "image_available": image_available,
        },
        "clone_fidelity": asdict(clone_metrics),
        "sync_scope": "episode_start_init_state_only",
        "mid_episode_sync": "future_work",
    }


def reset_direct_env_to_init_state(env: Any, init_state: Any) -> Any:
    observation = env.reset()
    if init_state is not None and hasattr(env, "set_init_state"):
        observation = env.set_init_state(init_state)
    return observation


def apply_candidate_to_direct_env(
    env: Any,
    candidate: ActionChunkCandidate,
    max_steps: int,
    np_module: Any,
    *,
    initial_observation: Any,
) -> dict[str, Any]:
    observation = initial_observation
    info: Any = {}
    reward: Any = 0.0
    done: Any = False
    for row in candidate.action_chunk[:max_steps]:
        action = np_module.asarray(row, dtype=getattr(np_module, "float32", None))
        result = env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = is_done(terminated) or is_done(truncated)
        else:
            observation, reward, done, info = result
            done = is_done(done)
        if done:
            break
    state_vector, state_source = extract_state_vector(env, observation, info)
    image_vector, image_source = extract_image_vector(observation)
    rgb_image_matrix, rgb_image_source = extract_rgb_image_matrix(observation)
    success_proxy, success_source = extract_success_proxy(info, reward, state_vector)
    privileged_state_vector, privileged_state_source = extract_privileged_state_vector(env)
    privileged_success = extract_privileged_success_proxy(env)
    return {
        "candidate_id": candidate.candidate_id,
        "state_vector": state_vector,
        "state_source": state_source,
        "privileged_state_vector": privileged_state_vector[:64],
        "privileged_state_source": privileged_state_source,
        "image_vector": image_vector,
        "image_source": image_source,
        "rgb_image_matrix": rgb_image_matrix,
        "rgb_image_source": rgb_image_source,
        "success_proxy": success_proxy,
        "success_proxy_source": success_source,
        "privileged_success_proxy": privileged_success[0],
        "privileged_success_proxy_source": privileged_success[1],
        "done": bool(done),
        "info_summary": summarize_value(info),
    }


def build_libero_risk_probe_rollout(
    *,
    config: RiskProbeConfig,
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
        clone_handle = find_sim_clone_handle(env)
        selected = select_actual_probe_candidates(candidates)
        candidate_results: dict[str, dict[str, Any]] = {}
        for candidate in selected:
            start_observation, start_info = env.reset(seed=rollout_seed)
            start_state = capture_sim_state(clone_handle)
            committed = apply_candidate_to_env(
                env,
                candidate,
                rollout_seed,
                max_steps,
                np,
                reset_before=False,
                initial_observation=start_observation,
                initial_info=start_info,
            )
            restore_result = restore_sim_state(clone_handle, start_state)
            clone = apply_candidate_to_env(
                env,
                candidate,
                rollout_seed,
                max_steps,
                np,
                reset_before=not restore_result["restored"],
                initial_observation=start_observation if restore_result["restored"] else None,
                initial_info=start_info if restore_result["restored"] else None,
            )
            state_l2 = l2(committed["state_vector"], clone["state_vector"]) if committed["state_vector"] and clone["state_vector"] else 0.0
            image_mse, image_mae = image_errors_from_vectors(committed["image_vector"], clone["image_vector"])
            success_delta = abs(committed["success_proxy"] - clone["success_proxy"])
            candidate_results[candidate.candidate_id] = {
                "committed": committed,
                "clone_or_imagination": clone,
                "snapshot_restore": restore_result,
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
        restored_exact_state = bool(clone_record["snapshot_restore"].get("restored"))
        clone_verdict = (
            PASS
            if exact_clone_available
            and restored_exact_state
            and clone_record["state_l2"] == 0.0
            and clone_record["image_mse"] == 0.0
            else BLOCKED
        )
        clone_rationale = (
            "Actual LIBERO exact state clone/replay matched for the selected candidate."
            if clone_verdict == PASS
            else (
                "Actual LIBERO exact simulator state clone API was not restored for the selected candidate; "
                "deterministic seed replay evidence was recorded but is not a clone-fidelity pass."
            )
        )
        clone_restore_evidence = {
            "selected_candidate_id": clone_candidate.candidate_id,
            "snapshot_restored": restored_exact_state,
            "snapshot_source": clone_record["snapshot_restore"].get("path", "unavailable"),
            "restore_error": clone_record["snapshot_restore"].get("reason"),
        }
        outcomes = {
            candidate_id: {
                "success_proxy": record["committed"]["success_proxy"],
                "state": record["committed"]["state_vector"][:3],
                "image": vector_to_image_matrix(record["committed"]["image_vector"]),
            }
            for candidate_id, record in candidate_results.items()
        }
        oracle_metrics = compute_actual_oracle_or_proxy_metrics(selected, outcomes, introspection)
        direct_double_sim = (
            run_direct_libero_double_sim_probe(config, selected, evidence_path.parent)
            if config.direct_libero_double_sim
            else {"enabled": False}
        )
        if direct_double_sim.get("clone_fidelity"):
            direct_clone = CloneFidelityMetrics(**direct_double_sim["clone_fidelity"])
            if direct_clone.verdict == PASS:
                clone_verdict = PASS
                clone_record = {
                    **clone_record,
                    "state_l2": direct_clone.state_l2,
                    "image_mse": direct_clone.image_mse,
                    "image_mae": direct_clone.image_mae,
                    "success_proxy_delta": direct_clone.success_proxy_delta,
                }
                clone_rationale = direct_clone.rationale
                clone_restore_evidence = direct_double_sim.get("clone_restore_evidence", clone_restore_evidence)
        evidence = {
            "mode": "libero_actual_adapter",
            "available": True,
            "blockers": [] if clone_verdict == PASS else [clone_rationale],
            "introspection": introspection,
            "candidate_results": candidate_results,
            "outcomes": outcomes,
            "clone_restore_evidence": clone_restore_evidence,
            "direct_libero_double_sim": direct_double_sim,
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


def apply_candidate_to_env(
    env: Any,
    candidate: ActionChunkCandidate,
    seed_value: Any,
    max_steps: int,
    np_module: Any,
    *,
    reset_before: bool = True,
    initial_observation: Any = None,
    initial_info: Any = None,
) -> dict[str, Any]:
    if reset_before:
        observation, reset_info = env.reset(seed=seed_value)
        info: Any = reset_info
    else:
        observation = initial_observation
        info = initial_info if initial_info is not None else {}
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
    privileged_state_vector, privileged_state_source = extract_privileged_state_vector(env)
    privileged_success = extract_privileged_success_proxy(env)
    if privileged_success[1] != "unavailable" and success_source == "state_norm_proxy":
        success_proxy, success_source = privileged_success
    return {
        "candidate_id": candidate.candidate_id,
        "state_vector": state_vector,
        "state_source": state_source,
        "privileged_state_vector": privileged_state_vector[:64],
        "privileged_state_source": privileged_state_source,
        "image_vector": image_vector,
        "image_source": image_source,
        "success_proxy": success_proxy,
        "success_proxy_source": success_source,
        "privileged_success_proxy": privileged_success[0],
        "privileged_success_proxy_source": privileged_success[1],
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
    privileged_state_vector, privileged_state_source = extract_privileged_state_vector(env)
    privileged_success = extract_privileged_success_proxy(env)
    exact_clone_available = bool(internal.get("sim_get_state_available") and internal.get("sim_set_state_available"))
    call_privileged_available = any(
        not isinstance(call_probe.get(name), dict) or "error" not in call_probe.get(name, {})
        for name in ("get_sim_state", "get_state", "get_object_state")
    )
    privileged_state_available = bool(
        exact_clone_available
        or call_privileged_available
        or privileged_state_vector
        or privileged_success[1] != "unavailable"
    )
    return {
        "env_type": f"{type(env).__module__}.{type(env).__name__}",
        "num_envs": getattr(env, "num_envs", None),
        "call_probe": call_probe,
        "interesting_attrs": attr_names,
        "internal_sim": internal,
        "exact_state_clone_available": exact_clone_available,
        "privileged_state_available": privileged_state_available,
        "privileged_state_source": privileged_state_source,
        "privileged_state_preview": privileged_state_vector[:24],
        "privileged_success_proxy_source": privileged_success[1],
        "privileged_success_proxy": privileged_success[0],
    }


def inspect_internal_sim(env: Any) -> dict[str, Any]:
    nodes = traverse_env_object_graph(env)
    sim_candidates = [
        node
        for node in nodes
        if has_callable(node["object"], "get_state") and has_callable(node["object"], "set_state")
    ]
    interesting = [
        {
            "path": node["path"],
            "type": object_type_name(node["object"]),
            "has_sim": hasattr(node["object"], "sim"),
            "has_check_success": has_callable(node["object"], "check_success") or has_callable(node["object"], "_check_success"),
            "has_reward": has_callable(node["object"], "reward"),
        }
        for node in nodes[:80]
    ]
    if not sim_candidates:
        return {
            "root_type": object_type_name(env),
            "visited_count": len(nodes),
            "sim_get_state_available": False,
            "sim_set_state_available": False,
            "candidate_paths": interesting,
        }
    node = sim_candidates[0]
    sim = node["object"]
    return {
        "root_type": object_type_name(env),
        "visited_count": len(nodes),
        "sim_path": node["path"],
        "sim_type": object_type_name(sim),
        "sim_get_state_available": hasattr(sim, "get_state"),
        "sim_set_state_available": hasattr(sim, "set_state"),
        "sim_forward_available": has_callable(sim, "forward"),
        "candidate_paths": interesting,
    }


def traverse_env_object_graph(root: Any, *, max_depth: int = 7, max_nodes: int = 240) -> list[dict[str, Any]]:
    relation_names = (
        "envs",
        "_env",
        "env",
        "unwrapped",
        "gym",
        "venv",
        "sim",
        "model",
        "data",
        "task",
        "robots",
        "robot",
        "arena",
        "objects",
        "object",
        "obj",
    )
    queue: list[tuple[str, Any, int]] = [("root", root, 0)]
    seen: set[int] = set()
    nodes: list[dict[str, Any]] = []
    while queue and len(nodes) < max_nodes:
        path, value, depth = queue.pop(0)
        if value is None:
            continue
        obj_id = id(value)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        nodes.append({"path": path, "object": value})
        if depth >= max_depth or is_scalar_like(value):
            continue
        for child_path, child in iter_known_children(value, relation_names):
            queue.append((f"{path}.{child_path}", child, depth + 1))
    return nodes


def iter_known_children(value: Any, relation_names: tuple[str, ...]) -> list[tuple[str, Any]]:
    children: list[tuple[str, Any]] = []
    for name in relation_names:
        try:
            child = getattr(value, name)
        except Exception:  # noqa: BLE001
            continue
        children.append((name, child))
    if isinstance(value, dict):
        for key, child in list(value.items())[:30]:
            if any(token in str(key).lower() for token in ("env", "sim", "state", "object", "robot", "task")):
                children.append((f"[{key!r}]", child))
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value[:16]):
            children.append((f"[{index}]", child))
    return children


def is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, bool))


def has_callable(value: Any, name: str) -> bool:
    try:
        attr = getattr(value, name)
    except Exception:  # noqa: BLE001
        return False
    return callable(attr)


def object_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def find_sim_clone_handle(env: Any) -> dict[str, Any] | None:
    for node in traverse_env_object_graph(env):
        candidate = node["object"]
        if has_callable(candidate, "get_state") and has_callable(candidate, "set_state"):
            return {"path": node["path"], "sim": candidate}
    return None


def capture_sim_state(handle: dict[str, Any] | None) -> dict[str, Any]:
    if not handle:
        return {"captured": False, "reason": "sim_handle_unavailable"}
    sim = handle["sim"]
    try:
        state = sim.get_state()
        return {
            "captured": True,
            "path": handle["path"],
            "state": copy.deepcopy(state),
            "state_summary": summarize_value(state),
        }
    except Exception as exc:  # noqa: BLE001
        return {"captured": False, "path": handle["path"], "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}


def restore_sim_state(handle: dict[str, Any] | None, snapshot: dict[str, Any]) -> dict[str, Any]:
    if not handle:
        return {"restored": False, "reason": "sim_handle_unavailable"}
    if not snapshot.get("captured"):
        return {"restored": False, "path": handle["path"], "reason": snapshot.get("reason", "snapshot_not_captured")}
    sim = handle["sim"]
    try:
        sim.set_state(snapshot["state"])
        if has_callable(sim, "forward"):
            sim.forward()
        return {"restored": True, "path": handle["path"], "forward_called": has_callable(sim, "forward")}
    except Exception as exc:  # noqa: BLE001
        return {"restored": False, "path": handle["path"], "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}


def extract_state_vector(env: Any, observation: Any, info: Any) -> tuple[list[float], str]:
    for name in ("get_sim_state", "get_state", "get_object_state"):
        try:
            value = env.call(name)
            vector = numeric_vector(value, limit=256)
            if vector:
                return vector, f"env.call({name})"
        except Exception:  # noqa: BLE001
            continue
    privileged_vector, privileged_source = extract_privileged_state_vector(env)
    if privileged_vector:
        return privileged_vector, privileged_source
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


def extract_rgb_image_matrix(observation: Any) -> tuple[list[list[list[int]]], str]:
    if not isinstance(observation, dict):
        return [], "unavailable"
    for key, value in observation.items():
        if "image" not in str(key).lower():
            continue
        matrix = rgb_matrix_from_value(value)
        if matrix:
            return matrix, str(key)
    return [], "unavailable"


def rgb_matrix_from_value(value: Any, *, max_size: int = 128) -> list[list[list[int]]]:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and value[0]
        and isinstance(value[0][0], list)
        and value[0][0]
        and isinstance(value[0][0][0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or not value:
        return []
    rows: list[list[list[int]]] = []
    for row in value[:max_size]:
        if not isinstance(row, list):
            return []
        pixels: list[list[int]] = []
        for pixel in row[:max_size]:
            if not isinstance(pixel, list):
                return []
            channels = numeric_vector(pixel, limit=4)
            if len(channels) < 3:
                return []
            if max(channels[:3]) <= 1.0:
                channels = [channel * 255.0 for channel in channels]
            pixels.append([max(0, min(255, int(round(channel)))) for channel in channels[:3]])
        if pixels:
            rows.append(pixels)
    return rows


def write_future_image_artifact(output_dir: Path, candidate_id: str, committed: dict[str, Any]) -> Path:
    rgb_matrix = committed.get("rgb_image_matrix") or []
    if rgb_matrix:
        artifact_path = output_dir / f"direct_future_{candidate_id}.ppm"
        write_ppm_image(artifact_path, rgb_matrix)
        return artifact_path
    artifact_path = output_dir / f"direct_future_{candidate_id}.svg"
    artifact_path.write_text(
        render_image_matrix_svg(
            vector_to_image_matrix(committed["image_vector"]),
            f"direct future {candidate_id}",
        ),
        encoding="utf-8",
    )
    return artifact_path


def write_ppm_image(path: Path, rgb_matrix: list[list[list[int]]]) -> None:
    height = len(rgb_matrix)
    width = len(rgb_matrix[0]) if height else 0
    header = f"P3\n{width} {height}\n255\n"
    rows = []
    for row in rgb_matrix:
        rows.append(" ".join(f"{pixel[0]} {pixel[1]} {pixel[2]}" for pixel in row))
    path.write_text(header + "\n".join(rows) + "\n", encoding="ascii")


def extract_success_proxy(info: Any, reward: Any, state_vector: list[float]) -> tuple[float, str]:
    for key in ("is_success", "success", "task_success"):
        value = find_key(info, key)
        if value is not None:
            numbers = numeric_vector(value, limit=10)
            if numbers:
                return max(0.0, min(1.0, max(numbers))), f"info.{key}"
    privileged_score, privileged_source = extract_privileged_success_proxy_from_value(info)
    if privileged_source != "unavailable":
        return privileged_score, f"info.{privileged_source}"
    reward_values = numeric_vector(reward, limit=10)
    if reward_values:
        return max(0.0, min(1.0, max(reward_values))), "reward_clamped"
    if state_vector:
        return max(0.0, min(1.0, 1.0 / (1.0 + math.sqrt(sum(value * value for value in state_vector[:8]))))), "state_norm_proxy"
    return 0.0, "unavailable"


def extract_privileged_state_vector(env: Any) -> tuple[list[float], str]:
    for node in traverse_env_object_graph(env):
        value = node["object"]
        sim_vector = extract_sim_data_vector(value)
        if sim_vector:
            return sim_vector, f"{node['path']}.sim_data_pose"
        attr_vector = extract_privileged_attrs_vector(value)
        if attr_vector:
            return attr_vector, f"{node['path']}.privileged_attrs"
    return [], "unavailable"


def extract_sim_data_vector(value: Any) -> list[float]:
    try:
        data = getattr(value, "data")
    except Exception:  # noqa: BLE001
        return []
    vectors: list[float] = []
    for name in ("body_xpos", "site_xpos", "geom_xpos", "qpos", "qvel"):
        try:
            vectors.extend(numeric_vector(getattr(data, name), limit=128))
        except Exception:  # noqa: BLE001
            continue
        if len(vectors) >= 256:
            break
    return vectors[:256]


def extract_privileged_attrs_vector(value: Any) -> list[float]:
    vectors: list[float] = []
    for name in dir(value):
        lowered = name.lower()
        if not any(token in lowered for token in ("target", "goal", "object", "obj", "site", "body")):
            continue
        if name.startswith("__"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:  # noqa: BLE001
            continue
        if callable(attr):
            continue
        vectors.extend(numeric_vector(attr, limit=64))
        if len(vectors) >= 256:
            break
    return vectors[:256]


def extract_privileged_success_proxy(env: Any) -> tuple[float, str]:
    for node in traverse_env_object_graph(env):
        value = node["object"]
        for name in ("check_success", "_check_success", "is_success", "_is_success"):
            if not has_callable(value, name):
                continue
            try:
                score, source = extract_privileged_success_proxy_from_value(getattr(value, name)())
                if source != "unavailable":
                    return score, f"{node['path']}.{name}"
            except Exception:  # noqa: BLE001
                continue
        if has_callable(value, "reward"):
            try:
                reward = value.reward()
                numbers = numeric_vector(reward, limit=10)
                if numbers:
                    return max(0.0, min(1.0, max(numbers))), f"{node['path']}.reward"
            except Exception:  # noqa: BLE001
                continue
    return 0.0, "unavailable"


def extract_privileged_success_proxy_from_value(value: Any) -> tuple[float, str]:
    for key in ("is_success", "success", "task_success", "check_success"):
        found = find_key(value, key)
        if found is None:
            continue
        numbers = numeric_vector(found, limit=10)
        if numbers:
            return max(0.0, min(1.0, max(numbers))), key
    numbers = numeric_vector(value, limit=10)
    if numbers:
        return max(0.0, min(1.0, max(numbers))), "numeric"
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


def render_image_matrix_svg(matrix: list[list[int]], title: str) -> str:
    cell = 8
    height_cells = len(matrix)
    width_cells = len(matrix[0]) if matrix else 1
    rows = [f"<text x='8' y='18' font-size='13'>{html.escape(title)}</text>"]
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            color = max(0, min(255, int(value)))
            rows.append(
                f'<rect x="{8 + x * cell}" y="{28 + y * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({color},{color},{color})"/>'
            )
    return svg_document(16 + width_cells * cell, 38 + height_cells * cell, "".join(rows))


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
    if "direct_libero_double_sim_evidence" in artifact:
        actual_link += (
            "<h2>Direct LIBERO Double-Sim Evidence</h2>"
            f"<p><a href=\"{html.escape(artifact['direct_libero_double_sim_evidence'])}\">direct_libero_double_sim_evidence.json</a></p>"
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
