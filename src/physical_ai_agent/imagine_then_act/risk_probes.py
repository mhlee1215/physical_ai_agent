from __future__ import annotations

import html
import json
import math
import random
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


def run_risk_probes(config: RiskProbeConfig) -> RiskProbeReport:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_mock_candidates(config)
    outcomes = {candidate.candidate_id: simulate_mock_env(candidate.action_chunk) for candidate in candidates}
    diversity = compute_diversity_metrics(config, candidates)
    clone_fidelity = compute_clone_fidelity_metrics(config, candidates[-1])
    oracle_upper_bound = compute_oracle_upper_bound_metrics(candidates, outcomes)
    artifacts = write_visual_artifacts(output_dir, candidates, outcomes, diversity, clone_fidelity, oracle_upper_bound)
    blockers: list[str] = []
    if config.backend != "mock":
        blockers.append(
            "LIBERO simulator/oracle adapter is not attached locally; RunPod smoke should wire clone state/image capture before benchmark claims."
        )
    risk_verdicts = {
        "risk_1_candidate_diversity": diversity.verdict,
        "risk_2_clone_fidelity": clone_fidelity.verdict if not blockers else BLOCKED,
        "risk_5_oracle_selector_upper_bound": oracle_upper_bound.verdict if not blockers else BLOCKED,
    }
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
</body>
</html>
"""
