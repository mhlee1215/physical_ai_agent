from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    mode: str
    target: str
    eval_method: str
    policy_path: str
    env_type: str
    task_suite: str
    task_id: int | None
    num_candidates: int
    candidate_seeds: tuple[int, ...]
    imagination_backend: str
    judge_backend: str
    post_check_backend: str
    retry_budget: int
    output_dir: str
    dry_run: bool
    episode_seed: int
    chunk_steps: int
    action_dim: int
    policy_num_steps: int | None
    policy_n_action_steps: int | None
    instruction: str
    selector_strategy: str


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: str
    config_path: str
    execution_contract_path: str
    trace_path: str
    report_path: str
    summary_path: str
    command_path: str
    blocker_path: str
    benchmark_command_path: str
    benchmark_log_path: str
    benchmark_trace_path: str
    benchmark_eval_info_path: str
    benchmark_result_path: str


@dataclass(frozen=True)
class ExecutionContract:
    entrypoint: str
    working_dir: str
    python_bin: str
    current_command: str
    benchmark_command: str | None
    backend_command: str | None
    environment_exports: dict[str, str]
    local_output_dir: str
    remote_output_dir: str | None
    requires_linux: bool
    notes: list[str]


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    seed: int
    action_chunk: list[list[float]]
    summary: dict[str, float]
    source: str = "deterministic_seeded_chunk_generator"
    is_baseline: bool = False


@dataclass(frozen=True)
class ImaginedCandidate:
    candidate_id: str
    backend: str
    predicted_progress: float
    predicted_alignment: float
    predicted_success_proxy: float
    rationale: str


@dataclass(frozen=True)
class JudgedCandidate:
    candidate_id: str
    backend: str
    score: float
    rank: int
    rationale: str


@dataclass(frozen=True)
class SelectionDecision:
    candidate_id: str
    score: float
    rank: int
    rationale: str
    selector_strategy: str
    confidence: float
    fallback_used: bool
    baseline_candidate_available: bool
    baseline_candidate_selected: bool
    method_claim_ready: bool


@dataclass(frozen=True)
class PostCheckResult:
    backend: str
    passed: bool
    score: float
    rationale: str


@dataclass(frozen=True)
class BenchmarkResult:
    available: bool
    success: bool | None
    source: str
    rationale: str
    command: str | None
    log_path: str | None
    trace_path: str | None
    eval_info_path: str | None
    pc_success: float | None
    eval_seconds: float | None
    action_steps: int | None
    seed: int | None
    exit_code: int | None
    selected_candidate_applied: bool
    selected_candidate_id: str | None
    selected_action_shape: list[int] | None
    committed_action_steps: int
    candidate_generation_source: str | None
    baseline_candidate_available: bool
    baseline_candidate_selected: bool
    selector_strategy: str | None
    selector_confidence: float | None
    selector_fallback_used: bool
    method_claim_ready: bool


@dataclass(frozen=True)
class RunReport:
    status: str
    mode: str
    target: str
    eval_method: str
    env_type: str
    task_suite: str
    task_id: int | None
    policy_path: str
    policy_num_steps: int | None
    policy_n_action_steps: int | None
    dry_run: bool
    candidate_count: int
    selected_candidate_id: str | None
    selected_score: float | None
    baseline_candidate_available: bool
    baseline_candidate_selected: bool
    selector_strategy: str | None
    selector_confidence: float | None
    selector_fallback_used: bool
    method_claim_ready: bool
    benchmark_success_available: bool
    benchmark_success: bool | None
    execution_readiness: str
    blockers: list[str]
    notes: list[str]
    stage_backends: dict[str, str]
    artifacts: dict[str, str]
    current_command: str
    benchmark_command: str | None
    backend_command: str | None
    trace_event_count: int
    post_check_passed: bool | None
    post_check_score: float | None
    post_check_rationale: str | None
    benchmark_result: BenchmarkResult
