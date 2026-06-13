#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from physical_ai_agent.imagine_then_act.utils import (
    build_execution_contract,
    build_run_config,
    build_run_report,
    evaluate_execution_readiness,
    execute_real_backend,
    generate_candidate_chunks,
    imagine_candidates,
    judge_candidates,
    prepare_run_artifacts,
    run_post_check,
    select_candidate,
    should_execute_real_backend,
    trace_event,
    write_config_snapshot,
    write_execution_contract,
    write_run_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entrypoint for Imagine-Then-Act chunk-selection experiments.")
    parser.add_argument("--mode", choices=("smoke", "local-dry-run", "libero", "runpod-libero"), default="smoke")
    parser.add_argument("--target", choices=("local", "runpod"), default="local")
    parser.add_argument(
        "--eval-method",
        choices=("policy_only", "ita_baseline_fallback"),
        default="ita_baseline_fallback",
        help="Backend evaluation method. policy_only omits ITA action injection; ita_baseline_fallback preserves baseline candidate selection.",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--env-type", default=None)
    parser.add_argument("--task-suite", default=None)
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--candidate-seeds", default=None)
    parser.add_argument("--imagination-backend", choices=("none", "sim-rollout", "learned-placeholder"), default="sim-rollout")
    parser.add_argument("--judge-backend", choices=("heuristic", "vlm-placeholder", "oracle-state-placeholder"), default="heuristic")
    parser.add_argument("--post-check-backend", choices=("none", "heuristic", "vlm-placeholder", "oracle-state-placeholder"), default="heuristic")
    parser.add_argument("--retry-budget", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force contract-only execution for LIBERO/RunPod modes before the real backend is attached.",
    )
    parser.add_argument("--episode-seed", type=int, default=1200)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--policy-num-steps",
        type=int,
        default=None,
        help="Optional fair-comparison pass-through to the backend as --policy.num_steps=<N>.",
    )
    parser.add_argument(
        "--policy-n-action-steps",
        type=int,
        default=None,
        help="Optional fair-comparison pass-through to the backend as --policy.n_action_steps=<N>.",
    )
    parser.add_argument("--instruction", default="Move the target object toward the receptacle without overcommitting the chunk.")
    parser.add_argument(
        "--selector-strategy",
        "--ita-selector-strategy",
        dest="selector_strategy",
        choices=("baseline_fallback", "progress_proxy_or_baseline", "debug_min_action_norm"),
        default="baseline_fallback",
        help=(
            "Candidate selector. baseline_fallback preserves policy-only behavior; "
            "progress_proxy_or_baseline only switches when an observation progress proxy proves improvement. "
            "--ita-selector-strategy is accepted as a backward-compatible alias."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Resolve and validate the shared experiment configuration.
    # ------------------------------------------------------------------
    try:
        config = build_run_config(args)
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    artifacts = prepare_run_artifacts(config)
    contract = build_execution_contract(config)
    write_config_snapshot(config, artifacts)
    write_execution_contract(contract, artifacts)

    # ------------------------------------------------------------------
    # 2. Materialize the candidate-generation and imagination stages.
    # ------------------------------------------------------------------
    trace_events = [
        trace_event(
            "config",
            {
                "mode": config.mode,
                "target": config.target,
                "eval_method": config.eval_method,
                "output_dir": config.output_dir,
                "policy_num_steps": config.policy_num_steps,
                "policy_n_action_steps": config.policy_n_action_steps,
            },
        ),
        trace_event("execution_contract", {"benchmark_command": contract.benchmark_command}),
    ]
    candidates = generate_candidate_chunks(config)
    imagined_candidates = imagine_candidates(config, candidates)
    trace_events.append(trace_event("candidate_generation", {"count": len(candidates), "candidate_ids": [item.candidate_id for item in candidates]}))
    trace_events.append(
        trace_event(
            "imagination",
            {
                "backend": config.imagination_backend,
                "predicted_success_proxy": {
                    item.candidate_id: item.predicted_success_proxy for item in imagined_candidates
                },
            },
        )
    )

    # ------------------------------------------------------------------
    # 3. Run the judge, selection, and post-check stages from the same
    #    entrypoint so every experiment shares one interface contract.
    # ------------------------------------------------------------------
    judged_candidates = judge_candidates(config, candidates, imagined_candidates)
    selection = select_candidate(
        judged_candidates,
        candidates=candidates,
        selector_strategy=config.selector_strategy,
    )
    post_check = run_post_check(config, selection, judged_candidates)
    trace_events.append(
        trace_event(
            "judge",
            {"backend": config.judge_backend, "ranking": {item.candidate_id: item.rank for item in judged_candidates}},
        )
    )
    trace_events.append(
        trace_event(
            "selection",
            {
                "candidate_id": selection.candidate_id,
                "score": selection.score,
                "rationale": selection.rationale,
                "selector_strategy": selection.selector_strategy,
                "selector_confidence": selection.confidence,
                "selector_fallback_used": selection.fallback_used,
                "baseline_candidate_available": selection.baseline_candidate_available,
                "baseline_candidate_selected": selection.baseline_candidate_selected,
                "method_claim_ready": selection.method_claim_ready,
            },
        )
    )
    trace_events.append(
        trace_event(
            "post_check",
            {"backend": post_check.backend, "passed": post_check.passed, "score": post_check.score},
        )
    )

    # ------------------------------------------------------------------
    # 4. Finalize the execution-readiness report, persist traces, and
    #    emit a PR/researcher-friendly artifact bundle under _workspace/.
    # ------------------------------------------------------------------
    execution_readiness, blockers, readiness_notes = evaluate_execution_readiness(config)
    trace_events.append(trace_event("execution_readiness", {"state": execution_readiness, "blockers": blockers}))
    benchmark_result = None
    if should_execute_real_backend(config, blockers):
        benchmark_result = execute_real_backend(config, artifacts)
        trace_events.append(
            trace_event(
                "benchmark_execution",
                {
                    "available": benchmark_result.available,
                    "success": benchmark_result.success,
                    "source": benchmark_result.source,
                    "exit_code": benchmark_result.exit_code,
                    "pc_success": benchmark_result.pc_success,
                },
            )
        )
    report = build_run_report(
        config=config,
        artifacts=artifacts,
        contract=contract,
        selected_candidate=selection,
        post_check=post_check,
        trace_events=trace_events,
        blockers=blockers,
        notes=readiness_notes + contract.notes,
        benchmark_result=benchmark_result,
    )
    write_run_outputs(artifacts, trace_events, report)

    if args.json:
        print(json.dumps({
            "status": report.status,
            "report_path": artifacts.report_path,
            "summary_path": artifacts.summary_path,
            "eval_method": report.eval_method,
            "selected_candidate_id": report.selected_candidate_id,
            "baseline_candidate_available": report.baseline_candidate_available,
            "baseline_candidate_selected": report.baseline_candidate_selected,
            "selector_strategy": report.selector_strategy,
            "selector_fallback_used": report.selector_fallback_used,
            "method_claim_ready": report.method_claim_ready,
            "policy_num_steps": report.policy_num_steps,
            "policy_n_action_steps": report.policy_n_action_steps,
        }, indent=2, sort_keys=True))
    else:
        print(f"status={report.status}")
        print(f"output_dir={artifacts.output_dir}")
        print(f"selected_candidate={report.selected_candidate_id}")
        print(f"report={artifacts.report_path}")
        print(f"summary={artifacts.summary_path}")
    return 0 if report.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
