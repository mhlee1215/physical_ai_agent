# Project Summary

Last updated: 2026-06-11

This repository is currently being used to build and evaluate an agentic
physical-AI wrapper around lightweight vision-language-action policies. The
near-term paper target is RSS SemRob 2026.

## Paper Target

- Target venue: RSS SemRob 2026, the 3rd Workshop on Semantic Reasoning and
  Goal Understanding in Robotics.
- Website: https://semrob.github.io/
- Submission deadline: 18 June 2026, 23:59 AoE.
- Notification: 25 June 2026.
- Camera-ready: 1 July 2026, 23:59 AoE.
- Workshop date: 17 July 2026.
- Format: 4+N or 8+N pages, double-blind, non-archival workshop report.
- Paper workspace: `papers/rss2026_semrob/`.
- Main draft: `papers/rss2026_semrob/main.tex`.
- Overview figure: `papers/rss2026_semrob/figures/overview.png`.
- TODO/checklist: `papers/rss2026_semrob/manuscript_todo_checklist_2026_06_10.md`.

## Current Research Idea

The working method is **Imagine-Then-Act**: an agentic wrapper around a frozen
lightweight VLA policy.

Paper-facing method framing:

1. Read the current observation and task instruction.
2. Generate a portfolio of plausible subgoal or strategy variants with an
   external multimodal planner, primarily Qwen2.5-VL.
3. Feed each strategy prompt into the same frozen SmolVLA executor to produce
   candidate action chunks.
4. Predict or simulate likely outcomes for candidate chunks when possible.
5. Select the most promising candidate before execution.
6. Verify progress after execution and replan if progress is insufficient.

Internal management labels such as `Risk1`, `Risk1-A`, `Risk1-B`, `Risk1-C`,
`Risk2`, and `Risk5` must not appear in paper prose. Translate them into
paper-facing concepts:

- candidate generation
- template prompt portfolio
- external VLM subgoal generation
- simulation-based candidate selection
- context-capture and rendering gate
- state-restoration validation
- oracle/proxy evaluation boundary

## Current Evidence State

### Completed or Stable

- Native SmolVLA noise path was tested and reached actual inference, but
  produced weak diversity. Treat as diagnostic baseline, not method evidence.
- Template prompt portfolio produced meaningfully more diverse policy-generated
  candidates than native noise. Treat as candidate-generation evidence, not
  benchmark success.
- Direct simulator snapshot/restore and scoped LeRobot episode-start checks
  passed previously. Keep those as simulator-control evidence, not benchmark
  success.
- Qwen2.5-VL-7B model cache and model-load preflight passed on the new 60GB
  RunPod volume.
- A paper-ready overview figure has been generated and copied to
  `papers/rss2026_semrob/figures/overview.png`.
- Install surface has been simplified. The canonical install/check scripts are:
  - `scripts/install/local_install.sh`
  - `scripts/install/runpod_install.sh`
  - `scripts/install/local_check.sh`
  - `scripts/install/runpod_check.sh`

### Currently Blocked or Pending

- The paper-facing Qwen2.5-VL-7B external subgoal generation chain has not yet
  completed because actual LIBERO context capture is blocked by EGL render-device
  permissions on the tested RunPod hosts.
- SmolVLA candidate chunks from Qwen2.5-VL-7B-generated subgoals have not been
  produced yet.
- Simulation-based candidate selection over Qwen2.5-VL-7B candidates has not
  run yet.
- Benchmark/environment success for the full Imagine-Then-Act method has not
  been demonstrated yet.

Current blocker:

- Category: `CONTEXT_CAPTURE_LIBERO_BLOCKED_EGL_DEVICE_PERMISSION`.
- Symptom: the full `capture_risk1b_context.py --backend libero` path cannot
  initialize EGL because the container cannot open `/dev/dri/renderD*`.
- Important distinction: Qwen2.5-VL-7B readiness is an infrastructure PASS, but
  the external-VLM experiment itself is still blocked before generation.

Latest mitigation:

- Tech Lead pushed `80af42561fa6084981ddb6ea2eba123da0cae86b`.
- `scripts/preflight_risk1b_context_capture.py` now performs a cheap
  `/dev/dri/renderD*` read/write gate before launching the full context-capture
  path.
- Manager must reject pods with inaccessible render nodes before handoff.
- OSMesa is a limited plumbing fallback only; do not present it as benchmark or
  deployable EGL evidence.

## RunPod Infrastructure State

- Current persistent RunPod volume:
  - name: `physical_ai_volume_60`
  - id: `otq1k142hf`
  - size: 60GB
  - datacenter: `EU-RO-1`
  - mount: `/workspace`
- Old 40GB volume was deleted:
  - old name: `physical_ai_network_volume`
  - old id: `w59nxx3o43`
- Persistent paths on the new volume:
  - canonical LIBERO/SmolVLA env:
    `/workspace/physical-ai/envs/lerobot_py312`
  - Risk1-B VLM env:
    `/workspace/physical-ai/envs/risk1b_vlm_py312`
  - LIBERO config:
    `/workspace/physical-ai/libero_config/config.yaml`
  - LIBERO assets:
    `/workspace/physical-ai/libero_assets`
  - Hugging Face/model cache:
    `/workspace/physical-ai/hf_home`
  - pip cache:
    `/workspace/physical-ai/pip_cache`
- Qwen2.5-VL-7B weights fit on the 60GB volume and model-load preflight has
  passed once.
- RunPod pods must be stopped after artifact fetch or blocker. No pod should be
  left running while ownership is unclear.

## Thread Topology

New conversations should start as ordinary Codex collaboration. The
postdoc/orchestrator workflow is activated only when the user explicitly says:

- `포닥 모드로 전환해서 PM과 스레드들을 오케스트레이션해줘.`
- `포닥 모드 켜줘.`

After that trigger, the user mainly talks to the postdoc/orchestrator thread.
The orchestrator should proactively delegate work to specialist threads when a
task crosses role boundaries.

Known thread roles:

- Postdoc/orchestrator: this conversation. Owns high-level decisions, cold
  judgment, PM steering, paper claim boundaries, and user-facing synthesis.
- PM/watchdog: `019eaf36-bb40-7182-8c92-a40550455ca3`.
  Owns checklists, owner tracking, nudge loops, worker report collection, and
  final status boards. All specialist threads report task completion/blockers
  to PM; PM reports closed or blocked tasks back to the postdoc.
- Tech Lead: `019eab52-10fb-78f0-923d-678ab82cc70f`.
  Owns repo-backed implementation, tests, harness/code patches, and diagnostics.
  Reports commits, tests, and blockers to PM.
- RunPod Manager: `019eaf4d-4157-7f51-9099-561b7c3a9c1e`.
  Owns RunPod allocation, source staging, install/bootstrap, env gates, handoff,
  stop, and no-RUNNING confirmation. Reports allocation/handoff/cleanup status
  to PM.
- RunPod Researcher: `019eab62-ca0b-7770-ad27-5d95f66ebd62`.
  Runs approved experiments only after manager handoff, fetches artifacts, and
  reports commands, artifacts, and verdicts to PM.
- Evaluation Results Manager: `019eb3e5-a8fa-7d01-b1bd-ee52d73319cc`.
  Classifies paper-facing evaluation result packages as paper-ready,
  diagnostic-only, or blocked and reports classifications to PM. This manager
  is not a general inbox for RunPod allocation, install, cache, volume, or
  cleanup status.
- Paper-writing thread: `019e9e01-32c0-7641-92b7-0f6c23800122`.
  Owns the manuscript draft and separate TODO/checklist. Reports completed
  sections and unresolved TODOs to PM.
- Related works thread: `019eaa3f-c52b-7d43-ae62-da4fa5532ec6`.
  Supplies citation clusters and overlap analysis. Reports paper-ready synthesis
  and overlap risks to PM.

## Operating Rules

- Do not run RunPod install/allocation from researcher threads. Researchers must
  wait for a manager handoff.
- RunPod Manager must use GitHub remote source staging only unless the user
  separately approves local workspace upload.
- RunPod Manager must use the new 60GB volume `otq1k142hf`.
- RunPod Manager must stage the relevant pushed commit exactly and report the
  source path plus `.source_commit`.
- Install/check commands should use the canonical `scripts/install/*` entrypoints.
- When RunPod capacity is the only blocker and the user/PM says to continue,
  RunPod Manager remains the owner until a usable instance is secured or the
  user/PM cancels. Retries run in repeated 10-attempt batches with 20-30 minute
  sleep/backoff, no idle Pod while sleeping, and a PM report after each attempt
  and batch. Capacity shortage is a polling state, not a terminal experiment
  blocker.
- Paper-facing evaluation result packages must be reported to PM and Evaluation
  Results Manager by the RunPod Researcher or another result reviewer after
  artifacts/metrics exist. Routine infra status goes to PM only.
- Paper drafts must not contain internal management labels, blockers, TODOs, or
  status-board language. TODOs belong in a separate checklist.
- Mock, fixture, proxy-only, debug, OSMesa fallback, or plumbing evidence must
  not be promoted to benchmark success.
- Treat development logs and dated intermediate artifacts as legacy unless this
  summary or the Evaluation Results Manager marks them current.

## Main vs Legacy

Current state is managed through:

- `Summary.md`
- `AGENTS.md`
- `docs/harness/physical-ai/team-spec.md`
- `papers/rss2026_semrob/main.tex`
- `papers/rss2026_semrob/manuscript_todo_checklist_2026_06_10.md`

Legacy by default:

- dated research notes under `docs/research/`
- old paper outlines, progress ledgers, handoff reports, and HTML reports
- old RunPod worklogs and setup-recovery notes
- old `_workspace/` checkpoint or run artifacts
- prior 40GB network-volume notes and quota history

Legacy artifacts are still useful for provenance, but they must not define the
current plan. If a legacy result matters for the paper, the Evaluation Results
Manager must explicitly classify it before the paper-writing thread uses it.
New development logs should stay out of the main state surface. Put run
artifacts under `_workspace/`, historical notes under `docs/legacy/` or dated
archive locations, and promote only distilled current facts back into this
summary.

## Immediate Next Steps

1. Continue searching for an EGL-capable RunPod host/runtime where
   `scripts/preflight_risk1b_context_capture.py` passes.
2. Only after that context-capture preflight passes, hand off to the Researcher
   for the Qwen2.5-VL-7B subgoal generation -> SmolVLA candidate chunks ->
   selector/probe chain.
3. Keep Qwen2.5-VL-3B as a lighter fallback/baseline if 7B execution remains
   impractical, but label it separately from 7B evidence.
4. Paper writing should continue with conservative claims: describe the method
   clearly, use the overview figure, and avoid claiming final benchmark gains
   until the evaluation manager marks evidence paper-ready.
