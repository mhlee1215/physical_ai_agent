# Project Summary

Last updated: 2026-06-16

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

The immediate collaboration goal is to produce manuscript-table experiment data
as quickly and efficiently as possible. Research and orchestration choices should
favor real task rows, metrics, videos, summaries, and artifact bundles over
additional infrastructure diagnostics unless those diagnostics are the shortest
path to unblock a data-producing run.

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

### Active SO101 SmolVLA Fine-Tuning Lane

- RunPod SO101 SmolVLA fine-tuning is being monitored through the training
  dashboard and JSONL metrics under the active run directory.
- Checkpoint retention policy for this lane is validation-best checkpoint plus
  one latest checkpoint for crash recovery. Do not keep every checkpoint unless
  the user explicitly asks for archival storage.
- Supervised validation loss may run after checkpoint save on CPU. Closed-loop
  evaluation is intentionally sparse: run it only when a checkpoint becomes the
  new validation-best candidate, or for explicit manual checks, and avoid
  overlapping full CUDA closed-loop rollouts with active GPU training.
- Current observed best validation checkpoint is `001490` (`val_loss=0.083125`).
  Checkpoint `002682` saved and validated but did not improve
  (`val_loss=0.139522`), so training was stopped as likely overfitting.
- Final current-best closed-loop evaluation
  `final_best_001490_cuda_ep24_step160` completed on CUDA with
  `success_rate=0.0`, `grasp_rate=0.041667`, over 24 episodes. The best
  checkpoint was downloaded locally to
  `_workspace/so101_smolvla_pure/runpod5_best_checkpoints/001490_pretrained_model`;
  final rollout artifacts were downloaded to
  `_workspace/so101_smolvla_pure/runpod5_final_evals/final_best_001490_cuda_ep24_step160`.
- The next SO101 SmolVLA run must start from the checked pipeline contract in
  `docs/so101_smolvla_training_pipeline.md`: 256x256 SmolVLA inputs,
  512-padding preprocessing, sample-time CUDA/MPS augmentation, doubled
  recovery-aware train data, dataset manifest validation, and monitor-managed
  validation/closed-loop/overfit stopping.
- RunPod experiment-data storage policy: past remote experiment results are not
  required as the system of record. For all future RunPod data-generation,
  training, evaluation, and closed-loop runs, download the dataset/result bundle
  to the local repo at run completion, verify the local manifest or metrics,
  then delete the completed remote RunPod artifact directory. Keep reusable
  Python environments on the Pod's local disk when practical; use the network
  volume only for active handoff/cache, not long-term experiment-data storage.

### What We Have Learned

- A plain frozen SmolVLA policy is not enough by itself for the paper story; it
  gives us a useful baseline but weak candidate diversity.
- Template-based strategy portfolios can create more diverse SmolVLA action
  chunks. This supports the idea that language-conditioned variation can widen
  the action space of a frozen policy.
- External VLM-generated strategy variants are feasible, but still inconsistent:
  some tasks produce usable variants and policy-generated chunks, while others
  fail schema, grounding, or task-relation checks.
- Simulation-based candidate selection is partially implemented, but current
  selector evidence is still diagnostic. It is not yet strong enough to claim a
  reliable selector improvement.
- The system can now run paper-shaped diagnostic sweeps and produce comparison
  tables, videos, summaries, and artifact bundles. The current evidence is
  useful for deciding the next research direction, not yet for claiming benchmark
  gains.

### Current Claim Boundary

- Do claim: we are testing whether external strategy generation plus frozen VLA
  execution can produce a useful set of candidate first-action chunks.
- Do claim: diagnostic runs show modest candidate diversity improvements on some
  tasks.
- Do not claim yet: benchmark success, deployment-ready rendering, reliable
  selector improvement, or full-task improvement.
- Do not describe internal blockers or management labels in the manuscript.

### Current Research Direction

- The top priority is fast, efficient production of experiment data that can
  support or rule out paper-table claims.
- Prioritize real experimental rows for the paper table: task coverage, candidate
  validity, action-chunk diversity, selector behavior, and full-task or smoke
  outcomes when available.
- Stop spending research time on repeated environment or smoke diagnostics unless
  they are the smallest necessary step before a real data-producing run.
- Use the current shallow-rendering route as a diagnostic data-production lane.
  Treat it as useful for research iteration, not as deployment or benchmark
  renderer evidence.
- Keep full EGL/deployment evidence separate. It can be revisited later, but it
  is not the active path for generating the next paper table.

## Collaboration Model

This summary is for collaborators who need research context, not operational
details. Low-level orchestration rules, thread IDs, install commands, cloud
volume paths, and RunPod lifecycle policy live in
`docs/harness/physical-ai/team-spec.md`.

Working collaboration assumptions:

- New conversations start as ordinary Codex collaboration unless the user
  explicitly asks for postdoc/orchestrator mode.
- The postdoc/orchestrator owns high-level research judgment, claim boundaries,
  and user-facing synthesis.
- PM owns coordination and concise status tracking.
- Tech Lead owns code fixes and tests.
- RunPod Manager owns cloud resource setup and handoff.
- RunPod Researcher owns experiment execution and artifact-backed result reports.
- Evaluation Results Manager classifies final evaluation packages before they
  are used as paper evidence.

Research mode means an active cloud experiment environment should be preserved
for researcher/tech-lead debugging unless the user or PM explicitly requests
cleanup or an urgent cost/security risk appears.

Any future orchestration policy change must be recorded in repo docs, not only
in chat. Use this file for collaborator-facing current state and
`docs/harness/physical-ai/team-spec.md` for the detailed operating rules.

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

1. Continue the shallow-OSMesa paper-data lane: run actual manifest rows that
   produce Qwen variants, SmolVLA action chunks, selector outputs, metrics, and
   table-ready artifacts.
2. Do not reopen COMMUNITY/EGL render-device hunting unless the user explicitly
   asks for EGL/deployment evidence.
3. If a run blocks, assign the concrete blocker to Tech Lead or RunPod Manager,
   then resume data production after the fix; do not replace the experiment with
   more smoke diagnostics.
4. Paper writing should continue with conservative claims: describe the method
   clearly, use the overview figure, and avoid claiming final benchmark gains
   until the evaluation manager marks evidence paper-ready.
