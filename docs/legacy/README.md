# Legacy Materials

This directory documents the project's legacy classification policy.

## Meaning of Legacy

Legacy does not mean useless or deleted. It means the artifact is historical
evidence, an old development log, an intermediate experiment note, or a past
handoff that should not be treated as the current operating state.

Use legacy material only to recover provenance, exact commands, old blocker
details, or prior evidence. Do not use it as the source of truth for the current
paper, RunPod setup, thread topology, or next action.

## Current Sources of Truth

Use these files first:

- `Summary.md`
- `AGENTS.md`
- `docs/harness/physical-ai/team-spec.md`
- `papers/rss2026_semrob/main.tex`
- `papers/rss2026_semrob/manuscript_todo_checklist_2026_06_10.md`

## Legacy by Default

The following classes are legacy unless `Summary.md` or the PM explicitly marks
one item current:

- dated research notes under `docs/research/`
- old experiment matrices, handoff reports, HTML reports, and summary JSON files
- historical RunPod worklogs and setup-recovery notes
- `_workspace/` checkpoint artifacts and fetched run artifacts from prior runs
- old paper outlines, ledgers, and progress logs
- previous 40GB RunPod volume notes

## Promotion Rule

A legacy result may become paper-facing evidence only after the Evaluation
Results Manager classifies it as paper-ready. Otherwise it remains diagnostic,
blocked, or historical context.

