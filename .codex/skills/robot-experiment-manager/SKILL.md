---
name: robot-experiment-manager
description: Use when working on the physical_ai_agent Robot Experiment Manager dashboard, including dataset viewer additions, dataset registration/discovery, SO101 or MyCobot platform filtering, training run/checkpoint display, closed-loop test analysis, loop-test media, or interactive policy rollout UI and API wiring.
---

# Robot Experiment Manager

Use this skill when changing or inspecting the unified dashboard served by
`scripts/serve_so101_dataset_viewer.py`. The dashboard is intentionally broader
than SO101: keep the product name and header generic enough for SO101, MyCobot,
and future robot platforms.

## Core Rules

- Keep one web surface: `Robot Experiment Manager`.
- Keep the visible dashboard tabs aligned with their backing contracts:
  `Data Viewer`, `Training Manager`, `Loop Test Analyzer`, and
  `Interactive Simulator`.
- Do not make display-only fixes for camera labels, prompts, object metadata,
  or loop-test starts. Verify the API payload and source artifact are consistent.
- Do not delete or rewrite raw datasets under `_workspace/` unless the user
  explicitly asks for that exact cleanup.
- Keep raw datasets, checkpoints, rollout media, TensorBoard logs, and other
  artifacts out of PRs.
- When a loop-test GIF is checked into a PR or research note as review evidence,
  generate it with the same TensorBoard closed-loop media renderer used for
  `closed_loop/<test_id>/rollout_camera1_camera2_episode_*`. Do not copy the raw
  per-episode rollout GIF directly if TensorBoard shows a side-by-side labeled
  camera view.
- Prefer config/contract registration for repeatable datasets and tests. Use
  env discovery only for temporary local POC inspection.

## Quick Start

Run the dashboard through `launchctl`. Do not use `nohup ... &` for this
server; Codex command cleanup can reap that child process even when the app did
not crash.

```bash
sh scripts/launch_so101_dataset_viewer.sh restart
```

Check the current server:

```bash
lsof -nP -iTCP:8768 -sTCP:LISTEN || true
curl -s http://127.0.0.1:8768/api/datasets | python3 -m json.tool >/dev/null
```

Lifecycle commands:

```bash
sh scripts/launch_so101_dataset_viewer.sh status
sh scripts/launch_so101_dataset_viewer.sh stop
sh scripts/launch_so101_dataset_viewer.sh restart
```

The LaunchAgent label is `com.physical-ai-agent.dataset-viewer`; logs are
written to `_workspace/logs/dataset_viewer_8768.launchd.log` and
`_workspace/logs/dataset_viewer_8768.launchd.err.log`.

Validate edits before reporting completion:

```bash
python3 -m py_compile scripts/serve_so101_dataset_viewer.py scripts/serve_loop_test_analyzer.py
git diff --check -- scripts/serve_so101_dataset_viewer.py scripts/serve_loop_test_analyzer.py
```

## What To Read

Read `references/experiment-manager.md` before making non-trivial changes to:

- dataset registration or platform filtering
- SO101/MyCobot dataset preview support
- training run or checkpoint discovery
- closed-loop test case configuration or analyzer export wiring
- interactive simulator presets, prompts, continuation, or media rendering
