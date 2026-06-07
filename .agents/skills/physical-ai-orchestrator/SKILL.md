---
name: physical-ai-orchestrator
description: Orchestrate checkpoint-driven development for the agentic physical AI simulation stack, including mandatory executable verification steps.
---

# Physical AI Orchestrator

## When to Use

Use this skill when implementing or reviewing milestones for the agentic physical AI project, especially LIBERO, LeRobot, policy evaluation, planner/verifier/retry, or local VLM checkpoints.

## Required Inputs

- Current checkpoint from `docs/agentic_physical_ai_plan.md`
- Team spec: `docs/harness/physical-ai/team-spec.md`
- Existing code/config under `src/physical_ai_agent/` and `configs/`

## Fast Context Lookup

When the task mentions SmolVLA baseline execution, LIBERO baseline parity, or
"how did we run SmolVLA?", open this handoff first:

```text
docs/research/smolvla_baseline_handoff_2026_06_07.md
```

That file records the frozen focused baseline, RunPod environment variables,
required camera mapping, exact `lerobot/smolvla_libero` command shape, current
`libero_goal` task 6 seed results, and the wrapper comparison commands.

Short anchor:

```bash
cd /workspace/physical-ai/physical_ai_agent
PY=/root/physical-ai/envs/lerobot_py312/bin/python

$PY scripts/run_libero_in_episode_smolvla_instrumented.py \
  --trace-path "$OUT/in_episode_trace.jsonl" \
  --trigger-mode semantic_no_progress \
  --intervention-mode none \
  --semantic-min-step 220 \
  --semantic-window 20 \
  --semantic-progress-threshold 0.002 \
  --output_dir="$OUT/eval_logs" \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[6]" \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --policy.empty_cameras=0 \
  --seed=1200
```

Do not change this focused baseline unless the user explicitly asks. Use it as
the fixed policy-only reference while improving or comparing agentic wrappers.

## Workflow

1. Select the smallest unchecked checkpoint that moves the MVP forward.
2. Implement only the code, config, docs, and tests needed for that checkpoint.
3. Register or update the executable verification command in the team spec when the checkpoint adds a new runnable path.
4. Run the required verification command before marking the checkpoint complete.
5. Report whether the checkpoint is passed, failed, or blocked by missing external dependencies.

## Image Artifact Verification

When a checkpoint or milestone produces images, videos, overlays, contact
sheets, GIFs, plots, or visual reports, do not call it passed from file
existence, JSON metrics, or command exit status alone.

Required behavior:

- Visually inspect representative artifacts before claiming success.
- State what was inspected in the response.
- If the visual artifact contradicts the claim, report failed or blocked even
  when the manifest says `passed`.
- For projection/overlay work, treat wrong-object, wrong-side, vertical mirror,
  center-bias, and visually implausible points as bugs, not acceptable evidence.
- Paper-facing visual evidence requires both same-timestep metadata provenance
  and human-visible semantic alignment.

## Checkpoint 01 Required Verification

Always run:

```bash
sh scripts/checkpoint_01.sh
```

When claiming LIBERO itself is executable, also run:

```bash
sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env
```

When claiming checkpoint 01 works on the target Mac, run:

```bash
sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco
```

If the Mac-local command fails because MuJoCo is missing, treat checkpoint 01 as not fully complete. If the LIBERO strict command fails on macOS because LIBERO/LeRobot requires Linux, treat that as a future Linux/cloud blocker rather than a Mac-local checkpoint failure.

## Expected Outputs

- Updated repo files
- Verification command results
- Clear next blocker or next checkpoint

## Validation Notes

- Do not claim Mac-local simulation readiness from import-free tests alone.
- Do not install or download simulation dependencies without user approval.
- Keep validation commands deterministic and repo-local.
