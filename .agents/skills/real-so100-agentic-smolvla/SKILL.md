---
name: real-so100-agentic-smolvla
description: Use for real SO-100 follower hardware experiments with SmolVLA and an agentic layer, including camera routing, policy/observer separation, metadata-gated action execution, Codex observer feedback loops, and checkpoint-26 real robot evidence.
---

# Real SO-100 Agentic SmolVLA

## When to Use

Use this skill whenever a task touches the real SO-100 follower arm, real
cameras, green-object/grasp experiments, SmolVLA action chunks, or the
agentic-layer improvement loop around real robot attempts.

Before touching hardware or changing real SO-100 code, read:

```text
.agents/skills/real-so100-agentic-smolvla/references/hardware_contract.md
```

When comparing the real robot path against validated SmolVLA evaluation, also
read:

```text
docs/research/smolvla_baseline_handoff_2026_06_07.md
docs/real_so100_smolvla_execution_comparison_2026_06_07.md
```

## Core Contract

- Build the agentic layer around SmolVLA; do not drift into dashboards unless
  they directly support robot iteration evidence.
- Policy cameras are Innomaker U20CAM indexes `0` and `1`.
- iPhone camera index `3` is Codex observer/debug evidence only.
- Do not feed camera `3` to SmolVLA.
- If camera `3` is temporarily off, continue only no-actuation agentic-layer
  development with cameras `0` and `1`; use camera `1` as the wide context
  feedback source and keep physical execution / final task-success claims
  blocked until observer evidence returns.
- Do not use the operator camera or legacy camera `2` for the current policy
  loop unless a historical artifact explicitly requires it.
- In-loop prompts target the in-loop agent or SmolVLA, not the human operator
  and not Codex.
- Semantic decisions must come from an LLM/VLM layer. Codex may act as a
  Pseudo-LLM during development, but the final runtime must be replaceable by
  an on-device lightweight LLM/VLM.
- Treat `move object right` as an object-frame task. Do not hard-code it as a
  fixed robot-arm direction or joint sign.
- Use SmolVLA action chunks, normally 10 steps at a time. Do not execute only a
  single isolated `select_action()` output for the real loop.
- After every robot-arm movement task, return the arm to the user-defined
  canonical home pose, then disable torque on all SO-100 motors before handing
  control back to the user. Execution reports must include the home-return
  artifact and `post_task_torque_disabled`.
- Do not disable torque at the end of an intermediate policy/action chunk when
  a home-return step still needs to run. Policy execution primitives must keep
  torque on for the immediate home-return handoff, and only the final
  home-return/recovery step may disable torque.
- Calibration is performed by the user/operator. Codex may launch the
  calibration script, verify outputs, and record metadata, but must not claim
  that Codex physically performed calibration.

## Safety and Execution Gate

Real physical execution is blocked unless all are true:

- workspace is physically clear and the user has confirmed execution;
- emergency stop / power / serial / calibration state is known;
- camera `3` before/during/after evidence will be recorded;
- SmolVLA output has passed the same kind of processor/postprocessor contract
  used by validated LeRobot evaluation, or an equivalent verified
  unnormalization path exists;
- SO-100 follower joint order is confirmed;
- gripper open/close semantics are confirmed;
- final motor targets are clipped inside `_workspace/real_so100/calibration/so100_local.json`;
- the command report will preserve readbacks, targets, videos, blockers, and
  post-task torque-off status.

Do not interpret SmolVLA outputs with arbitrary raw tick scaling. Earlier
`raw_action * N ticks` attempts are historical motor-communication evidence
only, not a valid SmolVLA execution method.

## Standard Loop

1. Capture camera `0` and `1` policy evidence; capture camera `3` observer
   evidence when it is available.
2. Build the policy input only from camera `0`, camera `1`, state, and the
   in-loop prompt.
3. Run SmolVLA through the correct LeRobot-compatible pre/postprocessing path.
4. If action execution is blocked, write a blocker artifact and improve the
   agentic layer rather than forcing motion.
5. If camera `3` is unavailable, stop before physical execution and use camera
   `1` context plus intermediate policy-camera data to write Pseudo-LLM
   feedback.
6. If execution is allowed and camera `3` is available, record camera `3`
   before/during/after, execute the 10-step chunk, and save readbacks plus
   visual evidence.
7. At the end of the movement task, return to the canonical home pose, then
   disable torque on all SO-100 motors and record the result in the execution
   report.
8. Produce the next agentic-layer version and repeat.
9. Report policy execution, verifier progress, grasp outcome, object relocation,
   and final task success separately.

## Required Artifacts

- Observation episode JSONL with camera roles and calibration paths.
- SmolVLA proposal/action chunk report.
- Metadata/processor/action-semantics report before any hardware execution.
- Execution report with `send_action_called`, `policy_actions_executed`,
  per-step targets, before/after readbacks, and blockers.
- Camera `3` observer video and before/after frames for every physical motion.
- Feedback / next-plan record under `_workspace/real_so100/reports/`.
- Task-level verifier output for transport goals, especially object relocation
  in the observer image frame.
- Home-return report plus `post_task_torque_disabled=true` for every completed
  movement task, or an explicit home-return / torque-off failure record and
  immediate follow-up recovery.

## Validation Commands

Run the narrow tests relevant to the touched path, typically:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_smolvla_metadata_adapter.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execute_chunk.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_iteration.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_vla_prompt_packet.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_relocation_verifier_packet.py'
```

For hardware-visible claims, command success is not enough. Inspect the camera
`3` video/frames and state what was visually inspected.
