# Imagine-Then-Act Chunk Selection Positioning

Date: 2026-06-08

## Working Title

- Imagine-Then-Act Chunk Selection for Lightweight Vision-Language-Action Policies
- VLM-Guided Imagined Rollout Selection for Lightweight VLA Policies

## Core Idea

A lightweight VLA policy, such as SmolVLA, samples multiple stochastic action
chunk candidates from the same current observation and subgoal. Before any
candidate is committed to the benchmark environment, the system imagines each
candidate's visual outcome. A VLM or visual judge scores the predicted outcome
image against the current subgoal, and the system executes only the selected
chunk.

The first proof can use a simulator clone as oracle imagination:

1. clone the simulator state;
2. roll out each candidate chunk in the clone;
3. render each candidate outcome image;
4. score the outcome image against the subgoal;
5. commit only the selected chunk to the real evaluation environment.

This should be described as an upper-bound diagnostic, not as a deployable real
robot method. The deployable follow-on is to learn an action-conditioned visual
dynamics or video prediction model from simulation data.

## Claim Boundary

Use the VLM judge only as a pre-execution candidate selector. Do not use it as
the final success detector.

Final reported success must remain the benchmark/environment success signal,
such as `info["success"]`, `is_success`, or environment-specific
`pc_success`.

## Why This Is Stronger Than Retry

The previous retry-wrapper direction produced useful controls, but blind retry
was competitive. Imagine-then-act moves the intervention earlier:

- retry asks "what should we do after failure?";
- imagine-then-act asks "which candidate chunk should we execute before failure
  is committed?";
- visual candidate selection can be compared against random, oracle-state, and
  VLM-based scoring under the same action budget.

## Related Work Sorting Plan

When the related-work search returns, sort papers into these buckets:

1. Visual MPC / visual foresight / video prediction.
2. World models and imagination-based planning.
3. Best-of-N / CEM / candidate trajectory selection.
4. VLM-as-critic / reward / verifier in robotics.
5. VLA and lightweight VLA background.
6. Agentic LLM concepts only where directly relevant.

The paper should read primarily as model-based candidate selection and visual
foresight for lightweight VLA control. Agentic LLM concepts are supporting
context only if they clarify subgoals, judging, or test-time selection.

## Minimal Experiment Matrix

| Condition | Imagination source | Selector | Committed chunk budget | Final success source |
| --- | --- | --- | --- | --- |
| Policy-only | none | base policy first chunk | 1 chunk | environment |
| Random chunk | simulator clone | random candidate | 1 selected chunk | environment |
| Oracle imagined selector | simulator clone | privileged outcome rule | 1 selected chunk | environment |
| VLM imagined selector | simulator clone render | VLM/visual judge | 1 selected chunk | environment |
| Learned imagined selector | learned video/dynamics model | VLM/visual judge | 1 selected chunk | environment |

The first SemRob submission can focus on the first four rows. The learned model
row is the future-work bridge.

## Metrics

- Benchmark success rate.
- Success per committed action step.
- Wall-clock latency.
- Candidate count.
- Imagination rollout cost.
- VLM/rule selector agreement with final environment outcome.
- Failure categories where selection chose a visually plausible but
  environment-failing chunk.

## Paper Safety Notes

- Do not claim real-robot readiness from simulator-clone imagination.
- Do not hide extra compute: report candidate count and imagination budget.
- Do not compare against policy-only as if the selector uses no extra test-time
  compute.
- Do compare under fixed committed action budget.
- Do keep `policy_only`, random chunk, and oracle selector baselines; they make
  the VLM contribution interpretable.
