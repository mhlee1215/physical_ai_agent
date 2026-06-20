# Imagine-Then-Act Chunk Selection Positioning

Date: 2026-06-08

## Evidence Governance

Evaluation Results Manager is active:

- Thread: `019eb3e5-a8fa-7d01-b1bd-ee52d73319cc`
- Name: `평가 결과 관리자`

Do not use any evaluation result as paper evidence unless the Evaluation
Results Manager classifies it as paper-ready, or explicitly diagnostic-only
with claim boundaries. If a metric or claim is needed, ask the Evaluation
Results Manager or PM instead of inferring from raw reports.

Safe current stance:

- Risk1-0 native noise: diagnostic WARN baseline.
- Risk1-A template prompt portfolio: candidate-generation PASS only, not
  selection or benchmark success.
- Risk1-B: pending/blocked until Evaluation Results Manager classifies the
  actual Qwen chain result.
- Risk2 scoped evidence: separate; do not overstate.
- Risk5: WARN/proxy-only.

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

## Risk1 Evidence: Native Policy Noise Is Weak

Current evidence says not to build the method claim on SmolVLA's native
`noise=` knob alone.

- RunPod/LIBERO test commit: `6308f6e86...`.
- Explicit seeded noise reached `smolvla.predict_action_chunk` inference.
- Candidate chunks were actual policy-generated chunks, not mock samples.
- Diversity remained weak:
  - `mean_normalized_pairwise_l2=0.048124`
  - `min_pairwise_l2=0.0`
  - `selected_vs_policy_l2=0.0`
  - `mean_pairwise_cosine_distance=0.001218`
- Risk1 status: WARN, not PASS.

Interpretation: native SmolVLA noise can be reported as tested plumbing, but it
cannot support a claim that the frozen policy naturally yields useful candidate
diversity for best-of-N selection.

## Claim Boundary

Use the VLM judge only as a pre-execution candidate selector. Do not use it as
the final success detector.

Final reported success must remain the benchmark/environment success signal,
such as `info["success"]`, `is_success`, or environment-specific
`pc_success`.

Do not claim that native SmolVLA noise gives useful candidate diversity. The
honest method direction is external proposal generation around a frozen VLA:
structured post-policy perturbation, subgoal/instruction portfolios, or
MPPI/CEM-style sampling around the nominal SmolVLA chunk.

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

## Related-Work Paragraph For Manuscript Drafting

Drafting note: rewrite this in author voice before submission.

Visual MPC and visual foresight methods typically generate action-sequence
candidates outside the policy and rank them with a learned dynamics model,
simulator, or task cost. PETS, MPPI, CEM, and related model-predictive methods
follow the same broad pattern: propose multiple trajectories, predict their
consequences, and execute only the selected action or short horizon. This
differs from relying on a single native stochasticity knob inside a VLA policy.
Diffusion-based policies and planners can also produce multiple trajectories
through stochastic denoising, while ACT, Behavior Transformer, and IBC model
multimodal action structure during training. High-level language-agent and VLM
robotics systems, including SayCan, Inner Monologue, Code as Policies, and
VoxPoser, often create diversity at the skill, subgoal, or affordance level
rather than at the low-level action-chunk level. Our method should therefore be
positioned as wrapper-level candidate proposal and imagined outcome selection
around a frozen lightweight VLA, not as a claim that SmolVLA's native sampling
already provides sufficient diverse actions.

## Method-Risk Note For Manuscript Drafting

Risk: if candidate chunks are produced only by native SmolVLA noise, the
selection layer may be ranking near-duplicates rather than meaningfully
different futures. The current seeded-noise evidence reached real
`predict_action_chunk` inference but produced weak diversity, so this path is a
WARN. The method should therefore separate two components:

1. candidate proposal generation, which may require structured post-policy
   perturbations, subgoal/instruction portfolios, MPPI/CEM around the nominal
   chunk, ensembles, or test-time adaptation;
2. imagined outcome scoring, which can use simulator-clone rollouts for an
   oracle upper bound and later learned visual dynamics for deployability.

This boundary prevents novelty overclaim against visual MPC and world-model
planning, where external candidate generation and cost-based ranking are
standard.

## Minimal Experiment Matrix

| Condition | Imagination source | Selector | Committed chunk budget | Final success source |
| --- | --- | --- | --- | --- |
| Policy-only | none | base policy first chunk | 1 chunk | environment |
| Native-noise chunks | simulator clone | random or VLM candidate | 1 selected chunk | environment |
| Structured perturbation chunks | simulator clone | random or VLM candidate | 1 selected chunk | environment |
| Subgoal/instruction portfolio | simulator clone | random or VLM candidate | 1 selected chunk | environment |
| MPPI/CEM around nominal chunk | simulator clone | model/simulator/VLM cost | 1 selected chunk | environment |
| Oracle imagined selector | simulator clone | privileged outcome rule | 1 selected chunk | environment |
| VLM imagined selector | simulator clone render | VLM/visual judge | 1 selected chunk | environment |
| Learned imagined selector | learned video/dynamics model | VLM/visual judge | 1 selected chunk | environment |

The first SemRob submission can include native-noise chunks as a negative or
warning control, but it should not depend on them as the only diversity source.
The learned model row is the future-work bridge.

## Metrics

- Benchmark success rate.
- Success per committed action step.
- Wall-clock latency.
- Candidate count.
- Candidate diversity: normalized pairwise L2, minimum pairwise L2, selected
  vs nominal policy L2, and cosine distance.
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
- Do keep the native-noise diversity result as WARN unless later runs show
  substantially stronger candidate separation.
- Do not overclaim novelty against visual MPC, visual foresight, world-model
  planning, PETS, MPPI, or CEM. The novelty, if any, is applying imagined
  outcome selection to frozen lightweight VLA action chunks under clear
  benchmark-success semantics.
