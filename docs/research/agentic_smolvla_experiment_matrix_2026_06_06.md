# Agentic SmolVLA Experiment Matrix

Date: 2026-06-06

## Purpose

This matrix converts the related-work finding into an executable paper plan.
Since visual prompting and affordance overlays already have close precedent,
the project should evaluate overlays as one intervention inside a broader
agentic lightweight VLA control study.

## Main hypothesis

Lightweight VLA policies may fail because they entangle instruction grounding,
spatial localization, action execution, and recovery in one forward-control
loop. A modular agentic layer can improve reliability by separating:

- subgoal selection,
- policy execution,
- state verification,
- retry/replan decisions,
- optional spatial cue injection.

The paper should measure which component actually helps.

## Experimental conditions

### C0: Policy-only baseline

SmolVLA receives the original observation and task instruction. No overlay,
no retry, no verifier-driven replan.

Purpose:

- Establish the benchmark success rate and failure distribution.
- Provide the denominator for all improvement claims.

Claim allowed:

- "The base lightweight VLA achieves X under this task/seed/action budget."

Claim not allowed:

- "The agentic layer helps."

### C1: Overlay-only heuristic

SmolVLA receives an actual-simulation RGB frame with a heuristic visual cue.
No verifier-driven retry/replan.

Purpose:

- Test whether cheap image-space spatial cueing alone changes behavior.
- Separate visual input conditioning from agentic recovery.

Claim allowed:

- "A heuristic cue changes or improves policy behavior under controlled
  conditions."

Claim not allowed:

- "The cue is an oracle."
- "The cue proves object-pose projection correctness."

### C2: Overlay-only true oracle

SmolVLA receives an actual-simulation RGB frame with a projected cue computed
from same-step object pose and camera metadata.

Purpose:

- Estimate the upper bound of perfect spatial cueing.
- Define what a learned affordance predictor must approximate.

Claim allowed:

- "Perfect spatial cueing would provide an upper-bound improvement of X."

Claim not allowed:

- "This is deployable without privileged simulator state."

### C3: Agentic verifier/retry only

SmolVLA receives the original observation, but an external verifier controls
bounded retry or replan decisions. No overlay.

Purpose:

- Test whether recovery helps independently of visual cueing.
- Separate agentic control from spatial input modification.

Claim allowed:

- "Verifier-driven retry recovers Y% of first-failure cases."

Claim not allowed:

- "The internal verifier success is benchmark success."

### C4: Agentic verifier/retry plus heuristic overlay

The agentic layer can retry/replan and inject a cheap image-space spatial cue.

Purpose:

- Test whether practical, non-privileged cueing and agentic recovery interact.
- This is the most deployable variant if it works.

Claim allowed:

- "A lightweight test-time wrapper improves success with Z latency/memory
  overhead."

Claim not allowed:

- "The method relies on oracle simulator state."

### C5: Agentic verifier/retry plus true-oracle overlay

The agentic layer can retry/replan and inject the true-oracle projected cue.

Purpose:

- Upper-bound the combined effect of perfect spatial cueing and recovery.
- Diagnose whether failures are from spatial grounding or from policy/control
  limits.

Claim allowed:

- "Even with perfect spatial cueing, remaining failures are likely control,
  horizon, or verifier problems."

Claim not allowed:

- "This is a practical method without replacing the oracle."

## Required evidence tiers

### Tier S: Synthetic diagnostic

Allowed use:

- Projection math smoke checks.
- Rendering/encoding visual tests.
- Policy-input tensor formatting.

Forbidden use:

- Paper-facing success claims.
- Actual-simulation behavior claims.

### Tier A: Actual-sim RGB fallback

Allowed use:

- Confirms overlays can render on saved simulator RGB.

Forbidden use:

- Oracle projection claims.

### Tier H: Actual-sim heuristic

Allowed use:

- Confirms a non-privileged cue can be generated on real simulator frames.
- Supports practical overlay feasibility.

Forbidden use:

- Object-pose/camera-correct oracle claims.

### Tier O: Actual-sim true oracle

Allowed use:

- Upper-bound diagnostic.
- Learned affordance target definition.

Required fields:

- same-step RGB,
- object pose,
- camera metadata,
- overlay frame,
- episode id,
- step id,
- environment success flag.

## Metrics

- Environment success rate.
- Recovery success after first detected failure.
- Mean retries per episode.
- Action budget consumed.
- Wall-clock latency.
- Additional memory overhead.
- Verifier false-positive rate.
- Verifier false-negative rate.
- Cue generation latency.
- Failure categories after final attempt.

## Paper framing guardrails

- Do not title the method as Oracle Point Overlay.
- Do not claim visual prompting as novel.
- Cite VP-VLA, TraceVLA, AVP, RoboPoint, CLIPort, VoxPoser, AffordVLA,
  AffordanceVLA, and CoA-VLA as close or foundational related work.
- Present overlays as a controlled interface inside a test-time agentic layer.
- Report final success only from benchmark or simulator success flags.
- Keep internal verifier success separate from final success.

## Immediate implementation implication

The current true-oracle projection work remains useful, but its role changes:

- Old role: proposed main method.
- New role: upper-bound diagnostic and evidence gate.

The next implementation milestone should therefore prioritize producing the
Tier O manifest for at least 10 actual-simulation samples, not because that is
the final deployable method, but because it makes the ablation scientifically
interpretable.

