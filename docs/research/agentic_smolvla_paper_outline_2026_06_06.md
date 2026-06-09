# Agentic SmolVLA Paper Outline

Date: 2026-06-06

## Working title

Agentic Recovery and Spatial-Cue Ablations for Lightweight Vision-Language-Action Policies

## Thesis

This paper studies whether compact VLA policies can be made more reliable by a
test-time agentic wrapper that separates execution, verification, retry, and
optional spatial cueing.

## Claim-safe abstract skeleton

Current safe version:

> Lightweight vision-language-action policies are attractive for accessible
> robotics, but compact policies may struggle with spatial grounding,
> partial failures, and long-horizon recovery. We develop an evaluation harness
> for studying test-time agentic wrapping around SmolVLA-class policies,
> separating policy-only execution, visual cueing, verifier-driven retry, and
> oracle upper-bound diagnostics. We present a claim-gated experimental
> protocol that distinguishes benchmark success from internal verifier success
> and separates synthetic diagnostics, actual-simulation RGB evidence, visual
> heuristic overlays, and privileged true-oracle projection.

Blocked until experiments:

> Our method improves benchmark success.

Blocked until Tier O:

> True-oracle projection is demonstrated on 10+ actual simulation timesteps.

## Section plan

### 1. Introduction

Safe content:

- VLAs are promising but often large and expensive.
- SmolVLA-class models motivate low-cost, deployable robotics research.
- Reliability failures can come from grounding, partial execution, recovery,
  and state tracking, not only model capacity.
- This work studies external test-time structure around compact policies.

Avoid for now:

- Claiming final success improvement.
- Claiming visual overlays are novel.

### 2. Related Work

Required groups:

- Generalist VLA policies: RT-1, RT-2, Open X-Embodiment, OpenVLA, Octo.
- Lightweight/efficient VLA: TinyVLA, SmolVLA, efficient VLA surveys.
- Agentic embodied control: SayCan, Inner Monologue, Code as Policies, ReAct,
  Reflexion, Voyager.
- Visual prompting / affordance interfaces: VP-VLA, TraceVLA, AVP, RoboPoint,
  AffordVLA, AffordanceVLA, CoA-VLA, CLIPort, VoxPoser.
- Evaluation caution: LIBERO, LIBERO-PRO, VLA benchmarking papers.

Safe positioning:

> Unlike work that proposes visual prompting itself as the method, we treat
> visual cueing as an ablated interface inside a test-time wrapper and measure
> when cueing, verification, retry, and oracle upper bounds explain behavior.

### 3. Method

Core components:

- Base policy: SmolVLA-class actor.
- Wrapper: planner / verifier / retry / optional cue injector.
- Cue types: none, heuristic, true-oracle upper bound.
- Success semantics: final success comes only from environment success flags.
- Evidence tiers: synthetic diagnostic, actual RGB fallback, actual heuristic,
  actual true oracle.

Safe content:

- Describe architecture and evaluation protocol.
- Describe current upper-bound role of oracle cues.

Avoid for now:

- Calling the oracle cue deployable.

### 4. Experiment Matrix

Required conditions:

- C0: policy-only SmolVLA.
- C1: overlay-only heuristic.
- C2: overlay-only true oracle.
- C3: agentic verifier/retry only.
- C4: agentic + heuristic overlay.
- C5: agentic + true-oracle overlay.

Metrics:

- Environment success rate.
- Recovery after first failure.
- Mean retries.
- Action budget.
- Wall-clock latency.
- Memory overhead.
- Verifier false positive / false negative.
- Failure categories.

Current status:

- Matrix scaffold exists.
- Agentic schema readiness exists from CP22/CP23.
- Actual SmolVLA matrix results are pending.

### 5. Evidence and Claim Gates

Safe content:

- Actual RGB exists.
- Heuristic overlays can be generated on actual sim RGB.
- Synthetic metadata diagnostic shows projection codepath can work.
- Mac-local true-oracle probe is blocked by SAPIEN/Vulkan.
- Remote evidence pack exists for renderer-capable environments.

Blocked:

- Tier O true-oracle success.
- Agentic success improvement.

### 6. Results

Do not write final result claims yet.

Placeholder tables:

- Table 1: C0-C5 success rates.
- Table 2: latency/memory overhead.
- Table 3: recovery/failure taxonomy.
- Figure 1: baseline vs agentic architecture.
- Figure 2: evidence tier separation.
- Figure 3: actual RGB / heuristic / oracle upper-bound examples.

### 7. Discussion

Safe content:

- Visual cueing should be interpreted cautiously due to close prior work.
- Privileged oracle cues are diagnostic upper bounds.
- Agentic wrappers can improve interpretability of failure and recovery
  attempts even before final success improvements are established.

Avoid:

- Overgeneralizing from synthetic diagnostics.

### 8. Limitations

Required limitations:

- Tier O requires renderer-capable environment and privileged simulator state.
- Heuristic cueing may not localize true affordances.
- Agentic retry can increase latency and action budget.
- Internal verifier success is not final task success.
- Current Mac-local renderer stack blocks true-oracle capture.

### 9. Reproducibility

Required artifacts:

- `scripts/run_renderer_env_preflight.sh`
- `scripts/run_actual_sim_true_oracle_probe_cp24.sh`
- `scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh`
- `scripts/run_remote_true_oracle_evidence_pack.sh`
- `scripts/build_paper_claim_gate_report.py`
- `scripts/build_agentic_smolvla_experiment_matrix_result_report.py`

## Immediate writing rule

Every paper paragraph must map to one of:

- supported claim,
- diagnostic-only claim,
- blocked future claim,
- hypothesis.

If a sentence does not fit one of those four categories, it should not be in the
draft yet.

