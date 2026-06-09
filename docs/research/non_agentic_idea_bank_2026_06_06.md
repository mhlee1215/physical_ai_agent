# Non-Agentic / Adjacent Idea Bank for Lightweight VLA

Date: 2026-06-06

Scope: This bank deliberately avoids the already-discussed ideas:

- generic planner / verifier / retry / replan loop
- π0.7-style structured control context as the main idea
- baseline SmolVLA-only evaluation

The goal is to find distinct mechanisms that could improve or analyze a
lightweight VLA such as SmolVLA without simply restating the current agentic
wrapper plan.

## Iteration 01 - Safety Filter Around VLA Actions

Source fields: control theory, safe RL, control barrier functions.

Observed materials:

- ConBaT: Control Barrier Transformer for Safe Policy Learning, arXiv:2303.04212.
- Learning for Safety-Critical Control with Control Barrier Functions, CaltechAUTHORS.
- Control Barrier Function safety filters for robot manipulators and safety-critical control.

Idea:

Treat SmolVLA as an unconstrained proposal policy, then pass each action through
a lightweight safety/action-feasibility filter before execution. The filter
minimally modifies actions that violate workspace, joint, gripper, collision, or
object-contact constraints.

Apply to this project:

- Add `policy=smolvla_safety_filtered`.
- Input: raw SmolVLA action, current state, action bounds, optional object pose.
- Output: clipped or projected action plus `filter_delta`.
- Metric: task success, safety violation count, action modification norm,
  latency overhead.

Why it is different:

This is not retry or replanning. It is online action correction at the control
interface.

Critical evaluation:

- Strong for avoiding obviously bad actions.
- Weak if the VLA chooses semantically wrong but safe actions.
- Easy to implement in ManiSkill/SO101 because action bounds and state are
  already available.
- Paper risk: may look like generic action clipping unless the safety
  predicates are manipulation-specific.

Verdict:

High feasibility, medium novelty. Good as a robustification baseline.

## Iteration 02 - Conformal Uncertainty Gate

Source fields: conformal prediction, uncertainty quantification, safety
assurance.

Observed materials:

- Sample-Efficient Safety Assurances using Conformal Prediction, NVIDIA Research.
- Safe Planning in Dynamic Environments Using Conformal Prediction, IEEE RA-L.
- Learning Robot Safety from Sparse Human Feedback using Conformal Prediction,
  arXiv:2501.04823.
- Uncertainty-Aware Policy Steering, arXiv:2602.22474.

Idea:

Train or calibrate a small uncertainty gate that predicts whether the next
SmolVLA chunk is likely to fail. The gate does not need to know the correct
action. It only needs calibrated abstention: execute, slow down, ask for another
view, switch policy, or stop.

Apply to this project:

- Collect baseline rollouts and label chunk outcomes: success, progress,
  no-progress, violation.
- Build a nonconformity score from state delta, reward delta, action norm,
  distance-to-goal, and visual embedding distance.
- Use split calibration to choose a threshold with a target false-negative rate.

Why it is different:

This gives a statistical uncertainty layer rather than a heuristic verifier.

Critical evaluation:

- Strong for paper credibility because it gives calibrated risk language.
- Requires data collection before it works.
- Calibration may not transfer across tasks unless task-conditioned.
- Works best with simulator labels, so LIBERO/ManiSkill are appropriate.

Verdict:

High paper value, medium implementation effort. This is one of the best ideas.

## Iteration 03 - Retrieval Cache of Successful Action Snippets

Source fields: case-based reasoning, retrieval-augmented embodied agents,
robot experience memory.

Observed materials:

- RT-Cache: Training-Free Retrieval for Real-Time Manipulation.
- RAM: Retrieval-Based Affordance Transfer for Generalizable Zero-Shot Robotic
  Manipulation, arXiv:2407.04689.
- Retrieval-Augmented Embodied Agents, CVPR 2024.
- Qualitative Case-Based Reasoning and Learning, Artificial Intelligence 2020.

Idea:

Store short successful observation-action snippets from previous rollouts.
At runtime, retrieve the nearest snippet by visual/state/task embedding and use
it to bias or replace SmolVLA's next action chunk.

Apply to this project:

- Build `_workspace/memory/action_snippets.jsonl`.
- Each snippet stores task, image embedding, state summary, action chunk,
  outcome label.
- Runtime policy: if nearest positive snippet similarity is high, replay or
  blend snippet action with SmolVLA action.

Why it is different:

This is not reasoning. It is experience reuse / case-based control.

Critical evaluation:

- Strong for repeated benchmark tasks such as LIBERO.
- Risk: can overfit to task IDs or seeds.
- Must report train/test task split clearly.
- Needs careful comparison against simply adding demonstrations.

Verdict:

High practicality, medium novelty. Good if framed as "training-free retrieval
augmentation for lightweight VLA."

## Iteration 04 - Active Perception Before Action

Source fields: active vision, next-best-view planning, interactive perception.

Observed materials:

- ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for
  Precise 3D Robotic Manipulation, arXiv:2601.08325.
- Active-Perceptive Motion Generation for Mobile Manipulation, arXiv:2310.00433.
- View planning in robot active vision: a survey, Computational Visual Media
  2020.
- Object-Aware Interactive Perception System for tabletop scene exploration.

Idea:

Before calling SmolVLA for a manipulation chunk, execute a cheap observation
action: move wrist camera, choose top-down view, zoom/crop object, or request an
extra render. Then feed the improved observation to SmolVLA.

Apply to this project:

- In SO101/ManiSkill, add an `observe_first` mode.
- Compare static camera vs active camera/crop selection.
- Use uncertainty, occlusion, or object-size heuristics to trigger the extra
  view.

Why it is different:

It improves input quality, not action selection or retry.

Critical evaluation:

- Very relevant because lightweight VLAs are perception-limited.
- Easy in simulation where cameras are controllable.
- Real robot transfer requires actual camera motion or multi-camera setup.
- Needs action budget accounting: observation actions must count as overhead.

Verdict:

High novelty for this repo, high relevance. Strong candidate.

## Iteration 05 - Affordance Map as Intermediate Representation

Source fields: affordance learning, object-centric manipulation, spatial VLMs.

Observed materials:

- RT-Affordance: Affordances are Versatile Intermediate Representations for
  Robot Manipulation, arXiv:2411.02704.
- RoboPoint: A VLM for Spatial Affordance Prediction for Robotics,
  arXiv:2406.10721.
- Instruction-Guided Affordance Net, arXiv:2408.10658.
- Learning Precise Affordances from Egocentric Videos for Robotic Manipulation,
  ICCV 2025.

Idea:

Insert an affordance predictor before SmolVLA. It produces a contact point,
grasp region, placement region, or object interaction mask. SmolVLA receives
the same image plus an affordance overlay/crop/coordinate hint.

Apply to this project:

- For PickCube/LIBERO, derive affordance maps from simulator object positions
  first.
- Later swap in a learned affordance model.
- Compare `image_only` vs `image_plus_affordance_overlay`.

Why it is different:

This is perception grounding, not agentic control.

Critical evaluation:

- Strong for grasp/place tasks.
- Risk: simulator-derived affordances may leak privileged information.
- Needs separate "privileged oracle" vs "learned/perceptual" experiments.
- Could be a compelling bridge from VLA to classic manipulation.

Verdict:

High idea quality. Implement oracle version first, learned version later.

## Iteration 06 - Residual Corrector on Top of SmolVLA

Source fields: residual policy learning, model predictive residual control,
flow policy correction.

Observed materials:

- Residual Policy Learning, arXiv:1812.06298.
- Iterative Residual Policy for Goal-Conditioned Dynamic Manipulation,
  arXiv:2203.00663.
- FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for
  Robotic Manipulation, 2026.
- Compliant Residual DAgger for long-horizon contact-rich manipulation.

Idea:

Do not replace SmolVLA. Learn a small residual policy that predicts
`delta_action` from state error and raw SmolVLA action. The executed action is:

```text
action = smolvla_action + residual_delta
```

Apply to this project:

- Train residual on failed/near-miss simulator rollouts.
- Start with linear or MLP residual using state predicates.
- Keep residual small and bounded to preserve base policy behavior.

Why it is different:

It is a learned local correction layer, not a planner.

Critical evaluation:

- Good for systematic bias: too high grasp, too shallow push, action scaling.
- Bad for semantic failures: wrong object, wrong phase.
- Requires training data, but much less than full VLA fine-tuning.
- Strong ablation against direct SmolVLA fine-tuning.

Verdict:

High feasibility and good paper angle: "low-rank/local residual adaptation."

## Iteration 07 - Adversarial Scenario Mining for Failure Discovery

Source fields: autonomous driving safety testing, adversarial scenario
generation, policy falsification.

Observed materials:

- DRAGEN: Distributionally Robust Policy Learning via Adversarial Environment
  Generation, IEEE RA-L 2022.
- Adaptive generation of challenging scenarios for autonomous systems with
  RAPT.
- SAGE: Steerable Adversarial Scenario Generation through Test-Time Preference
  Alignment.
- RoboLab: high-fidelity simulation benchmark with controlled perturbations.

Idea:

Use a red-team scenario generator to find where SmolVLA fails: object pose,
lighting, distractors, camera angle, initial gripper pose, task phrasing. Then
evaluate whether proposed wrappers improve those discovered slices.

Apply to this project:

- Add a `scenario_miner` that sweeps object pose, camera noise, distractor
  placement, and instruction paraphrase.
- Score scenarios by harm/rarity/ambiguity: failure rate, recovery difficulty,
  semantic confusion.
- Produce a failure atlas instead of only mean success rate.

Why it is different:

It improves evaluation and robustness discovery, not the policy directly.

Critical evaluation:

- Very valuable for a paper because it avoids cherry-picked tasks.
- Implementation can start simple with randomized seeds and perturbations.
- Risk: adversarial scenarios may become unrealistic unless constrained.
- Need report both natural and stress-test distributions.

Verdict:

Very strong for paper credibility. Implement early.

## Iteration 08 - Lightweight World-Model Preview

Source fields: visual foresight, model-based RL, robotic world models.

Observed materials:

- Visual Foresight, arXiv:1812.00568.
- tau_0-WM: Unified Video-Action World Model for Robotic Manipulation,
  arXiv:2606.01027.
- H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model,
  arXiv:2602.11291.
- World Model for Robot Learning survey, arXiv:2605.00080.

Idea:

Before executing a SmolVLA action chunk, use a tiny learned or simulator-derived
forward model to predict whether the chunk is likely to improve the task state.
Reject or down-rank chunks that predict collision, no object motion, or moving
away from goal.

Apply to this project:

- Short-term version: use simulator state transition rollout for candidate
  chunks where available.
- Longer-term version: train a cheap state-delta predictor from rollout traces.
- Avoid full video generation at first; use state/object deltas.

Why it is different:

This is "look ahead before acting", not retry after failure.

Critical evaluation:

- Full video world models are too heavy for our speed goal.
- State-level preview is feasible and aligned with lightweight theme.
- Needs candidate action chunks; SmolVLA may only output one chunk unless
  sampling can be enabled.

Verdict:

Medium feasibility, high conceptual value. Use state-level model, not video.

## Iteration 09 - Test-Time Adaptation by Online Normalization / Calibration

Source fields: test-time adaptation, sim-to-real calibration, online policy
steering.

Observed materials:

- Test-time adaptation literature in vision/robotics.
- Latent Policy Barrier, arXiv:2508.05941.
- Bayesian Disturbance Injection for robust imitation learning.
- Robust imitation and calibration papers for distribution shift.

Idea:

Many lightweight VLA failures may come from input distribution mismatch:
brightness, camera crop, state scale, action scale, or environment-specific
normalization. Add a test-time calibration layer that adapts observation
preprocessing and action scaling without updating SmolVLA weights.

Apply to this project:

- Calibrate image brightness/crop to match training statistics.
- Calibrate action scale per environment using first few safe steps.
- Track embedding OOD score and switch preprocessing profiles.

Why it is different:

It treats VLA failure as distribution shift rather than reasoning failure.

Critical evaluation:

- Very plausible for SmolVLA because deployment conditions matter.
- Hard to make novel unless tied to measurable VLA failure modes.
- Easy to accidentally tune on test tasks; protocol must be strict.

Verdict:

Medium novelty, high practical payoff. Good support component.

## Iteration 10 - Mixture of Cheap Controllers with Learned Gating

Source fields: hybrid control, ensembles, options, mixture-of-experts.

Observed materials:

- Classical hybrid systems and options-style hierarchical RL.
- Residual policy and safe-control literature.
- Robot policy evaluation papers showing task-dependent failure modes.

Idea:

Instead of asking SmolVLA to do every sub-behavior, maintain a small portfolio:

```text
SmolVLA actor
center / hold controller
visual servo controller
grasp-close primitive
zero / safe stop
retrieved snippet replay
```

A lightweight gate chooses which controller should act for the next short
horizon.

Apply to this project:

- Start with three controllers: SmolVLA, safe-stop, visual-servo/reach.
- Gate features: distance to object, action confidence, progress, safety filter
  delta, task phase.
- Evaluate success and controller usage distribution.

Why it is different:

This is not agentic planning. It is a hybrid action source architecture.

Critical evaluation:

- Strong for robustness and interpretability.
- Risk: if primitives are too strong, reviewer may say SmolVLA is not central.
- Need ablation: gate-only, primitive-only, SmolVLA-only, mixture.

Verdict:

High practicality, medium paper risk. Good if positioned as "lightweight VLA as
one expert in a hybrid manipulation stack."

## Ranked Shortlist

Top candidates for this project:

1. Conformal uncertainty gate.
2. Active perception before action.
3. Adversarial scenario mining / failure atlas.
4. Affordance-map intermediate representation.
5. Residual corrector on top of SmolVLA.

Lower priority but useful:

6. Retrieval cache of successful action snippets.
7. Safety filter around VLA actions.
8. Lightweight state world-model preview.
9. Test-time calibration.
10. Mixture of cheap controllers with learned gating.

## Suggested Paper Angles

Angle A:

"Uncertainty-Gated Lightweight VLA" - SmolVLA plus conformal risk gating and
selective action abstention.

Angle B:

"Perception-First Lightweight VLA" - active camera/crop/affordance cues before
action prediction.

Angle C:

"Failure Atlas for Lightweight VLA" - adversarial scenario mining plus wrapper
evaluation over discovered failure modes.

Angle D:

"Residual Adaptation for Lightweight VLA" - small bounded delta-action policy
trained from failures, compared against full fine-tuning.

