# Related-work scan: affordance / visual-prompt overlays for VLA

Date: 2026-06-06

## Question

Before continuing the Oracle Point Overlay direction, check whether the core idea is already covered by prior work:

- Add a point, trace, heatmap, affordance marker, or visual primitive to the policy image input.
- Use that added visual structure to improve low-level robot control in a VLA or language-conditioned policy.
- Keep the target policy lightweight, ideally SmolVLA-class, and improve success using an external agentic/perception layer.

## Immediate finding

The broad idea of using spatial affordance signals or visual prompts for robot control is already active and partially crowded.

Therefore, the paper should not claim novelty as simply:

> "We overlay an affordance point on the VLA input."

That claim would overlap strongly with recent visual-prompting and affordance-aware VLA work.

The safer positioning is:

> "We study a lightweight, test-time agentic wrapper around SmolVLA-class policies, separating policy-only control from verifier-driven retry/replan and explicitly comparing no-overlay, heuristic-overlay, and oracle-overlay evidence under actual simulation rollouts."

## Closest overlaps

### VP-VLA / visual prompting interface for VLA

VP-VLA is directly relevant because it frames visual prompting as an interface between high-level reasoning and low-level VLA control.

Overlap risk:

- High if our method only adds visual markers to VLA input.
- Medium if our method uses visual prompts but contributes a controlled lightweight policy evaluation protocol and agentic retry loop.

Useful distinction:

- Our current direction can emphasize SmolVLA-class low-cost deployment, actual-simulation evidence gates, and success semantics that separate benchmark success from internal verifier success.

Reference:

- https://arxiv.org/abs/2603.22003

### TraceVLA / visual trace prompting

TraceVLA overlays active point trajectories on an initial observation and feeds both original and trace-overlaid images into a VLA.

Overlap risk:

- High for "draw visual hints into the VLA input."
- Especially high if we use temporal traces rather than a single affordance point.

Useful distinction:

- Our current Oracle Point Overlay is a single affordance/contact target or policy-input readiness probe, not necessarily a learned trajectory-trace finetuning dataset.
- But if we move toward temporal trails, we must cite TraceVLA prominently and narrow our claim.

Reference:

- https://tracevla.github.io/

### AVP / Action with Visual Primitives

AVP uses spatially grounded visual-primitive tokens emitted by a VLM to condition an action expert.

Overlap risk:

- High at the conceptual interface level: symbolic/spatial visual primitives guide low-level action.

Useful distinction:

- If we do not introduce a new learned visual-primitive vocabulary and instead evaluate cheap test-time overlays for existing SmolVLA inference, the claim is more about wrapper design and empirical compute/success tradeoff.

Reference:

- https://kingdroper.github.io/AVP/

### AffordVLA / AffordanceVLA / CoA-VLA

Recent VLA papers explicitly inject affordance representations, forecast affordance maps, or use visual-text chains of affordance for VLA action generation.

Overlap risk:

- High for affordance-aware VLA improvement.
- Very high if we train an affordance head and integrate it into VLA features.

Useful distinction:

- We can keep our contribution test-time/modular rather than retraining the VLA.
- If we train an affordance predictor, we should present it as a small auxiliary module and compare against oracle/heuristic/no-overlay controls.

References:

- https://arxiv.org/abs/2605.17517
- https://arxiv.org/abs/2606.06155
- https://openaccess.thecvf.com/content/ICCV2025/papers/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.pdf

## Older but foundational related work

### Transporter Networks and CLIPort

Transporter-style methods and CLIPort are important because they predict spatial action/affordance maps for pick/place manipulation, often with strong spatial precision.

Overlap risk:

- Medium. These are not necessarily VLA image-overlay methods, but they establish that spatial affordance/action maps are a standard robot-control representation.

Useful distinction:

- Our overlay is an input-conditioning mechanism for an existing VLA, not a full spatial action-map policy.

Reference:

- https://github.com/cliport/cliport

### VoxPoser

VoxPoser uses LLM/VLM-generated 3D value maps and constraints grounded in RGB-D observations, then uses planning to synthesize trajectories.

Overlap risk:

- Medium. It is not SmolVLA overlay input, but it is strongly related to language-conditioned affordance/value maps and closed-loop replanning.

Useful distinction:

- Our target is lightweight learned VLA control, not model-based 3D value-map planning.
- But the "agentic map + planner/verifier" framing must cite VoxPoser.

Reference:

- https://voxposer.github.io/
- https://arxiv.org/abs/2307.05973

### RoboPoint

RoboPoint predicts 2D spatial affordance/action points from image and instruction and projects them into 3D using depth.

Overlap risk:

- High for "point affordance prediction."
- Medium for our full system if the point is only one component of a SmolVLA wrapper/evaluation protocol.

Useful distinction:

- RoboPoint is an affordance predictor; our paper can use such a model as a candidate auxiliary module rather than claiming point prediction itself.

Reference:

- https://robo-point.github.io/

## Baseline VLA context

### OpenVLA

OpenVLA is a 7B open-source VLA trained on a large robot-demonstration mixture and is an important baseline/context for open VLA work.

Reference:

- https://arxiv.org/abs/2406.09246

### SmolVLA

SmolVLA motivates the low-cost / lightweight direction. It is explicitly designed for affordable and efficient robotics and uses asynchronous inference / action chunking.

Reference:

- https://arxiv.org/abs/2506.01844
- https://huggingface.co/blog/smolvla

## Novelty risk assessment

### Unsafe claim

"A visual affordance overlay improves a VLA."

Reason:

- This is too close to VP-VLA, TraceVLA, affordance-aware VLA papers, and older affordance-map robot policies.

### Safer claim

"A modular agentic layer can improve or diagnose lightweight VLA behavior by injecting external spatial cues only at test time, while a verifier controls retry/replan and final success remains benchmark-defined."

Why safer:

- It is not claiming that visual prompting itself is new.
- It focuses on low-cost deployment, wrapper-level intervention, success-semantics discipline, and controlled ablations.

### Strongest paper angle

The most defensible direction is a controlled empirical paper:

1. Policy-only SmolVLA baseline.
2. SmolVLA + heuristic visual overlay.
3. SmolVLA + oracle visual overlay.
4. SmolVLA + agentic verifier/retry without overlay.
5. SmolVLA + agentic verifier/retry + overlay.

Report:

- Benchmark success only from environment success flags.
- Internal verifier success only as a control signal.
- Overlay source separated into actual-sim true oracle, actual-sim heuristic, and synthetic diagnostics.
- Runtime and memory overhead relative to SmolVLA.

## Direction decision

Do not abandon the project yet.

But pivot the framing immediately:

- From: "new affordance overlay method."
- To: "agentic lightweight VLA control study with explicit spatial-cue interfaces and strict evidence gates."

The overlay becomes one intervention in an agentic wrapper, not the whole contribution.

