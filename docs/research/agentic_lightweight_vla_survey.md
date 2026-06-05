# Agentic Lightweight VLA Survey

Date: 2026-06-05

## Working Thesis

This project should be framed as a system-level study of whether lightweight
vision-language-action policies can be made more reliable by adding an external
agentic loop: task decomposition, state verification, retry, and replanning.

The core claim should not be that a wrapper makes a weak policy universally
strong. The defensible claim is narrower: for long-horizon or partially failed
manipulation trials, a lightweight policy may recover more often when its action
chunks are selected, checked, and retried under explicit task-state feedback.

## Introduction Progress

The field has moved from single-task imitation policies toward large robot
foundation policies trained on diverse robot data. RT-1 established transformer
policies for real-world control at scale, RT-2 introduced the VLA framing by
co-training vision-language models to emit robot actions, and Open X-Embodiment
showed that cross-embodiment data can support broader robot policy transfer.
OpenVLA then made a 7B open-source VLA available on 970k robot episodes.

The recent counter-trend is efficiency. TinyVLA and SmolVLA argue that real
deployment is constrained by latency, memory, data, and hardware access.
SmolVLA is especially relevant here because it is a 450M open model designed
for consumer hardware, LeRobot datasets, action chunks, and asynchronous
inference. This makes it a practical base policy for research on wrappers:
small enough to iterate on locally, but strong enough to run nontrivial robot
benchmarks such as LIBERO and real SO100/SO101 setups.

The gap is that most VLA papers optimize the policy itself, while long-horizon
robotic execution also depends on detecting failed subgoals, retrying from
partial state, and choosing when to replan. Agentic language-model work has
already shown that reasoning traces, environment feedback, memory, and
self-reflection can improve sequential decision making without updating model
weights. Robotics work such as SayCan and Inner Monologue similarly separates
semantic planning from grounded low-level execution. This project sits at their
intersection: use a lightweight VLA as the low-level actor and add a measurable
agentic control layer around it.

## Related Work Progress

### 1. Generalist Robot Policies and VLA Models

RT-1 is the pre-VLA scaling baseline: transformer-based real-world robot control
trained on large-scale demonstrations. RT-2 formalized the VLA recipe by
connecting web-scale VLM knowledge to robotic action prediction. Open
X-Embodiment supplied the multi-robot data substrate for cross-embodiment
training and RT-X models. OpenVLA then provided an open-source 7B VLA trained
on real robot demonstrations.

Relevant citations:

- Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale", arXiv:2212.06817.
- Zitkovich et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", arXiv:2307.15818.
- Open X-Embodiment Collaboration et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", arXiv:2310.08864.
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", arXiv:2406.09246.
- Octo Model Team et al., "Octo: An Open-Source Generalist Robot Policy", arXiv:2405.12213.

### 2. Efficient and Lightweight VLA Models

Efficient VLA work is becoming its own line because large VLAs are expensive to
train, fine-tune, and deploy on robot hardware. TinyVLA targets fast,
data-efficient manipulation. SmolVLA is the strongest match for this project:
it uses a compact model, public LeRobot community data, action chunks, and
deployment-oriented inference. Hugging Face documentation explicitly treats it
as a base model that should be fine-tuned for the user's task, which means a
wrapper study should report whether agentic recovery helps before or alongside
task-specific fine-tuning.

Relevant citations:

- Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics", arXiv:2506.01844.
- Hugging Face LeRobot, "SmolVLA" documentation.
- Hugging Face blog, "SmolVLA: Efficient Vision-Language-Action Model trained on LeRobot Community Data", 2025.
- Wen et al., "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation", arXiv:2409.12514.
- Guan et al., "Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey", arXiv:2510.17111.

### 3. Agentic Planning Around Low-Level Skills

SayCan is the key robotics precedent: a language model proposes high-level
steps, but feasibility is grounded by skill value functions. Inner Monologue
adds feedback into the planning loop. Code as Policies shows that language
models can synthesize robot programs around lower-level APIs. These papers
support our design principle: do not ask the VLA to solve every part of
long-horizon task execution internally; expose the environment state and wrap
the low-level policy with planning and verification logic.

Relevant citations:

- Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", arXiv:2204.01691.
- Huang et al., "Inner Monologue: Embodied Reasoning through Planning with Language Models", arXiv:2207.05608.
- Liang et al., "Code as Policies: Language Model Programs for Embodied Control", arXiv:2209.07753.
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", arXiv:2210.03629.
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", arXiv:2303.11366.

### 4. Hierarchical and Reasoning-Augmented VLA Systems

Newer VLA papers increasingly move away from purely monolithic action
prediction. HAMSTER uses a high-level VLM to generate coarse trajectory
guidance and a lower-level controller for precise manipulation. Long-horizon
VLA systems explicitly add reasoning, acting, and memory layers. ChatVLA-2 and
similar work push reasoning into the model itself. This project differs by
keeping the base policy lightweight and adding reasoning externally, making the
intervention easier to ablate.

Physical Intelligence's π0.7 is especially relevant because its main idea is
not just a larger VLA, but steerability through diverse context conditioning.
The model is trained and prompted with task language, subtask language,
metadata such as speed and quality, control modality labels, memory/context,
and visual subgoal images. This suggests a concrete design direction for this
project: the agentic layer should not merely retry the same natural-language
instruction. It should produce richer control context for SmolVLA-like actors:
current subtask, desired execution style, retry reason, visual or state subgoal,
and a bounded action horizon.

Relevant citations:

- Li et al., "HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation", arXiv:2502.05485.
- Li et al., "Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory", ICCV 2025.
- Zhou et al., "ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge", arXiv:2505.21906.
- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models", arXiv:2305.16291.
- Physical Intelligence et al., "π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities", arXiv:2604.15483.

### 5. Benchmarks and Evaluation Risk

LIBERO is relevant because it is widely used for language-conditioned robot
learning and lifelong manipulation transfer. However, recent benchmarking work
warns that VLA performance is task-, platform-, action-space-, and evaluation-
protocol-sensitive. This is important for our paper: claims should use
policy-only vs agentic-wrapper comparisons under the same environment seeds,
same base policy, same action budget, and same success detector.

Relevant citations:

- Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", arXiv:2306.03310.
- "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks", arXiv:2411.05821.
- "LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization", arXiv:2510.03827.
- Li et al., "Survey of Vision-Language-Action Models for Embodied Manipulation", arXiv:2508.15201.

## Community Reading

The community signal is mixed but useful:

- Positive: there is clear demand for smaller open VLAs because robot labs need
  models that run on consumer GPUs, edge devices, or low-cost arms. SmolVLA's
  Hugging Face release and LeRobot integration are strong adoption signals.
- Positive: agentic control is considered credible when the agent has grounded
  feedback, executable skills, and measurable success checks, as in SayCan,
  Inner Monologue, ReAct, and Reflexion.
- Skeptical: many discussions and benchmark papers point out that demos can
  overstate generality, LIBERO numbers may be hard to compare, and long-horizon
  failures often come from state tracking, grounding, and recovery rather than
  single-step perception.
- Strategic implication: our paper should avoid claiming "agentic VLA
  generality." It should claim "recovery-oriented wrapper improves measured
  robustness for a fixed lightweight VLA under controlled long-horizon tasks."

## Proposed Paper Positioning

Potential title direction:

"Agentic Recovery for Lightweight Vision-Language-Action Policies"

Potential contribution statements:

1. A modular agentic wrapper for lightweight VLA policies that separates
   planning, execution, verification, retry, and replanning.
2. A controlled comparison of policy-only SmolVLA execution against an
   agentic-retry variant under identical seeds, action budgets, and success
   criteria.
3. A failure taxonomy for lightweight VLA manipulation: perception mismatch,
   action saturation, object miss, partial progress, verifier false positive,
   retry exhaustion, and replan failure.
4. An open, reproducible evaluation harness over LIBERO and/or ManiSkill/SO101
   that records traces, videos, verifier decisions, retries, and latency.

## Method Progress

The method should use the current repository pipeline as follows:

- Base policy: `lerobot/smolvla_base` or LIBERO-tuned SmolVLA.
- Actor API: observation images, robot state, language instruction, action chunk.
- Planner: decomposes task into subgoals or retry intentions.
- Verifier: reads simulator state and/or image observations to decide whether
  the current subgoal is complete, failed, or uncertain.
- Retry loop: re-executes policy with bounded retries and a changed prompt,
  reset condition, or subgoal.
- Replanner: updates the next instruction when verifier detects a recoverable
  failure.

π0.7-inspired control context:

- `task_instruction`: original language command.
- `subtask_instruction`: current decomposed step.
- `retry_reason`: verifier-produced failure label.
- `desired_quality`: conservative, fast, precise, exploratory.
- `desired_speed`: slow, normal, fast.
- `action_horizon`: max chunk/step budget for this attempt.
- `state_subgoal`: simulator-derived target condition.
- `visual_subgoal`: optional image target generated from a prior successful
  frame, simulator render, or lightweight world model.

This should be implemented as an external prompt/control packet, not by
modifying SmolVLA weights at first. The ablation can compare plain language
retry against π0.7-style enriched retry context.

Primary metrics:

- Task success rate.
- Recovery success after first failure.
- Mean retries per episode.
- Action budget and wall-clock latency.
- Verifier false positive / false negative rate.
- Failure category distribution.

## Immediate Next Steps

1. Use RunPod LIBERO baseline to obtain policy-only SmolVLA numbers.
2. Freeze the baseline action budget and seed protocol.
3. Add the agentic wrapper only after baseline reproducibility is documented.
4. Evaluate policy-only vs agentic retry on the same LIBERO task IDs.
5. Write the first related-work section around three groups: efficient VLAs,
   agentic embodied planning, and long-horizon VLA evaluation.
