# SO101 Qwen3 Tool Planner

This pipeline uses Qwen3-8B as a high-level planner only. It does not send
robot actions. It converts a user task into the approved fixed-jaw edge
primitive chain:

```text
move("green cube") -> align("green cube") -> pick_up("green cube")
```

The executor should map those function calls to the existing primitive policy
prompts:

| Function | Primitive dataset | Policy prompt |
| --- | --- | --- |
| `move` | `move_over_cube_edge` | `Move the static finger pad above one visible {object} edge.` |
| `align` | `align_fixed_jaw_cube_edge` | `Align the static finger pad with one visible {object} edge.` |
| `pick_up` | `grip_from_edge_cube` | `Keep the static finger pad at the {object} edge, close the gripper, and lift.` |

## Contract

- Qwen3 should run in non-thinking mode for tool routing.
- The tool list is intentionally narrow for 8B/14B models.
- The planner validates the exact order `move -> align -> pick_up`.
- All calls must use the same target object.
- If the model returns prose, malformed JSON, unsupported tools, or the wrong
  order, the planner raises an error instead of silently executing a bad plan.

## Local vLLM-style usage

Start a Qwen3-8B OpenAI-compatible server separately, then run:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_so101_qwen_tool_plan.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --task "pick and lift the green cube" \
  --object "green cube" \
  --output _workspace/qwen_plans/green_cube_edge_chain.json
```

The output is a validated primitive plan JSON that a closed-loop executor can
consume later.

## Qwen + SmolVLA E2E smoke

The default E2E unit test uses a saved Qwen response so it still runs when no
Qwen3 server is listening. It verifies the same validated tool order and checks
that the derived primitive-chain prompt is passed to the SmolVLA probe:

```bash
PYTHONPATH=src .venv/bin/python -B -m unittest tests.test_qwen_smolvla_e2e
```

The saved response lives at:

```text
configs/agent/qwen3_so101_tool_planner_mock_response.json
```

The real-model E2E branch remains gated so normal unit tests do not download or
load large models. It requires an OpenAI-compatible Qwen endpoint and a loadable
LeRobot SmolVLA checkpoint:

```bash
RUN_QWEN_SMOLVLA_E2E=1 \
QWEN_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
QWEN_MODEL=Qwen/Qwen3-8B \
SMOLVLA_MODEL_ID=lerobot/smolvla_base \
PYTHONPATH=src .venv/bin/python -B -m unittest tests.test_qwen_smolvla_e2e
```

To permit Hugging Face downloads in a networked RunPod/local environment, add:

```bash
SMOLVLA_ALLOW_DOWNLOAD=1
```

The equivalent direct runner writes `qwen_tool_plan.json`,
`qwen_smolvla_e2e_report.json`, and the SmolVLA rollout artifacts:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_so101_qwen_smolvla_e2e.py \
  --qwen-base-url http://127.0.0.1:8000/v1 \
  --qwen-model Qwen/Qwen3-8B \
  --smolvla-model-id lerobot/smolvla_base \
  --output-dir _workspace/qwen_smolvla_e2e \
  --rollout-steps 1 \
  --require-pass
```

When Qwen is not running, replay the saved planner response instead:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_so101_qwen_smolvla_e2e.py \
  --qwen-response-json configs/agent/qwen3_so101_tool_planner_mock_response.json \
  --smolvla-model-id lerobot/smolvla_base \
  --output-dir _workspace/qwen_smolvla_e2e_mock_qwen \
  --rollout-steps 1 \
  --require-pass
```

## Closed-loop primitive-chain evaluation

For training-time closed-loop checks, use the Qwen plan as the orchestration
layer and route each primitive to its separately trained SmolVLA checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_so101_qwen_closed_loop_eval.py \
  --qwen-base-url http://127.0.0.1:1234/v1 \
  --qwen-model qwen3-vl-8b-instruct-mlx \
  --primitive-policy move_over_cube_edge=<move_checkpoint>/pretrained_model \
  --primitive-policy align_fixed_jaw_cube_edge=<align_checkpoint>/pretrained_model \
  --primitive-policy grip_from_edge_cube=<grip_checkpoint>/pretrained_model \
  --episodes 8 \
  --max-steps-per-primitive 90 \
  --device cuda \
  --output-dir _workspace/qwen_so101_closed_loop/train_eval
```

The runner keeps one SO101 environment alive per episode and executes the
validated `move -> align -> pick_up` plan in sequence. Qwen chooses and
validates the primitive chain; SmolVLA checkpoints produce the robot actions.

Use `--plan-only` for CI/local preflight when SO101-Nexus or SmolVLA weights are
not available:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_so101_qwen_closed_loop_eval.py \
  --qwen-response-json configs/agent/qwen3_so101_tool_planner_mock_response.json \
  --primitive-policy move_over_cube_edge=<move_checkpoint>/pretrained_model \
  --primitive-policy align_fixed_jaw_cube_edge=<align_checkpoint>/pretrained_model \
  --primitive-policy grip_from_edge_cube=<grip_checkpoint>/pretrained_model \
  --plan-only \
  --require-pass
```

## Primitive Dataset Training Plan

Canonical name: `primitive training with qwen validation v1`.

For `scenario=pick_up_cube` and `execution_policy=qwen_edge_chain`, train one
SmolVLA checkpoint on the three primitive datasets together:

```text
move_over_cube_edge/train
align_fixed_jaw_cube_edge/train
grip_from_edge_cube/train
```

This is not three independent policies. The combined dataset config is:

```text
configs/so101/training_datasets/qwen_edge_primitives.json
```

Generate the concrete run plan:

```bash
PYTHONPATH=src .venv/bin/python scripts/plan_so101_qwen_edge_primitive_training.py \
  --runtime-platform linux \
  --base-train-config <base_smolvla_train_config.json> \
  --steps 50000 \
  --output _workspace/so101_qwen_edge_training/plan.json
```

The plan contains:

- one train command over the three primitive train splits declared through
  `hf_merge_sources`;
- launcher-managed `hf_merge_sources` resolution for both train and validation,
  without a separate manual pre-merge command;
- one final Qwen validation command that routes every Qwen primitive prompt to the
  same trained checkpoint with `--policy-path`.

On local macOS, run this outside the Codex sandbox with
`--runtime-platform macos`; the launcher then selects MPS for training.

The final evaluation row should be recorded as:

```text
name=primitive training with qwen validation v1
scenario=pick_up_cube
execution_policy=qwen_edge_chain
training_policy=single_smolvla_checkpoint_trained_on_three_primitive_datasets
```
