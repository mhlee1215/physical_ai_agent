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
PYTHONPATH=src python scripts/build_so101_qwen_tool_plan.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --task "pick and lift the green cube" \
  --object "green cube" \
  --output _workspace/qwen_plans/green_cube_edge_chain.json
```

The output is a validated primitive plan JSON that a closed-loop executor can
consume later.
