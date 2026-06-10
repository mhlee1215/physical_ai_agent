# Risk1-B External VLM Subgoal Generator

Risk1-B validates candidate generation only: an external VLM proposes grounded
subgoal prompts, frozen SmolVLA produces one action chunk per prompt, and
diversity is compared against Risk1-A and native-noise references. Do not count
mock or fixture outputs as Risk1-B PASS evidence.

## Required Schema

Each generated subgoal must include:

- `subgoal_text`
- `strategy_axis`
- `target_object`
- `target_region_or_point`
- `stop_condition`
- `confidence`
- optional `rationale`

The generator writes `risk1b_subgoals_<model>_<suite>_task<id>_seed<seed>.json`
plus a sibling `.raw.txt` file with the raw model output.

## Local Contract Smoke

```bash
PYTHONPATH=src python3 -B scripts/generate_risk1b_vlm_subgoals.py \
  --backend mock \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite libero_goal \
  --task-id 6 \
  --seed 1201 \
  --output-dir /tmp/risk1b_generator_mock \
  --json
```

This checks schema plumbing only. The output provenance is `mock_contract` and
cannot drive a Risk1-B PASS claim.

## Required Order

Risk1-B actual validation must run in this order:

1. Capture actual LIBERO/LeRobot context artifacts.
2. Generate external VLM subgoal JSON from those artifacts.
3. Run the frozen SmolVLA Risk1-B probe with `--risk1b-subgoals-json`.

If context artifacts are missing or have `provenance.actual_context=false`,
the transformers generator must stay blocked. Do not fabricate context for a
Risk1-B PASS claim.

## RunPod Context Capture

```bash
PYTHONPATH=src /root/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/capture_risk1b_context.py \
  --backend libero \
  --suite libero_goal \
  --task-id 6 \
  --seed 1201 \
  --policy-path lerobot/smolvla_libero \
  --policy-num-steps 10 \
  --policy-n-action-steps 15 \
  --renderer-backend egl \
  --output-dir _workspace/runpod_results/ita_risk_probes/risk1b_context \
  --json
```

Expected outputs:

- `_workspace/runpod_results/ita_risk_probes/risk1b_context/contact_sheet_task6_seed1201.png`
- `_workspace/runpod_results/ita_risk_probes/risk1b_context/context_task6_seed1201.json`

Local mock context capture is available for schema plumbing only:

```bash
PYTHONPATH=src python3 -B scripts/capture_risk1b_context.py \
  --backend mock \
  --output-dir /tmp/risk1b_context_mock \
  --json
```

The mock context has `provenance.actual_context=false` and the real
transformers generator will reject it.

## RunPod Generation

Run this after the RunPod VLM environment has `torch`, `transformers`, and
`pillow` available and a start-observation/contact-sheet image or task context
is ready.

Before loading model weights, run the dependency/class preflight. This checks
`torch`, `PIL.Image`, `transformers`, `AutoProcessor`, and the available model
loader class (`AutoModelForImageTextToText`, `AutoModelForVision2Seq`, or
`AutoModelForCausalLM`):

```bash
PYTHONPATH=src /root/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/generate_risk1b_vlm_subgoals.py \
  --backend transformers \
  --dependency-check-only \
  --json
```

If this fails, report `RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED` with stderr and do
not run the generator. If it passes but model loading fails, report the model
load/download error separately.

```bash
PYTHONPATH=src /root/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/generate_risk1b_vlm_subgoals.py \
  --backend transformers \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite libero_goal \
  --task-id 6 \
  --seed 1201 \
  --num-subgoals 5 \
  --task-description "Complete the LIBERO goal task from the current observation." \
  --context-image _workspace/runpod_results/ita_risk_probes/risk1b_context/contact_sheet_task6_seed1201.png \
  --context-json _workspace/runpod_results/ita_risk_probes/risk1b_context/context_task6_seed1201.json \
  --output-dir _workspace/runpod_results/ita_risk_probes \
  --json
```

Supported model ids:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `google/gemma-3-4b-it`

If model loading or generation fails, stop before the SmolVLA probe and report
the generator error. Invalid JSON exits nonzero and must not drive SmolVLA.

## SmolVLA Risk1-B Probe

Use the generated JSON as input:

```bash
PYTHONPATH=src /root/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/run_imagine_then_act_risk_probes.py \
  --preset runpod-libero-smoke \
  --backend libero-contract \
  --suite libero_goal \
  --task-ids 6 \
  --seed 1201 \
  --num-candidates 5 \
  --chunk-steps 15 \
  --action-dim 7 \
  --policy-path lerobot/smolvla_libero \
  --policy-num-steps 10 \
  --policy-n-action-steps 15 \
  --renderer-backend egl \
  --risk1b-vlm-subgoals \
  --risk1b-generator-backend json \
  --risk1b-model Qwen/Qwen2.5-VL-7B-Instruct \
  --risk1b-subgoals-json _workspace/runpod_results/ita_risk_probes/risk1b_subgoals_qwen2_5_vl_7b_instruct_libero_goal_task6_seed1201.json \
  --output-dir _workspace/runpod_results/ita_risk_probes/risk1b_vlm_subgoals_seed1201 \
  --json
```
