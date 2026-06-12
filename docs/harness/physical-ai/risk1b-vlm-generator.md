# Risk1-B External VLM Strategy-Portfolio Generator

Risk1-B validates candidate generation only: an external VLM proposes a
grounded strategy portfolio for the same immediate LIBERO subgoal, frozen
SmolVLA produces one action chunk per prompt, and diversity is compared against
Risk1-A and native-noise references. Do not count mock or fixture outputs as
Risk1-B PASS evidence.

The generator must not decompose the task into a temporal plan such as "pick,
move, place". All entries should target the same object, same target relation,
and same stop condition while varying the approach strategy. Risk1-B is testing
solution-mode diversity, not plan-step enumeration.

The actual context task description is the source of truth for object roles.
If the context says `put the cream cheese in the bowl`, the manipulated object
is the cream cheese and the bowl is the destination/container. The generator
must not switch to a visually salient or first-listed state key such as the bowl
as the object to pick up. The generator command should still pass
`--task-description`, but the script prioritizes `context_json.task_description`
when actual context is available.

## Required Schema

Each generated strategy entry must include:

- `subgoal_text`
- `strategy_axis`
- `target_object`
- `target_region_or_point`
- `stop_condition`
- `confidence`
- optional `rationale`

The generator writes `risk1b_subgoals_<model>_<suite>_task<id>_seed<seed>.json`
plus a sibling `.raw.txt` file with the raw model output. The filename keeps
`subgoals` for backward compatibility, but the intended semantics are
same-subgoal strategy alternatives.

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

## Task0 Repair-Only Handoff Preflight

For the Risk1-B/C task0 repair-only lane, RunPod Manager should run one bounded
handoff-preflight command instead of a loose sequence of shell steps:

```bash
cd /workspace/physical-ai/physical_ai_agent
/usr/bin/python3.12 -B scripts/runpod_risk1bc_task0_repair_handoff_preflight.py \
  --project-dir /workspace/physical-ai/physical_ai_agent \
  --work-root /workspace/physical-ai \
  --output-dir /workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/ita_risk_probes/risk1bc_task0_repair_handoff_preflight \
  --suite libero_goal \
  --task-id 0 \
  --seed 1201 \
  --renderer-backend osmesa \
  --qwen-readiness-mode model-load \
  --json
```

The wrapper writes one machine-readable final report:
`risk1bc_task0_repair_handoff_preflight.json`. It may return
`ENV_READY_HANDOFF_READY` or `BLOCKED`, with the blocked phase and log paths.
It verifies readiness only and does not run Qwen generation, SmolVLA Risk1-B/C
probes, benchmark evaluation, or any paper-facing experiment.

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

Run this after the actual context artifacts exist. Keep the canonical
LeRobot/LIBERO/SmolVLA env pinned to `torch==2.5.1+cu124`; do not upgrade its
torch stack to satisfy external VLM dependencies. Risk1-B JSON generation is a
separate process and should use a separate VLM env.

Recommended one-time VLM env setup on the RunPod volume:

```bash
cd /workspace/physical-ai/physical_ai_agent
PROJECT_DIR="$PWD" \
WORK_ROOT=/workspace/physical-ai \
VLM_VENV=/workspace/physical-ai/envs/risk1b_vlm_py312 \
RISK1B_VLM_HF_HOME=/tmp/risk1b_vlm_hf_home \
sh scripts/install/runpod_install.sh --component risk1b-vlm
```

The network volume preserves reusable envs, LIBERO config/assets, and fetched
artifacts. Large Qwen/Gemma model weights are cache-on-demand and are not
persistent assets by default. For VLM generation, use pod-local cache such as
`RISK1B_VLM_HF_HOME=/tmp/risk1b_vlm_hf_home`, or delete the model cache after
the generated JSON and artifacts are fetched. Persist Qwen/Gemma weights on the
network volume only with explicit PM approval.

Default VLM env pins:

- `torch==2.5.1+cu124` and `torchvision==0.20.1+cu124`
- `transformers==4.49.0`
- `Pillow>=10,<13`, `accelerate>=1.0,<2`, `huggingface_hub>=0.26,<1.0`
- `qwen-vl-utils[decord]>=0.0.8,<0.1`

Do not use the previously observed `transformers==5.10.2` with
`torch==2.5.1+cu124` for Qwen/Gemma generation: its lazy processor path can
require `torch.float8_e8m0fnu`, which is absent from torch 2.5.1. The generator
preflight reports this as
`RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED_COMPATIBILITY`.

Before loading model weights, run the dependency/class preflight. This checks
`torch`, `PIL.Image`, `transformers`, torch/Transformers float8 compatibility,
`AutoProcessor` or model-specific processor fallbacks, and the available model
loader class (`AutoModelForImageTextToText`, `AutoModelForVision2Seq`, or
`AutoModelForCausalLM`). It does not download model weights:

```bash
HF_HOME=/tmp/risk1b_vlm_hf_home \
HF_HUB_CACHE=/tmp/risk1b_vlm_hf_home/hub \
TRANSFORMERS_CACHE=/tmp/risk1b_vlm_hf_home/transformers \
PYTHONPATH=src /workspace/physical-ai/envs/risk1b_vlm_py312/bin/python -B \
  scripts/generate_risk1b_vlm_subgoals.py \
  --backend transformers \
  --dependency-check-only \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --json
```

If this fails, report `RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED` with stderr and do
not run the generator. If it passes but model loading fails, report the model
load/download error separately.
For Qwen/Gemma, package-level imports are not enough: the preflight also
resolves the processor loader. If `AutoProcessor` fails through Transformers'
lazy loader, the script tries model-specific fallbacks such as
`Qwen2_5_VLProcessor` before reporting
`RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED_PROCESSOR_LOADER`.

```bash
PYTHONPATH=src /workspace/physical-ai/envs/risk1b_vlm_py312/bin/python -B \
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
