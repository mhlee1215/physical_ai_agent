# SO101 SmolVLA Evaluator Action Contract Audit

Date: 2026-06-24

## Context

The `move_and_align_cube_edge_only` training run showed decreasing supervised validation loss while closed-loop success stayed low or worsened. The suspected failure mode was an evaluator-side action contract mismatch.

Run audited:

- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813`
- policy checkpoint: `model/checkpoints/best_val_loss/pretrained_model`
- closed-loop start set: `move_and_align_cube_edge_loop_validation10_ego_wrist_256_seed123500`

## Bug Found

The Qwen closed-loop evaluator loaded `SmolVLAPolicy` directly and called `policy.select_action(batch)`, then passed the result to `env.step(action)` after only truncating/padding action length.

That bypassed the saved LeRobot processor contract:

- policy preprocessor: state/image/language batching, tokenization, state normalization
- policy postprocessor: action unnormalization

The checkpoint contains:

- `policy_preprocessor.json`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

The direct evaluator path skipped the postprocessor, so normalized-ish model outputs were executed as raw SO101 joint targets.

## Fix

Updated `src/physical_ai_agent/agent_core/qwen_so101_closed_loop.py` to execute policy actions through `LeRobotPolicyRunner` when available:

`raw observation -> saved preprocessor -> policy.select_action -> saved postprocessor -> env.step`

The trace now records:

- `policy_output.processor_raw_action`
- `policy_output.processor_postprocessed_action`
- `policy_output.action`
- `policy_output.processor_source`
- preprocessor/postprocessor step names

Added evaluator modes in `scripts/run_so101_qwen_closed_loop_eval.py`:

- `--action-contract-mode processor` (default)
- `--action-contract-mode legacy`
- `--action-contract-mode processor_dataset_clamp`

The report now includes `action_contract` summary with dataset/env bounds and out-of-range ratios.

## Quantitative Check

Single-episode comparison, same checkpoint/start:

| Mode | Dataset out-of-range | Env out-of-range | Final tcp_to_obj_dist |
|---|---:|---:|---:|
| legacy | 0.9000 | 0.1875 | 0.1778 |
| processor | 0.0375 | 0.0000 | 0.1818 |
| processor_dataset_clamp | 0.0000 | 0.0000 | 0.1809 |

Ten-episode comparison:

| Mode | threshold pass (`tcp_to_obj_dist <= 0.06`) | mean dist | min dist | max dist | Dataset out-of-range | Env out-of-range |
|---|---:|---:|---:|---:|---:|---:|
| processor | 1/10 | 0.1194 | 0.0598 | 0.2384 | 0.16875 | 0.0 |
| processor_dataset_clamp | 2/10 | 0.1159 | 0.0532 | 0.2421 | 0.0 | 0.0 |

Artifacts:

- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813/action_contract_audit/legacy_ep1_compare/qwen_closed_loop_eval_report.json`
- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813/action_contract_audit/processor_ep1_compare/qwen_closed_loop_eval_report.json`
- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813/action_contract_audit/processor_dataset_clamp_ep1_compare/qwen_closed_loop_eval_report.json`
- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813/action_contract_audit/processor_ep10_contract_eval/qwen_closed_loop_eval_report.json`
- `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_only_val25_loop10_20260624_011813/action_contract_audit/processor_dataset_clamp_ep10_contract_eval/qwen_closed_loop_eval_report.json`

## Interpretation

The evaluator action contract bug was real and severe. The old path executed actions outside env range and far outside the training action distribution.

After the fix, env action range is valid. However, closed-loop performance remains low: 1/10 with the processor path, 2/10 with dataset clamp. Therefore, action contract mismatch was a blocker, but not the only reason the policy fails.

Remaining likely issues:

1. The learned policy is still not closed-loop stable enough for the move-and-align primitive.
2. Even after unnormalization, some postprocessed actions drift outside the dataset action range unless clamped.
3. Validation loss can continue decreasing while rollout performance does not improve, because small action errors compound under closed-loop execution.
4. The primitive may need either better closed-loop recovery data, stricter action smoothness/temporal consistency, or a shorter/easier subgoal boundary.

## Verification

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_so101_action_contract.py'`
- `.venv/bin/python -m py_compile src/physical_ai_agent/agent_core/qwen_so101_closed_loop.py src/physical_ai_agent/policies/lerobot_policy_runner.py scripts/run_so101_qwen_closed_loop_eval.py`
