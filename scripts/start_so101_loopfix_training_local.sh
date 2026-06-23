#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python scripts/start_so101_training.py start \
  --json \
  --dataset-config configs/so101/training_datasets/qwen_edge_primitives.json \
  --run-dir _workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/qwen_edge_primitives_resume_009632_loopfix_30000 \
  --host 0.0.0.0 \
  --tensorboard-port 6015 \
  --runtime-platform macos \
  --training-device mps \
  --closed-loop-every-epochs 1 \
  --closed-loop-episodes 5 \
  --closed-loop-steps 120 \
  --closed-loop-env-id MuJoCoPickLift-v1 \
  --closed-loop-mujoco-gl glfw \
  --closed-loop-runner qwen_chain \
  --closed-loop-policy best_or_periodic \
  --closed-loop-subgoal-chain-mode valid-mask \
  --closed-loop-valid-mask-checkpoint _workspace/so101_valid_mask_head/qwen_edge_primitives/valid_mask_head.pt \
  --closed-loop-policy-n-action-steps 15 \
  --closed-loop-policy-num-steps 10 \
  --record-loop-artifacts \
  --render-loop-media \
  --loop-artifact-width 128 \
  --loop-artifact-height 128 \
  --loop-artifact-fps 12 \
  --max-monitored-checkpoints 160 \
  --hf-local-files-only \
  --skip-hf-dataset-download \
  --python .venv/bin/python \
  -- \
  --config_path=_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/qwen_edge_primitives_suite3_resume_009408_aug_full_virtual_statsfix_30000/model/checkpoints/009632/pretrained_model/train_config.json \
  --resume=true \
  --so101-resume-checkpoint-path=_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/qwen_edge_primitives_suite3_resume_009408_aug_full_virtual_statsfix_30000/model/checkpoints/009632 \
  --steps=30000
