# myCobot 280 SmolVLA Tiny Fine-Tune Smoke

This note records the first true optimizer-step readiness gate for the myCobot
280 SmolVLA lane. It complements the existing supervised-loss smoke:

- supervised-loss smoke: model + dataset load, one forward/loss computation
- tiny fine-tune smoke: model + dataset load, forward, `backward()`,
  `optimizer.step()`, checkpoint/log artifact writes

## Command

Use the native LeRobot export produced by the readiness converter:

```bash
PYTHONPATH=src:. _workspace/local_envs/lerobot_py312/bin/python \
  scripts/train_mycobot280_smolvla_tiny.py \
  --dataset-root _workspace/mycobot280_lerobot/ground_pickup_tiny_smoke_native \
  --dataset-repo-id physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke \
  --policy-path lerobot/smolvla_base \
  --output-dir _workspace/mycobot280_training/ground_pickup_tiny_smoke \
  --batch-size 1 \
  --steps 2 \
  --device cpu \
  --local-files-only \
  --require-runtime
```

## Expected Artifacts

The run is only passed when it writes:

- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/tiny_finetune.json`
- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/train.log`
- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/checkpoints/latest/pretrained_model/`
- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/checkpoints/latest/optimizer_state.pt`
- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/checkpoints/latest/training_state.json`
- `_workspace/mycobot280_training/ground_pickup_tiny_smoke/tensorboard/` when TensorBoard is available

## Claim Boundary

This is a plumbing gate only. It proves that the myCobot native LeRobot dataset
can drive a SmolVLA optimizer step and checkpoint write. It does not prove
closed-loop task success, randomized-data generalization, or publication-level
learning improvement.
