# LIBERO RunPod Variants

These scripts are exploratory LIBERO throughput, retry, routing, and sweep
drivers. They are not the canonical paper-facing evaluation entrypoints.

Core SmolVLA evaluation entrypoints live in `scripts/`:

- `scripts/eval_smolvla_libero_linux.sh`
- `scripts/eval_smolvla_metaworld_linux.sh`

Both core runners generate their final `lerobot-eval` command through:

```bash
python -m physical_ai_agent.evaluation.lerobot_eval
```

Keep new benchmark-level evaluation behavior in the shared module first. Add
scripts here only for experiments that intentionally vary throughput, retry
budget, routing, or subset scheduling around the core LIBERO evaluator.
