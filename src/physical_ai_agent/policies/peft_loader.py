from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


BasePolicyLoader = Callable[[str, str | None], Any]


def load_policy_artifact(
    policy_path: str,
    *,
    base_loader: BasePolicyLoader,
    device: str,
) -> Any:
    """Load a normal LeRobot policy or a PEFT adapter checkpoint."""

    adapter_config_path = Path(policy_path) / "adapter_config.json"
    if not adapter_config_path.is_file():
        return base_loader(policy_path, None)
    config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_path = str(config.get("base_model_name_or_path") or "").strip()
    if not base_path:
        raise ValueError(f"PEFT adapter does not declare base_model_name_or_path: {adapter_config_path}")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Loading a LoRA checkpoint requires the LeRobot 'peft' extra") from exc
    # Preserve the adapter's policy config (including its active cameras)
    # instead of silently restoring the base checkpoint's stale contract.
    base_policy = base_loader(base_path, policy_path)
    policy = PeftModel.from_pretrained(base_policy, policy_path, is_trainable=False)
    if hasattr(policy, "to"):
        policy = policy.to(device)
    return policy
