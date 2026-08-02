from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from physical_ai_agent.policies.peft_loader import load_policy_artifact
from physical_ai_agent.policies.lerobot_policy_runner import (
    _load_base_policy_from_pretrained,
    _set_policy_runtime_device,
)
from physical_ai_agent.policies.smolvla_real import _peft_policy_config_kwargs


class _FakePolicy:
    def __init__(self, source: str, policy_config_path: str | None = None) -> None:
        self.source = source
        self.policy_config_path = policy_config_path
        self.device = None

    def to(self, device: str) -> _FakePolicy:
        self.device = device
        return self


class SO101PeftLoaderTest(unittest.TestCase):
    def test_policy_runner_runtime_device_overrides_checkpoint_device(self) -> None:
        policy = types.SimpleNamespace(config=types.SimpleNamespace(device="mps"))

        _set_policy_runtime_device(policy, "cpu")

        self.assertEqual(policy.config.device, "cpu")

    def test_policy_runner_adapter_config_uses_lerobot_choice_loader(self) -> None:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"type": "smolvla"}), encoding="utf-8")
            with patch.object(SmolVLAPolicy, "from_pretrained", return_value="loaded") as loader:
                loaded = _load_base_policy_from_pretrained(
                    policy_cls=SmolVLAPolicy,
                    policy_path="base-policy",
                    policy_config_path=tmp,
                    local_files_only=False,
                    map_location="cpu",
                    device="cpu",
                )

        self.assertEqual(loaded, "loaded")
        self.assertIsInstance(loader.call_args.kwargs["config"], SmolVLAPolicy.config_class)

    def test_adapter_policy_config_uses_lerobot_choice_loader(self) -> None:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"type": "smolvla"}), encoding="utf-8")
            kwargs = _peft_policy_config_kwargs(
                SmolVLAPolicy,
                tmp,
                local_files_only=True,
            )

        self.assertIsInstance(kwargs["config"], SmolVLAPolicy.config_class)

    def test_normal_checkpoint_uses_direct_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_policy_artifact(
                tmp,
                base_loader=_FakePolicy,
                device="cpu",
            )

        self.assertEqual(loaded.source, tmp)
        self.assertIsNone(loaded.policy_config_path)

    def test_adapter_checkpoint_loads_declared_base_then_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/stable/base-policy"}),
                encoding="utf-8",
            )
            calls: list[tuple[object, str, bool]] = []

            class FakePeftModel:
                @staticmethod
                def from_pretrained(base_policy: object, path: str, *, is_trainable: bool):
                    calls.append((base_policy, path, is_trainable))
                    return base_policy

            fake_peft = types.SimpleNamespace(PeftModel=FakePeftModel)
            previous = sys.modules.get("peft")
            sys.modules["peft"] = fake_peft
            try:
                loaded = load_policy_artifact(
                    str(adapter),
                    base_loader=_FakePolicy,
                    device="mps",
                )
            finally:
                if previous is None:
                    sys.modules.pop("peft", None)
                else:
                    sys.modules["peft"] = previous

        self.assertEqual(loaded.source, "/stable/base-policy")
        self.assertEqual(loaded.policy_config_path, str(adapter))
        self.assertEqual(loaded.device, "mps")
        self.assertEqual(calls[0][1], str(adapter))
        self.assertFalse(calls[0][2])


if __name__ == "__main__":
    unittest.main()
