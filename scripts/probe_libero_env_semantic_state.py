#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe semantic state access inside a LeRobot LIBERO rollout.")
    parser.add_argument("--output-path", type=Path, required=True)
    args, lerobot_args = parser.parse_known_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("", encoding="utf-8")

    from lerobot.scripts import lerobot_eval

    lerobot_eval.rollout = build_probe_rollout(args.output_path)
    sys.argv = ["lerobot-eval", *lerobot_args]
    lerobot_eval.main()


def build_probe_rollout(output_path: Path):
    def probe_rollout(
        env,
        policy,
        env_preprocessor,
        env_postprocessor,
        preprocessor,
        postprocessor,
        seeds=None,
        return_observations=False,
        render_callback=None,
    ) -> dict:
        import numpy as np
        import torch
        from torch import nn

        from lerobot.envs import preprocess_observation
        from lerobot.utils.constants import ACTION

        assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

        policy.reset()
        observation, reset_info = env.reset(seed=seeds)
        records: list[dict[str, Any]] = [
            {
                "event": "reset_probe",
                "env_type": type(env).__name__,
                "env_module": type(env).__module__,
                "num_envs": getattr(env, "num_envs", None),
                "reset_info": summarize_value(reset_info),
                "call_probe": call_probe(env),
                "attr_probe": attr_probe(env),
                "internal_probe": internal_probe(env),
            }
        ]

        observation = preprocess_observation(observation)
        observation["task"] = safe_task_descriptions(env)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        action_transition = env_postprocessor({ACTION: action})
        action_numpy = action_transition[ACTION].to("cpu").numpy()
        next_observation, reward, terminated, truncated, info = env.step(action_numpy)
        records.append(
            {
                "event": "step_probe",
                "reward": summarize_value(reward),
                "terminated": summarize_value(terminated),
                "truncated": summarize_value(truncated),
                "info": summarize_value(info),
                "next_observation": summarize_value(next_observation),
                "call_probe": call_probe(env),
                "attr_probe": attr_probe(env),
                "internal_probe": internal_probe(env),
            }
        )

        output_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")

        return {
            ACTION: torch.from_numpy(np.expand_dims(action_numpy, axis=1)),
            "reward": torch.from_numpy(np.expand_dims(reward, axis=1)),
            "success": torch.zeros((env.num_envs, 1), dtype=torch.bool),
            "done": torch.ones((env.num_envs, 1), dtype=torch.bool),
        }

    return probe_rollout


def call_probe(env: Any) -> dict[str, Any]:
    names = [
        "_max_episode_steps",
        "task_description",
        "task",
        "get_sim_state",
        "get_robot_state",
        "get_object_state",
        "get_state",
    ]
    result: dict[str, Any] = {}
    for name in names:
        try:
            result[name] = summarize_value(env.call(name))
        except Exception as exc:  # noqa: BLE001 - this is a probe script.
            result[name] = {"error": type(exc).__name__, "message": str(exc)[:500]}
    return result


def attr_probe(env: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"direct_attrs": sorted(key for key in dir(env) if interesting_name(key))[:200]}
    for method_name in ("get_attr", "call"):
        result[f"has_{method_name}"] = hasattr(env, method_name)
    for attr in ("envs", "unwrapped", "single_action_space", "single_observation_space"):
        try:
            result[attr] = summarize_value(getattr(env, attr))
        except Exception as exc:  # noqa: BLE001
            result[attr] = {"error": type(exc).__name__, "message": str(exc)[:500]}
    return result


def internal_probe(env: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        inner = env.envs[0]
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "message": str(exc)[:500]}
    result["inner_type"] = f"{type(inner).__module__}.{type(inner).__name__}"
    result["inner_attrs"] = sorted(key for key in dir(inner) if interesting_name(key))[:200]
    offscreen = getattr(inner, "_env", None)
    result["offscreen_type"] = None if offscreen is None else f"{type(offscreen).__module__}.{type(offscreen).__name__}"
    if offscreen is None:
        return result
    for attr in ("robots", "sim", "model", "env"):
        try:
            result[f"offscreen_{attr}"] = summarize_value(getattr(offscreen, attr))
        except Exception as exc:  # noqa: BLE001
            result[f"offscreen_{attr}"] = {"error": type(exc).__name__, "message": str(exc)[:500]}
    try:
        raw_obs = offscreen.env._get_observations()
        result["raw_obs_keys"] = sorted(raw_obs.keys())
        result["raw_obs_state_keys"] = {
            key: summarize_value(value)
            for key, value in raw_obs.items()
            if any(part in key.lower() for part in ("pos", "quat", "state", "joint", "gripper", "eef", "cream", "bowl"))
        }
    except Exception as exc:  # noqa: BLE001
        result["raw_obs"] = {"error": type(exc).__name__, "message": str(exc)[:500]}
    try:
        sim = offscreen.sim
        model = sim.model
        result["body_names"] = summarize_names(model, "body")
        result["site_names"] = summarize_names(model, "site")
        result["geom_names"] = summarize_names(model, "geom")
    except Exception as exc:  # noqa: BLE001
        result["model_names"] = {"error": type(exc).__name__, "message": str(exc)[:500]}
    return result


def summarize_names(model: Any, prefix: str) -> list[str]:
    count = int(getattr(model, f"n{prefix}"))
    names = []
    for idx in range(count):
        try:
            name = model.id2name(idx, prefix)
        except Exception:  # noqa: BLE001
            name = None
        if name:
            names.append(str(name))
    return names[:300]


def safe_task_descriptions(env: Any) -> list[str]:
    try:
        return list(env.call("task_description"))
    except Exception:  # noqa: BLE001
        try:
            return list(env.call("task"))
        except Exception:  # noqa: BLE001
            return [""] * env.num_envs


def interesting_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in ("sim", "model", "data", "obj", "body", "site", "robot", "grip", "env"))


def summarize_value(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return repr(value)[:300]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): summarize_value(item, depth + 1) for key, item in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [summarize_value(item, depth + 1) for item in list(value)[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "module": type(value).__module__, "repr": repr(value)[:500]}


if __name__ == "__main__":
    main()
