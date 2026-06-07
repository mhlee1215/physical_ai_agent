#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lerobot-eval with an instrumented in-episode rollout hook.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--intervention-step", type=int, default=3)
    parser.add_argument("--intervention-scale", type=float, default=1.0)
    args, lerobot_args = parser.parse_known_args()

    args.trace_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace_path.write_text("", encoding="utf-8")

    from lerobot.scripts import lerobot_eval

    lerobot_eval.rollout = build_instrumented_rollout(
        trace_path=args.trace_path,
        intervention_step=args.intervention_step,
        intervention_scale=args.intervention_scale,
    )
    sys.argv = ["lerobot-eval", *lerobot_args]
    lerobot_eval.main()


def build_instrumented_rollout(trace_path: Path, intervention_step: int, intervention_scale: float):
    def instrumented_rollout(
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
        import einops
        import numpy as np
        import torch
        from torch import nn
        from tqdm import trange

        from lerobot.envs import check_env_attributes_and_types, preprocess_observation
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.utils import inside_slurm

        assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

        policy.reset()
        observation, _info = env.reset(seed=seeds)
        if render_callback is not None:
            render_callback(env)

        all_observations = []
        all_actions = []
        all_rewards = []
        all_successes = []
        all_dones = []
        trace_records: list[dict[str, Any]] = []

        step = 0
        done = np.array([False] * env.num_envs)
        max_steps = int(env.call("_max_episode_steps")[0])
        progbar = trange(
            max_steps,
            desc=f"Running instrumented rollout with at most {max_steps} steps",
            disable=inside_slurm(),
            leave=False,
        )
        check_env_attributes_and_types(env)
        while not np.all(done) and step < max_steps:
            observation = preprocess_observation(observation)
            if return_observations:
                all_observations.append(deepcopy(observation))

            try:
                observation["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    observation["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""] * env.num_envs

            observation = env_preprocessor(observation)
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            verifier_triggered = step == intervention_step and not np.all(done)
            intervention_type = None
            if verifier_triggered:
                intervention_type = f"scale_action_{intervention_scale:g}"
                action = action * intervention_scale

            action_numpy = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env)

            if "final_info" in info:
                final_info = info["final_info"]
                if not isinstance(final_info, dict):
                    raise RuntimeError("Unsupported final_info format: expected dict.")
                successes = final_info["is_success"].tolist()
            elif "is_success" in info:
                is_success = info["is_success"]
                successes = is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
            else:
                successes = [False] * env.num_envs

            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_numpy))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor(successes))
            trace_records.append(
                {
                    "step": step,
                    "action_shape": list(action_numpy.shape),
                    "action_norm": float(np.linalg.norm(action_numpy)),
                    "reward": _tolist(reward),
                    "terminated": _tolist(terminated),
                    "truncated": _tolist(truncated),
                    "done": _tolist(done),
                    "successes": [bool(value) for value in successes],
                    "verifier_triggered": bool(verifier_triggered),
                    "verifier_reason": "timeout_risk_step_threshold" if verifier_triggered else "not_triggered",
                    "intervention_type": intervention_type,
                }
            )

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()

        if return_observations:
            observation = preprocess_observation(observation)
            all_observations.append(deepcopy(observation))

        ret = {
            ACTION: torch.stack(all_actions, dim=1),
            "reward": torch.stack(all_rewards, dim=1),
            "success": torch.stack(all_successes, dim=1),
            "done": torch.stack(all_dones, dim=1),
        }
        if return_observations:
            stacked_observations = {}
            for key in all_observations[0]:
                stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
            ret[OBS_STR] = stacked_observations

        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "rollout_summary",
                        "seeds": list(seeds) if seeds else None,
                        "max_steps": max_steps,
                        "action_step_count": len(trace_records),
                        "verifier_trigger_count": sum(1 for record in trace_records if record["verifier_triggered"]),
                        "intervention_count": sum(1 for record in trace_records if record["intervention_type"]),
                        "success": bool(any(any(record["successes"]) for record in trace_records)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for record in trace_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        if hasattr(policy, "use_original_modules"):
            policy.use_original_modules()
        return ret

    return instrumented_rollout


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_tolist(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    main()
