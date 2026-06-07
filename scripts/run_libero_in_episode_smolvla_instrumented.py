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
    parser.add_argument(
        "--trigger-mode",
        choices=("fixed_step", "action_norm_threshold", "fixed_or_action_norm", "semantic_no_progress"),
        default="fixed_step",
        help="Verifier trigger rule used before env.step().",
    )
    parser.add_argument(
        "--action-norm-threshold",
        type=float,
        default=1.0,
        help="Trigger threshold for action-norm verifier modes.",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=(
            "none",
            "scale",
            "clamp",
            "smooth",
            "policy_reset",
            "semantic_reach_target",
            "semantic_push_receptacle",
        ),
        default="scale",
        help="Action-space intervention applied after the verifier triggers.",
    )
    parser.add_argument("--intervention-scale", type=float, default=1.0)
    parser.add_argument("--action-clamp-norm", type=float, default=1.0)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="For smooth mode: alpha * previous_action + (1 - alpha) * current_action.",
    )
    parser.add_argument("--target-object-key", default="cream_cheese_1_pos")
    parser.add_argument("--receptacle-object-key", default="akita_black_bowl_1_pos")
    parser.add_argument("--semantic-min-step", type=int, default=40)
    parser.add_argument("--semantic-window", type=int, default=30)
    parser.add_argument("--semantic-progress-threshold", type=float, default=0.01)
    parser.add_argument("--semantic-reach-gain", type=float, default=4.0)
    parser.add_argument("--semantic-gripper-command", type=float, default=1.0)
    args, lerobot_args = parser.parse_known_args()

    args.trace_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace_path.write_text("", encoding="utf-8")

    from lerobot.scripts import lerobot_eval

    lerobot_eval.rollout = build_instrumented_rollout(
        trace_path=args.trace_path,
        intervention_step=args.intervention_step,
        trigger_mode=args.trigger_mode,
        action_norm_threshold=args.action_norm_threshold,
        intervention_mode=args.intervention_mode,
        intervention_scale=args.intervention_scale,
        action_clamp_norm=args.action_clamp_norm,
        smooth_alpha=args.smooth_alpha,
        target_object_key=args.target_object_key,
        receptacle_object_key=args.receptacle_object_key,
        semantic_min_step=args.semantic_min_step,
        semantic_window=args.semantic_window,
        semantic_progress_threshold=args.semantic_progress_threshold,
        semantic_reach_gain=args.semantic_reach_gain,
        semantic_gripper_command=args.semantic_gripper_command,
    )
    sys.argv = ["lerobot-eval", *lerobot_args]
    lerobot_eval.main()


def build_instrumented_rollout(
    trace_path: Path,
    intervention_step: int,
    trigger_mode: str,
    action_norm_threshold: float,
    intervention_mode: str,
    intervention_scale: float,
    action_clamp_norm: float,
    smooth_alpha: float,
    target_object_key: str,
    receptacle_object_key: str,
    semantic_min_step: int,
    semantic_window: int,
    semantic_progress_threshold: float,
    semantic_reach_gain: float,
    semantic_gripper_command: float,
):
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
        semantic_history: list[dict[str, Any]] = []
        previous_action = None

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
            semantic_state = collect_semantic_state(
                env,
                target_object_key=target_object_key,
                receptacle_object_key=receptacle_object_key,
            )
            semantic_history.append(semantic_state)
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
            policy_reset_reselected = False

            pre_intervention_action_norm = float(torch.linalg.vector_norm(action).item())
            verifier_triggered, verifier_reason = should_trigger_verifier(
                step=step,
                action_norm=pre_intervention_action_norm,
                intervention_step=intervention_step,
                trigger_mode=trigger_mode,
                action_norm_threshold=action_norm_threshold,
                is_done=bool(np.all(done)),
                semantic_history=semantic_history,
                semantic_min_step=semantic_min_step,
                semantic_window=semantic_window,
                semantic_progress_threshold=semantic_progress_threshold,
            )
            intervention_type = None
            if verifier_triggered and intervention_mode != "none":
                intervention_type = format_intervention_type(
                    mode=intervention_mode,
                    intervention_scale=intervention_scale,
                    action_clamp_norm=action_clamp_norm,
                    smooth_alpha=smooth_alpha,
                )
                if intervention_mode == "scale":
                    action = action * intervention_scale
                elif intervention_mode == "clamp":
                    action = clamp_action_norm(action, action_clamp_norm)
                elif intervention_mode == "smooth":
                    if previous_action is not None:
                        action = smooth_alpha * previous_action + (1.0 - smooth_alpha) * action
                    else:
                        intervention_type = "smooth_skipped_no_previous_action"
                elif intervention_mode == "policy_reset":
                    policy.reset()
                    with torch.inference_mode():
                        action = policy.select_action(observation)
                    action = postprocessor(action)
                    action_transition = {ACTION: action}
                    action_transition = env_postprocessor(action_transition)
                    action = action_transition[ACTION]
                    policy_reset_reselected = True
                elif intervention_mode == "semantic_reach_target":
                    action = semantic_reach_target_action(
                        action,
                        semantic_state=semantic_state,
                        gain=semantic_reach_gain,
                        gripper_command=semantic_gripper_command,
                    )
                elif intervention_mode == "semantic_push_receptacle":
                    action = semantic_push_receptacle_action(
                        action,
                        semantic_state=semantic_state,
                        gain=semantic_reach_gain,
                        gripper_command=semantic_gripper_command,
                    )
            post_intervention_action_norm = float(torch.linalg.vector_norm(action).item())

            action_numpy = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
            previous_action = action.detach().clone()

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
                    "pre_intervention_action_norm": pre_intervention_action_norm,
                    "post_intervention_action_norm": post_intervention_action_norm,
                    "reward": _tolist(reward),
                    "terminated": _tolist(terminated),
                    "truncated": _tolist(truncated),
                    "done": _tolist(done),
                    "successes": [bool(value) for value in successes],
                    "verifier_triggered": bool(verifier_triggered),
                    "verifier_reason": verifier_reason,
                    "intervention_type": intervention_type,
                    "policy_reset_reselected": policy_reset_reselected,
                    "semantic_state": semantic_state,
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
                        "trigger_mode": trigger_mode,
                        "action_norm_threshold": action_norm_threshold,
                        "intervention_mode": intervention_mode,
                        "intervention_step": intervention_step,
                        "intervention_scale": intervention_scale,
                        "action_clamp_norm": action_clamp_norm,
                        "smooth_alpha": smooth_alpha,
                        "target_object_key": target_object_key,
                        "receptacle_object_key": receptacle_object_key,
                        "semantic_min_step": semantic_min_step,
                        "semantic_window": semantic_window,
                        "semantic_progress_threshold": semantic_progress_threshold,
                        "semantic_reach_gain": semantic_reach_gain,
                        "semantic_gripper_command": semantic_gripper_command,
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


def should_trigger_verifier(
    *,
    step: int,
    action_norm: float,
    intervention_step: int,
    trigger_mode: str,
    action_norm_threshold: float,
    is_done: bool,
    semantic_history: list[dict[str, Any]] | None = None,
    semantic_min_step: int = 0,
    semantic_window: int = 1,
    semantic_progress_threshold: float = 0.0,
) -> tuple[bool, str]:
    if is_done:
        return False, "episode_done"
    fixed_step = step == intervention_step
    action_norm_spike = action_norm >= action_norm_threshold
    if trigger_mode == "fixed_step" and fixed_step:
        return True, "fixed_step_threshold"
    if trigger_mode == "action_norm_threshold" and action_norm_spike:
        return True, "action_norm_threshold"
    if trigger_mode == "fixed_or_action_norm":
        if fixed_step and action_norm_spike:
            return True, "fixed_step_and_action_norm_threshold"
        if fixed_step:
            return True, "fixed_step_threshold"
        if action_norm_spike:
            return True, "action_norm_threshold"
    if trigger_mode == "semantic_no_progress":
        triggered, reason = semantic_no_progress_trigger(
            step=step,
            semantic_history=semantic_history or [],
            min_step=semantic_min_step,
            window=semantic_window,
            progress_threshold=semantic_progress_threshold,
        )
        if triggered:
            return True, reason
        return False, reason
    return False, "not_triggered"


def clamp_action_norm(action: Any, max_norm: float) -> Any:
    if max_norm <= 0:
        raise ValueError("--action-clamp-norm must be positive")
    norm = action.norm()
    if float(norm.item()) <= max_norm:
        return action
    return action * (max_norm / norm.clamp_min(1e-8))


def format_intervention_type(
    *,
    mode: str,
    intervention_scale: float,
    action_clamp_norm: float,
    smooth_alpha: float,
) -> str:
    if mode == "scale":
        return f"scale_action_{intervention_scale:g}"
    if mode == "clamp":
        return f"clamp_action_norm_{action_clamp_norm:g}"
    if mode == "smooth":
        return f"smooth_action_alpha_{smooth_alpha:g}"
    if mode == "policy_reset":
        return "policy_reset_reselect_action"
    if mode == "semantic_reach_target":
        return "semantic_reach_target"
    if mode == "semantic_push_receptacle":
        return "semantic_push_receptacle"
    return mode


def collect_semantic_state(env: Any, *, target_object_key: str, receptacle_object_key: str) -> dict[str, Any]:
    raw_obs = get_raw_observation(env)
    state: dict[str, Any] = {
        "available": raw_obs is not None,
        "target_object_key": target_object_key,
        "receptacle_object_key": receptacle_object_key,
    }
    if raw_obs is None:
        return state
    target_pos = vector3(raw_obs.get(target_object_key))
    receptacle_pos = vector3(raw_obs.get(receptacle_object_key))
    eef_pos = vector3(raw_obs.get("robot0_eef_pos"))
    gripper_qpos = raw_obs.get("robot0_gripper_qpos")
    state.update(
        {
            "target_pos": target_pos,
            "receptacle_pos": receptacle_pos,
            "eef_pos": eef_pos,
            "gripper_qpos": _tolist(gripper_qpos),
            "target_to_receptacle_dist": distance(target_pos, receptacle_pos),
            "eef_to_target_dist": distance(eef_pos, target_pos),
        }
    )
    return state


def get_raw_observation(env: Any) -> dict[str, Any] | None:
    try:
        inner = env.envs[0]
        offscreen = getattr(inner, "_env", None)
        if offscreen is None:
            return None
        return dict(offscreen.env._get_observations())
    except Exception:  # noqa: BLE001 - best-effort instrumentation.
        return None


def semantic_no_progress_trigger(
    *,
    step: int,
    semantic_history: list[dict[str, Any]],
    min_step: int,
    window: int,
    progress_threshold: float,
) -> tuple[bool, str]:
    if step < min_step:
        return False, "semantic_before_min_step"
    if window <= 0:
        return False, "semantic_invalid_window"
    if len(semantic_history) <= window:
        return False, "semantic_waiting_for_window"
    current = semantic_history[-1]
    previous = semantic_history[-1 - window]
    current_pos = current.get("target_pos")
    previous_pos = previous.get("target_pos")
    movement = distance(current_pos, previous_pos)
    if movement is None:
        return False, "semantic_target_unavailable"
    if movement < progress_threshold:
        return True, f"semantic_no_target_progress_{movement:.6f}_lt_{progress_threshold:.6f}"
    return False, f"semantic_target_progress_{movement:.6f}"


def semantic_reach_target_action(action: Any, *, semantic_state: dict[str, Any], gain: float, gripper_command: float) -> Any:
    target_pos = semantic_state.get("target_pos")
    eef_pos = semantic_state.get("eef_pos")
    if not target_pos or not eef_pos:
        return action
    delta = [target_pos[i] - eef_pos[i] for i in range(3)]
    for index, value in enumerate(delta):
        action[:, index] = clamp_scalar(value * gain, -1.0, 1.0)
    if action.shape[-1] >= 7:
        action[:, 6] = clamp_scalar(gripper_command, -1.0, 1.0)
    return action


def semantic_push_receptacle_action(
    action: Any, *, semantic_state: dict[str, Any], gain: float, gripper_command: float
) -> Any:
    target_pos = semantic_state.get("target_pos")
    receptacle_pos = semantic_state.get("receptacle_pos")
    if not target_pos or not receptacle_pos:
        return action
    delta = [receptacle_pos[i] - target_pos[i] for i in range(3)]
    for index, value in enumerate(delta):
        action[:, index] = clamp_scalar(value * gain, -1.0, 1.0)
    if action.shape[-1] >= 7:
        action[:, 6] = clamp_scalar(gripper_command, -1.0, 1.0)
    return action


def vector3(value: Any) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def distance(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)) ** 0.5


def clamp_scalar(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


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
