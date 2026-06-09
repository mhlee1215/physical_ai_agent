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
        choices=(
            "fixed_step",
            "action_norm_threshold",
            "fixed_or_action_norm",
            "semantic_no_progress",
            "semantic_near_receptacle",
        ),
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
            "semantic_reach_then_push",
            "semantic_place_receptacle",
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
    parser.add_argument("--semantic-distance-threshold", type=float, default=0.07)
    parser.add_argument("--semantic-reach-gain", type=float, default=4.0)
    parser.add_argument("--semantic-push-gain", type=float, default=4.0)
    parser.add_argument("--semantic-contact-threshold", type=float, default=0.06)
    parser.add_argument("--semantic-place-z-command", type=float, default=-0.2)
    parser.add_argument("--semantic-gripper-command", type=float, default=1.0)
    parser.add_argument(
        "--ita-enable",
        action="store_true",
        help="Enable Imagine-Then-Act candidate generation and selected-action execution.",
    )
    parser.add_argument(
        "--ita-candidate-seeds",
        default="",
        help="Comma-separated candidate seeds used to sample policy action chunks from the same observation.",
    )
    parser.add_argument("--ita-num-candidates", type=int, default=0)
    parser.add_argument(
        "--ita-commit-steps",
        type=int,
        default=1,
        help="Number of selected candidate actions to commit open-loop before normal policy resumes.",
    )
    parser.add_argument(
        "--ita-selected-candidate-id",
        default="",
        help="Optional candidate id to force; use only for debug/sanity checks.",
    )
    parser.add_argument(
        "--ita-selector-strategy",
        choices=("baseline_fallback", "debug_min_action_norm"),
        default="baseline_fallback",
        help="baseline_fallback preserves policy-only chunks; debug_min_action_norm is not a method selector.",
    )
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
        semantic_distance_threshold=args.semantic_distance_threshold,
        semantic_reach_gain=args.semantic_reach_gain,
        semantic_push_gain=args.semantic_push_gain,
        semantic_contact_threshold=args.semantic_contact_threshold,
        semantic_place_z_command=args.semantic_place_z_command,
        semantic_gripper_command=args.semantic_gripper_command,
        ita_enable=args.ita_enable,
        ita_candidate_seeds=parse_ita_candidate_seeds(args.ita_candidate_seeds),
        ita_num_candidates=args.ita_num_candidates,
        ita_commit_steps=args.ita_commit_steps,
        ita_selected_candidate_id=args.ita_selected_candidate_id or None,
        ita_selector_strategy=args.ita_selector_strategy,
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
    semantic_distance_threshold: float,
    semantic_reach_gain: float,
    semantic_push_gain: float,
    semantic_contact_threshold: float,
    semantic_place_z_command: float,
    semantic_gripper_command: float,
    ita_enable: bool = False,
    ita_candidate_seeds: list[int] | None = None,
    ita_num_candidates: int = 0,
    ita_commit_steps: int = 1,
    ita_selected_candidate_id: str | None = None,
    ita_selector_strategy: str = "baseline_fallback",
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
        ita_action_queue: list[Any] = []
        ita_active_candidate_id: str | None = None
        ita_selected_candidate_applied = False
        ita_committed_action_steps = 0
        ita_candidate_generation_source = "disabled"
        ita_selected_action_shape: list[int] | None = None
        ita_candidate_generation_done = False
        ita_baseline_candidate_available = False
        ita_baseline_candidate_selected = False
        ita_selector_confidence: float | None = None
        ita_selector_fallback_used = False
        ita_method_claim_ready = False

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
            ita_step_record: dict[str, Any] = {
                "enabled": bool(ita_enable),
                "action_source": "normal_policy",
                "selected_candidate_id": None,
                "selected_candidate_applied": False,
                "candidate_generation_source": "disabled" if not ita_enable else None,
                "candidate_seeds": normalize_ita_candidate_seeds(
                    ita_candidate_seeds,
                    ita_num_candidates,
                )
                if ita_enable
                else [],
                "candidate_count": 0,
                "committed_action_steps_count": ita_committed_action_steps,
                "selected_action_shape": None,
                "baseline_candidate_available": False,
                "baseline_candidate_selected": False,
                "selector_strategy": ita_selector_strategy,
                "selector_confidence": None,
                "selector_fallback_used": False,
                "method_claim_ready": False,
                "limitation": None,
            }
            if ita_enable and not ita_candidate_generation_done:
                decision = build_ita_candidate_decision(
                    policy=policy,
                    observation=observation,
                    postprocessor=postprocessor,
                    env_postprocessor=env_postprocessor,
                    action_key=ACTION,
                    torch_module=torch,
                    numpy_module=np,
                    candidate_seeds=ita_candidate_seeds,
                    num_candidates=ita_num_candidates,
                    commit_steps=ita_commit_steps,
                    forced_candidate_id=ita_selected_candidate_id,
                    selector_strategy=ita_selector_strategy,
                )
                ita_action_queue = list(decision["selected_actions"])
                ita_active_candidate_id = decision["selected_candidate_id"]
                ita_candidate_generation_source = str(decision["candidate_generation_source"])
                ita_selected_action_shape = list(decision["selected_action_shape"])
                ita_baseline_candidate_available = bool(decision["baseline_candidate_available"])
                ita_baseline_candidate_selected = bool(decision["baseline_candidate_selected"])
                ita_selector_confidence = decision["selector_confidence"]
                ita_selector_fallback_used = bool(decision["selector_fallback_used"])
                ita_candidate_generation_done = True
                ita_step_record.update(
                    {
                        "candidate_generation_source": decision["candidate_generation_source"],
                        "candidate_count": decision["candidate_count"],
                        "candidates": decision["candidates"],
                        "selected_candidate_id": decision["selected_candidate_id"],
                        "selected_action_shape": decision["selected_action_shape"],
                        "baseline_candidate_available": decision["baseline_candidate_available"],
                        "baseline_candidate_selected": decision["baseline_candidate_selected"],
                        "selector_strategy": decision["selector_strategy"],
                        "selector_confidence": decision["selector_confidence"],
                        "selector_fallback_used": decision["selector_fallback_used"],
                        "method_claim_ready": False,
                        "limitation": decision["limitation"],
                    }
                )

            if ita_enable and ita_action_queue:
                action = ita_action_queue.pop(0)
                ita_selected_candidate_applied = True
                ita_committed_action_steps += 1
                ita_step_record.update(
                    {
                        "action_source": "ita_selected_candidate",
                        "selected_candidate_id": ita_active_candidate_id,
                        "selected_candidate_applied": True,
                        "committed_action_steps_count": ita_committed_action_steps,
                        "remaining_selected_actions": len(ita_action_queue),
                        "selected_action_shape": _shape_list(action),
                        "baseline_candidate_available": ita_baseline_candidate_available,
                        "baseline_candidate_selected": ita_baseline_candidate_selected,
                        "selector_strategy": ita_selector_strategy,
                        "selector_confidence": ita_selector_confidence,
                        "selector_fallback_used": ita_selector_fallback_used,
                    }
                )
                if not ita_action_queue and not (
                    ita_selector_strategy == "baseline_fallback" and ita_baseline_candidate_selected
                ):
                    policy.reset()
            else:
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
                semantic_distance_threshold=semantic_distance_threshold,
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
                        gain=semantic_push_gain,
                        gripper_command=semantic_gripper_command,
                    )
                elif intervention_mode == "semantic_reach_then_push":
                    action = semantic_reach_then_push_action(
                        action,
                        semantic_state=semantic_state,
                        reach_gain=semantic_reach_gain,
                        push_gain=semantic_push_gain,
                        contact_threshold=semantic_contact_threshold,
                        gripper_command=semantic_gripper_command,
                    )
                elif intervention_mode == "semantic_place_receptacle":
                    action = semantic_place_receptacle_action(
                        action,
                        semantic_state=semantic_state,
                        reach_gain=semantic_reach_gain,
                        push_gain=semantic_push_gain,
                        contact_threshold=semantic_contact_threshold,
                        place_z_command=semantic_place_z_command,
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
                    "ita": ita_step_record,
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

        rollout_success = bool(any(any(record["successes"]) for record in trace_records))
        ita_method_claim_ready = False
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
                        "semantic_distance_threshold": semantic_distance_threshold,
                        "semantic_reach_gain": semantic_reach_gain,
                        "semantic_push_gain": semantic_push_gain,
                        "semantic_contact_threshold": semantic_contact_threshold,
                        "semantic_place_z_command": semantic_place_z_command,
                        "semantic_gripper_command": semantic_gripper_command,
                        "action_step_count": len(trace_records),
                        "verifier_trigger_count": sum(1 for record in trace_records if record["verifier_triggered"]),
                        "intervention_count": sum(1 for record in trace_records if record["intervention_type"]),
                        "success": rollout_success,
                        "ita_enabled": bool(ita_enable),
                        "ita_candidate_generation_source": ita_candidate_generation_source,
                        "ita_candidate_seeds": normalize_ita_candidate_seeds(
                            ita_candidate_seeds,
                            ita_num_candidates,
                        )
                        if ita_enable
                        else [],
                        "ita_selected_candidate_id": ita_active_candidate_id,
                        "ita_selected_action_shape": ita_selected_action_shape,
                        "ita_committed_action_steps": ita_committed_action_steps,
                        "selected_candidate_applied": bool(ita_selected_candidate_applied),
                        "baseline_candidate_available": bool(ita_baseline_candidate_available),
                        "baseline_candidate_selected": bool(ita_baseline_candidate_selected),
                        "selector_strategy": ita_selector_strategy,
                        "selector_confidence": ita_selector_confidence,
                        "selector_fallback_used": bool(ita_selector_fallback_used),
                        "method_claim_ready": ita_method_claim_ready,
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


def parse_ita_candidate_seeds(raw_value: str) -> list[int]:
    if not raw_value.strip():
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def normalize_ita_candidate_seeds(candidate_seeds: list[int] | None, num_candidates: int) -> list[int]:
    seeds = list(candidate_seeds or [])
    if num_candidates <= 0:
        return seeds
    if not seeds:
        seeds = list(range(num_candidates))
    while len(seeds) < num_candidates:
        seeds.append(seeds[-1] + 1)
    return seeds[:num_candidates]


def build_ita_candidate_decision(
    *,
    policy: Any,
    observation: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    action_key: str,
    torch_module: Any,
    numpy_module: Any,
    candidate_seeds: list[int] | None,
    num_candidates: int,
    commit_steps: int,
    forced_candidate_id: str | None = None,
    selector_strategy: str = "baseline_fallback",
) -> dict[str, Any]:
    seeds = normalize_ita_candidate_seeds(candidate_seeds, num_candidates)
    if not seeds:
        seeds = [0]
    commit_steps = max(1, int(commit_steps))
    candidates = []
    baseline_candidate = sample_policy_candidate_chunk(
        policy=policy,
        observation=observation,
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
        action_key=action_key,
        torch_module=torch_module,
        commit_steps=commit_steps,
    )
    baseline_candidate.update(
        {
            "candidate_id": "candidate_00_policy_only",
            "seed": None,
            "is_baseline": True,
            "selection_role": "policy_only_baseline",
        }
    )
    candidates.append(baseline_candidate)
    should_sample_nonbaseline = bool(forced_candidate_id and forced_candidate_id != "candidate_00_policy_only")
    should_sample_nonbaseline = should_sample_nonbaseline or selector_strategy == "debug_min_action_norm"
    if should_sample_nonbaseline:
        for index, seed in enumerate(seeds):
            candidate_id = f"candidate_{index + 1:02d}"
            set_ita_policy_seed(seed, numpy_module=numpy_module, torch_module=torch_module)
            policy.reset()
            candidate = sample_policy_candidate_chunk(
                policy=policy,
                observation=observation,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
                action_key=action_key,
                torch_module=torch_module,
                commit_steps=commit_steps,
            )
            candidate.update({"candidate_id": candidate_id, "seed": seed, "is_baseline": False})
            candidates.append(candidate)
    selected = choose_ita_candidate(candidates, forced_candidate_id, selector_strategy=selector_strategy)
    selected_actions = list(selected["actions"])
    baseline_candidate_selected = selected["candidate_id"] == "candidate_00_policy_only"
    selector_fallback_used = selector_strategy == "baseline_fallback" and not forced_candidate_id
    selector_confidence = 1.0 if baseline_candidate_selected and selector_fallback_used else 0.0
    return {
        "candidate_generation_source": selected["source"],
        "candidate_count": len(candidates),
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "seed": candidate["seed"],
                "source": candidate["source"],
                "action_shape": candidate["action_shape"],
                "score": candidate["score"],
                "is_baseline": bool(candidate.get("is_baseline", False)),
                "selection_role": candidate.get("selection_role"),
                "limitation": candidate["limitation"],
            }
            for candidate in candidates
        ],
        "selected_candidate_id": selected["candidate_id"],
        "selected_action_shape": selected["action_shape"],
        "selected_actions": selected_actions,
        "baseline_candidate_available": True,
        "baseline_candidate_selected": baseline_candidate_selected,
        "selector_strategy": selector_strategy,
        "selector_confidence": selector_confidence,
        "selector_fallback_used": selector_fallback_used,
        "limitation": selected["limitation"],
    }


def sample_policy_candidate_chunk(
    *,
    policy: Any,
    observation: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    action_key: str,
    torch_module: Any,
    commit_steps: int,
) -> dict[str, Any]:
    limitation = None
    with torch_module.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            raw_chunk = policy.predict_action_chunk(clone_policy_observation(observation))
            source = f"{getattr(policy, 'name', 'policy')}.predict_action_chunk"
        else:
            raw_action = policy.select_action(clone_policy_observation(observation))
            raw_chunk = raw_action.unsqueeze(1) if hasattr(raw_action, "unsqueeze") else [raw_action]
            source = f"{getattr(policy, 'name', 'policy')}.select_action"
            limitation = "Policy does not expose predict_action_chunk; ITA MVP used a one-step selected action."
    actions = extract_env_action_sequence(
        raw_chunk=raw_chunk,
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
        action_key=action_key,
        commit_steps=commit_steps,
    )
    return {
        "source": source,
        "actions": actions,
        "action_shape": _shape_list(actions[0]) if actions else [],
        "score": score_ita_action_sequence(actions),
        "limitation": limitation,
    }


def extract_env_action_sequence(
    *,
    raw_chunk: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    action_key: str,
    commit_steps: int,
) -> list[Any]:
    actions = []
    chunk_steps = int(raw_chunk.shape[1]) if hasattr(raw_chunk, "shape") and len(raw_chunk.shape) >= 3 else 1
    for index in range(min(commit_steps, chunk_steps)):
        if chunk_steps == 1 and not (hasattr(raw_chunk, "shape") and len(raw_chunk.shape) >= 3):
            raw_step = raw_chunk
        else:
            raw_step = raw_chunk[:, index, :]
        action = postprocessor(raw_step)
        action_transition = {action_key: action}
        action_transition = env_postprocessor(action_transition)
        actions.append(action_transition[action_key])
    return actions


def choose_ita_candidate(
    candidates: list[dict[str, Any]],
    forced_candidate_id: str | None = None,
    selector_strategy: str = "baseline_fallback",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("ITA candidate generation produced no candidates")
    if forced_candidate_id:
        for candidate in candidates:
            if candidate["candidate_id"] == forced_candidate_id:
                return candidate
        raise ValueError(f"forced ITA candidate id not found: {forced_candidate_id}")
    if selector_strategy == "baseline_fallback":
        for candidate in candidates:
            if candidate["candidate_id"] == "candidate_00_policy_only":
                return candidate
    return min(candidates, key=lambda candidate: (candidate["score"], candidate["candidate_id"]))


def score_ita_action_sequence(actions: list[Any]) -> float:
    values = []
    for action in actions:
        if hasattr(action, "detach"):
            action = action.detach()
        if hasattr(action, "to"):
            action = action.to("cpu")
        if hasattr(action, "numpy"):
            action = action.numpy()
        if hasattr(action, "tolist"):
            action = action.tolist()
        values.extend(_flatten_numeric(action))
    return sum(abs(value) for value in values) / max(len(values), 1)


def set_ita_policy_seed(seed: int, *, numpy_module: Any, torch_module: Any) -> None:
    import random

    random.seed(seed)
    if hasattr(numpy_module, "random") and hasattr(numpy_module.random, "seed"):
        numpy_module.random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        try:
            cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001 - CUDA may be unavailable in CPU-only jobs.
            pass


def clone_policy_observation(observation: Any) -> Any:
    cloned: dict[str, Any] = {}
    for key, value in observation.items():
        if hasattr(value, "detach") and hasattr(value, "clone"):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = deepcopy(value)
    return cloned


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
    semantic_distance_threshold: float = 0.0,
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
    if trigger_mode == "semantic_near_receptacle":
        triggered, reason = semantic_near_receptacle_trigger(
            step=step,
            semantic_history=semantic_history or [],
            min_step=semantic_min_step,
            distance_threshold=semantic_distance_threshold,
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
    if mode == "semantic_reach_then_push":
        return "semantic_reach_then_push"
    if mode == "semantic_place_receptacle":
        return "semantic_place_receptacle"
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


def semantic_near_receptacle_trigger(
    *,
    step: int,
    semantic_history: list[dict[str, Any]],
    min_step: int,
    distance_threshold: float,
) -> tuple[bool, str]:
    if step < min_step:
        return False, "semantic_before_min_step"
    if not semantic_history:
        return False, "semantic_waiting_for_state"
    distance_value = semantic_history[-1].get("target_to_receptacle_dist")
    if distance_value is None:
        return False, "semantic_target_to_receptacle_unavailable"
    if float(distance_value) <= distance_threshold:
        return True, f"semantic_near_receptacle_{float(distance_value):.6f}_le_{distance_threshold:.6f}"
    return False, f"semantic_far_from_receptacle_{float(distance_value):.6f}"


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


def semantic_reach_then_push_action(
    action: Any,
    *,
    semantic_state: dict[str, Any],
    reach_gain: float,
    push_gain: float,
    contact_threshold: float,
    gripper_command: float,
) -> Any:
    target_pos = semantic_state.get("target_pos")
    eef_pos = semantic_state.get("eef_pos")
    if not target_pos or not eef_pos:
        return action
    eef_to_target = semantic_state.get("eef_to_target_dist")
    if eef_to_target is None or float(eef_to_target) > contact_threshold:
        return semantic_reach_target_action(
            action,
            semantic_state=semantic_state,
            gain=reach_gain,
            gripper_command=gripper_command,
        )
    return semantic_push_receptacle_action(
        action,
        semantic_state=semantic_state,
        gain=push_gain,
        gripper_command=gripper_command,
    )


def semantic_place_receptacle_action(
    action: Any,
    *,
    semantic_state: dict[str, Any],
    reach_gain: float,
    push_gain: float,
    contact_threshold: float,
    place_z_command: float,
    gripper_command: float,
) -> Any:
    eef_to_target = semantic_state.get("eef_to_target_dist")
    if eef_to_target is None or float(eef_to_target) > contact_threshold:
        return semantic_reach_target_action(
            action,
            semantic_state=semantic_state,
            gain=reach_gain,
            gripper_command=gripper_command,
        )
    target_pos = semantic_state.get("target_pos")
    receptacle_pos = semantic_state.get("receptacle_pos")
    if not target_pos or not receptacle_pos:
        return action
    delta = [receptacle_pos[i] - target_pos[i] for i in range(3)]
    action[:, 0] = clamp_scalar(delta[0] * push_gain, -1.0, 1.0)
    action[:, 1] = clamp_scalar(delta[1] * push_gain, -1.0, 1.0)
    action[:, 2] = clamp_scalar(place_z_command, -1.0, 1.0)
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


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [len(value), len(value[0])]
        return [len(value)]
    return []


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_flatten_numeric(item))
        return values
    return []


if __name__ == "__main__":
    main()
