#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_policy_camera_feedback(
    *,
    observation_manifest: Path,
    output: Path,
    observations: list[str],
    diagnosis: list[str],
    next_prompt: str,
    frame_index: int | None = None,
    observer_camera_status: str = "temporarily_unavailable",
) -> dict[str, Any]:
    manifest = _load_json(observation_manifest)
    episode_path = Path(manifest.get("episode_jsonl", ""))
    episode_frame = _select_episode_frame(episode_path=episode_path, frame_index=frame_index) if episode_path else {}
    observation = episode_frame.get("observation") or {}
    policy_camera_indexes = [int(index) for index in manifest.get("policy_camera_indexes") or observation.get("policy_camera_indexes") or [0, 1]]
    observer_camera_indexes = [int(index) for index in manifest.get("observer_camera_indexes") or observation.get("observer_camera_indexes") or []]
    status = "passed" if manifest.get("ok") and observations and next_prompt else "blocked"
    result = {
        "operation": "real_so100_policy_camera_pseudo_llm_feedback",
        "status": status,
        "purpose": "convert policy-camera evidence into replaceable LLM/VLM feedback for the real SO-100 SmolVLA agentic layer",
        "observation_manifest": str(observation_manifest),
        "frame_index": episode_frame.get("frame_index"),
        "camera_contract": {
            "policy_camera_indexes": policy_camera_indexes,
            "observer_camera_indexes": observer_camera_indexes,
            "observer_camera_status": observer_camera_status,
            "policy_camera_roles": observation.get("camera_roles") or manifest.get("camera_roles") or {},
            "camera_3_is_policy_input": False,
        },
        "visual_evidence": {
            "images": observation.get("images") or {},
            "image_shapes": observation.get("image_shapes") or {},
            "state_available": bool(observation.get("state_available", observation.get("state") is not None)),
            "state_source": observation.get("state_source"),
        },
        "pseudo_llm_feedback": {
            "development_role": "Codex acts as a temporary Pseudo-LLM/VLM; final runtime should replace this with an on-device lightweight LLM/VLM.",
            "target": "in_loop_agent_or_smolvla",
            "does_not_prompt_operator": True,
            "observations": observations,
            "diagnosis": diagnosis,
            "next_smolvla_prompt": next_prompt,
        },
        "execution_outcome": {
            "send_action_called": bool(manifest.get("send_action_called")),
            "policy_actions_executed": bool(manifest.get("policy_actions_executed")),
            "physical_robot_motion": False,
        },
        "task_success_claim_allowed": False,
        "next_agentic_layer_step": _next_step(status),
        "guardrails": [
            "use cameras 0 and 1 as policy inputs only",
            "camera 3 is observer-only and currently unavailable",
            "do not claim grasp, relocation, or task success from policy-camera-only evidence",
            "do not convert object-right goals into fixed robot-arm directions",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    result["manifest_path"] = str(output)
    result["markdown_path"] = str(output.with_suffix(".md"))
    return result


def render_markdown(report: dict[str, Any]) -> str:
    feedback = report.get("pseudo_llm_feedback", {})
    lines = [
        "# Real SO-100 Policy-Camera Pseudo-LLM Feedback",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Policy cameras: `{report.get('camera_contract', {}).get('policy_camera_indexes')}`",
        f"- Observer cameras: `{report.get('camera_contract', {}).get('observer_camera_indexes')}`",
        f"- Observer status: `{report.get('camera_contract', {}).get('observer_camera_status')}`",
        f"- Target: `{feedback.get('target')}`",
        f"- Sends action: `{report.get('execution_outcome', {}).get('send_action_called')}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed')}`",
        "",
        "## Observations",
        "",
    ]
    lines.extend(f"- {item}" for item in feedback.get("observations", []))
    lines.extend(["", "## Diagnosis", ""])
    lines.extend(f"- {item}" for item in feedback.get("diagnosis", []))
    lines.extend(
        [
            "",
            "## Next SmolVLA Prompt",
            "",
            feedback.get("next_smolvla_prompt", ""),
            "",
        ]
    )
    return "\n".join(lines)


def _next_step(status: str) -> dict[str, str]:
    if status == "passed":
        return {
            "type": "run_no_actuation_smolvla_from_policy_camera_feedback",
            "reason": "Use the policy-camera feedback as in-loop agent input while camera 3 remains unavailable.",
        }
    return {
        "type": "repair_policy_camera_feedback_capture",
        "reason": "A valid policy-camera observation and Pseudo-LLM feedback are required before the next no-actuation SmolVLA pass.",
    }


def _select_episode_frame(*, episode_path: Path, frame_index: int | None) -> dict[str, Any]:
    frames = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not frames:
        return {}
    if frame_index is None:
        return frames[-1]
    for frame in frames:
        if int(frame.get("frame_index", -1)) == frame_index:
            return frame
    raise ValueError(f"frame_index {frame_index} not found in {episode_path}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pseudo-LLM feedback from real SO-100 policy cameras.")
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--diagnosis", action="append", default=[])
    parser.add_argument("--next-prompt", required=True)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--observer-camera-status", default="temporarily_unavailable")
    args = parser.parse_args()
    print(
        json.dumps(
            build_policy_camera_feedback(
                observation_manifest=args.observation_manifest,
                output=args.output,
                observations=args.observation,
                diagnosis=args.diagnosis,
                next_prompt=args.next_prompt,
                frame_index=args.frame_index,
                observer_camera_status=args.observer_camera_status,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
