#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_jaw_readiness import assess_jaw_readiness
from scripts.real_so100_pregrasp_probe import assess_episode_frame


def build_policy_camera_task_state_packet(
    *,
    development_lane: Path,
    observation_manifest: Path,
    output: Path,
    frame_index: int | None = None,
    min_area_px: int = 800,
    edge_margin_px: int = 8,
    min_jaw_marker_area_px: int = 500,
    markdown: Path | None = None,
) -> dict[str, Any]:
    lane = _load_json(development_lane)
    manifest = _load_json(observation_manifest)
    episode = Path(manifest["episode_jsonl"])
    selected_frame = _select_frame(episode, frame_index)
    selected_frame_index = int(selected_frame["frame_index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pregrasp_path = output.parent / f"pregrasp_probe_frame_{selected_frame_index:06d}.json"
    pregrasp = assess_episode_frame(
        episode=episode,
        frame_index=selected_frame_index,
        output=pregrasp_path,
        min_area_px=min_area_px,
        edge_margin_px=edge_margin_px,
    )
    images = (selected_frame.get("observation") or {}).get("images") or {}
    wrist_camera = _wrist_camera(lane, manifest)
    wrist_image = Path(images[str(wrist_camera)]) if str(wrist_camera) in images else None
    jaw_path = output.parent / f"camera_{wrist_camera}_jaw_readiness_frame_{selected_frame_index:06d}.json"
    jaw = (
        assess_jaw_readiness(
            image_path=wrist_image,
            output=jaw_path,
            min_object_area_px=min_area_px,
            min_jaw_marker_area_px=min_jaw_marker_area_px,
            edge_margin_px=edge_margin_px,
        )
        if wrist_image
        else _missing_jaw_readiness(wrist_camera)
    )
    blockers = _blockers(lane=lane, manifest=manifest, images=images)
    object_state = _object_state(pregrasp)
    packet = {
        "operation": "real_so100_policy_camera_task_state_packet",
        "status": "passed" if not blockers else "blocked",
        "purpose": "structure policy-camera observations for a replaceable in-loop LLM/VLM without deciding robot actions",
        "source_development_lane": str(development_lane),
        "source_observation_manifest": str(observation_manifest),
        "episode_jsonl": str(episode),
        "frame_index": selected_frame_index,
        "task": selected_frame.get("task") or manifest.get("task"),
        "task_goal": _task_goal(selected_frame.get("task") or manifest.get("task") or ""),
        "camera_contract": {
            "policy_camera_indexes": [0, 1],
            "policy_camera_roles": _policy_roles(lane, manifest),
            "observer_camera_indexes": [],
            "observer_camera_status": "temporarily_unavailable",
            "camera_3_is_policy_input": False,
        },
        "robot_state": (selected_frame.get("observation") or {}).get("state") or {},
        "robot_state_source": (selected_frame.get("observation") or {}).get("state_source"),
        "policy_camera_images": {str(key): value for key, value in images.items() if str(key) in {"0", "1"}},
        "object_state": object_state,
        "jaw_context": _jaw_context(jaw, wrist_camera=wrist_camera),
        "llm_vlm_input_packet": {
            "consumer": "in_loop_agent_or_smolvla_prompt_builder",
            "does_not_prompt_operator": True,
            "allowed_reasoning": [
                "decide whether the target object is visible in policy cameras",
                "decide whether to preserve the best historical prompt family",
                "decide whether a no-actuation prompt packet should ask for search, reframe, approach, or pre-grasp",
            ],
            "forbidden_reasoning_outputs": [
                "physical execution authorization",
                "task success claim",
                "fixed robot-frame direction mapping for object relocation",
                "human operator instruction",
            ],
        },
        "pregrasp_probe_path": str(pregrasp_path),
        "jaw_readiness_path": str(jaw_path) if wrist_image else None,
        "next_agentic_layer_step": _next_step(object_state=object_state, jaw=jaw),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "blockers": blockers,
        "guardrails": [
            "camera 0 and camera 1 are SmolVLA/policy inputs",
            "camera 3 is not used because it is observer-only and currently off",
            "this packet is LLM/VLM input, not an action executor",
            "object-right remains an object-frame task relation, not a hard-coded robot-arm direction",
        ],
    }
    output.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(packet), encoding="utf-8")
    packet["json_path"] = str(output)
    packet["markdown_path"] = str(md_path)
    return packet


def render_markdown(packet: dict[str, Any]) -> str:
    object_state = packet.get("object_state") or {}
    jaw = packet.get("jaw_context") or {}
    lines = [
        "# Real SO-100 Policy-Camera Task-State Packet",
        "",
        f"- Status: `{packet.get('status')}`",
        f"- Frame: `{packet.get('frame_index')}`",
        f"- Policy cameras: `{packet.get('camera_contract', {}).get('policy_camera_indexes')}`",
        f"- Object visible: `{object_state.get('visible')}`",
        f"- Primary object camera: `{object_state.get('primary_camera')}`",
        f"- Jaw readiness: `{jaw.get('status')}`",
        f"- Next step: `{packet.get('next_agentic_layer_step', {}).get('type')}`",
        f"- Physical robot motion: `{packet.get('physical_robot_motion')}`",
        "",
    ]
    if object_state.get("camera_observations"):
        lines.extend(["## Camera Observations", ""])
        for item in object_state["camera_observations"]:
            lines.append(
                f"- camera `{item.get('camera')}` visible=`{item.get('object_visible')}` "
                f"usable=`{item.get('usable_for_pregrasp')}` center=`{item.get('center_px')}` "
                f"edge_clipped=`{item.get('edge_clipped')}`"
            )
    if packet.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in packet["blockers"])
    lines.append("")
    return "\n".join(lines)


def _task_goal(task: str) -> dict[str, Any]:
    lowered = task.lower()
    direction = "right" if "right" in lowered or "오른쪽" in task else None
    target = "green Android figure" if "green" in lowered or "녹색" in task else None
    return {
        "target_object": target or "target object",
        "manipulation": "pick_and_transport" if ("pick" in lowered or "집" in task or "들" in task) else "manipulate",
        "target_relation": {"type": "move_object", "direction": direction or "unspecified"},
        "relation_frame": "object_or_observer_image_frame",
        "requires_grasp_verifier": True,
        "requires_relocation_verifier": True,
    }


def _object_state(pregrasp: dict[str, Any]) -> dict[str, Any]:
    assessments = pregrasp.get("assessments") or []
    primary = pregrasp.get("primary_camera")
    primary_item = next((item for item in assessments if item.get("camera") == primary), None)
    return {
        "visible": any(bool(item.get("object_visible")) for item in assessments),
        "usable_for_pregrasp": bool(primary),
        "primary_camera": primary,
        "primary_center_px": (primary_item or {}).get("center_px"),
        "primary_bbox_xyxy": (primary_item or {}).get("bbox_xyxy"),
        "usable_cameras": pregrasp.get("usable_cameras") or [],
        "camera_observations": assessments,
        "notes": pregrasp.get("notes") or [],
    }


def _jaw_context(jaw: dict[str, Any], *, wrist_camera: int) -> dict[str, Any]:
    return {
        "wrist_camera": wrist_camera,
        "status": jaw.get("status"),
        "object_candidate": jaw.get("object_candidate"),
        "jaw_marker_candidate": jaw.get("jaw_marker_candidate"),
        "object_edge_clipped": jaw.get("object_edge_clipped"),
        "blockers": jaw.get("blockers") or [],
        "notes": jaw.get("notes") or [],
    }


def _next_step(*, object_state: dict[str, Any], jaw: dict[str, Any]) -> dict[str, str]:
    if not object_state.get("visible"):
        return {
            "type": "ask_llm_vlm_for_no_actuation_search_prompt_packet",
            "reason": "The target object is not visible in policy cameras; search reasoning is needed before another SmolVLA pass.",
        }
    if not object_state.get("usable_for_pregrasp"):
        return {
            "type": "ask_llm_vlm_for_no_actuation_reframe_or_approach_prompt_packet",
            "reason": "The target is visible but not usable for pregrasp, often because it is edge-clipped.",
        }
    if jaw.get("status") != "ready":
        return {
            "type": "ask_llm_vlm_for_no_actuation_jaw_alignment_prompt_packet",
            "reason": "The target is usable in policy cameras, but wrist/jaw context is not ready.",
        }
    return {
        "type": "ask_llm_vlm_for_no_actuation_best_prompt_evaluation_packet",
        "reason": "The policy-camera task state is usable; evaluate the best historical prompt family before any mutation.",
    }


def _blockers(*, lane: dict[str, Any], manifest: dict[str, Any], images: dict[str, Any]) -> list[str]:
    blockers = []
    if lane.get("operation") != "real_so100_agentic_development_lane":
        blockers.append(f"Development lane operation is {lane.get('operation')!r}.")
    if lane.get("status") != "passed":
        blockers.append(f"Development lane status is {lane.get('status')!r}.")
    if manifest.get("ok") is not True:
        blockers.append("Observation manifest is not ok.")
    missing = [index for index in ("0", "1") if index not in {str(key) for key in images}]
    if missing:
        blockers.append(f"Policy camera image(s) missing: {missing}.")
    if bool(lane.get("physical_robot_motion")) or bool(manifest.get("send_action_called")):
        blockers.append("Source artifacts record physical motion or action sending.")
    return blockers


def _policy_roles(lane: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return (
        (lane.get("camera_contract") or {}).get("policy_camera_roles")
        or manifest.get("camera_roles")
        or {"0": "wrist_cam", "1": "egocentric_wide_context"}
    )


def _wrist_camera(lane: dict[str, Any], manifest: dict[str, Any]) -> int:
    roles = _policy_roles(lane, manifest)
    for camera, role in roles.items():
        if role == "wrist_cam":
            return int(camera)
    return 0


def _missing_jaw_readiness(wrist_camera: int) -> dict[str, Any]:
    return {
        "status": "blocked",
        "object_candidate": None,
        "jaw_marker_candidate": None,
        "object_edge_clipped": None,
        "blockers": [f"wrist camera {wrist_camera} image missing"],
        "notes": ["Jaw readiness could not run without the wrist camera image."],
    }


def _select_frame(episode: Path, frame_index: int | None) -> dict[str, Any]:
    frames = [json.loads(line) for line in episode.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not frames:
        raise ValueError(f"no frames found in {episode}")
    if frame_index is None:
        return frames[-1]
    for frame in frames:
        if int(frame.get("frame_index", -1)) == frame_index:
            return frame
    raise ValueError(f"frame_index={frame_index} not found in {episode}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a policy-camera task-state packet for the real SO-100 agentic layer.")
    parser.add_argument("--development-lane", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--min-area-px", type=int, default=800)
    parser.add_argument("--edge-margin-px", type=int, default=8)
    parser.add_argument("--min-jaw-marker-area-px", type=int, default=500)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_policy_camera_task_state_packet(
                development_lane=args.development_lane,
                observation_manifest=args.observation_manifest,
                output=args.output,
                frame_index=args.frame_index,
                min_area_px=args.min_area_px,
                edge_margin_px=args.edge_margin_px,
                min_jaw_marker_area_px=args.min_jaw_marker_area_px,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
