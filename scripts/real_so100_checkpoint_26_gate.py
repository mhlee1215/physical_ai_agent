#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_jaw_readiness import assess_jaw_readiness
from scripts.real_so100_next_action_gate import decide_next_action
from scripts.real_so100_observe import record_observation
from scripts.real_so100_pregrasp_probe import assess_episode_frame


def run_checkpoint_26_gate(
    *,
    output_dir: Path,
    port: str | None,
    episode: Path | None,
    frame_index: int | None,
    grasp_outcome: Path | None,
    calibration_file: Path | None,
    duration_seconds: float,
    fps: float,
    task: str,
    policy_camera_indexes: list[int] | None = None,
    observer_camera_indexes: list[int] | None = None,
    wrist_camera_index: str = "0",
    egocentric_camera_index: str = "1",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_camera_indexes = policy_camera_indexes or [int(wrist_camera_index), int(egocentric_camera_index)]
    observer_camera_indexes = observer_camera_indexes or []
    camera_indexes = _dedupe(policy_camera_indexes + observer_camera_indexes)
    camera_roles = {
        wrist_camera_index: "wrist_cam",
        egocentric_camera_index: "egocentric_cam",
    }
    for index in observer_camera_indexes:
        camera_roles[str(index)] = "codex_observer"
    manifest: dict[str, Any] = {
        "operation": "real_so100_checkpoint_26_no_actuation_gate",
        "output_dir": str(output_dir),
        "actuation_enabled": False,
        "policy_actions_executed": False,
        "send_action_called": False,
        "observer_camera_status": "available" if observer_camera_indexes else "temporarily_unavailable",
        "observer_camera_note": (
            "observer/debug camera indexes are unavailable; use policy cameras only for no-actuation "
            "agentic-layer development and do not claim physical task success"
            if not observer_camera_indexes
            else "observer/debug camera indexes are excluded from SmolVLA policy input"
        ),
        "camera_indexes": camera_indexes,
        "policy_camera_indexes": policy_camera_indexes,
        "observer_camera_indexes": observer_camera_indexes,
        "camera_roles": camera_roles,
        "wrist_camera_index": wrist_camera_index,
        "egocentric_camera_index": egocentric_camera_index,
        "task": task,
        "grasp_outcome": str(grasp_outcome) if grasp_outcome else None,
        "calibration_file": str(calibration_file) if calibration_file else None,
    }

    if episode is None:
        if port is None:
            raise ValueError("either --episode or --port is required")
        observation = record_observation(
            port=port,
            camera_indexes=camera_indexes,
            output_dir=output_dir,
            duration_seconds=duration_seconds,
            fps=fps,
            task=task,
            calibration_file=calibration_file,
            camera_roles=camera_roles,
            policy_camera_indexes=policy_camera_indexes,
            observer_camera_indexes=observer_camera_indexes,
        )
        manifest["observation_manifest"] = observation
        episode_path = Path(observation["episode_jsonl"])
    else:
        episode_path = episode
        manifest["observation_manifest"] = None

    selected_frame_index = frame_index if frame_index is not None else _last_frame_index(episode_path)
    pregrasp_path = output_dir / f"pregrasp_probe_frame_{selected_frame_index:06d}.json"
    pregrasp = assess_episode_frame(
        episode=episode_path,
        frame_index=selected_frame_index,
        output=pregrasp_path,
        min_area_px=800,
        edge_margin_px=8,
    )
    jaw_image = _image_path_for_camera(episode_path, selected_frame_index, wrist_camera_index)
    jaw_path = output_dir / f"camera_{wrist_camera_index}_jaw_readiness_frame_{selected_frame_index:06d}.json"
    jaw = assess_jaw_readiness(image_path=jaw_image, output=jaw_path)
    next_action_path = output_dir / "next_action_gate.json"
    next_action = decide_next_action(
        pregrasp_probe=pregrasp_path,
        jaw_readiness=jaw_path,
        grasp_outcome=grasp_outcome,
        output=next_action_path,
        object_view_camera=egocentric_camera_index,
        jaw_camera=wrist_camera_index,
    )

    manifest.update(
        {
            "status": next_action["status"],
            "recommended_action": next_action["recommended_action"],
            "episode_jsonl": str(episode_path),
            "frame_index": selected_frame_index,
            "pregrasp_probe": str(pregrasp_path),
            "jaw_readiness": str(jaw_path),
            "next_action_gate": str(next_action_path),
            "pregrasp_status": pregrasp["status"],
            "jaw_status": jaw["status"],
            "blockers": next_action["blockers"],
            "allowed_physical_action": next_action["allowed_physical_action"],
            "vla_prompt_allowed": next_action.get("vla_prompt_allowed"),
            "vla_prompt_gate": next_action.get("vla_prompt_gate"),
            "physical_execution_gate": next_action.get("physical_execution_gate"),
        }
    )
    manifest_path = output_dir / "checkpoint_26_gate_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _last_frame_index(episode: Path) -> int:
    last: int | None = None
    for line in episode.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = int(json.loads(line)["frame_index"])
    if last is None:
        raise ValueError(f"no frames found in {episode}")
    return last


def _image_path_for_camera(episode: Path, frame_index: int, camera: str) -> Path:
    for line in episode.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["frame_index"]) != frame_index:
            continue
        image_path = record.get("observation", {}).get("images", {}).get(camera)
        if not image_path:
            raise ValueError(f"camera {camera} image missing for frame {frame_index} in {episode}")
        return Path(image_path)
    raise ValueError(f"frame_index={frame_index} not found in {episode}")


def _dedupe(items: list[int]) -> list[int]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real SO-100 CP26 no-actuation next-action gate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--grasp-outcome", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--task", default="checkpoint_26_no_actuation_gate")
    parser.add_argument("--policy-camera-index", type=int, action="append", default=[])
    parser.add_argument("--observer-camera-index", type=int, action="append", default=[])
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--egocentric-camera-index", default="1")
    args = parser.parse_args()
    print(
        json.dumps(
            run_checkpoint_26_gate(
                output_dir=args.output_dir,
                port=args.port,
                episode=args.episode,
                frame_index=args.frame_index,
                grasp_outcome=args.grasp_outcome,
                calibration_file=args.calibration_file,
                duration_seconds=args.duration_seconds,
                fps=args.fps,
                task=args.task,
                policy_camera_indexes=args.policy_camera_index or None,
                observer_camera_indexes=args.observer_camera_index,
                wrist_camera_index=args.wrist_camera_index,
                egocentric_camera_index=args.egocentric_camera_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
