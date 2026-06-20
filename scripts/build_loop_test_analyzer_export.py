#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "loop-test-analyzer-v0.1"
DEFAULT_QWEN_SYSTEM_PROMPT = (
    "You are a robot task planner. Use only the provided tools. "
    "Plan short SO101 primitive calls for the task. "
    "Qwen3 non-thinking mode: do not emit hidden reasoning, prose, or markdown. "
    "Return tool calls only. /no_think"
)
DEFAULT_QWEN_USER_PROMPT_TEMPLATE = (
    "Task: {task}\n"
    "Target object: {target_object}\n"
    "Use the narrow SO101 edge-grasp primitive set in order when appropriate: "
    "move, align, pick_up."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized loop-test analyzer export.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-source", action="store_true")
    parser.add_argument("--generate-media", action="store_true")
    parser.add_argument("--media-width", type=int, default=128)
    parser.add_argument("--media-height", type=int, default=128)
    parser.add_argument("--media-fps", type=int, default=12)
    parser.add_argument("--media-every-n-steps", type=int, default=1)
    args = parser.parse_args()

    manifest = build_export(
        args.run_dir,
        args.output_dir,
        copy_source=args.copy_source,
        generate_media=args.generate_media,
        media_width=args.media_width,
        media_height=args.media_height,
        media_fps=args.media_fps,
        media_every_n_steps=args.media_every_n_steps,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / "manifest.json"), **manifest["summary"]}, indent=2))


def build_export(
    run_dir: Path,
    output_dir: Path,
    *,
    copy_source: bool = False,
    generate_media: bool = False,
    media_width: int = 128,
    media_height: int = 128,
    media_fps: int = 12,
    media_every_n_steps: int = 1,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validations = _validation_by_checkpoint(run_dir)
    loop_tests = []
    for report_path in _closed_loop_report_paths(run_dir):
        report = _read_json(report_path)
        loop_test = _build_loop_test(
            run_dir,
            output_dir,
            report_path,
            report,
            validations,
            copy_source=copy_source,
            generate_media=generate_media,
            media_width=media_width,
            media_height=media_height,
            media_fps=media_fps,
            media_every_n_steps=media_every_n_steps,
        )
        loop_tests.append(loop_test)
    loop_tests.sort(key=lambda row: (row.get("training_step") or -1, row.get("checkpoint") or ""))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "loop_tests": loop_tests,
        "summary": {
            "loop_tests": len(loop_tests),
            "checkpoints": [row["checkpoint"] for row in loop_tests],
            "successes": sum(1 for row in loop_tests if (row.get("success_rate") or 0) > 0),
            "latest_checkpoint": loop_tests[-1]["checkpoint"] if loop_tests else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_tables(output_dir, loop_tests)
    _write_standalone_report(output_dir, manifest)
    _write_zip_bundle(output_dir)
    return manifest


def _build_loop_test(
    run_dir: Path,
    output_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    validations: dict[str, dict[str, Any]],
    *,
    copy_source: bool,
    generate_media: bool,
    media_width: int,
    media_height: int,
    media_fps: int,
    media_every_n_steps: int,
) -> dict[str, Any]:
    checkpoint = _checkpoint_from_report_path(report_path)
    step = _checkpoint_to_step(checkpoint)
    loop_test_id = _loop_test_id_from_report_path(report_path, checkpoint)
    loop_dir = output_dir / "loop_tests" / loop_test_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    source_dir = loop_dir / "source"
    if copy_source:
        source_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, source_dir / report_path.name)
        _copy_qwen_artifacts(report_path.parent, source_dir)

    episode_rows = []
    for episode in report.get("episodes") or []:
        episode_rows.append(
            _write_episode_timeline(
                loop_dir=loop_dir,
                source_dir=source_dir,
                episode=episode,
                report=report,
                checkpoint=checkpoint,
                step=step,
                copy_source=copy_source,
                generate_media=generate_media,
                media_width=media_width,
                media_height=media_height,
                media_fps=media_fps,
                media_every_n_steps=media_every_n_steps,
            )
        )

    validation = validations.get(checkpoint, {})
    loop_manifest = {
        "schema_version": SCHEMA_VERSION,
        "loop_test_id": loop_test_id,
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "pick_up_cube",
        "policy_type": "qwen_chain",
        "policy_label": "Qwen chain + SmolVLA",
        "checkpoint": checkpoint,
        "training_step": step,
        "validation_loss": validation.get("loss"),
        "success_rate": report.get("success_rate"),
        "status": report.get("status"),
        "status_meaning": "evaluator_completed" if report.get("status") == "passed" else report.get("status"),
        "episodes_requested": report.get("episodes_requested"),
        "episodes_completed": report.get("episodes_completed"),
        "seed": report.get("seed"),
        "qwen_plan": report.get("plan"),
        "qwen_prompts": _qwen_prompts(report.get("plan") or {}),
        "qwen_raw": _qwen_raw_paths(source_dir if copy_source else report_path.parent),
        "report_path": str(report_path),
        "source_report_path": str(source_dir / report_path.name) if copy_source else str(report_path),
        "episodes": episode_rows,
    }
    (loop_dir / "loop_test_manifest.json").write_text(
        json.dumps(loop_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "loop_test_id": loop_test_id,
        "manifest_path": str(loop_dir / "loop_test_manifest.json"),
        "checkpoint": checkpoint,
        "training_step": step,
        "scenario": loop_manifest["scenario"],
        "policy_type": loop_manifest["policy_type"],
        "policy_label": loop_manifest["policy_label"],
        "validation_loss": loop_manifest["validation_loss"],
        "success_rate": loop_manifest["success_rate"],
        "status": loop_manifest["status"],
        "episodes_completed": loop_manifest["episodes_completed"],
    }


def _write_episode_timeline(
    *,
    loop_dir: Path,
    source_dir: Path,
    episode: dict[str, Any],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    copy_source: bool,
    generate_media: bool,
    media_width: int,
    media_height: int,
    media_fps: int,
    media_every_n_steps: int,
) -> dict[str, Any]:
    episode_index = int(episode.get("episode") or 0)
    episode_dir = loop_dir / "episodes" / f"episode_{episode_index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(episode.get("trace_path") or "")
    if copy_source and trace_path.exists():
        shutil.copy2(trace_path, source_dir / trace_path.name)
    records = _read_jsonl(trace_path)
    if generate_media:
        records = _generate_media_for_records(
            records=records,
            report=report,
            episode=episode,
            episode_dir=episode_dir,
            width=media_width,
            height=media_height,
            fps=media_fps,
            every_n_steps=media_every_n_steps,
        )
    timeline = _timeline_rows(
        records,
        report,
        checkpoint,
        step,
        episode_index,
        episode_dir=episode_dir,
        copy_source=copy_source,
    )
    timeline_path = episode_dir / "timeline.jsonl"
    timeline_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in timeline) + "\n", encoding="utf-8")
    episode_manifest = {
        "episode_index": episode_index,
        "final_success": episode.get("final_success"),
        "total_reward": episode.get("total_reward"),
        "steps": episode.get("steps"),
        "reset_info": episode.get("reset_info"),
        "final_info": episode.get("final_info"),
        "timeline_path": str(timeline_path),
        "source_trace_path": str(source_dir / trace_path.name) if copy_source and trace_path.exists() else str(trace_path),
        "media_root": str(episode_dir / "media"),
        "iterations": _iteration_summary(timeline),
    }
    (episode_dir / "episode_manifest.json").write_text(
        json.dumps(episode_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return episode_manifest


def _timeline_rows(
    records: list[dict[str, Any]],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
    *,
    episode_dir: Path,
    copy_source: bool,
) -> list[dict[str, Any]]:
    plan = report.get("plan") or {}
    rows: list[dict[str, Any]] = [
        {
            "type": "planner_call",
            "iteration": 0,
            "checkpoint": checkpoint,
            "training_step": step,
            "episode": episode_index,
            "policy": {"type": "qwen_chain", "model": plan.get("model"), "thinking_mode": plan.get("thinking_mode")},
            "policy_input": {
                "task": plan.get("task"),
                "system_prompt": DEFAULT_QWEN_SYSTEM_PROMPT,
                "user_prompt": DEFAULT_QWEN_USER_PROMPT_TEMPLATE.format(
                    task=plan.get("task") or "",
                    target_object=_target_object(plan),
                ),
            },
            "policy_output": {"tool_calls": [_tool_call_payload(call) for call in plan.get("calls") or []]},
            "robot": None,
            "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
        }
    ]
    current_key: tuple[str | None, str | None] | None = None
    iteration = 0
    primitive_step_counts: dict[tuple[str | None, str | None], int] = defaultdict(int)
    for record in records:
        key = (record.get("primitive_id"), record.get("fn"))
        if key != current_key:
            if current_key is not None:
                rows.append(_tool_end_row(iteration, current_key, record, checkpoint, step, episode_index))
            iteration += 1
            current_key = key
            rows.append(_tool_start_row(iteration, record, report, checkpoint, step, episode_index))
        primitive_step_counts[key] += 1
        rows.append(
            _policy_step_row(
                iteration,
                record,
                report,
                checkpoint,
                step,
                episode_index,
                episode_dir=episode_dir,
                copy_source=copy_source,
            )
        )
    if current_key is not None and records:
        rows.append(_tool_end_row(iteration, current_key, records[-1], checkpoint, step, episode_index))
    rows.append(
        {
            "type": "episode_end",
            "iteration": iteration + 1,
            "checkpoint": checkpoint,
            "training_step": step,
            "episode": episode_index,
            "policy": None,
            "policy_input": None,
            "policy_output": None,
            "robot": {"final_info": (report.get("episodes") or [{}])[episode_index].get("final_info")},
            "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
        }
    )
    return rows


def _tool_start_row(
    iteration: int,
    record: dict[str, Any],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
) -> dict[str, Any]:
    rollout_config = _rollout_config_for_record(record, report)
    return {
        "type": "tool_call_start",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "tool_call": record.get("fn"),
        "tool_parameters": _tool_parameters_from_record(record),
        "primitive_id": record.get("primitive_id"),
        "policy": {"type": "smolvla", "policy_path": record.get("policy_path")},
        "policy_input": {"prompt": record.get("prompt")},
        "policy_output": {
            "tool_call": _tool_call_from_record(iteration, record),
            "rollout_config": rollout_config,
            "action_chunk_contract": _action_chunk_contract(rollout_config),
        },
        "robot": None,
        "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
    }


def _policy_step_row(
    iteration: int,
    record: dict[str, Any],
    report: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
    *,
    episode_dir: Path,
    copy_source: bool,
) -> dict[str, Any]:
    rollout_config = _rollout_config_for_record(record, report)
    contract = _action_chunk_contract(rollout_config)
    primitive_step = _int_or_none(record.get("primitive_step"))
    used_per_chunk = _int_or_none(contract.get("used_per_chunk"))
    chunk_index = primitive_step // used_per_chunk if primitive_step is not None and used_per_chunk else None
    chunk_step_index = primitive_step % used_per_chunk if primitive_step is not None and used_per_chunk else None
    media = _media_payload(record, episode_dir=episode_dir, copy_source=copy_source)
    return {
        "type": "policy_step",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "primitive_step": record.get("primitive_step"),
        "tool_call": record.get("fn"),
        "tool_parameters": _tool_parameters_from_record(record),
        "primitive_id": record.get("primitive_id"),
        "policy": {"type": "smolvla", "policy_path": record.get("policy_path")},
        "policy_input": {
            "prompt": record.get("prompt"),
            "observation": record.get("observation"),
            "image_feature_mapping": record.get("image_feature_mapping"),
            "images": {},
        },
        "policy_output": {
            "action": record.get("action"),
            "action_chunk": {
                **contract,
                "chunk_index": chunk_index,
                "chunk_step_index": chunk_step_index,
                "executed_step_count": 1,
            },
        },
        "robot": {
            "reward": record.get("reward"),
            "info": record.get("info"),
            "terminated": record.get("terminated"),
            "truncated": record.get("truncated"),
        },
        "media": media,
        "source": {"record": record},
    }


def _tool_end_row(
    iteration: int,
    key: tuple[str | None, str | None],
    record: dict[str, Any],
    checkpoint: str,
    step: int | None,
    episode_index: int,
) -> dict[str, Any]:
    primitive_id, fn = key
    return {
        "type": "tool_call_end",
        "iteration": iteration,
        "checkpoint": checkpoint,
        "training_step": step,
        "episode": episode_index,
        "global_step": record.get("global_step"),
        "tool_call": fn,
        "tool_parameters": _tool_parameters_from_record(record),
        "primitive_id": primitive_id,
        "policy": None,
        "policy_input": None,
        "policy_output": None,
        "robot": {"last_info": record.get("info"), "last_reward": record.get("reward")},
        "media": {"available": False, "reason": "legacy rollout has no saved frames or videos"},
    }


def _iteration_summary(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in timeline:
        if row.get("type") == "tool_call_start":
            action_steps = [
                step
                for step in timeline
                if step.get("type") == "policy_step" and step.get("iteration") == row.get("iteration")
            ]
            contract = ((row.get("policy_output") or {}).get("action_chunk_contract") or {})
            chunks = _chunk_groups(action_steps)
            rows.append(
                {
                    "iteration": row.get("iteration"),
                    "tool_call": row.get("tool_call"),
                    "tool_parameters": row.get("tool_parameters"),
                    "primitive_id": row.get("primitive_id"),
                    "start_global_step": row.get("global_step"),
                    "action_chunk_summary": {
                        **contract,
                        "chunk_count": len(chunks),
                        "executed_action_steps": len(action_steps),
                        "recorded_policy_steps": len(action_steps),
                        "chunks": chunks,
                    },
                }
            )
    return rows


def _rollout_config_for_record(record: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    for source, value in (
        ("recorded_rollout_config", record.get("policy_rollout_config")),
        ("recorded_report_config", report.get("policy_rollout_config")),
        ("policy_metadata_config", _metadata_rollout_config(report, record.get("policy_path"))),
    ):
        if isinstance(value, dict) and value:
            return {**value, "source": source, "confirmed_in_rollout": source.startswith("recorded_")}
    config = _checkpoint_config(record.get("policy_path"))
    if config:
        return {**config, "source": "checkpoint_config_file", "confirmed_in_rollout": False}
    return {
        "chunk_size": None,
        "n_action_steps": None,
        "num_steps": None,
        "source": "not_recorded",
        "confirmed_in_rollout": False,
    }


def _metadata_rollout_config(report: dict[str, Any], policy_path: Any) -> dict[str, Any] | None:
    metadata = report.get("policy_metadata") or {}
    if policy_path in metadata and isinstance(metadata[policy_path], dict):
        value = metadata[policy_path].get("rollout_config")
        return value if isinstance(value, dict) else None
    return None


def _checkpoint_config(policy_path: Any) -> dict[str, Any]:
    if not policy_path:
        return {}
    config_path = Path(str(policy_path)) / "pretrained_model" / "config.json"
    if not config_path.exists():
        config_path = Path(str(policy_path)) / "config.json"
    if not config_path.exists():
        return {}
    try:
        payload = _read_json(config_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "chunk_size": payload.get("chunk_size"),
        "n_action_steps": payload.get("n_action_steps"),
        "num_steps": payload.get("num_steps"),
    }


def _action_chunk_contract(rollout_config: dict[str, Any]) -> dict[str, Any]:
    generated = _int_or_none(rollout_config.get("chunk_size"))
    used = _int_or_none(rollout_config.get("n_action_steps"))
    confirmed = bool(rollout_config.get("confirmed_in_rollout"))
    source = rollout_config.get("source") or "not_recorded"
    if confirmed:
        note = "rollout report explicitly records this SmolVLA action horizon"
    elif source == "checkpoint_config_file":
        note = "checkpoint config only; legacy rollout did not explicitly record the applied horizon"
    else:
        note = "legacy rollout did not record SmolVLA chunk horizon"
    return {
        "generated_count": generated,
        "used_per_chunk": used,
        "rollout_config_source": source,
        "confirmed_in_rollout": confirmed,
        "note": note,
    }


def _chunk_groups(action_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    no_chunk_rows = []
    for row in action_steps:
        action_chunk = ((row.get("policy_output") or {}).get("action_chunk") or {})
        chunk_index = action_chunk.get("chunk_index")
        if isinstance(chunk_index, int):
            chunks[chunk_index].append(row)
        else:
            no_chunk_rows.append(row)
    if no_chunk_rows and not chunks:
        return [
            {
                "chunk_index": None,
                "used_count": len(no_chunk_rows),
                "start_global_step": no_chunk_rows[0].get("global_step"),
                "end_global_step": no_chunk_rows[-1].get("global_step"),
            }
        ]
    out = []
    for chunk_index in sorted(chunks):
        rows = chunks[chunk_index]
        out.append(
            {
                "chunk_index": chunk_index,
                "used_count": len(rows),
                "start_global_step": rows[0].get("global_step"),
                "end_global_step": rows[-1].get("global_step"),
            }
        )
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _copy_qwen_artifacts(report_dir: Path, source_dir: Path) -> None:
    qwen_dir = report_dir / "qwen"
    if not qwen_dir.exists():
        return
    target = source_dir / "qwen"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(qwen_dir, target)


def _qwen_raw_paths(root: Path) -> dict[str, str | None]:
    qwen_dir = root / "qwen"
    return {
        "manifest_path": str(qwen_dir / "qwen_artifacts_manifest.json")
        if (qwen_dir / "qwen_artifacts_manifest.json").exists()
        else None,
        "request_path": str(qwen_dir / "qwen_raw_request.json")
        if (qwen_dir / "qwen_raw_request.json").exists()
        else None,
        "response_path": str(qwen_dir / "qwen_raw_response.json")
        if (qwen_dir / "qwen_raw_response.json").exists()
        else None,
        "plan_path": str(qwen_dir / "qwen_plan.json") if (qwen_dir / "qwen_plan.json").exists() else None,
    }


def _media_payload(record: dict[str, Any], *, episode_dir: Path, copy_source: bool) -> dict[str, Any]:
    source = record.get("media") if isinstance(record.get("media"), dict) else {}
    if not source:
        return {"available": False, "reason": "legacy rollout has no saved frames or videos"}
    media_root = episode_dir / "media"
    policy_images = {
        str(name): _copy_media_path(path, media_root / "policy_inputs", copy_source=copy_source)
        for name, path in (source.get("policy_input_images") or {}).items()
        if path
    }
    robot_frame = _copy_media_path(source.get("robot_frame"), media_root / "robot_frames", copy_source=copy_source)
    iteration_video_gif = _copy_media_path(
        source.get("iteration_video_gif"),
        media_root / "videos",
        copy_source=copy_source,
    )
    iteration_video_mp4 = _copy_media_path(
        source.get("iteration_video_mp4"),
        media_root / "videos",
        copy_source=copy_source,
    )
    available = bool(policy_images or robot_frame or iteration_video_gif or iteration_video_mp4)
    return {
        "available": available,
        "reason": None if available else "no saved media paths in trace",
        "policy_input_images": policy_images,
        "robot_frame": robot_frame,
        "iteration_video_gif": iteration_video_gif,
        "iteration_video_mp4": iteration_video_mp4,
    }


def _copy_media_path(path_value: Any, target_dir: Path, *, copy_source: bool) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return str(path)
    if not copy_source:
        return str(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return str(target)


def _generate_media_for_records(
    *,
    records: list[dict[str, Any]],
    report: dict[str, Any],
    episode: dict[str, Any],
    episode_dir: Path,
    width: int,
    height: int,
    fps: int,
    every_n_steps: int,
) -> list[dict[str, Any]]:
    if not records:
        return records
    try:
        import mujoco

        from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
        from physical_ai_agent.sim.so101_nexus_env import SO101NexusEnv
    except Exception:
        return records

    env_id = str(report.get("env_id") or (records[0].get("render_replay") or {}).get("env_id") or "MuJoCoReach-v1")
    seed = int(episode.get("seed") or report.get("seed") or (records[0].get("render_replay") or {}).get("seed") or 0)
    media_root = episode_dir / "media"
    env = None
    renderers: dict[str, Any] = {}
    generated = [dict(record) for record in records]
    primitive_frames: dict[tuple[Any, Any], list[str]] = defaultdict(list)
    try:
        env = SO101NexusEnv(env_id, None)
        raw_env = getattr(env, "env", env)
        renderers = {
            name: mujoco.Renderer(raw_env.unwrapped.model, height=int(height), width=int(width))
            for name in ("egocentric_cam", "wrist_cam", "top_down")
        }
        env.reset(seed=seed)
        for index, record in enumerate(generated):
            global_step = _int_or_none(record.get("global_step")) or index
            primitive_step = _int_or_none(record.get("primitive_step")) or 0
            should_render = primitive_step % max(1, int(every_n_steps)) == 0
            media = dict(record.get("media") or {})
            media["render_mode"] = "generated_local" if should_render else media.get("render_mode", "deferred")
            if should_render:
                policy_images = {}
                for camera_name in ("egocentric_cam", "wrist_cam"):
                    renderer = renderers[camera_name]
                    renderer.update_scene(raw_env.unwrapped.data, camera=_make_camera(raw_env, camera_name))
                    pixels = postprocess_camera_frame(camera_name, renderer.render())
                    path = media_root / "policy_inputs" / f"step_{global_step:04d}_{camera_name}.png"
                    _write_image(path, pixels)
                    policy_images[camera_name] = str(path)
                media["policy_input_images"] = policy_images

            env.step(record.get("action") or [])

            if should_render:
                renderer = renderers["top_down"]
                renderer.update_scene(raw_env.unwrapped.data, camera=_make_camera(raw_env, "top_down"))
                frame_path = media_root / "robot_frames" / f"step_{global_step:04d}_top_down.png"
                _write_image(frame_path, renderer.render())
                media["robot_frame"] = str(frame_path)
                primitive_key = (record.get("primitive_id"), record.get("fn"))
                primitive_frames[primitive_key].append(str(frame_path))
            record["media"] = media

        videos_by_primitive = _write_generated_primitive_videos(
            primitive_frames=primitive_frames,
            media_root=media_root,
            fps=fps,
        )
        for record in generated:
            videos = videos_by_primitive.get((record.get("primitive_id"), record.get("fn")))
            if videos:
                record.setdefault("media", {}).update(videos)
    except Exception:
        return records
    finally:
        for renderer in renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        if env is not None:
            env.close()
    return generated


def _write_generated_primitive_videos(
    *,
    primitive_frames: dict[tuple[Any, Any], list[str]],
    media_root: Path,
    fps: int,
) -> dict[tuple[Any, Any], dict[str, str]]:
    try:
        import imageio.v2 as imageio
    except Exception:
        return {}
    out = {}
    videos_dir = media_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for index, (key, frame_paths) in enumerate(primitive_frames.items(), start=1):
        if not frame_paths:
            continue
        frames = [imageio.imread(path) for path in frame_paths]
        primitive_id = str(key[0] or "primitive")
        stem = f"iteration_{index:02d}_{primitive_id}"
        gif_path = videos_dir / f"{stem}.gif"
        mp4_path = videos_dir / f"{stem}.mp4"
        imageio.mimsave(gif_path, frames, fps=max(1, int(fps)))
        imageio.mimsave(mp4_path, frames, fps=max(1, int(fps)))
        out[key] = {
            "iteration_video_gif": str(gif_path),
            "iteration_video_mp4": str(mp4_path),
        }
    return out


def _write_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(path)


def _qwen_prompts(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": DEFAULT_QWEN_SYSTEM_PROMPT,
        "user": DEFAULT_QWEN_USER_PROMPT_TEMPLATE.format(
            task=plan.get("task") or "",
            target_object=_target_object(plan),
        ),
    }


def _target_object(plan: dict[str, Any]) -> str:
    calls = plan.get("calls") or []
    for call in calls:
        if isinstance(call, dict) and call.get("object"):
            return str(call["object"])
    return "green cube"


def _tool_call_payload(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "function": call.get("fn"),
        "parameters": {
            "object": call.get("object"),
            "primitive_id": call.get("primitive_id"),
            "prompt": call.get("prompt"),
            "max_steps": call.get("max_steps"),
        },
        "index": call.get("index"),
    }


def _tool_call_from_record(iteration: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": iteration - 1,
        "function": record.get("fn"),
        "parameters": _tool_parameters_from_record(record),
    }


def _tool_parameters_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": _object_from_prompt(record.get("prompt")),
        "primitive_id": record.get("primitive_id"),
        "prompt": record.get("prompt"),
    }


def _object_from_prompt(prompt: Any) -> str | None:
    text = str(prompt or "")
    for marker in ("green cube", "cube"):
        if marker in text:
            return marker
    return None


def _closed_loop_report_paths(run_dir: Path) -> list[Path]:
    root = run_dir / "closed_loop_evals"
    if not root.exists():
        return []
    return sorted(root.glob("*/qwen_closed_loop_eval_report.json"))


def _validation_by_checkpoint(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl")
    return {str(row["checkpoint"]): row for row in rows if row.get("checkpoint")}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_from_report_path(path: Path) -> str:
    match = path.parent.name.rsplit("_", 1)
    if len(match) == 2 and match[1].isdigit():
        return match[1]
    return path.parent.name


def _loop_test_id_from_report_path(path: Path, checkpoint: str) -> str:
    parent = path.parent.name
    canonical = f"qwen_chain_seed98100_{checkpoint}"
    if parent == canonical:
        return f"qwen_chain_{checkpoint}"
    if parent.startswith("qwen_chain_") and parent.endswith(f"_{checkpoint}"):
        suffix = parent.removeprefix("qwen_chain_").removesuffix(f"_{checkpoint}")
        suffix = suffix.removeprefix("seed98100_").strip("_")
        if suffix:
            return f"qwen_chain_{suffix}_{checkpoint}"
    return f"qwen_chain_{checkpoint}"


def _checkpoint_to_step(checkpoint: str) -> int | None:
    try:
        return int(checkpoint)
    except ValueError:
        return None


def _write_summary_tables(output_dir: Path, loop_tests: list[dict[str, Any]]) -> None:
    metrics_path = output_dir / "joined_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "loop_test_id",
                "checkpoint",
                "training_step",
                "scenario",
                "policy_type",
                "validation_loss",
                "success_rate",
                "status",
                "episodes_completed",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in loop_tests)

    episode_rows = []
    primitive_rows = []
    for row in loop_tests:
        detail = _read_json(Path(row["manifest_path"]))
        for episode in detail.get("episodes") or []:
            episode_rows.append(
                {
                    "loop_test_id": detail.get("loop_test_id"),
                    "checkpoint": detail.get("checkpoint"),
                    "episode": episode.get("episode_index"),
                    "final_success": episode.get("final_success"),
                    "total_reward": episode.get("total_reward"),
                    "steps": episode.get("steps"),
                    "final_tcp_to_target_dist": (episode.get("final_info") or {}).get("tcp_to_target_dist"),
                }
            )
            for iteration in episode.get("iterations") or []:
                summary = iteration.get("action_chunk_summary") or {}
                primitive_rows.append(
                    {
                        "loop_test_id": detail.get("loop_test_id"),
                        "checkpoint": detail.get("checkpoint"),
                        "episode": episode.get("episode_index"),
                        "iteration": iteration.get("iteration"),
                        "tool_call": iteration.get("tool_call"),
                        "primitive_id": iteration.get("primitive_id"),
                        "executed_action_steps": summary.get("executed_action_steps"),
                        "chunk_count": summary.get("chunk_count"),
                        "generated_count": summary.get("generated_count"),
                        "used_per_chunk": summary.get("used_per_chunk"),
                        "confirmed_in_rollout": summary.get("confirmed_in_rollout"),
                    }
                )

    _write_csv(output_dir / "episode_summary.csv", episode_rows)
    _write_csv(output_dir / "primitive_summary.csv", primitive_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_standalone_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("loop_tests") or []
    table_rows = "\n".join(
        "<tr>"
        f"<td>{_html(row.get('loop_test_id'))}</td>"
        f"<td>{_html(row.get('checkpoint'))}</td>"
        f"<td>{_html(row.get('training_step'))}</td>"
        f"<td>{_html(row.get('validation_loss'))}</td>"
        f"<td>{_html(row.get('success_rate'))}</td>"
        f"<td>{_html(row.get('status'))}</td>"
        "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Loop Test Analyzer Export</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8dde7;padding:6px 8px;text-align:left}}th{{background:#f5f7fb}}</style>
</head><body>
<h1>Loop Test Analyzer Export</h1>
<p>Generated at {_html(manifest.get('generated_at'))}. Loop tests: {_html(len(rows))}.</p>
<table><thead><tr><th>Loop</th><th>Checkpoint</th><th>Step</th><th>Val loss</th><th>Success</th><th>Status</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    (output_dir / "standalone_report.html").write_text(html, encoding="utf-8")


def _html(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_zip_bundle(output_dir: Path) -> None:
    zip_path = output_dir / "loop_test_analyzer_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            bundle.write(path, path.relative_to(output_dir))


if __name__ == "__main__":
    main()
