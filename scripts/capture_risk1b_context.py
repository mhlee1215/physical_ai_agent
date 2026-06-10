#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from physical_ai_agent.imagine_then_act.libero_config import ensure_noninteractive_libero_config
from physical_ai_agent.imagine_then_act.risk_probes import (
    RiskProbeConfig,
    apply_torch_transformers_import_compatibility_patch,
    build_lerobot_eval_argv,
    renderer_env,
    renderer_env_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Risk1-B VLM context artifacts. mock is contract-only; libero "
            "captures actual LeRobot/LIBERO start observation."
        )
    )
    parser.add_argument("--backend", choices=("mock", "artifact", "libero"), default="mock")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--camera-mapping", default='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}')
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--renderer-backend", choices=("egl", "osmesa", "auto"), default="egl")
    parser.add_argument("--actual-timeout-sec", type=int, default=600)
    parser.add_argument("--output-dir", default="_workspace/runpod_results/ita_risk_probes/risk1b_context")
    parser.add_argument("--artifact-image", default=None)
    parser.add_argument("--artifact-context-json", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def contact_sheet_path(output_dir: Path, task_id: int, seed: int) -> Path:
    return output_dir / f"contact_sheet_task{task_id}_seed{seed}.png"


def context_json_path(output_dir: Path, task_id: int, seed: int) -> Path:
    return output_dir / f"context_task{task_id}_seed{seed}.json"


def write_context_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_png(path: Path, image: list[list[list[int]]]) -> None:
    if not image or not image[0]:
        image = solid_image(32, 32, (0, 0, 0))
    height = len(image)
    width = len(image[0])
    raw = b"".join(bytes([0]) + bytes(clamp_channel(value) for pixel in row for value in pixel[:3]) for row in image)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def clamp_channel(value: Any) -> int:
    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        numeric *= 255.0
    return max(0, min(255, int(round(numeric))))


def solid_image(width: int, height: int, color: tuple[int, int, int]) -> list[list[list[int]]]:
    return [[list(color) for _x in range(width)] for _y in range(height)]


def make_contact_sheet(images: list[list[list[list[int]]]]) -> list[list[list[int]]]:
    if not images:
        return solid_image(96, 64, (32, 32, 32))
    height = max(len(image) for image in images)
    widths = [len(image[0]) if image and image[0] else 1 for image in images]
    total_width = sum(widths) + 4 * (len(images) - 1)
    sheet = solid_image(total_width, height, (245, 245, 245))
    cursor = 0
    for image, width in zip(images, widths):
        for y, row in enumerate(image):
            for x, pixel in enumerate(row[:width]):
                if y < height and cursor + x < total_width:
                    sheet[y][cursor + x] = [clamp_channel(value) for value in pixel[:3]]
        cursor += width + 4
    return sheet


def mock_context(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        solid_image(64, 48, (80, 120, 190)),
        solid_image(64, 48, (190, 120, 80)),
    ]
    image_paths = []
    for name, image in zip(("agentview_image", "robot0_eye_in_hand_image"), images):
        image_path = output_dir / f"camera_{name}_task{args.task_id}_seed{args.seed}.png"
        write_png(image_path, image)
        image_paths.append({"camera_key": name, "path": str(image_path), "source": "mock_contract"})
    sheet_path = contact_sheet_path(output_dir, args.task_id, args.seed)
    write_png(sheet_path, make_contact_sheet(images))
    context_path = context_json_path(output_dir, args.task_id, args.seed)
    write_context_json(
        context_path,
        {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "task_description": "mock contract context; not actual LIBERO observation",
            "observation_source": "mock_contract",
            "camera_images": image_paths,
            "contact_sheet": str(sheet_path),
            "timestamp_unix": round(time.time(), 3),
            "provenance": {
                "backend": "mock",
                "actual_context": False,
                "claim_boundary": "mock context is local plumbing only and must not drive Risk1-B PASS",
            },
        },
    )
    return sheet_path, context_path


def artifact_context(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.artifact_image or not args.artifact_context_json:
        raise ValueError("--artifact-image and --artifact-context-json are required for backend=artifact")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = contact_sheet_path(output_dir, args.task_id, args.seed)
    context_path = context_json_path(output_dir, args.task_id, args.seed)
    shutil.copyfile(args.artifact_image, sheet_path)
    payload = json.loads(Path(args.artifact_context_json).read_text(encoding="utf-8"))
    payload.setdefault("suite", args.suite)
    payload.setdefault("task_id", args.task_id)
    payload.setdefault("seed", args.seed)
    payload["contact_sheet"] = str(sheet_path)
    payload["observation_source"] = payload.get("observation_source", "existing_actual_artifact")
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    provenance.setdefault("backend", "artifact")
    provenance.setdefault("actual_context", bool(provenance.get("actual_context")))
    payload["provenance"] = provenance
    write_context_json(context_path, payload)
    return sheet_path, context_path


def libero_context(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = RiskProbeConfig(
        preset="runpod-libero-smoke",
        backend="libero-contract",
        suite=args.suite,
        task_ids=(args.task_id,),
        seed=args.seed,
        num_candidates=2,
        chunk_steps=args.policy_n_action_steps,
        action_dim=7,
        output_dir=str(output_dir),
        policy_path=args.policy_path,
        camera_mapping=args.camera_mapping,
        policy_num_steps=args.policy_num_steps,
        policy_n_action_steps=args.policy_n_action_steps,
        actual_timeout_sec=args.actual_timeout_sec,
        renderer_backend=args.renderer_backend,
    )
    import_compat = apply_torch_transformers_import_compatibility_patch()
    old_argv = sys.argv[:]
    try:
        with renderer_env(args.renderer_backend):
            ensure_noninteractive_libero_config()
            from lerobot.scripts import lerobot_eval

            old_rollout = getattr(lerobot_eval, "rollout", None)
            try:
                lerobot_eval.rollout = build_context_rollout(args, output_dir, import_compat)
                sys.argv = ["lerobot-eval", *build_lerobot_eval_argv(config, output_dir)]
                lerobot_eval.main()
            finally:
                if old_rollout is not None:
                    lerobot_eval.rollout = old_rollout
    finally:
        sys.argv = old_argv
    return contact_sheet_path(output_dir, args.task_id, args.seed), context_json_path(output_dir, args.task_id, args.seed)


def build_context_rollout(args: argparse.Namespace, output_dir: Path, import_compat: dict[str, Any]):
    def rollout(
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
        del env_preprocessor, env_postprocessor, preprocessor, postprocessor, return_observations
        import numpy as np
        import torch
        from lerobot.utils.constants import ACTION

        if hasattr(policy, "reset"):
            policy.reset()
        rollout_seed = seeds if seeds is not None else [args.seed]
        observation, info = env.reset(seed=rollout_seed)
        if render_callback is not None:
            try:
                render_callback(env)
            except Exception:  # noqa: BLE001
                pass
        task_descriptions = safe_env_call(env, "task_description") or safe_env_call(env, "task") or []
        images = extract_images_from_observation(observation)
        camera_images = []
        image_matrices = []
        for key, image in images:
            image_path = output_dir / f"camera_{sanitize_name(key)}_task{args.task_id}_seed{args.seed}.png"
            write_png(image_path, image)
            image_matrices.append(image)
            camera_images.append({"camera_key": key, "path": str(image_path), "source": "libero_start_observation"})
        sheet_path = contact_sheet_path(output_dir, args.task_id, args.seed)
        write_png(sheet_path, make_contact_sheet(image_matrices))
        context_path = context_json_path(output_dir, args.task_id, args.seed)
        write_context_json(
            context_path,
            {
                "suite": args.suite,
                "task_id": args.task_id,
                "seed": args.seed,
                "task_description": str(task_descriptions[0]) if task_descriptions else "",
                "task_descriptions": [str(item) for item in task_descriptions],
                "observation_source": "lerobot_libero_env_reset",
                "camera_mapping": args.camera_mapping,
                "camera_images": camera_images,
                "camera_count": len(camera_images),
                "contact_sheet": str(sheet_path),
                "info_summary": summarize_context_value(info),
                "renderer_env": renderer_env_snapshot(config_like(args)),
                "import_compat": import_compat,
                "timestamp_unix": round(time.time(), 3),
                "provenance": {
                    "backend": "libero",
                    "actual_context": True,
                    "env_type": "LeRobot LIBERO",
                    "observation_step": "episode_start_reset",
                },
            },
        )
        action_dim = 1
        action = np.zeros((getattr(env, "num_envs", 1), 1, action_dim), dtype=np.float32)
        zeros = np.zeros((getattr(env, "num_envs", 1), 1), dtype=np.float32)
        done = np.ones((getattr(env, "num_envs", 1), 1), dtype=bool)
        return {
            ACTION: torch.from_numpy(action),
            "reward": torch.from_numpy(zeros),
            "success": torch.from_numpy(zeros.astype(bool)),
            "done": torch.from_numpy(done),
        }

    return rollout


def config_like(args: argparse.Namespace) -> RiskProbeConfig:
    return RiskProbeConfig(
        preset="runpod-libero-smoke",
        backend="libero-contract",
        suite=args.suite,
        task_ids=(args.task_id,),
        seed=args.seed,
        num_candidates=2,
        chunk_steps=args.policy_n_action_steps,
        action_dim=7,
        output_dir=args.output_dir,
        renderer_backend=args.renderer_backend,
    )


def safe_env_call(env: Any, name: str) -> list[Any]:
    try:
        values = env.call(name)
        return list(values)
    except Exception:  # noqa: BLE001
        return []


def extract_images_from_observation(observation: Any) -> list[tuple[str, list[list[list[int]]]]]:
    images: list[tuple[str, list[list[list[int]]]]] = []
    visit_observation(observation, "observation", images)
    return images[:4]


def visit_observation(value: Any, key: str, images: list[tuple[str, list[list[list[int]]]]]) -> None:
    if len(images) >= 4:
        return
    converted = to_rgb_image(value)
    if converted is not None:
        images.append((key, converted))
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            visit_observation(child_value, str(child_key), images)
    elif isinstance(value, (list, tuple)) and len(value) <= 8:
        for index, item in enumerate(value):
            visit_observation(item, f"{key}_{index}", images)


def to_rgb_image(value: Any) -> list[list[list[int]]] | None:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    shape = nested_shape(value)
    if len(shape) == 4 and shape[0] == 1:
        value = value[0]
        shape = nested_shape(value)
    if len(shape) == 3 and shape[0] in {1, 3} and shape[-1] not in {1, 3}:
        value = transpose_chw_to_hwc(value)
        shape = nested_shape(value)
    if len(shape) != 3 or shape[-1] not in {1, 3, 4}:
        return None
    height, width, _channels = shape
    if height < 8 or width < 8:
        return None
    image: list[list[list[int]]] = []
    for row in value:
        image_row = []
        for pixel in row:
            if not isinstance(pixel, list):
                return None
            rgb = pixel[:3] if len(pixel) >= 3 else [pixel[0], pixel[0], pixel[0]]
            image_row.append([clamp_channel(channel) for channel in rgb])
        image.append(image_row)
    return image


def nested_shape(value: Any) -> tuple[int, ...]:
    shape = []
    cursor = value
    while isinstance(cursor, list):
        shape.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]
    return tuple(shape)


def transpose_chw_to_hwc(value: list[Any]) -> list[list[list[Any]]]:
    channels = len(value)
    height = len(value[0])
    width = len(value[0][0])
    return [[[value[channel][y][x] for channel in range(channels)] for x in range(width)] for y in range(height)]


def summarize_context_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 2:
        return str(type(value).__name__)
    if isinstance(value, dict):
        return {str(key): summarize_context_value(child, depth=depth + 1) for key, child in list(value.items())[:16]}
    if isinstance(value, (list, tuple)):
        return [summarize_context_value(item, depth=depth + 1) for item in list(value)[:8]]
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": [int(dim) for dim in value.shape]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "camera"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.backend == "mock":
            sheet_path, context_path = mock_context(args)
        elif args.backend == "artifact":
            sheet_path, context_path = artifact_context(args)
        else:
            sheet_path, context_path = libero_context(args)
    except Exception as exc:  # noqa: BLE001
        print(f"context_capture_error: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
        return 2
    payload = {
        "status": "PASS",
        "backend": args.backend,
        "contact_sheet": str(sheet_path),
        "context_json": str(context_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status=PASS contact_sheet={sheet_path} context_json={context_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
