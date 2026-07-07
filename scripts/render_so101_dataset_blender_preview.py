#!/usr/bin/env python3
"""Render selected SO101 LeRobot dataset rows through Blender Cycles Metal."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from physical_ai_agent.sim.so101_camera_input import EGOCENTRIC_CAMERA1_POSE, _make_camera
from render_so101_blender_probe import BLENDER_DRIVER, _pbr_paths, _write_photo_tabletop_texture
from render_so101_mitsuba_probe import _export_mesh_geoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SO101 dataset row previews with Blender Cycles Metal.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_dataset_blender_preview"))
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument(
        "--env-source",
        choices=("gym", "high_contrast_picklift"),
        default="gym",
        help="Environment source. Use high_contrast_picklift for export_so101_teacher_rollouts_lerobot datasets.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        help="Comma-separated episode indices. Overrides --episode when set.",
    )
    parser.add_argument(
        "--frames",
        default="0,22,44,66,87",
        help="Comma-separated frame indices or labels: start, open, grip, final, all.",
    )
    parser.add_argument("--camera-keys", default="observation.images.camera1,observation.images.camera2")
    parser.add_argument("--duplicate-camera3-from-camera2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-base", type=int, help="Reset seed base. Defaults to seedNNN parsed from dataset root.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--robot-material", choices=("plastic", "matte_pla", "metal"), default="matte_pla")
    parser.add_argument("--camera-lens", type=float, default=48.0)
    parser.add_argument("--asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    args = parser.parse_args()

    blender_bin = shutil.which(args.blender_bin) or args.blender_bin
    episodes = _parse_int_csv(args.episodes) if args.episodes else [int(args.episode)]
    frame_tokens = [item.strip() for item in args.frames.split(",") if item.strip()]
    result = render_dataset_preview(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        env_id=args.env_id,
        env_source=args.env_source,
        episodes=episodes,
        frame_tokens=frame_tokens,
        camera_keys=[item.strip() for item in args.camera_keys.split(",") if item.strip()],
        duplicate_camera3_from_camera2=args.duplicate_camera3_from_camera2,
        seed_base=args.seed_base,
        width=args.width,
        height=args.height,
        samples=args.samples,
        denoise=args.denoise,
        robot_material=args.robot_material,
        camera_lens=args.camera_lens,
        asset_root=args.asset_root,
        blender_bin=blender_bin,
        max_mesh_geoms=args.max_mesh_geoms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_dataset_preview(
    *,
    dataset_root: Path,
    output_dir: Path,
    env_id: str,
    env_source: str,
    episodes: list[int],
    frame_tokens: list[str],
    camera_keys: list[str],
    duplicate_camera3_from_camera2: bool,
    seed_base: int | None,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    robot_material: str,
    camera_lens: float,
    asset_root: Path,
    blender_bin: str,
    max_mesh_geoms: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries = _episode_summaries(dataset_root)
    episode_frames = _resolve_episode_frames(
        dataset_root,
        episodes=episodes,
        frame_tokens=frame_tokens,
        episode_summaries=episode_summaries,
    )
    rows = _dataset_rows_for_replay(dataset_root, episode_frames=episode_frames)
    expected = sum(len(frames) for frames in episode_frames.values())
    selected = [
        row
        for row in rows
        if int(row["frame_index"]) in set(episode_frames[int(row["episode_index"])])
    ]
    if len(selected) != expected:
        found = [(int(row["episode_index"]), int(row["frame_index"])) for row in selected]
        raise ValueError(f"missing requested episode frames: requested={episode_frames}, found={found}")
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)

    rendered: list[dict[str, Any]] = []
    driver_path = output_dir / "blender_driver.py"
    driver_path.write_text(BLENDER_DRIVER, encoding="utf-8")
    texture_path = _write_photo_tabletop_texture(output_dir / "tabletop_texture.png")
    hdri_path = asset_root / "polyhaven" / "studio_small_08_2k.hdr"
    table_pbr_dir = asset_root / "ambientcg" / "Wood008_1K-JPG"
    plastic_pbr_dir = asset_root / "ambientcg" / "Plastic013A_1K-JPG"

    env = _make_env(env_id=env_id, env_source=env_source)
    mujoco_renderers: dict[str, Any] = {}
    try:
        for camera_name in ("egocentric_cam", "wrist_cam"):
            mujoco_renderers[camera_name] = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
        for episode in sorted(rows_by_episode):
            seed = _seed_for_episode(
                episode,
                seed_base=seed_base,
                dataset_root=dataset_root,
                episode_summaries=episode_summaries,
            )
            env.reset(seed=seed)
            unwrapped = env.unwrapped
            actuator_ids = getattr(unwrapped, "_actuator_ids", None)
            for row in rows_by_episode[episode]:
                frame_index = int(row["frame_index"])
                state = [float(value) for value in row["observation.state"]]
                action = [float(value) for value in row["action"]]
                replay_state = unwrapped.data.ctrl[actuator_ids] if actuator_ids is not None else unwrapped.data.ctrl
                state_error = _state_replay_error(replay_state, state)

                if frame_index in episode_frames[episode]:
                    frame_dir = output_dir / f"episode_{episode:04d}_frame_{frame_index:04d}"
                    mesh_dir = frame_dir / "ply"
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    mujoco.mj_forward(unwrapped.model, unwrapped.data)
                    exported = _export_mesh_geoms(
                        unwrapped.model,
                        unwrapped.data,
                        mesh_dir,
                        max_mesh_geoms=max_mesh_geoms,
                    )
                    primitives = _export_primitive_geoms(unwrapped.model, unwrapped.data)
                    camera_specs = _camera_specs_from_mujoco_scene(
                        env,
                        mujoco_renderers,
                        camera_lens=camera_lens,
                    )
                    render_specs = []
                    image_paths: dict[str, str] = {}
                    for camera_key in camera_keys:
                        if camera_key not in camera_specs:
                            raise ValueError(f"unsupported camera key: {camera_key}")
                        image_path = frame_dir / f"episode_{episode:04d}_frame_{frame_index:04d}_{_camera_slug(camera_key)}.png"
                        render_specs.append({"image_path": str(image_path.resolve()), "camera": camera_specs[camera_key]})
                        image_paths[camera_key] = str(image_path)
                    blender_report_path = frame_dir / "blender_device_report.json"
                    spec_path = frame_dir / "blender_scene_spec.json"
                    spec = {
                        "width": width,
                        "height": height,
                        "samples": samples,
                        "denoise": denoise,
                        "cycles_seed": 98200,
                        "sample_clamp_indirect": 0.85,
                        "background_wall": False,
                        "stable_tabletop": True,
                        "robot_material": robot_material,
                        "camera_lens": camera_lens,
                        "texture_path": str(texture_path.resolve()),
                        "hdri_path": str(hdri_path.resolve()) if hdri_path.exists() else None,
                        "table_pbr": _pbr_paths(table_pbr_dir, "Wood008_1K-JPG"),
                        "plastic_pbr": _pbr_paths(plastic_pbr_dir, "Plastic013A_1K-JPG"),
                        "image_path": str(Path(next(iter(image_paths.values()))).resolve()),
                        "blender_report_path": str(blender_report_path.resolve()),
                        "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
                        "primitives": primitives,
                        "target_site": None,
                        "renders": render_specs,
                    }
                    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
                    command = [blender_bin, "--background", "--python", str(driver_path), "--", str(spec_path)]
                    started = time.perf_counter()
                    completed = subprocess.run(command, text=True, capture_output=True, check=False)
                    render_seconds = time.perf_counter() - started
                    log_path = frame_dir / "blender_render.log"
                    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                    if completed.returncode != 0:
                        raise RuntimeError(f"Blender render failed with exit code {completed.returncode}; see {log_path}")
                    shutil.rmtree(mesh_dir, ignore_errors=True)
                    for camera_key, image_path in image_paths.items():
                        rotation = camera_specs[camera_key].get("rotation_degrees", 0)
                        if rotation:
                            _rotate_image(Path(image_path), int(rotation))
                    if duplicate_camera3_from_camera2 and "observation.images.camera2" in image_paths:
                        camera2_path = Path(image_paths["observation.images.camera2"])
                        camera3_path = frame_dir / f"episode_{episode:04d}_frame_{frame_index:04d}_camera3.png"
                        shutil.copyfile(camera2_path, camera3_path)
                        image_paths["observation.images.camera3"] = str(camera3_path)
                    rendered.append(
                        {
                            "episode": episode,
                            "seed": seed,
                            "frame": frame_index,
                            "frame_label": _frame_label(episode_frames, episode=episode, frame=frame_index),
                            "timestamp": float(row["timestamp"]),
                            "state": state,
                            "action": action,
                            "replay_state_error": state_error,
                            "image_path": image_paths.get("observation.images.camera1") or next(iter(image_paths.values())),
                            "image_paths": image_paths,
                            "mesh_geoms_exported": len(exported),
                            "primitive_geoms_exported": len(primitives),
                            "render_seconds": render_seconds,
                        }
                    )
                env.step(np.asarray(action, dtype=float))
    finally:
        for renderer in mujoco_renderers.values():
            renderer.close()
        env.close()
    if len(rendered) != expected:
        raise RuntimeError(f"rendered {len(rendered)} frames, expected {expected}")

    contact_sheet = _write_contact_sheet(rendered, output_dir / "so101_dataset_photoreal_contact_sheet.png")
    report = {
        "dataset_root": str(dataset_root),
        "env_id": env_id,
        "env_source": env_source,
        "episodes": episodes,
        "frame_tokens": frame_tokens,
        "camera_keys": camera_keys,
        "duplicate_camera3_from_camera2": duplicate_camera3_from_camera2,
        "episode_frames": episode_frames,
        "seed_base": seed_base if seed_base is not None else _seed_from_name(dataset_root),
        "renderer": "blender_cycles",
        "robot_material": robot_material,
        "camera_lens": camera_lens,
        "samples": samples,
        "denoise": denoise,
        "width": width,
        "height": height,
        "contact_sheet": str(contact_sheet),
        "renders": rendered,
        "note": (
            "Episode state is replayed from the source LeRobot actions: each episode "
            "resets once with the recorded seed, renders the pre-action frame, then "
            "steps the source action so cube/contact dynamics remain continuous."
        ),
    }
    (output_dir / "so101_dataset_blender_preview_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _dataset_rows_for_replay(dataset_root: Path, *, episode_frames: dict[int, list[int]]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing LeRobot parquet data files under {dataset_root}")
    table = pq.read_table(
        [str(path) for path in data_files],
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    max_frame_by_episode = {int(episode): max(int(frame) for frame in frames) for episode, frames in episode_frames.items()}
    rows = [
        row
        for row in _rows(table.to_pydict())
        if int(row["episode_index"]) in max_frame_by_episode
        and int(row["frame_index"]) <= max_frame_by_episode[int(row["episode_index"])]
    ]
    return sorted(rows, key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))


def _state_replay_error(qpos: Any, state: list[float]) -> float:
    total = 0.0
    for observed, expected in zip(qpos, state, strict=False):
        delta = float(observed) - float(expected)
        total += delta * delta
    return total**0.5


def _export_primitive_geoms(model: Any, data: Any) -> list[dict[str, Any]]:
    import mujoco

    names = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    }
    primitives: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        kind = names.get(int(model.geom_type[geom_id]))
        if kind is None:
            continue
        rgba = _geom_rgba(model, geom_id)
        if len(rgba) >= 4 and rgba[3] <= 0.01:
            continue
        name = model.geom(geom_id).name or f"geom_{geom_id:03d}"
        semantic_color = "red_cube" if "cube" in name or "pick_slot" in name else None
        primitives.append(
            {
                "geom_id": geom_id,
                "name": name,
                "type": kind,
                "position": [float(value) for value in data.geom_xpos[geom_id]],
                "xmat": [float(value) for value in data.geom_xmat[geom_id]],
                "size": [float(value) for value in model.geom_size[geom_id]],
                "rgba": rgba,
                "semantic_color": semantic_color,
            }
        )
    return primitives


def _geom_rgba(model: Any, geom_id: int) -> list[float]:
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return [float(value) for value in model.mat_rgba[mat_id]]
    return [float(value) for value in model.geom_rgba[geom_id]]


def _make_env(*, env_id: str, env_source: str) -> Any:
    if env_source == "high_contrast_picklift":
        from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

        return make_high_contrast_picklift_env()
    import gymnasium as gym

    return gym.make(env_id, render_mode=None)


def _write_contact_sheet(rendered: list[dict[str, Any]], output_path: Path) -> Path:
    images = [Image.open(item["image_path"]).convert("RGB") for item in rendered]
    cell_w, cell_h, label_h = 320, 240, 38
    episode_counts: dict[int, int] = {}
    for item in rendered:
        episode_counts[int(item["episode"])] = episode_counts.get(int(item["episode"]), 0) + 1
    columns = max(1, max(episode_counts.values()) if episode_counts else 1)
    if len(images) <= 5:
        columns = len(images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, (cell_h + label_h) * rows), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, (image, item) in enumerate(zip(images, rendered, strict=True)):
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        col = index % columns
        row = index // columns
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * (cell_h + label_h) + label_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))
        label = str(item.get("frame_label") or item["frame"])
        draw.text(
            (col * cell_w + 10, row * (cell_h + label_h) + 12),
            f"ep {item['episode']} | {label} | frame {item['frame']}",
            fill=(25, 25, 25),
        )
    sheet.save(output_path)
    return output_path


def _seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        return 0
    return int(match.group(1))


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _episode_summaries(dataset_root: Path) -> dict[int, dict[str, Any]]:
    report_path = dataset_root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    return {index: episode for index, episode in enumerate(episodes) if isinstance(episode, dict)}


def _seed_for_episode(
    episode: int,
    *,
    seed_base: int | None,
    dataset_root: Path,
    episode_summaries: dict[int, dict[str, Any]],
) -> int:
    if episode in episode_summaries and "seed" in episode_summaries[episode]:
        return int(episode_summaries[episode]["seed"])
    return (seed_base if seed_base is not None else _seed_from_name(dataset_root)) + int(episode)


def _resolve_episode_frames(
    dataset_root: Path,
    *,
    episodes: list[int],
    frame_tokens: list[str],
    episode_summaries: dict[int, dict[str, Any]],
) -> dict[int, list[int]]:
    ranges = _episode_frame_ranges(dataset_root)
    episode_frames: dict[int, list[int]] = {}
    for episode in episodes:
        if episode not in ranges:
            raise ValueError(f"episode {episode} not found in {dataset_root}")
        first_frame, last_frame = ranges[episode]
        summary = episode_summaries.get(episode, {})
        if len(frame_tokens) == 1 and frame_tokens[0].lower() == "all":
            frames = list(range(first_frame, last_frame + 1))
        else:
            frames = [_resolve_frame_token(token, first_frame=first_frame, last_frame=last_frame, summary=summary) for token in frame_tokens]
        episode_frames[int(episode)] = sorted(dict.fromkeys(frames))
    return episode_frames


def _episode_frame_ranges(dataset_root: Path) -> dict[int, tuple[int, int]]:
    import pyarrow.parquet as pq

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing LeRobot parquet data files under {dataset_root}")
    table = pq.read_table([str(path) for path in data_files], columns=["episode_index", "frame_index"])
    ranges: dict[int, tuple[int, int]] = {}
    columns = table.to_pydict()
    for episode, frame in zip(columns["episode_index"], columns["frame_index"]):
        episode = int(episode)
        frame = int(frame)
        current = ranges.get(episode)
        ranges[episode] = (frame, frame) if current is None else (min(current[0], frame), max(current[1], frame))
    return ranges


def _resolve_frame_token(token: str, *, first_frame: int, last_frame: int, summary: dict[str, Any]) -> int:
    lowered = token.lower()
    if lowered == "start":
        return first_frame
    if lowered in {"final", "last"}:
        return last_frame
    phase_counts = summary.get("phase_counts") if isinstance(summary, dict) else None
    if lowered == "open" and isinstance(phase_counts, dict):
        return min(last_frame, first_frame + int(phase_counts.get("approach", 0)) + int(phase_counts.get("settle", 0)) - 1)
    if lowered == "grip" and isinstance(phase_counts, dict):
        return min(
            last_frame,
            first_frame
            + int(phase_counts.get("approach", 0))
            + int(phase_counts.get("settle", 0))
            + int(phase_counts.get("close", 0))
            - 1,
        )
    return int(token)


def _frame_label(episode_frames: dict[int, list[int]], *, episode: int, frame: int) -> str:
    frames = episode_frames.get(int(episode), [])
    if len(frames) == 1:
        return "selected"
    if frame == frames[0]:
        return "start"
    if frame == frames[-1]:
        return "grip"
    return str(frame)


def _camera_specs_from_mujoco_scene(
    env: Any,
    renderers: dict[str, Any],
    *,
    camera_lens: float,
) -> dict[str, dict[str, Any]]:
    unwrapped = env.unwrapped
    camera1 = _scene_camera_spec(
        unwrapped.model,
        unwrapped.data,
        renderers["egocentric_cam"],
        "egocentric_cam",
    )
    camera1["rotation_degrees"] = int(EGOCENTRIC_CAMERA1_POSE["rotation_degrees"])
    return {
        "observation.images.camera1": camera1,
        "observation.images.camera2": _fixed_mujoco_camera_spec(
            unwrapped.model,
            unwrapped.data,
            "wrist_cam",
        ),
    }


def _scene_camera_spec(model: Any, data: Any, renderer: Any, camera_name: str) -> dict[str, Any]:
    renderer.update_scene(data, camera=_make_camera(_EnvView(model=model, data=data), camera_name))
    scene = getattr(renderer, "scene", None) or getattr(renderer, "_scene", None)
    if scene is None:
        raise RuntimeError("MuJoCo renderer scene is unavailable")
    camera = scene.camera[0]
    return {
        "mode": "forward_up",
        "location": [float(value) for value in camera.pos],
        "forward": [float(value) for value in camera.forward],
        "up": [float(value) for value in camera.up],
        "fovy": _camera_fovy(model, camera_name),
        "focus_distance": 0.63 if camera_name == "egocentric_cam" else 0.20,
        "aperture_fstop": 10.0,
        "use_dof": False,
        "clip_start": 0.001,
    }


def _fixed_mujoco_camera_spec(model: Any, data: Any, camera_name: str) -> dict[str, Any]:
    camera_id = _camera_id(model, camera_name)
    xmat = [float(value) for value in data.cam_xmat[camera_id]]
    rows = [xmat[0:3], xmat[3:6], xmat[6:9]]
    columns = [
        [rows[0][0], rows[1][0], rows[2][0]],
        [rows[0][1], rows[1][1], rows[2][1]],
        [rows[0][2], rows[1][2], rows[2][2]],
    ]
    forward = [-float(value) for value in columns[2]]
    up = [float(value) for value in columns[1]]
    return {
        "mode": "forward_up",
        "location": [float(value) for value in data.cam_xpos[camera_id]],
        "forward": forward,
        "up": up,
        "fovy": _camera_fovy(model, camera_name),
        "focus_distance": 0.20,
        "aperture_fstop": 10.0,
        "use_dof": False,
        "clip_start": 0.001,
    }


def _camera_fovy(model: Any, camera_name: str) -> float:
    if camera_name == "egocentric_cam":
        return float(model.vis.global_.fovy)
    for index in range(model.ncam):
        if model.camera(index).name == camera_name:
            return float(model.cam_fovy[index])
    return float(model.vis.global_.fovy)


def _camera_id(model: Any, camera_name: str) -> int:
    for index in range(model.ncam):
        if model.camera(index).name == camera_name:
            return int(index)
    raise ValueError(f"unknown MuJoCo camera: {camera_name}")


class _EnvView:
    def __init__(self, *, model: Any, data: Any) -> None:
        self.unwrapped = self
        self.model = model
        self.data = data


def _camera_slug(camera_key: str) -> str:
    return camera_key.removeprefix("observation.images.")


def _rotate_image(path: Path, degrees: int) -> None:
    if degrees % 360 == 0:
        return
    image = Image.open(path).convert("RGB")
    if degrees % 360 == 90:
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif degrees % 360 == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif degrees % 360 == 270:
        image = image.transpose(Image.Transpose.ROTATE_90)
    else:
        image = image.rotate(-degrees, expand=False)
    image.save(path)


if __name__ == "__main__":
    main()
