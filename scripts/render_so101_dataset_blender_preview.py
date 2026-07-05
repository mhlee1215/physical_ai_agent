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

from render_so101_blender_probe import BLENDER_DRIVER, _pbr_paths, _write_photo_tabletop_texture
from render_so101_mitsuba_probe import _export_mesh_geoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SO101 dataset row previews with Blender Cycles Metal.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_dataset_blender_preview"))
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--frames",
        default="0,22,44,66,87",
        help="Comma-separated frame indices from the selected episode.",
    )
    parser.add_argument("--seed-base", type=int, help="Reset seed base. Defaults to seedNNN parsed from dataset root.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--robot-material", choices=("plastic", "matte_pla", "metal"), default="matte_pla")
    parser.add_argument("--asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    args = parser.parse_args()

    blender_bin = shutil.which(args.blender_bin) or args.blender_bin
    frames = [int(item.strip()) for item in args.frames.split(",") if item.strip()]
    result = render_dataset_preview(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        env_id=args.env_id,
        episode=args.episode,
        frames=frames,
        seed_base=args.seed_base,
        width=args.width,
        height=args.height,
        samples=args.samples,
        denoise=args.denoise,
        robot_material=args.robot_material,
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
    episode: int,
    frames: list[int],
    seed_base: int | None,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    robot_material: str,
    asset_root: Path,
    blender_bin: str,
    max_mesh_geoms: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import mujoco
    import pyarrow.parquet as pq
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _dataset_rows(dataset_root, episode=episode, frames=frames)
    if len(rows) != len(frames):
        found = sorted(int(row["frame_index"]) for row in rows)
        raise ValueError(f"missing requested frames: requested={frames}, found={found}")

    seed = (seed_base if seed_base is not None else _seed_from_name(dataset_root)) + episode
    rendered: list[dict[str, Any]] = []
    driver_path = output_dir / "blender_driver.py"
    driver_path.write_text(BLENDER_DRIVER, encoding="utf-8")
    texture_path = _write_photo_tabletop_texture(output_dir / "tabletop_texture.png")
    hdri_path = asset_root / "polyhaven" / "studio_small_08_2k.hdr"
    table_pbr_dir = asset_root / "ambientcg" / "Wood008_1K-JPG"
    plastic_pbr_dir = asset_root / "ambientcg" / "Plastic013A_1K-JPG"

    env = gym.make(env_id, render_mode=None)
    try:
        for row in rows:
            env.reset(seed=seed)
            unwrapped = env.unwrapped
            state = [float(value) for value in row["observation.state"]]
            unwrapped.data.qpos[: len(state)] = state
            actuator_ids = getattr(unwrapped, "_actuator_ids", None)
            if actuator_ids is not None:
                unwrapped.data.ctrl[actuator_ids] = state
            mujoco.mj_forward(unwrapped.model, unwrapped.data)

            frame_index = int(row["frame_index"])
            frame_dir = output_dir / f"episode_{episode:04d}_frame_{frame_index:04d}"
            mesh_dir = frame_dir / "ply"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            exported = _export_mesh_geoms(
                unwrapped.model,
                unwrapped.data,
                mesh_dir,
                max_mesh_geoms=max_mesh_geoms,
            )
            primitives = _export_primitive_geoms(unwrapped.model, unwrapped.data)
            image_path = frame_dir / "so101_dataset_blender_cycles_metal.png"
            blender_report_path = frame_dir / "blender_device_report.json"
            spec_path = frame_dir / "blender_scene_spec.json"
            spec = {
                "width": width,
                "height": height,
                "samples": samples,
                "denoise": denoise,
                "robot_material": robot_material,
                "texture_path": str(texture_path.resolve()),
                "hdri_path": str(hdri_path.resolve()) if hdri_path.exists() else None,
                "table_pbr": _pbr_paths(table_pbr_dir, "Wood008_1K-JPG"),
                "plastic_pbr": _pbr_paths(plastic_pbr_dir, "Plastic013A_1K-JPG"),
                "image_path": str(image_path.resolve()),
                "blender_report_path": str(blender_report_path.resolve()),
                "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
                "primitives": primitives,
                "target_site": None,
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
            rendered.append(
                {
                    "episode": episode,
                    "frame": frame_index,
                    "timestamp": float(row["timestamp"]),
                    "state": state,
                    "action": [float(value) for value in row["action"]],
                    "image_path": str(image_path),
                    "mesh_geoms_exported": len(exported),
                    "primitive_geoms_exported": len(primitives),
                    "render_seconds": render_seconds,
                }
            )
    finally:
        env.close()

    contact_sheet = _write_contact_sheet(rendered, output_dir / "so101_dataset_photoreal_5frame_contact_sheet.png")
    report = {
        "dataset_root": str(dataset_root),
        "env_id": env_id,
        "episode": episode,
        "frames": frames,
        "seed": seed,
        "renderer": "blender_cycles",
        "robot_material": robot_material,
        "samples": samples,
        "denoise": denoise,
        "width": width,
        "height": height,
        "contact_sheet": str(contact_sheet),
        "renders": rendered,
        "note": (
            "Robot qpos/action are read from the LeRobot dataset rows. The cube/object "
            "pose comes from resetting the MuJoCoPickLift env with the dataset seed."
        ),
    }
    (output_dir / "so101_dataset_blender_preview_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _dataset_rows(dataset_root: Path, *, episode: int, frames: list[int]) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing LeRobot parquet data files under {dataset_root}")
    table = pq.read_table(
        [str(path) for path in data_files],
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    frame_set = pa.array(frames, type=pa.int64())
    mask = pc.and_(
        pc.equal(table["episode_index"], episode),
        pc.is_in(table["frame_index"], value_set=frame_set),
    )
    filtered = table.filter(mask)
    rows = _rows(filtered.to_pydict())
    return sorted(rows, key=lambda row: int(row["frame_index"]))


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
        semantic_color = "green_cube" if "cube" in name or "pick_slot" in name else None
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


def _write_contact_sheet(rendered: list[dict[str, Any]], output_path: Path) -> Path:
    images = [Image.open(item["image_path"]).convert("RGB") for item in rendered]
    cell_w, cell_h, label_h = 320, 240, 38
    sheet = Image.new("RGB", (cell_w * len(images), cell_h + label_h), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, (image, item) in enumerate(zip(images, rendered, strict=True)):
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = index * cell_w + (cell_w - image.width) // 2
        y = label_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * cell_w + 10, 12), f"ep {item['episode']} | frame {item['frame']}", fill=(25, 25, 25))
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


if __name__ == "__main__":
    main()
