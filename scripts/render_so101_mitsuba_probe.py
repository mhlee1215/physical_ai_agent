#!/usr/bin/env python3
"""Render the current SO101 MuJoCo state through Mitsuba.

This is an offline visual probe: MuJoCo still owns physics/state, while Mitsuba
only path-traces exported world-space mesh geometry.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SO101 MuJoCo state to Mitsuba and render it.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_mitsuba_probe"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--spp", type=int, default=96)
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    parser.add_argument(
        "--mujoco-reference",
        type=Path,
        default=Path("_workspace/so101_realistic_render_probe_v2/enhanced_scene.png"),
        help="Optional MuJoCo image to place next to the Mitsuba render.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render_mitsuba_probe(
        env_id=args.env_id,
        output_dir=args.output_dir,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_mesh_geoms=args.max_mesh_geoms,
        mujoco_reference=args.mujoco_reference,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_mitsuba_probe(
    *,
    env_id: str,
    output_dir: Path,
    seed: int,
    warmup_steps: int,
    width: int,
    height: int,
    spp: int,
    max_mesh_geoms: int,
    mujoco_reference: Path | None,
) -> dict[str, Any]:
    import gymnasium as gym
    import mitsuba as mi
    import mujoco
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    mi.set_variant("scalar_rgb")

    mesh_dir = output_dir / "ply"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_path = _write_tabletop_texture(output_dir / "tabletop_texture.png")

    env = gym.make(env_id, render_mode=None)
    try:
        env.reset(seed=seed)
        for step in range(warmup_steps):
            env.step(sample_action(env.action_space, step / max(1, warmup_steps - 1)))
        model, data = env.unwrapped.model, env.unwrapped.data
        exported = _export_mesh_geoms(model, data, mesh_dir, max_mesh_geoms=max_mesh_geoms)
        target = _target_site(model, data)
    finally:
        env.close()

    scene_dict = _mitsuba_scene(
        exported,
        target=target,
        texture_path=texture_path,
        width=width,
        height=height,
        spp=spp,
    )
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)

    image_path = output_dir / "so101_mitsuba_render.png"
    exr_path = output_dir / "so101_mitsuba_render.exr"
    mi.Bitmap(image).write(str(exr_path))
    mi.util.write_bitmap(str(image_path), image)
    comparison_path = _write_comparison(mujoco_reference, image_path, output_dir / "mujoco_vs_mitsuba.png")

    report = {
        "env_id": env_id,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "width": width,
        "height": height,
        "spp": spp,
        "renderer": "mitsuba",
        "variant": "scalar_rgb",
        "image_path": str(image_path),
        "exr_path": str(exr_path),
        "mesh_dir": str(mesh_dir),
        "texture_path": str(texture_path),
        "comparison_path": str(comparison_path) if comparison_path else None,
        "mesh_geoms_exported": len(exported),
        "mesh_format": "binary_little_endian_ply",
        "target_site": target,
        "material_profile": "roughplastic robot materials, bitmap tabletop texture, softbox area lights",
        "note": "MuJoCo physics/state exported to world-space PLY; Mitsuba path-traces the static frame.",
    }
    (output_dir / "mitsuba_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _export_mesh_geoms(model: Any, data: Any, mesh_dir: Path, *, max_mesh_geoms: int) -> list[dict[str, Any]]:
    import mujoco

    exported: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        if len(exported) >= max_mesh_geoms:
            break
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue

        vertices, faces = _world_mesh(model, data, geom_id, mesh_id)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        if not np.isfinite(vertices).all():
            continue

        name = model.geom(geom_id).name or f"geom_{geom_id:03d}"
        path = mesh_dir / f"{geom_id:03d}_{_safe_name(name)}.ply"
        _write_ply(path, vertices, faces)
        rgba = _geom_rgba(model, geom_id)
        exported.append(
            {
                "geom_id": geom_id,
                "mesh_id": mesh_id,
                "name": name,
                "path": str(path),
                "rgba": rgba,
            }
        )
    return exported


def _world_mesh(model: Any, data: Any, geom_id: int, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    local_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32)

    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    if not np.isfinite(local_vertices).all() or not np.isfinite(xmat).all() or not np.isfinite(xpos).all():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        world_vertices = local_vertices @ xmat.T + xpos
    return world_vertices, faces


def _geom_rgba(model: Any, geom_id: int) -> list[float]:
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return [float(value) for value in model.mat_rgba[mat_id]]
    return [float(value) for value in model.geom_rgba[geom_id]]


def _target_site(model: Any, data: Any) -> dict[str, Any] | None:
    for site_id in range(model.nsite):
        if model.site(site_id).name == "reach_target":
            return {
                "position": [float(value) for value in data.site_xpos[site_id]],
                "radius": float(model.site_size[site_id][0]),
                "rgba": [float(value) for value in model.site_rgba[site_id]],
            }
    return None


def _mitsuba_scene(
    exported: list[dict[str, Any]],
    *,
    target: dict[str, Any] | None,
    texture_path: Path,
    width: int,
    height: int,
    spp: int,
) -> dict[str, Any]:
    import mitsuba as mi

    transform = mi.ScalarTransform4f
    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": transform.look_at(
                origin=[0.45, -0.48, 0.34],
                target=[0.12, 0.02, 0.08],
                up=[0.0, 0.0, 1.0],
            ),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "rfilter": {"type": "tent"},
            },
        },
        "env": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.035, 0.038, 0.045]},
        },
        "floor": {
            "type": "rectangle",
            "to_world": transform.translate([0.0, 0.0, -0.002]) @ transform.scale([1.25, 1.25, 1.0]),
            "bsdf": {
                "type": "roughplastic",
                "diffuse_reflectance": {
                    "type": "bitmap",
                    "filename": str(texture_path),
                    "raw": True,
                    "filter_type": "bilinear",
                    "wrap_mode": "repeat",
                },
                "alpha": 0.55,
            },
        },
        "softbox": {
            "type": "rectangle",
            "to_world": transform.translate([0.1, -0.35, 0.95])
            @ transform.rotate([1.0, 0.0, 0.0], 180)
            @ transform.scale([0.45, 0.32, 1.0]),
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [7.0, 6.5, 5.6]},
            },
        },
        "fill": {
            "type": "rectangle",
            "to_world": transform.translate([-0.45, 0.42, 0.55])
            @ transform.rotate([1.0, 0.0, 0.0], 180)
            @ transform.scale([0.30, 0.30, 1.0]),
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [1.4, 1.6, 2.0]},
            },
        },
    }

    for index, item in enumerate(exported):
        color, material = _material(item["rgba"])
        scene[f"mesh_{index:03d}"] = {
            "type": "ply",
            "filename": item["path"],
            "face_normals": False,
            "bsdf": _roughplastic(color, alpha=material["alpha"], specular=material["specular"]),
        }

    if target is not None:
        scene["reach_target"] = {
            "type": "sphere",
            "center": target["position"],
            "radius": max(float(target["radius"]), 0.018),
            "bsdf": {
                "type": "roughplastic",
                "diffuse_reflectance": {"type": "rgb", "value": [1.0, 0.46, 0.04]},
                "alpha": 0.32,
            },
        }
    return scene


def _material_color(rgba: list[float]) -> list[float]:
    rgb = [float(value) for value in rgba[:3]]
    if rgb[0] > 0.8 and rgb[1] > 0.6 and rgb[2] < 0.25:
        return [0.95, 0.68, 0.16]
    if max(rgb) < 0.2:
        return [0.025, 0.026, 0.026]
    return rgb


def _material(rgba: list[float]) -> tuple[list[float], dict[str, float]]:
    color = _material_color(rgba)
    if color[0] > 0.7 and color[1] > 0.45:
        return color, {"alpha": 0.42, "specular": 0.42}
    if max(color) < 0.1:
        return color, {"alpha": 0.34, "specular": 0.18}
    return color, {"alpha": 0.55, "specular": 0.24}


def _roughplastic(rgb: list[float], *, alpha: float, specular: float) -> dict[str, Any]:
    return {
        "type": "roughplastic",
        "diffuse_reflectance": {"type": "rgb", "value": rgb},
        "specular_reflectance": {"type": "rgb", "value": [specular, specular, specular]},
        "alpha": alpha,
    }


def _diffuse(rgb: list[float]) -> dict[str, Any]:
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": rgb}}


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices32 = np.asarray(vertices, dtype="<f4")
    faces32 = np.asarray(faces, dtype="<i4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices32)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces32)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        file.write(vertices32.tobytes())
        for face in faces32:
            file.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def _write_tabletop_texture(path: Path, size: int = 1024) -> Path:
    if path.exists():
        return path
    image = Image.new("RGB", (size, size), (154, 148, 136))
    pixels = image.load()
    rng = np.random.default_rng(1215)
    noise = rng.normal(0.0, 6.0, (size, size))
    for y in range(size):
        grain = 10.0 * np.sin(y / 23.0) + 5.0 * np.sin(y / 71.0)
        for x in range(size):
            checker = 8 if ((x // 128) + (y // 128)) % 2 == 0 else -5
            value = int(np.clip(150 + grain + checker + noise[y, x], 95, 205))
            pixels[x, y] = (value, int(value * 0.94), int(value * 0.82))
    draw = ImageDraw.Draw(image, "RGBA")
    for line in range(0, size, 128):
        draw.line((line, 0, line, size), fill=(80, 70, 55, 30), width=2)
        draw.line((0, line, size, line), fill=(80, 70, 55, 24), width=2)
    image.save(path)
    return path


def _write_comparison(reference: Path | None, mitsuba_path: Path, output_path: Path) -> Path | None:
    if reference is None or not reference.exists():
        return None
    pairs = [("MuJoCo enhanced", reference), ("Mitsuba path-traced", mitsuba_path)]
    cell_w, cell_h, label_h = 640, 480, 38
    sheet = Image.new("RGB", (cell_w * 2, cell_h + label_h), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(pairs):
        image = _open_image_with_retry(path)
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = index * cell_w + (cell_w - image.width) // 2
        y = label_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * cell_w + 14, 12), label, fill=(25, 25, 25))
    sheet.save(output_path)
    return output_path


def _open_image_with_retry(path: Path, attempts: int = 5) -> Image.Image:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)


def _safe_name(name: str) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return text[:80] or "mesh"


if __name__ == "__main__":
    main()
