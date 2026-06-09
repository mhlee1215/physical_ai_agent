#!/usr/bin/env python3
"""Capture LIBERO/MuJoCo actual-sim RGB frames with simulator-state oracle overlays.

This avoids SAPIEN/Vulkan entirely. It uses LIBERO's MuJoCo/robosuite renderer,
extracts actual object/body/site world positions from the same simulator state,
projects one candidate point into the active camera, and writes raw/overlay
frames plus a strict evidence manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ProbeSample:
    suite: str
    task_id: int
    task_name: str
    episode: int
    step: int
    camera_name: str
    raw_frame_path: str
    overlay_frame_path: str
    object_name: str
    object_pose_xyz: list[float]
    object_geom_id: int | None
    mask_pixel_count: int
    camera_position_xyz: list[float]
    camera_fovy_degrees: float
    point_xy: list[int]
    image_shape: list[int]
    projection_source: str
    strict_true_oracle_ready: bool
    figure_orientation: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_benchmark(suite: str):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite not in benchmark_dict:
        raise KeyError(f"Unknown LIBERO suite {suite!r}; available={sorted(benchmark_dict)}")
    return benchmark_dict[suite]()


def _patch_robosuite_segmentation_uint8_overflow() -> None:
    """Patch robosuite segmentation decoding for NumPy uint8 overflow.

    Some robosuite releases decode MuJoCo ID-color segmentation with uint8
    arithmetic, e.g. ``rgb[:, :, 1] * 256``. Newer NumPy raises on that
    operation. Keep the original path for RGB/depth rendering and replace only
    the segmentation decode with int32 arithmetic.
    """

    try:
        import mujoco
        from robosuite.utils import binding_utils
    except Exception:
        return

    cls = getattr(binding_utils, "MjRenderContext", None)
    if cls is None or getattr(cls, "_physical_ai_segmentation_patch", False):
        return

    original_read_pixels = cls.read_pixels

    def read_pixels_safe(self: Any, width: int, height: int, depth: bool = False, segmentation: bool = False) -> Any:
        if not segmentation:
            return original_read_pixels(self, width, height, depth=depth, segmentation=segmentation)

        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

        rgb_i32 = rgb_img.astype(np.int32)
        seg_img = rgb_i32[:, :, 0] + rgb_i32[:, :, 1] * (2**8) + rgb_i32[:, :, 2] * (2**16)
        seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
        seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
        for idx in range(self.scn.ngeom):
            geom = self.scn.geoms[idx]
            if geom.segid != -1:
                seg_ids[geom.segid + 1, 0] = geom.objtype
                seg_ids[geom.segid + 1, 1] = geom.objid
        ret_img = seg_ids[seg_img]
        if depth:
            return ret_img, depth_img
        return ret_img

    cls.read_pixels = read_pixels_safe
    cls._physical_ai_segmentation_patch = True


def _make_env(
    task: Any,
    width: int,
    height: int,
    use_segmentation_env: bool = False,
    segmentation_mode: str = "instance",
):
    from pathlib import Path

    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
    from libero.libero import get_libero_path

    _patch_robosuite_segmentation_uint8_overflow()

    bddl_file = Path(str(task.bddl_file))
    if not bddl_file.is_absolute():
        bddl_root = Path(get_libero_path("bddl_files"))
        direct = bddl_root / bddl_file
        if direct.exists():
            bddl_file = direct
        else:
            matches = sorted(bddl_root.rglob(bddl_file.name))
            if not matches:
                raise FileNotFoundError(f"Could not resolve LIBERO BDDL file: {bddl_file}")
            bddl_file = matches[0]

    env_cls = SegmentationRenderEnv if use_segmentation_env else OffScreenRenderEnv
    kwargs = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": height,
        "camera_widths": width,
    }
    if use_segmentation_env:
        kwargs["camera_segmentations"] = segmentation_mode
    return env_cls(**kwargs)


def _reset_env(env: Any, bench: Any, task_id: int, episode: int, seed: int) -> Any:
    env.seed(seed + task_id * 1000 + episode)
    obs = env.reset()
    if getattr(env, "_oracle_skip_init_states", False):
        return obs
    init_states = bench.get_task_init_states(task_id)
    if len(init_states) > 0:
        obs = env.set_init_state(init_states[episode % len(init_states)])
    return obs


def _zero_action(env: Any) -> np.ndarray:
    if hasattr(env, "action_dim"):
        return np.zeros(int(env.action_dim), dtype=np.float32)
    if hasattr(env, "action_spec"):
        spec = env.action_spec
        if callable(spec):
            spec = spec()
        if isinstance(spec, tuple) and len(spec) >= 1:
            return np.zeros_like(np.asarray(spec[0], dtype=np.float32))
    return np.zeros(7, dtype=np.float32)


def _camera_image(obs: dict[str, Any], preferred: list[str]) -> tuple[str, np.ndarray]:
    for name in preferred:
        key = f"{name}_image"
        if key in obs:
            return name, _as_rgb(obs[key])
    for key, value in sorted(obs.items()):
        if key.endswith("_image"):
            return key[: -len("_image")], _as_rgb(value)
    raise RuntimeError(f"No *_image camera observation found. keys={sorted(obs)}")


def _segmentation_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    preferred_keys = (
        f"{camera_name}_segmentation_instance",
        f"{camera_name}_segmentation",
        f"{camera_name}_segmentation_class",
        f"{camera_name}_segmentation_element",
    )
    for key in preferred_keys:
        if key in obs:
            return np.asarray(obs[key])
    for key, value in sorted(obs.items()):
        if key.startswith(f"{camera_name}_") and "segmentation" in key:
            return np.asarray(value)
    return None


def _mask_centroid_from_obs_segmentation(env: Any, obs: dict[str, Any], camera_name: str) -> tuple[list[int] | None, int, str]:
    segmentation = _segmentation_from_obs(obs, camera_name)
    if segmentation is None:
        return None, 0, "obs_segmentation_missing"
    array = np.asarray(segmentation)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        return None, 0, f"obs_segmentation_bad_shape:{list(segmentation.shape)}"

    masks: list[tuple[str, np.ndarray]] = []
    if hasattr(env, "get_segmentation_of_interest"):
        try:
            interest = np.asarray(env.get_segmentation_of_interest(array.copy()))
            masks.append(("obj_of_interest", interest > 0))
        except Exception:
            pass
    values, counts = np.unique(array, return_counts=True)
    # Background/robot commonly dominate. Keep nonzero moderate-size masks as fallback.
    for value, count in sorted(zip(values, counts), key=lambda item: int(item[1])):
        if int(value) <= 0 or int(count) <= 0:
            continue
        masks.append((f"seg_id:{int(value)}", array == value))

    for label, mask in masks:
        count = int(mask.sum())
        if count <= 0:
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x = int(round(float(xs.mean())))
        y = int(round(float(ys.mean())))
        height, width = array.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            return [x, y], count, f"obs_segmentation_{label}"
    return None, 0, "obs_segmentation_no_positive_mask"


def _as_rgb(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] > 3:
        array = array[..., :3]
    return np.ascontiguousarray(array)


def _sim(env: Any) -> Any:
    current = env
    for attr in ("env", "_env"):
        if hasattr(current, attr):
            maybe = getattr(current, attr)
            if hasattr(maybe, "sim"):
                return maybe.sim
    if hasattr(env, "sim"):
        return env.sim
    raise RuntimeError("Could not find MuJoCo sim on LIBERO env")


def _names(model: Any, obj_type: str) -> list[str]:
    count_attr = {"body": "nbody", "site": "nsite", "geom": "ngeom"}[obj_type]
    id2name = getattr(model, f"{obj_type}_id2name", None)
    result = []
    for idx in range(int(getattr(model, count_attr))):
        name = id2name(idx) if callable(id2name) else None
        if name:
            result.append(str(name))
    return result


def _candidate_points(sim: Any) -> list[tuple[str, np.ndarray, int | None]]:
    model = sim.model
    data = sim.data
    deny = (
        "world",
        "robot",
        "gripper",
        "eef",
        "camera",
        "mount",
        "base",
        "link",
        "finger",
        "joint",
        "floor",
        "wall",
        "table",
        "workspace",
        "arena",
    )
    prefer = (
        "drawer",
        "handle",
        "cabinet",
        "door",
        "cube",
        "mug",
        "bowl",
        "plate",
        "book",
        "box",
        "can",
        "object",
        "obj",
        "target",
        "main",
    )
    candidates: list[tuple[int, str, np.ndarray, int | None]] = []

    geom_names = _names(model, "geom")
    for idx, name in enumerate(geom_names):
        lower = name.lower()
        if any(token in lower for token in deny):
            continue
        if hasattr(data, "geom_xpos") and idx < len(data.geom_xpos):
            score = 0 if any(token in lower for token in prefer) else 3
            candidates.append((score, f"geom:{name}", np.asarray(data.geom_xpos[idx], dtype=float), idx))

    body_names = _names(model, "body")
    for idx, name in enumerate(body_names):
        lower = name.lower()
        if any(token in lower for token in deny):
            continue
        if hasattr(data, "body_xpos") and idx < len(data.body_xpos):
            score = 1 if any(token in lower for token in prefer) else 4
            candidates.append((score, f"body:{name}", np.asarray(data.body_xpos[idx], dtype=float), None))

    site_names = _names(model, "site")
    for idx, name in enumerate(site_names):
        lower = name.lower()
        if any(token in lower for token in deny):
            continue
        if hasattr(data, "site_xpos") and idx < len(data.site_xpos):
            score = 2 if any(token in lower for token in prefer) else 5
            candidates.append((score, f"site:{name}", np.asarray(data.site_xpos[idx], dtype=float), None))

    candidates = [(s, n, p, g) for s, n, p, g in candidates if np.isfinite(p).all()]
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(name, point, geom_id) for _, name, point, geom_id in candidates]


def _camera_id(model: Any, camera_name: str) -> int:
    fn = getattr(model, "camera_name2id", None)
    if callable(fn):
        try:
            return int(fn(camera_name))
        except Exception:
            pass
    names = _names(model, "camera") if hasattr(model, "ncam") else []
    if camera_name in names:
        return names.index(camera_name)
    return 0


def _camera_pose(sim: Any, camera_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    cam_id = _camera_id(sim.model, camera_name)
    data = sim.data
    model = sim.model
    if hasattr(data, "cam_xpos"):
        pos = np.asarray(data.cam_xpos[cam_id], dtype=float)
        mat = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    elif hasattr(data, "camera_xpos"):
        pos = np.asarray(data.camera_xpos[cam_id], dtype=float)
        mat = np.asarray(data.camera_xmat[cam_id], dtype=float).reshape(3, 3)
    else:
        raise RuntimeError("MuJoCo camera pose arrays not found")
    fovy = float(np.asarray(model.cam_fovy)[cam_id]) if hasattr(model, "cam_fovy") else 45.0
    return pos, mat, fovy


def _project(point_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, fovy_deg: float, width: int, height: int) -> tuple[list[int] | None, str]:
    rel = np.asarray(point_world, dtype=float) - np.asarray(cam_pos, dtype=float)
    cam = np.asarray(cam_mat, dtype=float).T @ rel
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    candidates = []
    for z_sign, label in ((-1.0, "mujoco_neg_z_forward"), (1.0, "pos_z_forward_fallback")):
        z = z_sign * float(cam[2])
        if z <= 1e-8:
            continue
        u = int(round(width / 2.0 + f * float(cam[0]) / z))
        v = int(round(height / 2.0 - f * float(cam[1]) / z))
        candidates.append(([u, v], label))
    for point, label in candidates:
        if 0 <= point[0] < width and 0 <= point[1] < height:
            return point, label
    return None, "projection_out_of_frame"


def _segmentation_centroid(sim: Any, camera_name: str, geom_id: int, width: int, height: int) -> tuple[list[int] | None, int]:
    rendered = _render_segmentation(sim, camera_name, width, height)
    if rendered is None:
        return None, 0
    array = _segmentation_array(rendered)
    if array is None:
        return None, 0

    object_masks = _candidate_segmentation_masks(array, geom_id)
    best_mask = None
    best_count = 0
    for mask in object_masks:
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count <= 0:
        return None, 0

    ys, xs = np.nonzero(best_mask)
    if len(xs) == 0:
        return None, 0
    x = int(round(float(xs.mean())))
    y = int(round(float(ys.mean())))
    if not (0 <= x < width and 0 <= y < height):
        return None, best_count
    return [x, y], best_count


def _render_segmentation(sim: Any, camera_name: str, width: int, height: int) -> Any | None:
    try:
        return sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            segmentation=True,
        )
    except TypeError:
        try:
            return sim.render(width, height, camera_name=camera_name, segmentation=True)
        except Exception:
            return None
    except Exception:
        return None


def _segmentation_array(rendered: Any) -> np.ndarray | None:
    if isinstance(rendered, tuple):
        segmentation = rendered[-1]
    else:
        segmentation = rendered

    array = np.asarray(segmentation)
    if array.ndim != 3 or array.shape[-1] < 2:
        return None
    return array


def _candidate_segmentation_masks(array: np.ndarray, geom_id: int) -> list[np.ndarray]:
    candidate_masks = []
    for channel in range(min(array.shape[-1], 3)):
        candidate_masks.append(array[..., channel] == geom_id)
    return candidate_masks


def _task_target_tokens(task_name: str) -> list[str]:
    name = str(task_name).lower().replace("-", "_")
    patterns = (
        r"^pick_up_the_(.+?)(?:_and_|_between_|_next_to_|_on_|_in_|$)",
        r"^open_the_(.+?)(?:_of_|_and_|$)",
        r"^close_the_(.+?)(?:_of_|_and_|$)",
        r"^put_the_(.+?)(?:_in_|_on_|_and_|$)",
        r"^push_the_(.+?)(?:_to_|_in_|_on_|_and_|$)",
        r"^turn_on_the_(.+?)(?:_and_|$)",
    )
    phrase = ""
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            phrase = match.group(1)
            break
    if not phrase:
        phrase = name
    stop = {"the", "a", "an", "of", "and", "to", "from", "in", "on", "into", "place", "pick", "up"}
    tokens = [token for token in re.split(r"[^a-z0-9]+", phrase) if token and token not in stop]
    return tokens


def _semantic_candidate_order(
    candidates: list[tuple[str, np.ndarray, int | None]],
    task_name: str,
    require_match: bool = False,
) -> list[tuple[str, np.ndarray, int | None]]:
    tokens = _task_target_tokens(task_name)
    if not tokens:
        return candidates

    def score(candidate: tuple[str, np.ndarray, int | None]) -> tuple[int, int, str]:
        name = candidate[0].lower()
        matched = sum(1 for token in tokens if token in name)
        missing = len(tokens) - matched
        # Prefer concrete geoms for pixel masks, then candidates that match all
        # task-target tokens. Name is the stable tie-breaker.
        no_geom = 1 if candidate[2] is None else 0
        return missing, no_geom, name

    ordered = sorted(candidates, key=score)
    best_missing = score(ordered[0])[0] if ordered else len(tokens)
    if best_missing < len(tokens):
        if require_match:
            return [candidate for candidate in ordered if score(candidate)[0] < len(tokens)]
        return ordered
    if require_match:
        return []
    return candidates


def _mask_centroid_for_geom(array: np.ndarray, geom_id: int) -> tuple[list[int] | None, int]:
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        masks = [array == geom_id]
    elif array.ndim == 3:
        masks = _candidate_segmentation_masks(array, geom_id)
    else:
        return None, 0

    best_mask = None
    best_count = 0
    for mask in masks:
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count <= 0:
        return None, 0
    ys, xs = np.nonzero(best_mask)
    if len(xs) == 0:
        return None, best_count
    return [int(round(float(xs.mean()))), int(round(float(ys.mean())))], best_count


def _drawer_handle_candidate(
    sim: Any,
    segmentation: np.ndarray,
    task_name: str,
) -> tuple[str, np.ndarray, int, int, list[int], str] | None:
    """Select a visible cabinet drawer/handle mask for open-drawer tasks.

    LIBERO cabinet geoms are named generically (``wooden_cabinet_1_g*``), so
    task tokens like "top drawer" do not directly match geom names. The visible
    cabinet masks form height buckets. Within the target bucket, the smaller
    visible masks correspond better to handle/front affordance points than the
    large cabinet body mask.
    """

    lowered = str(task_name).lower()
    if not lowered.startswith(("open_the_", "close_the_")) or "drawer" not in lowered:
        return None

    model = sim.model
    data = sim.data
    visible: list[tuple[float, int, list[int], int, str, np.ndarray]] = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id)
        if not name or "cabinet" not in name.lower():
            continue
        point_world = np.asarray(data.geom_xpos[geom_id], dtype=float)
        point_xy, mask_count = _mask_centroid_for_geom(segmentation, geom_id)
        if point_xy is None or mask_count <= 0:
            continue
        # Skip the very large cabinet carcass/background face; keep smaller
        # parts that visually behave like drawer fronts or handles.
        if mask_count > 2500:
            continue
        visible.append((float(point_world[2]), mask_count, point_xy, geom_id, f"geom:{name}", point_world))

    if not visible:
        return None

    # Cluster visible cabinet parts by world z. Descending order corresponds to
    # top / middle / bottom drawer levels in the LIBERO cabinet asset.
    clusters: list[list[tuple[float, int, list[int], int, str, np.ndarray]]] = []
    for item in sorted(visible, key=lambda row: row[0], reverse=True):
        if not clusters or abs(clusters[-1][0][0] - item[0]) > 0.04:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    if "bottom" in lowered:
        cluster_idx = min(2, len(clusters) - 1)
    elif "middle" in lowered:
        cluster_idx = min(1, len(clusters) - 1)
    else:
        cluster_idx = 0

    target_cluster = clusters[cluster_idx]
    # Prefer the smallest visible mask in the bucket: empirically this lands on
    # the handle/front affordance rather than the larger drawer slab.
    z, mask_count, point_xy, geom_id, object_name, point_world = min(target_cluster, key=lambda row: row[1])
    return object_name, point_world, geom_id, mask_count, point_xy, f"obs_segmentation_drawer_handle:{object_name}"


def _write_segmentation_debug(
    sim: Any,
    rgb: np.ndarray,
    camera_name: str,
    candidates: list[tuple[str, np.ndarray, int | None]],
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    debug_dir = output_dir / "segmentation_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_path = debug_dir / f"{prefix}_raw.png"
    Image.fromarray(rgb).save(raw_path)
    rendered = _render_segmentation(sim, camera_name, width, height)
    array = _segmentation_array(rendered) if rendered is not None else None
    info: dict[str, Any] = {
        "camera_name": camera_name,
        "raw_frame_path": str(raw_path),
        "rendered_type": type(rendered).__name__ if rendered is not None else "none",
        "segmentation_shape": list(array.shape) if array is not None else None,
        "candidate_geom_ids": [
            {"name": name, "geom_id": geom_id}
            for name, _point, geom_id in candidates[:40]
            if geom_id is not None
        ],
        "channel_unique_values": [],
        "mask_tiles": [],
    }
    if array is None:
        return info

    tiles: list[tuple[str, Image.Image]] = [("rgb", Image.fromarray(rgb).convert("RGB"))]
    for channel in range(array.shape[-1]):
        values, counts = np.unique(array[..., channel], return_counts=True)
        order = np.argsort(counts)[::-1]
        top = [
            {"value": int(values[idx]), "count": int(counts[idx])}
            for idx in order[:20]
        ]
        info["channel_unique_values"].append({"channel": channel, "top_values": top})
        norm = array[..., channel].astype(float)
        norm_min = float(norm.min(initial=0))
        norm_max = float(norm.max(initial=0))
        if norm_max > norm_min:
            norm = (norm - norm_min) / (norm_max - norm_min) * 255.0
        channel_img = Image.fromarray(np.asarray(norm, dtype=np.uint8)).convert("RGB")
        tiles.append((f"channel_{channel}", channel_img))

    for name, _point, geom_id in candidates[:24]:
        if geom_id is None:
            continue
        best = None
        best_count = 0
        best_label = ""
        for channel in range(array.shape[-1]):
            mask = array[..., channel] == geom_id
            count = int(mask.sum())
            if count > best_count:
                best = mask
                best_count = count
                best_label = f"{name} ch{channel} id{geom_id} n{count}"
        if best is None or best_count <= 0:
            continue
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask_rgb[..., 0] = np.asarray(best, dtype=np.uint8) * 255
        overlay = Image.fromarray(rgb).convert("RGB")
        red = Image.fromarray(mask_rgb).convert("RGB")
        blended = Image.blend(overlay, red, 0.38)
        tiles.append((best_label, blended))
        info["mask_tiles"].append({"label": best_label, "count": best_count})

    cell_w, cell_h = 260, 300
    cols = 3
    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, tile_img) in enumerate(tiles):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        draw.text((x + 8, y + 8), label[:42], fill=(20, 18, 14))
        thumb = tile_img.copy()
        thumb.thumbnail((cell_w - 16, cell_h - 44))
        sheet.paste(thumb, (x + 8, y + 34))
    sheet_path = debug_dir / f"{prefix}_segmentation_contact_sheet.jpg"
    sheet.save(sheet_path)
    info["contact_sheet"] = str(sheet_path)
    return info


def _draw_overlay(rgb: np.ndarray, point_xy: list[int], label: str) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    x, y = int(point_xy[0]), int(point_xy[1])
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(255, 210, 0, 210), outline=(20, 20, 20, 255), width=3)
    draw.line((x - 24, y, x + 24, y), fill=(20, 20, 20, 220), width=2)
    draw.line((x, y - 24, x, y + 24), fill=(20, 20, 20, 220), width=2)
    draw.rectangle((8, 8, 8 + 9 * len(label), 32), fill=(255, 255, 255, 210))
    draw.text((14, 13), label, fill=(10, 10, 10, 255))
    return image


def _figure_rgb_and_point(
    rgb: np.ndarray,
    point_xy: list[int],
    flip_vertical: bool,
) -> tuple[np.ndarray, list[int], str]:
    if not flip_vertical:
        return rgb, point_xy, "native_agentview"
    height = int(rgb.shape[0])
    flipped = np.ascontiguousarray(rgb[::-1, :, :])
    return flipped, [int(point_xy[0]), int(height - 1 - int(point_xy[1]))], "vertical_flip_for_libero_figure"


def _capture_task(bench: Any, suite: str, task_id: int, args: argparse.Namespace, output_dir: Path) -> list[ProbeSample]:
    task = bench.get_task(task_id)
    task_name = getattr(task, "name", f"task_{task_id}")
    env = _make_env(
        task,
        args.width,
        args.height,
        use_segmentation_env=args.use_segmentation_env,
        segmentation_mode=args.segmentation_mode,
    )
    setattr(env, "_oracle_skip_init_states", bool(args.skip_init_states))
    samples: list[ProbeSample] = []
    try:
        for episode in range(args.episodes_per_task):
            obs = _reset_env(env, bench, task_id, episode, args.seed)
            for step in range(args.steps):
                camera_name, rgb = _camera_image(obs, args.cameras)
                sim = _sim(env)
                candidates = _candidate_points(sim)
                cam_pos, cam_mat, fovy = _camera_pose(sim, camera_name)
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
                if args.write_segmentation_debug and not getattr(args, "_segmentation_debug_written", False):
                    prefix = f"{suite}_task{task_id:02d}_ep{episode:02d}_step{step:02d}_{camera_name}"
                    args._segmentation_debug_info = _write_segmentation_debug(
                        sim=sim,
                        rgb=rgb,
                        camera_name=camera_name,
                        candidates=candidates,
                        output_dir=output_dir,
                        prefix=prefix,
                    )
                    args._segmentation_debug_written = True
                chosen = None
                if args.use_segmentation_env:
                    segmentation = _segmentation_array(_render_segmentation(sim, camera_name, width, height))
                    if segmentation is not None:
                        drawer_candidate = _drawer_handle_candidate(sim, np.asarray(segmentation), str(task_name))
                        if drawer_candidate is not None:
                            chosen = drawer_candidate
                        else:
                            for object_name, point_world, geom_id in _semantic_candidate_order(
                                candidates,
                                str(task_name),
                                require_match=True,
                            ):
                                if geom_id is None:
                                    continue
                                seg_point, mask_count = _mask_centroid_for_geom(np.asarray(segmentation), geom_id)
                                if seg_point is not None:
                                    chosen = (
                                        object_name,
                                        point_world,
                                        geom_id,
                                        mask_count,
                                        seg_point,
                                        f"obs_segmentation_semantic_geom:{object_name}",
                                    )
                                    break
                    if chosen is None:
                        seg_point, mask_count, seg_source = _mask_centroid_from_obs_segmentation(env, obs, camera_name)
                        if seg_point is not None:
                            object_name = "obj_of_interest"
                            point_world = candidates[0][1] if candidates else np.zeros(3)
                            geom_id = candidates[0][2] if candidates else None
                            chosen = (object_name, point_world, geom_id, mask_count, seg_point, seg_source)
                for object_name, point_world, geom_id in _semantic_candidate_order(candidates, str(task_name)):
                    if chosen is not None:
                        break
                    if geom_id is not None:
                        seg_point, mask_count = _segmentation_centroid(sim, camera_name, geom_id, width, height)
                        if seg_point is not None:
                            chosen = (object_name, point_world, geom_id, mask_count, seg_point, "segmentation_mask_centroid")
                            break
                    point_xy, projection_source = _project(point_world, cam_pos, cam_mat, fovy, width, height)
                    if point_xy is not None:
                        chosen = (object_name, point_world, geom_id, 0, point_xy, projection_source)
                        break
                if chosen is None:
                    object_name, point_world, geom_id = candidates[0] if candidates else ("missing_object_candidate", np.zeros(3), None)
                    point_xy = [width // 2, height // 2]
                    mask_count = 0
                    projection_source = "fallback_center_no_visible_candidate"
                else:
                    object_name, point_world, geom_id, mask_count, point_xy, projection_source = chosen
                raw_path = output_dir / "frames" / f"{suite}_task{task_id:02d}_ep{episode:02d}_step{step:02d}_{camera_name}_raw.png"
                overlay_path = output_dir / "overlays" / f"{suite}_task{task_id:02d}_ep{episode:02d}_step{step:02d}_{camera_name}_oracle.png"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                figure_rgb, figure_point_xy, figure_orientation = _figure_rgb_and_point(
                    rgb,
                    point_xy,
                    bool(args.figure_flip_vertical),
                )
                Image.fromarray(figure_rgb).save(raw_path)
                _draw_overlay(figure_rgb, figure_point_xy, object_name).save(overlay_path)
                if args.use_segmentation_env:
                    strict = projection_source.startswith("obs_segmentation_") and mask_count > 0
                else:
                    strict = projection_source not in {"fallback_center_no_visible_candidate", "projection_out_of_frame"}
                samples.append(ProbeSample(
                    suite=suite,
                    task_id=task_id,
                    task_name=str(task_name),
                    episode=episode,
                    step=step,
                    camera_name=camera_name,
                    raw_frame_path=str(raw_path),
                    overlay_frame_path=str(overlay_path),
                    object_name=object_name,
                    object_pose_xyz=[float(x) for x in point_world[:3]],
                    object_geom_id=geom_id,
                    mask_pixel_count=mask_count,
                    camera_position_xyz=[float(x) for x in cam_pos[:3]],
                    camera_fovy_degrees=float(fovy),
                    point_xy=point_xy,
                    image_shape=[height, width, 3],
                    projection_source=projection_source,
                    strict_true_oracle_ready=strict,
                    figure_orientation=figure_orientation,
                ))
                action = _zero_action(env)
                obs, _reward, _done, _info = env.step(action)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return samples


def _write_report(output_dir: Path, manifest: dict[str, Any], samples: list[ProbeSample]) -> None:
    rows = samples
    cell_w, cell_h = 760, 250
    sheet = Image.new("RGB", (cell_w, max(1, len(rows)) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(sheet)
    for idx, sample in enumerate(rows):
        y = idx * cell_h
        draw.text((16, y + 10), f"{sample.suite} task={sample.task_id} ep={sample.episode} step={sample.step} point={sample.point_xy} {sample.object_name}", fill=(20, 18, 14))
        for col, path in enumerate((sample.raw_frame_path, sample.overlay_frame_path)):
            img = Image.open(path).convert("RGB")
            img.thumbnail((340, 185))
            x = 16 + col * 370
            sheet.paste(img, (x, y + 42))
            draw.text((x, y + 230), "actual LIBERO/MuJoCo RGB" if col == 0 else "MuJoCo-state oracle overlay", fill=(20, 18, 14))
    contact = output_dir / "libero_mujoco_oracle_contact_sheet.jpg"
    sheet.save(contact)
    html_path = output_dir / "libero_mujoco_oracle_report.html"
    status = manifest["status"]
    cards = []
    for sample in rows:
        cards.append(f"""
<article><h2>{sample.suite} task {sample.task_id}, ep {sample.episode}, step {sample.step}</h2>
<p>object={sample.object_name}; pose={sample.object_pose_xyz}; camera={sample.camera_name}; point={sample.point_xy}; projection={sample.projection_source}</p>
<div class='pair'><img src='{Path(sample.raw_frame_path).relative_to(output_dir).as_posix()}'><img src='{Path(sample.overlay_frame_path).relative_to(output_dir).as_posix()}'></div></article>
""")
    html_path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LIBERO MuJoCo Oracle Evidence</title>
<style>body{{margin:0;font-family:Georgia,serif;background:#f3ead8;color:#18120b}}header,main{{padding:34px 48px}}h1{{font-size:56px;letter-spacing:-.05em;margin:0}}.pill{{display:inline-block;border:1px solid #743;padding:7px 12px;border-radius:999px;background:white}}article{{background:#fffaf0;border:1px solid #d8c09b;border-radius:22px;padding:18px;margin:18px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%;border-radius:14px}}</style></head><body>
<header><h1>LIBERO/MuJoCo Oracle Overlay Evidence</h1><p class='pill'>{status}</p><p>Actual LIBERO/MuJoCo RGB plus same-timestep MuJoCo body/site pose projected into the camera. This is the SAPIEN-free evidence path.</p><img src='{contact.name}'></header><main>{''.join(cards)}</main></body></html>""", encoding="utf-8")
    manifest["html"] = str(html_path)
    manifest["contact_sheet"] = str(contact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="libero_spatial")
    parser.add_argument("--task-ids", default="0,1,2")
    parser.add_argument(
        "--suite-task-ids",
        default="",
        help="Optional semicolon-separated suite-specific task IDs, e.g. 'libero_spatial:0,3;libero_object:0,6'.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--min-strict-samples", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cameras", default="agentview,robot0_eye_in_hand")
    parser.add_argument("--write-segmentation-debug", action="store_true")
    parser.add_argument("--use-segmentation-env", action="store_true")
    parser.add_argument("--segmentation-mode", choices=("instance", "element", "class"), default="instance")
    parser.add_argument("--skip-init-states", action="store_true")
    parser.add_argument("--no-figure-flip-vertical", dest="figure_flip_vertical", action="store_false")
    parser.set_defaults(figure_flip_vertical=True)
    args = parser.parse_args()
    args.cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")

    all_samples: list[ProbeSample] = []
    errors: list[dict[str, Any]] = []
    default_task_ids = [int(item.strip()) for item in args.task_ids.split(",") if item.strip()]
    suite_task_ids: dict[str, list[int]] = {}
    if args.suite_task_ids.strip():
        for chunk in args.suite_task_ids.split(";"):
            if not chunk.strip():
                continue
            suite_name, ids = chunk.split(":", 1)
            suite_task_ids[suite_name.strip()] = [int(item.strip()) for item in ids.split(",") if item.strip()]
    for suite in [item.strip() for item in args.suites.split(",") if item.strip()]:
        try:
            bench = _load_benchmark(suite)
        except Exception as exc:
            errors.append({"suite": suite, "error": repr(exc), "traceback": traceback.format_exc()})
            continue
        for task_id in suite_task_ids.get(suite, default_task_ids):
            try:
                all_samples.extend(_capture_task(bench, suite, task_id, args, output_dir))
            except Exception as exc:
                errors.append({"suite": suite, "task_id": task_id, "error": repr(exc), "traceback": traceback.format_exc()})

    strict_count = sum(1 for sample in all_samples if sample.strict_true_oracle_ready)
    status = "passed" if strict_count >= args.min_strict_samples else "blocked"
    manifest = {
        "status": status,
        "source_type": "actual_libero_mujoco_true_oracle_projection",
        "real_sim_episode": True,
        "true_oracle_projection": status == "passed",
        "strict_true_oracle_step_count": strict_count,
        "sample_count": len(all_samples),
        "min_strict_samples": args.min_strict_samples,
        "suites": args.suites,
        "task_ids": default_task_ids,
        "suite_task_ids": suite_task_ids,
        "episodes_per_task": args.episodes_per_task,
        "steps": args.steps,
        "errors": errors,
        "samples": [asdict(sample) for sample in all_samples],
    }
    if getattr(args, "_segmentation_debug_info", None):
        manifest["segmentation_debug"] = args._segmentation_debug_info
    _write_report(output_dir, manifest, all_samples)
    manifest_path = output_dir / "libero_mujoco_oracle_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "strict_true_oracle_step_count": strict_count, "sample_count": len(all_samples), "manifest": str(manifest_path), "html": manifest.get("html")}, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
