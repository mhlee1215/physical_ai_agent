#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.sim.so101_camera_input import _make_camera
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info
from train_so101_visual_picklift_bc import _compose_picklift_frame
from train_so101_visual_picklift_delta import _object_set_description, make_diverse_picklift_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SO101 PickLift using object position estimated from the input image."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/picklift_image_policy"))
    parser.add_argument("--calibration-samples", type=int, default=160)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20400)
    parser.add_argument("--render-seed", type=int, default=20400)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_image_policy(
        output_dir=args.output_dir,
        calibration_samples=args.calibration_samples,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        render_seed=args.render_seed,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_image_policy(
    *,
    output_dir: Path,
    calibration_samples: int,
    episodes: int,
    steps: int,
    seed: int,
    render_seed: int,
    fps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    config = SO101VisualRLConfig(
        env_id="MuJoCoPickLift-v1",
        camera_name="top_down",
        width=128,
        height=128,
        include_state=False,
        channel_first=True,
    )
    calibration = calibrate_image_to_object_pose(
        config=config,
        samples=calibration_samples,
        seed=seed - 5_000,
    )
    episodes_report = []
    for episode in range(episodes):
        result = run_image_policy_episode(
            config=config,
            calibration=calibration,
            seed=seed + episode,
            steps=steps,
        )
        episodes_report.append({"episode": episode, "seed": seed + episode, **result})
    rollout = render_image_policy_rollout(
        config=config,
        calibration=calibration,
        seed=render_seed,
        steps=steps,
        fps=fps,
        output_dir=videos_dir,
    )
    success_plot = plots_dir / "image_policy_success_curve.png"
    _write_success_plot(episodes_report, success_plot)
    success_steps = [
        item["steps_to_success"] for item in episodes_report if item["steps_to_success"] is not None
    ]
    manifest = {
        "operation": "evaluate_so101_picklift_image_policy",
        "runtime_input": "top_down RGB image only for object localization; robot state is used only for IK/control geometry",
        "object_set": _object_set_description(),
        "calibration": calibration,
        "episodes": episodes_report,
        "success_rate": float(np.mean([item["success"] for item in episodes_report])),
        "mean_final_lift_height": float(np.mean([item["final_lift_height"] for item in episodes_report])),
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "artifacts": {
            "success_plot": str(success_plot),
            "rollout_gif": rollout["gif_path"],
            "rollout_mp4": rollout["mp4_path"],
            "rollout_manifest": rollout["manifest_path"],
        },
    }
    manifest_path = output_dir / "image_policy_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def calibrate_image_to_object_pose(
    *,
    config: SO101VisualRLConfig,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import mujoco

    env = make_diverse_picklift_env(config)
    renderer = mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width)
    pixel_rows = []
    xy_rows = []
    area_rows = []
    z_rows = []
    failures = 0
    try:
        for index in range(samples):
            env.reset(seed=seed + index)
            image = render_policy_image(env, renderer, config.camera_name)
            detection = detect_colored_object(image)
            if detection is None:
                failures += 1
                continue
            obj = env.unwrapped._get_target_pose()[:3].copy()
            u, v = detection["centroid"]
            pixel_rows.append([u, v, 1.0])
            xy_rows.append([float(obj[0]), float(obj[1])])
            area_rows.append([float(detection["area"]), 1.0])
            z_rows.append(float(obj[2]))
    finally:
        renderer.close()
        env.close()
    if len(pixel_rows) < 8:
        raise RuntimeError(f"Not enough image detections for calibration: {len(pixel_rows)}")
    pixel_matrix = np.asarray(pixel_rows, dtype=float)
    xy_matrix = np.asarray(xy_rows, dtype=float)
    affine, *_ = np.linalg.lstsq(pixel_matrix, xy_matrix, rcond=None)
    z_fit, *_ = np.linalg.lstsq(np.asarray(area_rows, dtype=float), np.asarray(z_rows, dtype=float), rcond=None)
    pred_xy = pixel_matrix @ affine
    pred_z = np.asarray(area_rows, dtype=float) @ z_fit
    xy_rmse = float(np.sqrt(np.mean(np.sum((pred_xy - xy_matrix) ** 2, axis=1))))
    z_rmse = float(np.sqrt(np.mean((pred_z - np.asarray(z_rows, dtype=float)) ** 2)))
    return {
        "samples_requested": samples,
        "samples_used": len(pixel_rows),
        "detection_failures": failures,
        "pixel_to_xy_affine": affine.tolist(),
        "area_to_z_linear": z_fit.tolist(),
        "xy_rmse_m": xy_rmse,
        "z_rmse_m": z_rmse,
    }


def run_image_policy_episode(
    *,
    config: SO101VisualRLConfig,
    calibration: dict[str, Any],
    seed: int,
    steps: int,
) -> dict[str, Any]:
    import mujoco

    env = make_diverse_picklift_env(config)
    renderer = mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width)
    try:
        env.reset(seed=seed)
        image = render_policy_image(env, renderer, config.camera_name)
        estimated_obj = estimate_object_pose_from_image(image, calibration)
        initial_true_obj = env.unwrapped._get_target_pose()[:3].copy()
        q_open = solve_pregrasp_qpos_for_object(env, estimated_obj)
        q_open[-1] = 0.25
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        success_step = None
        info: dict[str, Any] = {}
        for step in range(steps):
            action = image_policy_action(env, step=step, q_open=q_open, q_close=q_close)
            _obs, _reward, terminated, truncated, info = env.step(action)
            if bool(info.get("success", False)) and success_step is None:
                success_step = step + 1
            if terminated or truncated:
                break
        return {
            "success": bool(info.get("success", False)),
            "steps_to_success": success_step,
            "final_is_grasped": float(info.get("is_grasped", 0.0)),
            "final_lift_height": float(info.get("lift_height", 0.0)),
            "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
            "estimated_object_xyz": [float(value) for value in estimated_obj],
            "initial_object_xyz_error_m": float(np.linalg.norm(estimated_obj - initial_true_obj)),
        }
    finally:
        renderer.close()
        env.close()


def render_image_policy_rollout(
    *,
    config: SO101VisualRLConfig,
    calibration: dict[str, Any],
    seed: int,
    steps: int,
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_diverse_picklift_env(config)
    renderers = {
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "top_down": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "policy": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
    }
    frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        policy_image = render_policy_image(env, renderers["policy"], config.camera_name)
        estimated_obj = estimate_object_pose_from_image(policy_image, calibration)
        q_open = solve_pregrasp_qpos_for_object(env, estimated_obj)
        q_open[-1] = 0.25
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        info = env.unwrapped._get_info()
        for step in range(steps):
            action = image_policy_action(env, step=step, q_open=q_open, q_close=q_close)
            views = {}
            for name in ("scene_3d", "wrist_cam", "egocentric_cam", "top_down"):
                renderer = renderers[name]
                if name == "scene_3d":
                    renderer.update_scene(env.unwrapped.data)
                else:
                    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
                views[name] = renderer.render()
            composed = _compose_picklift_frame(
                views=views,
                step=step,
                action=action,
                info=_json_safe_info(info),
                panel_width=480,
                panel_height=360,
            )
            canvas = Image.fromarray(composed).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (16, 360 * 2 + 74),
                "image estimate xyz "
                + ", ".join(f"{float(value):+.3f}" for value in estimated_obj),
                fill=(25, 25, 25),
            )
            frames.append(np.asarray(canvas))
            _obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "success": bool(info.get("success", False)),
                    "is_grasped": float(info.get("is_grasped", 0.0)),
                    "lift_height": float(info.get("lift_height", 0.0)),
                    "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
                }
            )
            if terminated or truncated:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    gif_path = output_dir / "picklift_image_policy_multiview.gif"
    mp4_path = output_dir / "picklift_image_policy_multiview.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_so101_picklift_image_policy_rollout",
        "seed": seed,
        "steps": len(records),
        "success": any(record["success"] for record in records),
        "final_is_grasped": records[-1]["is_grasped"] if records else 0.0,
        "final_lift_height": records[-1]["lift_height"] if records else 0.0,
        "estimated_object_xyz": [float(value) for value in estimated_obj],
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
        "records": records,
    }
    manifest_path = output_dir / "picklift_image_policy_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def render_policy_image(env: Any, renderer: Any, camera_name: str) -> np.ndarray:
    renderer.update_scene(env.unwrapped.data, camera=make_fixed_policy_camera())
    return renderer.render()


def make_fixed_policy_camera() -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.35, 0.0, 0.02]
    camera.distance = 0.85
    camera.azimuth = 90
    camera.elevation = -90
    return camera


def detect_colored_object(image: np.ndarray) -> dict[str, Any] | None:
    from scipy import ndimage

    rgb = np.asarray(image, dtype=np.float32)
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = maxc - minc
    mask = (saturation > 45.0) & (maxc > 70.0)
    labels, count = ndimage.label(mask)
    if count <= 0:
        return None
    height, width = mask.shape
    candidates = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        area = int(len(xs))
        if area < 6:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        touches_border = x_min <= 1 or y_min <= 1 or x_max >= width - 2 or y_max >= height - 2
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        too_large = bbox_w > width * 0.22 or bbox_h > height * 0.22
        if touches_border or too_large:
            continue
        candidates.append((area, xs, ys, (x_min, y_min, x_max, y_max)))
    if not candidates:
        return None
    _area, xs, ys, bbox = max(candidates, key=lambda item: item[0])
    return {
        "centroid": [float(xs.mean()), float(ys.mean())],
        "area": int(len(xs)),
        "bbox": [int(value) for value in bbox],
    }


def estimate_object_pose_from_image(image: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    detection = detect_colored_object(image)
    if detection is None:
        raise RuntimeError("No colored object detected in policy image")
    u, v = detection["centroid"]
    affine = np.asarray(calibration["pixel_to_xy_affine"], dtype=float)
    z_fit = np.asarray(calibration["area_to_z_linear"], dtype=float)
    xy = np.asarray([u, v, 1.0], dtype=float) @ affine
    z = float(np.asarray([float(detection["area"]), 1.0], dtype=float) @ z_fit)
    return np.asarray([xy[0], xy[1], z], dtype=float)


def solve_pregrasp_qpos_for_object(env: Any, obj_pos: np.ndarray) -> np.ndarray:
    import math
    import mujoco
    from scipy.optimize import least_squares

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    joint_addrs = [model.jnt_qposadr[jid] for jid in unwrapped._joint_ids]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    q_seed = np.asarray([data.qpos[addr] for addr in joint_addrs], dtype=float)
    axis = np.asarray([1.0, -1.0, 0.0], dtype=float) / math.sqrt(2.0)
    open_value = 0.25
    gap = 0.034
    desired_static = obj_pos - axis * gap * 0.5 + np.asarray([0.0, 0.0, 0.004])
    desired_moving = obj_pos + axis * gap * 0.5 + np.asarray([0.0, 0.0, 0.004])

    def set_qpos(qpos: np.ndarray) -> None:
        for addr, value in zip(joint_addrs, qpos):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
        set_qpos(qpos)
        static_pos = data.geom_xpos[static_pad]
        moving_pos = data.geom_xpos[moving_pad]
        return np.concatenate(
            [
                (static_pos - desired_static) * 20.0,
                (moving_pos - desired_moving) * 20.0,
                np.maximum(0.0, 0.004 - static_pos[2:3]) * 20.0,
                np.maximum(0.0, 0.004 - moving_pos[2:3]) * 20.0,
                (arm_qpos - q_seed[:5]) * 0.05,
            ]
        )

    starts = [
        q_seed[:5],
        np.asarray([-0.5, 0.4, 0.1, 0.5, -1.3]),
        np.asarray([0.0, 0.55, -0.25, 0.85, 1.2]),
        np.asarray([0.6, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([-0.8, 0.2, 0.2, 0.6, -1.0]),
    ]
    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, low[:5], high[:5]),
            bounds=(low[:5], high[:5]),
            max_nfev=180,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate([result.x, np.asarray([open_value])])
        if best is None or cost < best[0]:
            best = (cost, candidate)
    assert best is not None
    return np.clip(best[1], low, high)


def image_policy_action(env: Any, *, step: int, q_open: np.ndarray, q_close: np.ndarray) -> np.ndarray:
    if step < 58:
        return q_open.copy()
    if step < 118:
        return q_close.copy()
    action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
    action[-1] = q_close[-1]
    return np.clip(action, env.action_space.low, env.action_space.high)


def _write_success_plot(episodes: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    xs = np.arange(len(episodes))
    success = [1.0 if item["success"] else 0.0 for item in episodes]
    lift = [float(item["final_lift_height"]) for item in episodes]
    err = [float(item["initial_object_xyz_error_m"]) for item in episodes]
    plt.figure(figsize=(9, 4))
    plt.bar(xs - 0.25, success, width=0.25, label="success")
    plt.bar(xs, lift, width=0.25, label="final lift height (m)")
    plt.bar(xs + 0.25, err, width=0.25, label="initial xyz error (m)")
    plt.xlabel("eval episode")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
