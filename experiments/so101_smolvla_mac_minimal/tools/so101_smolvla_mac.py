#!/usr/bin/env python3
"""Compact SO-101 + SmolVLA Mac runner.

This file is intentionally standalone. It does not import physical_ai_agent
modules or any existing real SO100/SO101 scripts from the parent repo.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
CONFIRM_TEXT = "I understand this can move the SO-101"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg


def output_dir(cfg: dict[str, Any], stem: str) -> Path:
    root = Path(cfg["runtime"].get("output_dir", "runs"))
    if not root.is_absolute():
        root = ROOT / root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"{ts}_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def import_status(module: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(module)
        return {"ok": True, "path": getattr(mod, "__file__", None)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = {
        "created": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(args.config),
        "robot_port": cfg["robot"]["port"],
        "robot_port_exists": Path(cfg["robot"]["port"]).exists(),
        "imports": {
            name: import_status(name)
            for name in [
                "cv2",
                "torch",
                "lerobot",
                "lerobot.policies.factory",
                "lerobot.robots.so_follower",
            ]
        },
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # noqa: BLE001
        report["torch"] = {"error": repr(exc)}

    out = output_dir(cfg, "doctor")
    (out / "doctor.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {out / 'doctor.json'}")
    return 0 if all(v["ok"] for v in report["imports"].values()) else 2


def cmd_snap_cameras(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    import cv2

    out = output_dir(cfg, "camera_snap")
    width = int(cfg["camera"]["width"])
    height = int(cfg["camera"]["height"])
    fps = int(cfg["camera"]["fps"])
    indexes = args.indexes if args.indexes else default_camera_indexes(cfg)
    results: list[dict[str, Any]] = []

    for idx in indexes:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        path = out / f"camera_{idx}.jpg"
        if ok and frame is not None:
            cv2.imwrite(str(path), frame)
        cap.release()
        results.append({"index": idx, "ok": bool(ok), "image": str(path) if ok else None})

    manifest = {"created": datetime.now().isoformat(), "results": results}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"output: {out}")
    return 0 if any(r["ok"] for r in results) else 2


def load_lerobot_policy(cfg: dict[str, Any], ds_meta: Any | None = None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_path = cfg["policy"]["path"]
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = cfg["policy"].get("device", "mps")
    if hasattr(policy_cfg, "num_steps"):
        policy_cfg.num_steps = int(cfg["runtime"].get("policy_num_steps", 10))

    policy = make_policy(policy_cfg, ds_meta=ds_meta) if ds_meta is not None else make_policy(policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": cfg["policy"].get("device", "mps")},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    return policy_cfg, policy, preprocessor, postprocessor


def cmd_dry_policy(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = output_dir(cfg, "dry_policy")
    try:
        policy_cfg, policy, preprocessor, postprocessor = load_lerobot_policy(cfg)
        report = {
            "created": datetime.now().isoformat(),
            "status": "loaded",
            "policy_path": cfg["policy"]["path"],
            "policy_class": policy.__class__.__name__,
            "config_class": policy_cfg.__class__.__name__,
            "device": getattr(policy_cfg, "device", None),
            "has_preprocessor": preprocessor is not None,
            "has_postprocessor": postprocessor is not None,
            "input_features": str(getattr(policy_cfg, "input_features", "")),
            "output_features": str(getattr(policy_cfg, "output_features", "")),
            "normalization_mapping": str(getattr(policy_cfg, "normalization_mapping", "")),
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "created": datetime.now().isoformat(),
            "status": "failed",
            "policy_path": cfg["policy"]["path"],
            "error": repr(exc),
        }
    (out / "dry_policy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {out / 'dry_policy.json'}")
    return 0 if report["status"] == "loaded" else 2


def default_camera_indexes(cfg: dict[str, Any]) -> list[int]:
    camera_cfg = cfg["camera"]
    return [int(camera_cfg.get("scene_index", 0))]


def open_camera(cfg: dict[str, Any], index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["camera"]["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["camera"]["height"]))
    cap.set(cv2.CAP_PROP_FPS, int(cfg["camera"]["fps"]))
    return cap


def make_robot(cfg: dict[str, Any]):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    calibration_dir = cfg["robot"].get("calibration_dir")
    if calibration_dir is not None:
        calibration_dir = Path(calibration_dir)
        if not calibration_dir.is_absolute():
            calibration_dir = ROOT / calibration_dir

    robot_cfg = SOFollowerRobotConfig(
        id=cfg["robot"].get("id", "so101_follower"),
        port=cfg["robot"]["port"],
        disable_torque_on_disconnect=bool(cfg["robot"].get("disable_torque_on_disconnect", True)),
        max_relative_target=cfg["robot"].get("max_relative_target"),
        use_degrees=bool(cfg["robot"].get("use_degrees", True)),
        calibration_dir=calibration_dir,
        cameras={},
    )
    return SOFollower(robot_cfg)


def create_minimal_dataset_features(robot: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    from lerobot.datasets.feature_utils import combine_feature_dicts, hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    observation_hw = {
        **robot.observation_features,
        "camera1": (
            int(cfg["camera"]["height"]),
            int(cfg["camera"]["width"]),
            3,
        ),
    }
    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        hw_to_dataset_features(observation_hw, OBS_STR, use_video=False),
    )


def cmd_run_policy(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not args.execute or args.confirm != CONFIRM_TEXT:
        print("Refusing to move hardware.")
        print("--execute and exact --confirm text are required:")
        print(f'  --confirm "{CONFIRM_TEXT}"')
        return 2

    import cv2
    import torch
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.robot_utils import precise_sleep

    out = output_dir(cfg, "run_policy")
    robot = make_robot(cfg)
    scene_index = int(cfg["camera"].get("scene_index", 0))
    cap = open_camera(cfg, scene_index)
    frames = 0
    actions_sent = 0
    results: list[dict[str, Any]] = []

    try:
        robot.connect(calibrate=False)
        features = create_minimal_dataset_features(robot, cfg)

        class Meta:
            pass

        class DatasetStub:
            pass

        dataset = DatasetStub()
        dataset.features = features
        dataset.meta = Meta()
        dataset.meta.features = features

        policy_cfg, policy, preprocessor, postprocessor = load_lerobot_policy(cfg, ds_meta=dataset.meta)
        device = torch.device(cfg["policy"].get("device", "mps"))
        max_seconds = float(cfg["runtime"].get("max_seconds", 5.0))
        fps = float(cfg["runtime"].get("fps", 5))
        task = cfg["policy"]["task"]
        start = time.perf_counter()

        while time.perf_counter() - start < max_seconds:
            loop_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Scene camera index {scene_index} failed to return a frame")
            obs = robot.get_observation()
            obs["camera1"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            observation_frame = build_dataset_frame(features, obs, prefix=OBS_STR)
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(getattr(policy_cfg, "use_amp", cfg["runtime"].get("use_amp", False))),
                task=task,
                robot_type=robot.robot_type,
            )
            action = make_robot_action(action_values, features)
            returned = robot.send_action(action)
            actions_sent += 1
            frames += 1
            results.append({"step": actions_sent, "action": action, "returned": returned})
            precise_sleep(max((1.0 / fps) - (time.perf_counter() - loop_start), 0.0))

        report = {
            "created": datetime.now().isoformat(),
            "status": "completed",
            "policy_path": cfg["policy"]["path"],
            "task": task,
            "frames": frames,
            "actions_sent": actions_sent,
            "max_seconds": max_seconds,
            "fps": fps,
            "results": results,
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "created": datetime.now().isoformat(),
            "status": "failed",
            "error": repr(exc),
            "frames": frames,
            "actions_sent": actions_sent,
            "results": results,
        }
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            cap.release()

    (out / "run_policy.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"report: {out / 'run_policy.json'}")
    return 0 if report["status"] == "completed" else 2


def cmd_propose_action(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    import cv2
    import torch
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action

    out = output_dir(cfg, "propose_action")
    robot = make_robot(cfg)
    scene_index = int(cfg["camera"].get("scene_index", 0))
    cap = open_camera(cfg, scene_index)
    report: dict[str, Any]

    try:
        robot.connect(calibrate=False)
        features = create_minimal_dataset_features(robot, cfg)

        class Meta:
            pass

        class DatasetStub:
            pass

        dataset = DatasetStub()
        dataset.features = features
        dataset.meta = Meta()
        dataset.meta.features = features

        policy_cfg, policy, preprocessor, postprocessor = load_lerobot_policy(cfg, ds_meta=dataset.meta)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Scene camera index {scene_index} failed to return a frame")
        image_path = out / f"camera1_index_{scene_index}.jpg"
        cv2.imwrite(str(image_path), frame_bgr)

        obs = robot.get_observation()
        obs["camera1"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        observation_frame = build_dataset_frame(features, obs, prefix=OBS_STR)
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=torch.device(cfg["policy"].get("device", "mps")),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(policy_cfg, "use_amp", cfg["runtime"].get("use_amp", False))),
            task=cfg["policy"]["task"],
            robot_type=robot.robot_type,
        )
        action = make_robot_action(action_values, features)
        report = {
            "created": datetime.now().isoformat(),
            "status": "completed",
            "send_action_called": False,
            "policy_actions_executed": False,
            "policy_path": cfg["policy"]["path"],
            "task": cfg["policy"]["task"],
            "scene_camera_index": scene_index,
            "camera_image": str(image_path),
            "robot_port": cfg["robot"]["port"],
            "device": cfg["policy"].get("device", "mps"),
            "action": action,
            "features": str(features),
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "created": datetime.now().isoformat(),
            "status": "failed",
            "send_action_called": False,
            "policy_actions_executed": False,
            "error": repr(exc),
        }
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            cap.release()

    (out / "propose_action.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"report: {out / 'propose_action.json'}")
    return 0 if report["status"] == "completed" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["doctor", "dry-policy", "propose-action", "run-policy", "snap-cameras"]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "config.local.toml")
        if name == "run-policy":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--confirm", default="")
        if name == "snap-cameras":
            p.add_argument("--indexes", type=int, nargs="*")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "snap-cameras":
        return cmd_snap_cameras(args)
    if args.cmd == "dry-policy":
        return cmd_dry_policy(args)
    if args.cmd == "propose-action":
        return cmd_propose_action(args)
    if args.cmd == "run-policy":
        return cmd_run_policy(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
