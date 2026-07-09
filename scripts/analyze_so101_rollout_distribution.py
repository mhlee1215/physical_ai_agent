#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SO101 closed-loop rollout state/action values against a LeRobot training dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=None, help="Optional train episode for trajectory RMSE.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze_distribution(
        dataset_root=args.dataset_root,
        rollout_jsonl=args.rollout_jsonl,
        episode_index=args.episode_index,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


def analyze_distribution(*, dataset_root: Path, rollout_jsonl: Path, episode_index: int | None = None) -> dict[str, Any]:
    data = _load_dataset_table(dataset_root)
    rollout = _load_rollout(rollout_jsonl)
    report: dict[str, Any] = {
        "operation": "analyze_so101_rollout_distribution",
        "dataset_root": str(dataset_root),
        "rollout_jsonl": str(rollout_jsonl),
        "dataset_rows": int(len(data)),
        "rollout_rows": int(len(rollout)),
        "env_observation_first6": _compare(
            reference=np.stack(data["observation.state"].to_numpy()).astype(float)[:, :6],
            values=np.asarray([row["observation"][:6] for row in rollout], dtype=float),
        ),
        "action": _compare(
            reference=np.stack(data["action"].to_numpy()).astype(float)[:, :6],
            values=np.asarray([row["action"][:6] for row in rollout], dtype=float),
        ),
    }
    policy_state_values = [
        (
            (row.get("policy_output") or {}).get("policy_input_motor_state_raw")
            or (row.get("policy_output") or {}).get("policy_input_state_raw")
            or []
        )[:6]
        for row in rollout
        if (row.get("policy_output") or {}).get("policy_input_motor_state_raw")
        or (row.get("policy_output") or {}).get("policy_input_state_raw")
    ]
    if policy_state_values:
        report["policy_input_motor_state"] = _compare(
            reference=np.stack(data["observation.state"].to_numpy()).astype(float)[:, :6],
            values=np.asarray(policy_state_values, dtype=float),
        )
    qpos_values = [
        ((row.get("sim_snapshot") or {}).get("qpos") or [])[:6]
        for row in rollout
        if (row.get("sim_snapshot") or {}).get("qpos")
    ]
    if qpos_values:
        report["sim_qpos"] = _compare(
            reference=np.stack(data["observation.state"].to_numpy()).astype(float)[:, :6],
            values=np.asarray(qpos_values, dtype=float),
        )
    if episode_index is not None:
        episode = data[data["episode_index"] == int(episode_index)].sort_values("frame_index")
        if not episode.empty:
            teacher_state = np.stack(episode["observation.state"].to_numpy()).astype(float)[:, :6]
            teacher_action = np.stack(episode["action"].to_numpy()).astype(float)[:, :6]
            rollout_state = np.asarray([row["observation"][:6] for row in rollout], dtype=float)
            rollout_policy_state = (
                np.asarray(policy_state_values, dtype=float) if policy_state_values else rollout_state
            )
            rollout_action = np.asarray([row["action"][:6] for row in rollout], dtype=float)
            horizon = min(len(teacher_state), len(rollout_state))
            policy_state_horizon = min(len(teacher_state), len(rollout_policy_state))
            report["teacher_episode"] = {
                "episode_index": int(episode_index),
                "teacher_frames": int(len(episode)),
                "compared_frames": int(horizon),
                "env_observation_state_rmse": _rmse_summary(rollout_state[:horizon], teacher_state[:horizon]),
                "policy_input_state_rmse": _rmse_summary(
                    rollout_policy_state[:policy_state_horizon],
                    teacher_state[:policy_state_horizon],
                ),
                "action_rmse": _rmse_summary(rollout_action[:horizon], teacher_action[:horizon]),
            }
    return report


def _load_dataset_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {root / 'data'}")
    return pd.concat(
        [
            pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state", "action"])
            for path in files
        ],
        ignore_index=True,
    )


def _load_rollout(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compare(*, reference: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    low = reference.min(axis=0)
    high = reference.max(axis=0)
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std < 1e-8] = 1.0
    below = values < low
    above = values > high
    z = np.abs((values - mean) / std)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return {
        "count": int(values.shape[0]),
        "reference_min": _round(low),
        "reference_max": _round(high),
        "value_min": _round(values.min(axis=0)),
        "value_max": _round(values.max(axis=0)),
        "outside_minmax_ratio_any_dim": float(np.round((below | above).any(axis=1).mean(), 6)),
        "outside_minmax_ratio_by_dim": dict(zip(names, _round((below | above).mean(axis=0)), strict=False)),
        "max_abs_z_by_dim": dict(zip(names, _round(z.max(axis=0)), strict=False)),
        "mean_abs_z_by_dim": dict(zip(names, _round(z.mean(axis=0)), strict=False)),
        "frames_with_abs_z_gt_3_ratio": float(np.round((z > 3.0).any(axis=1).mean(), 6)),
        "frames_with_abs_z_gt_5_ratio": float(np.round((z > 5.0).any(axis=1).mean(), 6)),
    }


def _rmse_summary(values: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    rmse = np.sqrt(((values - reference) ** 2).mean(axis=1))
    return {
        "mean": float(np.round(rmse.mean(), 6)),
        "max": float(np.round(rmse.max(), 6)),
        "step0": float(np.round(rmse[0], 6)),
        "last": float(np.round(rmse[-1], 6)),
    }


def _round(values: Any) -> list[float]:
    return [float(value) for value in np.round(np.asarray(values, dtype=float), 6).tolist()]


if __name__ == "__main__":
    main()
