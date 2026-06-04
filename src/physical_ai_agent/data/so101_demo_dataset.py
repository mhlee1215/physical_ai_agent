from __future__ import annotations

import json
from pathlib import Path

from physical_ai_agent.sim.so101_nexus_env import SO101Step


def write_demo_dataset(output_dir: Path, steps: list[SO101Step], task: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    meta_path = output_dir / "metadata.json"
    episodes_path.write_text("", encoding="utf-8")
    for index, step in enumerate(steps):
        record = {
            "episode_index": 0,
            "frame_index": index,
            "timestamp": index / 20.0,
            "task": task,
            "observation": {
                "state": step.observation,
                "images": {},
            },
            "action": step.action,
            "reward": step.reward,
            "done": step.terminated or step.truncated or index == len(steps) - 1,
        }
        with episodes_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    meta = {
        "format": "lerobot_like_jsonl",
        "robot": "SO-100/101",
        "task": task,
        "episodes": 1,
        "frames": len(steps),
        "notes": "Intermediate CP13 artifact; convert to LeRobotDataset parquet/video in later training checkpoints.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return {"episodes": str(episodes_path), "metadata": str(meta_path)}
