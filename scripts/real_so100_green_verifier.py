#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.perception.green_object_verifier import (
    image_paths_from_episode_record,
    verify_green_object_images,
    write_verifier_result,
)


def _load_episode_record(path: Path, frame_index: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["frame_index"]) == frame_index:
            return record
    raise ValueError(f"frame_index={frame_index} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify green object visibility in a real SO-100 episode.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-px", type=int, default=800)
    args = parser.parse_args()

    record = _load_episode_record(args.episode, args.frame_index)
    result = verify_green_object_images(
        image_paths_from_episode_record(record),
        min_area_px=args.min_area_px,
    )
    write_verifier_result(result, args.output)
    print(json.dumps(result, default=lambda obj: obj.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
