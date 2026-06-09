#!/usr/bin/env python3
"""Extract representative midpoint frames from Meta-World eval videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frame(run_dir: Path, split: str, episode: int) -> Path:
    video = run_dir / "eval" / "videos" / f"{split}_0" / f"eval_episode_{episode}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video}")

    out_dir = run_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}_0_eval_episode_{episode}_mid.png"
    cv2.imwrite(str(out), frame)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--splits",
        default="easy,medium,hard,very_hard",
        help="Comma-separated split names to sample.",
    )
    args = parser.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(extract_frame(args.run_dir, split, args.episode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
