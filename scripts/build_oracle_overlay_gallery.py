#!/usr/bin/env python3
"""Build an HTML gallery for oracle overlay frame directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.perception.overlay_gallery import build_overlay_gallery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Oracle Overlay Gallery")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()

    manifest = build_overlay_gallery(
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        title=args.title,
        limit=args.limit,
        min_frames=args.min_frames,
    )
    print(json.dumps({"frame_count": manifest["frame_count"], "html": manifest.get("html")}, indent=2))
    return 0 if manifest.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
