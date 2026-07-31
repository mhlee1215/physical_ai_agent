#!/usr/bin/env python3
"""Create deterministic, coordinate-disjoint partitions of an SO101 spawn catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _bucket(xy: list[float], *, salt: str, count: int) -> int:
    token = f"{salt}:{float(xy[0]):.12f}:{float(xy[1]):.12f}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % int(count)


def _rank(xy: list[float], *, salt: str) -> bytes:
    token = f"order:{salt}:{float(xy[0]):.12f}:{float(xy[1]):.12f}".encode()
    return hashlib.sha256(token).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--partition-indices", required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()

    count = int(args.partition_count)
    if count < 2:
        raise ValueError("--partition-count must be at least two")
    selected = {
        int(value.strip()) for value in args.partition_indices.split(",") if value.strip()
    }
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError("partition indices must be within [0, partition-count)")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source.get("format") != "so101_spawn_catalog_v1":
        raise ValueError("source must be an so101_spawn_catalog_v1 file")
    output = dict(source)
    output["catalog_id"] = str(args.catalog_id)
    output["partition"] = {
        "source": str(args.source),
        "count": count,
        "indices": sorted(selected),
        "salt": str(args.salt),
        "contract": "sha256(world_xy) modulo partition count",
    }
    output["lookup"] = {
        key: sorted([
            xy for xy in values
            if _bucket(xy, salt=str(args.salt), count=count) in selected
        ], key=lambda xy: _rank(xy, salt=str(args.salt)))
        for key, values in source["lookup"].items()
    }
    output["candidate_counts"] = {
        key: len(values) for key, values in output["lookup"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "candidate_counts": output["candidate_counts"]}, indent=2))


if __name__ == "__main__":
    main()
