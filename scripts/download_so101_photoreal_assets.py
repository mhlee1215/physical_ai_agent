#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


ASSETS = {
    "polyhaven/studio_small_08_2k.hdr": {
        "url": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_08_2k.hdr",
        "md5": "18571cc63a24b0b7b9e9e7c909b3e8ee",
    },
    "ambientcg/Wood008_1K-JPG.zip": {
        "url": "https://ambientcg.com/get?file=Wood008_1K-JPG.zip",
        "extract_to": "ambientcg/Wood008_1K-JPG",
    },
    "ambientcg/Plastic013A_1K-JPG.zip": {
        "url": "https://ambientcg.com/get?file=Plastic013A_1K-JPG.zip",
        "extract_to": "ambientcg/Plastic013A_1K-JPG",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CC0 HDRI/PBR assets for SO101 photoreal previews.")
    parser.add_argument("--asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.asset_root.mkdir(parents=True, exist_ok=True)
    records = []
    for relative, spec in ASSETS.items():
        path = args.asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not path.exists():
            _download(spec["url"], path)
        md5 = _md5(path)
        expected_md5 = spec.get("md5")
        status = "passed" if expected_md5 is None or md5 == expected_md5 else "failed"
        if "extract_to" in spec:
            extract_dir = args.asset_root / str(spec["extract_to"])
            if args.overwrite or not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(extract_dir)
        records.append(
            {
                "path": str(path),
                "url": spec["url"],
                "md5": md5,
                "expected_md5": expected_md5,
                "status": status,
                "extract_to": str(args.asset_root / str(spec["extract_to"])) if "extract_to" in spec else None,
            }
        )
    report = {"asset_root": str(args.asset_root), "assets": records}
    report_path = args.asset_root / "so101_photoreal_assets_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "physical-ai-agent-photoreal-assets/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
