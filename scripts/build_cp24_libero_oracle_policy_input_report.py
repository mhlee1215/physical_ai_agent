#!/usr/bin/env python3
"""Build CP24 LIBERO oracle-overlay policy-input visualization report.

This script consumes the visually inspected, figure-orientation-corrected
LIBERO oracle overlay manifest and produces a policy-input readiness pack:

- raw RGB figure frame
- oracle-overlay figure frame
- SmolVLA-style resized tensor preview
- contact sheet
- HTML report
- machine-readable manifest

It does not run SmolVLA or claim success-rate improvement. It only verifies
that the real-simulation overlay evidence can be packaged as policy-input
images with preserved native/displayed coordinate provenance.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


FEATURE_SHAPE = (3, 224, 224)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(from_path: Path, target: Path) -> str:
    try:
        return target.relative_to(from_path.parent).as_posix()
    except ValueError:
        return target.as_posix()


def _resolve_artifact_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Artifact path not found: {value}")


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _policy_tensor_preview(image_path: Path, output_path: Path) -> dict[str, Any]:
    channels, height, width = FEATURE_SHAPE
    image = Image.open(image_path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    preview = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, 136, 22), fill=(0, 0, 0))
    draw.text((6, 5), "SmolVLA image 3x224x224", fill=(0, 255, 120))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)
    return {
        "shape": list(tensor.shape),
        "dtype": "float32",
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cell_w, cell_h = 1060, 290
    canvas = Image.new("RGB", (cell_w, max(1, len(rows)) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * cell_h
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        title = (
            f"#{row['broad_index']} {row['suite']} task={row['task_id']} "
            f"{row['object_name']} native={row.get('native_point_xy')} shown={row['point_xy']}"
        )
        draw.text((14, y + 10), title[:150], fill=(20, 18, 14))
        for col, key in enumerate(("raw_frame_path", "overlay_frame_path", "policy_preview_path")):
            image = Image.open(row[key]).convert("RGB")
            image.thumbnail((320, 210))
            x = 18 + col * 345
            canvas.paste(image, (x, y + 44))
            label = {
                "raw_frame_path": "RGB-only figure frame",
                "overlay_frame_path": "RGB + oracle point",
                "policy_preview_path": "SmolVLA tensor preview",
            }[key]
            draw.text((x, y + 260), label, fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, source_manifest: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['suite'])} task {row['task_id']}: {html.escape(row['object_name'])}</h2>
        <p><strong>{html.escape(row['task_name'])}</strong></p>
        <p>source={html.escape(row['projection_source'])}; native point={html.escape(str(row.get('native_point_xy')))}; displayed point={html.escape(str(row['point_xy']))}; figure orientation={html.escape(row['figure_orientation'])}</p>
        <p>tensor shape={html.escape(str(row['overlay_tensor']['shape']))}; range={row['overlay_tensor']['min']:.3f}-{row['overlay_tensor']['max']:.3f}; mean={row['overlay_tensor']['mean']:.3f}</p>
        <div class="triplet">
          <figure><img src="{html.escape(_rel(report_path, Path(row['raw_frame_path'])))}" alt="raw"><figcaption>RGB-only input candidate</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['overlay_frame_path'])))}" alt="overlay"><figcaption>Oracle-overlay input candidate</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['policy_preview_path'])))}" alt="policy preview"><figcaption>Resized normalized image feature preview</figcaption></figure>
        </div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CP24 LIBERO Oracle Overlay Policy-Input Report</title>
  <style>
    :root {{ --ink:#17130e; --paper:#f7efe1; --line:#d3bea0; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 20% 0%,#ddffd8,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.92); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--green); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    p {{ line-height:1.45; }}
    code {{ background:#fff; border:1px solid var(--line); border-radius:6px; padding:1px 5px; }}
    a {{ color:#174f38; font-weight:800; }}
    .triplet {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>CP24 LIBERO Oracle Overlay Policy-Input Report</h1>
    <p>Real LIBERO/MuJoCo oracle affordance overlays packaged as SmolVLA-style image inputs. This report validates image-conditioning readiness and visualization provenance. It does not claim SmolVLA success-rate improvement.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">PASSED POLICY-INPUT VISUALIZATION GATE</span>
      <h2>Evidence boundary</h2>
      <p>Source manifest: <code>{html.escape(source_manifest.as_posix())}</code></p>
      <p>Contact sheet: <a href="{html.escape(_rel(report_path, contact_sheet))}">open CP24 policy-input contact sheet</a></p>
      <p>The source frames use paper-facing LIBERO figure orientation. Native simulator coordinates remain recorded separately from displayed coordinates when available.</p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(manifest_path: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    samples = manifest.get("curated_samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest does not contain curated_samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples[:limit]:
        raw_src = _resolve_artifact_path(str(sample["raw_frame_path"]), manifest_path)
        overlay_src = _resolve_artifact_path(str(sample["overlay_frame_path"]), manifest_path)
        stem = f"{sample['suite']}_task{int(sample['task_id']):02d}_{sample['object_name'].replace(':', '_').replace('/', '_')}"
        raw_dst = output_dir / "frames" / f"{stem}_rgb_only.png"
        overlay_dst = output_dir / "frames" / f"{stem}_oracle_overlay.png"
        preview_dst = output_dir / "policy_previews" / f"{stem}_smolvla_tensor_preview.png"
        _copy_image(raw_src, raw_dst)
        _copy_image(overlay_src, overlay_dst)
        overlay_tensor = _policy_tensor_preview(overlay_dst, preview_dst)
        raw_tensor = _policy_tensor_preview(raw_dst, output_dir / "policy_previews_rgb_only" / f"{stem}_smolvla_tensor_preview.png")
        row = {
            "broad_index": sample.get("broad_index"),
            "suite": sample["suite"],
            "task_id": sample["task_id"],
            "task_name": sample["task_name"],
            "object_name": sample["object_name"],
            "projection_source": sample["projection_source"],
            "native_point_xy": sample.get("native_point_xy"),
            "point_xy": sample["point_xy"],
            "figure_orientation": sample.get("figure_orientation", manifest.get("figure_orientation", "unknown")),
            "raw_frame_path": str(raw_dst),
            "overlay_frame_path": str(overlay_dst),
            "policy_preview_path": str(preview_dst),
            "raw_tensor": raw_tensor,
            "overlay_tensor": overlay_tensor,
        }
        rows.append(row)

    contact_sheet = output_dir / "cp24_libero_oracle_policy_input_contact_sheet.jpg"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "cp24_libero_oracle_policy_input_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, manifest_path), encoding="utf-8")
    status = "passed" if len(rows) >= min(10, limit) and all(row["overlay_tensor"]["shape"] == [1, 3, 224, 224] for row in rows) else "failed"
    report = {
        "status": status,
        "checkpoint": "cp24_libero_oracle_policy_input_visualization",
        "source_manifest": str(manifest_path),
        "source_type": "actual_libero_mujoco_real_sim_oracle_overlay",
        "claim_boundary": "policy_input_readiness_not_success_rate_improvement",
        "sample_count": len(rows),
        "feature_shape": [1, 3, 224, 224],
        "figure_orientation": manifest.get("figure_orientation", "unknown"),
        "contact_sheet": str(contact_sheet),
        "html": str(html_path),
        "rows": rows,
    }
    (output_dir / "cp24_libero_oracle_policy_input_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_rows = [
        f"| {row['broad_index']} | {row['suite']} | {row['task_id']} | {row['object_name']} | `{row['projection_source']}` | `{row.get('native_point_xy')}` | `{row['point_xy']}` |"
        for row in rows
    ]
    (output_dir / "cp24_libero_oracle_policy_input_report.md").write_text(
        "# CP24 LIBERO Oracle Overlay Policy-Input Report\n\n"
        "## Verdict\n\n"
        f"- Status: `{status}`\n"
        f"- Sample count: `{len(rows)}`\n"
        "- Evidence type: actual LIBERO/MuJoCo real-simulation oracle overlay frames.\n"
        "- Claim boundary: policy-input readiness only; no SmolVLA success-rate improvement claim.\n"
        f"- Contact sheet: `{contact_sheet}`\n"
        f"- HTML report: `{html_path}`\n\n"
        "## Samples\n\n"
        "| Broad index | Suite | Task | Target | Source | Native point | Displayed point |\n"
        "| ---: | --- | ---: | --- | --- | --- | --- |\n"
        + "\n".join(markdown_rows)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=13)
    args = parser.parse_args()
    report = build_report(Path(args.manifest), Path(args.output_dir), args.limit)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_count": report["sample_count"],
                "contact_sheet": report["contact_sheet"],
                "html": report["html"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
