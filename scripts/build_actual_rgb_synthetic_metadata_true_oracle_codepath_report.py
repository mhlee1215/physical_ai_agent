#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.checkpoints.checkpoint_24 import _augment_obs_with_oracle_state
from physical_ai_agent.perception.affordance_overlay import build_oracle_affordance_overlay


DEFAULT_ROOT = Path("_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z")
FRAME_RE = re.compile(
    r"(?P<policy>[a-zA-Z0-9_]+)_episode_(?P<episode>\d+)_(?P<kind>reset|step)_(?P<step>\d+)\.png$"
)


@dataclass(frozen=True)
class CodepathSample:
    sample_id: str
    source_frame: str
    overlay_frame: str
    panel_frame: str
    policy: str
    episode: int
    step: int
    mode: str
    point_xy: list[int]
    object_pose_xyz: list[float] | None
    camera_metadata_keys: list[str] | None
    strict_codepath_ready: bool


class _FakePose:
    def __init__(self, p: list[float]) -> None:
        self.p = np.asarray(p, dtype=np.float32)


class _FakeCube:
    def __init__(self, p: list[float]) -> None:
        self.pose = _FakePose(p)


class _FakeCamera:
    def __init__(self, intrinsic: Any, extrinsic: Any) -> None:
        self.intrinsic_cv = np.asarray(intrinsic, dtype=np.float32)
        self.extrinsic_cv = np.asarray(extrinsic, dtype=np.float32)


class _FakeEnv:
    def __init__(self, object_pose: list[float], camera_name: str, intrinsic: Any, extrinsic: Any) -> None:
        self.cube = _FakeCube(object_pose)
        self._sensors = {camera_name: _FakeCamera(intrinsic, extrinsic)}

    @property
    def unwrapped(self) -> "_FakeEnv":
        return self


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "actual_rgb_synthetic_metadata_true_oracle_codepath"
    overlay_dir = output_dir / "overlay"
    panel_dir = output_dir / "panels"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    frames = _select_frames(_discover_frames(), args.limit)
    samples = [
        _build_sample(index, frame, overlay_dir, panel_dir)
        for index, frame in enumerate(frames)
    ]
    contact_sheet = output_dir / "actual_rgb_synthetic_metadata_codepath_contact_sheet.jpg"
    _write_contact_sheet([Path(sample.panel_frame) for sample in samples], contact_sheet)

    strict_count = sum(1 for sample in samples if sample.strict_codepath_ready)
    manifest = {
        "status": "passed" if strict_count >= min(args.limit, 10) else "failed",
        "source_type": "actual_sim_rgb_synthetic_metadata_codepath_diagnostic",
        "real_sim_episode": True,
        "synthetic_metadata": True,
        "true_oracle_projection": False,
        "codepath_projection_ready": strict_count >= min(args.limit, 10),
        "sample_count": len(samples),
        "strict_codepath_ready_count": strict_count,
        "minimum_required_samples": min(args.limit, 10),
        "contact_sheet": str(contact_sheet),
        "samples": [asdict(sample) for sample in samples],
        "claim_boundary": (
            "Uses actual simulator RGB frames but synthetic object pose/camera metadata. "
            "This validates the augmentation/projection codepath only and is not Tier O evidence."
        ),
    }
    (output_dir / "actual_rgb_synthetic_metadata_codepath_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "actual_rgb_synthetic_metadata_codepath_report.html"
    html_path.write_text(_html(samples, manifest), encoding="utf-8")
    print(html_path)
    return 0 if manifest["status"] == "passed" else 1


def _discover_frames() -> list[Path]:
    root = Path("_workspace/checkpoints")
    if not root.exists():
        return []
    return sorted(path for path in root.glob("**/maniskill_rollout/rollout_frames/*.png") if FRAME_RE.search(path.name))


def _select_frames(frames: list[Path], limit: int) -> list[Path]:
    chosen: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for path in frames:
        match = FRAME_RE.search(path.name)
        if not match:
            continue
        key = (match.group("policy"), match.group("episode"))
        if key in seen:
            continue
        chosen.append(path)
        seen.add(key)
        if len(chosen) >= limit:
            return chosen
    for path in frames:
        if path not in chosen:
            chosen.append(path)
        if len(chosen) >= limit:
            break
    return chosen


def _build_sample(index: int, frame: Path, overlay_dir: Path, panel_dir: Path) -> CodepathSample:
    match = FRAME_RE.search(frame.name)
    if not match:
        raise RuntimeError(f"Unexpected frame name: {frame}")
    policy = match.group("policy")
    episode = int(match.group("episode"))
    step = int(match.group("step"))

    pixels = np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8)
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    camera_name = "base_camera"
    fx = float(max(width, height))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    target_u = width * (0.22 + 0.56 * ((index % 4) / 3.0))
    target_v = height * (0.25 + 0.5 * (((index // 4) % 3) / 2.0))
    z = 1.0
    object_pose = [(target_u - cx) * z / fx, (target_v - cy) * z / fy, z]
    intrinsic = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    extrinsic = np.eye(4, dtype=np.float32).tolist()

    obs = {"sensor_data": {camera_name: {"rgb": pixels}}}
    env = _FakeEnv(object_pose, camera_name, intrinsic, extrinsic)
    augmented = _augment_obs_with_oracle_state(env, obs)
    overlay_path = overlay_dir / f"codepath_{index:03d}.png"
    _overlay_images, metadata = build_oracle_affordance_overlay(
        augmented,
        output_path=overlay_path,
        preferred_camera=camera_name,
        label="codepath diagnostic",
    )
    panel_path = panel_dir / f"codepath_panel_{index:03d}.jpg"
    _write_panel(frame, overlay_path, panel_path, metadata.metadata())
    strict_ready = (
        metadata.mode == "projected_object_pose"
        and bool(metadata.object_pose_xyz)
        and bool(metadata.camera_metadata_keys)
        and overlay_path.exists()
    )
    return CodepathSample(
        sample_id=f"{policy}_ep{episode:03d}_step{step:04d}",
        source_frame=str(frame),
        overlay_frame=str(overlay_path),
        panel_frame=str(panel_path),
        policy=policy,
        episode=episode,
        step=step,
        mode=metadata.mode,
        point_xy=metadata.point_xy,
        object_pose_xyz=metadata.object_pose_xyz,
        camera_metadata_keys=metadata.camera_metadata_keys,
        strict_codepath_ready=strict_ready,
    )


def _write_panel(raw_path: Path, overlay_path: Path, panel_path: Path, metadata: dict[str, Any]) -> None:
    raw = Image.open(raw_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")
    raw.thumbnail((360, 260))
    overlay.thumbnail((360, 260))
    panel = Image.new("RGB", (760, 390), (248, 239, 218))
    panel.paste(raw, (14, 40))
    panel.paste(overlay, (386, 40))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 14), "actual sim RGB", fill=(32, 39, 31))
    draw.text((386, 14), "projected overlay from synthetic metadata", fill=(32, 39, 31))
    draw.rectangle((14, 314, 746, 376), fill=(255, 250, 240))
    draw.text((26, 326), f"mode={metadata.get('mode')} point={metadata.get('point_xy')}", fill=(34, 111, 76))
    draw.text((26, 350), "NOT Tier O: actual RGB + synthetic pose/camera metadata codepath diagnostic", fill=(178, 63, 48))
    panel.save(panel_path, quality=92)


def _write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((380, 195))
        tile = Image.new("RGB", (380, 195), (255, 250, 240))
        tile.paste(image, ((380 - image.width) // 2, (195 - image.height) // 2))
        thumbs.append(tile)
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 380, rows * 195), (234, 218, 185))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 380, (index // columns) * 195))
    sheet.save(output_path, quality=92)


def _html(samples: list[CodepathSample], manifest: dict[str, Any]) -> str:
    cards = "\n".join(_card(sample) for sample in samples)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Actual RGB + Synthetic Metadata True-Oracle Codepath Diagnostic</title>
  <style>
    :root {{ --ink:#18201b; --paper:#f8eed8; --panel:#fffaf0; --line:#d0bd92; --green:#2d7a55; --red:#b23f30; --muted:#637060; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:linear-gradient(135deg,#fbf3df,#ead8b1); }}
    header {{ padding:42px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(34px,6vw,74px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:980px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .stats {{ max-width:1180px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:20px; }}
    .tile, .card {{ background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:24px; padding:16px; box-shadow:0 18px 48px rgba(50,35,10,.11); }}
    .tile b {{ display:block; font-size:30px; margin-bottom:7px; }}
    .tile span {{ color:var(--muted); line-height:1.35; }}
    .contact {{ max-width:1180px; margin-bottom:20px; }}
    .contact img, .card img {{ width:100%; display:block; border-radius:16px; background:#222; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .card h2 {{ font-size:20px; margin:12px 0 8px; }}
    .card p {{ color:var(--muted); line-height:1.38; margin:5px 0; }}
    .ok {{ color:var(--green); }} .bad {{ color:var(--red); }}
    code {{ background:rgba(24,32,27,.08); padding:2px 5px; border-radius:6px; }}
    footer {{ max-width:1180px; color:var(--muted); margin-top:28px; }}
    @media (max-width:900px) {{ .stats,.grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Actual RGB + Synthetic Metadata True-Oracle Codepath Diagnostic</h1>
    <p class="lead">This verifies the new augmentation/projection code path on real simulator RGB frames. It is intentionally not Tier O because object pose and camera metadata are synthetic.</p>
  </header>
  <main>
    <section class="stats">
      <div class="tile"><b class="ok">{manifest['sample_count']}</b><span>actual simulator RGB frames used</span></div>
      <div class="tile"><b class="ok">{manifest['strict_codepath_ready_count']}</b><span>projected_object_pose codepath samples</span></div>
      <div class="tile"><b class="bad">false</b><span>Tier O true-oracle claim</span></div>
      <div class="tile"><b class="bad">synthetic</b><span>object pose and camera metadata</span></div>
    </section>
    <section class="contact card">
      <img src="actual_rgb_synthetic_metadata_codepath_contact_sheet.jpg" alt="codepath contact sheet" />
      <h2>10+ representative codepath samples</h2>
      <p>Left panels are actual simulator RGB. Right panels show projected overlays generated after synthetic env metadata augmentation.</p>
    </section>
    <section class="grid">
      {cards}
    </section>
    <footer>Manifest: actual_rgb_synthetic_metadata_codepath_manifest.json. Claim boundary: {manifest['claim_boundary']}</footer>
  </main>
</body>
</html>
"""


def _card(sample: CodepathSample) -> str:
    panel = Path(sample.panel_frame).name
    return f"""
      <article class="card">
        <img src="panels/{panel}" alt="{sample.sample_id}" />
        <h2>{sample.policy} episode {sample.episode}, step {sample.step}</h2>
        <p>mode=<code>{sample.mode}</code>; point=<code>{sample.point_xy}</code></p>
        <p>camera keys=<code>{sample.camera_metadata_keys}</code></p>
      </article>
    """


if __name__ == "__main__":
    raise SystemExit(main())
