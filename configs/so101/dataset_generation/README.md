# SO101 Dataset Recipes

Every new dataset starts as one JSON file in this directory. Use
`schema_version: 2`; schema-v1 files are historical recipes and must not be
edited in place.

Choose exactly one source contract:

```json
"source": {"mode": "from_scratch"}
```

Use this when MuJoCo creates the spawn state and teacher trajectory without
reading another dataset.

```json
"source": {
  "mode": "from_spawn_catalog",
  "catalogs": ["configs/so101/spawn_catalogs/task_v1.json"]
}
```

Use this when only validated world-XY spawn candidates are reused. A spawn
catalog contains `bin -> [[x, y], ...]` only: no source frames, actions, states,
or seeds. The generated split must point `lookup_cache` at the same catalog, so
the exporter creates new simulator state, teacher trajectories, images, and
episode seeds.

```json
"source": {
  "mode": "from_existing_dataset",
  "operation": "regenerate_teacher",
  "datasets": ["_workspace/so101_lerobot/source_v1"]
}
```

Use `regenerate_teacher` to reuse declared source reports/spawn layouts while
writing new trajectories. Use `render_derivative` to preserve source
state/action/timeline and replace only configured image streams.

Use `episode_subset` to materialize only episodes that pass a declared metric
into a new append-only LeRobot root. The output preserves every retained
frame/action/state unchanged, writes the source episode index as provenance,
and rebuilds metadata, stats, audit, and camera-grid sidecars.

Prefer `from_spawn_catalog` over `regenerate_teacher` when the only reusable
information is object placement. Build the catalog once with
`scripts/extract_so101_spawn_catalog.py`, then keep dataset roots out of the new
recipe.

New generated `grip_the_cube_v1` recipes require one geometry gate and may add
one camera2 visual gate in `common.inspection_gates`:

```json
"inspection_gates": [
  {
    "kind": "geometry_contact_alignment",
    "contract": "jaw_line_vs_contact_face_normal_through_cube_center",
    "max_pre_close_error_deg": 3.0
  },
  {
    "kind": "camera2_visual_alignment",
    "camera_key": "observation.images.camera2",
    "edge_mode": "top_contact",
    "strategy": "constructive_refine_then_probe",
    "mode": "preclose_and_early_trace",
    "pre_close_max_deg": 8.0,
    "close_25_max_deg": 8.0,
    "close_50_max_deg": 8.0
  }
]
```

Place this object under `common`. Pydantic rejects missing, unknown, or
contradictory fields before generation. The generator forwards these limits to
the teacher exporter, so rejected trajectories never enter the dataset.

Run a config-first dry run before generating:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/generate_so101_dataset_recipe.py \
  --recipe configs/so101/dataset_generation/<dataset>.json \
  --dry-run
```

The recipe must use new append-only output roots. A successful real run ends at
the mandatory registry, training-ready, and viewer completion gate.
