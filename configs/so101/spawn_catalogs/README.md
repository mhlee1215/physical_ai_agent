# SO101 Spawn Catalogs

These checked-in JSON files are generation inputs, not LeRobot datasets.

`so101_spawn_catalog_v1` contains only:

- camera1 grid geometry;
- MuJoCo world-frame object spawn `[x, y]` candidates grouped by bin.

It must not contain source episode seeds, frames, actions, observations, or
dataset paths. Generate one with `scripts/extract_so101_spawn_catalog.py`, then
reference it through recipe `source.mode: from_spawn_catalog` and the generated
split's `lookup_cache`.
