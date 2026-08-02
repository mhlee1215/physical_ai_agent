# SO101 Workspace Probe Selections

This directory stores seed-free placement selections and evidence derived from
completed workspace probes.

- Executable `WorkspaceProbeConfig` files belong in
  `configs/so101/workspace_probes/`.
- Selected placements, probe hashes, and generation provenance belong here.
- A spawn catalog may reference a file here, but it must not copy source
  frames, actions, states, or episode seeds.

Keeping these artifacts separate lets the checked-in probe-config test parse
every JSON file under `workspace_probes/` without guessing its document type.
