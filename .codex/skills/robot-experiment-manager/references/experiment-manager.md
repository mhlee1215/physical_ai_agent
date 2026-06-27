# Robot Experiment Manager Reference

## Scope

The Experiment Manager is the single browser UI for robot policy work in this
repo. It should cover SO101, MyCobot, and future robot platforms without
renaming the product back to an SO101-only tool.

Main files:

- `scripts/serve_so101_dataset_viewer.py`: unified dashboard, data viewer,
  training manager panel, embedded loop analyzer route, interactive simulator.
- `scripts/serve_loop_test_analyzer.py`: standalone loop-test analyzer reused
  under `/loop-analyzer/`.
- `configs/so101/training_datasets/*.json`: training dataset configs.
- `configs/so101/training_datasets/skill_dataset_contract.json`: SO101
  primitive dataset contract.
- `configs/so101/training_datasets/README.md`: durable dataset/training rules.

Server:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_so101_dataset_viewer.py --host 0.0.0.0 --port 8768
```

Primary local URL:

```text
http://127.0.0.1:8768/
```

## Dashboard Identity

Keep:

- HTML title: `Robot Experiment Manager`
- Header: `Experiment Manager`
- Subtitle: generic robot-policy wording, not SO101-only wording.

Good subtitle shape:

```text
Unified workspace for robot policy datasets, training runs, loop-test analysis, and interactive rollouts.
```

Do not use text like `One place for SO101...` because the dashboard must cover
SO101 and MyCobot.

## Data Viewer

### Purpose

The Data Viewer previews stored robot datasets. It must show the actual dataset
contents: prompt, camera frames, state/action, episode/frame metadata, and
dataset volume. Do not correct labels only in UI if the backend payload is
wrong.

### API shape

`GET /api/datasets` returns:

- `datasets`: map from split name to summary.
- `dataset_groups`: grouped catalog items.
- `camera_view_note`: note about camera semantics.

Each available catalog item should include at least:

- `name`
- `root`
- `category`
- `status`
- `detail`
- `platform`
- `platform_label`
- `summary`

Each summary should include at least:

- `name`
- `root`
- `platform`
- `platform_label`
- `episodes`
- `frames`
- `fps`
- `size_bytes`
- `size_human`
- `data_bytes` / `data_human` when available
- `image_bytes` / `image_human` when available
- `features`
- `image_shapes`
- `episode_lengths`

`GET /api/frame?split=<name>&episode=<i>&frame=<j>` returns:

- `split`
- `episode`
- `frame`
- `episode_length`
- `row_index`
- `timestamp`
- `task`
- `prompt`
- `task_index` if available
- `images`
- `state`
- `action`

### Platform Filtering

The Data Viewer has a `Platform` select. Current platform ids:

- `so101` with label `SO101`
- `mycobot` with label `MyCobot`

Backend catalog items and summaries must set `platform` and `platform_label`.
Frontend filtering should use those fields. If no dataset exists for a platform,
show an empty state; do not leave stale SO101 content visible.

Platform inference can be path/name-based for temporary POC data, but durable
configs should declare platform metadata explicitly once a platform has a
contract.

### Adding SO101 Datasets

For repeatable SO101 datasets:

- Register train/validation/loop-validation roots in config or contract files.
- Use `configs/so101/training_datasets/skill_dataset_contract.json` for
  additive primitive datasets.
- Use `closed_loop.test_cases` in the training dataset config for official loop
  tests.
- Keep raw LeRobot data under `_workspace/`; commit configs, recipes, tests, and
  checksum manifests only.

For temporary local SO101 inspection:

```bash
SO101_TEMP_DATASETS=name=/abs/path/to/lerobot_dataset \
PYTHONPATH=src .venv/bin/python scripts/serve_so101_dataset_viewer.py --host 0.0.0.0 --port 8768
```

### Adding MyCobot Datasets

For MyCobot POC datasets, especially JSONL teacher exports:

- It is acceptable to expose them in the viewer before LeRobot parquet export.
- Mark the item as `platform: "mycobot"` and `platform_label: "MyCobot"`.
- Mark the format clearly, for example `dataset_format:
  "mycobot_jsonl_v1"`.
- Treat them as previewable teacher datasets, not SmolVLA training-ready
  LeRobot datasets.
- Add a format-specific adapter instead of pretending JSONL is a LeRobot
  parquet dataset.

Recommended POC discovery:

```bash
MYCOBOT_TEMP_DATASETS=name=/abs/path/to/mycobot_jsonl_dataset \
PYTHONPATH=src .venv/bin/python scripts/serve_so101_dataset_viewer.py --host 0.0.0.0 --port 8768
```

If the current branch does not yet implement `MYCOBOT_TEMP_DATASETS`, add it as
a separate MyCobot JSONL discovery path. Do not overload `SO101_TEMP_DATASETS`
for MyCobot.

Recommended JSONL summary extras:

- `dataset_format`
- `robot_model`
- `gripper`
- `gate`
- `rendered_frames`
- `failed_episodes`

## Training Manager

### Purpose

The Training Manager lists training runs and their checkpoints. It should make
the relationship explicit:

```text
training run -> checkpoint -> policy path
```

The Interactive Simulator and Loop Test Analyzer should show which training run
and checkpoint a policy came from when that metadata is available.

### Discovery

Current discovery reads:

- `_workspace/so101_training/runs/**/training_run_summary.json`
- `_workspace/so101_training/training_runs_index.json`
- checkpoint folders under run directories

Checkpoint folders are valid policy candidates when they contain:

```text
*/pretrained_model/config.json
*/pretrained_model/model.safetensors
```

### Training Launcher Rule

For SO101 training, use:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_so101_training.py start --preset <preset>
```

Do not add another shell wrapper with the same purpose. If a launch shape becomes
common, add a preset to `scripts/start_so101_training.py`.

## Loop Test Analyzer

### Purpose

Loop tests are official closed-loop evaluations, not arbitrary replay media.
The analyzer should expose:

- latest matching rollout per configured test case
- success rate and status
- prompt/scenario
- start contract
- training step/checkpoint
- training source metadata when available
- media generation status and artifacts
- raw payload for debugging

### Official Test Case Contract

Closed-loop tests belong in the same dataset config contract as training and
validation datasets:

```json
{
  "closed_loop": {
    "test_cases": [
      {
        "id": "...",
        "episodes": 10,
        "steps": 60,
        "seed": 98100,
        "start_contract": "...",
        "task_prompt": "...",
        "qwen_object": "green cube",
        "env_object_color": "green",
        "plan_json": "...",
        "start_dataset": {
          "root": "..."
        }
      }
    ]
  }
}
```

Rules:

- Do not create dashboard-only loop test definitions.
- Do not rename, remove, or reinterpret loop tests without explicit user
  approval.
- For dataset-aligned loop tests, create a separate loop-validation split and
  initialize episode `i` from that split's episode `i`.
- Keep prompts, object color/shape, camera contract, start contract, and dataset
  source index in the rollout report.
- If a media-render flag is disabled, still ensure policy camera renderers exist
  when the policy needs image input. Avoid zero-tensor policy input bugs.

Integrated route in Experiment Manager:

```text
/loop-analyzer/
/loop-analyzer/api/loop-tests
/loop-analyzer/api/loop-test?id=<id>
/loop-analyzer/api/generate-media?loop=<id>
/loop-analyzer/api/generate-media-status?loop=<id>
```

## Interactive Simulator

### Purpose

The Interactive Simulator lets the user select:

- start preset
- policy prompt
- training task/run
- checkpoint
- device
- number of chunks/frames

It runs a real policy rollout and appends the result to a timeline. Continue
should start from the previous rollout end state, not discard the old timeline.

### API

`GET /api/simulator/config` returns:

- `presets`
- `training_runs`
- `valid_mask_checkpoint`
- `default_output_root`
- `defaults`

`POST /api/simulator/run` launches:

```text
scripts/run_so101_qwen_closed_loop_eval.py
```

with loop artifact recording and media rendering enabled.

### Invariants

- `policy_n_action_steps` defaults to 15.
- User-facing `Frames / chunks` should control both the policy chunk count and
  maximum steps per primitive when that is the intended interaction.
- Continue must use the previous `continuation_start_report.json`.
- Timeline should accumulate across runs.
- Each inference segment should preserve prompt, input camera images, checkpoint,
  seed, episode, and frame metadata.
- Model execution may take time; show processing state while subprocess runs.

## Verification Checklist

After changing the dashboard:

```bash
python3 -m py_compile scripts/serve_so101_dataset_viewer.py scripts/serve_loop_test_analyzer.py
git diff --check -- scripts/serve_so101_dataset_viewer.py scripts/serve_loop_test_analyzer.py
curl -s http://127.0.0.1:8768/ | rg -n "Robot Experiment Manager|Platform|MyCobot"
curl -s http://127.0.0.1:8768/api/datasets | python3 -m json.tool >/dev/null
```

For UI changes, open or reload:

```text
http://127.0.0.1:8768/
```

Use Browser/in-app browser verification when the user asks to see the dashboard
or when visual layout/interaction is the actual deliverable.

