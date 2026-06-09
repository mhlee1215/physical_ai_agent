# Physical AI Harness Team Spec

## Goal

Make the agentic physical AI workflow repeatable. Every implementation checkpoint must include an executable verification step before the work is considered complete.

## Inputs

- Project plan: `docs/agentic_physical_ai_plan.md`
- Simulation config: `configs/sim/libero.yaml`
- Current smoke command: `PYTHONPATH=src python3 -B -m unittest tests.test_checkpoint_25 tests.test_real_so100_checkpoint_26_gate`

## Outputs

- Source changes under `src/physical_ai_agent/`
- Tests under `tests/`
- Checkpoint evidence in the final response or `_workspace/` when a long run needs preserved artifacts

## Roles

| Role | Responsibility | Reusable skill | Writes |
| --- | --- | --- | --- |
| Orchestrator | Select the next checkpoint and enforce validation | `.agents/skills/physical-ai-orchestrator/SKILL.md` | final response or `_workspace/*` |
| Implementer | Add code, config, and tests for the checkpoint | n/a | repo files |
| Verifier | Run required commands and report pass/fail/blockers | n/a | command evidence |

## Phase Order

### Phase 1: Select Checkpoint

- input sources: `docs/agentic_physical_ai_plan.md`
- actions: choose the smallest unchecked checkpoint that advances the MVP
- output files: none required
- completion criteria: target checkpoint and verification command are known

### Phase 2: Implement

- input sources: selected checkpoint, existing package scaffold
- actions: add focused code, config, docs, and tests
- output files: repo-local source and test files
- completion criteria: code path is runnable without hidden setup when possible

### Phase 3: Verify

- input sources: implementation output
- actions: run the required checkpoint command and relevant tests
- output files: command output summarized in final response
- completion criteria: verification passes, or blocker is explicit and reproducible

## Retired Checkpoint Gates

CP01-CP24 were completed milestone gates and are no longer part of the active verification surface. Do not add new work that depends on deleted `scripts/checkpoint_*.sh` commands or retired `physical_ai_agent.checkpoints.checkpoint_01` through `checkpoint_24` modules. Use the maintained evaluation, CP25 RoboCasa, and CP26 real SO-100 paths below.

## Failure Policy

- retry policy: fix local code/config failures and rerun verification once
- partial completion policy: scaffold smoke can pass while Mac-local MuJoCo smoke is blocked by missing external dependencies
- conflict resolution policy: the Mac-local MuJoCo smoke is authoritative for claiming checkpoint 01 works on the target Mac; the LIBERO strict gate is authoritative only for claiming full LIBERO execution
- escalation trigger: request permission before installing or downloading simulation dependencies

## RunPod Orchestration

Use RunPod as the Linux/NVIDIA execution lane for LIBERO, LeRobot/SmolVLA,
ManiSkill GPU rendering, Isaac, and paper-comparable evaluation runs.

Maintain `docs/runpod_worklog.md` as the durable handoff journal for cloud work.
Update it whenever a RunPod evaluation, setup fix, result fetch, or lifecycle
decision changes the state a future conversation needs to recover. Do not
include API keys, private SSH keys, full `.env` contents, or raw secret-bearing
API responses.

Default workflow:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_pod.sh start
sh scripts/runpod_pod.sh status
RUNPOD_SSH="$RUNPOD_SSH" sh scripts/runpod_check.sh
```

Then, on the Pod:

```bash
cd /workspace/physical-ai/physical_ai_agent
git fetch origin
git checkout main
git pull --ff-only origin main
```

Run heavy evaluation under `/workspace` only. Keep environments, model caches,
datasets, logs, videos, and benchmark outputs under `/workspace`, not `/`.

Before stopping the Pod, consolidate outputs under a network-volume result
directory and fetch them back to the local repo:

```bash
cd /workspace/physical-ai/physical_ai_agent
mkdir -p _workspace/runpod_results
cp -R _workspace/checkpoints _workspace/runpod_results/checkpoints
cp -R outputs _workspace/runpod_results/outputs 2>/dev/null || true
```

Write a concise Markdown report under `_workspace/runpod_results/` with:

- git commit evaluated on RunPod
- exact command line
- policy checkpoint/model id
- benchmark suite and episode count
- success-rate table
- blocker list, if any
- artifact paths for logs, videos, and JSON metrics

Fetch results locally before stopping:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_fetch_results.sh
```

For long-running baseline debugging, archive completed result directories
locally and delete those completed remote directories before the network volume
fills. Preserve the active run explicitly:

```bash
set -a
. ./.env
set +a
RUNPOD_ACTIVE_RESULT_DIR=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/<active-run> \
  sh scripts/runpod_archive_results.sh --delete-remote --yes-delete
```

Do not delete model caches or LIBERO assets during an active parity-debugging
loop unless the volume is blocked; deleting them usually costs more time than it
saves space.

Then stop the Pod:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_pod.sh stop
sh scripts/runpod_pod.sh status
```

RunPod API responses can include sensitive env values. `scripts/runpod_pod.sh`
redacts API response fields by default; do not set `RUNPOD_RAW_RESPONSE=1`
unless raw JSON is explicitly needed for debugging.

For paper-comparable numbers, only count runs from a committed Git revision
that has been pulled on RunPod. Ad hoc `rsync` or uncommitted remote edits may
be used for quick debugging, but they are not acceptable evidence for a
reported benchmark table.

## Retired Oracle Overlay Workflow

The CP24-era oracle overlay workflow is retired from the active checkpoint surface. Historical reports may remain under `docs/research/`, but new validation should use maintained benchmark evaluation entrypoints or the CP26 real SO-100 gate rather than CP24 overlay wrappers.

## Validation Checks

- `sh scripts/checkpoint_25.sh`
- `sh scripts/checkpoint_25.sh --probe-reset-step --require-robocasa --task CloseFridge`
- `sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2 --max-steps 1`
- `sh scripts/eval_smolvla_libero_mac.sh`
- `LIBERO_TASKS=libero_spatial LIBERO_N_EPISODES=1 sh scripts/eval_smolvla_libero_linux.sh`
- `LIBERO_TASKS=libero_spatial LIBERO_TASK_IDS='[0,1,2]' LIBERO_N_EPISODES=5 sh scripts/eval_smolvla_libero_linux.sh`
- `METAWORLD_TASKS=easy METAWORLD_N_EPISODES=1 sh scripts/eval_smolvla_metaworld_linux.sh`
- `sh scripts/eval_smolvla_lerobot_linux.sh --benchmark libero --agentic-layer baseline`
- `sh scripts/eval_smolvla_lerobot_linux.sh --benchmark metaworld --agentic-layer baseline`
- `sh scripts/runpod_smolvla_libero_eval.sh`
- `PYTHONPATH=src python3 -m physical_ai_agent.evaluation.lerobot_eval --benchmark libero --output-dir /tmp/libero --print-command`
- `PYTHONPATH=src python3 -m physical_ai_agent.evaluation.lerobot_eval --benchmark metaworld --output-dir /tmp/metaworld --n-action-steps 15 --print-command`
- `PYTHONPATH=src python3 -B -m unittest tests.test_real_so100_contract`
- `PYTHONPATH=src /Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest discover -s tests`
- `PYTHONPATH=src python3 -B -m pytest` when pytest is available
- `PYTHONPATH=src python3 -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text()) for p in files]; print(f'parsed {len(files)} files')"` as a no-dependency fallback syntax check


## Planned Checkpoint 26 Verification

Checkpoint 26 will register the real SO-100 pilot. The first gate should use a calibrated `so100_follower` arm with two arm-mounted OpenCV cameras as policy inputs and the MacBook built-in webcam as a separate Codex/operator observer channel. The smoke gate must capture all camera streams, save a manifest with camera indexes, feature keys, frame sizes, calibration file paths, and observer-frame artifacts, then run planner/verifier logic without sending robot actions. The strict gate may execute physical actions only after emergency stop, joint limits, action rate limits, safety clipping, and human confirmation are wired and evidenced. Any paper claim from CP26 must distinguish quantitative policy results from qualitative Codex observer-assisted demonstrations.

## AI & Coding-Agent Disclosure Policy for Papers

Purpose:

- Keep publication artifacts reproducible and transparent.
- Prevent policy drift between simulation runs, benchmark claims, and manuscript text.
- Ensure top-tier journal expectations are met for LLM/coding-agent usage.

Mandatory rules for any manuscript draft in this repo:

- Never list AI tools as authors.
- Always disclose AI contributions in a stable manuscript section (`Methods`, `Acknowledgements`, `Author contributions`, or a dedicated `AI Disclosure` section).
- Record at least:
  - tool name + model/version,
  - date used,
  - task type (e.g., drafting, proofreading, code assist, experiment interpretation, figure caption drafting),
  - scope/amount (entire draft vs. partial section, prompt count, or file list),
  - whether outputs were independently verified.
- Keep a human-in-the-loop trace:
  - all AI outputs must be reviewed and edited by the authors;
  - generated code must be executed, and reported metrics/visuals linked as evidence.

Top-tier examples to mirror (Nature portfolio examples already observed):

- Nature Communications 2025: `Acknowledgements` disclosure of ChatGPT-based assistance and explicit statement that core scientific conclusions were written by authors.
- Nature Communications 2025 (another article): script assistance with explicit tool/version/task disclosure in `Acknowledgements`.
- Communications Biology 2026: explicit `Acknowledgements` disclosure of ChatGPT/Perplexity/Elicit for idea generation and literature scan.
- Scientific Reports 2024: explicit AI-use section with model/version and non-authorship statement.

Example templates (adapt per journal):

- `We used [Tool] ([model/version], [date]) for [task], and then manually reviewed, edited, and finalized all outputs before submission.`
- `No text or claims in the manuscript were copied directly from model output without human verification; all figures and code were executed/validated, and evidence is reported in _workspace/checkpoints.`
- `No AI system is listed as an author.`

Venue alignment:

- For journal-specific policies (Nature/Elsevier/IEEE/OUP/MDPI/ACM), place disclosure in the required location and mirror wording in final manuscript evidence notes.

Required checkpoint-local record:

- Add a short disclosure note to the checkpoint evidence bundle whenever AI or coding-agent outputs were used during manuscript preparation.
- Add the same note to final response artifacts when publishing claims or benchmark tables are produced.
