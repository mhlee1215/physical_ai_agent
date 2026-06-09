# RSS SemRob 2026 Submission Workspace

Target: **3rd Workshop on Semantic Reasoning and Goal Understanding in Robotics (SemRob), RSS 2026**.

## Source Rules

- Workshop site: <https://semrob.github.io/>
- Submission site: <https://openreview.net/group?id=roboticsfoundation.org/RSS/2026/Workshop/SemRob>
- Template source: RSS 2026 official LaTeX template from <https://roboticsconference.org/2026/information/authorinfo/>
- Local template archive: `paper-template-latex.tar.gz`
- Original template files kept in this directory:
  - `paper_template.tex`
  - `paper_template.pdf`
  - `IEEEtran.cls`

## Current Plan

- Submit a **4+N non-archival workshop paper** unless the final evidence clearly needs 8 pages.
- Keep the initial submission **double-blind**.
- Use `main.tex` as the working manuscript.
- Use `references.bib` as the working BibTeX database.
- Do not modify `IEEEtran.cls` or RSS formatting.

## Deadline

Official SemRob page currently lists:

- Submission deadline: **2026-06-18 23:59 AoE** (extended)
- Notification: **2026-06-24**
- Camera ready: **2026-07-01 23:59 AoE**
- Workshop: **2026-07-17**

## Paper Angle

Working title:

> Imagine-Then-Act Chunk Selection for Lightweight Vision-Language-Action Policies

Core claim:

> A fixed lightweight VLA policy can sample multiple candidate action chunks, imagine each candidate's visual outcome before execution, and execute the chunk whose predicted outcome best satisfies the current subgoal.

Do **not** claim:

- The selector universally fixes weak VLA policies.
- The verifier is the final success judge.
- The current simulator evidence proves real-robot success.
- The method trains a new VLA model.
- The VLM judge determines benchmark success.
- Oracle simulator-clone imagination is already deployable on a real robot.

## Evidence To Fill

- Policy-only baseline:
  - Current internal report: `docs/smolvla_libero_baseline_report.md`
  - Current best average: `87.75%` over 400 LIBERO episodes.
- Prior agentic retry evidence:
  - Current internal report: `docs/smolvla_libero_agentic_retry_report.md`
  - Use this as motivation/control context, not as the main novelty.
- Imagine-then-act evidence:
  - TODO: add simulator-clone candidate rollout, rendered outcome images, VLM/rule/oracle selector scores, and committed-environment success results.

The new main idea is pre-execution chunk selection, not post-failure retry. Keep retry-budget results as a baseline/control only if they help interpret cost and success.

## Related Work Buckets

When the delegated related-work search returns, sort papers into:

1. visual MPC / visual foresight / video prediction
2. world models and imagination-based planning
3. best-of-N / CEM / candidate trajectory selection
4. VLM-as-critic / reward / verifier in robotics
5. VLA and lightweight VLA background
6. agentic LLM concepts only where directly relevant

The final related-work story should make the method look like visual MPC / model-based candidate selection for lightweight VLAs, with VLM judging as a semantic scorer. Do not frame the VLM as the task success oracle.

## Authoring Safety

SemRob follows RSS double-blind review policy. RSS author information also includes an LLM-use policy. Treat `main.tex` as a structure scaffold, not final generated prose. Before submission:

- Rewrite all paper prose manually.
- Remove author-identifying paths, repository URLs, institution names, and acknowledgments.
- Do not include local artifact paths in the blinded submission.
- Keep artifact paths and repo links for camera-ready or supplement only if allowed.

## Build

From this directory:

```sh
make
```

Before uploading, check:

```sh
make fonts
```

RSS asks for embedded fonts and no page numbers.
