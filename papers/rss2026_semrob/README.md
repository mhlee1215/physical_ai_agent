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

> Agentic Recovery for Lightweight Vision-Language-Action Policies

Core claim:

> A fixed lightweight VLA policy can be evaluated under an external recovery loop that detects failures, retries bounded attempts, and replans when needed, while final task success remains the benchmark environment success signal.

Do **not** claim:

- The wrapper universally fixes weak VLA policies.
- The verifier is the final success judge.
- The current simulator evidence proves real-robot success.
- The method trains a new VLA model.

## Evidence To Fill

- Policy-only baseline:
  - Current internal report: `docs/smolvla_libero_baseline_report.md`
  - Current best average: `87.75%` over 400 LIBERO episodes.
- Agentic retry probe:
  - Current internal report: `docs/smolvla_libero_agentic_retry_report.md`
  - Small 30-episode probe: baseline `50.00%`, success-once `70.00%`, recovery `6/15`.

The 30-episode retry probe is promising but too small for a strong final claim. Treat it as a pilot unless it is scaled or carefully framed.

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
