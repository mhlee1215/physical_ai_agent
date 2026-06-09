# RSS SemRob 2026 Submission Checklist

## Venue Fit

- [ ] The paper directly addresses SemRob themes: semantic goal understanding, policy steering, failure monitoring, online correction, or test-time adaptation.
- [ ] The contribution is framed as a recovery/evaluation layer around a fixed lightweight VLA policy.
- [ ] Claims are bounded to simulator evidence unless real-robot evidence is added.

## Format

- [ ] Use the RSS official LaTeX template.
- [ ] Do not edit `IEEEtran.cls` or spacing/font/page geometry.
- [ ] Keep the main submission PDF anonymous.
- [ ] Remove author names, affiliations, acknowledgments, institution-identifying phrases, repository links, and local machine paths.
- [ ] Add the OpenReview Paper-ID to the title/author line after assignment.
- [ ] Include a `Limitations` section.

## Experiments

- [ ] Freeze policy checkpoint.
- [ ] Freeze action budget.
- [ ] Freeze seeds and task ordering.
- [ ] Report policy-only baseline.
- [ ] Report retry/recovery condition under matched settings or explicitly label unmatched settings.
- [ ] Report final success using benchmark `info["success"]` / `is_success`.
- [ ] Report wrapper verifier outputs only as control signals, not final success.
- [ ] Include failure categories if there is room.

## Tables and Figures

- [ ] Table 1: policy-only baseline vs external/reference baseline.
- [ ] Table 2: policy-only vs agentic retry/replan ablation.
- [ ] Optional figure: wrapper control loop.
- [ ] Optional figure: failure/recovery trace.

## Submission

- [ ] Confirm final SemRob deadline on <https://semrob.github.io/>.
- [ ] Confirm OpenReview profile and conflict domains.
- [ ] Upload PDF to <https://openreview.net/group?id=roboticsfoundation.org/RSS/2026/Workshop/SemRob>.
- [ ] Save submitted PDF and OpenReview metadata under `_workspace` or another non-paper artifact folder.

## Camera Ready

- [ ] Restore author names and affiliations only after acceptance.
- [ ] Add acknowledgments if appropriate.
- [ ] Add repository/artifact links if allowed.
- [ ] Add AI/coding-agent disclosure according to `docs/harness/physical-ai/team-spec.md`.
