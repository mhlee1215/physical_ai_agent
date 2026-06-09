# RSS SemRob 2026 Submission Checklist

## Venue Fit

- [ ] The paper directly addresses SemRob themes: semantic goal understanding, policy steering, failure monitoring, online correction, or test-time adaptation.
- [ ] The contribution is framed as pre-execution imagined chunk selection around a fixed lightweight VLA policy.
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
- [ ] Report random chunk, oracle imagined selector, and VLM imagined selector under matched settings or explicitly label unmatched settings.
- [ ] Report final success using benchmark `info["success"]` / `is_success`.
- [ ] Report VLM judge outputs only as candidate-selection signals, not final success.
- [ ] Report candidate count, committed action budget, total simulated/imagination budget, and wall-clock latency.
- [ ] Include failure categories if there is room.

## Tables and Figures

- [ ] Table 1: policy-only baseline vs external/reference baseline.
- [ ] Table 2: policy-only vs random chunk vs oracle imagined selector vs VLM imagined selector.
- [ ] Optional figure: imagine-then-act control loop.
- [ ] Optional figure: candidate predicted outcome image grid with selected chunk.

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
