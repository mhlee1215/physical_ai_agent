# Manuscript TODO Checklist

Date: 2026-06-10

This file holds missing items and claim-boundary checks that should not appear
inside the manuscript prose.

## Citations

- [ ] Confirm final citation metadata for SmolVLA.
- [ ] Confirm final citation metadata for FAST.
- [ ] Confirm final citation metadata for pi0.
- [ ] Add the closest VLM-as-critic / VLM-as-reward robotics citations once
  the related-work thread selects them.
- [ ] Add paper-ready citations for prompt portfolios, test-time adaptation,
  dropout, or ensembles in robot policy/VLA diversity if used in final claims.
- [ ] Decide how to cite Gemma 3 4B if it appears in the VLM comparison.

## Experiments

- [ ] Add classified candidate-generation metrics for template prompt
  portfolios.
- [ ] Add classified external VLM subgoal-generation metrics.
- [ ] Add classified simulator-based candidate-selection results.
- [ ] Add state restoration validation if used to support simulator-clone
  selection.
- [ ] Add end-to-end LIBERO benchmark success results.
- [ ] Add action budget, candidate count, wall-clock cost, and environment
  reset accounting for every comparison.

## Figures

- [x] Insert current overview figure as `figures/overview.png`.
- [ ] Recreate the overview as a publication-style vector figure if time
  permits.
- [ ] Add a candidate outcome grid figure if selection experiments produce
  paper-ready images.
- [ ] Add a compact evaluation-axis diagram or table if space permits.

## Claim Boundaries

- [ ] Candidate diversity is not candidate selection.
- [ ] Candidate selection is not benchmark success.
- [ ] VLM scores are not final task success.
- [ ] Simulator-clone selection is an oracle/proxy unless replaced by a
  deployable model.
- [ ] The frozen lightweight VLA remains the only execution policy.
- [ ] No fine-tuning claim unless a fine-tuning experiment is added.
- [ ] No success-rate improvement claim without classified benchmark evidence.

## Current TODO Counts

- Citations: 6
- Experiments: 6
- Figures: 3 open, 1 done
- Claim boundaries: 7
