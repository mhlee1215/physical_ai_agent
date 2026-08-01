# myCobot 280 Main Policy-Progression Figure

**Prepared:** 2026-08-01 PDT

The main PR figure adds the unfine-tuned base policy to the three paired
deterministic-versus-randomized training-seed results under fresh randomized
physics.

- The base point is one shared, fixed policy evaluated on 11 matched unfiltered
  candidates: 0/11 strict and 0/11 pickup-and-hold.
- Each colored line is one independently fine-tuned seed. Dotted segments show
  the shared base reference to deterministic fine-tuning; solid segments show
  the paired deterministic-to-randomized comparison.
- Deterministic strict successes are 3/11, 1/11, and 0/11.
- Randomized strict successes are 5/11, 5/11, and 4/11.
- Randomized training is higher in 3/3 paired training seeds. The exact
  two-sided sign test remains `p=0.25` because the independent training unit is
  `n=3`.

The three dotted segments do not represent three baseline models. They connect
the same shared base reference to each independently trained deterministic
checkpoint for visual comparison.

## Reproduction

```bash
python scripts/render_mycobot280_policy_progression_figure.py \
  --claim-summary docs/research/2026_08_01/mycobot280_policy_claims_summary.json \
  --output-figure docs/research/2026_08_01/mycobot280_policy_progression_main.png
```
