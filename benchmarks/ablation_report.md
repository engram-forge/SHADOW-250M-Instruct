# Anti-repetition staged ablation

All generation gates use the same 50-prompt suite, 256-token greedy decoding, and seed 0.
Run A is the 150-step recovery-plus-3-gram-UL model.

| Run | Change from Run A | Ordinary loops | Long-form loops | Lists/code loops | Stress loops | Mean tokens | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| A | Champion baseline | 10% | 10% | 30% | 0% | 155.1 | Keep |
| B | Restart base; 180 duplicated structured examples; 50 steps | 52.5% | 80% | 40% | 50% | 163.8 | Reject |
| B2 | Continue A; 36 structured examples; LR 3e-6; 50 steps | 12.5% | 0% | 40% | 0% | 151.2 | Reject |
| C | Continue A; 40 targeted recovery examples; LR 1e-6; 50 steps | 10% | 0% | 20% | 20% | 159.9 | Reject |

## Findings

- Repeated template variants in Run B caused severe distribution shift and memorization.
- A small structured mix in B2 fixed the known long-form loop but did not improve aggregate
  behavior and worsened lists/code.
- Targeted recovery in C improved both named weak categories, but displaced failures into the
  repetition-stress and legitimate-repetition groups. It is useful evidence, not a promotable
  checkpoint.
- Run A remains the champion because no candidate improves the target categories without a
  new category regression.

## Revised next experiment

Do not stack low-order UL onto B2 or C. Build a broader recovery corpus with diverse lexical
forms and held-out templates, then train from Run A at LR 5e-7 to 1e-6. Separate validation
by source before training: original chat, structured tasks, recovery tasks, and ordinary
anti-loop prompts. Promote only when every ordinary category is non-regressing. Low-order UL
should then be tested as an independent branch from Run A, with explicit exact-format scoring
for tables, JSON, refrains, and repeated constants.
