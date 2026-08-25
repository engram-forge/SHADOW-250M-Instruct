# Release qualification: anti-repeat candidate

The released base and Pirate anti-repeat candidate were evaluated with guarded generation on
100 deterministic cases. Generated output and the blinded review packet remain local under
the ignored candidate directory.

| Gate | Base | Candidate | Candidate result |
|---|---:|---:|---|
| Exact-format compliance | 20% | 4% | Fail |
| Guarded loop rate | 0% | 0% | Pass |
| Pirate-style leakage | 0% | 72% | Fail |
| Mean output length | 142.1 | 86.7 (61% of base) | Fail |

Automatic promotion decision: **fail**. The candidate passes only the loop gate. It must not
replace either released deployment weights or the original fine-tuning master checkpoint.

The low absolute format scores also show that the 250M base has limited strict instruction
following. Promotion should use relative non-regression gates, but a future neutral model needs
dedicated structured-output training to improve the absolute baseline.

## Required next candidate

- Train from the original master checkpoint, not the Pirate candidate.
- Use neutral, licensed instruction data with source-separated validation.
- Retain anti-repetition UL/recovery but remove Pirate transformations.
- Run this suite, the 50-prompt repetition suite, public capability benchmarks, export
  round-trip validation, and blinded completeness review before promotion.
