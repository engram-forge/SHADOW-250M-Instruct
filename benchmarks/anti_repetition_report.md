# Anti-repetition model evaluation

This report compares the released SHADOW 250M Instruct model with the final retained
`finetune/anti_repeat_model` candidate. Intermediate and rejected experiment results are
intentionally omitted.

## Model and training

The candidate was trained for 300 steps from the original master checkpoint with:

- 1,974 filtered English OASST1 conversations (90% of the training mixture);
- 219 Pirate-style conversations (10%);
- OASST1 revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, Apache-2.0;
- 209 OASST1 validation conversations, split by conversation tree with no tree overlap;
- repetition-completion unlikelihood loss with alpha 0.2;
- corrected recovery perturbations on 20% of sampled examples;
- quantization active throughout training.

Recovery corruption repeats an early answer span two to four times with zero loss, then starts
supervision at a later non-overlapping continuation. It never rewards another copy of the
repeated span.

## Final qualification

| Metric | Released base | Retained candidate | Gate | Result |
|---|---:|---:|---:|---|
| OASST1 fixed-batch MLE loss | 3.1626 | 2.7658 | no material regression | Pass |
| OASST1 perplexity | 23.63 | 15.89 | lower is better | Pass |
| Exact-format compliance | 20% | 21% | no more than 5 points below base | Pass |
| Pirate-style leakage | 0% | 0% | below 1% | Pass |
| Qualification mean tokens | 142.1 | 135.3 | 70–120% of base | Pass |
| Ordinary greedy loop rate | 70% | 7.5% | at most 10% | Pass |
| Greedy repeat 4-gram ratio | 47.5% | 14.8% | lower is better | Pass |
| Guarded loop rate | 0% | 0% | 0% | Pass |
| Export round-trip maximum error | — | 0.0 | 0.0 | Pass |

The OASST1 loss comparison uses the same 52,813 assistant target tokens, fixed packing seed,
pure assistant-token MLE, and trainer-equivalent quantization path for both checkpoints.

## Greedy repetition results

The 50-prompt suite contains ten prompts in each category. The first four categories are
ordinary tasks; legitimate-repetition prompts are reported separately and excluded from the
ordinary aggregate. Generation is greedy with a limit of 256 tokens.

| Category | Base loop rate | Candidate loop rate | Base repeat 4-gram | Candidate repeat 4-gram |
|---|---:|---:|---:|---:|
| Long-form writing | 80% | 0% | 50.0% | 13.3% |
| Explanation/summary | 50% | 0% | 39.4% | 6.9% |
| Lists/code | 90% | 20% | 55.6% | 19.1% |
| Repetition stress | 60% | 10% | 50.8% | 13.9% |
| Legitimate repetition | 50% | 40% | 41.8% | 21.0% |
| **Ordinary aggregate** | **70%** | **7.5%** | — | — |

Lists and code remain the weakest category. Repetition metrics do not measure factual or code
correctness, and legitimate-repetition prompts require exact task-specific scoring to distinguish
correct refrains from unwanted loops.

## Artifacts and reproduction

Retained local artifacts:

```text
finetune/anti_repeat_model/finetuned.pt
finetune/anti_repeat_model/anti_repeat.shdw
finetune/anti_repeat_model/audit.json
finetune/anti_repeat_model/eval50_greedy.json
finetune/anti_repeat_model/release_qualification.json
finetune/anti_repeat_model/blind_review.jsonl
```

SHA-256:

```text
82fb96c5599bdf3af83acfd356a6fb0da34365774693ff163f90d1b0b2211fed  finetuned.pt
4ca240ec9bdaafcedafc23709e440d252d68b1d98c990977b82f685e27084d63  anti_repeat.shdw
```

Use `anti_repetition_eval.py` with `anti_repetition_prompts.json` for repetition evaluation,
`release_qualification.py` for deterministic format/style gates and blinded review generation,
and `finetune/evaluate_loss.py` for checkpoint MLE comparison. Generated model outputs and
checkpoints are ignored by Git.
