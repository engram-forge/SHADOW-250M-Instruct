# Anti-repetition evaluation: base vs tuned

Evaluated 2026-08-25 with 50 prompts from `anti_repetition_prompts.json`, 256 generated
tokens per prompt, and the released base model versus `anti_repeat.shdw`. The suite has 10
prompts in each category: long-form writing, explanation/summary, lists/code, repetition
stress, and legitimate-repetition controls.

## Aggregate results

| Profile | Model | All loop rate | Ordinary loop rate | Repeat 4-gram ratio | Mean tokens | Retry rate |
|---|---|---:|---:|---:|---:|---:|
| Greedy | Base | 66% | 70% | 47.51% | 229.8 | 0% |
| Greedy | Tuned | 16% | 10% | 18.54% | 155.1 | 0% |
| Sampled | Base | 0% | 0% | 6.93% | 191.4 | 0% |
| Sampled | Tuned | 6% | 7.5% | 0.39% | 129.7 | 0% |
| Guarded | Base | 0% | 0% | 6.93% | 191.4 | 0% |
| Guarded | Tuned | 0% | 0% | 0.33% | 122.6 | 6% |

On the 40 ordinary prompts, tuned greedy decoding reduces loop rate from 70% to 10%, an
85.7% relative reduction. Guarded decoding catches the three remaining sampled attractors
and retries them, producing a zero detected-loop rate.

## Greedy results by category

| Category | Base loop rate | Tuned loop rate | Base repeat 4-gram | Tuned repeat 4-gram |
|---|---:|---:|---:|---:|
| Long-form writing | 80% | 10% | 50.00% | 11.05% |
| Explanation/summary | 50% | 0% | 39.35% | 6.24% |
| Lists/code | 90% | 30% | 55.63% | 36.58% |
| Repetition stress | 60% | 0% | 50.80% | 6.40% |
| Legitimate repetition | 50% | 40% | 41.76% | 32.42% |

Lists/code remains the weakest category. The sampled tuned model entered three local
attractors on BFS pseudocode, SQL, and Wi-Fi troubleshooting prompts. These outputs also
contained substantive correctness problems independent of repetition. The runtime guard
retried all three.

## Interpretation limits

- Exact loop and n-gram metrics measure repetition, not factual or code correctness.
- A legitimate-repetition control that is not flagged may still be an instruction-following
  failure if the model omits the requested repeated structure. These controls require human
  or task-specific exact-match scoring before drawing compliance conclusions.
- Tuned outputs are shorter: guarded mean length is 122.6 tokens versus 191.4 for base.
  Completeness should be scored separately before selecting the final UL weight.
- Sampling results use one seed per prompt. Multi-seed evaluation is still required for a
  confidence interval.
