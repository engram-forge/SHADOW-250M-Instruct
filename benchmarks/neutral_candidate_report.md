# Neutral candidate experiment

Source: OpenAssistant OASST1, Apache-2.0, pinned revision
`fdf72ae0827c1cda404aff25b6603abec9e3399b`. The deterministic preprocessing pipeline
selected accepted English paths, filtered unsafe, low-quality, synthetic, repetitive, and
Pirate-like content, and split by conversation-tree hash. The resulting local dataset contains
1,974 training and 209 tree-disjoint validation conversations.

Both candidates trained for 300 steps from the original master checkpoint with 10% recovery
perturbation.

| Metric | Base | Neutral UL 0.1 | Neutral UL 0.2 |
|---|---:|---:|---:|
| Held-out MLE loss | 3.2201 | 2.7800 | 2.7772 |
| Exact-format compliance | 20% | 25% | 20% |
| Pirate leakage | 0% | 0% | 0% |
| Qualification mean tokens | 142.1 | 145.6 | 142.1 |
| Ordinary greedy loop rate | 70% | 30% | 30% |
| Greedy repeat 4-gram ratio | 47.5% | 28.8% | 24.7% |

Both candidates pass the automatic guarded qualification gates but fail the separate greedy
loop target of at most 10%. Raising UL from 0.1 to 0.2 improves repetition density but not the
binary attractor rate and removes the compliance gain. Seed 0 with UL 0.1 is the current
neutral experimental leader, but it is not ready for promotion.

The next experiment should keep UL at 0.1 and add a broad, lexically diverse neutral recovery
corpus. It should cover long-form, code/list formatting, and stress prompts using source-held-out
templates. Additional random seeds should run only after this data change clears the loop gate.
