# Benchmarks

`results.json` holds every number on the model card; `report.pdf` is the full report.
The evaluation archives are held out and not distributed.

`run.py` is the harness that produced the numbers. You can point it at your own archive
(a folder with a `tokens.u32` stream and a `bank_valid.jsonl` question bank in the same
format as `results.json` describes) to run the same evaluation on your own data.

`release_qualification.py` runs 100 deterministic instruction and exact-format checks against
deployed `.shdw` files and produces a machine-readable promotion decision plus a blinded human
review packet. It covers JSON, Python syntax, Markdown tables, numbered lists, compound
constraints, neutral style, repetition, output length, and pirate-style leakage. Example:

    python benchmarks/release_qualification.py \
      --model base=deployment/shadow250m_instruct.shdw \
      --model candidate=path/to/candidate.shdw \
      --out results/qualification.json --review-out results/blind_review.jsonl

These generative checks are not substitutes for canonical PIQA or ARC likelihood scoring.

`anti_repetition_eval.py` evaluates deployed models on the balanced 50-prompt suite in
`anti_repetition_prompts.json`. The final retained anti-repetition model results, training
recipe, qualification gates, and artifact hashes are consolidated in
`anti_repetition_report.md`; intermediate experiment reports are not retained.
