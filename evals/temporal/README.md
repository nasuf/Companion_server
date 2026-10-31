# Unified temporal evaluation

One entry point, three layers, preserved historical evidence:

```
python -m evals.temporal --inventory
python -m evals.temporal --suite contract --output-dir /fresh/truth
python -m evals.temporal --suite replay --output-dir /fresh/replay
python -m evals.temporal --suite binding --live --output-dir /fresh/model -- \
  --source-root /isolated/source --prompt-snapshot /private/prompts.json \
  --model MODEL --judge-model JUDGE --samples 3
```

1. **Truth**: contract calendar families and boundary cases. The original 240
   cases are 48 expression families, not 240 independent model conversations.
2. **Replay**: real graph/handler/rendering contracts with controlled IO, including
   reminder source clocks, expiry, evidence and explicit follow-up preferences.
   This is not production delivery or an actual model-quality verdict.
3. **Model**: dialogue, binding and incident runners retain their original input
   partitions, append/resume rules, native reports and judges. `--live` is required.
   Exit zero means collection finished; native quality review/aggregation is still
   required. No single averaged score can hide a failing or uncollected family.

All six legacy families are retained under this directory. `evidence/` contains
frozen counterexamples and acceptance criteria consumed by the existing harness,
not a seventh independent model cohort. `awareness` and `recall` include historical
DB-backed entry points; the unified isolated gate does not silently connect them
to production. They remain accessible via their documented explicit setup. Their
fixtures and unit evaluators remain covered in the release regression suite.

Old Python entry points forward to canonical implementations; old data paths are
relative symlinks to the same bytes. `corpus_manifest.json` pins every JSON/JSONL
file against the pre-migration hash. Old IDs, partitions and failures are not
rewritten. Historical source/harness hashes remain historical; new reports bind
the actual canonical source rather than hashing only forwarding files.

Outputs are fresh directories. Never overwrite a failed run, tune on held-out
outputs and call them blind, sum repeated scenarios, treat missing judgments as
passes, or describe migrated historical failures as newly qualified capability.
Model raw reports may contain private prompts: keep them outside Git and use
private permissions. EVAL-only provider credentials and the existing cost gates
apply; no production chat/DB write is needed for this consolidation.
