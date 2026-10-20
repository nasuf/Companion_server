# Temporal dialogue evidence qualification

This suite complements the 240 deterministic parser properties in `evals/temporal_contract`.
It contains **240 authored multi-turn scenarios**: 120 development and 120 held-out,
12 categories, 10 scenarios per category and partition. Each scenario has a fixed
criterion and a distinct family ID. Inputs and partitions were authored before candidate
outputs. The same author sees both partitions; this is not an independently authored or
blinded benchmark. Held-out responses must not be used for tuning the same version.

## What is exercised

The actual LangGraph reply executor, prompt rendering, time context, memory projection,
reply generation and normalization. The shared harness stubs domain IO, routing,
background jobs, location verification and real delivery. For non-gap cases the
`statement_memories_v1` adapter supplies the authored user statements as **synthetic
retrieved memories** with statement timestamps and unknown event dates. The two explicit
event fixtures retain their independently supplied occurrence dates. Gap cases use
history only. This separates reply reasoning from recall availability; it does **not**
prove extraction, actual retrieval recall, durable state, privacy authorization, deletion,
resume, delivery, or production E2E. Critical-topic samples test the reply only.

A first development pilot without the adapter confirmed that the three-hour active
history cutoff makes old statements unavailable. Its missing-context outputs must be
retained as harness calibration, not counted as a paired qualification. Automatic judge
misclassification is possible; retain original verdicts and any independently recorded
reviews. A review must never silently rewrite the original result or relax its criterion.

## Execution

Use isolated source checkouts without `.env`, an explicitly exported production prompt
snapshot, and provider credentials in environment variables. The actual reply uses the
same model, temperature `.7`, max output 700 tokens and prompt snapshot for both versions.
The optional reply guard always uses `qwen3.5-flash`; the separate grading model is an
explicit argument. No provider credentials are stored. Raw prompts/results are private
local evidence, not Git artifacts.

```sh
python evals/temporal_dialogue/run.py --source-root /isolated/source \
  --prompt-snapshot /private/prompts.json --output /private/holdout-0.jsonl \
  --model MODEL --judge-model qwen3.5-plus --split holdout --samples 5 \
  --shards 2 --shard 0 --live
# Run shard 1 in a separate process, never threads sharing monkeypatches.
python evals/temporal_dialogue/report.py /private/holdout-0.jsonl /private/holdout-1.jsonl \
  --output /private/summary.json
```

Full held-out schedule is **120 scenarios × 5 samples = 600 replies per version**.
Development runs use one sample. Run identity binds corpus, source, harness, prompts,
models, endpoints, and parameters. The append-only journal retains failures; resume
skips recorded samples, including failed answers. Concurrent writers are excluded.
Missing judge evidence is not a pass. A torn journal requires explicit preservation and
repair; it is never discarded automatically. Source/harness must remain unchanged and a
checksum-bound completion seal is required for aggregation. Exit 0 from the runner only
means collection completed; the report exits 0 only for full-schedule qualification.

## Reporting and release scope

Missing results count against the fixed denominator. Report ≥95% overall, ≥90% each
category, and zero critical nonpasses; disclose automatic-judge limitations. The 95%
interval resamples **scenario clusters**, not 600 supposedly independent samples. Report
reply/guard tokens and provider latency separately from judge calls. This is not a
production chat latency or cost SLO measurement. Broad qualification additionally requires
the roadmap's deterministic, real persistence/delivery, review and observation gates.

A failing full-system benchmark remains open under TIME03–TIME11 as applicable. Passing a
narrow read-path regression does not close TIME02 or certify all temporal reasoning.
For TIME02C, the production change only preserves recall when a date may describe a
mention/record instead of an event. It adds no write, migration, prompt version, model
call or client protocol. Its independent release evidence must include both executor
paths, real retrieval regressions, full isolated gates and Web desktop rendering; wider
benchmark failures must be disclosed, not treated as successful qualification.
