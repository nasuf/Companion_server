# Context embedding baseline (CHAT01.09.1)

This suite records the original behavior before changing context windows or
prompt placement. A completed baseline is **not** a quality pass or permission
to enable CHAT01.09.2. Preserve failed responses and original reports.

## Scope and reproduction

`cases.json` contains 104 synthetic conversations: eight per category (the eleven
plan categories, low-information replies, and web search). Each scenario runs
two sequential evaluated turns, with the first actual answer appended to the
history before generating the second. Repeat three times by default. The corpus
is reproducible from `build_cases.py` and contains no production user content.

Use an isolated checkout without `.env`, an explicitly supplied read-only
production prompt snapshot (`key`, `content`, `is_enabled`), and provider keys
in the process environment. Do not pass production database/Redis credentials.

```sh
python -m evals.context_embedding.run_eval \
  --source-root "$PWD" --prompt-snapshot /absolute/path/prompts.json \
  --output /absolute/path/baseline/report.json --repeats 3 --live
```

The evaluator refuses to overwrite evidence, records original provider usage,
hashes prompts/corpus/source files, and writes after each turn. Synthetic IO,
fixed time/location, and forced routing isolate reply rendering. The actual
renderer, normalizer, location loading and model are exercised. Network requests
go to the configured Ark and DashScope endpoints; product delivery, real DB,
Redis and classification are not exercised. This is not a WebSocket E2E.

The four reply paths are main, tier, current-state short circuit and search.
Actual diagnostic paths are reported separately: a follow-up to a short circuit
may legitimately fall back to tier. Do not label it as a short circuit merely
because that was the scenario's initial route. Search enables the actual search
prompt branch, forwards its location parameters, and uses production's
nonstream Responses API settings.

## Measurements and cost

- Chat streaming TTFT measures the first nonempty content delta. Incomplete or
  length-truncated streams fail qualification.
- Production search is nonstream: TTFT is unavailable. Report completion time
  separately; it cannot be substituted silently for TTFT in a release gate.
- Each actual model/API/reply path reports its maximum input, count above
  24,000, count above 32,000, input/output/cache usage, cost and latency samples.
  Any request above the conservative 32,000-input guard is **price unknown** and
  fails resource qualification. Do not average it away.
- The original Ark model price is 0.8 / 0.16 / 2.0 CNY per million input/cached
  input/output tokens. It was present in production since July 18; this suite
  does not change that registry or claim a pricing-fix saving.
- Search cost uses provider `usage.tool_usage.web_search`, charged at the
  normalized 0.004 CNY per search. Baseline and candidate use the same basis.
  The monthly 20,000 free-search quota affects the separate invoice estimate,
  not optimization claims. A single reply can perform multiple searches.
- The product-cost report covers reply-generation calls. Judge and temporal
  verification costs, intent/relevance and other mocked calls are outside this
  metric; it is not the total cost of an end-to-end chat turn. The CLI's cost cap
  has the same scope and can overshoot by one bounded request.

### Cache-zero evidence

The sanitized fixture contains actual 2026-10-26 no-hit provider usage from
nonstream Chat Completions, streaming Chat Completions and Responses search.
Each explicitly reports `cached_tokens=0`. The former Trace parser discarded a
normalized `cache_read=0` via boolean `or`; the accompanying runtime fix preserves
it without inventing a value when both fields are absent.

For historical Ark traces, `recover_ark_trace_cache` can read the exact original
usage payload. It does not backfill production, extrapolate unknown incomplete
calls, or establish a default for other providers. Missing original evidence
stays unknown, while the recovered measured zero stays zero.

## Quality comparison and boundaries

`compare.py` matches conversation/sample/turn identities and bootstraps at the
conversation level with a fixed seed. Targets must be specified before running:

```sh
python -m evals.context_embedding.compare \
  --baseline /absolute/path/original/report.json \
  --candidate /absolute/path/candidate/report.json \
  --targets needs_5_turns,answer_question
```

It rejects missing judgements, duplicate or mismatched samples, price-unknown
calls, changed route populations, critical paired regressions, missing TTFT and
nonstream TTFT claims. Target mean quality must improve with 95% CI lower bound
at least zero; other categories may not regress more than two points. Resource
checks use the original report, not a later inflated candidate. The historical
5,441-input and 34.7%-cache references remain explicit additional checks.

Repeated identical synthetic cases warm provider caches. Their observed hit
rate is not an estimate of production's rate. Compare equally warmed sequences
and retain the original production reference; do not claim production savings
from this synthetic cache hit rate.

The original reused single-answer judge was overly strict on natural short
replies. The dedicated instruction and default qwen3.5-plus judge in `judge.py`
were calibrated on 14 manually reviewed examples; all original answers were
then rejudged, retaining both sets of verdicts. This selected calibration set
does not prove independent judge accuracy. Reports fingerprint the judge and
product model options; the comparator rejects protocol changes. Keep raw reasons
and review false positives; do not manually convert failures to passes in a
machine gate. Blinded candidate preference, deterministic semantic
checks, classification accuracy, broader safety/persona regressions, client E2E
and post-release observation are additional gates before behavior changes. The
baseline alone does not prove those have passed.

## Offline tests

```sh
python -m pytest -q evals/context_embedding tests/test_local_tracer.py \
  tests/test_temporal_awareness_eval.py
```

No production prompt, model routing, context window, historical price or
database schema is modified by this suite.
