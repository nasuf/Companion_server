# Context embedding evaluation (CHAT01.09.1–.2)

This suite records the original behavior before changing context windows or
prompt placement, then compares candidates against that original behavior.
A completed baseline is **not** a candidate quality pass. Preserve failed
responses and original reports.

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
because that was the scenario's initial route. The corresponding scenario
category is `current_state_requested`, not an assertion about the actual path.
Search enables the actual search
prompt branch, forwards its location parameters, and uses production's
nonstream Responses API settings.

## Measurements and cost

- Chat streaming TTFT measures the first nonempty content delta. Incomplete or
  length-truncated streams fail qualification.
- Production search is nonstream: TTFT is unavailable. As explicitly approved
  on 2026-10-26, this route uses p90 complete-response time (at most +10%).
  Streaming routes retain the p90 TTFT gate. Both require every call's valid
  latency sample; missing, negative or non-finite times cannot pass.
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

Run two independent baseline replays with `--judge-repeats 2` before enabling
critical-scene flip gates. Use the same source, corpus, prompts and model/judge
settings for both. Measure generation verdict flips separately from repeated
judgements of identical output; text variation alone is not a failure.

```sh
python -m evals.context_embedding.calibrate \
  --left /absolute/path/aa-a/report.json \
  --right /absolute/path/aa-b/report.json \
  --output /absolute/path/aa-calibration.json
```

Calibration reports noise and supplies a **pre-registered** allowance for other
categories; it never deletes failures or itself authorizes release. Preserve
malformed judge responses and any recovery replays separately rather than
silently rerunning until a pass appears.

Before new baseline/A/B replays, freeze `build_gate_policy(calibration)` and copy
that exact object into each report's `gate_policy`. The policy fingerprints the
whole calibration, including retained adverse recovery evidence. The comparator
rejects missing/changed policies. Do not retrofit this policy into old reports
and present them as new prospective evidence.

The approved 2026-10-26 noise rule uses each actual provider-route/reply-path's
generation-verdict flip rate. Same-output judge flips are diagnostic and **not
added** to generation flips, which already include judge variation. The existing
A/A measured: chat tier 8/96 (8.3333%), chat main 1/92 (1.0870%), search main
3/16 (18.75%), current-state short circuit 0/4 (0%). Main quality still has a
strict no-decline gate; its noise rate does not waive it. Other non-target,
non-main categories may decline by at most their own path's frozen A/A rate.
Missing or invalid actual-path calibration fails closed. If judge protocol
changes, rejudge the complete fixed corpora symmetrically and recalibrate before
freezing a new policy; preserve old versions.

`compare.py` matches conversation/sample/turn identities and bootstraps at the
conversation level with a fixed seed. Targets must be specified before running:

```sh
python -m evals.context_embedding.compare \
  --baseline /absolute/path/original/report.json \
  --candidate /absolute/path/candidate/report.json \
  --calibration /absolute/path/aa-calibration.json \
  --targets needs_5_turns,topic_interleave
```

It rejects missing judgements, duplicate or mismatched samples, price-unknown
calls, changed route populations, critical scene regressions, missing TTFT and
nonstream TTFT claims. Quality is reported by actual provider route, reply path
and category, with no unweighted all-path score. Main replies must not regress;
the designated tier targets must improve with 95% CI lower bound at least zero.
For each critical case/turn, exactly three repeat samples are required: a new
regression means candidate fails at least twice and the baseline fails zero
times. One noisy paired flip is retained but is not itself a critical regression.
Other categories use the fixed A/A allowance above. A five-turn
window is not expected to recover facts twelve turns back; keep those results
visible and evaluate longer configurations separately. Resource
checks use the original report, not a later inflated candidate. The historical
5,441-input and 34.7%-cache references remain explicit additional checks.

Every run must retain `previous_assistant_reply`: the actual previous whole AI
turn (merged bubbles), or explicit null when none exists. Absence is an evidence
failure, never a zero repetition count. Exact repetition compares text after
stripping only outer whitespace. High similarity is character-bigram **set
Dice** after Unicode NFKC, casefold and retaining Unicode letters/numbers; the
threshold is >=0.6. Fewer than two retained characters has similarity zero, but
can still be an exact repeat. Rates use only replies with a nonempty previous AI
turn; report excluded/no-previous counts separately. Every actual route must have
both rates no higher than baseline; the eligible population must match.

Current-state-short-circuit TTFT requires **at least 36 valid calls in each
comparison arm**, not 36 pooled across baseline and candidate. Calibration stores
raw state call identities and TTFT observations. A/A may supplement the baseline
only if source hashes, prompt snapshot, model/options, cache and prefix protocol
match; repeated call identities are deduplicated and conflicting duplicates
rejected. Every added observation enters the baseline p90 distribution, not just
its count. Candidate calls cannot be padded using unchanged-baseline A/A. Missing
latency or fewer than 36 samples remains an incomplete gate, not an exemption.

`select_variant(original, A, B, targets, calibration)` defaults to A. B is selected
only if its own complete gates pass, both repetition rates are no worse than A on
every path and at least one strictly improves, and each category's B-minus-A
quality difference is within its own path's frozen A/A allowance. The result
also reports `selected_passed`: selecting default A does **not** mean A passed.

Each conversation/sample replay has an independent request-prefix namespace,
retained across its consecutive turns. Cache gates use only the second and later
calls within that same replay; first-turn warming and repeated-case prewarming
are excluded. This controlled hit rate is not an estimate of production's rate;
retain the original production reference and do not claim production savings
from it. Offline prefix measurements cover the full wire-message list, including
role and all message fields: adjacent common-prefix bytes, equal leading messages
and complete-list equality. The first turn has no adjacent comparison. Comparing
only the first system message is insufficient.

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
