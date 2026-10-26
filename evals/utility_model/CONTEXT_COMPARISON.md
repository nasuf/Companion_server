# Classifier context-budget comparison

This screen is separate from the reply-quality release gate. It measures the two
hot-path judgement prompts `intent.unified` and `memory.relevance` against a
read-only production prompt snapshot. It does not publish prompts or modify a
runtime setting.

## Preregistered corpus and input arms

- Keep all 42 intent and 45 relevance cases from `tasks.py`. Their context is empty
  or at most one short turn. Report them as **existing label anchors**, not as
  multi-turn context coverage. One draw per arm is descriptive, not proof of a
  reproducible regression.
- Add 24 synthetic five-turn cases (12 per task), frozen before calls. Include old
  topic versus latest intent, merged bubbles, long assistant tails, 4–5-turn noun
  references, a newer noun replacing an old noun, and simple acknowledgement.
- For relevance, report expected noun grounding in `enhanced_query` separately
  from level accuracy. Returning the right level with an ungrounded query is not
  evidence that history was understood. Substring checks are conservative and
  can reject a valid paraphrase; retain raw outputs for review.
- `old`: frozen old formatter, 3 turns, 150 characters per role line, 900 total,
  no timestamps. `wide`: candidate renderer, 5 turns, 800/3000, timestamps.
  `tight`: the same candidate renderer, 5 turns, 150/1500, timestamps.
- `tight` keeps five-turn coverage while returning individual lines to their old
  cap. The extra 600 total characters permit those extra short turns plus date
  labels. The recommendation, if supported, is a **fixed classifier cap**, not a
  cap that keeps growing with the UI round count; N=20 must not make it 6000.

The comparison is end-to-end for the planned formatting change, so old versus new
also includes timestamps and the repaired bubble-merging boundaries. It does not
isolate the causal contribution of each formatting change. Main-reply history,
memory IO, proactive traffic and Web-search billing tiers are out of scope.

## Calls, scoring and failures

Freeze the manifest with `prepare` before `run`; it includes every rendered prompt
and SHA-256. Use production `dashscope:qwen3.5-flash`, temperature 0, non-thinking,
256 output tokens, at most 3 concurrent calls. Existing anchors use one draw per
arm; synthetic cases use three: **477 calls maximum**. Stable random order avoids
running all arms at different times. This does not constitute a replay-cache gate.

Report each task and corpus group separately: exact expected label, query noun
score, all-call denominator including errors, input tokens/max input, actual cache
field presence, full-response p50/p90 latency in seconds and estimated CNY. Never
remove a case because all arms fail. Reproducible label regression means old has
0/3 failures and candidate has >=2/3 failures. Prefer tight over wide if it avoids
that regression while preserving useful five-turn query grounding. Any concern
outside those screens needs inspection, not a claim of proof from 24 cases.

Transport, quota, rate-limit and API failures stop new dispatch; already in-flight
calls finish and all errors remain in JSONL. No automatic retries or overwrite of
evidence. No production DB, user data, notifications or environment mutation.

The listed utility price (CNY/million: input .2, cached .04, output 2.0) is the
existing evaluator's schedule, not an invoice claim. When provider cache usage is
missing, cost is an interval (all-hit to all-miss), not silently zero cache.
Credential values are supplied only through the runner process environment.

## Reproduction

```sh
python -m pytest evals/utility_model/test_context_comparison.py -q --noconftest
python -m evals.utility_model.context_comparison prepare \
  --snapshot /absolute/path/to/read-only-production-snapshot.json \
  --manifest /absolute/path/to/new-manifest.json
# DASHSCOPE_API_KEY must already be in process environment; do not echo it.
python -m evals.utility_model.context_comparison run \
  --manifest /absolute/path/to/new-manifest.json \
  --output /absolute/path/to/new-results.jsonl
python -m evals.utility_model.context_comparison summarize \
  --manifest /absolute/path/to/new-manifest.json \
  --input /absolute/path/to/new-results.jsonl \
  --output /absolute/path/to/summary.json
```

## Authorized rule clarification follow-up

A later explicitly authorized experiment retains every label and changes only the
candidate `memory.relevance` template to clarify the strong/medium boundary. The
original production prompt is always used by `old`; the candidate snapshot is
used only by `tight`. The tool rejects any change to `intent.unified`.

Use `prepare --candidate-snapshot ... --all-repeats 3 --variants old tight` and
`run --max-calls 666`: all 111 cases receive three calls per arm. Each substantive
candidate gets a new manifest/output path. Neither previous raw evidence nor its
latency failures are overwritten; there is no automatic retry-until-pass mode.
The output path is claimed exclusively before paid requests to prevent duplicate
concurrent launches. Full-task summaries must retain the original separate
single-utterance and multi-turn strata and disclose their different distributions.
