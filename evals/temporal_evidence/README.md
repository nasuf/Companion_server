# Evidence use with ambiguous query dates

TIME02E freezes 24 new, authored synthetic cases: 10 known-evidence questions,
10 counterexamples, and 4 ordinary-chat/time-boundary controls. They exercise
negation, hypotheses, quotation, speaker attribution, cancelled plans, unknown
event dates, conflicting evidence, multiple candidates and irrelevant recall.
They are a bounded generalization check, not an external blind benchmark or
proof of complete temporal reasoning. The previous 12 binding cases remain a
separate regression set.

Reuse the actual reply executor and renderer with explicit source/prompt snapshots:

```sh
python evals/temporal_binding/run.py --cases evals/temporal_evidence/cases.json \
  --source-root /path/to/isolated/source --prompt-snapshot /path/to/prompts.json \
  --output /path/to/fresh-results.json --model MODEL --judge-model JUDGE \
  --samples 3 --live
```

Run both prompt variants against identical cases and settings. Never include an
`.env` in the isolated source. Domain IO and routing are synthetic; prompt
rendering, model inference and reply normalization are real. Reports retain
source/corpus/prompt/harness hashes, all replies and judge outcomes. A completed
run is not a quality pass. Preserve rejected candidates, invalid judgments and
raw scores; record any manual adjudication separately. Do not equate a cautious
confirmation question with a false assertion, but do flag wrong speakers and
unsupported event dates. Missing evidence and multiple matching records must
remain uncertain rather than forcing one recalled record into an answer.

The initial prompt-only TIME02E candidates did not qualify for release; at that
checkpoint the production runtime and prompt defaults remained unchanged. The candidate snapshots and all
failed experiments are retained in the dated local evidence directory, not
published as production fixes. These cases describe required behavior; their
presence does not mean the model currently passes them.

Trace retains the existing editable prompt component and temporal diagnostics.
Any future prompt change must follow the production key's complete history
(code defaults only if never Web-managed; otherwise the normal versioned Web
save path). Never relax evidence attribution just to improve answer rate.

## Bounded recall-answer candidate

The subsequent implementation uses the same selected memories and current turn
anchor. It computes calendar day/month/year offsets in code, then asks a model
for a short answer and local memory indices. A second call checks the answer
against the original candidates, recent corrections and character context.
Candidate indices are mechanically validated; neither their presence nor a
model verdict proves that the underlying memory is true. No memory is changed.
The existing Web-managed reply style and character rules remain inputs.

This route is limited to ordinary, tier-eligible recall questions with at most
20 memories / 6000 evidence characters. Timeline queries, long-gap reunions,
current/future state, counts, order, duration, clock-time questions, special
handlers and empty recall retain their existing paths. Missing anchors, disabled
prompts and oversized material do not activate it. Invalid answers, a rejected
check or service failure yield an explicitly uncertain short answer. Both calls
have bounded timeouts and no automatic retry/rewrite loop.

`holdout.json` contains 24 additional authored scenarios. After those results
were used during development, it ceased to be an untouched holdout.
`acceptance.json` contains 16 further scenarios frozen before the final paired
runs, including recent corrections and cancellations. Both are same-author
synthetic tests, not an independent or blinded external benchmark. Qualify with
three samples per case against the production baseline and prompt snapshot:
qualified recall quality >=95%, zero new critical factual/ownership/action
errors, and no changed routing of the controls. Preserve raw model judgments;
separate explicit review of faulty labels or judgments from raw scores. The
ambiguous phrase `在石桥修鞋补过登山鞋` cannot by itself prove whether `石桥修鞋`
is a shop name; cautious answers must not be described as factual hallucinations.
Failures in the unchanged paths remain open in the wider temporal roadmap.

The actual-flow harness exercises the real reply executor, prompt store,
calendar projection, model calls and normalization. It replaces domain IO,
route selection and the final persona verifier; it is not a production delivery
E2E. Offline executor and isolated database tests cover complementary failure,
persistence and delivery contracts. Web Trace browser cases exercise both valid
and malformed diagnostics without exposing raw memory content in the summary.

`release_cases.json` adds 8 later scenarios frozen before the final candidate
sampling (earlier corpora had already become development regressions). The
candidate additionally blocks common unsubstantiated negative event/history
claims before the model check: e.g. an incomplete recall set does not justify
“you never mentioned it” or “I never went there”. Explicit negative material
from matched memories or recent **user** messages is necessary, not sufficient;
the model must still check its subject and date. This heuristic is intentionally
conservative and is not a complete entailment algorithm. Style preferences are
passed as tone data, not as an additional source of events. Failed and aborted
experiments remain in the local evidence directory and do not count as passes.

`final_review_cases.json` freezes 8 additional authored cases after the final
code review fixes, before their first live sampling. They focus on event versus
mention dates, question versus negative evidence, cancellation and subject
ownership. They remain synthetic and do not replace natural production use.

The final deterministic safeguard also rejects “that day you told me” when the
associated event and statement dates do not establish a shared day. All 222
eligible live outputs were replayed through that final helper; two previously
accepted date-confusion answers became explicit uncertainty. This replay is
not counted as fresh live model sampling. The 276 paired live samples include
unchanged controls; final eligible answer-quality accounting keeps conservative
rejections and raw judge errors in the denominator. Full regression and direct
provider smoke are rerun against final code.
