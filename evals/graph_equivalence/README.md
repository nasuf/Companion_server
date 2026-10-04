# G02 paired graph qualification

This runner compares the released legacy executor with the opt-in LangGraph
executor. It runs the actual intent/relevance classifiers, reply generator, prompt renderer
and reply-emotion model against the existing `reply_register` case bank. Domain
reads, account/profile/history data, persistence, background jobs, payments and
notifications use synthetic fixtures. It does not enable the production graph.

Run only in a separate, disposable process with the candidate application code
and its constrained dependencies. Never import the runner into an application
worker. A live run requires an explicit opt-in and consumes model-provider quota
only after case-evidence preflight passes:

```sh
python -m evals.graph_equivalence.run_eval \
  --read-only-production-snapshot \
  --paired-classifiers \
  --samples 5 \
  --output /tmp/g02-paired.json
```

The snapshot transaction is `REPEATABLE READ, READ ONLY` and loads only global
configuration, per-agent configuration overrides, prompt templates and model
prices. It reads prompt Redis keys to detect cache/DB discrepancies, then closes
both connections. It never loads production conversations, profiles or messages.
Per-agent overrides or stale prompt caches abort qualification instead of being
silently ignored or repaired. Production credentials remain in the process;
reports contain prompt hashes and synthetic content, not secrets or prompt text.

After the snapshot, a fail-closed fence permits only the configured model-provider
HTTPS origins on port 443. Provider DNS addresses are pinned for the run. Every
HTTP hop, including redirects, is checked; sockets to PostgreSQL, Redis and other
business services are blocked. Application DB/Redis roots are also denied. This
is a test-process guard, not a general-purpose sandbox for untrusted code.

Each case receives at least five samples per executor at fixed sequential load.
Pair order alternates to reduce cache/order bias. Both sides use the same
synthetic history, clock, profile, empty memory results, prompt versions and model
configuration. Production fast relevance gates and relevance-model routing are
retained. User-emotion context and web search are disabled in this synthetic scope.
Artificial reply sleeps and business background work are excluded. The existing
judge must first pass all its calibration cases; its quality thresholds are not
relaxed. A passing full-bank report additionally requires:

- Every expected case/sample/executor row, with no failed turn.
- A parsed judge verdict for every turn; an ungraded response cannot disappear
  from the quality denominator and still qualify.
- Equal logical model input sequences, excluding the reply-emotion input which
  depends on the stochastic generated text. Repeated adjacent retry requests are
  compared once within their role. Independent intent/relevance requests may start
  in either order; sequence order is preserved within each role. All attempts and
  model errors remain recorded.
- No detected persona leak and no network-fence violation, including recovered
  violations.
- Known nonzero model prices and no more than 10% increase in P95 turn latency,
  mean normalized model cost, or mean model call count.
- Passing existing format and quality gates for both executors.

Actual cached-input billing is reported separately. The cost gate uses the same
uncached-input price for both sides, since the second member of a pair can benefit
from provider prompt caching. `--case-limit`, a subset selected by repeatable
`--case-id`, and fewer than five samples are diagnostic runs; they cannot qualify
the full bank. Reports are written atomically after
each row so interrupted or partial runs remain inspectable and cannot pass.

This qualification complements the deterministic behavior matrix in
`tests/test_chat_graph_equivalence.py` and the real HTTP/WebSocket/Redis integration
harness. It does not qualify real retrieval, crisis detection, account-specific
personas, payments, notifications, checkpoint replay, or production concurrency.
Those require their own regression evidence and the G03 controlled production
observation. A failed gate keeps graph activation closed even when CI tests pass.

Schema v4 also records parsed intent/relevance decisions and reply paths to
diagnose unequal downstream inputs. Independently sampled classifiers can choose
different tiers even with equal classifier inputs. The report retains those
mismatches as failed gates; decision traces are evidence, not a waiver. The
full-bank controlled-classifier tests compare the actual rendered tier prompts
under the same decisions without model calls. They complement, and do not replace,
real-model quality qualification.

`--paired-classifiers` makes the migration comparison reproducible: each pair
consumes the first executor's parsed intent/relevance results on both sides.
Both sides still invoke the real classifiers and include their actual latency,
tokens and call counts. Independent and consumed decisions are recorded
separately; the second result is used only for diagnosis. The first executor
alternates across cases and samples. Changed inputs, extra or missing classifier
calls fail closed, including errors caught by application fallbacks. Results are
copied so an executor cannot change the other side's decision state.

Unused speculative relevance reads can finish or be cancelled before an ending
reply, depending on provider timing. Cancellations are recorded explicitly and
matched to their original input/call position; they are not missing calls. This
exception is limited to `memory_relevance` when neither reply path consumes its
context. A cancelled classifier required by an ordinary reply, cancelled intent,
changed input, extra/missing invocation or unequal actual provider request sequence
still fails qualification. If the source relevance read was cancelled, a completed
counterpart result may remain independent only on that unused path; equal response
inputs must prove it did not change the reply. The call-count gate counts actual
model starts, including cancelled attempts, rather than only completed usage rows.
Reported-token costs remain estimates; cancelled provider calls can have billing
not returned to the client, and their count is disclosed separately.

This is a controlled comparison of executors given the same model decisions,
not an end-to-end test of independent classifier sampling and not a checkpoint
replay. Omitting the option retains independent sampling for diagnosis; unequal
inputs still fail its gates. Reports from different modes or source/prompt
versions cannot be stitched together. Quality and performance thresholds remain
the same in both modes, and a passing subset is never full-bank qualification.

Schema v5 introduced an evidence gate before database/cache reads or model calls.
Every original false-premise case declares the synthetic persona fact or delivery
evidence required to disprove its claim. Unknown family/travel/preferences remain
unknown. A disabled photo feature is not a delivery record, and a voice message
is not a telephone call. The existing judge's text-only assumption is explicitly
checked against the released speech-output capability; rubric wording and quality
thresholds have not been changed. A missing or conflicting premise blocks the
entire selected bank, rather than dropping those cases from its denominator.

Run the offline audit without production credentials or provider quota:

```sh
python -m evals.graph_equivalence.run_eval \
  --preflight-only --output /tmp/g02-preconditions.json
```

An invalid preflight returns exit code 2 and preserves all selected IDs and
issues. A valid subset preflight is only an audit, never release qualification.
The historical v5 audit blocked all 12 false-premise cases: nine lacked declared
scenario facts, and the legacy judge's text-only assumption conflicted with speech
output. The three known age/city/occupation facts also failed the empty-memory
tier input check. Those failures are retained as evidence; they are not retroactively
converted into passing results.

At runtime, authoritative statements must reach both response and judge inputs.
Statements found only in user claims or generated replies are not trusted facts;
a dropped template parameter or an uninformed response attempt blocks grading.
Reports keep issue codes and hashes rather than raw reference or prompt text.
Older reports without this evidence cannot qualify. The original v2–v5
reports remain historical evidence, including their actual quality failures.
Judge parsing also fails closed: valid JSON or an exact bare label is required.
Malformed JSON, unknown verdicts and labels mentioned only in explanations do
not become grades. Existing verdict categories and thresholds are preserved.
An invalid judge output may be retried up to three attempts with the identical
frozen prompt. Every attempt's input/output hash and parsed verdict is retained;
a valid negative grade is final and is never retried to obtain a favorable grade.
After three unparsed outputs the row remains ungraded and blocks qualification.
Provider and network-fence exceptions fail the row immediately. Judge attempts
are separate evaluation work and do not enter production turn cost/latency.

Schema v6 uses the frozen `g02-grounded-v1` synthetic scenario. The nine previously
unsupported cases receive one declared earlier assistant turn containing the
relevant travel/family/preference/role or delivery fact. Original case IDs, user
messages and history entries remain intact; both executors see the same added
turn. No witness is added to out-of-window cases. A negative delivery fact comes
from this explicitly complete toy-world ledger, never from absent production
history or an assumption that media/calls/orders are unavailable. These added
premises make v6 a new qualification context; its results cannot be described as
an unchanged-context improvement over the older ungrounded false-premise scores.

The grounded false-premise judge receives the same authoritative fact explicitly.
It retains the four verdicts and existing numerical gates, while removing the
unsupported text-only capability assumption. Fictional character consistency
does not require denying the real product's AI identity. Legacy reply-register
callers retain their previous rubric unless they supply grounded facts. All 14
calibration samples and expected labels are unchanged; the two Antarctic control
samples receive their declared scene fact when this runner calibrates the judge.
The response model and judge must both receive the authoritative statement on
every attempt. Missing evidence, including evidence dropped by a template, still
blocks qualification.

The candidate reply fix reuses the main path's managed identity/personality section
in tier replies, including known age/job/city without new business reads or model
calls. It respects the section's admin enabled flag and keeps the main prompt lazy.
The tier already receives the shared Web reply policy through `get_prompt_text`;
no duplicate prefix is added. Four uncustomized tier defaults clarify known facts,
missing history and emotional acknowledgment. Customized shared Web policies are
preserved byte for byte.

For pre-release real-model runs, `prospective_prompt_snapshot` simulates these four
default updates only in process memory. A changed customized row, any unreviewed
default update or a missing tier row aborts. The report records the source and
candidate prompt hashes and row timestamps. Nothing is saved to production DB or
Redis. Evaluate the exact candidate application code in a disposable process;
baseline-only worker imports do not qualify candidate changes. Release must
revalidate the baseline and prompt customization immediately before deployment.
