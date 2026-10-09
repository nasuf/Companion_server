# M02.01 memory evaluation baseline

This batch adds isolated evaluation assets. It does not change production
retrieval, recording, prompts, schemas, schedulers or chat delivery. Baseline
collection is never evidence that a proposed algorithm is ready to activate.

## Dataset and evidence

`cases.jsonl` contains 240 authored Chinese synthetic scenarios in 12 strata,
20 independent scenario families per stratum. `dataset.py` freezes a seeded
168/72 development/holdout split. The holdout may be run for final assessment;
it must not be used to tune the next candidate. Seeds and labels are deliberately
small fixtures: they are not representative production traffic or proof of
large-corpus performance. Do not report repeated model calls as new scenarios.

Memory identities include the owner side (`user:` or `ai:`). The real PG fixture
uses an active workspace and an archived previous companion workspace of the
same user, plus another user's workspace. It respects the one-active-workspace
constraint; it does not enable multiple active companions or simulate that
unsupported regular-user configuration.

Results distinguish candidates, selected prompt memories, and answer usage.
Fixture seeding leaves extraction `not_run`; `fixture_ids` allows retrieval
loss attribution without claiming recording was tested. Empty evidence and an
unexecuted stage have different meanings. Long item truncation, unrelated
injection and existing quality failures remain in the baseline report.

## Run

Validate without application imports, databases or models:

```sh
python -m evals.memory_baseline.run_eval --validate-only
```

Freeze a private, qualified, **native read-only** production policy snapshot
(`read_only=on`, `business_mutations=0`; prompts, models/prices, migration
checksums, image identity and observation time; no user chats):

```sh
python -m evals.memory_baseline.run_eval --freeze /private/policies.json --output /private/freeze
```

Create disposable PostgreSQL/pgvector and Redis services locally, apply the
existing Prisma migrations, then run:

```sh
python -m evals.memory_baseline.run_eval --replay \
  --manifest /private/freeze/manifest.json \
  --database-url postgresql://user:password@127.0.0.1:55441/companion_memory_eval_run \
  --redis-url redis://127.0.0.1:56391/14 --output /private/replay
python -m evals.memory_baseline.run_eval --adapters \
  --manifest /private/freeze/manifest.json --output /private/adapters
```

Replay runs **the same production implementation in both arms** with alternating
arm order, frozen vectors/clock, real scoped SQL, ranking/selection and Redis
cache/invalidation. It measures retrieval-only P50/P95 and actual raw SQL call
counts; embedding inference and fixture writes are outside timing. It does not
measure complete chat latency, DB CPU/IO cost or recording. Parallel scope checks
run separately from latency samples. DB errors swallowed by the production
retriever still invalidate the evaluation. Source drift requires a new freeze.

The model-only opt-in runner requires explicit `MEMORY_EVAL_CHAT_URL`,
`MEMORY_EVAL_CHAT_KEY`, `MEMORY_EVAL_JUDGE_URL`, `MEMORY_EVAL_JUDGE_KEY` environment
variables. No credentials are loaded from `.env`. Use the exact providers and
models in the freeze (currently Ark chat and DashScope utility); the runner uses
the frozen `memory.strong_reply` text and actual tier renderer. It does not
simulate guards, L1 permanent injection, history assembly or the full chat graph.

```sh
python -m evals.memory_baseline.live --manifest /private/freeze/manifest.json \
  --policies /private/policies.json --retrieval /private/replay/observations.jsonl \
  --pairs 5 --output /private/live
```

The 12 marked critical scenarios receive five matched pairs each (120 chat calls
plus semantic judge calls). Returned model identity, token receipts, frozen
registry prices, rendered prompt hashes and all failures are retained. Semantic
stale/persona/usage judgments are **diagnostic**, not an independent correctness
oracle. Deterministic persona anchors are reported separately. Voice drift,
external LongMemEval data, Mem0 OSS and Mem0 Platform are `not_run` unless explicitly
executed; no production data is uploaded. Adapter registration alone is not a
passed live evaluation.

## Isolation and release gates

Run each tool in a fresh process. Replay rejects non-literal-loopback URLs,
implicit/default DBs and production database names, suppresses dotenv loading,
clears external credentials, and fences Python sockets to loopback. The Prisma
child process receives only the validated local database URL; the Python socket
fence does not sandbox arbitrary child processes. Never run this tool with a
production tunnel bound to an evaluation URL. Sidecars must be operator-owned,
disposable services. Cleanup only removes this run's fixtures and Redis prefix;
it never flushes a Redis database or touches shared application keys.

The separate live process pins only the declared HTTPS provider destinations,
blocks PG/Redis/business HTTP and checks redirects. Raw synthetic replies and
policy snapshots belong outside the repository with restricted permissions.

`comparison.compare` applies `gates.json` to a complete paired candidate run:
stratified development/holdout intervals, scenario-clustered bootstrap,
no candidate hard failure/core regression, latency, matched live evidence and
model cost/risk checks. The target group must be declared before the optimization.
A self-comparison has zero improvement and **cannot pass the +5pp target gate**.
This infrastructure batch has its own correctness/integration/review gate; it
never claims those algorithm activation gates have passed.

For candidate versions, retain each source identity separately from shared
conditions (same data, model, budget, prompts, embedding digest, clock). Release
evidence must bind the measured source tree to the actual candidate commit.

CI runs bank/grader/safety tests and a fresh-process fixture test against a
migrated disposable PG and Redis. Real embedding/model evaluations remain opt-in
and are recorded separately from deterministic CI tests.
