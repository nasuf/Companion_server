# Regression release gates

This gate covers Server and the associated Web administration workflows. It is
not a claim of zero production bugs or complete LLM behaviour coverage.
Deployment interruption is outside this batch; its existing roadmap risks stay open.

## What must pass

1. **Full backend collection**, not a handwritten file allowlist. New
   `tests/test_*.py` files are included automatically. Collection, JUnit test
   identities, failures, skips, source/test/config hashes and the Git commit must
   agree. A fresh output directory is mandatory.
2. **Real disposable dependencies**: pgvector/PostgreSQL 16, Redis 7, and the
   complete Prisma migration chain. Missing dependencies may cause a developer
   test to skip, but the release gate rejects that skip. No production snapshot
   or provider credentials are required.
3. **Line and branch coverage**, measured separately across all `app` and `jobs`
   Python files, including files that were never imported. Initial floors are
   77% executable lines and 64% branch edges, based on the measured full-suite
   baseline rather than an invented 90% claim. Critical files have individual
   floors in `tests/quality/coverage-policy.json`. Changed executable lines in
   `app/` and `jobs/` require 90%; branch edges starting on changed lines require
   80%. No executable changes means N/A, not 100%. The changed-branch metric does
   not prove all paths through adjacent unchanged branches.
4. **Web browser CI**: pinned Playwright/Chromium executes the real Trace and
   prompt React components against controlled API fixtures. It covers desktop
   scrolling/completeness, raw Trace preservation, status/version labels,
   optimistic conflicts, restore/enable behaviour, and single-step replay/save.
   Web's deployment job depends on this workflow, including manual deployments.
   Memory-category consumer contracts are checked by both repositories;
   `tests/quality/web-taxonomy.json` and Web's matching fixture must change
   together. Locally, `COMPANION_WEB_UTILS=/qualified/web/src/utils.ts` also
   runs the Server comparison against real consumer source. Independent CI
   proves the versioned contract, not a live checkout of the other repository.
5. **Browser business integration**: `test_browser_prompt_postgres_e2e.py` drives
   a small HTML test form through actual HTTP, production JWT/admin routes,
   Prisma, PostgreSQL and Redis. It proves publication audit/version,
   persistence/cache/hot reads, stale-write rejection and administrator
   permissions. It does not substitute for the separately tested production
   React components, and does not call or evaluate a real model.
6. **Packaged runtime integrations** remain mandatory in CI: independent roles
   with SQL/Redis, worker diagnostics, authentication, reconnect and graph
   execution over HTTP/WebSocket/Redis. The graph transport harness controls
   model/business dependencies; it does not claim a complete real-model,
   real-database chat journey. Domain PostgreSQL tests separately cover memory,
   ingress/effects, execution scopes, jobs/outbox, replay and rollback.

## Explicit exclusions, never counted as passed

Two existing sibling-Flutter tests require a Flutter checkout/SDK and are outside
this Server/Web job. Their exact IDs and reasons are allowlisted and reported as
skipped. No broad marker/module skip is accepted. The Linux `/proc` resource test
may skip locally on macOS but MUST execute in Linux CI. New unexpected skips and
xfails fail the gate. Cross-repository production-UI/business E2E and real-model
quality evaluations remain separate roadmap work; this gate does not close them.

## Reproduce locally

Use a clean disposable checkout without a `.env` file (including in child
processes). Install the constrained dev dependencies and
`python -m playwright install chromium`.
Create disposable loopback PostgreSQL/Redis services, then export explicitly:

| Variable | Required resource |
|---|---|
| `PROACTIVE_E2E_DATABASE_URL` | `postgresql://…@127.0.0.1:PORT/companion_proactive_e2e` |
| `MEMORY_EVAL_TEST_DATABASE_URL` | `…/companion_memory_eval_ci` or `…/companion_memory_eval_gate` |
| `G03_TEST_DATABASE_URL`, `TTS_TEST_DATABASE_URL` | Disposable `…/postgres` scratch database |
| `MEMORY_EVAL_TEST_REDIS_URL` | `redis://127.0.0.1:PORT/13` |
| `PROACTIVE_E2E_REDIS_URL` | `redis://127.0.0.1:PORT/14` |
| `RUNTIME_JOB_TEST_REDIS_URL` | `redis://127.0.0.1:PORT/15` |

Apply Prisma migrations to the two dedicated databases after creating the
`extensions.vector` extension. `psql` must be available for the migration-lock
integration. The CI workflow contains the complete provisioning commands.

```sh
python scripts/run_regression_gate.py --output reports/regression-UNIQUE
python scripts/check_regression_gate.py --output reports/regression-UNIQUE --base BASE_COMMIT
```

The runner strips inherited provider credentials and disables the repository
`.env`, remote sockets and scheduled production side effects. It preserves
explicit temporary dotenv fixtures used by configuration tests. Loopback URLs
must identify resources created specifically for testing; do not point them at
SSH tunnels. Run the gate in its own process, never inside a production worker.

## Evidence and publication

CI retains JUnit, JSON/HTML coverage, collection/source hashes, tool versions,
skip reasons, gate summary, and browser traces/failure screenshots for seven days.
Artifacts contain synthetic fixtures and synthetic JWTs only. The summary is
bound to a candidate SHA; absent, failed or stale evidence is not acceptance.

Server production deployments (automatic or manual) must find a completed,
successful CI run for the exact current remote-main SHA. A failed or unfinished
rerun cannot reuse an older success. This is a deployment gate; repository branch
protection is a separate setting and is not implied by the presence of CI.

Before committing, complete both reviews required by AGENTS.md. Do not lower
thresholds, silently extend skip allowances, or restore obsolete business rules
to make tests green. Changes to the policy require rationale and review. The
full-suite audit found stale fixture assumptions around staged consolidation,
pure cumulative decay, DB singleton locks, scheduler registration, lazy FastAPI
routers and isolated provider state; these fixtures were aligned with the
current contracts while real-dependency assertions were retained.
