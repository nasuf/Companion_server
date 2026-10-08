# Runtime roles: explicit startup and deployment boundary

The current production compose explicitly selects `APP_RUNTIME_ROLE=integrated`.
It still runs two API workers, existing Redis consumers, timers and WebSocket
subscribers. This release does not turn on SQL chat ingress, SQL consumers,
checkpoints or a new production process topology.

| Entry | Role | Owns |
| --- | --- | --- |
| `app.main:app` | `integrated` | Existing API, scheduler and Redis job consumer |
| `app.main:app` | `api` | API/WS, configuration refresh and local Redis recovery; no business timers or job consumer |
| `jobs.runtime:app` | `scheduler` | Existing distributed timers, delayed-turn scan, configuration refresh and dependency probes; no initialization queue consumer |
| `jobs.runtime:app` | `background` | Shared initialization handler registry, one sequential Redis consumer loop, configuration refresh and dependency probes |

A role passed to the wrong entry point fails before dependency startup. Unknown
roles, including unfinished SQL foreground/delivery roles, cannot activate a
consumer. Independent entries use one Uvicorn process per container, with
`restart: unless-stopped` and a private, un-published `/health` endpoint. A failed
runtime loop exits the process; the container supervisor owns replacement.
Dependency loss returns HTTP 503 and prevents new background scans; recovery
restores readiness. Dependency probes have four-second deadlines. Health is
process/dependency readiness, not proof of success for every scheduled task.

The initialization pipeline is in `app.services.agent_initialization`, shared by
the API's existing local fallback and registered queue handler. Both consumer
entry points call `register_runtime_handlers()` before starting consumption;
importing a router no longer installs handlers. Its old no-delay recovery policy,
lock grace period, progress steps and missing/archived-agent behavior are retained.

The scheduler's registration function only creates definitions. API/CLI health
reports can therefore use actual triggers to detect stale/drifted jobs without
starting timers. One-time startup catch-up definitions are excluded from that
static inventory. Existing cron times and distributed locks remain unchanged.
The standalone scheduler still owns legacy delayed chat generation until the
SQL foreground ingress/handler adaptation is accepted in R01.03.8/R01.06.7.

## Capacity and production handover

Do not independently enable these entries alongside `integrated` production.
That would duplicate timer owners and change available execution capacity. The
role API is ready for isolated validation; production container activation remains
R01.06.8, jointly gated by R01.07 Redis handover and R01.08 capacity/restore checks.

For a topology of two API workers, one scheduler and one background worker, all
four processes must receive `LLM_PROCESS_COUNT=4` and `WEB_CONCURRENCY=2`.
`WEB_CONCURRENCY` remains the API worker count. The explicit total divides the
existing global provider quotas; independent scheduler/background calls always
use background quotas, including usage scopes not yet classified centrally.
Rounding retains the existing ceil/minimum-one behavior. This local partition is
not a distributed provider rate limiter and must be recomputed when topology
changes. DB pools must also be budgeted across every process: a dedicated entry
should use a conservative pool, rather than inherit the API's 12-connection pool.

Shutdown marks readiness false before cancelling loops. It preserves in-flight
Redis leases for normal recovery. A consumer ignoring cancellation exceeds a
ten-second drain deadline and the owning process exits, rather than starting
another consumer beside it. SQL primary jobs/delivery are not registered here;
`WorkerStopRequired` is propagated to process supervision, never retried locally.
Existing initializer business-effect idempotence and Redis-to-SQL handover remain
separate unfinished gates.

## Verification

`tests/test_runtime_roles.py` checks role ownership, wrong entry points, quota
partitioning, service registration, dependency failures/recovery, stale health,
API-only startup, supervision and cooperative/uncooperative shutdown.
`scripts/test_runtime_roles_e2e.py --image IMAGE --output DIRECTORY` creates owned,
internal PostgreSQL/Redis containers, applies all Prisma migrations, then runs the
packaged two-worker API and independent background/scheduler entries. It checks a
real registered handler, Redis outage recovery, SIGKILL replacement, SIGTERM drain
and disabled SQL consumers. It never loads production environment files or exposes
host ports. Existing authenticated chat, graph and worker replacement suites also
remain release gates.
