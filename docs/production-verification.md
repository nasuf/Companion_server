# Production verification and worker diagnosis

Run provider evaluations, migrations, workload/fault injection and graph/ORM
integration probes only in disposable local/CI databases and containers. Production
acceptance is read-only. Do not start concurrent app, Prisma, graph or eval CLI
processes inside the serving API container: those processes share its memory and
CPU budget even when the queries are read-only.

Production checks run serially with deadlines. Compare committed source hashes
using Python's standard library without importing application modules. Query the
database with native `psql` in the database container using a short `READ ONLY`
transaction and statement/lock timeouts. Collect only selected Docker state,
cgroup counters, worker PIDs/RSS and lifecycle metadata; do not print environment,
process command lines, configuration or raw application logs.

Check host `http://127.0.0.1:8000/health` separately from the public
`https://banshengcomp.com/api/health`. An observer's TLS handshake timeout is not
an HTTP error response. Preserve HTTP 502 samples during a deployment and measure
the outage; the current single-container deployment does not provide zero downtime.

## API worker failure events

The image launches `python -m app.api_server`. It delegates worker checks,
replacement, startup-failure handling and signals to pinned Uvicorn 0.54.0.
Two workers and the existing 60-second healthcheck budget remain the defaults.
The adapter refuses an unqualified Uvicorn version; upgrades must rerun the actual
image cold-import, replacement, heartbeat, startup-failure and shutdown tests.

The parent emits `worker_diagnostic` JSON only after a failed worker check and
after the failed worker is joined. Fields are bounded operational metadata:

- `event`: `api_worker_unhealthy` or `api_worker_failure_joined`.
- `worker_pid`, `healthcheck_timeout_seconds`.
- `reason`: `exited_before_replacement` if an exit code was already observed;
  `unresponsive_before_replacement` when it was still absent.
- `exitcode_before_replacement`, and `exitcode_after_join` for the second event.

An absent pre-replacement code followed by `-9` distinguishes the supervisor's
replacement path from an already-observed process exit. It does not prove an
underlying memory, CPU, GIL or provider cause. A kernel/cgroup OOM counter and
resource timeline are separate evidence. Normal shutdown and rolling signals
retain Uvicorn's behavior and do not generate these failure events.

## R01.03.8.2 incident

On 2026-10-08 at 09:53:39 UTC, worker 9 was replaced by worker 908. The original
logs lack a pre-kill exit code and healthcheck outcome, so its historical cause
cannot be established retroactively. Container restarts and cgroup/kernel OOM
events were zero. Heavy parallel read-only acceptance probes were an unsuitable
method; their causal connection to the exit is unproven. The method was replaced
with serial standard-library metadata and native read-only SQL. Preserve this
event in the roadmap and do not reset a container to erase its history.

The follow-up qualifies operational diagnosis; it does not enable SQL ingress,
change prompts/schema or claim that the historical exit has a proven cause.
