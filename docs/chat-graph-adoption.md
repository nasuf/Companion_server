# Chat graph adoption — G01

Baseline: server `51a1501006b795d4daf1ec584fea4a588a571d47`.

The public `stream_chat_response` selects one executor for the entire logical
turn. The default is `legacy`. `CHAT_EXECUTOR=langgraph` additionally requires
an exact conversation ID in `CHAT_GRAPH_CONVERSATION_ALLOWLIST`; an empty list
selects legacy for every turn. Legacy recursive children remain in legacy.
Once graph execution starts, failure is surfaced without replaying legacy.
Production enablement remains gated on G02 paired model evaluations and G03
stable internal cohorts/rollback qualification.

G01 uses a real StateGraph with production phase adapters:

`load_turn → guard → pending → prepare_reads → identify → prepare_routes →
early_route → prepare_context → special_route → prepare_reply → generate_reply
→ normalize_reply → persist_reply → fragments → finish_turn`.

A guard or handler can short circuit to `fragments`. Pending fragments are
ordered by the existing intent priority and loop through the graph with a
forced intent; they do not recursively invoke the public stream. The parent
owns user persistence, usage, trace root, the final achievement, background
processing and the single `done` event. Per-message achievement/notification
hooks remain in the existing reply persistence layer.

The legacy implementation is retained as the migration reference and fallback.
`graph_phases.py` adapts the current production code without changing registry
prompts, model routing or intent rules. This temporarily duplicates orchestration
code. Subsequent domain changes must qualify both executors until adapters are
consolidated through later rollout stages; wholesale frozen-candidate merging
is prohibited. The source-reference tests explicitly inspect the legacy
implementation; graph behavior is checked with runtime tests and paired inputs.

Graph state contains only JSON decisions and diagnostics. Agent/service objects,
provider results and owned tasks live in invocation context. Each node declares
required inputs/outputs. Concurrent prefetch tasks have separate result handles;
classification and context preparation are joined explicitly. All owned tasks
are cancelled and awaited on failure/cancellation or fragment exit.

This is **not a durable execution contract**. There is no checkpointer, node
retry policy, resume API or checkpoint replay. `DURABLE_EXECUTION_READY=False`
is not configurable. R01/R02 must provide effect identities, business snapshots
and recovery qualification first. Reply persistence still uses the existing
non-transactional message saves; a failure can leave partial effects. A failed
turn is therefore never automatically re-executed.

Trace retains the existing root/run ID API. Graph/node/branch metadata uses
existing JSON fields. Raw HTTP model calls attach to the current graph node;
failed graph roots report an error. Old traces have no graph metadata and must
not be labelled as graph executions. Runtime context is excluded from trace
state inputs and client events. The client receives only the established reply,
delay and done protocol.

Relevant corrections qualified in this batch:

- Graph short-circuit replies await persistence before parent completion.
- A failed graph does not submit the success background/achievement hooks.
- Graph task cancellation waits for owned prefetch work to exit.
- Unbound short-circuit voice outputs are discarded on failed persistence.
- Mixed query/ordinary fragments retain the ordinary reply in AI memory;
  query-only turns still skip AI self-memory.
- Pure filler replies retain their existing proactive reason.
- Deployment transfers `runtime-constraints.txt` before building the image.

Tests use synthetic DB/domain IO, real StateGraph/LangChain callbacks and a
private Docker network for HTTP/Nginx/ticket/WebSocket/Redis flows. They do not
use production DB data or production credentials. Real model/cost/latency
qualification is a separate G02 gate. No deidentified production sample is
currently available. Passing these tests is not a claim of zero production risk.

Canonical task ledger and historical release records live in the user-facing
`companion-refactor-roadmap.json/html/csv` deliverables in this Codex thread.

Prompt changes follow the user's two-track rule. Audit the current production
text and complete version history before changing a registry key. Only keys
with no prior Web version update may change `defaults.py` and use `code_sync`.
Any prior Web save, restore or reset to default requires a new version through
the Web save path (`update_prompt_text` with `expected_updated_at`), even when
current text equals the default. Incomplete history uses the Web path. Validate
placeholders/rendering before publishing, preserve enabled state, and verify
DB, Redis, version provenance and the runtime getter afterward. Do not relabel
an existing `code_sync` revision as a Web save.

## G03: controlled activation and observation

The trace and read-only observation changes can be released independently while
keeping `CHAT_EXECUTOR=legacy` and the conversation allowlist empty. An internal
cohort and its observation/rollback gates are required to enable or expand graph
execution; they do not block deployment of these supporting changes. Graph
activation uses persistent deployment configuration rather than a manual `.env`
edit that would be lost on the next release.
The deployment now applies a private host configuration at
`/app/companion-secrets/chat-graph-rollout.json` after generating `.env`, before
building or stopping the server. Keep that file mode 0600 and outside Git.
Its exact shape is `{"executor":"langgraph","conversation_ids":["<UUID>"]}`.
The operator must verify every ID's account/workspace ownership before adding it.
Missing configuration selects legacy with an empty cohort. Invalid, wildcard,
empty graph or oversized cohorts abort deployment before server stop; only exact
conversation IDs can select graph. A new conversation requires a new authorized
cohort entry. The JSON file is not included in source synchronization or images.
To roll back, atomically replace it with
`{"executor":"legacy","conversation_ids":[]}`, apply
`scripts/chat_graph_deploy_config.py --env-file .env --config-file <host-file>`
and recreate only the server from the qualified retained image, without build,
pull or database migrations. Verify health and executor selection afterward.
Existing in-flight/failed turns must never be replayed. The private file remains
the source of rollout configuration for later deployments.

New chat roots explicitly record the selected executor, graph version when
applicable, and checkpoint-disabled flag in the existing JSON metadata. The
foreground session also records whether a usage row is expected. Historical
roots lacking these fields remain unmarked; absence of graph nodes alone does
not prove that a legacy turn was observed.

The read-only check requires exact, explicitly authorized conversation IDs:

    PYTHONPATH=/app python scripts/check_chat_graph_rollout.py \
      --conversation-id INTERNAL_CONVERSATION_ID \
      --expected-executor langgraph --expected-graph-version chat-g01-v1 \
      --since 2026-10-04T01:00:00+08:00 --until 2026-10-05T01:10:00+08:00

Use actual activation timestamps; repeat the conversation option for every
authorized internal conversation. End the window after late foreground writes
have settled. The report requires at least 24 hours, 20 traced turns, and a
sample from every specified conversation. These are telemetry minima, not
statistical proof of quality or performance. Zero traffic and missing evidence
never pass. Runtime cohort/version drift, failed or stale roots, missing
persisted replies, graph/finish-node inconsistencies and usage gaps hold the
gate. No model inputs, outputs, message text or exception messages are exported.

The collector opens a READ ONLY / REPEATABLE READ transaction, limits each
statement to five seconds, and reads at most 1000 roots over seven days for at
most 100 internal conversations. Truncation holds the gate. It has no Redis
writes, model invocations, scheduled work, notifications, activation or replay.
Exit 0 means telemetry is complete; exit 2 means hold; exit 3 means collection
failed. The report always keeps release_ready=false: controlled client smoke,
matched quality/cost/latency evidence, rollback and review remain separate gates.

Activation uses the existing startup environment configuration: set
CHAT_EXECUTOR=langgraph and the exact internal allowlist, deploy the qualified
artifact in the chosen production window, then verify actual marked traces.
Configuration edits do not hot-update existing workers. Reverting to legacy
requires applying startup configuration through the deployment mechanism and
may restart the container. Keep the previous artifact/configuration available.
Never rerun an in-flight or failed graph through legacy; its domain effects
may already exist. Do not describe this deployment path as zero downtime.

After controlled smoke and rollback qualification, observe the internal cohort
for at least 24 hours with actual traffic and the relevant scheduled jobs.
Record each later fixed cohort/coverage snapshot and its gates before expanding.
Any unresolved P0/P1, repeated failures, unexpected billing, unexplained trace,
or missing evidence stops expansion. R01/R02 still gate checkpoint replay.
