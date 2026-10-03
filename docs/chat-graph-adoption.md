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
