# Achievement Module Boundary

This package owns achievement definitions, counters, rule evaluation, unlocks,
and in-app notification records.

External application modules should only report domain events through
`app.services.achievements.service`:

- `handle_user_message_event`
- `handle_assistant_message_event`
- `handle_memory_changelog_event`
- `handle_intent_event`
- `handle_achievement_event`

The chat, memory, proactive, and intent systems must not import rule modules,
achievement ids, repository helpers, or unlock logic directly. They only know
that an event happened.

Internal layout:

- `events.py`: typed event payloads accepted by the achievement engine.
- `engine.py`: single dispatcher from public events to internal rules.
- `rules/`: private achievement rule implementations for realtime, intent,
  memory, assistant-message, and daily rollup achievements. This is the only
  layer where achievement ids, counters, and unlock conditions should be
  handled.
- `repository.py`: persistence helpers for events and unlocked achievements.
  `unlock_achievement` also owns the Redis unlocked-state cache. Cache keys are
  scoped by `user_id + agent_id` (`achievements:unlocked:{user_id}:{agent_id}`),
  so each user-agent relationship has an independent completion set. Redis is
  only an optimization: DB unique constraints remain the source of truth, Redis
  misses fall through to DB, DB conflicts backfill Redis, and Redis failures
  degrade to the DB path.
- `definitions.py`: achievement metadata loaded by UI/admin APIs.

## Persisted chat source boundary

Chat callbacks are references, not authoritative text, ownership or timestamps.
`source.resolve_message_event` binds user/assistant messages, intent events and
aggregation completions to an existing live conversation. The engine uses the
persisted message content, role, owner and UTC receipt time. Derived counters
inherit that source through a task-local context; duplicate evaluation may
continue rules after an earlier partial failure, while unique constraints keep
events and unlocks from multiplying. Database failures propagate to the existing
background-task failure handler instead of returning a duplicate result.

Assistant turn counts require explicit user-message links and the persisted
turn tag, with at most 100 messages per side and conversation scoping. There is
no callback-text or guessed-latest-message fallback inside the validated route.
Interaction-day callbacks use the persisted message ID in one guarded INSERT;
they do not reserve a Redis key before the database write or use callback time.

The admin source sample shows at most 20 recent primary message/intent/aggregation
events (API cap 50), with a sampling timestamp. `persisted_message` means a source
was bound and remains available; it does not prove an unlock, notification or
content truth. `legacy_unverified` preserves old records without guessing a link.
`source_missing` means the bound message is no longer available. The query uses
four index-backed bounded slices and never returns message text or raw metadata.

This does not add a durable background inbox or a notification outbox. A process
exit before a fire-and-forget callback runs can still lose its evaluation. Those
registration/recovery and delivery guarantees belong to roadmap R01.03.8.3.6;
SQL chat/checkpoint rollout remains independently gated. Daily and memory rules
keep their existing source contracts. Existing unlocks are not revoked when a
message is removed; new-contract counters stop using a removed source.
