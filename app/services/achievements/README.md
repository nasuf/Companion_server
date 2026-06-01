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
- `definitions.py`: achievement metadata loaded by UI/admin APIs.
