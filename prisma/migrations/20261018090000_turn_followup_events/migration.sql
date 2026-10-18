-- Explicit successful parent-turn completion only; never infer historical turns.
CREATE TABLE turn_followup_events (
  turn_id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE RESTRICT,
  user_message_ids JSONB NOT NULL CHECK (jsonb_typeof(user_message_ids)='array'
    AND jsonb_array_length(user_message_ids) BETWEEN 1 AND 100),
  scope JSONB NOT NULL,
  source_version TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending_dispatch'
    CHECK (status IN ('pending_dispatch','queued','failed','completed','cancelled')),
  generation INTEGER NOT NULL DEFAULT 0 CHECK (generation >= 0),
  job_id TEXT,
  dispatch_attempts INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_attempts >= 0),
  last_error TEXT,
  outcome TEXT,
  next_dispatch_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX turn_followup_dispatch_idx ON turn_followup_events(next_dispatch_at,turn_id)
  WHERE status IN ('pending_dispatch','queued');
CREATE INDEX turn_followup_recent_idx ON turn_followup_events(created_at DESC,turn_id DESC);
CREATE INDEX turn_followup_status_recent_idx ON turn_followup_events(status,created_at DESC,turn_id DESC);
CREATE INDEX turn_followup_retention_idx ON turn_followup_events(updated_at,turn_id)
  WHERE status IN ('completed','cancelled');
CREATE INDEX turn_followup_message_idx ON turn_followup_events(message_id);
-- The 101-row qualification cap must use a turn lookup, not scan chat history.
CREATE INDEX messages_achievement_turn_idx
  ON messages(conversation_id,(metadata->>'achievement_turn_id'),id)
  WHERE role='assistant';
