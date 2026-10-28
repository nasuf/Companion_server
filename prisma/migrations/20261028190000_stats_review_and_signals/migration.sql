-- Recomputable, text-free administrator statistics. No chat-path writes.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE user_daily_facts
  ADD COLUMN resend_60s INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN meta_comments INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN short_sessions INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN completed_sessions INTEGER NOT NULL DEFAULT 0;
CREATE TABLE stats_review_items (
  kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  local_date DATE NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  conversation_id TEXT,
  trace_id TEXT,
  occurred_at TIMESTAMP(3) NOT NULL,
  definition_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL,
  PRIMARY KEY(kind, source_id)
);
CREATE INDEX stats_review_items_day_kind_idx ON stats_review_items(local_date,kind,occurred_at DESC);
CREATE INDEX stats_review_items_user_idx ON stats_review_items(user_id,local_date);
COMMIT;
