-- CHAT01.10.1: per-turn repetition scores and per-day scoring runs.
-- Recomputable, text-free statistics written by a nightly job; no chat-path writes.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
CREATE TABLE reply_repetition_scores (
  turn_message_id TEXT PRIMARY KEY,
  local_date DATE NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  conversation_id TEXT NOT NULL,
  trace_id TEXT,
  occurred_at TIMESTAMP(3) NOT NULL,
  situation TEXT NOT NULL,
  bubbles INTEGER NOT NULL,
  candidate_bubbles INTEGER NOT NULL,
  judged_bubbles INTEGER NOT NULL,
  max_similarity DOUBLE PRECISION,
  label TEXT NOT NULL,
  repeated BOOLEAN NOT NULL,
  scorer_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL
);
CREATE INDEX reply_repetition_scores_day_idx ON reply_repetition_scores(local_date, repeated);
CREATE INDEX reply_repetition_scores_user_idx ON reply_repetition_scores(user_id);
CREATE TABLE reply_repetition_runs (
  local_date DATE PRIMARY KEY,
  scorer_version TEXT NOT NULL,
  status TEXT NOT NULL,
  eligible_turns INTEGER NOT NULL,
  candidate_bubbles INTEGER NOT NULL,
  judged_bubbles INTEGER NOT NULL,
  failed_bubbles INTEGER NOT NULL,
  unjudgeable_bubbles INTEGER NOT NULL,
  truncated BOOLEAN NOT NULL,
  attempts INTEGER NOT NULL,
  started_at TIMESTAMPTZ(6) NOT NULL,
  completed_at TIMESTAMPTZ(6) NOT NULL
);
COMMIT;
