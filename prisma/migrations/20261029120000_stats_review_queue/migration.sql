BEGIN;
SET LOCAL lock_timeout='5s';
CREATE TABLE stats_review_queue (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE ON UPDATE CASCADE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  source TEXT NOT NULL,
  counts JSONB NOT NULL,
  definition_version TEXT NOT NULL,
  created_by TEXT REFERENCES users(id) ON DELETE SET NULL ON UPDATE CASCADE,
  created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','reviewed','dismissed'))
);
CREATE INDEX stats_review_queue_status_idx ON stats_review_queue(status,created_at DESC);
COMMIT;
