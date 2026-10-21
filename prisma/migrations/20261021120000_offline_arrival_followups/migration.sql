-- Only the new arrival service registers receipts inside its status transaction.
-- No trigger/backfill: old synchronous workers cannot double-register during rollout.
CREATE TABLE offline_arrival_followups (
  activity_id TEXT PRIMARY KEY REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE ON UPDATE RESTRICT,
  source_workspace_id TEXT,
  arrived_at TIMESTAMPTZ NOT NULL,
  scope JSONB NOT NULL,
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
CREATE INDEX offline_arrival_dispatch_idx ON offline_arrival_followups(next_dispatch_at,activity_id)
  WHERE status IN ('pending_dispatch','queued');
CREATE INDEX offline_arrival_recent_idx ON offline_arrival_followups(created_at DESC,activity_id DESC);
CREATE INDEX offline_arrival_status_recent_idx ON offline_arrival_followups(status,created_at DESC,activity_id DESC);
CREATE INDEX offline_arrival_retention_idx ON offline_arrival_followups(updated_at,activity_id)
  WHERE status IN ('completed','cancelled');
