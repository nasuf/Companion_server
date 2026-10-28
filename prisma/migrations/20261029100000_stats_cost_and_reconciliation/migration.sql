BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Derived administrator data only. User erasure removes every owned detail.
CREATE TABLE stats_cost_daily (
  id TEXT PRIMARY KEY,
  local_date DATE NOT NULL,
  user_id TEXT REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  agent_id TEXT,
  category TEXT NOT NULL,
  scope TEXT NOT NULL,
  data JSONB NOT NULL,
  definition_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL
);
CREATE INDEX stats_cost_daily_day_idx ON stats_cost_daily(local_date,category);
CREATE INDEX stats_cost_daily_user_idx ON stats_cost_daily(user_id,local_date);
CREATE TABLE stats_token_reconciliation (
  trace_id TEXT PRIMARY KEY,
  local_date DATE NOT NULL,
  user_id TEXT REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  conversation_id TEXT,
  agent_id TEXT,
  scope TEXT NOT NULL,
  status TEXT NOT NULL,
  release_qualified BOOLEAN NOT NULL,
  data JSONB NOT NULL,
  definition_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL
);
CREATE INDEX stats_token_reconciliation_day_idx ON stats_token_reconciliation(local_date,status);
CREATE INDEX stats_token_reconciliation_user_idx ON stats_token_reconciliation(user_id,local_date);
COMMIT;
