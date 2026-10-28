BEGIN;
SET LOCAL lock_timeout='5s';
CREATE TABLE stats_proactive_daily (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  local_date DATE NOT NULL,
  trigger_type TEXT NOT NULL,
  data JSONB NOT NULL,
  definition_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL,
  PRIMARY KEY(user_id,local_date,trigger_type)
);
CREATE INDEX stats_proactive_daily_day_idx ON stats_proactive_daily(local_date);
COMMIT;
