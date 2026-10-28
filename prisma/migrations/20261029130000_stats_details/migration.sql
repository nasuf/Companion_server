BEGIN;
SET LOCAL lock_timeout='5s';
CREATE TABLE stats_detail_daily (
  local_date DATE NOT NULL,
  kind TEXT NOT NULL,
  owner_key TEXT NOT NULL,
  user_id TEXT REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  data JSONB NOT NULL,
  definition_version TEXT NOT NULL,
  population_fingerprint TEXT NOT NULL DEFAULT '-',
  computed_at TIMESTAMPTZ(6) NOT NULL,
  PRIMARY KEY (local_date,kind,owner_key)
);
CREATE INDEX stats_detail_daily_user_idx ON stats_detail_daily(user_id,local_date);
COMMIT;
