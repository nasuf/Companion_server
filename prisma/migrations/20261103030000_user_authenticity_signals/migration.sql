-- User authenticity signals for the admin "真人判定" view.
-- Network: one row per (user, keyed IP hash); the raw address is never stored,
-- only its /24 prefix and offline region/ISP lookup. Input: one row per user
-- message with character count and compose timing; never draft text.
-- Both are cleared with the account (FK cascade) and pruned after 180 days.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
CREATE TABLE user_network_observations (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  ip_hash TEXT NOT NULL,
  ip_prefix TEXT NOT NULL,
  country TEXT,
  province TEXT,
  city TEXT,
  isp TEXT,
  network_kind TEXT NOT NULL,
  client_kind TEXT NOT NULL,
  user_agent TEXT,
  source TEXT NOT NULL,
  hits INTEGER NOT NULL DEFAULT 1,
  first_seen_at TIMESTAMP(3) NOT NULL,
  last_seen_at TIMESTAMP(3) NOT NULL,
  PRIMARY KEY (user_id, ip_hash),
  CONSTRAINT user_network_observations_kind_check
    CHECK (network_kind IN ('carrier', 'campus', 'datacenter', 'overseas', 'unknown')),
  CONSTRAINT user_network_observations_client_check
    CHECK (client_kind IN ('app', 'browser', 'other'))
);
CREATE INDEX user_network_observations_hash_idx ON user_network_observations(ip_hash, last_seen_at);
CREATE INDEX user_network_observations_seen_idx ON user_network_observations(last_seen_at);
CREATE TABLE user_input_signals (
  message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
  created_at TIMESTAMP(3) NOT NULL,
  chars INTEGER NOT NULL CHECK (chars >= 0),
  server_compose_ms INTEGER CHECK (server_compose_ms >= 0),
  client_compose_ms INTEGER CHECK (client_compose_ms >= 0),
  max_insert_chars INTEGER CHECK (max_insert_chars >= 0),
  edit_count INTEGER CHECK (edit_count >= 0),
  source TEXT NOT NULL CHECK (source IN ('live', 'nginx_backfill'))
);
CREATE INDEX user_input_signals_user_idx ON user_input_signals(user_id, created_at);
CREATE INDEX user_input_signals_created_idx ON user_input_signals(created_at);
COMMIT;
