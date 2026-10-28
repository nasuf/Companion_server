-- UXM01.02: versioned derived caches for beta statistics. Both tables can be
-- fully recomputed from raw data (messages, users, user_daily_activity,
-- user_feedback, llm_usage); app/services/effect/daily_facts.py owns them.
-- user_daily_facts holds counts per user x local day, never message text, and
-- cascades with the account. metrics_daily holds cohort aggregates only and has
-- no user identifier. Additive only; old images ignore both tables.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE user_daily_facts (
  user_id TEXT NOT NULL,
  local_date DATE NOT NULL,
  opened BOOLEAN NOT NULL DEFAULT FALSE,
  chatted BOOLEAN NOT NULL DEFAULT FALSE,
  user_messages INTEGER NOT NULL DEFAULT 0,
  turns INTEGER NOT NULL DEFAULT 0,
  sessions INTEGER NOT NULL DEFAULT 0,
  bursts INTEGER NOT NULL DEFAULT 0,
  unreplied_bursts INTEGER NOT NULL DEFAULT 0,
  first_message_at TIMESTAMP(3),
  spontaneous BOOLEAN,
  proactive_received INTEGER NOT NULL DEFAULT 0,
  proactive_answered INTEGER NOT NULL DEFAULT 0,
  feedback_count INTEGER NOT NULL DEFAULT 0,
  cost_cny DOUBLE PRECISION NOT NULL DEFAULT 0,
  settled BOOLEAN NOT NULL DEFAULT FALSE,
  definition_version TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL,
  CONSTRAINT user_daily_facts_pkey PRIMARY KEY (user_id, local_date),
  CONSTRAINT user_daily_facts_user_id_fkey FOREIGN KEY (user_id)
    REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX user_daily_facts_local_date_idx ON user_daily_facts (local_date);
-- No erasure_owner_active trigger here: its FOR SHARE lock on users would block every
-- profile/role/status UPDATE for the whole computation. The FK takes KEY SHARE, which
-- only waits for account deletion and SELECT ... FOR UPDATE (erasure start) for the few
-- milliseconds a day takes; writers skip status='deleting'; rows cascade on delete.

CREATE TABLE metrics_daily (
  local_date DATE NOT NULL,
  metric TEXT NOT NULL,
  cohort TEXT NOT NULL,
  dimension TEXT NOT NULL DEFAULT '',
  dimension_value TEXT NOT NULL DEFAULT '',
  numerator DOUBLE PRECISION,
  denominator DOUBLE PRECISION,
  value DOUBLE PRECISION,
  sample_size INTEGER,
  settled BOOLEAN NOT NULL,
  definition_version TEXT NOT NULL,
  population_fingerprint TEXT NOT NULL,
  computed_at TIMESTAMPTZ(6) NOT NULL,
  CONSTRAINT metrics_daily_pkey PRIMARY KEY (local_date, metric, cohort, dimension, dimension_value),
  CONSTRAINT metrics_daily_cohort_check CHECK (cohort IN ('external', 'all'))
);
COMMIT;
