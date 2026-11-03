BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Additive configuration for chat/reply_repetition; NULL keeps the shipped defaults.
ALTER TABLE system_config ADD COLUMN reply_repetition_policy JSONB;
ALTER TABLE system_config ADD CONSTRAINT system_config_reply_repetition_policy_object
  CHECK (reply_repetition_policy IS NULL OR jsonb_typeof(reply_repetition_policy) = 'object');
CREATE TABLE reply_repetition_policy_changes (
  id TEXT PRIMARY KEY,
  admin_id TEXT,
  changes JSONB NOT NULL,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX reply_repetition_policy_changes_created_at_idx
  ON reply_repetition_policy_changes (created_at DESC);
-- Existing system_config update trigger bumps configuration_revisions; no new scheduler.
COMMIT;
