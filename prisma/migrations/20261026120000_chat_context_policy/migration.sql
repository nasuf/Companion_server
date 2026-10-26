BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Additive configuration; no chat or memory rows are rewritten.
ALTER TABLE system_config ADD COLUMN chat_context_policy JSONB;
ALTER TABLE system_config ADD CONSTRAINT system_config_chat_context_policy_object
  CHECK (chat_context_policy IS NULL OR jsonb_typeof(chat_context_policy) = 'object');
CREATE TABLE chat_context_policy_changes (
  id TEXT PRIMARY KEY,
  admin_id TEXT,
  changes JSONB NOT NULL,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX chat_context_policy_changes_created_at_idx
  ON chat_context_policy_changes (created_at DESC);
-- Existing system_config update trigger bumps configuration_revisions; no new scheduler.
COMMIT;
