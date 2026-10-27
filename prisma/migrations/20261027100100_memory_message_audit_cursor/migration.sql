-- The first versioned batch defines admission (not historical backfill permission).
-- Attaching discovery to that batch reuses its scoped erasure cascade without a
-- second conversation/side lifecycle. Deleting it intentionally resets discovery
-- to the next surviving admission floor; durable source receipts remain the
-- completion authority. A cursor is never evidence that messages were processed.
-- This cursor bounds discovery work; only source receipts prove completion.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
CREATE TABLE memory_message_audit_cursors (
  batch_id TEXT PRIMARY KEY REFERENCES memory_extraction_batches(id) ON DELETE CASCADE,
  after_at TIMESTAMPTZ,
  after_id TEXT,
  CHECK ((after_at IS NULL) = (after_id IS NULL))
);
COMMIT;
