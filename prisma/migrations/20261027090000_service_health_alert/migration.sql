BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Additive singleton: service-level alert state, serialized by its row lock.
CREATE TABLE ops_service_health (
    id TEXT PRIMARY KEY CHECK (id = 'primary'),
    sampled_at TIMESTAMPTZ,
    snapshot JSONB NOT NULL DEFAULT '{}',
    level TEXT NOT NULL DEFAULT 'normal' CHECK (level IN ('normal','warning','unknown')),
    notified_signature TEXT,
    generation INTEGER NOT NULL DEFAULT 0,
    notification_ids JSONB NOT NULL DEFAULT '[]',
    recipient_count INTEGER NOT NULL DEFAULT 0
);
-- The memory check scans only unfinished batches; completed rows are the bulk.
CREATE INDEX memory_extraction_batches_unfinished_idx ON memory_extraction_batches(prepared_at)
    WHERE status IN ('preparing','processing');
COMMIT;
