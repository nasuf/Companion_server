BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Constant default: metadata-only on PostgreSQL 11+, no table rewrite.
ALTER TABLE messages ADD COLUMN hidden_from_timeline BOOLEAN NOT NULL DEFAULT false;
-- Earlier realtime voice calls stored their transcripts as visible rows.
UPDATE messages SET hidden_from_timeline = true
 WHERE metadata->>'channel' = 'realtime_voice';
COMMIT;
