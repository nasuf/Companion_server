-- Immutable confirmation receipts reuse the existing scoped input ledger.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
ALTER TABLE memory_input_origins DROP CONSTRAINT memory_input_origins_source_status_check;
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_input_origins_source_status_check
    CHECK (source_status IN ('knowledge_document','admin_edit','user_edit','admin_repair','contradiction_confirmation'));
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_confirmation_ticket_check
    CHECK (source_status <> 'contradiction_confirmation' OR
        COALESCE(payload_text::jsonb->'input'->>'pending_id','') ~ '^[0-9a-f]{32}$');
CREATE UNIQUE INDEX memory_confirmation_ticket_key ON memory_input_origins
    (workspace_id, (payload_text::jsonb->'input'->>'pending_id'))
    WHERE source_status='contradiction_confirmation';
COMMIT;
