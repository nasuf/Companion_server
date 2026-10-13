-- Add one scoped input kind. Historic inputs and evidence remain immutable.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
ALTER TABLE memory_input_origins DROP CONSTRAINT memory_input_origins_source_status_check;
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_input_origins_source_status_check
    CHECK (source_status IN ('knowledge_document','admin_edit','user_edit','admin_repair'));
COMMIT;
