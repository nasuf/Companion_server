-- UXM01.01: beta statistics exclude internal accounts by default.
-- Admins are internal by rule (app/services/effect/definitions.py), so no
-- backfill is needed and existing rows keep the safe default. Additive only;
-- old application images ignore the column.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '15s';
ALTER TABLE users ADD COLUMN is_internal BOOLEAN NOT NULL DEFAULT FALSE;
COMMIT;
