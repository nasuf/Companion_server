BEGIN;
SET LOCAL lock_timeout='5s';
-- Contains metering contract only, never prompt bodies or credentials.
ALTER TABLE quality_charges ADD COLUMN request_policy JSONB NOT NULL DEFAULT '{}'::jsonb;
COMMIT;
