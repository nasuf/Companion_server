-- Preserve the explicit-open product rule after the transitional migration
-- that marked already-due sealed capsules as opened.
UPDATE "time_capsules"
SET "opened_at" = NULL,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "status" = 'sealed'
  AND "opened_at" IS NOT NULL
  AND (
    ("sealed_at" IS NOT NULL AND "opened_at" = "sealed_at")
    OR ("sealed_at" IS NULL AND "opened_at" = "updated_at")
  );
