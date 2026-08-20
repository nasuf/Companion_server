-- The app-level validation (LastWillCreate/LastWillUpdate) already allows
-- inactivity_days as low as 1, matching the Flutter ruler's 1-90 day range,
-- but the DB CHECK constraint added in 20260529143000_last_will_safety_constraints
-- still enforced BETWEEN 5 AND 365 — any save with 1-4 days passed Pydantic
-- validation and then failed at the database with a 500, not a clean 4xx.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_wills_inactivity_days_check'
    ) THEN
        ALTER TABLE "last_wills" DROP CONSTRAINT "last_wills_inactivity_days_check";
    END IF;

    ALTER TABLE "last_wills"
        ADD CONSTRAINT "last_wills_inactivity_days_check"
        CHECK ("inactivity_days" BETWEEN 1 AND 365)
        NOT VALID;
END $$;
