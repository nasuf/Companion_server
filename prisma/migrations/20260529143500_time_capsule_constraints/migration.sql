ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_status_check"
    CHECK ("status" IN ('draft', 'sealed')) NOT VALID;

ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_sealed_open_date_check"
    CHECK ("status" <> 'sealed' OR "open_date" IS NOT NULL) NOT VALID;
