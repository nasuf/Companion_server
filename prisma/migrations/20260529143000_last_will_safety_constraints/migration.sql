DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_wills_status_check'
    ) THEN
        ALTER TABLE "last_wills"
            ADD CONSTRAINT "last_wills_status_check"
            CHECK ("status" IN ('draft', 'active', 'paused', 'triggered', 'cancelled'))
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_wills_inactivity_days_check'
    ) THEN
        ALTER TABLE "last_wills"
            ADD CONSTRAINT "last_wills_inactivity_days_check"
            CHECK ("inactivity_days" BETWEEN 5 AND 365)
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_wills_contacts_shape_check'
    ) THEN
        ALTER TABLE "last_wills"
            ADD CONSTRAINT "last_wills_contacts_shape_check"
            CHECK (
                jsonb_typeof("contacts") = 'array'
                AND jsonb_array_length("contacts") <= 3
            )
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_wills_active_payload_check'
    ) THEN
        ALTER TABLE "last_wills"
            ADD CONSTRAINT "last_wills_active_payload_check"
            CHECK (
                "status" <> 'active'
                OR (
                    btrim("content") <> ''
                    AND jsonb_typeof("contacts") = 'array'
                    AND jsonb_array_length("contacts") >= 1
                )
            )
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_will_deliveries_channel_check'
    ) THEN
        ALTER TABLE "last_will_deliveries"
            ADD CONSTRAINT "last_will_deliveries_channel_check"
            CHECK ("channel" IN ('email', 'phone'))
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'last_will_deliveries_status_check'
    ) THEN
        ALTER TABLE "last_will_deliveries"
            ADD CONSTRAINT "last_will_deliveries_status_check"
            CHECK ("status" IN ('pending', 'sent', 'failed', 'cancelled'))
            NOT VALID;
    END IF;
END $$;
