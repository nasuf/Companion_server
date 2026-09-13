-- Chat management: random reply delay + user message aggregation toggles.
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "reply_delay_enabled" BOOLEAN;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "reply_delay_max_seconds" INTEGER;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "user_message_aggregation_enabled" BOOLEAN;
