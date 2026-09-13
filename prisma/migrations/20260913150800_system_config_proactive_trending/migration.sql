-- Proactive trending knobs (global SystemConfig, runtime-config admin API).
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "proactive_trending_enabled" BOOLEAN;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "proactive_trending_probability" DOUBLE PRECISION;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "proactive_trending_link_probability" DOUBLE PRECISION;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "proactive_trending_cache_ttl_s" INTEGER;
