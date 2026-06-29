ALTER TABLE real_world_gifts
    ADD COLUMN IF NOT EXISTS provider_product_id TEXT,
    ADD COLUMN IF NOT EXISTS product_url TEXT,
    ADD COLUMN IF NOT EXISTS product_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS logistics_provider TEXT,
    ADD COLUMN IF NOT EXISTS provider_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS logistics_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS last_tracking_synced_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS real_world_gifts_provider_order_idx
    ON real_world_gifts(provider, provider_order_id);
