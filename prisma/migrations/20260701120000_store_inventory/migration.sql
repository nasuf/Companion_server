CREATE TABLE IF NOT EXISTS user_store_inventory (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    product_kind TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    acquired_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_store_inventory_user_product_key UNIQUE (user_id, product_kind),
    CONSTRAINT user_store_inventory_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_store_inventory_quantity_nonnegative CHECK (quantity >= 0)
);

CREATE INDEX IF NOT EXISTS user_store_inventory_user_updated_idx
    ON user_store_inventory(user_id, updated_at DESC);
