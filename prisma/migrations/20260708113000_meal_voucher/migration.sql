-- 霸王餐 (free-meal voucher) campaign tables + activation-code kill switch.

-- Admin toggle: disabling invalidates all rotating activation codes at once.
ALTER TABLE system_config
    ADD COLUMN IF NOT EXISTS meal_code_enabled BOOLEAN NOT NULL DEFAULT true;

-- Merchants: one fixed 6-digit redeem code each (rotatable / deactivatable).
CREATE TABLE IF NOT EXISTS meal_merchants (
    id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name          TEXT NOT NULL,
    contact_name  TEXT,
    contact_phone TEXT,
    redeem_code   TEXT NOT NULL,
    code_active   BOOLEAN NOT NULL DEFAULT true,
    created_at    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS meal_merchants_redeem_code_key
    ON meal_merchants (redeem_code);

-- Vouchers: one per user; state machine inactive -> activated -> redeemed.
CREATE TABLE IF NOT EXISTS meal_vouchers (
    id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id      TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'inactive',
    activated_at TIMESTAMP(3),
    redeemed_at  TIMESTAMP(3),
    merchant_id  TEXT,
    created_at   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meal_vouchers_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT meal_vouchers_merchant_id_fkey FOREIGN KEY (merchant_id)
        REFERENCES meal_merchants (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS meal_vouchers_user_id_key
    ON meal_vouchers (user_id);
CREATE INDEX IF NOT EXISTS meal_vouchers_status_idx
    ON meal_vouchers (status);
CREATE INDEX IF NOT EXISTS meal_vouchers_merchant_idx
    ON meal_vouchers (merchant_id);
CREATE INDEX IF NOT EXISTS meal_vouchers_activated_at_idx
    ON meal_vouchers (activated_at);
