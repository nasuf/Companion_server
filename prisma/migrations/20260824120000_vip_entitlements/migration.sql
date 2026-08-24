-- VIP 权益与计量层：限时钞票、带过期的消耗品批次、对话额度、音乐时长
-- 详见 CLAUDE.md 无对应节，实施计划 vip-vip-1-1-dapper-hinton.md。

-- 1. 限时(赠送)钞票 + 超额小数累加器 + VIP 发放锚点
ALTER TABLE user_wallets
    ADD COLUMN IF NOT EXISTS gift_ticket_balance INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS overage_accrued NUMERIC(10, 2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS vip_last_grant_at TIMESTAMP(3);

ALTER TABLE user_wallets
    DROP CONSTRAINT IF EXISTS user_wallets_gift_ticket_nonnegative;
ALTER TABLE user_wallets
    ADD CONSTRAINT user_wallets_gift_ticket_nonnegative CHECK (gift_ticket_balance >= 0);

-- 2. 带过期的消耗品批次（音乐畅听券 / 补签卡）
CREATE TABLE IF NOT EXISTS user_consumable_batch (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    product_kind TEXT NOT NULL,          -- 'music_hour_coupon' | 'makeup_card'
    quantity INTEGER NOT NULL,           -- 剩余数量
    source TEXT NOT NULL,                -- 'purchase' | 'vip_grant'
    expires_at TIMESTAMP(3),             -- NULL=永久；购买券=+30天；VIP赠送=+1月
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_consumable_batch_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_consumable_batch_quantity_nonnegative CHECK (quantity >= 0)
);

CREATE INDEX IF NOT EXISTS user_consumable_batch_consume_idx
    ON user_consumable_batch(user_id, product_kind, expires_at);

-- 2.1 存量迁移：把 user_store_inventory 里的券/卡搬进批次表
--     音乐券按购买 30 天有效；补签卡按永久（购买无过期）。
INSERT INTO user_consumable_batch (user_id, product_kind, quantity, source, expires_at, created_at)
SELECT user_id, 'music_hour_coupon', quantity, 'purchase',
       CURRENT_TIMESTAMP + INTERVAL '30 days', acquired_at
FROM user_store_inventory
WHERE product_kind = 'music_hour_coupon' AND quantity > 0;

INSERT INTO user_consumable_batch (user_id, product_kind, quantity, source, expires_at, created_at)
SELECT user_id, 'makeup_card', quantity, 'purchase', NULL, acquired_at
FROM user_store_inventory
WHERE product_kind = 'makeup_card' AND quantity > 0;

DELETE FROM user_store_inventory
WHERE product_kind IN ('music_hour_coupon', 'makeup_card');

-- 3. 对话额度计数（day=非VIP免费 / month=VIP免费）
CREATE TABLE IF NOT EXISTS user_message_quota (
    user_id TEXT NOT NULL,
    period_scope TEXT NOT NULL,          -- 'day' | 'month'
    period_key TEXT NOT NULL,            -- UTC+8 'YYYY-MM-DD' | 'YYYY-MM'
    used INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, period_scope, period_key),
    CONSTRAINT user_message_quota_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- 4. 音乐时长计量（按 UTC+8 自然日）
--    免费额度(1800s)为隐式常量，不入表；provisioned_seconds 只记券+钞票额外覆盖。
--    券按 1h/张、钞票按 0.5h/块提供覆盖，均为整数，故无需小数累加器。
CREATE TABLE IF NOT EXISTS user_music_quota (
    user_id TEXT NOT NULL,
    day_key TEXT NOT NULL,                          -- UTC+8 'YYYY-MM-DD'
    listened_seconds INTEGER NOT NULL DEFAULT 0,    -- 今日累计收听秒数
    provisioned_seconds INTEGER NOT NULL DEFAULT 0, -- 券+钞票额外覆盖秒数(免费1800不计入)
    coupon_units INTEGER NOT NULL DEFAULT 0,        -- 审计: 今日消耗券数
    ticket_spent INTEGER NOT NULL DEFAULT 0,        -- 审计: 今日音乐钞票花费
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, day_key),
    CONSTRAINT user_music_quota_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
);
