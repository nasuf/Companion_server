-- Apple IAP（VIP 订阅 + 钞票充值）：交易主账、订阅状态、V2 通知审计。
-- Direct Apple 自建：所有交易/通知落自己的库，形成 通知→交易→账变 的完整审计链。
-- 沿用项目 raw-SQL + IF NOT EXISTS 幂等惯例（见 20260824120000_vip_entitlements）。

-- 0. 修历史遗留 bug：wallet_ledger 的 currency CHECK 从建表起只允许 ticket/point，
--    但 credit_gift_tickets/zero_gift_tickets 早已在写 gift_ticket（约束从未随之更新）。
--    IAP 充值到账走 ticket，不新增币种；此处顺手把 gift_ticket 补进合法集合。
ALTER TABLE wallet_ledger DROP CONSTRAINT IF EXISTS wallet_ledger_currency_check;
ALTER TABLE wallet_ledger
    ADD CONSTRAINT wallet_ledger_currency_check
    CHECK (currency IN ('ticket', 'point', 'gift_ticket'));

-- 1. IAP 交易主账 + 审计。transaction_id 唯一 = 到账幂等地基。
CREATE TABLE IF NOT EXISTS iap_transactions (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    provider                TEXT NOT NULL DEFAULT 'apple',   -- apple/google/wechat 预留
    transaction_id          TEXT NOT NULL,                   -- Apple transactionId, 幂等键
    original_transaction_id TEXT,                            -- 订阅家族键(消耗型=自身)
    web_order_line_item_id  TEXT,                            -- 订阅每期行项
    product_id              TEXT NOT NULL,
    kind                    TEXT NOT NULL,                   -- subscription | consumable
    environment             TEXT NOT NULL,                   -- Sandbox | Production
    user_id                 TEXT NOT NULL,
    quantity                INTEGER NOT NULL DEFAULT 1,
    purchase_date           TIMESTAMP(3),
    expires_date            TIMESTAMP(3),                    -- 订阅到期(消耗型 NULL)
    status                  TEXT NOT NULL,                   -- pending|granted|refunded|revoked|failed
    wallet_ledger_source_id TEXT,                            -- 回链 wallet_ledger.(source,source_id)
    notification_uuid       TEXT,                            -- 若由 webhook 创建/更新, 记来源通知
    raw_transaction_payload JSONB NOT NULL DEFAULT '{}',
    raw_jws                 TEXT,
    created_at              TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT iap_transactions_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT iap_transactions_provider_txn_unique UNIQUE (provider, transaction_id),
    CONSTRAINT iap_transactions_kind_check CHECK (kind IN ('subscription', 'consumable')),
    CONSTRAINT iap_transactions_status_check
        CHECK (status IN ('pending', 'granted', 'refunded', 'revoked', 'failed'))
);

CREATE INDEX IF NOT EXISTS iap_transactions_user_idx
    ON iap_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS iap_transactions_original_txn_idx
    ON iap_transactions(original_transaction_id);
CREATE INDEX IF NOT EXISTS iap_transactions_status_env_idx
    ON iap_transactions(status, environment, created_at DESC);

-- 2. 订阅当前真相（key = original_transaction_id）。vip_until 仍是 VIP 权益唯一源，
--    此表是 webhook 增量落点 + "为什么 vip_until 是这个值"的解释。
CREATE TABLE IF NOT EXISTS iap_subscription_state (
    original_transaction_id   TEXT PRIMARY KEY,
    provider                  TEXT NOT NULL DEFAULT 'apple',
    user_id                   TEXT NOT NULL,
    product_id                TEXT NOT NULL,
    environment               TEXT NOT NULL,
    status                    TEXT NOT NULL,   -- active|in_grace|expired|revoked|refunded
    auto_renew_status         BOOLEAN,
    auto_renew_product_id     TEXT,
    expires_date              TIMESTAMP(3),
    grace_period_expires_date TIMESTAMP(3),
    latest_transaction_id     TEXT,
    last_notification_type    TEXT,
    last_notification_subtype TEXT,
    updated_at                TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT iap_subscription_state_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS iap_subscription_state_user_idx
    ON iap_subscription_state(user_id);

-- 3. App Store Server Notifications V2 原始审计 + 幂等。收到即落库、再处理。
CREATE TABLE IF NOT EXISTS iap_notifications (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    provider                TEXT NOT NULL DEFAULT 'apple',
    notification_uuid       TEXT NOT NULL,      -- Apple notificationUUID, 幂等键
    notification_type       TEXT NOT NULL,
    subtype                 TEXT,
    environment             TEXT,
    original_transaction_id TEXT,
    transaction_id          TEXT,
    signed_payload          TEXT NOT NULL,
    decoded_payload         JSONB NOT NULL DEFAULT '{}',
    processed_at            TIMESTAMP(3),
    process_error           TEXT,
    received_at             TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT iap_notifications_uuid_unique UNIQUE (provider, notification_uuid)
);
CREATE INDEX IF NOT EXISTS iap_notifications_type_idx
    ON iap_notifications(notification_type, received_at DESC);
CREATE INDEX IF NOT EXISTS iap_notifications_unprocessed_idx
    ON iap_notifications(processed_at, received_at DESC);
