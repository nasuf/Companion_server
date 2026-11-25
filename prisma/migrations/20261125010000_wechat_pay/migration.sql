-- 服务号 H5 微信支付（JSAPI）：订单主账、回调审计、退款、服务号 openid 映射。
-- 与 Apple IAP 完全分表：iap_transactions / iap_subscription_state / iap_notifications
-- 一行不动，Apple 的验签、通知、退款、订阅链路不受影响。共享的只有 user_wallets
-- （vip_until / ticket_balance / vip_trial_used）与 wallet_ledger 账变流水。
-- 沿用项目 raw-SQL + IF NOT EXISTS 幂等惯例（见 20260905120000_apple_iap）。

-- 1. 服务号 openid 映射。
--    为什么单独建表而不复用 auth_identities.openid：那一列在每次微信登录时被
--    覆盖（wechat_auth.py 的 update），移动应用与服务号是两个不同 appid，openid
--    互不通用。JSAPI 下单必须用服务号 appid 下的 openid，所以单独存，且不碰
--    现有身份表，避免影响 App 登录链路。
--    账号归一仍由 auth_identities 的 unionid 负责：用户无论从 App 还是服务号
--    登录，unionid 相同 → 同一个 users.id → 这里的 user_id 也相同。
CREATE TABLE IF NOT EXISTS wechat_pay_openids (
    openid     TEXT PRIMARY KEY,                -- 服务号 appid 下的 openid
    app_id     TEXT NOT NULL,                   -- 服务号 appid（换号时能看出历史）
    user_id    TEXT NOT NULL,
    unionid    TEXT,                            -- 冗余留痕，便于排查账号归一问题
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT wechat_pay_openids_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS wechat_pay_openids_user_idx
    ON wechat_pay_openids(user_id, updated_at DESC);

-- 2. 订单主账。out_trade_no 由服务端生成且唯一 = 到账幂等地基。
--    金额、商品、user_id 全部在下单时由服务端写定，回调只按 out_trade_no 回链，
--    绝不信任回调里的金额或 openid 去决定发给谁。
CREATE TABLE IF NOT EXISTS wechat_pay_orders (
    id                TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    out_trade_no      TEXT NOT NULL,                 -- 商户订单号，幂等键
    transaction_id    TEXT,                          -- 微信支付订单号（支付成功后才有）
    mch_id            TEXT NOT NULL,
    app_id            TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    openid            TEXT NOT NULL,                 -- 下单时的付款人 openid
    product_id        TEXT NOT NULL,
    product_title     TEXT NOT NULL,
    kind              TEXT NOT NULL,                 -- consumable（内测不开订阅）
    amount_fen        INTEGER NOT NULL,              -- 应付金额（分），服务端定价
    paid_amount_fen   INTEGER,                        -- 实付金额（分），来自回调
    currency          TEXT NOT NULL DEFAULT 'CNY',
    vip_days          INTEGER NOT NULL DEFAULT 0,    -- 下单时快照的权益，改价不影响历史单
    ticket_amount     INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL,                 -- created|paid|granted|closed|refunded|failed
    trade_state       TEXT,                          -- 微信侧 trade_state 原值
    prepay_id         TEXT,
    success_time      TIMESTAMP(3),                  -- 微信侧支付完成时间
    granted_at        TIMESTAMP(3),                  -- 我方权益发放时间
    closed_at         TIMESTAMP(3),
    refunded_at       TIMESTAMP(3),
    refunded_fen      INTEGER NOT NULL DEFAULT 0,
    wallet_ledger_source_id TEXT,                    -- 回链 wallet_ledger.(source, source_id)
    fail_reason       TEXT,                          -- 发放失败原因（后台排查用）
    notify_id         TEXT,                          -- 最后一次处理的回调 id
    raw_payload       JSONB NOT NULL DEFAULT '{}',   -- 最近一次查单/回调的脱敏明文
    created_at        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 与 iap_transactions 一致随用户级联删除：账号注销（user_erasure.verify_erasure）
    -- 要求不残留任何 user_id 行。对账以微信商户平台的交易记录为准。
    CONSTRAINT wechat_pay_orders_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT wechat_pay_orders_out_trade_no_unique UNIQUE (out_trade_no),
    CONSTRAINT wechat_pay_orders_kind_check CHECK (kind IN ('consumable')),
    CONSTRAINT wechat_pay_orders_status_check
        CHECK (status IN ('created', 'paid', 'granted', 'closed', 'refunded', 'failed')),
    CONSTRAINT wechat_pay_orders_amount_check CHECK (amount_fen > 0),
    CONSTRAINT wechat_pay_orders_refunded_check
        CHECK (refunded_fen >= 0 AND refunded_fen <= amount_fen)
);
-- 微信支付订单号在有值时也必须唯一（同一笔支付不能回链到两张订单）。
CREATE UNIQUE INDEX IF NOT EXISTS wechat_pay_orders_transaction_id_unique
    ON wechat_pay_orders(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS wechat_pay_orders_user_idx
    ON wechat_pay_orders(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS wechat_pay_orders_status_idx
    ON wechat_pay_orders(status, created_at DESC);
-- 未支付订单的查单兜底扫描（回调丢失时由 cron 主动查单）。
CREATE INDEX IF NOT EXISTS wechat_pay_orders_pending_idx
    ON wechat_pay_orders(status, updated_at) WHERE status = 'created';

-- 3. 回调原始审计 + 幂等。收到即落库、再处理，失败留 process_error 供后台查。
CREATE TABLE IF NOT EXISTS wechat_pay_notifications (
    id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    notify_id     TEXT NOT NULL,                     -- 微信回调通知 id，幂等键
    event_type    TEXT NOT NULL,                     -- TRANSACTION.SUCCESS / REFUND.*
    summary       TEXT,
    out_trade_no  TEXT,
    transaction_id TEXT,
    decoded_payload JSONB NOT NULL DEFAULT '{}',     -- 解密后的脱敏明文
    processed_at  TIMESTAMP(3),
    process_error TEXT,
    received_at   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT wechat_pay_notifications_notify_id_unique UNIQUE (notify_id)
);
CREATE INDEX IF NOT EXISTS wechat_pay_notifications_type_idx
    ON wechat_pay_notifications(event_type, received_at DESC);
CREATE INDEX IF NOT EXISTS wechat_pay_notifications_unprocessed_idx
    ON wechat_pay_notifications(processed_at, received_at DESC);
CREATE INDEX IF NOT EXISTS wechat_pay_notifications_order_idx
    ON wechat_pay_notifications(out_trade_no, received_at DESC);

-- 4. 退款。内测规则：未使用的可全额退，由后台人工发起（见 docs）。
--    out_refund_no 唯一 = 退款幂等地基，避免重复提交造成多次出款。
CREATE TABLE IF NOT EXISTS wechat_pay_refunds (
    id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    out_refund_no  TEXT NOT NULL,
    refund_id      TEXT,                             -- 微信退款单号
    out_trade_no   TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    amount_fen     INTEGER NOT NULL,
    status         TEXT NOT NULL,                    -- pending|success|closed|abnormal|failed
    reason         TEXT,
    actor_id       TEXT,                             -- 发起退款的管理员
    fail_reason    TEXT,
    success_time   TIMESTAMP(3),
    raw_payload    JSONB NOT NULL DEFAULT '{}',
    created_at     TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT wechat_pay_refunds_out_refund_no_unique UNIQUE (out_refund_no),
    CONSTRAINT wechat_pay_refunds_order_fkey
        FOREIGN KEY (out_trade_no) REFERENCES wechat_pay_orders(out_trade_no)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT wechat_pay_refunds_amount_check CHECK (amount_fen > 0),
    CONSTRAINT wechat_pay_refunds_status_check
        CHECK (status IN ('pending', 'success', 'closed', 'abnormal', 'failed'))
);
CREATE INDEX IF NOT EXISTS wechat_pay_refunds_order_idx
    ON wechat_pay_refunds(out_trade_no, created_at DESC);
CREATE INDEX IF NOT EXISTS wechat_pay_refunds_status_idx
    ON wechat_pay_refunds(status, created_at DESC);
-- 同一订单同时只能有一张进行中/已成功的退款单（与 create_refund 的行锁双保险）。
CREATE UNIQUE INDEX IF NOT EXISTS wechat_pay_refunds_open_unique
    ON wechat_pay_refunds(out_trade_no) WHERE status IN ('pending', 'success');

-- 5. wallet_ledger 的 source 没有白名单约束（见 20260905120000_apple_iap），
--    新 source 'wechat_pay' / 'wechat_pay_refund' 无需迁移即可写入。
