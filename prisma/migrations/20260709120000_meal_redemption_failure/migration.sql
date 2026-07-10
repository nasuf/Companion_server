-- 霸王餐核销失败留痕: 当日核销总量达到上限 (先到先得, 默认 1000 份/日) 时,
-- 用户核销请求被拒, 在此记录供后台「数据统计」查看谁没抢到.
-- merchantId 为松引用 (无外键): 仅作展示线索, 商家删除不影响历史失败记录.
CREATE TABLE IF NOT EXISTS meal_redemption_failures (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    merchant_id TEXT,
    reason      TEXT NOT NULL DEFAULT 'daily_cap',
    created_at  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT meal_redemption_failures_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS meal_redeem_failures_created_at_idx
    ON meal_redemption_failures (created_at);
CREATE INDEX IF NOT EXISTS meal_redeem_failures_user_idx
    ON meal_redemption_failures (user_id);
