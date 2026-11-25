-- 微信支付与 Apple 内购分开开关。
-- real_money_payments_enabled 保留原名，从此只管 Apple 内购：已发布的 Flutter
-- 客户端读的就是它，改名会让旧版 App 的商城误判。
-- 新列初值取旧开关的当前值：上线前微信支付也由旧开关控制，部署后行为不变。
ALTER TABLE "system_config"
ADD COLUMN IF NOT EXISTS "wechat_pay_purchases_enabled" BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE "system_config"
SET "wechat_pay_purchases_enabled" = "real_money_payments_enabled"
WHERE "id" = 1;

-- 回填订单的 notify_id：此前回调结算没有把通知 id 传进订单行。
-- 取该订单最近一次处理成功的支付成功回调；已有值的不动。
UPDATE "wechat_pay_orders" o
SET "notify_id" = n."notify_id"
FROM (
    SELECT DISTINCT ON ("out_trade_no") "out_trade_no", "notify_id"
    FROM "wechat_pay_notifications"
    WHERE "event_type" = 'TRANSACTION.SUCCESS' AND "processed_at" IS NOT NULL
      AND "out_trade_no" IS NOT NULL
    ORDER BY "out_trade_no", "processed_at" DESC
) n
WHERE o."out_trade_no" = n."out_trade_no" AND o."notify_id" IS NULL;
