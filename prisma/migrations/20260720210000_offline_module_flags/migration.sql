-- 线下真实世界模块 (活动推荐 / 礼物推荐) 主动推送开关.
-- 默认关闭: 客户端 (H5) 无对应卡片 UI 时先整体暂停, admin 后台可动态开启,
-- 实时生效 (offline trigger scan 与手动/mock 接口每次读取当前值).
ALTER TABLE "system_config"
    ADD COLUMN IF NOT EXISTS "offline_activity_enabled" BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS "offline_gift_enabled" BOOLEAN NOT NULL DEFAULT false;
