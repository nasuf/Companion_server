-- 成就系统全局模式开关 (admin 后台「系统设置」动态切换, 与线下活动/礼物开关同面板).
-- 取值 on / silent / off; NULL = 跟随 .env ACHIEVEMENT_MODE 默认值.
-- silent 用于 H5 纯聊天上线: 成就照常实时评估并落库 (unlocked_at/conversation_id
-- 即真实达成点), 但通知/API/时间线/钱包积分全部抑制; 切回 on 后全量自动呈现.
ALTER TABLE "system_config"
    ADD COLUMN IF NOT EXISTS "achievement_mode" TEXT;
