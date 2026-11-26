-- 聊天加号面板「语音通话」全局开关，默认关闭。
-- 关闭时 App 不显示入口，/ws/voice 拒绝建立新通话。
ALTER TABLE "system_config"
ADD COLUMN IF NOT EXISTS "voice_call_enabled" BOOLEAN NOT NULL DEFAULT FALSE;
