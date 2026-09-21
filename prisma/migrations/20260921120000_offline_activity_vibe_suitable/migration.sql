-- 活动卡新增「氛围 / 适合」两行（打卡页详情卡展示，图标见前端）。
-- 由活动推荐大模型生成的短标签，如 vibe="人文、安静" / suitable="看展、拍细节"。
ALTER TABLE offline_activity_recommendations
    ADD COLUMN IF NOT EXISTS vibe TEXT,
    ADD COLUMN IF NOT EXISTS suitable TEXT;
