-- Drop unused `life_overview_data` column.
--
-- Spec 第一部分 §2.1 + 指令模版 P20「AI生活画像」要求输出纯文本。
-- 早期实现把生活画像存成 JSON ({description, weekday_schedule,
-- weekend_activities, holiday_habits}) 并写入 life_overview_data 列。
-- 4.19 对齐后 LIFE_OVERVIEW_PROMPT 输出已改回纯文本，只写入
-- life_overview TEXT 列，life_overview_data 已停止写入且无下游读取
-- (grep 过 weekday_schedule / weekend_activities / holiday_habits
-- 三个字段名均无 consumer)。此迁移移除这个废弃列。

ALTER TABLE "ai_agents" DROP COLUMN IF EXISTS "life_overview_data";
