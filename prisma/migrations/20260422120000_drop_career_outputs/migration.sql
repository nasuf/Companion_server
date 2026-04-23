-- Drop `outputs` (主要产出物) column from career_templates.
--
-- 2026-04-21 PM 发布新版《AI职业背景设定》(34 条)，模板字段精简为
-- 职业 / 工作内容 / 社会价值 / 服务对象 四项，去掉"主要产出物"。
--
-- 此迁移只 DROP COLUMN；新增职业 + 删除"陨石猎人"由启动时的
-- `ensure_default_careers` 幂等逻辑处理（title 不冲突则 CREATE；
-- "陨石猎人"若无 profile 引用则 DELETE，否则标记 archived）。

ALTER TABLE "career_templates" DROP COLUMN IF EXISTS "outputs";
