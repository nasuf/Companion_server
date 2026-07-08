-- 霸王餐校验码生成锚点: 重新开启功能时刷新, 使旧码立即失效并让新码
-- 从满 5 分钟倒计时开始 (码与窗口序号均相对锚点推导).
ALTER TABLE system_config
    ADD COLUMN IF NOT EXISTS meal_code_anchor BIGINT;
