-- 双人方块竞速胜利积分 4 → 3 (PM 2026-08-09).
--
-- 这局原先没有 PM 规格, 种子给的是占位的 +4。结算页的积分素材定稿为 "+3 / -3",
-- 以素材为准把胜利分改成 3, 界面与实际入账才一致; 输 / 中途退出维持 -3。
-- 同时清掉 pending_pm 标记 —— 后台"积分规则"页不必再提示待确认。
UPDATE game_point_rules
SET rules = (rules - 'pending_pm') || '{"win":3}'::jsonb,
    updated_at = NOW()
WHERE game_key = 'tetris_duel'
  AND rules ->> 'type' = 'outcome';
