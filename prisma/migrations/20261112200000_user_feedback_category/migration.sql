-- 意见反馈区分类型 (2026-11-12 合规: 投诉举报入口)。feedback = 普通意见反馈; report = 投诉举报。
-- 存量行默认 feedback, 行为不变。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE user_feedback ADD COLUMN category TEXT NOT NULL DEFAULT 'feedback'
  CHECK (category IN ('feedback', 'report'));
CREATE INDEX user_feedback_category_created_idx ON user_feedback (category, created_at DESC);
COMMIT;
