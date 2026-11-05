BEGIN;
SET LOCAL lock_timeout='5s';
-- CHAT01.10.2 / CHAT01.02 复读门禁: 手动触发的 GitHub 工作流, 用自己的执行锁 (不挡夜间回归)
-- 与独立的月度预算; 用例是人工复核过的脱敏回放用例, 只在后台库里, 不进公开仓库。
ALTER TABLE quality_jobs DROP CONSTRAINT quality_jobs_kind_check;
ALTER TABLE quality_jobs ADD CONSTRAINT quality_jobs_kind_check
  CHECK(kind IN ('daily','prompt','routing','nightly','calibration','repetition_gate'));
ALTER TABLE quality_executor DROP CONSTRAINT quality_executor_id_check;
ALTER TABLE quality_executor ADD CONSTRAINT quality_executor_id_check
  CHECK(id IN ('primary','repetition_gate'));
INSERT INTO quality_executor(id) VALUES('repetition_gate');
-- 不含账号、会话、消息 ID: 用户编号是按出现顺序的 u01…, 回查映射只在受控证据目录里且到期删除。
CREATE TABLE quality_repetition_cases (
 id TEXT PRIMARY KEY,
 case_set TEXT NOT NULL,
 case_data JSONB NOT NULL,
 context JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX quality_repetition_cases_set_idx ON quality_repetition_cases(case_set,id);
COMMIT;
