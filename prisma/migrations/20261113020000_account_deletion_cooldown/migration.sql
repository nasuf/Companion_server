-- 用户自助注销, 7 天冷静期 (2026-11-13)。申请时记下时间, 期间重新登录即撤销;
-- 到期由定时任务调用 hard_delete_user_data 彻底删除。NULL = 未申请。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE users ADD COLUMN deletion_requested_at TIMESTAMP(3), ADD COLUMN deletion_due_at TIMESTAMP(3);
CREATE INDEX users_deletion_due_idx ON users (deletion_due_at) WHERE deletion_due_at IS NOT NULL;
COMMIT;
