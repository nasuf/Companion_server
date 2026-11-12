-- 实名 v2 (2026-11-12):
-- 1) system_config.realname_policy: 后台实时开关 {mode, scope, user_ids}; NULL = 沿用环境变量 REALNAME_MODE。
--    改动经既有 system_config 更新触发器递增 configuration_revisions, 各 worker 2s 内重载。
-- 2) realname_policy_changes: 每次保存的修改记录。
-- 3) user_identity_verifications 增加证件号解析出的出生日期/性别/户籍地区, 以及加密保存的姓名与证件号
--    (AES-GCM, 密钥由 REALNAME_HASH_KEY 派生; 库里没有明文)。随账号注销删除。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

ALTER TABLE system_config ADD COLUMN realname_policy JSONB;
ALTER TABLE system_config ADD CONSTRAINT system_config_realname_policy_object
  CHECK (realname_policy IS NULL OR jsonb_typeof(realname_policy) = 'object');

CREATE TABLE realname_policy_changes (
  id TEXT PRIMARY KEY,
  admin_id TEXT,
  changes JSONB NOT NULL,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX realname_policy_changes_created_at_idx ON realname_policy_changes (created_at DESC);

ALTER TABLE user_identity_verifications
  ADD COLUMN birth_date DATE,
  ADD COLUMN gender TEXT CHECK (gender IS NULL OR gender IN ('male', 'female')),
  ADD COLUMN region_code TEXT CHECK (region_code IS NULL OR region_code ~ '^[0-9]{6}$'),
  ADD COLUMN province TEXT,
  ADD COLUMN name_encrypted TEXT,
  ADD COLUMN id_number_encrypted TEXT;

COMMIT;
