-- 实名核验结果 (COMP 实名, 2026-11-12)。每个用户至多一行。
-- 身份证号原文不落库: id_digest 是 HMAC-SHA256 摘要, 只用于"一张证只认证一个账号";
-- masked_name 只保留姓氏。随账号注销删除 (ON DELETE CASCADE + data_reset 显式删除)。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE user_identity_verifications (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    status         TEXT NOT NULL CHECK (status IN ('verified', 'minor', 'rejected')),
    masked_name    TEXT,
    id_digest      TEXT,
    is_adult       BOOLEAN,
    provider       TEXT NOT NULL DEFAULT 'tencent',
    failure_reason TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    verified_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 一张证只能对应一个已通过/已判未成年的账号 (rejected 行不存摘要)
CREATE UNIQUE INDEX user_identity_verifications_digest_uq
    ON user_identity_verifications(id_digest)
    WHERE id_digest IS NOT NULL AND status IN ('verified', 'minor');
CREATE INDEX user_identity_verifications_status_idx
    ON user_identity_verifications(status, updated_at DESC);

COMMIT;
