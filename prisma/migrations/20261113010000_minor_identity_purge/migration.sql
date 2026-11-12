-- 判定为未成年的账号不再保存身份信息 (2026-11-13): 只保留证件摘要 (防换号重注) 与锁定状态。
-- 清理此前写入的未成年记录中的姓名、证件号密文与证件解析信息。生产此时 0 行, 幂等。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
UPDATE user_identity_verifications
   SET masked_name = NULL, name_encrypted = NULL, id_number_encrypted = NULL,
       birth_date = NULL, gender = NULL, region_code = NULL, province = NULL, updated_at = now()
 WHERE status = 'minor'
   AND (masked_name IS NOT NULL OR name_encrypted IS NOT NULL OR id_number_encrypted IS NOT NULL
        OR birth_date IS NOT NULL OR gender IS NOT NULL OR region_code IS NOT NULL OR province IS NOT NULL);
COMMIT;
