-- 删除 users.name 列。历史上与 username 双写相同值, 留作 iOS 旧匿名路径
-- 兼容字段, 但 iOS 路径已因 hashedPassword 必填而隐式失效, 全栈消费方
-- 全部走 username, name 字段已无价值。
--
-- 安全性: 所有现存 row 的 name 与 username 一致 (auth.register 强制双
-- 写), 删列不丢任何独立信息。
ALTER TABLE "users" DROP COLUMN "name";
