-- 用户注册来源标注: 仅在创建账号时写入一次, 后续登录不更新.
-- signup_source:   密码注册 password[_app|_miniprogram|_h5|_web] / 微信注册 wechat_app|wechat_miniprogram|wechat_h5
-- signup_platform: 设备平台 ios / android / harmony / devtools / windows / mac / web ...
-- signup_os_version / signup_app_version: 注册时的系统版本与客户端版本 (尽力采集, 可空)
-- 历史用户四列均为 NULL (语义 = 未知/该功能上线前注册).
ALTER TABLE "users" ADD COLUMN "signup_source" TEXT;
ALTER TABLE "users" ADD COLUMN "signup_platform" TEXT;
ALTER TABLE "users" ADD COLUMN "signup_os_version" TEXT;
ALTER TABLE "users" ADD COLUMN "signup_app_version" TEXT;
