-- 用户自设昵称 / 头像的存储位。
--
-- 在此之前 user_display_name / user_avatar_url 唯一的来源是
-- auth_identities.raw_profile->>'nickname' / 'headimgurl' (provider=wechat),
-- update_wechat_profile() 对没有微信身份的用户直接 no-op 返回。生产上 47 个
-- 用户里 3 个手机号 + 9 个密码账号 (26%) 因此完全没有地方存这两个字段, 客户端
-- 一旦开放"修改昵称/头像"就会静默失败。
--
-- 这两列是所有身份类型共用的权威值; 微信 rawProfile 保留为首次登录时的默认来源
-- (读取侧优先本地列, 回落 rawProfile), 所以不需要回填 —— 存量微信用户在没自设
-- 之前照旧显示微信昵称头像。
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_key TEXT;
