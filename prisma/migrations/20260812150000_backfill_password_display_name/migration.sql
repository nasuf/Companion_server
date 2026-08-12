-- 给存量密码账号回填 display_name = username。
--
-- 背景: 读取链是 display_name → 微信昵称 → 用户{手机尾号}, 末尾刻意不再拿
-- users.username 兜底 —— 那是 `wx_89b939bc004` 这样的内部 hash, 塞进展示名字段
-- 会逼每个客户端写正则再把它过滤掉。
--
-- 但这样一来"三种登录方式全都没有名字"的那一格就露出来了: 密码账号既没有微信昵称
-- 也没有手机尾号。密码账号也是唯一没有**活来源**的类型 (微信昵称每次登录会刷新、
-- 手机号可改绑), 所以对它复制一份 username 不会过期。新注册走 auth.register 里的
-- 预写, 这里补存量。
--
-- 条件是"完全没有任何 auth_identities 行"而不是"username 不像 wx_/ph_": 一个有
-- 密码同时又绑了手机号的账号是有活来源的, 不该被写死一个副本。
--
-- 排除模板系统账号 (__companion_template_system__): 它不是人, 后台用户列表已经把
-- 它过滤掉, 给它一个展示名只会让它某天漏进某个界面。
UPDATE users u
SET display_name = u.username,
    updated_at = CURRENT_TIMESTAMP
WHERE u.display_name IS NULL
  AND u.username <> '__companion_template_system__'
  AND NOT EXISTS (
    SELECT 1 FROM auth_identities a WHERE a.user_id = u.id
  );
