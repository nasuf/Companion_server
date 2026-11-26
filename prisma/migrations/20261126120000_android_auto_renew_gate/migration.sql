-- 安卓端「连续包月」档位开关，默认关闭。
-- 安卓走服务号 H5 付款，没有自动续费能力，所以档位先隐藏；iOS 不读这个开关。
ALTER TABLE "system_config"
ADD COLUMN IF NOT EXISTS "android_auto_renew_enabled" BOOLEAN NOT NULL DEFAULT FALSE;
