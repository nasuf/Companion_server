-- 霸王餐商家「千味央厨」归属标志:
--   qwyc_member = 该商家属于「千味央厨」品牌下的成员门店 (计入汇总)。
--   qwyc_group  = 该商家是「千味央厨」总账号 (登录后查看汇总, 不扫码核销)。
ALTER TABLE meal_merchants
    ADD COLUMN IF NOT EXISTS qwyc_member BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS qwyc_group  BOOLEAN NOT NULL DEFAULT false;

-- 汇总查询按 qwyc_member 过滤成员门店, 建部分索引加速。
CREATE INDEX IF NOT EXISTS meal_merchants_qwyc_member_idx
    ON meal_merchants (qwyc_member)
    WHERE qwyc_member = true;
