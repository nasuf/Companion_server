-- Drop free-meal voucher campaign tables and system_config columns.
DROP TABLE IF EXISTS "meal_redemption_failures";
DROP TABLE IF EXISTS "meal_vouchers";
DROP TABLE IF EXISTS "meal_merchants";
ALTER TABLE "system_config" DROP COLUMN IF EXISTS "meal_code_enabled";
ALTER TABLE "system_config" DROP COLUMN IF EXISTS "meal_code_anchor";
