-- The campaign now has one redemption path only: an authenticated merchant
-- scans the customer's short-lived QR grant.
ALTER TABLE "meal_vouchers"
DROP COLUMN IF EXISTS "redeem_method";

DROP INDEX IF EXISTS "meal_merchants_redeem_code_key";

ALTER TABLE "meal_merchants"
DROP COLUMN IF EXISTS "redeem_code";
