-- New real-money checkout starts disabled, including existing installations.
-- Do not change existing VIP grants, wallets, renewals or paid transactions.
ALTER TABLE "system_config"
ADD COLUMN "real_money_payments_enabled" BOOLEAN NOT NULL DEFAULT FALSE;
