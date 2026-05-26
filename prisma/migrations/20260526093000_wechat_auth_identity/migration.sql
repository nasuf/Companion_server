ALTER TABLE "users" ALTER COLUMN "hashed_password" DROP NOT NULL;

CREATE TABLE "auth_identities" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "provider_account_id" TEXT NOT NULL,
    "openid" TEXT,
    "unionid" TEXT,
    "scope" TEXT,
    "raw_profile" JSONB,
    "last_login_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "auth_identities_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "auth_identities_provider_account_unique"
    ON "auth_identities"("provider", "provider_account_id");

CREATE INDEX "auth_identities_user_id_idx" ON "auth_identities"("user_id");
CREATE INDEX "auth_identities_openid_idx" ON "auth_identities"("openid");
CREATE INDEX "auth_identities_unionid_idx" ON "auth_identities"("unionid");

ALTER TABLE "auth_identities"
    ADD CONSTRAINT "auth_identities_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
