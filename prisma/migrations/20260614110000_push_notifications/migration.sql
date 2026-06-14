CREATE TABLE "push_devices" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "user_id" TEXT NOT NULL,
    "platform" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT 'apns',
    "token" TEXT NOT NULL,
    "environment" TEXT NOT NULL DEFAULT 'sandbox',
    "bundle_id" TEXT,
    "device_id" TEXT,
    "app_version" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "last_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "disabled_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "push_devices_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "notification_events" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "workspace_id" TEXT,
    "conversation_id" TEXT,
    "message_id" TEXT,
    "type" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "body" TEXT NOT NULL,
    "payload" JSONB NOT NULL DEFAULT '{}',
    "dedupe_key" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "provider" TEXT NOT NULL DEFAULT 'apns',
    "provider_message_id" TEXT,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "error" TEXT,
    "scheduled_for" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sent_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "notification_events_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "push_devices_provider_token_key" ON "push_devices"("provider", "token");
CREATE INDEX "push_devices_user_enabled_idx" ON "push_devices"("user_id", "enabled", "last_seen_at" DESC);
CREATE INDEX "push_devices_user_device_idx" ON "push_devices"("user_id", "device_id");

CREATE UNIQUE INDEX "notification_events_user_type_dedupe_key" ON "notification_events"("user_id", "type", "dedupe_key");
CREATE INDEX "notification_events_status_scheduled_idx" ON "notification_events"("status", "scheduled_for");
CREATE INDEX "notification_events_user_created_idx" ON "notification_events"("user_id", "created_at" DESC);

ALTER TABLE "push_devices"
ADD CONSTRAINT "push_devices_user_id_fkey"
FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "notification_events"
ADD CONSTRAINT "notification_events_user_id_fkey"
FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
