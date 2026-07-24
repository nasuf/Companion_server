-- speech_usage rows are inserted via raw SQL (record_speech_usage / backfill),
-- which bypasses Prisma's client-side @default(uuid()) id generation. Give the
-- column a DB-level default so raw inserts don't violate NOT NULL, matching the
-- existing chat_message_attachments pattern. Idempotent.
ALTER TABLE "speech_usage" ALTER COLUMN "id" SET DEFAULT gen_random_uuid()::text;
