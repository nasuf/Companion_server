ALTER TABLE chat_message_attachments
    ADD COLUMN IF NOT EXISTS duration_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS transcription_status TEXT,
    ADD COLUMN IF NOT EXISTS transcription_text TEXT,
    ADD COLUMN IF NOT EXISTS transcription_model TEXT,
    ADD COLUMN IF NOT EXISTS transcription_request_id TEXT,
    ADD COLUMN IF NOT EXISTS transcription_error TEXT;

ALTER TABLE chat_message_attachments
    ADD CONSTRAINT chat_message_attachments_duration_positive_check
    CHECK (duration_seconds IS NULL OR duration_seconds > 0);
