ALTER TABLE offline_activity_media
ADD COLUMN IF NOT EXISTS duration_seconds INTEGER;

ALTER TABLE offline_activity_feedback
ADD COLUMN IF NOT EXISTS audio_attachment_id TEXT;
