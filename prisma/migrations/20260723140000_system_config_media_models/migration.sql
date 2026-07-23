-- Multimodal model config (admin runtime config, global only):
-- vision_model  → image understanding (Doubao vision via Ark)
-- asr_model     → speech-to-text (Fun-ASR via Dashscope)
-- NULL = fall back to env DOUBAO_VISION_MODEL / DASHSCOPE_ASR_MODEL.
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "vision_model" TEXT;
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "asr_model" TEXT;
