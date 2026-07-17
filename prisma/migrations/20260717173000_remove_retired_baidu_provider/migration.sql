-- The retired provider briefly shipped before being removed. Clear any
-- selections made during that window before deleting all of its registry rows.
UPDATE "system_config"
SET
    "remote_chat_model" = CASE
        WHEN "remote_chat_provider" = 'qianfan'
          OR ("remote_chat_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_chat_model" END,
    "remote_chat_provider" = CASE
        WHEN "remote_chat_provider" = 'qianfan'
          OR ("remote_chat_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_chat_provider" END,
    "remote_small_model" = CASE
        WHEN "remote_small_provider" = 'qianfan'
          OR ("remote_small_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_small_model" END,
    "remote_small_provider" = CASE
        WHEN "remote_small_provider" = 'qianfan'
          OR ("remote_small_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_small_provider" END,
    "remote_provider" = CASE
        WHEN "remote_provider" = 'qianfan' THEN NULL ELSE "remote_provider" END
WHERE "remote_provider" = 'qianfan'
   OR "remote_chat_provider" = 'qianfan'
   OR "remote_small_provider" = 'qianfan';

UPDATE "agent_config_overrides"
SET
    "remote_chat_model" = CASE
        WHEN "remote_chat_provider" = 'qianfan'
          OR ("remote_chat_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_chat_model" END,
    "remote_chat_provider" = CASE
        WHEN "remote_chat_provider" = 'qianfan'
          OR ("remote_chat_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_chat_provider" END,
    "remote_small_model" = CASE
        WHEN "remote_small_provider" = 'qianfan'
          OR ("remote_small_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_small_model" END,
    "remote_small_provider" = CASE
        WHEN "remote_small_provider" = 'qianfan'
          OR ("remote_small_provider" IS NULL AND "remote_provider" = 'qianfan')
        THEN NULL ELSE "remote_small_provider" END,
    "remote_provider" = CASE
        WHEN "remote_provider" = 'qianfan' THEN NULL ELSE "remote_provider" END
WHERE "remote_provider" = 'qianfan'
   OR "remote_chat_provider" = 'qianfan'
   OR "remote_small_provider" = 'qianfan';

DELETE FROM "model_registry" WHERE "provider" = 'qianfan';
