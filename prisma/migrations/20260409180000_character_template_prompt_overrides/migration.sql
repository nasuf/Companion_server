-- Add prompt_header and prompt_requirements columns for editable prompt overrides.
-- NULL means "use system default" (defined in app/services/character.py).

ALTER TABLE "character_templates"
  ADD COLUMN "prompt_header" TEXT,
  ADD COLUMN "prompt_requirements" TEXT;
