-- Prompt ops closure: bind prompt changes to eval validation and support canary rollout.
ALTER TABLE "prompt_template_versions"
    ADD COLUMN "eval_result" JSONB;

ALTER TABLE "prompt_templates"
    ADD COLUMN "canary_config" JSONB;
