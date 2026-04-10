-- Add career_id foreign key to character_profiles
ALTER TABLE "character_profiles" ADD COLUMN "career_id" TEXT;

CREATE INDEX "character_profiles_career_id_idx" ON "character_profiles"("career_id");

ALTER TABLE "character_profiles" ADD CONSTRAINT "character_profiles_career_id_fkey"
  FOREIGN KEY ("career_id") REFERENCES "career_templates"("id") ON DELETE SET NULL ON UPDATE CASCADE;
