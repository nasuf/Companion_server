-- Additive immutable provider/session snapshot; old cards remain readable.
ALTER TABLE offline_activity_recommendations
  ADD COLUMN discovery_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
