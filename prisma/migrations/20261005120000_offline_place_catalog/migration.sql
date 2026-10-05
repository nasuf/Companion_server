CREATE TABLE offline_place_catalog (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX offline_place_catalog_expiry_idx ON offline_place_catalog(expires_at);

ALTER TABLE offline_activity_recommendations
    ADD COLUMN arrival_verified BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN travel_note_version INTEGER NOT NULL DEFAULT 0;
