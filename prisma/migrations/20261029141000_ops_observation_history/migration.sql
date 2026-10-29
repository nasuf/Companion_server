CREATE TABLE ops_service_observations (
 sampled_at timestamptz PRIMARY KEY,
 checks jsonb NOT NULL
);
CREATE INDEX quality_charges_month_idx ON quality_charges(month);
