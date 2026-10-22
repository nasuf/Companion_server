-- Additive; legacy rows remain explicitly uncollected. No inferred backfill.
ALTER TABLE memory_evidence_links
  ADD COLUMN source_statement_time TIMESTAMP(3),
  ADD COLUMN temporal_basis JSONB;
ALTER TABLE memory_evidence_links ADD CONSTRAINT memory_evidence_temporal_object
  CHECK (temporal_basis IS NULL OR jsonb_typeof(temporal_basis) = 'object');
