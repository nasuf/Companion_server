-- Additive ledger metadata. NULL means unknown historical provenance/outcome;
-- never infer success from the old Redis watermark or overwrite old receipts.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE memory_extraction_batches
 ADD COLUMN pipeline_version TEXT,
 ADD COLUMN processing_outcome TEXT CHECK(processing_outcome IN ('processed','no_fact','policy_skipped')),
 ADD COLUMN outcome_reason TEXT,
 ADD COLUMN extraction_provenance JSONB CHECK(extraction_provenance IS NULL OR
   (jsonb_typeof(extraction_provenance)='object' AND octet_length(extraction_provenance::text)<=8192));
COMMIT;
