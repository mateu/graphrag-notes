use super::Migration;

/// Job authority must not disappear when private execution input is damaged.
pub(super) const MIGRATION: Migration = Migration {
    version: 23,
    name: "remote_job_family",
    sql: r#"
DEFINE FIELD IF NOT EXISTS remote_enrichment_job ON processing_job TYPE option<bool>;
UPDATE processing_job SET remote_enrichment_job = (remote_input.enrichment IS NOT NONE AND remote_input.enrichment IS NOT NULL) WHERE job_type = 'remote_upload';
LET $enrichment_receipts = (SELECT instance_id, request_id FROM remote_mutation_receipt WHERE operation IN ['enrich_source', 'rollback_enrichment']);
FOR $receipt IN $enrichment_receipts {
    UPDATE processing_job SET remote_enrichment_job = true WHERE job_type = 'remote_upload' AND remote_instance_id = $receipt.instance_id AND remote_request_id = $receipt.request_id;
};
DEFINE FIELD OVERWRITE remote_enrichment_job ON processing_job TYPE option<bool> READONLY;
"#,
};
