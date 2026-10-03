use super::Migration;

/// User-owned uploads retain admission, execution input and outcomes across
/// restart/backup. Worker credentials are never accepted or persisted here.
pub(super) const MIGRATION: Migration = Migration {
    version: 18,
    name: "remote_upload_jobs",
    sql: r#"
DEFINE FIELD IF NOT EXISTS remote_instance_id ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_request_id ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_payload_fingerprint ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_input ON processing_job TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS remote_admission ON processing_job TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS remote_result ON processing_job TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS remote_source_id ON processing_job TYPE option<record<source>>;
DEFINE FIELD IF NOT EXISTS remote_source_uri ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_source_generation ON processing_job TYPE option<int>;
DEFINE FIELD IF NOT EXISTS remote_admission_order ON processing_job TYPE option<int>;
DEFINE FIELD IF NOT EXISTS remote_phase ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_cancel_requested ON processing_job TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS remote_service_epoch ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS remote_worker_token ON processing_job TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_remote_upload_request ON processing_job FIELDS remote_instance_id, remote_request_id UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_remote_upload_owner ON processing_job FIELDS remote_instance_id, updated_at;
"#,
};
