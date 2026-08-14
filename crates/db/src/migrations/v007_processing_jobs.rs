use super::Migration;

/// Durable local state for bounded inference work. Results remain local to the
/// SurrealDB store; this is deliberately not a distributed queue.
pub(super) const MIGRATION: Migration = Migration {
    version: 7,
    name: "processing_jobs",
    sql: r#"
DEFINE TABLE IF NOT EXISTS processing_job SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS job_type ON processing_job TYPE string;
DEFINE FIELD IF NOT EXISTS source_generation ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS scope ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS item_ids ON processing_job TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS status ON processing_job TYPE string;
DEFINE FIELD IF NOT EXISTS total_count ON processing_job TYPE int;
DEFINE FIELD IF NOT EXISTS completed_count ON processing_job TYPE int;
DEFINE FIELD IF NOT EXISTS failed_count ON processing_job TYPE int;
DEFINE FIELD IF NOT EXISTS checkpoint ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS last_error ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at ON processing_job TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON processing_job TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS finished_at ON processing_job TYPE option<datetime>;
DEFINE INDEX IF NOT EXISTS idx_processing_job_status ON processing_job FIELDS status, updated_at;
"#,
};
