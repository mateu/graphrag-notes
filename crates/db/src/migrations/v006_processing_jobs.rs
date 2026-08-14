use super::Migration;

/// Durable local state for bounded inference work.  Results deliberately stay
/// in the local SurrealDB store; this is not a distributed queue.
pub(super) const MIGRATION: Migration = Migration {
    version: 6,
    name: "processing_jobs_and_inference_cache",
    sql: r#"
DEFINE TABLE IF NOT EXISTS processing_job SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS job_type ON processing_job TYPE string;
DEFINE FIELD IF NOT EXISTS source_generation ON processing_job TYPE option<string>;
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

DEFINE TABLE IF NOT EXISTS inference_cache SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS cache_key ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS operation ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS provider ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS model ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS version ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS input_hash ON inference_cache TYPE string;
DEFINE FIELD IF NOT EXISTS cache_value ON inference_cache TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS created_at ON inference_cache TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON inference_cache TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_inference_cache_key ON inference_cache FIELDS cache_key UNIQUE;
"#,
};
