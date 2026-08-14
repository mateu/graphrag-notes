use super::Migration;

/// Durable local inference cache. Processing-job state is intentionally added
/// by a later migration so this cache foundation can stand alone.
pub(super) const MIGRATION: Migration = Migration {
    version: 6,
    name: "inference_cache",
    sql: r#"
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
