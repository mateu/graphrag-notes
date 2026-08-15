use super::Migration;

/// A resumable reindex must be pinned to one embedding identity. Otherwise a
/// changed runtime provider could mix vectors in the inactive staging fields.
pub(super) const MIGRATION: Migration = Migration {
    version: 11,
    name: "reindex_job_identity",
    sql: r#"
DEFINE FIELD IF NOT EXISTS target_embedding_provider ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS target_embedding_model ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS target_embedding_dimension ON processing_job TYPE option<int>;
"#,
};
