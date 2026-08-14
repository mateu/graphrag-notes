use super::Migration;

/// Runtime compatibility metadata is deliberately separate from the immutable
/// v001 baseline.  One row is retained for the active embedded corpus; a
/// future reindex command may advance its generation rather than changing the
/// vector index dimensions in place.
pub(super) const MIGRATION: Migration = Migration {
    version: 2,
    name: "embedding_metadata",
    sql: r#"
DEFINE TABLE IF NOT EXISTS graphrag_metadata SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS key ON graphrag_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS application_schema_version ON graphrag_metadata TYPE int;
DEFINE FIELD IF NOT EXISTS embedding_provider ON graphrag_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS embedding_model ON graphrag_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS embedding_dimension ON graphrag_metadata TYPE int;
DEFINE FIELD IF NOT EXISTS extraction_provider ON graphrag_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS extraction_model ON graphrag_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS generation ON graphrag_metadata TYPE int DEFAULT 1;
DEFINE FIELD IF NOT EXISTS last_reindex_at ON graphrag_metadata TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS last_reindex_status ON graphrag_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS updated_at ON graphrag_metadata TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_graphrag_metadata_key ON graphrag_metadata FIELDS key UNIQUE;
"#,
};
