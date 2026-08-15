use super::Migration;

/// Persist the source-text snapshot for each staged reindex item so resumed
/// jobs cannot publish vectors computed before an edit.
pub(super) const MIGRATION: Migration = Migration {
    version: 12,
    name: "reindex_item_fingerprints",
    sql: r#"
DEFINE FIELD IF NOT EXISTS reindex_item_fingerprints ON processing_job TYPE option<object> FLEXIBLE;
"#,
};
