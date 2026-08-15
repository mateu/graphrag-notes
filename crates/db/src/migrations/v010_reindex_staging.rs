use super::Migration;

/// Stage a complete reindex without exposing mixed-model vectors to search.
pub(super) const MIGRATION: Migration = Migration {
    version: 10,
    name: "reindex_staging",
    sql: r#"
DEFINE FIELD IF NOT EXISTS reindex_embedding ON note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS reindex_embedding ON message TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS reindex_summary_embedding ON conversation TYPE option<array<float>>;
"#,
};
