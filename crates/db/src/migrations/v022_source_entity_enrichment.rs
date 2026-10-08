use super::Migration;

/// Older workers must refuse stores containing generation-preserving graph policy state.
pub(super) const MIGRATION: Migration = Migration {
    version: 22,
    name: "source_entity_enrichment",
    sql: r#"
DEFINE FIELD IF NOT EXISTS metadata.graph_policy_revision ON source TYPE option<int> ASSERT $value IS NONE OR $value > 0;
"#,
};
