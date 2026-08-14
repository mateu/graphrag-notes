use super::Migration;

/// Local aliases used to seed bounded graph retrieval without an inference
/// provider call. This must remain additive: v001 is checksum-immutable for
/// databases that were initialized before aliases existed.
pub(super) const MIGRATION: Migration = Migration {
    version: 9,
    name: "entity_graph_aliases",
    sql: r#"
DEFINE FIELD IF NOT EXISTS metadata.aliases ON entity TYPE option<array<string>>;
"#,
};
