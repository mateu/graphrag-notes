use super::Migration;

/// Chat ingestion writes provider-shaped metadata on conversations and their
/// derived-note sources. Redefine only those object fields so existing rows
/// and the checksum-immutable v001 schema remain intact.
pub(super) const MIGRATION: Migration = Migration {
    version: 15,
    name: "chat_metadata",
    sql: r#"
DEFINE FIELD OVERWRITE metadata ON conversation TYPE option<object> FLEXIBLE;
DEFINE FIELD OVERWRITE metadata ON source TYPE option<object> FLEXIBLE;
"#,
};
