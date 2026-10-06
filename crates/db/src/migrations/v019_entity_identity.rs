use super::Migration;

/// Preserve every legacy entity ID and mention. This changes the uniqueness
/// contract, not the labels or extracted data: reprocessing remains explicit.
pub(super) const MIGRATION: Migration = Migration {
    version: 19,
    name: "entity_identity",
    sql: r#"
DEFINE FIELD IF NOT EXISTS identity_key ON entity TYPE string VALUE $value ?? string::concat('legacy:', canonical_name);
DEFINE FIELD IF NOT EXISTS metadata.extraction ON entity TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS metadata ON mentions TYPE option<object> FLEXIBLE;
UPDATE entity SET identity_key = string::concat('legacy:', canonical_name) WHERE identity_key IS NONE;
DEFINE INDEX IF NOT EXISTS idx_entity_identity ON entity FIELDS identity_key UNIQUE;
REMOVE INDEX idx_entity_canonical ON entity;
DEFINE INDEX idx_entity_canonical ON entity FIELDS canonical_name;
DEFINE INDEX IF NOT EXISTS idx_mentions_entity_note ON mentions FIELDS out, in;
"#,
};
