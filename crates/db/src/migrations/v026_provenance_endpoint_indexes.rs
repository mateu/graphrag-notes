use super::Migration;

/// Add non-unique endpoint indexes without rewriting provenance or record IDs.
pub(super) const MIGRATION: Migration = Migration {
    version: 26,
    name: "provenance_endpoint_indexes",
    sql: r#"
DEFINE INDEX IF NOT EXISTS idx_note_from_conversation_in ON note_from_conversation FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_note_from_message_in ON note_from_message FIELDS in;
"#,
};
