use super::Migration;

/// Successful authenticated edits, deletions and proposal decisions journal
/// their exact bounded response atomically with every effect. This namespace
/// deliberately leaves v016 capture request identities unchanged. Terminal
/// targets are audit references: deleting them must not invalidate replay.
pub(super) const MIGRATION: Migration = Migration {
    version: 17,
    name: "remote_mutation_receipts",
    sql: r#"
DEFINE TABLE IF NOT EXISTS remote_mutation_receipt SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS instance_id ON remote_mutation_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS request_id ON remote_mutation_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS operation ON remote_mutation_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS target ON remote_mutation_receipt TYPE record;
DEFINE FIELD IF NOT EXISTS payload_fingerprint ON remote_mutation_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS payload ON remote_mutation_receipt TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS result ON remote_mutation_receipt TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS created_at ON remote_mutation_receipt TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at ON remote_mutation_receipt TYPE datetime;
DEFINE INDEX IF NOT EXISTS idx_remote_mutation_request ON remote_mutation_receipt FIELDS instance_id, request_id UNIQUE;
"#,
};
