use super::Migration;

/// Successful remote captures retain their original safe response in the same
/// transaction as the note. Instance identity comes from authentication; no
/// credential material is stored. Receipts are logical backup data because
/// forgetting one during restore would make a lost-response replay duplicate
/// a previously committed capture.
pub(super) const MIGRATION: Migration = Migration {
    version: 16,
    name: "remote_capture_receipts",
    sql: r#"
DEFINE TABLE IF NOT EXISTS remote_capture_receipt SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS instance_id ON remote_capture_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS request_id ON remote_capture_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS payload_fingerprint ON remote_capture_receipt TYPE string;
DEFINE FIELD IF NOT EXISTS payload ON remote_capture_receipt TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS result ON remote_capture_receipt TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS note_id ON remote_capture_receipt TYPE record<note>;
DEFINE FIELD IF NOT EXISTS source_id ON remote_capture_receipt TYPE record<source>;
DEFINE FIELD IF NOT EXISTS created_at ON remote_capture_receipt TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at ON remote_capture_receipt TYPE datetime;
DEFINE INDEX IF NOT EXISTS idx_remote_capture_request ON remote_capture_receipt FIELDS instance_id, request_id UNIQUE;
"#,
};
