//! Canonical record-ID conversion shared by repository domains.
//!
//! Public entry points accept the user-facing `table:key` form; internal
//! calls use `RecordId` and never construct a second textual table prefix.

use crate::{DbError, Result};
use surrealdb::types::RecordId;

/// Parse a canonical `table:key` ID for proposal and edge CLI actions.
pub fn parse_record_id(value: &str, expected_table: Option<&str>) -> Result<RecordId> {
    let (table, key) = value.trim().split_once(':').ok_or_else(|| {
        DbError::QueryFailed(format!("expected table:key record id, got {value:?}"))
    })?;
    if table.is_empty()
        || key.is_empty()
        || expected_table.is_some_and(|expected| expected != table)
    {
        return Err(DbError::QueryFailed(format!(
            "unexpected record id {value:?}"
        )));
    }
    if let Ok(uuid) = key.parse::<surrealdb_types::Uuid>() {
        return Ok(RecordId::new(table, uuid));
    }
    if let Ok(number) = key.parse::<i64>() {
        return Ok(RecordId::new(table, number));
    }
    Ok(RecordId::new(table, key))
}

/// Accept either a raw note key or its public `note:key` serialization.
/// Callers that already hold a `RecordId` must bind it directly instead.
pub(crate) fn normalize_note_id(note_id: &str) -> RecordId {
    RecordId::new("note", note_id.strip_prefix("note:").unwrap_or(note_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::record_id_to_string;

    #[test]
    fn parses_raw_keys_and_canonical_table_keys_without_double_prefixing() {
        assert_eq!(
            record_id_to_string(&normalize_note_id("raw-key")),
            "note:raw-key"
        );
        assert_eq!(
            record_id_to_string(&normalize_note_id("note:raw-key")),
            "note:raw-key"
        );
        assert_eq!(
            record_id_to_string(&parse_record_id("note:raw-key", Some("note")).unwrap()),
            "note:raw-key"
        );
        assert!(parse_record_id("entity:raw-key", Some("note")).is_err());
    }
}
