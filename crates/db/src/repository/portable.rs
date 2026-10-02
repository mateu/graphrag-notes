//! Portable logical-data archive ownership.
//!
//! Archive restore is intentionally one-record-at-a-time. This keeps export
//! memory bounded and makes the timestamp/reference conversions part of the
//! same query that recreates each canonical record ID. It does not own schema
//! migrations; migration history remains in `crate::migrations`.

use super::*;
use surrealdb_types::ToSql;

/// Tables that form the portable, logical GraphRAG data model. Runtime caches,
/// processing-job checkpoints, and migration history are intentionally absent:
/// they are machine-local implementation state rather than recoverable user
/// knowledge.
pub const PORTABLE_TABLES: &[&str] = &[
    "source",
    "note",
    "entity",
    "conversation",
    "message",
    "supports",
    "contradicts",
    "derived_from",
    "related_to",
    "mentions",
    "note_from_conversation",
    "note_from_message",
    "proposed_edge",
    "graphrag_metadata",
];

fn validate_portable_table(table: &str) -> Result<()> {
    if PORTABLE_TABLES.contains(&table) {
        Ok(())
    } else {
        Err(DbError::QueryFailed(format!(
            "{table} is not a portable backup table"
        )))
    }
}

fn validate_portable_field(field: &str) -> Result<()> {
    let mut chars = field.bytes();
    let Some(first) = chars.next() else {
        return Err(DbError::QueryFailed(
            "portable record has an empty field name".into(),
        ));
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !chars.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(DbError::QueryFailed(format!(
            "portable record has unsafe field name {field:?}"
        )));
    }
    Ok(())
}

/// Decode the canonical SurrealQL IDs emitted by Surreal's JSON serializer.
/// Quoted string keys must remain strings even when they resemble a UUID or
/// number. The CLI's unquoted-ID convenience parser cannot make that distinction.
fn parse_portable_record_id(value: &str, expected_table: Option<&str>) -> Result<RecordId> {
    let value = value.trim();
    let (_, key) = value
        .split_once(':')
        .ok_or_else(|| DbError::QueryFailed(format!("invalid portable record id {value:?}")))?;
    let encoded_key = key.trim_start();
    let encoded_key = encoded_key.starts_with(['`', '⟨', '[', '{'])
        || encoded_key.starts_with("u'")
        || encoded_key.starts_with("u\"");
    let id = match surrealdb::parse::value(value) {
        // Require an exact canonical round trip: the SDK literal parser can
        // accept a prefix, which must never truncate an archive's record key.
        Ok(surrealdb_types::Value::RecordId(id)) if id.to_sql() == value => id,
        _ if !encoded_key => {
            // Older archives used the CLI's unquoted table:key convention.
            // Preserve its UUID/numeric inference and raw string-key behavior.
            parse_record_id(value, expected_table)?
        }
        _ => {
            return Err(DbError::QueryFailed(format!(
                "invalid canonical portable record id {value:?}"
            )));
        }
    };
    if expected_table.is_some_and(|expected| id.table.as_str() != expected) {
        return Err(DbError::QueryFailed(format!(
            "unexpected portable record id {value:?}"
        )));
    }
    Ok(id)
}

/// Remove JSONL timestamps so their explicit Surreal datetime casts are part
/// of the restore statement instead of relying on implicit string coercion.
fn portable_timestamps(
    table: &str,
    record: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<(String, String)>> {
    let fields: &[&str] = match table {
        "source" => &["created_at", "updated_at", "last_ingested_at"],
        "note" => &["created_at", "updated_at"],
        "entity" => &["created_at"],
        "conversation" => &["created_at", "updated_at", "ingested_at"],
        "message" => &["created_at", "updated_at", "ingested_at"],
        "supports"
        | "contradicts"
        | "derived_from"
        | "related_to"
        | "mentions"
        | "note_from_conversation"
        | "note_from_message" => &["created_at"],
        "proposed_edge" => &["created_at", "updated_at", "reviewed_at", "superseded_at"],
        "graphrag_metadata" => &["last_reindex_at", "updated_at"],
        _ => &[],
    };
    let mut values = Vec::new();
    for field in fields {
        let Some(value) = record.remove(*field) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let value = value.as_str().ok_or_else(|| {
            DbError::QueryFailed(format!(
                "portable {table}.{field} timestamp is not a string"
            ))
        })?;
        values.push(((*field).to_string(), value.to_string()));
    }
    Ok(values)
}

/// Convert JSONL's canonical `table:key` references back into typed Surreal
/// record IDs before binding schemafull record fields.
fn portable_record_ids(
    table: &str,
    record: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<(String, RecordId)>> {
    let fields: &[(&str, Option<&str>)] = match table {
        "note" => &[("source_id", Some("source"))],
        "message" => &[("conversation_id", Some("conversation"))],
        "supports" | "contradicts" | "derived_from" | "related_to" => &[
            ("in", Some("note")),
            ("out", Some("note")),
            ("proposal_id", Some("proposed_edge")),
        ],
        "mentions" => &[("in", Some("note")), ("out", Some("entity"))],
        "note_from_conversation" => &[("in", Some("note")), ("out", Some("conversation"))],
        "note_from_message" => &[("in", Some("note")), ("out", Some("message"))],
        "proposed_edge" => &[
            ("in", Some("note")),
            ("out", Some("note")),
            ("resulting_edge_id", None),
        ],
        _ => &[],
    };
    let mut values = Vec::new();
    for (field, expected_table) in fields {
        let Some(value) = record.remove(*field) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let value = value.as_str().ok_or_else(|| {
            DbError::QueryFailed(format!(
                "portable {table}.{field} reference is not a string"
            ))
        })?;
        values.push((
            (*field).to_string(),
            parse_portable_record_id(value, *expected_table)?,
        ));
    }
    Ok(values)
}

impl Repository {
    /// Read one bounded page of a portable logical table in deterministic
    /// record-id order. Callers advance `offset` instead of materializing a
    /// full database in memory while writing an archive.
    #[instrument(skip(self))]
    pub async fn portable_records_page(
        &self,
        table: &str,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        validate_portable_table(table)?;
        self.db
            .query(format!(
                "SELECT * FROM {table} ORDER BY id ASC LIMIT $limit START $offset"
            ))
            .bind(("limit", limit.max(1)))
            .bind(("offset", offset))
            .await?
            .take(0)
            .map_err(Into::into)
    }

    /// Insert one record from a validated portable archive while preserving
    /// its logical Surreal record ID. References in the archive therefore do
    /// not need a lossy best-effort remapping step.
    #[instrument(skip(self, record))]
    pub async fn restore_portable_record(
        &self,
        table: &str,
        record: serde_json::Value,
    ) -> Result<()> {
        validate_portable_table(table)?;
        let id = record.get("id").cloned().ok_or_else(|| {
            DbError::QueryFailed(format!("portable {table} record is missing its id"))
        })?;
        let id = if let Some(id) = id.as_str() {
            parse_portable_record_id(id, Some(table))?
        } else {
            serde_json::from_value::<RecordId>(id).map_err(|error| {
                DbError::QueryFailed(format!(
                    "portable {table} record has an invalid id: {error}"
                ))
            })?
        };
        if id.table.as_str() != table {
            return Err(DbError::QueryFailed(format!(
                "portable record id {} does not belong to {table}",
                record_id_to_string(&id)
            )));
        }
        let mut content = record;
        let object = content.as_object_mut().ok_or_else(|| {
            DbError::QueryFailed(format!("portable {table} record is not an object"))
        })?;
        object.remove("id");
        // Surreal represents NONE as JSON null, but a JSON null bound back
        // through CONTENT is SQL NULL and is rejected by many option fields.
        // Omission restores the same NONE/default state.
        object.retain(|_, value| !value.is_null());
        let timestamps = portable_timestamps(table, object)?;
        let references = portable_record_ids(table, object)?;
        let mut assignments = Vec::new();
        for field in object.keys() {
            validate_portable_field(field)?;
            assignments.push(format!("{field} = ${field}"));
        }
        for (field, _) in &timestamps {
            assignments.push(format!("{field} = <datetime>${field}"));
        }
        for (field, _) in &references {
            assignments.push(format!("{field} = ${field}"));
        }
        if assignments.is_empty() {
            return Err(DbError::QueryFailed(format!(
                "portable {table} record has no restorable fields"
            )));
        }
        let mut query = self
            .db
            .query(format!("CREATE $id SET {}", assignments.join(", ")))
            .bind(("id", id));
        for (field, value) in object {
            query = query.bind((field.as_str(), value.clone()));
        }
        for (field, timestamp) in timestamps {
            query = query.bind((field, timestamp));
        }
        for (field, reference) in references {
            query = query.bind((field, reference));
        }
        query.await?.check()?;
        Ok(())
    }

    /// Return persisted model metadata for a portable vector export. The
    /// caller must refuse `--include-embeddings` when this is absent, because
    /// an unlabelled vector payload is not safely portable.
    pub async fn portable_embedding_metadata(
        &self,
    ) -> Result<Option<crate::compatibility::EmbeddingMetadata>> {
        crate::compatibility::embedding_metadata(&self.db).await
    }

    /// Count active vector-bearing records independently of metadata. This is
    /// used to keep a legacy, unlabelled corpus from accepting a partial model
    /// cutover that would make the global identity dishonest.
    pub async fn vector_bearing_record_count(&self) -> Result<usize> {
        crate::compatibility::vector_bearing_record_count(&self.db).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;
    use graphrag_core::{record_id_to_string, Source};

    fn typed_note_ids() -> Vec<RecordId> {
        let uuid = "ab8d239c-c227-4642-a472-2672fba13f0b";
        vec![
            RecordId::new("note", uuid),
            RecordId::new("note", "42"),
            RecordId::new("note", 42_i64),
            RecordId::new("note", -7_i64),
            RecordId::new("note", uuid.parse::<surrealdb_types::Uuid>().unwrap()),
            RecordId::new("note", "literal ` tick \\ slash: café and space"),
            RecordId::new("note", "control\0\r\t\n\u{8}\u{c}characters"),
        ]
    }

    #[test]
    fn portable_id_decoder_preserves_canonical_key_types_and_escapes() {
        for id in typed_note_ids() {
            let canonical = id.to_sql();
            assert_eq!(
                parse_portable_record_id(&canonical, Some("note")).unwrap(),
                id,
                "canonical ID {canonical} must preserve its exact typed key"
            );
        }
    }

    #[test]
    fn portable_id_decoder_preserves_legacy_unquoted_archive_ids() {
        for legacy in [
            "note:old_key",
            "note:legacy-key with spaces",
            "note:42",
            "note:ab8d239c-c227-4642-a472-2672fba13f0b",
        ] {
            assert_eq!(
                parse_portable_record_id(legacy, Some("note")).unwrap(),
                parse_record_id(legacy, Some("note")).unwrap()
            );
        }
    }

    #[test]
    fn portable_id_decoder_rejects_malformed_encoded_keys_and_wrong_tables() {
        for malformed in [
            "note:`unterminated",
            "note:`valid` trailing",
            "note: `valid` trailing",
            "note:u'not-a-uuid'",
            "note:[unclosed",
            "note:{unclosed",
            "note:",
            "missing-table-prefix",
            "source:`wrong table`",
        ] {
            assert!(
                parse_portable_record_id(malformed, Some("note")).is_err(),
                "malformed ID {malformed:?} must not become a literal key"
            );
        }
    }

    #[tokio::test]
    async fn canonical_export_restore_preserves_typed_ids_and_references() {
        let original = Repository::new(init_memory().await.unwrap());
        let source_id = RecordId::new("source", "7c555ed2-cf07-4e5b-a446-5a552d2ac65e");
        original
            .db
            .query(
                "CREATE $id SET source_type = 'manual', title = 'typed-ID fixture', \
                 metadata = {}, created_at = time::now(), updated_at = time::now()",
            )
            .bind(("id", source_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let mut ids = typed_note_ids();
        for (ordinal, id) in ids.iter().enumerate() {
            original
                .db
                .query(
                    "CREATE $id SET content = $content, note_type = 'raw', tags = [], \
                     source_id = $source, created_at = time::now(), updated_at = time::now()",
                )
                .bind(("id", id.clone()))
                .bind(("content", format!("portable typed-ID note {ordinal}")))
                .bind(("source", source_id.clone()))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for target in &ids[1..] {
            original
                .create_edge(&ids[0], target, EdgeType::Supports, None)
                .await
                .unwrap();
        }

        let restored = Repository::new(init_memory().await.unwrap());
        for table in ["source", "note", "supports"] {
            let page = original.portable_records_page(table, 0, 100).await.unwrap();
            for record in &page {
                // Exercise the actual export serializer and JSONL decoding,
                // rather than manufacturing raw IDs with the CLI formatter.
                let jsonl = serde_json::to_vec(record).unwrap();
                let record = serde_json::from_slice(&jsonl).unwrap();
                restored
                    .restore_portable_record(table, record)
                    .await
                    .unwrap();
            }
            assert_eq!(
                restored.portable_records_page(table, 0, 100).await.unwrap(),
                page,
                "portable {table} records must export with identical IDs and references after restore"
            );
        }
        let restored_notes: Vec<Note> = restored.db.select("note").await.unwrap();
        assert!(restored_notes
            .iter()
            .all(|note| note.source_id.as_ref() == Some(&source_id)));
        let mut restored_ids = restored_notes
            .into_iter()
            .map(|note| note.id.unwrap())
            .collect::<Vec<_>>();
        ids.sort();
        restored_ids.sort();
        assert_eq!(restored_ids, ids);
        let mut response = restored
            .db
            .query("SELECT VALUE in FROM supports; SELECT VALUE out FROM supports;")
            .await
            .unwrap();
        let from_ids: Vec<RecordId> = response.take(0).unwrap();
        let mut to_ids: Vec<RecordId> = response.take(1).unwrap();
        assert!(from_ids.iter().all(|id| id == &typed_note_ids()[0]));
        let mut expected_targets = typed_note_ids()[1..].to_vec();
        expected_targets.sort();
        to_ids.sort();
        assert_eq!(to_ids, expected_targets);
    }

    #[tokio::test]
    async fn portable_pages_preserve_canonical_record_id_order() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_source(Source::manual().with_title("first"))
            .await
            .unwrap();
        repo.create_source(Source::manual().with_title("second"))
            .await
            .unwrap();

        let first = repo.portable_records_page("source", 0, 1).await.unwrap();
        let second = repo.portable_records_page("source", 1, 1).await.unwrap();
        let first_id = first[0]["id"].as_str().unwrap();
        let second_id = second[0]["id"].as_str().unwrap();
        let first_id = parse_portable_record_id(first_id, Some("source")).unwrap();
        let second_id = parse_portable_record_id(second_id, Some("source")).unwrap();

        assert!(record_id_to_string(&first_id) < record_id_to_string(&second_id));
    }
}
