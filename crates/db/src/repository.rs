//! Repository pattern for database operations

use crate::{
    compatibility::{
        check_embedding_compatibility, record_embedding_metadata, CompatibilityState,
        EmbeddingIdentity, ExtractionIdentity,
    },
    fusion::{self, FusionConfig, FusionEvidence, FusionRecord},
    migrations, DbConnection, DbError, Result,
};
use chrono::{DateTime, Utc};
use graphrag_core::{
    record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, Note, ProposedEdge,
    ProposedEdgeStatus, Source, SourceIngestionStatus, SourceType,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;
use tokio::sync::Mutex;
use tracing::instrument;

/// Repository for all database operations
#[derive(Clone)]
pub struct Repository {
    db: DbConnection,
    proposal_acceptance_lock: Arc<Mutex<()>>,
}

// A source generation becomes visible only after promotion. Legacy/manual
// notes have no generation and remain visible, while staged and superseded
// file-import notes are excluded from every user-facing scan.
const VISIBLE_NOTE_CONDITION: &str = "(source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)";

// Edge rows contain record references in `in` and `out`. Resolve both note
// endpoints through the note table before showing graph topology so an
// interrupted import cannot expose relationships owned by a staged or
// superseded generation.
const VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION: &str = "in IN (SELECT VALUE id FROM note WHERE (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)) AND out IN (SELECT VALUE id FROM note WHERE (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation))";

fn graph_query_normalize(query: &str) -> String {
    // Entity canonicalization retains arbitrary Unicode text. Split only at
    // lexical boundaries so adjacent punctuation (for example `Atlas?`) does
    // not become part of a term, while CJK and other non-Latin letters remain
    // intact rather than being treated as ASCII-only words.
    Entity::canonicalize(query)
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn graph_prefix_terms(normalized_query: &str) -> Vec<String> {
    normalized_query
        .split_whitespace()
        .filter(|term| term.chars().count() >= 4 && !is_graph_prefix_stop_word(term))
        .map(str::to_string)
        .collect()
}

fn is_graph_prefix_stop_word(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "can"
            | "changed"
            | "change"
            | "could"
            | "did"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "recent"
            | "show"
            | "tell"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "will"
            | "with"
            | "would"
            | "you"
    )
}

/// Match tiers for local graph-entity seeding. Keeping exact equality ahead
/// of contained phrases means a specific entity query cannot be crowded out
/// by a shorter entity name when the caller supplies a small seed cap.
#[derive(Clone, Copy)]
enum GraphEntityMatchTier {
    Exact,
    ContainedPhrase,
    Prefix,
}

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

/// Remove serialized ISO-8601 timestamps from raw JSON content and return the
/// values that must be restored through Surreal's explicit datetime cast.
/// JSONL deliberately uses ordinary JSON strings; binding those strings into a
/// schemafull datetime field would otherwise be rejected by SurrealDB.
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
/// record IDs. Generic JSON content bindings would otherwise treat them as
/// strings and schemafull record fields would reject the restored record.
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
            parse_record_id(value, *expected_table)?,
        ));
    }
    Ok(values)
}

fn count_to_i64(count: u64) -> Result<i64> {
    i64::try_from(count)
        .map_err(|_| DbError::QueryFailed("processing count exceeds database integer range".into()))
}

/// Derive the default full-text value from a note's displayed content and its
/// Markdown heading metadata.
fn derived_search_content(note: &Note) -> String {
    let headings = note
        .chunk_heading_path
        .iter()
        .filter(|heading| !heading.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ");
    if headings.is_empty() {
        note.content.clone()
    } else {
        format!("{headings}\n\n{}", note.content)
    }
}

/// Resolve search text for a note update without treating every existing
/// value as derived. A caller may intentionally supply aliases or other
/// custom searchable text; keep those for metadata-only updates and explicit
/// replacements. Rebuild only when the persisted value was the old derived
/// Markdown/body value carried through a content or heading-context edit.
fn search_content_for_note_update(existing: &Note, replacement: &Note) -> String {
    let existing_derived = derived_search_content(existing);
    let existing_search = existing
        .search_content
        .as_deref()
        .unwrap_or(existing_derived.as_str());
    if let Some(replacement_search) = replacement.search_content.as_deref() {
        if replacement_search != existing_search {
            return replacement_search.to_string();
        }
    }

    let source_changed = existing.content != replacement.content
        || existing.chunk_heading_path != replacement.chunk_heading_path;
    if source_changed && existing_search == existing_derived {
        derived_search_content(replacement)
    } else {
        existing_search.to_string()
    }
}

/// The coarse operation of a durable local processing job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingJobType {
    Embedding,
    EntityExtraction,
    Reindex,
}

impl ProcessingJobType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::EntityExtraction => "entity_extraction",
            Self::Reindex => "reindex",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "embedding" => Some(Self::Embedding),
            "entity_extraction" => Some(Self::EntityExtraction),
            "reindex" => Some(Self::Reindex),
            _ => None,
        }
    }
}

/// State transitions are intentionally small: workers own `running`, while
/// command handlers can request cancellation between atomic item mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ProcessingJobStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Persisted checkpoint and aggregate counts for a local processing run.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct ProcessingJob {
    pub id: Option<RecordId>,
    pub job_type: String,
    pub source_generation: Option<String>,
    pub scope: Option<String>,
    /// Reindex jobs pin this identity so a later resume cannot apply a
    /// different provider/model's staged vectors to the same generation.
    #[serde(default)]
    pub target_embedding_provider: Option<String>,
    #[serde(default)]
    pub target_embedding_model: Option<String>,
    #[serde(default)]
    pub target_embedding_dimension: Option<i64>,
    /// Snapshot text fingerprints for reindex items. They make a checkpoint
    /// safe to resume even when an item was edited after its vector staged.
    #[serde(default)]
    pub reindex_item_fingerprints: Option<std::collections::BTreeMap<String, String>>,
    /// Renewable owner lease for a reindex worker. An expired owner may be
    /// replaced after a crash, but a live owner cannot be resumed concurrently.
    #[serde(default)]
    pub reindex_lease_owner: Option<String>,
    #[serde(default)]
    pub reindex_lease_expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub item_ids: Vec<String>,
    pub status: String,
    pub total_count: i64,
    pub completed_count: i64,
    pub failed_count: i64,
    pub checkpoint: Option<String>,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl ProcessingJob {
    pub fn job_type_enum(&self) -> Option<ProcessingJobType> {
        ProcessingJobType::parse(&self.job_type)
    }
}

/// Partial update with explicit nested options for nullable fields: `None`
/// leaves a field unchanged; `Some(None)` clears it.
#[derive(Debug, Clone, Default)]
pub struct ProcessingJobUpdate {
    pub status: Option<ProcessingJobStatus>,
    pub completed_count: Option<u64>,
    pub failed_count: Option<u64>,
    pub checkpoint: Option<Option<String>>,
    pub last_error: Option<Option<String>>,
    pub finish: bool,
}

/// An audited cache entry. `cache_key` is computed by the processing layer;
/// remaining fields make collision/debug inspection deterministic.
#[derive(Debug, Clone)]
pub struct InferenceCacheEntry {
    pub cache_key: String,
    pub operation: String,
    pub provider: String,
    pub model: String,
    pub version: String,
    pub input_hash: String,
    pub value: serde_json::Value,
}

/// One durable reindex item. Its id determines whether `text` comes from a
/// note, chat message, or conversation summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexItem {
    pub id: String,
    /// Exact canonical input submitted to the embedding provider.
    pub text: String,
    /// Source fields from which `text` was derived. Commit validates this
    /// snapshot transactionally before making the staged vector visible.
    pub source_snapshot: serde_json::Value,
}

fn canonical_note_reindex_text(
    content: &str,
    search_content: Option<&str>,
    chunk_heading_path: &[String],
) -> String {
    search_content.map_or_else(
        || {
            let headings = chunk_heading_path
                .iter()
                .filter(|heading| !heading.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" > ");
            if headings.is_empty() {
                content.to_string()
            } else {
                format!("{headings}\n\n{content}")
            }
        },
        str::to_string,
    )
}

fn canonical_message_reindex_text(content: &str, content_blocks: &serde_json::Value) -> String {
    if !content.trim().is_empty() {
        return content.to_string();
    }
    let parts = content_blocks
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(serde_json::Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if parts.is_empty() {
        "[empty message]".to_string()
    } else {
        parts.join("\n\n")
    }
}

fn canonical_conversation_reindex_text(title: Option<&str>, summary: &str) -> String {
    let title = title
        .filter(|title| !title.is_empty())
        .unwrap_or("Untitled Conversation");
    format!("{title}\n\n{summary}")
}

fn source_content_value(source: &Source) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(source)
        .map_err(|error| DbError::QueryFailed(format!("source serialization failed: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| DbError::QueryFailed("source did not serialize as an object".into()))?;
    object.remove("id");
    object.remove("created_at");
    object.remove("updated_at");
    object.remove("last_ingested_at");
    for key in [
        "uri",
        "content",
        "normalized_uri",
        "content_hash",
        "last_error",
    ] {
        if object.get(key).is_some_and(serde_json::Value::is_null) {
            object.remove(key);
        }
    }
    if object
        .get("metadata")
        .is_some_and(serde_json::Value::is_null)
    {
        object.remove("metadata");
    }
    Ok(value)
}

impl Repository {
    /// Create a new repository
    pub fn new(db: DbConnection) -> Self {
        let proposal_acceptance_lock = db.proposal_lifecycle_lock();
        Self {
            db,
            proposal_acceptance_lock,
        }
    }

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
            parse_record_id(id, Some(table))?
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

    /// Check the active embedding identity before a vector read or write.
    /// This method is read-only and therefore safe to use for search paths.
    pub async fn check_embedding_compatibility(
        &self,
        embedding: &EmbeddingIdentity,
    ) -> Result<CompatibilityState> {
        check_embedding_compatibility(&self.db, embedding).await
    }

    /// Initialize empty-corpus metadata after a successful embedding probe.
    /// Existing metadata is never overwritten by a different model.
    pub async fn record_embedding_metadata(
        &self,
        embedding: &EmbeddingIdentity,
        extraction: Option<&ExtractionIdentity>,
    ) -> Result<CompatibilityState> {
        record_embedding_metadata(&self.db, embedding, extraction).await
    }

    // ==========================================
    // DURABLE INFERENCE PROCESSING
    // ==========================================

    /// Create a legacy/unscoped inference record. `source_generation` is an
    /// opaque durable identifier so jobs remain valid even when their source
    /// record is later reloaded by another process.
    ///
    /// This helper has no persisted item set and therefore cannot be resumed
    /// by a durable worker. New callers must use
    /// [`Self::create_processing_job_with_scope`] so retries have an exact,
    /// resumable scope.
    pub async fn create_processing_job(
        &self,
        job_type: ProcessingJobType,
        source_generation: Option<String>,
        total_count: u64,
    ) -> Result<ProcessingJob> {
        self.create_processing_job_with_scope(
            job_type,
            source_generation,
            total_count,
            None,
            Vec::new(),
        )
        .await
    }

    /// Persist the exact item set selected for a bounded job. Resume uses this
    /// durable scope instead of silently substituting a new page of notes.
    pub async fn create_processing_job_with_scope(
        &self,
        job_type: ProcessingJobType,
        source_generation: Option<String>,
        total_count: u64,
        scope: Option<String>,
        item_ids: Vec<String>,
    ) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "CREATE processing_job SET job_type = $job_type, source_generation = $source_generation, scope = $scope, item_ids = $item_ids, \
                 status = 'running', total_count = $total_count, completed_count = 0, failed_count = 0, \
                 checkpoint = NONE, last_error = NONE, created_at = time::now(), updated_at = time::now(), \
                 finished_at = NONE RETURN AFTER",
            )
            .bind(("job_type", job_type.as_str()))
            .bind(("source_generation", source_generation))
            .bind(("scope", scope))
            .bind(("item_ids", item_ids))
            .bind(("total_count", count_to_i64(total_count)?))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::QueryFailed("create_processing_job".into()))
    }

    /// Persist a reindex job together with the exact embedding identity it is
    /// allowed to stage and promote. This is intentionally separate from the
    /// generic inference-job constructor: regular jobs have no corpus-wide
    /// model cutover contract.
    pub async fn create_reindex_processing_job(
        &self,
        total_count: u64,
        scope: String,
        item_ids: Vec<String>,
        embedding: &EmbeddingIdentity,
        item_fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<ProcessingJob> {
        let dimension = i64::try_from(embedding.dimension).map_err(|_| {
            DbError::QueryFailed("embedding dimension exceeds database integer range".into())
        })?;
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "CREATE processing_job SET job_type = 'reindex', scope = $scope, item_ids = $item_ids, \
                 target_embedding_provider = $provider, target_embedding_model = $model, target_embedding_dimension = $dimension, \
                 reindex_item_fingerprints = $fingerprints, \
                 status = 'queued', total_count = $total_count, completed_count = 0, failed_count = 0, \
                 checkpoint = NONE, last_error = NONE, created_at = time::now(), updated_at = time::now(), \
                 finished_at = NONE RETURN AFTER",
            )
            .bind(("scope", scope))
            .bind(("item_ids", item_ids))
            .bind(("provider", embedding.provider.clone()))
            .bind(("model", embedding.model.clone()))
            .bind(("dimension", dimension))
            .bind(("fingerprints", item_fingerprints))
            .bind(("total_count", count_to_i64(total_count)?))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::QueryFailed("create reindex processing_job".into()))
    }

    /// Replace the job's fingerprint snapshot only after the corresponding
    /// staged vectors committed. A crash before this write repeats work; it
    /// never treats a stale staged vector as current.
    pub async fn update_reindex_item_fingerprints(
        &self,
        id: &RecordId,
        fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        self.db
            .query("UPDATE $id SET reindex_item_fingerprints = $fingerprints, updated_at = time::now()")
            .bind(("id", id.clone()))
            .bind(("fingerprints", fingerprints))
            .await?
            .check()?;
        Ok(())
    }

    /// Update a reindex checkpoint only while this worker still owns its
    /// durable lease. A recovered worker must never advance another owner's
    /// checkpoint.
    pub async fn update_reindex_item_fingerprints_owned(
        &self,
        id: &RecordId,
        owner: &str,
        fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET reindex_item_fingerprints = $fingerprints, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("fingerprints", fingerprints))
            .await?
            .take(0)?;
        if job.is_none() {
            return Err(DbError::NotFound(
                "owned running reindex processing_job".into(),
                record_id_to_string(id),
            ));
        }
        Ok(())
    }

    /// Atomically acquire a reindex job. A live lease cannot be stolen; an
    /// expired lease represents a worker killed between durable checkpoints.
    pub async fn claim_reindex_processing_job(
        &self,
        id: &RecordId,
        owner: &str,
        lease_expires_at: DateTime<Utc>,
    ) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'running', reindex_lease_owner = $owner, \
                 reindex_lease_expires_at = $lease_expires_at, last_error = NONE, finished_at = NONE, \
                 updated_at = time::now() WHERE job_type = 'reindex' AND (status = 'queued' OR \
                 status = 'cancelled' OR status = 'failed' OR (status = 'running' AND \
                 (reindex_lease_expires_at IS NONE OR reindex_lease_expires_at < time::now()))) RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("lease_expires_at", lease_expires_at))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound(
                "claimable reindex processing_job".into(),
                record_id_to_string(id),
            )
        })
    }

    /// Renew a running reindex lease. A false result means another worker has
    /// recovered the job or the lease expired before this worker could renew.
    pub async fn renew_reindex_processing_job_lease(
        &self,
        id: &RecordId,
        owner: &str,
        lease_expires_at: DateTime<Utc>,
    ) -> Result<bool> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET reindex_lease_expires_at = $lease_expires_at, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("lease_expires_at", lease_expires_at))
            .await?
            .take(0)?;
        Ok(job.is_some())
    }

    /// Transfer still-valid staged rows to a recovered worker. The exact
    /// source snapshot condition intentionally leaves edited rows for the new
    /// worker to re-embed rather than adopting a stale vector.
    pub async fn adopt_reindex_staging(&self, item_ids: &[String], owner: &str) -> Result<()> {
        let mut notes = Vec::new();
        let mut messages = Vec::new();
        let mut conversations = Vec::new();
        for id in item_ids {
            let record = parse_record_id(id, None)?;
            match record.table.as_str() {
                "note" => notes.push(record),
                "message" => messages.push(record),
                "conversation" => conversations.push(record),
                table => {
                    return Err(DbError::QueryFailed(format!(
                        "{table} is not a supported reindex record table"
                    )))
                }
            }
        }
        self.db
            .query(
                "BEGIN TRANSACTION; \
                 UPDATE note SET reindex_staging_owner = $owner WHERE id IN $notes AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.search_content = search_content AND reindex_source_snapshot.chunk_heading_path = chunk_heading_path; \
                 UPDATE message SET reindex_staging_owner = $owner WHERE id IN $messages AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.content_blocks = content_blocks; \
                 UPDATE conversation SET reindex_staging_owner = $owner WHERE id IN $conversations AND reindex_summary_embedding IS NOT NONE AND reindex_source_snapshot.title = title AND reindex_source_snapshot.summary = summary; \
                 COMMIT TRANSACTION;",
            )
            .bind(("owner", owner.to_string()))
            .bind(("notes", notes))
            .bind(("messages", messages))
            .bind(("conversations", conversations))
            .await?
            .check()?;
        Ok(())
    }

    /// Update progress only after an item's atomic database mutation has
    /// completed.  A crash can therefore repeat at most the checkpoint item;
    /// the processing callers use idempotent upserts/reconciliation.
    pub async fn update_processing_job(
        &self,
        id: &RecordId,
        update: ProcessingJobUpdate,
    ) -> Result<ProcessingJob> {
        let status = update.status.map(ProcessingJobStatus::as_str);
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = IF $status = NONE THEN status ELSE $status END, \
                 completed_count = IF $completed_count = NONE THEN completed_count ELSE $completed_count END, \
                 failed_count = IF $failed_count = NONE THEN failed_count ELSE $failed_count END, \
                 checkpoint = IF $checkpoint_set THEN $checkpoint ELSE checkpoint END, \
                 last_error = IF $last_error_set THEN $last_error ELSE last_error END, \
                 finished_at = IF $finish THEN time::now() ELSE finished_at END, updated_at = time::now() \
                 RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("status", status))
            .bind((
                "completed_count",
                update.completed_count.map(count_to_i64).transpose()?,
            ))
            .bind(("failed_count", update.failed_count.map(count_to_i64).transpose()?))
            .bind(("checkpoint_set", update.checkpoint.is_some()))
            .bind(("checkpoint", update.checkpoint.flatten()))
            .bind(("last_error_set", update.last_error.is_some()))
            .bind(("last_error", update.last_error.flatten()))
            .bind(("finish", update.finish))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::NotFound("processing_job".into(), record_id_to_string(id)))
    }

    /// Apply a reindex worker update only if its lease is still live.
    pub async fn update_owned_reindex_processing_job(
        &self,
        id: &RecordId,
        owner: &str,
        update: ProcessingJobUpdate,
    ) -> Result<ProcessingJob> {
        let status = update.status.map(ProcessingJobStatus::as_str);
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = IF $status = NONE THEN status ELSE $status END, \
                 completed_count = IF $completed_count = NONE THEN completed_count ELSE $completed_count END, \
                 failed_count = IF $failed_count = NONE THEN failed_count ELSE $failed_count END, \
                 checkpoint = IF $checkpoint_set THEN $checkpoint ELSE checkpoint END, \
                 last_error = IF $last_error_set THEN $last_error ELSE last_error END, \
                 finished_at = IF $finish THEN time::now() ELSE finished_at END, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("status", status))
            .bind((
                "completed_count",
                update.completed_count.map(count_to_i64).transpose()?,
            ))
            .bind(("failed_count", update.failed_count.map(count_to_i64).transpose()?))
            .bind(("checkpoint_set", update.checkpoint.is_some()))
            .bind(("checkpoint", update.checkpoint.flatten()))
            .bind(("last_error_set", update.last_error.is_some()))
            .bind(("last_error", update.last_error.flatten()))
            .bind(("finish", update.finish))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound(
                "owned running reindex processing_job".into(),
                record_id_to_string(id),
            )
        })
    }

    pub async fn get_processing_job(&self, id: &str) -> Result<Option<ProcessingJob>> {
        let raw = id.strip_prefix("processing_job:").unwrap_or(id);
        Ok(self.db.select(("processing_job", raw)).await?)
    }

    pub async fn list_processing_jobs(&self, limit: usize) -> Result<Vec<ProcessingJob>> {
        let limit = i64::try_from(limit)
            .map_err(|_| DbError::QueryFailed("job limit exceeds database integer range".into()))?;
        Ok(self
            .db
            .query("SELECT * FROM processing_job ORDER BY updated_at DESC LIMIT $limit")
            .bind(("limit", limit))
            .await?
            .take(0)?)
    }

    /// A cancelled job is never silently restarted. `resume` must explicitly
    /// move it back to running after the caller has reconstructed the work.
    pub async fn cancel_processing_job(&self, id: &RecordId) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'cancelled', updated_at = time::now(), finished_at = time::now() \
                 WHERE status = 'running' OR status = 'queued' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound("runnable processing_job".into(), record_id_to_string(id))
        })
    }

    pub async fn resume_processing_job(&self, id: &RecordId) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'running', last_error = NONE, finished_at = NONE, updated_at = time::now() \
                 WHERE status = 'cancelled' OR status = 'failed' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound("resumable processing_job".into(), record_id_to_string(id))
        })
    }

    /// Snapshot the exact visible corpus selected for a reindex job. The
    /// durable job stores these logical IDs so a resume never silently picks
    /// up unrelated later records.
    pub async fn snapshot_reindex_item_ids(
        &self,
        notes: bool,
        messages: bool,
        summaries: bool,
    ) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        if notes {
            // A corpus-wide identity cutover must cover hidden/pending file
            // generations too. Otherwise a later source promotion could make
            // an old-model vector visible under freshly committed metadata.
            let note_query = if messages && summaries {
                "SELECT VALUE id FROM note ORDER BY id ASC".to_string()
            } else {
                format!("SELECT VALUE id FROM note WHERE {VISIBLE_NOTE_CONDITION} ORDER BY id ASC")
            };
            ids.extend(self.reindex_ids_for_query(&note_query).await?);
        }
        if messages {
            ids.extend(
                self.reindex_ids_for_query("SELECT VALUE id FROM message ORDER BY id ASC")
                    .await?,
            );
        }
        if summaries {
            ids.extend(
                self.reindex_ids_for_query(
                    "SELECT VALUE id FROM conversation WHERE summary IS NOT NONE AND summary != '' ORDER BY id ASC",
                )
                .await?,
            );
        }
        Ok(ids)
    }

    /// Check whether a corpus-wide reindex snapshot has been widened by newly
    /// eligible records. The cutover transaction repeats this check atomically;
    /// this preflight gives the worker an actionable restart error instead of
    /// leaving a failed job with an opaque database transaction message.
    pub async fn full_reindex_scope_widened(&self, item_ids: &[String]) -> Result<bool> {
        let expected = item_ids.iter().collect::<HashSet<_>>();
        Ok(self
            .snapshot_reindex_item_ids(true, true, true)
            .await?
            .iter()
            .any(|id| !expected.contains(id)))
    }

    async fn reindex_ids_for_query(&self, query: &str) -> Result<Vec<String>> {
        let ids: Vec<RecordId> = self.db.query(query).await?.take(0)?;
        Ok(ids.iter().map(record_id_to_string).collect())
    }

    /// Load one previously snapshotted reindex item. A record removed after
    /// snapshot is reported as absent and can be reconciled as completed by
    /// the worker without widening its durable scope.
    pub async fn get_reindex_item(&self, id: &str) -> Result<Option<ReindexItem>> {
        let record = parse_record_id(id, None)?;
        match record.table.as_str() {
            "note" => {
                #[derive(Deserialize, SurrealValue)]
                struct NoteRow {
                    id: RecordId,
                    content: String,
                    #[serde(default)]
                    search_content: Option<String>,
                    #[serde(default)]
                    chunk_heading_path: Vec<String>,
                }
                let row: Option<NoteRow> = self
                    .db
                    .query("SELECT id, content, search_content, chunk_heading_path FROM $id")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_note_reindex_text(
                        &row.content,
                        row.search_content.as_deref(),
                        &row.chunk_heading_path,
                    ),
                    source_snapshot: serde_json::json!({
                        "content": row.content,
                        "search_content": row.search_content,
                        "chunk_heading_path": row.chunk_heading_path,
                    }),
                }))
            }
            "message" => {
                #[derive(Deserialize, SurrealValue)]
                struct MessageRow {
                    id: RecordId,
                    content: String,
                    #[serde(default)]
                    content_blocks: serde_json::Value,
                }
                let row: Option<MessageRow> = self
                    .db
                    .query("SELECT id, content, content_blocks FROM $id")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_message_reindex_text(&row.content, &row.content_blocks),
                    source_snapshot: serde_json::json!({
                        "content": row.content,
                        "content_blocks": row.content_blocks,
                    }),
                }))
            }
            "conversation" => {
                #[derive(Deserialize, SurrealValue)]
                struct ConversationRow {
                    id: RecordId,
                    #[serde(default)]
                    title: Option<String>,
                    summary: String,
                }
                let row: Option<ConversationRow> = self
                    .db
                    .query("SELECT id, title, summary FROM $id WHERE summary IS NOT NONE AND summary != ''")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_conversation_reindex_text(row.title.as_deref(), &row.summary),
                    source_snapshot: serde_json::json!({
                        "title": row.title,
                        "summary": row.summary,
                    }),
                }))
            }
            table => Err(DbError::QueryFailed(format!(
                "{table} is not a supported reindex record table"
            ))),
        }
    }

    /// Persist a newly computed vector in the inactive reindex field. Search
    /// continues to read the old active vector until [`Self::commit_reindex`]
    /// validates and swaps the full selected generation.
    pub async fn stage_reindex_embedding(
        &self,
        item: &ReindexItem,
        embedding: Vec<f32>,
        owner: &str,
    ) -> Result<()> {
        let record = parse_record_id(&item.id, None)?;
        let query = match record.table.as_str() {
            "note" => "UPDATE $id SET reindex_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE content = $snapshot.content AND search_content = $snapshot.search_content AND chunk_heading_path = $snapshot.chunk_heading_path RETURN AFTER",
            "message" => "UPDATE $id SET reindex_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE content = $snapshot.content AND content_blocks = $snapshot.content_blocks RETURN AFTER",
            "conversation" => "UPDATE $id SET reindex_summary_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE title = $snapshot.title AND summary = $snapshot.summary RETURN AFTER",
            table => {
                return Err(DbError::QueryFailed(format!(
                    "{table} is not a supported reindex record table"
                )))
            }
        };
        #[derive(Deserialize, SurrealValue)]
        struct StagedRow {
            id: RecordId,
        }
        let staged: Option<StagedRow> = self
            .db
            .query(query)
            .bind(("id", record))
            .bind(("embedding", embedding))
            .bind(("text", item.text.clone()))
            .bind(("snapshot", item.source_snapshot.clone()))
            .bind(("owner", owner.to_string()))
            .await?
            .take(0)?;
        if staged.is_none() {
            return Err(DbError::QueryFailed(
                "reindex item changed or disappeared while its embedding was being staged".into(),
            ));
        }
        Ok(())
    }

    /// Atomically publish all staged vectors and the corresponding model
    /// identity. Any provider failure or cancellation before this method leaves
    /// every active vector and metadata field on the last known-good model.
    #[allow(clippy::too_many_arguments)]
    pub async fn commit_reindex(
        &self,
        job_id: &RecordId,
        owner: &str,
        item_ids: &[String],
        embedding: &EmbeddingIdentity,
        clear_entity_embeddings: bool,
        validate_full_scope_membership: bool,
        completed_count: u64,
    ) -> Result<()> {
        let mut notes = Vec::new();
        let mut messages = Vec::new();
        let mut conversations = Vec::new();
        for id in item_ids {
            let record = parse_record_id(id, None)?;
            match record.table.as_str() {
                "note" => notes.push(record),
                "message" => messages.push(record),
                "conversation" => conversations.push(record),
                table => {
                    return Err(DbError::QueryFailed(format!(
                        "{table} is not a supported reindex record table"
                    )))
                }
            }
        }
        let dimension = i64::try_from(embedding.dimension).map_err(|_| {
            DbError::QueryFailed("embedding dimension exceeds database integer range".into())
        })?;
        let completed_count = count_to_i64(completed_count)?;
        self.db
            .query(
                "BEGIN TRANSACTION; \
                 LET $job_valid = (SELECT VALUE count() FROM processing_job WHERE id = $job_id AND job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner AND reindex_lease_expires_at >= time::now() GROUP ALL)[0] = 1; \
                 LET $notes_expected = (SELECT VALUE count() FROM note WHERE id IN $notes GROUP ALL)[0]; \
                 LET $messages_expected = (SELECT VALUE count() FROM message WHERE id IN $messages GROUP ALL)[0]; \
                 LET $conversations_expected = (SELECT VALUE count() FROM conversation WHERE id IN $conversations GROUP ALL)[0]; \
                 LET $notes_widened = (SELECT VALUE count() FROM note WHERE id NOT IN $notes GROUP ALL)[0] != 0; \
                 LET $messages_widened = (SELECT VALUE count() FROM message WHERE id NOT IN $messages GROUP ALL)[0] != 0; \
                 LET $conversations_widened = (SELECT VALUE count() FROM conversation WHERE summary IS NOT NONE AND summary != '' AND id NOT IN $conversations GROUP ALL)[0] != 0; \
                 LET $notes_valid = (SELECT VALUE count() FROM note WHERE id IN $notes AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.search_content = search_content AND reindex_source_snapshot.chunk_heading_path = chunk_heading_path AND reindex_staging_owner = $owner GROUP ALL)[0] = $notes_expected; \
                 LET $messages_valid = (SELECT VALUE count() FROM message WHERE id IN $messages AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.content_blocks = content_blocks AND reindex_staging_owner = $owner GROUP ALL)[0] = $messages_expected; \
                 LET $conversations_valid = (SELECT VALUE count() FROM conversation WHERE id IN $conversations AND reindex_summary_embedding IS NOT NONE AND reindex_source_snapshot.title = title AND reindex_source_snapshot.summary = summary AND reindex_staging_owner = $owner GROUP ALL)[0] = $conversations_expected; \
                 IF !$job_valid OR !$notes_valid OR !$messages_valid OR !$conversations_valid THEN THROW 'reindex staging is stale, incomplete, or no longer owned' END; \
                 IF $validate_full_scope_membership AND ($notes_widened OR $messages_widened OR $conversations_widened) THEN THROW 'full reindex scope widened after snapshot; start a new reindex job' END; \
                 UPDATE note SET embedding = reindex_embedding, reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $notes; \
                 UPDATE message SET embedding = reindex_embedding, reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $messages; \
                 UPDATE conversation SET summary_embedding = reindex_summary_embedding, reindex_summary_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $conversations; \
                 UPDATE entity SET embedding = NONE WHERE $clear_entity_embeddings AND embedding IS NOT NONE; \
                 UPSERT graphrag_metadata SET key = 'active_embedding', \
                     application_schema_version = $schema_version, embedding_provider = $provider, \
                     embedding_model = $model, embedding_dimension = $dimension, \
                     generation = IF generation = NONE THEN 1 ELSE generation + 1 END, last_reindex_at = time::now(), \
                     last_reindex_status = 'completed', updated_at = time::now() WHERE key = 'active_embedding'; \
                 UPDATE $job_id SET status = 'completed', completed_count = $completed_count, checkpoint = NONE, last_error = NONE, finished_at = time::now(), updated_at = time::now() \
                     WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner AND reindex_lease_expires_at >= time::now(); \
                 COMMIT TRANSACTION;",
            )
            .bind(("job_id", job_id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("notes", notes))
            .bind(("messages", messages))
            .bind(("conversations", conversations))
            .bind(("schema_version", i64::from(migrations::latest_version())))
            .bind(("provider", embedding.provider.clone()))
            .bind(("model", embedding.model.clone()))
            .bind(("dimension", dimension))
            .bind(("clear_entity_embeddings", clear_entity_embeddings))
            .bind((
                "validate_full_scope_membership",
                validate_full_scope_membership,
            ))
            .bind(("completed_count", completed_count))
            .await?
            .check()?;
        Ok(())
    }

    /// Read a durable local inference result by its fully semantic cache key.
    pub async fn get_inference_cache(&self, cache_key: &str) -> Result<Option<serde_json::Value>> {
        #[derive(Deserialize, SurrealValue)]
        struct CacheValue {
            cache_value: serde_json::Value,
        }
        let row: Option<CacheValue> = self
            .db
            .query("SELECT cache_value FROM inference_cache WHERE cache_key = $cache_key LIMIT 1")
            .bind(("cache_key", cache_key.to_string()))
            .await?
            .take(0)?;
        Ok(row.map(|row| row.cache_value))
    }

    /// Store a JSON result under its semantic key. The unique index makes
    /// concurrent misses converge without changing the cached result.
    pub async fn put_inference_cache(&self, entry: InferenceCacheEntry) -> Result<()> {
        self.db
            .query(
                "UPSERT type::record('inference_cache', $cache_key) SET cache_key = $cache_key, operation = $operation, \
                 provider = $provider, model = $model, version = $version, input_hash = $input_hash, \
                 cache_value = $cache_value, updated_at = time::now(), created_at = IF created_at = NONE THEN time::now() ELSE created_at END",
            )
            .bind(("cache_key", entry.cache_key))
            .bind(("operation", entry.operation))
            .bind(("provider", entry.provider))
            .bind(("model", entry.model))
            .bind(("version", entry.version))
            .bind(("input_hash", entry.input_hash))
            .bind(("cache_value", entry.value))
            .await?
            .check()?;
        Ok(())
    }

    // ==========================================
    // NOTE OPERATIONS
    // ==========================================

    /// Create a new note
    #[instrument(skip(self, note))]
    pub async fn create_note(&self, note: Note) -> Result<Note> {
        // Source ownership is written in the same CREATE statement as the
        // note. Splitting this into a later UPDATE leaves an interruption
        // window where a staged import leaks an unowned, visible note.
        let created: Option<Note> = self
            .db
            .query(
                "CREATE note SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, source_id = $source_id, \
                    source_generation = $source_generation, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at \
                 RETURN AFTER",
            )
            .bind((
                "note_type",
                serde_json::to_value(&note.note_type)
                    .map_err(|error| DbError::QueryFailed(error.to_string()))?,
            ))
            .bind(("title", note.title.clone()))
            .bind(("content", note.content.clone()))
            .bind((
                "embedding",
                (!note.embedding.is_empty()).then_some(note.embedding.clone()),
            ))
            .bind(("source_id", note.source_id.clone()))
            .bind((
                "source_generation",
                note.source_generation.map(|generation| generation as i64),
            ))
            .bind(("chunk_key", note.chunk_key.clone()))
            .bind(("chunk_location_key", note.chunk_location_key.clone()))
            .bind(("chunk_ordinal", note.chunk_ordinal.map(|value| value as i64)))
            .bind(("chunk_heading_path", note.chunk_heading_path.clone()))
            .bind(("source_start_line", note.source_start_line.map(|value| value as i64)))
            .bind(("source_end_line", note.source_end_line.map(|value| value as i64)))
            .bind(("source_start_byte", note.source_start_byte.map(|value| value as i64)))
            .bind(("source_end_byte", note.source_end_byte.map(|value| value as i64)))
            .bind(("chunk_overlap_from", note.chunk_overlap_from.clone()))
            .bind(("chunk_overlap_chars", note.chunk_overlap_chars.map(|value| value as i64)))
            .bind(("split_fenced_code", note.split_fenced_code))
            .bind(("content_hash", note.content_hash.clone()))
            .bind(("search_content", note.search_content.clone()))
            .bind(("tags", note.tags.clone()))
            .bind(("created_at", note.created_at.to_rfc3339()))
            .bind(("updated_at", note.updated_at.to_rfc3339()))
            .await?
            .take(0)?;
        created.ok_or_else(|| DbError::QueryFailed("create_note".into()))
    }

    /// Get a note by ID
    #[instrument(skip(self))]
    pub async fn get_note(&self, id: &str) -> Result<Option<Note>> {
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note: Option<Note> = self.db.select(("note", raw_id)).await?;
        Ok(note)
    }

    /// Get a note only when its source generation is currently visible.
    #[instrument(skip(self))]
    pub async fn get_visible_note(&self, id: &str) -> Result<Option<Note>> {
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note: Option<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE id = $id AND {VISIBLE_NOTE_CONDITION} LIMIT 1"
            ))
            .bind(("id", RecordId::new("note", raw_id)))
            .await?
            .take(0)?;
        Ok(note)
    }

    /// Update a note
    #[instrument(skip(self, note))]
    pub async fn update_note(&self, id: &str, note: Note) -> Result<Note> {
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let existing = self
            .get_note(raw_id)
            .await?
            .ok_or_else(|| DbError::NotFound("note".into(), id.into()))?;
        let search_content = search_content_for_note_update(&existing, &note);
        let updated: Option<Note> = self
            .db
            .query(
                "UPDATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    source_id = IF $source_id = NONE THEN source_id ELSE $source_id END, \
                    source_generation = IF $source_generation = NONE THEN source_generation ELSE $source_generation END, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at \
                 RETURN AFTER",
            )
            .bind(("id", RecordId::new("note", raw_id)))
            .bind(("note_type", serde_json::to_value(&note.note_type).map_err(|error| DbError::QueryFailed(error.to_string()))?))
            .bind(("title", note.title.clone()))
            .bind(("content", note.content.clone()))
            .bind(("embedding", (!note.embedding.is_empty()).then_some(note.embedding.clone())))
            .bind(("tags", note.tags.clone()))
            .bind(("source_id", note.source_id.clone()))
            .bind(("source_generation", note.source_generation.map(|generation| generation as i64)))
            .bind(("chunk_key", note.chunk_key.clone()))
            .bind(("chunk_location_key", note.chunk_location_key.clone()))
            .bind(("chunk_ordinal", note.chunk_ordinal.map(|value| value as i64)))
            .bind(("chunk_heading_path", note.chunk_heading_path.clone()))
            .bind(("source_start_line", note.source_start_line.map(|value| value as i64)))
            .bind(("source_end_line", note.source_end_line.map(|value| value as i64)))
            .bind(("source_start_byte", note.source_start_byte.map(|value| value as i64)))
            .bind(("source_end_byte", note.source_end_byte.map(|value| value as i64)))
            .bind(("chunk_overlap_from", note.chunk_overlap_from.clone()))
            .bind(("chunk_overlap_chars", note.chunk_overlap_chars.map(|value| value as i64)))
            .bind(("split_fenced_code", note.split_fenced_code))
            .bind(("content_hash", note.content_hash.clone()))
            .bind(("search_content", search_content))
            .bind(("created_at", note.created_at.to_rfc3339()))
            .bind(("updated_at", note.updated_at.to_rfc3339()))
            .await?
            .take(0)?;

        updated.ok_or_else(|| DbError::NotFound("note".into(), id.into()))
    }

    /// Atomically replace a note's searchable payload and its complete entity
    /// mention set. Entity upserts are completed before the transaction; the
    /// visible note update and mention replacement then commit together, so a
    /// failed mention write cannot expose new content with old evidence (or
    /// vice versa).
    #[instrument(skip(self, note, entities))]
    pub async fn update_note_and_replace_entities(
        &self,
        id: &str,
        note: Note,
        entities: Vec<Entity>,
    ) -> Result<Note> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note_id = RecordId::new("note", raw_id);
        let existing = self
            .get_note(raw_id)
            .await?
            .ok_or_else(|| DbError::NotFound("note".into(), id.into()))?;
        if !self.note_is_writable(&note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a note update endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        let entity_ids = self.replacement_entity_ids(entities).await?;
        let search_content = search_content_for_note_update(&existing, &note);

        let mut response = self
            .db
            .query(
                "BEGIN TRANSACTION; \
                 UPDATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    source_id = IF $source_id = NONE THEN source_id ELSE $source_id END, \
                    source_generation = IF $source_generation = NONE THEN source_generation ELSE $source_generation END, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at; \
                 DELETE mentions WHERE in = $id; \
                 FOR $entity_id IN $entity_ids { CREATE mentions SET in = $id, out = $entity_id; }; \
                 COMMIT TRANSACTION;",
            )
            .bind(("id", note_id.clone()))
            .bind(("note_type", serde_json::to_value(&note.note_type).map_err(|error| DbError::QueryFailed(error.to_string()))?))
            .bind(("title", note.title.clone()))
            .bind(("content", note.content.clone()))
            .bind(("embedding", (!note.embedding.is_empty()).then_some(note.embedding.clone())))
            .bind(("tags", note.tags.clone()))
            .bind(("source_id", note.source_id.clone()))
            .bind(("source_generation", note.source_generation.map(|generation| generation as i64)))
            .bind(("chunk_key", note.chunk_key.clone()))
            .bind(("chunk_location_key", note.chunk_location_key.clone()))
            .bind(("chunk_ordinal", note.chunk_ordinal.map(|value| value as i64)))
            .bind(("chunk_heading_path", note.chunk_heading_path.clone()))
            .bind(("source_start_line", note.source_start_line.map(|value| value as i64)))
            .bind(("source_end_line", note.source_end_line.map(|value| value as i64)))
            .bind(("source_start_byte", note.source_start_byte.map(|value| value as i64)))
            .bind(("source_end_byte", note.source_end_byte.map(|value| value as i64)))
            .bind(("chunk_overlap_from", note.chunk_overlap_from.clone()))
            .bind(("chunk_overlap_chars", note.chunk_overlap_chars.map(|value| value as i64)))
            .bind(("split_fenced_code", note.split_fenced_code))
            .bind(("content_hash", note.content_hash.clone()))
            .bind(("search_content", search_content))
            .bind(("created_at", note.created_at.to_rfc3339()))
            .bind(("updated_at", note.updated_at.to_rfc3339()))
            .bind(("entity_ids", entity_ids))
            .await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            return Err(DbError::QueryFailed(format!(
                "atomic note-and-mention update failed: {}",
                errors
                    .into_iter()
                    .map(|(statement, error)| format!("statement {statement}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }

        self.get_note(raw_id)
            .await?
            .ok_or_else(|| DbError::NotFound("note".into(), id.into()))
    }

    /// Delete a note
    #[instrument(skip(self))]
    pub async fn delete_note(&self, id: &str) -> Result<()> {
        self.delete_note_with_summary(id).await.map(|_| ())
    }

    /// Return the exact cascade that a single-note deletion would perform.
    /// This is read-only and powers the CLI's non-mutating default preview.
    #[instrument(skip(self))]
    pub async fn preview_note_delete(&self, id: &str) -> Result<SourceDeleteSummary> {
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note_id = RecordId::new("note", raw_id);
        if !self.note_is_visible(&note_id).await? {
            return Err(DbError::NotFound("note".into(), id.into()));
        }
        self.delete_summary_for_notes(std::slice::from_ref(&note_id))
            .await
    }

    /// Delete one visible note and return the same exact cascade reported by
    /// [`Self::preview_note_delete`]. Proposal retirement happens before the
    /// physical dependent cleanup, so accepted-edge audits never dangle.
    #[instrument(skip(self))]
    pub async fn delete_note_with_summary(&self, id: &str) -> Result<SourceDeleteSummary> {
        // Serialize endpoint removal with proposal acceptance. Without this,
        // deletion could run after acceptance checks existence but before the
        // accepted edge write, leaving a dangling endpoint reference.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note_id = RecordId::new("note", raw_id);
        if !self.note_is_visible(&note_id).await? {
            return Err(DbError::NotFound("note".into(), id.into()));
        }
        let summary = self
            .delete_summary_for_notes(std::slice::from_ref(&note_id))
            .await?;
        self.supersede_proposals_for_removed_notes(std::slice::from_ref(&note_id))
            .await?;
        self.delete_notes_and_dependents(std::slice::from_ref(&note_id))
            .await?;
        Ok(summary)
    }

    /// List recent notes (basic fields only, for CLI)
    #[instrument(skip(self))]
    pub async fn list_notes(&self, limit: usize) -> Result<Vec<SearchResult>> {
        self.list_notes_filtered(limit, &[], None).await
    }

    /// List visible notes with deterministic, CLI-oriented tag/source filters.
    #[instrument(skip(self, tags))]
    pub async fn list_notes_filtered(
        &self,
        limit: usize,
        tags: &[String],
        source_uri: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut notes: Vec<SearchResult> = self
            .db
            .query(format!(
                "SELECT *, source_id.uri AS source_uri FROM note WHERE {VISIBLE_NOTE_CONDITION}"
            ))
            .await?
            .take(0)?;

        // Sort by creation time descending and apply limit in Rust to avoid
        // SurrealDB multi-result `take` issues and deserialization problems
        // with full `Note` records.
        notes.sort_by_key(|note| std::cmp::Reverse(note.created_at));
        notes.retain(|note| {
            source_uri.is_none_or(|source_uri| note.source_uri.as_deref() == Some(source_uri))
                && tags
                    .iter()
                    .all(|tag| note.tags.iter().any(|note_tag| note_tag == tag))
        });
        if notes.len() > limit {
            notes.truncate(limit);
        }

        Ok(notes)
    }

    /// Get notes without embeddings (for processing)
    #[instrument(skip(self))]
    pub async fn get_notes_without_embeddings(&self) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE ({VISIBLE_NOTE_CONDITION}) AND (embedding IS NONE OR array::len(embedding) = 0)"
            ))
            .await?
            .take(0)?;

        Ok(notes)
    }

    /// Read one stable page while building a durable pending-embedding
    /// snapshot. Callers persist only the page's record IDs, keeping initial
    /// job selection bounded before any inference work begins.
    pub async fn get_notes_without_embeddings_page(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Note>> {
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("embedding page limit exceeds database integer range".into())
        })?;
        let offset = i64::try_from(offset).map_err(|_| {
            DbError::QueryFailed("embedding page offset exceeds database integer range".into())
        })?;
        Ok(self
            .db
            .query(format!(
                "SELECT * FROM note WHERE ({VISIBLE_NOTE_CONDITION}) AND (embedding IS NONE OR array::len(embedding) = 0) ORDER BY created_at ASC, id ASC LIMIT $limit START $offset"
            ))
            .bind(("limit", limit))
            .bind(("offset", offset))
            .await?
            .take(0)?)
    }

    /// Fetch one bounded work window. Repeating this query is safe because a
    /// successful item no longer matches it, avoiding an unbounded in-memory
    /// import queue and making interruption reconciliation natural.
    pub async fn get_notes_without_embeddings_limit(&self, limit: usize) -> Result<Vec<Note>> {
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("embedding page limit exceeds database integer range".into())
        })?;
        Ok(self
            .db
            .query(format!(
                "SELECT * FROM note WHERE ({VISIBLE_NOTE_CONDITION}) AND (embedding IS NONE OR array::len(embedding) = 0) ORDER BY id LIMIT $limit"
            ))
            .bind(("limit", limit))
            .await?
            .take(0)?)
    }

    pub async fn count_notes_without_embeddings(&self) -> Result<u64> {
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: i64,
        }
        let row: Option<CountRow> = self
            .db
            .query(format!(
                "SELECT count() AS count FROM note WHERE ({VISIBLE_NOTE_CONDITION}) AND (embedding IS NONE OR array::len(embedding) = 0) GROUP ALL"
            ))
            .await?
            .take(0)?;
        let count = row.map(|row| row.count).unwrap_or(0);
        u64::try_from(count)
            .map_err(|_| DbError::QueryFailed("negative pending embedding count".into()))
    }

    /// Get notes without entity links (for extraction)
    #[instrument(skip(self))]
    pub async fn get_notes_without_entities(&self, limit: usize) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE ({VISIBLE_NOTE_CONDITION}) AND id NOT IN (SELECT in FROM mentions) LIMIT $limit"
            ))
            .bind(("limit", limit))
            .await?
            .take(0)?;

        Ok(notes)
    }

    /// Get notes in a stable order (for full extraction passes)
    #[instrument(skip(self))]
    pub async fn get_notes_page(&self, limit: usize, offset: usize) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE {VISIBLE_NOTE_CONDITION} ORDER BY created_at ASC LIMIT $limit START $offset"
            ))
            .bind(("limit", limit))
            .bind(("offset", offset))
            .await?
            .take(0)?;

        Ok(notes)
    }

    /// Return the current persisted Markdown chunks for one source. The caller
    /// uses this before staging a new source generation to retain IDs and
    /// embeddings for chunks whose deterministic key/content are unchanged.
    ///
    /// Pre-v008 Markdown imports did not persist `chunk_key`, but they did set
    /// `source_generation`. Include those successful legacy notes so their
    /// first v008-era refresh can reconcile safe successors instead of
    /// deleting their graph dependents as an unrelated generation.
    #[instrument(skip(self, source_id))]
    pub async fn get_source_chunks(&self, source_id: &RecordId) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(
                "SELECT * FROM note WHERE source_id = $source_id \
                 AND source_generation = source_id.successful_generation \
                 ORDER BY chunk_ordinal ASC, created_at ASC, id ASC",
            )
            .bind(("source_id", source_id.clone()))
            .await?
            .take(0)?;
        Ok(notes)
    }

    /// Update note embedding
    #[instrument(skip(self, embedding))]
    pub async fn update_note_embedding(
        &self,
        id: &surrealdb::types::RecordId,
        embedding: Vec<f32>,
    ) -> Result<()> {
        self.db
            .query(
                "UPDATE note SET embedding = $embedding, updated_at = time::now() WHERE id = $id",
            )
            .bind(("id", id.clone()))
            .bind(("embedding", embedding))
            .await?;

        Ok(())
    }

    // ==========================================
    // SEARCH OPERATIONS
    // ==========================================

    /// Hybrid search combining vector similarity and full-text
    #[instrument(skip(self, embedding))]
    pub async fn hybrid_search(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_search_notes(query_text, embedding, limit, None, None)
            .await
    }

    /// Hybrid search for notes with optional temporal/source filters.
    #[instrument(skip(self, embedding))]
    pub async fn hybrid_search_notes(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_search_notes_with_weights(
            query_text, embedding, limit, since, source_uri, 0.65, 0.35,
        )
        .await
    }

    /// Hybrid note search using explicitly configured vector and full-text
    /// weights. The caller is responsible for validating that they sum to one.
    #[instrument(skip(self, embedding))]
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search_notes_with_weights(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        vector_weight: f32,
        fulltext_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        let fusion = FusionConfig {
            vector_weight,
            fulltext_weight,
            ..FusionConfig::default()
        };
        self.hybrid_search_notes_with_fusion(
            query_text, embedding, limit, since, source_uri, &fusion,
        )
        .await
    }

    /// Hybrid note search with one configurable, deterministic fusion policy.
    #[instrument(skip(self, embedding, fusion))]
    pub async fn hybrid_search_notes_with_fusion(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>> {
        let candidate_limit = fusion.candidate_limit(limit);

        let vec_results = self
            .vector_search_notes(
                embedding.clone(),
                candidate_limit,
                since,
                source_uri.clone(),
            )
            .await?;

        let fts_results = self
            .fulltext_search_notes(query_text, candidate_limit, since, source_uri)
            .await?;

        let mut results = fusion::fuse(vec_results, fts_results, fusion, |existing, incoming| {
            if existing.title.is_none() {
                existing.title = incoming.title;
            }
            if existing.content.is_empty() {
                existing.content = incoming.content;
            }
            if existing.tags.is_empty() {
                existing.tags = incoming.tags;
            }
            if incoming.fts_score.is_some() {
                existing.fts_score = incoming.fts_score;
            }
        });
        if results.len() > limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    #[instrument(skip(self, embedding))]
    pub async fn vector_search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.vector_search_notes(embedding, limit, None, None).await
    }

    #[instrument(skip(self, embedding))]
    pub async fn vector_search_notes(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        // SurrealQL requires a literal KNN candidate count. `limit` is a
        // usize calculated by FusionConfig, so interpolating it is safe and
        // keeps KNN's pool aligned with the query LIMIT.
        let query = format!(
            r#"
                SELECT 
                    id,
                    title,
                    content,
                    note_type,
                    tags,
                    created_at,
                    source_id.uri AS source_uri,
                    vector::distance::knn() AS vec_distance
                FROM note
                WHERE embedding <|{limit},COSINE|> $embedding
                  AND ($since = NONE OR created_at >= <datetime>$since)
                  AND ($source_uri = NONE OR source_id.uri = $source_uri)
                  AND (
                    source_id IS NONE
                    OR source_generation IS NONE
                    OR source_generation = source_id.successful_generation
                  )
                ORDER BY vec_distance ASC, id ASC
                LIMIT $limit
            "#
        );
        let results: Vec<SearchResult> = self
            .db
            .query(query)
            .bind(("embedding", embedding))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;

        Ok(results)
    }

    /// Full-text search only
    #[instrument(skip(self))]
    pub async fn fulltext_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.fulltext_search_notes(query, limit, None, None).await
    }

    #[instrument(skip(self))]
    pub async fn fulltext_search_notes(
        &self,
        query: &str,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        let results: Vec<SearchResult> = self
            .db
            .query(
                r#"
                SELECT 
                    id,
                    title,
                    content,
                    note_type,
                    tags,
                    created_at,
                    source_id.uri AS source_uri,
                    (search::score(0) * 0.7 + search::score(1) * 0.2 + search::score(2) * 0.1) AS fts_score
                FROM note
                WHERE (search_content @0@ $query OR content @1@ $query OR title @2@ $query)
                  AND ($since = NONE OR created_at >= <datetime>$since)
                  AND ($source_uri = NONE OR source_id.uri = $source_uri)
                  AND (
                    source_id IS NONE
                    OR source_generation IS NONE
                    OR source_generation = source_id.successful_generation
                  )
                ORDER BY fts_score DESC, id ASC
                LIMIT $limit
            "#,
            )
            .bind(("query", query.to_string()))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;

        Ok(results)
    }

    /// Hybrid search across persisted chat messages.
    #[instrument(skip(self, embedding))]
    pub async fn hybrid_search_messages(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<MessageSearchResult>> {
        self.hybrid_search_messages_with_weights(
            query_text, embedding, limit, since, source_uri, 0.65, 0.35,
        )
        .await
    }

    /// Hybrid message search using explicitly configured ranking weights.
    #[instrument(skip(self, embedding))]
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search_messages_with_weights(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        vector_weight: f32,
        fulltext_weight: f32,
    ) -> Result<Vec<MessageSearchResult>> {
        let fusion = FusionConfig {
            vector_weight,
            fulltext_weight,
            ..FusionConfig::default()
        };
        self.hybrid_search_messages_with_fusion(
            query_text, embedding, limit, since, source_uri, &fusion,
        )
        .await
    }

    /// Hybrid message search with one configurable, deterministic fusion policy.
    #[instrument(skip(self, embedding, fusion))]
    pub async fn hybrid_search_messages_with_fusion(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        fusion: &FusionConfig,
    ) -> Result<Vec<MessageSearchResult>> {
        let candidate_limit = fusion.candidate_limit(limit);

        let vec_results = self
            .vector_search_messages(
                embedding.clone(),
                candidate_limit,
                since,
                source_uri.clone(),
            )
            .await?;
        let fts_results = self
            .fulltext_search_messages(query_text, candidate_limit, since, source_uri)
            .await?;

        let mut results = fusion::fuse(vec_results, fts_results, fusion, |existing, incoming| {
            if incoming.fts_score.is_some() {
                existing.fts_score = incoming.fts_score;
            }
        });
        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    #[instrument(skip(self, embedding))]
    pub async fn vector_search_messages(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<MessageSearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        let query = format!(
            r#"
                SELECT
                    id,
                    conversation_id,
                    conversation_uuid,
                    message_index,
                    role,
                    content,
                    created_at,
                    conversation_id.source_uri AS source_uri,
                    vector::distance::knn() AS vec_distance
                FROM message
                WHERE embedding <|{limit},COSINE|> $embedding
                  AND ($since = NONE OR (created_at != NONE AND created_at >= <datetime>$since))
                  AND ($source_uri = NONE OR conversation_id.source_uri = $source_uri)
                ORDER BY vec_distance ASC, id ASC
                LIMIT $limit
            "#
        );
        let results: Vec<MessageSearchResult> = self
            .db
            .query(query)
            .bind(("embedding", embedding))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;

        Ok(results)
    }

    #[instrument(skip(self))]
    pub async fn fulltext_search_messages(
        &self,
        query: &str,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<MessageSearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        let results: Vec<MessageSearchResult> = self
            .db
            .query(
                r#"
                SELECT
                    id,
                    conversation_id,
                    conversation_uuid,
                    message_index,
                    role,
                    content,
                    created_at,
                    conversation_id.source_uri AS source_uri,
                    search::score(0) AS fts_score
                FROM message
                WHERE content @0@ $query
                  AND ($since = NONE OR (created_at != NONE AND created_at >= <datetime>$since))
                  AND ($source_uri = NONE OR conversation_id.source_uri = $source_uri)
                ORDER BY fts_score DESC, id ASC
                LIMIT $limit
            "#,
            )
            .bind(("query", query.to_string()))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;
        Ok(results)
    }

    /// Hybrid search across conversation summaries.
    #[instrument(skip(self, embedding))]
    pub async fn hybrid_search_conversation_summaries(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<ConversationSearchResult>> {
        self.hybrid_search_conversation_summaries_with_weights(
            query_text, embedding, limit, since, source_uri, 0.65, 0.35,
        )
        .await
    }

    /// Hybrid conversation-summary search using explicitly configured ranking
    /// weights.
    #[instrument(skip(self, embedding))]
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search_conversation_summaries_with_weights(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        vector_weight: f32,
        fulltext_weight: f32,
    ) -> Result<Vec<ConversationSearchResult>> {
        let fusion = FusionConfig {
            vector_weight,
            fulltext_weight,
            ..FusionConfig::default()
        };
        self.hybrid_search_conversation_summaries_with_fusion(
            query_text, embedding, limit, since, source_uri, &fusion,
        )
        .await
    }

    /// Hybrid conversation-summary search with one configurable, deterministic
    /// fusion policy.
    #[instrument(skip(self, embedding, fusion))]
    pub async fn hybrid_search_conversation_summaries_with_fusion(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        fusion: &FusionConfig,
    ) -> Result<Vec<ConversationSearchResult>> {
        let candidate_limit = fusion.candidate_limit(limit);

        let vec_results = self
            .vector_search_conversation_summaries(
                embedding.clone(),
                candidate_limit,
                since,
                source_uri.clone(),
            )
            .await?;
        let fts_results = self
            .fulltext_search_conversation_summaries(query_text, candidate_limit, since, source_uri)
            .await?;

        let mut results = fusion::fuse(vec_results, fts_results, fusion, |existing, incoming| {
            if incoming.fts_score.is_some() {
                existing.fts_score = incoming.fts_score;
            }
        });
        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    #[instrument(skip(self, embedding))]
    pub async fn vector_search_conversation_summaries(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<ConversationSearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        let query = format!(
            r#"
                SELECT
                    id,
                    uuid,
                    title,
                    summary,
                    source_uri,
                    updated_at,
                    vector::distance::knn() AS vec_distance
                FROM conversation
                WHERE summary_embedding <|{limit},COSINE|> $embedding
                  AND ($since = NONE OR updated_at >= <datetime>$since)
                  AND ($source_uri = NONE OR source_uri = $source_uri)
                ORDER BY vec_distance ASC, id ASC
                LIMIT $limit
            "#
        );
        let results: Vec<ConversationSearchResult> = self
            .db
            .query(query)
            .bind(("embedding", embedding))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;
        Ok(results)
    }

    #[instrument(skip(self))]
    pub async fn fulltext_search_conversation_summaries(
        &self,
        query: &str,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<ConversationSearchResult>> {
        let since = since.map(|ts| ts.to_rfc3339());
        let results: Vec<ConversationSearchResult> = self
            .db
            .query(
                r#"
                SELECT
                    id,
                    uuid,
                    title,
                    summary,
                    source_uri,
                    updated_at,
                    (search::score(0) * 0.7 + search::score(1) * 0.3) AS fts_score
                FROM conversation
                WHERE (summary @0@ $query OR title @1@ $query)
                  AND ($since = NONE OR updated_at >= <datetime>$since)
                  AND ($source_uri = NONE OR source_uri = $source_uri)
                ORDER BY fts_score DESC, id ASC
                LIMIT $limit
            "#,
            )
            .bind(("query", query.to_string()))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)?;
        Ok(results)
    }

    // ==========================================
    // GRAPH OPERATIONS
    // ==========================================

    /// Create a relationship between notes
    #[instrument(skip(self))]
    pub async fn create_edge(
        &self,
        from_id: &surrealdb::types::RecordId,
        to_id: &surrealdb::types::RecordId,
        edge_type: EdgeType,
        confidence: Option<f32>,
    ) -> Result<()> {
        // A source reconciliation snapshots old-generation dependents before
        // atomically promoting and retiring that generation. Serialize manual
        // graph writes with that transition so a write cannot be accepted in
        // the snapshot/cleanup window and then silently removed by cleanup.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        validate_note_edge(from_id, to_id, &edge_type)?;
        if !self.note_is_writable(from_id).await? || !self.note_is_writable(to_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a graph edge endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        self.create_audited_edge(
            from_id,
            to_id,
            edge_type,
            confidence,
            None,
            "manual_api",
            None,
            true,
        )
        .await?;
        Ok(())
    }

    /// Persist a similarity-derived Gardener proposal. Similarity may only
    /// produce `related_to`, never a logical support/contradiction assertion.
    #[instrument(skip(self))]
    pub async fn upsert_gardener_proposal(
        &self,
        from_id: &RecordId,
        to_id: &RecordId,
        confidence: f32,
        reason: String,
        generator_version: Option<String>,
        model: Option<String>,
    ) -> Result<ProposedEdge> {
        self.upsert_edge_proposal(EdgeProposalDraft {
            from_id: from_id.clone(),
            to_id: to_id.clone(),
            edge_type: EdgeType::RelatedTo,
            confidence,
            reason,
            generator: "gardener-similarity".into(),
            generator_version,
            model,
        })
        .await
    }

    /// Create or update a proposal identified by its stable canonical key.
    /// Terminal proposals are returned unchanged: a repeated scan must not
    /// silently resurrect a user decision or create an equivalent duplicate.
    #[instrument(skip(self, draft))]
    pub async fn upsert_edge_proposal(&self, mut draft: EdgeProposalDraft) -> Result<ProposedEdge> {
        validate_note_edge(&draft.from_id, &draft.to_id, &draft.edge_type)?;
        canonicalize_note_edge(&mut draft.from_id, &mut draft.to_id, &draft.edge_type);
        draft.confidence = draft.confidence.clamp(0.0, 1.0);
        let dedupe_key = edge_dedupe_key(&draft.from_id, &draft.to_id, &draft.edge_type);

        if let Some(existing) = self.find_proposal_by_dedupe_key(&dedupe_key).await? {
            if existing.status == ProposedEdgeStatus::Pending {
                #[derive(Deserialize, SurrealValue)]
                struct UpdatedRow {
                    id: RecordId,
                }
                let id = existing.id.clone().expect("stored proposal has id");
                let updated: Option<UpdatedRow> = self
                    .db
                    .query(
                        "UPDATE $id SET confidence = $confidence, reason = $reason, generator = $generator, generator_version = $generator_version, model = $model, updated_at = time::now() WHERE status = 'pending' RETURN AFTER",
                    )
                    .bind(("id", id.clone()))
                    .bind(("confidence", draft.confidence))
                    .bind(("reason", draft.reason))
                    .bind(("generator", draft.generator))
                    .bind(("generator_version", draft.generator_version))
                    .bind(("model", draft.model))
                    .await?
                    .take(0)?;
                return self
                    .get_edge_proposal(&updated.map(|row| row.id).unwrap_or(id))
                    .await?
                    .ok_or_else(|| DbError::QueryFailed("updated proposal disappeared".into()));
            }
            return Ok(existing);
        }

        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }
        let insert = self
            .db
            .query(
                "INSERT INTO proposed_edge (dedupe_key, in, out, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at) VALUES ($dedupe_key, $from, $to, $edge_type, $confidence, $reason, $generator, $generator_version, $model, 'pending', time::now(), time::now()) RETURN id",
            )
            .bind(("dedupe_key", dedupe_key.clone()))
            .bind(("from", draft.from_id))
            .bind(("to", draft.to_id))
            .bind(("edge_type", draft.edge_type.to_string()))
            .bind(("confidence", draft.confidence))
            .bind(("reason", draft.reason))
            .bind(("generator", draft.generator))
            .bind(("generator_version", draft.generator_version))
            .bind(("model", draft.model))
            .await;
        let created_result: Result<Vec<IdRow>> = match insert {
            Ok(mut response) => response.take(0).map_err(Into::into),
            Err(error) => Err(error.into()),
        };
        let created = match created_result {
            Ok(created) => created,
            Err(error) => {
                // A concurrent scan can pass the lookup above at the same
                // time. The unique dedupe index elects one winner; reload it
                // instead of surfacing a spurious duplicate-key failure.
                if let Some(existing) = self.find_proposal_by_dedupe_key(&dedupe_key).await? {
                    return Ok(existing);
                }
                return Err(error);
            }
        };
        let id = created
            .into_iter()
            .next()
            .ok_or_else(|| DbError::QueryFailed("create proposed_edge".into()))?
            .id;
        self.get_edge_proposal(&id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("created proposal disappeared".into()))
    }

    /// Fetch a proposal by record id.
    #[instrument(skip(self))]
    pub async fn get_edge_proposal(&self, id: &RecordId) -> Result<Option<ProposedEdge>> {
        let proposals: Vec<ProposedEdgeRow> = self
            .db
            .query(proposal_select_sql("WHERE id = $id"))
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        proposals
            .into_iter()
            .next()
            .map(ProposedEdgeRow::into_domain)
            .transpose()
    }

    /// List proposals, optionally filtering by lifecycle status.
    #[instrument(skip(self))]
    pub async fn list_edge_proposals(
        &self,
        status: Option<ProposedEdgeStatus>,
        limit: usize,
    ) -> Result<Vec<ProposedEdge>> {
        let where_clause = if status.is_some() {
            "WHERE status = $status"
        } else {
            ""
        };
        let mut query = self
            .db
            .query(format!(
                "{} ORDER BY updated_at DESC LIMIT $limit",
                proposal_select_sql(where_clause)
            ))
            .bind(("limit", limit.max(1)));
        if let Some(status) = status {
            query = query.bind(("status", status.to_string()));
        }
        let rows: Vec<ProposedEdgeRow> = query.await?.take(0)?;
        rows.into_iter().map(ProposedEdgeRow::into_domain).collect()
    }

    /// Accept a pending proposal, creating one auditable accepted edge. Calling
    /// it again after acceptance returns the same accepted proposal unchanged.
    #[instrument(skip(self))]
    pub async fn accept_edge_proposal(
        &self,
        id: &RecordId,
        reviewer: Option<String>,
        action_reason: Option<String>,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        // All clones of one repository serialize completion. The database
        // state claim below remains the cross-process guard, while this lock
        // prevents an in-process loser from releasing a shared claim.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let proposal = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        match proposal.status {
            ProposedEdgeStatus::Accepted => {
                return self
                    .recover_or_return_accepted_proposal(id, proposal, is_manual)
                    .await;
            }
            ProposedEdgeStatus::Accepting => {
                return self.resume_acceptance(id, proposal, is_manual).await;
            }
            ProposedEdgeStatus::Pending => {}
            _ => {
                return Err(DbError::QueryFailed(format!(
                    "proposal {} is {}, not pending",
                    record_id_to_string(id),
                    proposal.status
                )));
            }
        }
        // Claim the pending row before creating the accepted edge. `accepting`
        // is recoverable: retries resume its idempotent edge creation rather
        // than exposing a completed `accepted` proposal without an edge id.
        if !self
            .claim_pending_proposal(
                id,
                ProposedEdgeStatus::Accepting,
                reviewer,
                action_reason,
                Some(is_manual),
            )
            .await?
        {
            return self.acceptance_claim_lost(id, is_manual).await;
        }
        let claimed = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("acceptance claim disappeared".into()))?;
        self.resume_acceptance(id, claimed, is_manual).await
    }

    /// Complete (or retry) an `accepting` claim. The edge upsert is idempotent
    /// by dedupe key, so an interruption after edge creation can be recovered
    /// safely by repeating this method.
    async fn resume_acceptance(
        &self,
        id: &RecordId,
        proposal: ProposedEdge,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        if !self.note_is_visible(&proposal.from_id).await?
            || !self.note_is_visible(&proposal.to_id).await?
        {
            self.mark_claimed_proposal_stale(id, "proposal endpoint is no longer visible")
                .await?;
            return Err(DbError::QueryFailed(format!(
                "proposal {} is stale: an endpoint is no longer visible",
                record_id_to_string(id)
            )));
        }
        let (edge_id, edge_proposal_id) = match self
            .create_audited_edge(
                &proposal.from_id,
                &proposal.to_id,
                proposal.edge_type.clone(),
                Some(proposal.confidence),
                Some(&proposal.reason),
                &proposal.generator,
                Some(id),
                proposal.acceptance_is_manual.unwrap_or(is_manual),
            )
            .await
        {
            Ok(edge) => edge,
            // Keep the durable `accepting` claim intact. A transient failure
            // is recoverable by a later retry/batch run; releasing it here
            // could clear another caller's in-flight completion.
            Err(error) => return Err(error),
        };
        if edge_proposal_id.as_ref() != Some(id) {
            self.mark_claimed_proposal_materialized(id).await?;
            return Err(DbError::QueryFailed(format!(
                "proposal {} is superseded because an independent equivalent edge already exists",
                record_id_to_string(id)
            )));
        }
        if !self.finalize_acceptance_claim(id, edge_id).await? {
            let current = self.get_edge_proposal(id).await?.ok_or_else(|| {
                DbError::QueryFailed("acceptance finalization disappeared".into())
            })?;
            if current.status == ProposedEdgeStatus::Accepted && current.resulting_edge_id.is_some()
            {
                return Ok(current);
            }
            return Err(DbError::QueryFailed(format!(
                "proposal {} acceptance remains recoverable; retry the operation",
                record_id_to_string(id)
            )));
        }
        self.get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("accepted proposal disappeared".into()))
    }

    /// Reject a pending proposal. Repeating the same rejection is a no-op.
    #[instrument(skip(self))]
    pub async fn reject_edge_proposal(
        &self,
        id: &RecordId,
        reviewer: Option<String>,
        action_reason: Option<String>,
    ) -> Result<ProposedEdge> {
        let proposal = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        if proposal.status == ProposedEdgeStatus::Rejected {
            return Ok(proposal);
        }
        if proposal.status != ProposedEdgeStatus::Pending {
            return Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                proposal.status
            )));
        }
        if !self
            .claim_pending_proposal(
                id,
                ProposedEdgeStatus::Rejected,
                reviewer,
                action_reason,
                None,
            )
            .await?
        {
            let current = self.get_edge_proposal(id).await?.ok_or_else(|| {
                DbError::NotFound("proposed_edge".into(), record_id_to_string(id))
            })?;
            if current.status == ProposedEdgeStatus::Rejected {
                return Ok(current);
            }
            return Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                current.status
            )));
        }
        self.get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("rejected proposal disappeared".into()))
    }

    /// Atomically transition a pending proposal to a claimed state. Rejection
    /// is terminal immediately; acceptance is finalized only after its edge
    /// has been created and recorded.
    async fn claim_pending_proposal(
        &self,
        id: &RecordId,
        status: ProposedEdgeStatus,
        reviewer: Option<String>,
        action_reason: Option<String>,
        acceptance_is_manual: Option<bool>,
    ) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct ClaimRow {
            id: RecordId,
        }
        let claimed: Option<ClaimRow> = self
            .db
            .query(
                "UPDATE $id SET status = $status, reviewed_at = time::now(), reviewer = $reviewer, action_reason = $action_reason, acceptance_is_manual = $acceptance_is_manual, updated_at = time::now() WHERE status = 'pending' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("status", status.to_string()))
            .bind(("reviewer", reviewer))
            .bind(("action_reason", action_reason))
            .bind(("acceptance_is_manual", acceptance_is_manual))
            .await?
            .take(0)?;
        Ok(claimed.is_some())
    }

    async fn acceptance_claim_lost(&self, id: &RecordId, is_manual: bool) -> Result<ProposedEdge> {
        let current = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        match current.status {
            ProposedEdgeStatus::Accepted => {
                self.recover_or_return_accepted_proposal(id, current, is_manual)
                    .await
            }
            ProposedEdgeStatus::Accepting => self.resume_acceptance(id, current, is_manual).await,
            _ => Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                current.status
            ))),
        }
    }

    /// Repair legacy/incomplete `accepted` rows that have no resulting edge
    /// reference. New code never leaves this state, but retries can safely
    /// convert it into an `accepting` claim and recreate or rediscover the
    /// deduplicated edge.
    async fn recover_or_return_accepted_proposal(
        &self,
        id: &RecordId,
        proposal: ProposedEdge,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        if proposal.resulting_edge_id.is_some() {
            return Ok(proposal);
        }
        #[derive(Deserialize, SurrealValue)]
        struct RecoveryRow {
            id: RecordId,
        }
        let recovered: Option<RecoveryRow> = self
            .db
            .query(
                "UPDATE $id SET status = 'accepting', updated_at = time::now() WHERE status = 'accepted' AND resulting_edge_id IS NONE RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        if recovered.is_some() {
            let claimed = self
                .get_edge_proposal(id)
                .await?
                .ok_or_else(|| DbError::QueryFailed("acceptance recovery disappeared".into()))?;
            return self.resume_acceptance(id, claimed, is_manual).await;
        }
        let current = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        if current.status == ProposedEdgeStatus::Accepted && current.resulting_edge_id.is_some() {
            return Ok(current);
        }
        if current.status == ProposedEdgeStatus::Accepting {
            return self.resume_acceptance(id, current, is_manual).await;
        }
        Err(DbError::QueryFailed(format!(
            "proposal {} changed while acceptance recovery was being claimed",
            record_id_to_string(id)
        )))
    }

    async fn mark_claimed_proposal_stale(&self, id: &RecordId, reason: &str) -> Result<()> {
        self.db
            .query(
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = $reason, resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting'",
            )
            .bind(("id", id.clone()))
            .bind(("reason", reason.to_string()))
            .await?
            .check()?;
        Ok(())
    }

    async fn mark_claimed_proposal_materialized(&self, id: &RecordId) -> Result<()> {
        self.db
            .query(
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'equivalent edge already materialized independently', resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting'",
            )
            .bind(("id", id.clone()))
            .await?
            .check()?;
        Ok(())
    }

    async fn finalize_acceptance_claim(&self, id: &RecordId, edge_id: RecordId) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct FinalizedRow {
            id: RecordId,
        }
        let finalized: Option<FinalizedRow> = self
            .db
            .query(
                "UPDATE $id SET status = 'accepted', resulting_edge_id = $edge_id, updated_at = time::now() WHERE status = 'accepting' AND resulting_edge_id IS NONE RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("edge_id", edge_id))
            .await?
            .take(0)?;
        Ok(finalized.is_some())
    }

    /// Accept all pending similarity proposals at or above a configured threshold.
    /// This is intentionally restricted to canonical `related_to` proposals.
    #[instrument(skip(self))]
    pub async fn accept_gardener_proposals_above(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
    ) -> Result<usize> {
        self.accept_gardener_proposals_above_with_audit(
            min_confidence,
            reviewer,
            "configured gardener auto-apply policy".into(),
            false,
        )
        .await
    }

    /// Accept every matching proposal with the supplied, auditable reviewer
    /// decision. Interactive/manual workflows must set `is_manual` to true;
    /// scheduled policy application uses the automatic default above.
    #[instrument(skip(self))]
    pub async fn accept_gardener_proposals_above_with_audit(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
        action_reason: String,
        is_manual: bool,
    ) -> Result<usize> {
        self.accept_gardener_proposals_above_in_pages(
            min_confidence,
            reviewer,
            action_reason,
            is_manual,
            250,
        )
        .await
    }

    /// Accept every matching pending proposal, using a stable record-id cursor
    /// so a large batch cannot silently stop at an arbitrary first page.
    async fn accept_gardener_proposals_above_in_pages(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
        action_reason: String,
        is_manual: bool,
        page_size: usize,
    ) -> Result<usize> {
        let page_size = page_size.max(1);
        let mut accepted = 0;
        let mut after_id = None;

        loop {
            let proposals = self
                .list_pending_gardener_proposals_page(min_confidence, after_id.clone(), page_size)
                .await?;
            if proposals.is_empty() {
                break;
            }
            let last_id = proposals
                .last()
                .and_then(|proposal| proposal.id.clone())
                .expect("stored proposal has id");

            for proposal in proposals {
                let id = proposal.id.expect("stored proposal has id");
                self.accept_edge_proposal(
                    &id,
                    reviewer.clone(),
                    Some(action_reason.clone()),
                    is_manual,
                )
                .await?;
                accepted += 1;
            }

            after_id = Some(last_id);
        }
        Ok(accepted)
    }

    async fn list_pending_gardener_proposals_page(
        &self,
        min_confidence: f32,
        after_id: Option<RecordId>,
        limit: usize,
    ) -> Result<Vec<ProposedEdge>> {
        let rows: Vec<ProposedEdgeRow> = self
            .db
            .query(format!(
                "{} WHERE (status = 'pending' OR status = 'accepting' OR (status = 'accepted' AND resulting_edge_id IS NONE)) AND edge_type = 'related_to' AND generator = 'gardener-similarity' AND confidence >= $min_confidence AND ($after_id = NONE OR id > $after_id) ORDER BY id ASC LIMIT $limit",
                proposal_select_sql("")
            ))
            .bind(("min_confidence", min_confidence.clamp(0.0, 1.0)))
            .bind(("after_id", after_id))
            .bind(("limit", limit.max(1)))
            .await?
            .take(0)?;
        rows.into_iter().map(ProposedEdgeRow::into_domain).collect()
    }

    /// Delete an accepted edge and mark its source proposal superseded. This is
    /// idempotent for a proposal that was already undone.
    #[instrument(skip(self))]
    pub async fn undo_edge(
        &self,
        edge_id: &RecordId,
        action_reason: Option<String>,
    ) -> Result<bool> {
        // Keep physical edge deletion and proposal audit retirement in the
        // same in-process lifecycle critical section as acceptance completion.
        // Otherwise an acceptance could finalize after this method deletes its
        // edge but before it supersedes the proposal, restoring a dangling
        // accepted state.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let table = edge_id.table.as_str();
        if !matches!(
            table,
            "supports" | "contradicts" | "derived_from" | "related_to"
        ) {
            return Err(DbError::QueryFailed(format!(
                "{} is not a note-edge record id",
                record_id_to_string(edge_id)
            )));
        }
        let reason = action_reason.unwrap_or_else(|| "accepted edge undone".into());
        let edge_existed = self.note_edge_exists(edge_id).await?;
        if edge_existed {
            self.db
                .query("DELETE $id")
                .bind(("id", edge_id.clone()))
                .await?
                .check()?;
        }
        // If a prior attempt deleted the edge but failed before this update,
        // a retry sees the absent edge and still repairs the proposal audit.
        let proposal_updated = self
            .supersede_proposal_for_undone_edge(edge_id, &reason)
            .await?;
        Ok(edge_existed || proposal_updated)
    }

    async fn supersede_proposal_for_undone_edge(
        &self,
        edge_id: &RecordId,
        reason: &str,
    ) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct UpdatedRow {
            id: RecordId,
        }
        let updated: Option<UpdatedRow> = self
            .db
            .query(
                "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = $reason, resulting_edge_id = NONE, updated_at = time::now() WHERE resulting_edge_id = $id RETURN AFTER",
            )
            .bind(("id", edge_id.clone()))
            .bind(("reason", reason.to_string()))
            .await?
            .take(0)?;
        Ok(updated.is_some())
    }

    /// Return whether a supported note-edge record currently exists. This is
    /// intentionally read-only so CLI dry-runs can report their real outcome
    /// without relying on the mutating undo path.
    #[instrument(skip(self))]
    pub async fn note_edge_exists(&self, edge_id: &RecordId) -> Result<bool> {
        let table = edge_id.table.as_str();
        if !matches!(
            table,
            "supports" | "contradicts" | "derived_from" | "related_to"
        ) {
            return Err(DbError::QueryFailed(format!(
                "{} is not a note-edge record id",
                record_id_to_string(edge_id)
            )));
        }
        #[derive(Deserialize, SurrealValue)]
        struct ExistingEdge {
            id: RecordId,
        }
        let existing: Option<ExistingEdge> = self
            .db
            .query("SELECT id FROM $id")
            .bind(("id", edge_id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    async fn find_proposal_by_dedupe_key(&self, dedupe_key: &str) -> Result<Option<ProposedEdge>> {
        let proposals: Vec<ProposedEdgeRow> = self
            .db
            .query(proposal_select_sql("WHERE dedupe_key = $dedupe_key"))
            .bind(("dedupe_key", dedupe_key.to_string()))
            .await?
            .take(0)?;
        proposals
            .into_iter()
            .next()
            .map(ProposedEdgeRow::into_domain)
            .transpose()
    }

    /// Proposal acceptance may only materialize relationships between notes
    /// that are currently visible to the corpus. This also makes a retry safe
    /// if a different process is interrupted after source promotion changes
    /// visibility but before it retires old-generation proposals.
    async fn note_is_visible(&self, id: &RecordId) -> Result<bool> {
        let existing: Option<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE id = $id AND {VISIBLE_NOTE_CONDITION} LIMIT 1"
            ))
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    /// A graph/mention write may address a visible note or the source's
    /// current pending generation. Older hidden and failed generations are
    /// deliberately not writable, even when physical cleanup is deferred.
    async fn note_is_writable(&self, id: &RecordId) -> Result<bool> {
        let existing: Option<Note> = self
            .db
            .query(
                "SELECT * FROM note WHERE id = $id AND (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation OR (source_generation = source_id.generation AND source_id.status = 'pending')) LIMIT 1",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    #[allow(clippy::too_many_arguments)]
    async fn create_audited_edge(
        &self,
        from_id: &RecordId,
        to_id: &RecordId,
        edge_type: EdgeType,
        confidence: Option<f32>,
        reason: Option<&str>,
        provenance: &str,
        proposal_id: Option<&RecordId>,
        is_manual: bool,
    ) -> Result<(RecordId, Option<RecordId>)> {
        validate_note_edge(from_id, to_id, &edge_type)?;
        let mut from_id = from_id.clone();
        let mut to_id = to_id.clone();
        canonicalize_note_edge(&mut from_id, &mut to_id, &edge_type);
        let table = note_edge_table(&edge_type)?;
        let dedupe_key = edge_dedupe_key(&from_id, &to_id, &edge_type);
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
            #[serde(default)]
            proposal_id: Option<RecordId>,
        }
        let existing: Option<IdRow> = self
            .db
            .query(format!(
                "SELECT id, proposal_id FROM {table} WHERE dedupe_key = $dedupe_key LIMIT 1"
            ))
            .bind(("dedupe_key", dedupe_key.clone()))
            .await?
            .take(0)?;
        if let Some(existing) = existing {
            return Ok((existing.id, existing.proposal_id));
        }
        let insert = self.db.query(format!("INSERT INTO {table} (in, out, confidence, reason, provenance, proposal_id, is_manual, dedupe_key, created_at) VALUES ($from, $to, $confidence, $reason, $provenance, $proposal_id, $is_manual, $dedupe_key, time::now()) RETURN id"))
            .bind(("from", from_id)).bind(("to", to_id)).bind(("confidence", confidence.map(|value| value.clamp(0.0, 1.0))))
            .bind(("reason", reason.map(str::to_owned))).bind(("provenance", provenance.to_string())).bind(("proposal_id", proposal_id.cloned()))
            .bind(("is_manual", is_manual)).bind(("dedupe_key", dedupe_key.clone())).await;
        let created_result: Result<Vec<IdRow>> = match insert {
            Ok(mut response) => response.take(0).map_err(Into::into),
            Err(error) => Err(error.into()),
        };
        let created = match created_result {
            Ok(created) => created,
            Err(error) => {
                let existing: Option<IdRow> = self
                    .db
                    .query(format!(
                        "SELECT id, proposal_id FROM {table} WHERE dedupe_key = $dedupe_key LIMIT 1"
                    ))
                    .bind(("dedupe_key", dedupe_key))
                    .await?
                    .take(0)?;
                if let Some(existing) = existing {
                    return Ok((existing.id, existing.proposal_id));
                }
                return Err(error);
            }
        };
        created
            .into_iter()
            .next()
            .map(|row| (row.id, proposal_id.cloned()))
            .ok_or_else(|| DbError::QueryFailed(format!("create {table}")))
    }

    /// Get notes related to a given note (any direction)
    #[instrument(skip(self))]
    pub async fn get_related_notes(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<RelatedNotes> {
        let result: Vec<RelatedNotes> = self
            .db
            .query(format!(
                r#"
                SELECT 
                    (SELECT * FROM ->supports->note WHERE {VISIBLE_NOTE_CONDITION}) AS supporting,
                    (SELECT * FROM <-supports<-note WHERE {VISIBLE_NOTE_CONDITION}) AS supported_by,
                    (SELECT * FROM ->contradicts->note WHERE {VISIBLE_NOTE_CONDITION}) AS contradicting,
                    (SELECT * FROM <-contradicts<-note WHERE {VISIBLE_NOTE_CONDITION}) AS contradicted_by,
                    (SELECT * FROM ->related_to->note WHERE {VISIBLE_NOTE_CONDITION}) AS related,
                    (SELECT * FROM <-related_to<-note WHERE {VISIBLE_NOTE_CONDITION}) AS related_from
                FROM note
                WHERE id = $id AND {VISIBLE_NOTE_CONDITION}
            "#,
            ))
            .bind(("id", note_id.clone()))
            .await?
            .take(0)?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| DbError::NotFound("note".into(), record_id_to_string(note_id)))
    }

    /// Find orphan notes (no connections)
    #[instrument(skip(self))]
    pub async fn find_orphan_notes(&self) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(format!(
                r#"
                SELECT * FROM note 
                WHERE 
                    {VISIBLE_NOTE_CONDITION} AND
                    array::len((SELECT * FROM ->supports->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-supports<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM ->contradicts->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-contradicts<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM ->related_to->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-related_to<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0
            "#
            ))
            .await?
            .take(0)?;

        Ok(notes)
    }

    /// Find potentially related notes (for gardener suggestions)
    #[instrument(skip(self, embedding))]
    pub async fn find_similar_notes(
        &self,
        note_id: &str,
        embedding: Vec<f32>,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<SimilarNote>> {
        let results: Vec<SimilarNote> = self
            .db
            .query(format!(
                r#"
                SELECT
                    id,
                    title,
                    content,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM note
                WHERE
                    {VISIBLE_NOTE_CONDITION} AND
                    id != $note_id AND
                    embedding IS NOT NONE AND
                    vector::similarity::cosine(embedding, $embedding) > $threshold
                ORDER BY similarity DESC
                LIMIT $limit
            "#
            ))
            .bind(("note_id", normalize_note_id(note_id)))
            .bind(("embedding", embedding))
            .bind(("threshold", threshold))
            .bind(("limit", limit))
            .await?
            .take(0)?;

        Ok(results)
    }

    // ==========================================
    // ENTITY OPERATIONS
    // ==========================================

    /// Create or get existing entity by canonical name
    #[instrument(skip(self))]
    pub async fn upsert_entity(&self, entity: Entity) -> Result<Entity> {
        let Entity {
            id: _,
            entity_type,
            name,
            canonical_name,
            embedding,
            metadata,
            created_at: _,
        } = entity;

        let result: Option<Entity> = self.db
            .query(r#"
                INSERT INTO entity (entity_type, name, canonical_name, embedding, metadata, created_at)
                VALUES ($entity_type, $name, $canonical_name, $embedding, $metadata, time::now())
                ON DUPLICATE KEY UPDATE 
                    name = $name,
                    embedding = $embedding,
                    metadata = object::extend(
                        object::extend(metadata ?? {}, $metadata ?? {}),
                        {
                            aliases: array::distinct(array::concat(
                                metadata.aliases ?? [],
                                $metadata.aliases ?? []
                            ))
                        }
                    )
            "#)
            .bind(("entity_type", entity_type.clone()))
            .bind(("name", name.clone()))
            .bind(("canonical_name", canonical_name.clone()))
            .bind(("embedding", embedding.clone()))
            .bind(("metadata", metadata.clone()))
            .await?
            .take(0)?;

        if let Some(entity) = result {
            return Ok(entity);
        }

        // If SurrealDB doesn't return the id on upsert, look it up by canonical name
        let fetched: Option<Entity> = self
            .db
            .query("SELECT * FROM entity WHERE canonical_name = $canonical_name LIMIT 1")
            .bind(("canonical_name", canonical_name))
            .await?
            .take(0)?;

        fetched.ok_or_else(|| DbError::CreateFailed("entity".into()))
    }

    /// Link a note to an entity
    #[instrument(skip(self))]
    pub async fn link_note_to_entity(
        &self,
        note_id: &surrealdb::types::RecordId,
        entity_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_entity_locked(note_id, entity_id).await
    }

    /// Upsert extracted entities and attach the complete result set to a note
    /// while holding the source lifecycle lock.  Reconciliation snapshots
    /// dependent records under that same lock, so a concurrent import sees
    /// either the entire extraction result or none of it; it can never copy a
    /// prefix of a multi-entity extraction to a successor generation.
    #[instrument(skip(self, entities))]
    #[allow(clippy::mutable_key_type)] // Surreal `RecordId` is the database's canonical edge key.
    pub async fn upsert_entities_and_link_note(
        &self,
        note_id: &RecordId,
        entities: Vec<Entity>,
    ) -> Result<usize> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "an entity-link endpoint is hidden, failed, or no longer exists".into(),
            ));
        }

        // Preserve links that predate this batch: a rollback may only remove
        // records this call actually added, never a concurrent/manual link
        // that happened to target the same entity. The lifecycle lock keeps
        // this snapshot stable with respect to supported graph mutations.
        let existing_links: HashSet<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        let mut created_links = Vec::new();
        let mut linked = 0;
        let result: Result<()> = async {
            for entity in entities {
                let entity = self.upsert_entity(entity).await?;
                let entity_id = entity.id.as_ref().ok_or_else(|| {
                    DbError::CreateFailed("upserted entity did not receive an id".into())
                })?;
                if !existing_links.contains(entity_id) {
                    self.link_note_to_entity_locked(note_id, entity_id).await?;
                    created_links.push(entity_id.clone());
                }
                linked += 1;
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            for entity_id in created_links {
                self.db
                    .query("DELETE mentions WHERE in = $note_id AND out = $entity_id")
                    .bind(("note_id", note_id.clone()))
                    .bind(("entity_id", entity_id))
                    .await?;
            }
            return Err(error);
        }
        Ok(linked)
    }

    /// Replace a note's extracted entity mention set after inference has
    /// completed. The complete replacement is applied under the same source
    /// lifecycle lock as reconciliation, so a source refresh can snapshot
    /// either the old complete set or the new complete set, never the old
    /// delete/infer/insert gap.
    #[instrument(skip(self, entities))]
    #[allow(clippy::mutable_key_type)] // Deduplication must retain typed `RecordId`s for writes.
    pub async fn replace_note_entities(
        &self,
        note_id: &RecordId,
        entities: Vec<Entity>,
    ) -> Result<usize> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "an entity-replacement endpoint is hidden, failed, or no longer exists".into(),
            ));
        }

        // Complete all fallible inference-result persistence before replacing
        // mentions. A malformed entity therefore leaves the prior extraction
        // intact instead of clearing it first.
        let entity_ids = self.replacement_entity_ids(entities).await?;

        let previous_ids: Vec<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        self.delete_mentions_for_note_locked(note_id).await?;

        let result: Result<()> = async {
            for entity_id in &entity_ids {
                self.link_note_to_entity_locked(note_id, entity_id).await?;
            }
            Ok(())
        }
        .await;
        if let Err(error) = result {
            // Restore the pre-replacement set if a database failure occurs
            // after deletion. The lock prevents source reconciliation from
            // observing the transient empty set.
            self.delete_mentions_for_note_locked(note_id).await?;
            for entity_id in previous_ids {
                self.link_note_to_entity_locked(note_id, &entity_id).await?;
            }
            return Err(error);
        }
        Ok(entity_ids.len())
    }

    async fn replacement_entity_ids(&self, entities: Vec<Entity>) -> Result<Vec<RecordId>> {
        let mut entity_ids = Vec::with_capacity(entities.len());
        let mut seen = HashSet::new();
        for entity in entities {
            let entity = self.upsert_entity(entity).await?;
            let entity_id = entity.id.ok_or_else(|| {
                DbError::CreateFailed("upserted entity did not receive an id".into())
            })?;
            if seen.insert(entity_id.clone()) {
                entity_ids.push(entity_id);
            }
        }
        Ok(entity_ids)
    }

    async fn link_note_to_entity_locked(
        &self,
        note_id: &surrealdb::types::RecordId,
        entity_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a mention endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM mentions WHERE in = $note_id AND out = $entity_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("entity_id", entity_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count == 0 {
            self.db
                .query("CREATE mentions SET in = $note_id, out = $entity_id")
                .bind(("note_id", note_id.clone()))
                .bind(("entity_id", entity_id.clone()))
                .await?;
        }

        Ok(())
    }

    /// Remove all mention links for a note
    #[instrument(skip(self))]
    pub async fn delete_mentions_for_note(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.delete_mentions_for_note_locked(note_id).await
    }

    async fn delete_mentions_for_note_locked(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a mention endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        self.db
            .query("DELETE mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?;

        Ok(())
    }

    /// Get entities linked to a note
    #[instrument(skip(self))]
    pub async fn get_entities_for_note(&self, note_id: &str) -> Result<Vec<Entity>> {
        let raw = note_id.strip_prefix("note:").unwrap_or(note_id).to_string();
        let note_record_id = RecordId::new("note", raw);

        let entity_ids: Vec<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_record_id))
            .await?
            .take(0)?;

        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut entities = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            let entity: Option<Entity> = self.db.select(entity_id).await?;
            if let Some(entity) = entity {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// Check whether a note has at least one linked entity matching the query.
    #[instrument(skip(self))]
    pub async fn note_has_entity_name(&self, note_id: &str, entity_query: &str) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            #[serde(default)]
            count: Option<u64>,
        }

        let raw = note_id.strip_prefix("note:").unwrap_or(note_id).to_string();
        let note_record_id = RecordId::new("note", raw);
        let normalized = entity_query.trim().to_lowercase();

        if normalized.is_empty() {
            return Ok(true);
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                r#"
                SELECT count() AS count
                FROM mentions
                WHERE in = $note_id
                  AND out IN (
                    SELECT VALUE id
                    FROM entity
                    WHERE canonical_name CONTAINS $entity_query
                  )
                GROUP ALL
            "#,
            )
            .bind(("note_id", note_record_id))
            .bind(("entity_query", normalized))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        Ok(count > 0)
    }

    /// Find a bounded, deterministic set of canonical entities that occur in
    /// the normalized query. Aliases are optional values stored in
    /// `entity.metadata.aliases`; this keeps query-time matching local and
    /// avoids requiring the extraction provider for search.
    #[instrument(skip(self))]
    pub async fn find_graph_entities(
        &self,
        normalized_query: &str,
        limit: usize,
    ) -> Result<Vec<GraphEntityMatch>> {
        let normalized_query = graph_query_normalize(normalized_query);
        if normalized_query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("graph entity limit exceeds database integer range".into())
        })?;

        // Whole-query canonical/alias equality is always preferred over a
        // shorter entity phrase contained in the query. Only when neither
        // lexical tier produces a local entity do we consider typo-style
        // prefix seeds, preventing ordinary sentence words from crowding the
        // cap.
        let exact = self
            .query_graph_entities(&normalized_query, GraphEntityMatchTier::Exact, &[], limit)
            .await?;
        if !exact.is_empty() {
            return Ok(exact);
        }

        let phrases = self
            .query_graph_entities(
                &normalized_query,
                GraphEntityMatchTier::ContainedPhrase,
                &[],
                limit,
            )
            .await?;
        if !phrases.is_empty() {
            return Ok(phrases);
        }

        // Prefixes are a narrow recovery path for partial entity terms such
        // as `atla`. They require four characters and exclude common query
        // words, preserving the `ai`/`chair` boundary safety while avoiding
        // `what` -> `Whatever` false seeds.
        let prefixes = graph_prefix_terms(&normalized_query);
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        self.query_graph_entities(
            &normalized_query,
            GraphEntityMatchTier::Prefix,
            &prefixes,
            limit,
        )
        .await
    }

    async fn query_graph_entities(
        &self,
        normalized_query: &str,
        tier: GraphEntityMatchTier,
        prefixes: &[String],
        limit: i64,
    ) -> Result<Vec<GraphEntityMatch>> {
        // Keep stored names and aliases on the exact same lexical boundary
        // contract as `graph_query_normalize`: punctuation becomes a space,
        // while Unicode letters and numbers remain terms. This lets `GPT-4`
        // be found from either a punctuated or a sentence query.
        let canonical_lexical = r#"string::trim(string::replace(string::lowercase(canonical_name), <regex>"[^\\p{L}\\p{N}]+", ' '))"#;
        let alias_lexical = r#"string::trim(string::replace(string::lowercase($alias), <regex>"[^\\p{L}\\p{N}]+", ' '))"#;
        let match_condition = match tier {
            GraphEntityMatchTier::Exact => format!(
                "{canonical_lexical} = $query OR array::any(metadata.aliases ?? [], |$alias| {alias_lexical} = $query)"
            ),
            GraphEntityMatchTier::ContainedPhrase => format!(
                "string::contains(string::concat(' ', $query, ' '), string::concat(' ', {canonical_lexical}, ' ')) OR array::any(metadata.aliases ?? [], |$alias| string::contains(string::concat(' ', $query, ' '), string::concat(' ', {alias_lexical}, ' ')))"
            ),
            GraphEntityMatchTier::Prefix => prefixes
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    format!("string::starts_with({canonical_lexical}, $prefix_{index})")
                })
                .collect::<Vec<_>>()
                .join(" OR "),
        };
        let phrase_specificity = match tier {
            GraphEntityMatchTier::ContainedPhrase => {
                // A sentence can contain both `New` and `New York`. Rank by
                // the lexical phrase that actually matched (canonical or
                // alias), rather than the stored canonical name, so aliases
                // receive the same specificity treatment before the cap.
                let matching_alias_lengths = format!(
                    "array::map(array::filter(metadata.aliases ?? [], |$alias| string::contains(string::concat(' ', $query, ' '), string::concat(' ', {alias_lexical}, ' '))), |$alias| string::len({alias_lexical}))"
                );
                let phrase_specificity = format!(
                    "array::max([IF string::contains(string::concat(' ', $query, ' '), string::concat(' ', {canonical_lexical}, ' ')) THEN string::len({canonical_lexical}) ELSE 0 END, array::max({matching_alias_lengths})])"
                );
                Some(phrase_specificity)
            }
            GraphEntityMatchTier::Exact | GraphEntityMatchTier::Prefix => None,
        };
        let prefix_plausibility = match tier {
            GraphEntityMatchTier::Prefix => Some(format!(
                "array::min([{}])",
                prefixes
                    .iter()
                    .enumerate()
                    .map(|(index, prefix)| {
                        // Prefix recovery is for short, likely truncated
                        // entity fragments. Prefer the shortest matching
                        // query fragment so a context word such as
                        // `deployed` cannot crowd out `zeta` at a small cap.
                        format!(
                            "IF string::starts_with({canonical_lexical}, $prefix_{index}) THEN {} ELSE 2147483647 END",
                            prefix.chars().count()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            GraphEntityMatchTier::Exact | GraphEntityMatchTier::ContainedPhrase => None,
        };
        let select_specificity = phrase_specificity
            .as_ref()
            .map(|specificity| format!(", {specificity} AS graph_match_specificity"))
            .unwrap_or_default();
        let select_prefix_plausibility = prefix_plausibility
            .as_ref()
            .map(|plausibility| format!(", {plausibility} AS graph_prefix_plausibility"))
            .unwrap_or_default();
        let ordering = if phrase_specificity.is_some() {
            "graph_match_specificity DESC, canonical_name ASC, id ASC"
        } else if prefix_plausibility.is_some() {
            "graph_prefix_plausibility ASC, canonical_name ASC, id ASC"
        } else {
            "canonical_name ASC, id ASC"
        };
        let query = format!(
            r#"
                SELECT id, name, canonical_name, metadata{select_specificity}{select_prefix_plausibility}
                FROM entity
                WHERE {match_condition}
                ORDER BY {ordering}
                LIMIT $limit
                "#,
        );
        let mut query = self
            .db
            .query(query)
            .bind(("query", normalized_query.to_string()))
            .bind(("limit", limit));
        for (index, prefix) in prefixes.iter().enumerate() {
            query = query.bind((format!("prefix_{index}"), prefix.clone()));
        }
        query.await?.take(0).map_err(Into::into)
    }

    /// Fetch visible note IDs mentioned by any supplied entities in one
    /// query. The caller owns the cap, so an entity with a high degree cannot
    /// cause an unbounded graph seed set.
    #[instrument(skip(self, entity_ids))]
    pub async fn graph_notes_for_entities(
        &self,
        entity_ids: &[RecordId],
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<GraphEntityNoteSeed>> {
        if entity_ids.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        // Query a bounded page for every ranked entity. A single global
        // relation `LIMIT` ordered by note ID lets a high-degree, lower-priority
        // entity starve later matches before the caller can allocate the
        // unique-note cap across them. `entity_ids` is itself bounded by graph
        // configuration, so this remains bounded while preserving coverage.
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("graph note limit exceeds database integer range".into())
        })?;
        let since = since.map(|timestamp| timestamp.to_rfc3339());
        let mut seeds = Vec::new();
        for entity_id in entity_ids {
            let mut entity_seeds: Vec<GraphEntityNoteSeed> = self
                .db
                .query(format!(
                    "SELECT in AS note_id, out AS entity_id FROM mentions WHERE out = $entity_id AND in IN (SELECT VALUE id FROM note WHERE ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION}) ORDER BY in ASC LIMIT $limit"
                ))
                .bind(("entity_id", entity_id.clone()))
                .bind(("limit", limit))
                .bind(("since", since.clone()))
                .bind(("source_uri", source_uri.clone()))
                .await?
                .take(0)?;
            seeds.append(&mut entity_seeds);
        }
        Ok(seeds)
    }

    /// Fetch accepted persisted note-edge rows for many frontier notes.
    /// Every frontier note gets its own deterministic per-table budget; a
    /// high-degree early note cannot consume a shared table `LIMIT` and starve
    /// later seeds. Proposal tables are deliberately never consulted.
    #[instrument(skip(self, note_ids, edge_types))]
    #[allow(clippy::too_many_arguments)]
    pub async fn graph_note_edges(
        &self,
        note_ids: &[RecordId],
        edge_types: &[String],
        per_table_limit: usize,
        allow_outbound: bool,
        allow_inbound: bool,
        min_confidence: f32,
        since: Option<DateTime<Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<NoteEdgeRow>> {
        self.graph_note_edges_excluding_visited(
            note_ids,
            edge_types,
            per_table_limit,
            allow_outbound,
            allow_inbound,
            min_confidence,
            since,
            source_uri,
            &HashMap::new(),
        )
        .await
    }

    /// Fetch bounded graph edges while excluding endpoint notes already on the
    /// path for each current source. Applying this exclusion in the database
    /// is necessary: otherwise a high-confidence back-edge can consume the
    /// per-source table limit before traversal rejects the cycle in memory.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, note_ids, edge_types, visited_note_ids))]
    pub async fn graph_note_edges_excluding_visited(
        &self,
        note_ids: &[RecordId],
        edge_types: &[String],
        per_table_limit: usize,
        allow_outbound: bool,
        allow_inbound: bool,
        min_confidence: f32,
        since: Option<DateTime<Utc>>,
        source_uri: Option<String>,
        visited_note_ids: &HashMap<String, Vec<RecordId>>,
    ) -> Result<Vec<NoteEdgeRow>> {
        if note_ids.is_empty()
            || edge_types.is_empty()
            || per_table_limit == 0
            || (!allow_outbound && !allow_inbound)
        {
            return Ok(Vec::new());
        }
        let since = since.map(|timestamp| timestamp.to_rfc3339());
        let mut rows = HashMap::<String, NoteEdgeRow>::new();
        for table in ["supports", "contradicts", "related_to", "derived_from"] {
            if !edge_types.iter().any(|edge_type| edge_type == table) {
                continue;
            }
            let limit = i64::try_from(per_table_limit).map_err(|_| {
                DbError::QueryFailed("graph edge limit exceeds database integer range".into())
            })?;
            // SurrealDB executes these per-source bounded statements in one
            // request/response. A single global `LIMIT` is incorrect here:
            // it can return only high-degree early sources. Keeping the
            // statements batched avoids client round-trips while making the
            // per-source budget exact and deterministic.
            let eligible_notes = format!(
                "(SELECT VALUE id FROM note WHERE ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION})"
            );
            let query = (0..note_ids.len())
                .map(|index| {
                    let direction = match (allow_outbound, allow_inbound) {
                        (true, true) => format!(
                            "((in = $note_{index} AND out IN {eligible_notes} AND out NOT IN $visited_{index}) OR (out = $note_{index} AND in IN {eligible_notes} AND in NOT IN $visited_{index}))"
                        ),
                        (true, false) => {
                            format!("in = $note_{index} AND out IN {eligible_notes} AND out NOT IN $visited_{index}")
                        }
                        (false, true) => {
                            format!("out = $note_{index} AND in IN {eligible_notes} AND in NOT IN $visited_{index}")
                        }
                        (false, false) => "false".to_string(),
                    };
                    format!(
                        "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at, IF confidence = NONE THEN 1.0 ELSE confidence END AS graph_confidence \
                         FROM {table} WHERE {direction} AND (confidence = NONE OR confidence >= $min_confidence) AND {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION} \
                         ORDER BY graph_confidence DESC, id ASC LIMIT $limit;"
                    )
                })
                .collect::<String>();
            let mut query = self
                .db
                .query(query)
                .bind(("limit", limit))
                .bind(("min_confidence", min_confidence))
                .bind(("since", since.clone()))
                .bind(("source_uri", source_uri.clone()));
            for (index, note_id) in note_ids.iter().enumerate() {
                query = query.bind((format!("note_{index}"), note_id.clone()));
                query = query.bind((
                    format!("visited_{index}"),
                    visited_note_ids
                        .get(&record_id_to_string(note_id))
                        .cloned()
                        .unwrap_or_default(),
                ));
            }
            let mut response = query.await?;
            for index in 0..note_ids.len() {
                let edges: Vec<NoteEdgeRow> = response.take(index)?;
                for edge in edges {
                    rows.entry(record_id_to_string(&edge.id)).or_insert(edge);
                }
            }
        }
        let mut rows = rows.into_values().collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            left.edge_type
                .cmp(&right.edge_type)
                .then_with(|| record_id_to_string(&left.id).cmp(&record_id_to_string(&right.id)))
        });
        Ok(rows)
    }

    /// Load graph-selected notes in one visibility-aware query so deleted or
    /// superseded endpoints are silently excluded from retrieval.
    #[instrument(skip(self, note_ids))]
    pub async fn graph_notes_by_ids(
        &self,
        note_ids: &[RecordId],
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let since = since.map(|timestamp| timestamp.to_rfc3339());
        self.db
            .query(format!(
                "SELECT id, title, content, note_type, tags, created_at, source_id.uri AS source_uri \
                 FROM note WHERE id IN $note_ids AND ($since = NONE OR created_at >= <datetime>$since) \
                 AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION} \
                 ORDER BY id ASC"
            ))
            .bind(("note_ids", note_ids.to_vec()))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)
            .map_err(Into::into)
    }

    /// Return the original chat provenance record IDs for graph-selected
    /// notes in two bounded set queries. File-backed notes retain their
    /// source URI in [`SearchResult`]; chat-derived notes need these record
    /// IDs to keep an augmentation citation reconstructable without loading
    /// each note individually.
    #[instrument(skip(self, note_ids))]
    pub async fn graph_note_provenance_ids(
        &self,
        note_ids: &[RecordId],
    ) -> Result<HashMap<String, Vec<String>>> {
        #[derive(Deserialize, SurrealValue)]
        struct ProvenanceRow {
            r#in: RecordId,
            out: RecordId,
        }

        if note_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut provenance = HashMap::<String, Vec<String>>::new();
        for table in ["note_from_conversation", "note_from_message"] {
            let rows: Vec<ProvenanceRow> = self
                .db
                .query(format!("SELECT in, out FROM {table} WHERE in IN $note_ids"))
                .bind(("note_ids", note_ids.to_vec()))
                .await?
                .take(0)?;
            for row in rows {
                provenance
                    .entry(record_id_to_string(&row.r#in))
                    .or_default()
                    .push(record_id_to_string(&row.out));
            }
        }
        for ids in provenance.values_mut() {
            ids.sort();
            ids.dedup();
        }
        Ok(provenance)
    }

    /// List note-to-note edges across all edge tables
    #[instrument(skip(self))]
    pub async fn list_note_edges(&self, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let mut edges: Vec<NoteEdgeRow> = Vec::new();
        let limit = limit.max(1);

        edges.extend(self.query_edges_table("supports", limit).await?);
        edges.extend(self.query_edges_table("contradicts", limit).await?);
        edges.extend(self.query_edges_table("related_to", limit).await?);
        edges.extend(self.query_edges_table("derived_from", limit).await?);

        Ok(edges)
    }

    /// Get note-to-note edges for a specific note id (in or out)
    #[instrument(skip(self))]
    pub async fn get_note_edges(&self, note_id: &str) -> Result<Vec<NoteEdgeRow>> {
        let note_id = normalize_note_id(note_id);
        let mut edges: Vec<NoteEdgeRow> = Vec::new();

        edges.extend(self.query_edges_for_note("supports", &note_id).await?);
        edges.extend(self.query_edges_for_note("contradicts", &note_id).await?);
        edges.extend(self.query_edges_for_note("related_to", &note_id).await?);
        edges.extend(self.query_edges_for_note("derived_from", &note_id).await?);

        Ok(edges)
    }

    /// Copy relationships owned by stable source chunks to their staged
    /// successors before a copy-on-write source generation is promoted. The
    /// old generation remains authoritative until promotion/cleanup; should a
    /// copy fail, removing the staged generation leaves its dependents intact.
    ///
    /// The mapping may contain only chunks that reconciled successfully. Its
    /// third value records whether the successor's displayed content is
    /// byte-for-byte unchanged. A removed or ambiguous chunk deliberately has
    /// no successor, so its dependents follow the normal source-lifecycle
    /// cascade instead of being attached to unrelated content.
    #[instrument(skip(self, successors))]
    pub async fn copy_note_dependents_to_successors(
        &self,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.copy_note_dependents_to_successors_locked(successors)
            .await
    }

    #[allow(clippy::mutable_key_type)] // Typed `RecordId` maps are copied directly into edge writes.
    async fn copy_note_dependents_to_successors_locked(
        &self,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<()> {
        if successors.is_empty() {
            return Ok(());
        }
        let exact_content_successors = successors
            .iter()
            .filter_map(|(old_id, _, exact_content)| exact_content.then_some(old_id.clone()))
            .collect::<HashSet<_>>();
        let successors = successors
            .iter()
            .map(|(old_id, new_id, _)| (old_id.clone(), new_id.clone()))
            .collect::<HashMap<_, _>>();

        // Entity mentions describe extracted source text, so only carry them
        // across when that displayed text is exactly unchanged. A changed
        // successor remains mention-free and is therefore eligible for a new
        // entity-extraction pass. Chat provenance identifies origin rather
        // than extracted content, so it follows every safely reconciled chunk.
        for (old_id, new_id) in &successors {
            if exact_content_successors.contains(old_id) {
                let entity_ids: Vec<RecordId> = self
                    .db
                    .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
                    .bind(("note_id", old_id.clone()))
                    .await?
                    .take(0)?;
                for entity_id in entity_ids {
                    self.link_note_to_entity_locked(new_id, &entity_id).await?;
                }
            }

            let conversation_ids: Vec<RecordId> = self
                .db
                .query("SELECT VALUE out FROM note_from_conversation WHERE in = $note_id")
                .bind(("note_id", old_id.clone()))
                .await?
                .take(0)?;
            for conversation_id in conversation_ids {
                self.link_note_to_conversation_locked(new_id, &conversation_id)
                    .await?;
            }

            let message_ids: Vec<RecordId> = self
                .db
                .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
                .bind(("note_id", old_id.clone()))
                .await?
                .take(0)?;
            for message_id in message_ids {
                self.link_note_to_message_locked(new_id, &message_id)
                    .await?;
            }
        }

        // Snapshot each edge once before writing successors. This handles an
        // edge whose two endpoints are both reconciled chunks and preserves
        // manual as well as generated graph relationships.
        #[allow(clippy::mutable_key_type)] // Edge ids remain typed for successor copying.
        let mut seen_edges = HashSet::new();
        let mut edges = Vec::new();
        for old_id in successors.keys() {
            for edge in self.get_note_edges(&record_id_to_string(old_id)).await? {
                if seen_edges.insert(edge.id.clone()) {
                    edges.push(edge);
                }
            }
        }
        for edge in edges {
            let from_id = successors.get(&edge.in_id).cloned().unwrap_or(edge.in_id);
            let to_id = successors.get(&edge.out_id).cloned().unwrap_or(edge.out_id);
            if from_id == to_id {
                // A many-to-one reconciliation must not manufacture an
                // invalid self-edge. Current Markdown keys are one-to-one,
                // but retaining this guard makes the copy routine safe for
                // future reconciliation strategies.
                continue;
            }
            self.create_audited_edge(
                &from_id,
                &to_id,
                persisted_note_edge_type(&edge.edge_type)?,
                edge.confidence,
                edge.reason.as_deref(),
                edge.provenance
                    .as_deref()
                    .unwrap_or("source-reconciliation"),
                edge.proposal_id.as_ref(),
                edge.is_manual,
            )
            .await?;
        }
        Ok(())
    }

    // ==========================================
    // SOURCE OPERATIONS
    // ==========================================

    /// Create a source and return its database-assigned id.
    #[instrument(skip(self, source))]
    pub async fn create_source(&self, source: Source) -> Result<Source> {
        let created: Option<Source> = self
            .db
            .query("CREATE source CONTENT $source RETURN AFTER")
            .bind(("source", source_content_value(&source)?))
            .await?
            .take(0)?;
        created.ok_or_else(|| DbError::CreateFailed("source".into()))
    }

    /// List sources in stable identity order for human and JSON CLI output.
    #[instrument(skip(self))]
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        Ok(self
            .db
            .query("SELECT * FROM source ORDER BY normalized_uri, created_at, id")
            .await?
            .take(0)?)
    }

    /// Resolve a source by record id or by its normalized/legacy URI.
    #[instrument(skip(self))]
    pub async fn get_source(&self, id_or_uri: &str) -> Result<Option<Source>> {
        if let Some(raw_id) = id_or_uri.strip_prefix("source:") {
            let source: Option<Source> = self.db.select(("source", raw_id)).await?;
            return Ok(source);
        }

        let source: Option<Source> = self
            .db
            .query("SELECT * FROM source WHERE normalized_uri = $key OR uri = $key LIMIT 1")
            .bind(("key", id_or_uri.to_string()))
            .await?
            .take(0)?;
        Ok(source)
    }

    /// Begin a staged import. The old successful generation remains untouched
    /// until `complete_file_import` succeeds, making failed refreshes safe.
    #[instrument(skip(self, content))]
    pub async fn begin_file_import(
        &self,
        source_type: SourceType,
        title: String,
        normalized_uri: String,
        content: String,
        content_hash: String,
        force: bool,
    ) -> Result<SourceImportPlan> {
        if let Some(mut existing) = self.get_source(&normalized_uri).await? {
            if existing.source_type == SourceType::Manual {
                return Err(DbError::QueryFailed(format!(
                    "refusing to replace manual source at {normalized_uri}"
                )));
            }
            if !force
                && existing.status == SourceIngestionStatus::Ready
                && existing.content_hash.as_deref() == Some(content_hash.as_str())
            {
                // A process can stop after promotion and before old-generation
                // cleanup. An otherwise unchanged retry is the natural
                // recovery path; finish that deferred cleanup before reporting
                // a no-op so stale records cannot accumulate indefinitely.
                let cleanup = self.cleanup_non_successful_generations(&existing).await?;
                return Ok(SourceImportPlan {
                    source: existing,
                    action: SourceImportAction::Unchanged,
                    cleanup,
                });
            }

            existing.generation = existing.generation.saturating_add(1).max(1);
            existing.source_type = source_type;
            existing.title = Some(title);
            existing.uri = Some(normalized_uri.clone());
            existing.normalized_uri = Some(normalized_uri);
            existing.content = Some(content);
            existing.content_hash = Some(content_hash);
            existing.status = SourceIngestionStatus::Pending;
            existing.last_error = None;
            existing.updated_at = chrono::Utc::now();
            self.replace_source(&existing).await?;
            return Ok(SourceImportPlan {
                source: existing,
                action: SourceImportAction::Updated,
                cleanup: SourceDeleteSummary::default(),
            });
        }

        let now = chrono::Utc::now();
        let source = Source {
            id: None,
            source_type,
            title: Some(title),
            uri: Some(normalized_uri.clone()),
            normalized_uri: Some(normalized_uri),
            content: Some(content),
            content_hash: Some(content_hash),
            generation: 1,
            successful_generation: 0,
            status: SourceIngestionStatus::Pending,
            last_error: None,
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            last_ingested_at: None,
        };
        Ok(SourceImportPlan {
            source: self.create_source(source).await?,
            action: SourceImportAction::Created,
            cleanup: SourceDeleteSummary::default(),
        })
    }

    /// Promote an import before removing superseded records. If cleanup is
    /// interrupted, the new generation is already searchable and the old
    /// generation remains recoverable (but hidden) for a later cleanup.
    /// Manual and legacy notes have no generation and survive.
    #[instrument(skip(self, source))]
    pub async fn complete_file_import(&self, source: &mut Source) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.complete_file_import_locked(source).await
    }

    /// Atomically carry reconciled dependents into a staged source generation,
    /// promote it, then retire the superseded generation. A graph mutation
    /// therefore either lands before the dependent snapshot and is copied, or
    /// observes the completed transition instead of being silently discarded
    /// during old-generation cleanup.
    #[instrument(skip(self, source, successors))]
    pub async fn reconcile_file_import(
        &self,
        source: &mut Source,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.copy_note_dependents_to_successors_locked(successors)
            .await?;
        self.complete_file_import_locked(source).await
    }

    async fn complete_file_import_locked(
        &self,
        source: &mut Source,
    ) -> Result<SourceDeleteSummary> {
        // Promotion, proposal retargeting, and old-generation cleanup are one
        // lifecycle transition. In particular, acceptance/undo must not run
        // after the old endpoints become hidden but before an accepted
        // proposal is retargeted to its staged replacement edge.
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        let summary = self
            .source_delete_summary(&source_id, Some(source.generation), true)
            .await?;
        self.promote_file_import_locked(source).await?;
        // Proposal-backed edges are staged copy-on-write with their new note
        // endpoints. Once promotion makes those endpoints authoritative,
        // retarget the accepted proposal before deleting the old generation so
        // its audit row and undo path follow the replacement edge.
        self.retarget_reconciled_proposals(&source_id, source.generation)
            .await?;
        // Do this only after durable promotion. A failure here can leave old
        // records behind, but cannot leave the corpus with no visible complete
        // generation; visibility selects `successful_generation`.
        self.delete_source_notes_locked(&source_id, Some(source.generation), true)
            .await?;
        Ok(summary)
    }

    async fn retarget_reconciled_proposals(
        &self,
        source_id: &RecordId,
        generation: u64,
    ) -> Result<()> {
        let note_ids = self
            .source_owned_note_ids(source_id, Some(generation), false)
            .await?;
        #[allow(clippy::mutable_key_type)] // Edge ids remain typed for proposal retargeting.
        let mut seen_edges = HashSet::new();
        for note_id in note_ids {
            for edge in self.get_note_edges(&record_id_to_string(&note_id)).await? {
                let Some(proposal_id) = edge.proposal_id.as_ref() else {
                    continue;
                };
                if !seen_edges.insert(edge.id.clone()) {
                    continue;
                }
                let edge_type = persisted_note_edge_type(&edge.edge_type)?;
                let mut from_id = edge.in_id.clone();
                let mut to_id = edge.out_id.clone();
                canonicalize_note_edge(&mut from_id, &mut to_id, &edge_type);
                let dedupe_key = edge_dedupe_key(&from_id, &to_id, &edge_type);
                #[derive(Deserialize, SurrealValue)]
                struct UpdatedRow {
                    id: RecordId,
                }
                let updated: Option<UpdatedRow> = self
                    .db
                    .query(
                        "UPDATE $proposal SET in = $from, out = $to, dedupe_key = $dedupe_key, resulting_edge_id = $edge, updated_at = time::now() WHERE status = 'accepted' RETURN AFTER",
                    )
                    .bind(("proposal", proposal_id.clone()))
                    .bind(("from", from_id))
                    .bind(("to", to_id))
                    .bind(("dedupe_key", dedupe_key))
                    .bind(("edge", edge.id.clone()))
                    .await?
                    .take(0)?;
                if updated.is_none() {
                    let proposal = self.get_edge_proposal(proposal_id).await?.ok_or_else(|| {
                        DbError::NotFound(
                            "reconciled proposal".into(),
                            record_id_to_string(proposal_id),
                        )
                    })?;
                    if matches!(
                        proposal.status,
                        ProposedEdgeStatus::Rejected | ProposedEdgeStatus::Superseded
                    ) {
                        // A crash can leave a staged copy of an accepted edge
                        // after the original edge was undone. Its proposal is
                        // terminal, so preserve that user decision by dropping
                        // the stale copied edge rather than making recovery
                        // fail forever trying to retarget it.
                        self.db
                            .query("DELETE $edge")
                            .bind(("edge", edge.id.clone()))
                            .await?
                            .check()?;
                        continue;
                    }
                    return Err(DbError::QueryFailed(format!(
                        "reconciled proposal {} is no longer accepted",
                        record_id_to_string(proposal_id)
                    )));
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    async fn promote_file_import(&self, source: &mut Source) -> Result<()> {
        // Promotion changes which source generation is visible. Keep that
        // transition and retirement of proposals for the newly hidden notes
        // atomic with respect to proposal acceptance in this repository.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.promote_file_import_locked(source).await
    }

    async fn promote_file_import_locked(&self, source: &mut Source) -> Result<()> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        // Do not update the caller's in-memory source until the visibility
        // transition is durable. Callers use this state to distinguish a
        // pre-promotion failure (safe to discard staged notes) from a later
        // cleanup failure (new generation must remain intact for recovery).
        let mut promoted = source.clone();
        promoted.successful_generation = promoted.generation;
        promoted.status = SourceIngestionStatus::Ready;
        promoted.last_error = None;
        promoted.updated_at = chrono::Utc::now();
        promoted.last_ingested_at = Some(promoted.updated_at);
        self.replace_source(&promoted).await?;
        *source = promoted;
        // Promotion makes older source generations invisible even if their
        // destructive cleanup is interrupted. Retire their pending proposals
        // at the same durable boundary so batch acceptance cannot create an
        // edge for a hidden endpoint in that window.
        let superseded_notes = self
            .source_owned_note_ids(&source_id, Some(source.generation), true)
            .await?;
        self.supersede_pending_proposals_for_notes(&superseded_notes)
            .await
    }

    async fn cleanup_non_successful_generations(
        &self,
        source: &Source,
    ) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        // A process can stop after promotion but before proposal retargeting
        // or old-generation deletion. Retry retargeting before cleanup so an
        // accepted proposal's audit and undo path follows its visible staged
        // replacement instead of being retired with the old generation.
        self.retarget_reconciled_proposals(source_id, source.successful_generation)
            .await?;
        self.delete_source_notes_locked(source_id, Some(source.successful_generation), true)
            .await
    }

    /// Remove partially-created notes for a failed generation and retain the
    /// last successful generation. The source remains resumable via reimport.
    #[instrument(skip(self, source, error))]
    pub async fn fail_file_import(&self, source: &mut Source, error: impl ToString) -> Result<()> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        self.delete_source_notes(source_id, Some(source.generation), false)
            .await?;
        source.status = SourceIngestionStatus::Failed;
        source.last_error = Some(error.to_string());
        source.updated_at = chrono::Utc::now();
        self.replace_source(source).await
    }

    /// Count every record that source deletion would mutate. This is used for
    /// dry-run output and intentionally excludes manual/legacy notes.
    #[instrument(skip(self, source))]
    pub async fn preview_source_delete(&self, source: &Source) -> Result<SourceDeleteSummary> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::NotFound("source".into(), "missing id".into()))?;
        self.source_delete_summary(source_id, None, false).await
    }

    /// Delete a source and the records it owns. Edges/provenance/mentions are
    /// removed before notes, so no dangling graph records remain. Shared entity
    /// records are deliberately retained: without per-source entity ownership,
    /// deleting an unmentioned entity could erase a user-authored entity.
    #[instrument(skip(self, source))]
    pub async fn delete_source(&self, source: &Source) -> Result<SourceDeleteSummary> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::NotFound("source".into(), "missing id".into()))?;
        let summary = self.delete_source_notes(source_id, None, false).await?;
        let _: Option<Source> = self.db.delete(source_id.clone()).await?;
        Ok(summary)
    }

    async fn replace_source(&self, source: &Source) -> Result<()> {
        let id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        let content = source_content_value(source)?;
        self.db
            .query("UPDATE $id MERGE $source")
            .bind(("id", id.clone()))
            .bind(("source", content))
            .await?
            .check()?;
        self.db
            .query(
                "UPDATE $id SET updated_at = <datetime>$updated_at, \
                 last_error = $last_error, \
                 last_ingested_at = IF $last_ingested_at = NONE THEN NONE ELSE <datetime>$last_ingested_at END",
            )
            .bind(("id", id.clone()))
            .bind(("updated_at", source.updated_at.to_rfc3339()))
            .bind(("last_error", source.last_error.clone()))
            .bind((
                "last_ingested_at",
                source.last_ingested_at.map(|time| time.to_rfc3339()),
            ))
            .await?
            .check()?;
        Ok(())
    }

    async fn delete_source_notes(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        // Source cleanup shares the same endpoint/acceptance critical section
        // as single-note deletion.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.delete_source_notes_locked(source_id, generation, older_than_generation)
            .await
    }

    async fn delete_source_notes_locked(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        let summary = self
            .source_delete_summary(source_id, generation, older_than_generation)
            .await?;
        let notes = self
            .source_owned_note_ids(source_id, generation, older_than_generation)
            .await?;
        self.supersede_proposals_for_removed_notes(&notes).await?;
        self.delete_notes_and_dependents(&notes).await?;
        Ok(summary)
    }

    /// Delete note rows and every relationship/provenance record owned by or
    /// incident on them. Proposal retirement is deliberately separate so
    /// callers can choose the lifecycle transition before this physical
    /// cascade runs.
    async fn delete_notes_and_dependents(&self, notes: &[RecordId]) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "DELETE supports WHERE in = $note OR out = $note; \
                     DELETE contradicts WHERE in = $note OR out = $note; \
                     DELETE derived_from WHERE in = $note OR out = $note; \
                     DELETE related_to WHERE in = $note OR out = $note; \
                     DELETE mentions WHERE in = $note; \
                     DELETE note_from_conversation WHERE in = $note; \
                     DELETE note_from_message WHERE in = $note; \
                     DELETE $note;",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    /// Retire every mutable proposal whose source-owned endpoint is being
    /// removed. Accepted proposals lose their resulting edge reference along
    /// with the edge itself, matching [`Self::undo_edge`] semantics.
    async fn supersede_proposals_for_removed_notes(&self, notes: &[RecordId]) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'proposal endpoint removed by source lifecycle', resulting_edge_id = NONE, updated_at = time::now() WHERE (status = 'pending' OR status = 'accepting' OR status = 'accepted') AND (in = $note OR out = $note)",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    /// Retire pending suggestions whose source-owned endpoint is no longer
    /// usable. Promotion uses this narrower transition because old-generation
    /// accepted edges remain intact until deferred destructive cleanup runs.
    async fn supersede_pending_proposals_for_notes(&self, notes: &[RecordId]) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'proposal endpoint removed by source lifecycle', updated_at = time::now() WHERE (status = 'pending' OR status = 'accepting') AND (in = $note OR out = $note)",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    async fn source_owned_note_ids(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<Vec<RecordId>> {
        let condition = match (generation, older_than_generation) {
            (Some(_), true) => "source_generation IS NOT NONE AND source_generation != $generation",
            (Some(_), false) => "source_generation = $generation",
            (None, _) => "source_generation IS NOT NONE",
        };
        let query =
            format!("SELECT VALUE id FROM note WHERE source_id = $source_id AND {condition}");
        let mut request = self.db.query(query).bind(("source_id", source_id.clone()));
        if let Some(generation) = generation {
            request = request.bind(("generation", generation as i64));
        }
        Ok(request.await?.take(0)?)
    }

    async fn source_delete_summary(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        let notes = self
            .source_owned_note_ids(source_id, generation, older_than_generation)
            .await?;
        self.delete_summary_for_notes(&notes).await
    }

    /// Count the relationships and provenance records that physical note
    /// cleanup removes. Shared source/note accounting keeps dry-run previews
    /// exact and makes confirmed single-note deletion report the same shape.
    async fn delete_summary_for_notes(&self, notes: &[RecordId]) -> Result<SourceDeleteSummary> {
        let mut summary = SourceDeleteSummary::default();
        summary.notes = notes.len() as u64;
        summary.note_edges = self.count_note_edges_for_notes(&notes).await?;
        summary.proposals = self.count_mutable_proposals_for_notes(&notes).await?;
        for note_id in notes {
            let counts: Vec<SourceDeleteCount> = self
                .db
                .query(
                    "RETURN [\
                       { kind: 'mentions', count: (SELECT count() FROM mentions WHERE in = $note GROUP ALL)[0].count },\
                       { kind: 'conversation_provenance', count: (SELECT count() FROM note_from_conversation WHERE in = $note GROUP ALL)[0].count },\
                       { kind: 'message_provenance', count: (SELECT count() FROM note_from_message WHERE in = $note GROUP ALL)[0].count }\
                     ];",
                )
                .bind(("note", note_id.clone()))
                .await?
                .take(0)?;
            for count in counts {
                match count.kind.as_str() {
                    "mentions" => summary.mentions += count.count,
                    "conversation_provenance" => {
                        summary.note_conversation_provenance += count.count
                    }
                    "message_provenance" => summary.note_message_provenance += count.count,
                    _ => {}
                }
            }
        }
        Ok(summary)
    }

    /// Count each edge row once across all owned notes. An internal edge is
    /// reachable from two endpoints but is deleted exactly once, so summing
    /// per-note counts would make dry-run output inaccurate.
    async fn count_note_edges_for_notes(&self, notes: &[RecordId]) -> Result<u64> {
        let mut total = 0_u64;
        for table in ["supports", "contradicts", "derived_from", "related_to"] {
            #[derive(Deserialize, SurrealValue)]
            struct CountRow {
                #[serde(default)]
                count: Option<u64>,
            }

            let query = format!(
                "SELECT count() FROM {table} WHERE in IN $notes OR out IN $notes GROUP ALL"
            );
            let row: Option<CountRow> = self
                .db
                .query(query)
                .bind(("notes", notes.to_vec()))
                .await?
                .take(0)?;
            total += row.and_then(|row| row.count).unwrap_or(0);
        }
        Ok(total)
    }

    /// Count proposal records that source cleanup will transition. Proposal
    /// rows are not deleted: they retain an auditable terminal decision.
    async fn count_mutable_proposals_for_notes(&self, notes: &[RecordId]) -> Result<u64> {
        if notes.is_empty() {
            return Ok(0);
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            #[serde(default)]
            count: Option<u64>,
        }
        let row: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM proposed_edge WHERE (status = 'pending' OR status = 'accepting' OR status = 'accepted') AND (in IN $notes OR out IN $notes) GROUP ALL",
            )
            .bind(("notes", notes.to_vec()))
            .await?
            .take(0)?;
        Ok(row.and_then(|row| row.count).unwrap_or(0))
    }

    // ==========================================
    // CHAT IMPORT OPERATIONS
    // ==========================================

    /// Upsert a conversation record from a chat export conversation.
    #[instrument(skip(self, conversation, metadata))]
    pub async fn upsert_conversation(
        &self,
        conversation: &ChatConversation,
        source_uri: Option<String>,
        metadata: serde_json::Value,
        summary_embedding: Option<Vec<f32>>,
    ) -> Result<RecordId> {
        #[derive(Debug, Deserialize, SurrealValue)]
        struct ConversationIdRow {
            id: RecordId,
        }

        let account_uuid = conversation
            .account
            .as_ref()
            .map(|account| account.uuid.clone());
        let summary = if conversation.summary.is_empty() {
            None
        } else {
            Some(conversation.summary.clone())
        };

        let upserted: Option<ConversationIdRow> = self
            .db
            .query(
                r#"
                INSERT INTO conversation (
                    uuid, title, summary, source_uri, account_uuid, metadata, summary_embedding, created_at, updated_at, ingested_at
                )
                VALUES (
                    $uuid, $title, $summary, $source_uri, $account_uuid, $metadata, $summary_embedding, <datetime>$created_at, <datetime>$updated_at, time::now()
                )
                ON DUPLICATE KEY UPDATE
                    title = $title,
                    summary = $summary,
                    source_uri = $source_uri,
                    account_uuid = $account_uuid,
                    metadata = $metadata,
                    summary_embedding = $summary_embedding,
                    created_at = <datetime>$created_at,
                    updated_at = <datetime>$updated_at,
                    ingested_at = time::now()
            "#,
            )
            .bind(("uuid", conversation.uuid.clone()))
            .bind(("title", conversation.display_title()))
            .bind(("summary", summary))
            .bind(("source_uri", source_uri))
            .bind(("account_uuid", account_uuid))
            .bind(("metadata", metadata))
            .bind(("summary_embedding", summary_embedding))
            .bind(("created_at", conversation.created_at.to_rfc3339()))
            .bind(("updated_at", conversation.updated_at.to_rfc3339()))
            .await?
            .take(0)?;

        if let Some(row) = upserted {
            return Ok(row.id);
        }

        let fetched: Option<ConversationIdRow> = self
            .db
            .query("SELECT id FROM conversation WHERE uuid = $uuid LIMIT 1")
            .bind(("uuid", conversation.uuid.clone()))
            .await?
            .take(0)?;

        fetched
            .map(|row| row.id)
            .ok_or_else(|| DbError::CreateFailed("conversation".into()))
    }

    /// Upsert a message record from a chat export message.
    #[instrument(skip(self, message))]
    pub async fn upsert_message(
        &self,
        conversation_id: &RecordId,
        conversation_uuid: &str,
        index: usize,
        message: &ChatMessage,
        embedding: Option<Vec<f32>>,
    ) -> Result<RecordId> {
        #[derive(Debug, Deserialize, SurrealValue)]
        struct MessageIdRow {
            id: RecordId,
        }

        let message_uuid = message.uuid.clone();
        let message_key = message_uuid
            .clone()
            .unwrap_or_else(|| format!("{}:{}", conversation_uuid, index));

        let role = serde_json::to_string(&message.role)
            .unwrap_or_else(|_| "\"system\"".to_string())
            .trim_matches('"')
            .to_string();

        let content_blocks = message
            .content_blocks
            .as_array()
            .cloned()
            .unwrap_or_default();
        let attachments = message.attachments.clone();
        let files = message.files.clone();

        let upserted: Option<MessageIdRow> = self
            .db
            .query(
                r#"
                INSERT INTO message (
                    message_key, message_uuid, conversation_id, conversation_uuid, message_index, role,
                    content, embedding, content_blocks, attachments, files, has_files, created_at, updated_at, ingested_at
                )
                VALUES (
                    $message_key, $message_uuid, $conversation_id, $conversation_uuid, $message_index, $role,
                    $content, $embedding, $content_blocks, $attachments, $files, $has_files,
                    IF $created_at = NONE THEN NONE ELSE <datetime>$created_at END,
                    IF $updated_at = NONE THEN NONE ELSE <datetime>$updated_at END,
                    time::now()
                )
                ON DUPLICATE KEY UPDATE
                    message_uuid = $message_uuid,
                    conversation_id = $conversation_id,
                    conversation_uuid = $conversation_uuid,
                    message_index = $message_index,
                    role = $role,
                    content = $content,
                    embedding = $embedding,
                    content_blocks = $content_blocks,
                    attachments = $attachments,
                    files = $files,
                    has_files = $has_files,
                    created_at = IF $created_at = NONE THEN NONE ELSE <datetime>$created_at END,
                    updated_at = IF $updated_at = NONE THEN NONE ELSE <datetime>$updated_at END,
                    ingested_at = time::now()
            "#,
            )
            .bind(("message_key", message_key.clone()))
            .bind(("message_uuid", message_uuid))
            .bind(("conversation_id", conversation_id.clone()))
            .bind(("conversation_uuid", conversation_uuid.to_string()))
            .bind(("message_index", index as i64))
            .bind(("role", role))
            .bind(("content", message.content.clone()))
            .bind(("embedding", embedding))
            .bind(("content_blocks", content_blocks))
            .bind(("attachments", attachments))
            .bind(("files", files.clone()))
            .bind(("has_files", !files.is_empty()))
            .bind((
                "created_at",
                message.created_at.as_ref().map(|dt| dt.to_rfc3339()),
            ))
            .bind((
                "updated_at",
                message.updated_at.as_ref().map(|dt| dt.to_rfc3339()),
            ))
            .await?
            .take(0)?;

        if let Some(row) = upserted {
            return Ok(row.id);
        }

        let fetched: Option<MessageIdRow> = self
            .db
            .query("SELECT id FROM message WHERE message_key = $message_key LIMIT 1")
            .bind(("message_key", message_key))
            .await?
            .take(0)?;

        fetched
            .map(|row| row.id)
            .ok_or_else(|| DbError::CreateFailed("message".into()))
    }

    /// Link note provenance to a conversation.
    #[instrument(skip(self))]
    pub async fn link_note_to_conversation(
        &self,
        note_id: &RecordId,
        conversation_id: &RecordId,
    ) -> Result<bool> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_conversation_locked(note_id, conversation_id)
            .await
    }

    async fn link_note_to_conversation_locked(
        &self,
        note_id: &RecordId,
        conversation_id: &RecordId,
    ) -> Result<bool> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a conversation-provenance endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_conversation WHERE in = $note_id AND out = $conversation_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("conversation_id", conversation_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count > 0 {
            return Ok(false);
        }

        self.db
            .query("CREATE note_from_conversation SET in = $note_id, out = $conversation_id")
            .bind(("note_id", note_id.clone()))
            .bind(("conversation_id", conversation_id.clone()))
            .await?;

        Ok(true)
    }

    /// Link note provenance to a message.
    #[instrument(skip(self))]
    pub async fn link_note_to_message(
        &self,
        note_id: &RecordId,
        message_id: &RecordId,
    ) -> Result<bool> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_message_locked(note_id, message_id).await
    }

    async fn link_note_to_message_locked(
        &self,
        note_id: &RecordId,
        message_id: &RecordId,
    ) -> Result<bool> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a message-provenance endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_message WHERE in = $note_id AND out = $message_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("message_id", message_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count > 0 {
            return Ok(false);
        }

        self.db
            .query("CREATE note_from_message SET in = $note_id, out = $message_id")
            .bind(("note_id", note_id.clone()))
            .bind(("message_id", message_id.clone()))
            .await?;

        Ok(true)
    }

    /// Check whether a conversation already has any linked notes.
    #[instrument(skip(self))]
    pub async fn conversation_has_note_links(&self, conversation_id: &RecordId) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_conversation WHERE out = $conversation_id GROUP ALL",
            )
            .bind(("conversation_id", conversation_id.clone()))
            .await?
            .take(0)?;

        Ok(existing.and_then(|row| row.count).unwrap_or(0) > 0)
    }

    // ==========================================
    // STATS
    // ==========================================

    /// Get database statistics
    #[instrument(skip(self))]
    pub async fn get_stats(&self) -> Result<DbStats> {
        let stats: Vec<DbStats> = self
            .db
            .query(
                r#"
                RETURN {
                    note_count: (SELECT count() FROM note GROUP ALL)[0].count,
                    entity_count: (SELECT count() FROM entity GROUP ALL)[0].count,
                    source_count: (SELECT count() FROM source GROUP ALL)[0].count,
                    conversation_count: (SELECT count() FROM conversation GROUP ALL)[0].count,
                    message_count: (SELECT count() FROM message GROUP ALL)[0].count,
                    mention_count: (SELECT count() FROM mentions GROUP ALL)[0].count,
                    note_conversation_link_count: (SELECT count() FROM note_from_conversation GROUP ALL)[0].count,
                    note_message_link_count: (SELECT count() FROM note_from_message GROUP ALL)[0].count,
                    edge_count: (
                        (SELECT count() FROM supports GROUP ALL)[0].count +
                        (SELECT count() FROM contradicts GROUP ALL)[0].count +
                        (SELECT count() FROM related_to GROUP ALL)[0].count
                    )
                }
            "#,
            )
            .await?
            .take(0)?;

        stats
            .into_iter()
            .next()
            .ok_or_else(|| DbError::QueryFailed("stats".into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct NoteEdgeRow {
    pub id: RecordId,
    pub edge_type: String,
    pub in_id: RecordId,
    pub out_id: RecordId,
    #[serde(default)]
    pub proposal_id: Option<RecordId>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub provenance: Option<String>,
    #[serde(default)]
    pub is_manual: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Local lexical entity match used as a graph-retrieval seed. `metadata` is
/// retained only to make alias evidence inspectable by higher layers.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct GraphEntityMatch {
    pub id: RecordId,
    pub name: String,
    pub canonical_name: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A visible note directly mentioned by one query-matched entity. Retaining
/// both IDs prevents query-wide entity labels from being attached to unrelated
/// seeds when several matched entities are present.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct GraphEntityNoteSeed {
    pub note_id: RecordId,
    pub entity_id: RecordId,
}

fn normalize_note_id(note_id: &str) -> RecordId {
    RecordId::new("note", note_id.strip_prefix("note:").unwrap_or(note_id))
}

/// Data required to persist an edge proposal. The repository canonicalizes
/// symmetric endpoint order before deriving the proposal's stable key.
#[derive(Debug, Clone)]
pub struct EdgeProposalDraft {
    pub from_id: RecordId,
    pub to_id: RecordId,
    pub edge_type: EdgeType,
    pub confidence: f32,
    pub reason: String,
    pub generator: String,
    pub generator_version: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct ProposedEdgeRow {
    id: Option<RecordId>,
    dedupe_key: String,
    from_id: RecordId,
    to_id: RecordId,
    edge_type: String,
    confidence: f32,
    reason: String,
    generator: String,
    #[serde(default)]
    generator_version: Option<String>,
    #[serde(default)]
    model: Option<String>,
    status: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    reviewed_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    reviewer: Option<String>,
    #[serde(default)]
    action_reason: Option<String>,
    #[serde(default)]
    acceptance_is_manual: Option<bool>,
    #[serde(default)]
    resulting_edge_id: Option<RecordId>,
    #[serde(default)]
    superseded_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    supersession_reason: Option<String>,
}

impl ProposedEdgeRow {
    fn into_domain(self) -> Result<ProposedEdge> {
        let edge_type = match self.edge_type.as_str() {
            "supports" => EdgeType::Supports,
            "contradicts" => EdgeType::Contradicts,
            "derived_from" => EdgeType::DerivedFrom,
            "related_to" => EdgeType::RelatedTo,
            other => {
                return Err(DbError::QueryFailed(format!(
                    "unknown proposed edge type {other:?}"
                )))
            }
        };
        let status = match self.status.as_str() {
            "pending" => ProposedEdgeStatus::Pending,
            "accepting" => ProposedEdgeStatus::Accepting,
            "accepted" => ProposedEdgeStatus::Accepted,
            "rejected" => ProposedEdgeStatus::Rejected,
            "superseded" => ProposedEdgeStatus::Superseded,
            other => {
                return Err(DbError::QueryFailed(format!(
                    "unknown proposed edge status {other:?}"
                )))
            }
        };
        Ok(ProposedEdge {
            id: self.id,
            dedupe_key: self.dedupe_key,
            from_id: self.from_id,
            to_id: self.to_id,
            edge_type,
            confidence: self.confidence,
            reason: self.reason,
            generator: self.generator,
            generator_version: self.generator_version,
            model: self.model,
            status,
            created_at: self.created_at,
            updated_at: self.updated_at,
            reviewed_at: self.reviewed_at,
            reviewer: self.reviewer,
            action_reason: self.action_reason,
            acceptance_is_manual: self.acceptance_is_manual,
            resulting_edge_id: self.resulting_edge_id,
            superseded_at: self.superseded_at,
            supersession_reason: self.supersession_reason,
        })
    }
}

fn proposal_select_sql(where_clause: &str) -> String {
    format!(
        "SELECT id, dedupe_key, in AS from_id, out AS to_id, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at, reviewed_at, reviewer, action_reason, acceptance_is_manual, resulting_edge_id, superseded_at, supersession_reason FROM proposed_edge {where_clause}"
    )
}

fn note_edge_table(edge_type: &EdgeType) -> Result<&'static str> {
    match edge_type {
        EdgeType::Supports => Ok("supports"),
        EdgeType::Contradicts => Ok("contradicts"),
        EdgeType::DerivedFrom => Ok("derived_from"),
        EdgeType::RelatedTo => Ok("related_to"),
        EdgeType::References | EdgeType::Mentions | EdgeType::TaggedWith => {
            Err(DbError::QueryFailed(format!(
                "{edge_type} is not a persisted note-to-note edge type"
            )))
        }
    }
}

fn persisted_note_edge_type(value: &str) -> Result<EdgeType> {
    match value {
        "supports" => Ok(EdgeType::Supports),
        "contradicts" => Ok(EdgeType::Contradicts),
        "derived_from" => Ok(EdgeType::DerivedFrom),
        "related_to" => Ok(EdgeType::RelatedTo),
        other => Err(DbError::QueryFailed(format!(
            "unknown persisted note edge type {other:?}"
        ))),
    }
}

fn validate_note_edge(from_id: &RecordId, to_id: &RecordId, edge_type: &EdgeType) -> Result<()> {
    if !edge_type.is_note_edge() || matches!(edge_type, EdgeType::References) {
        return Err(DbError::QueryFailed(format!(
            "{edge_type} is not supported for persisted note edges"
        )));
    }
    if from_id.table.as_str() != "note" || to_id.table.as_str() != "note" {
        return Err(DbError::QueryFailed(
            "note edges require note record ids".into(),
        ));
    }
    if from_id == to_id {
        return Err(DbError::QueryFailed("self-edges are not allowed".into()));
    }
    Ok(())
}

fn canonicalize_note_edge(from_id: &mut RecordId, to_id: &mut RecordId, edge_type: &EdgeType) {
    if edge_type.is_symmetric() && record_id_to_string(from_id) > record_id_to_string(to_id) {
        std::mem::swap(from_id, to_id);
    }
}

fn edge_dedupe_key(from_id: &RecordId, to_id: &RecordId, edge_type: &EdgeType) -> String {
    format!(
        "{}:{}:{}",
        edge_type,
        record_id_to_string(from_id),
        record_id_to_string(to_id)
    )
}

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

impl Repository {
    async fn query_edges_table(&self, table: &str, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let query = format!(
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at \
             FROM {table} WHERE {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION} LIMIT $limit"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("limit", limit))
            .await?
            .take(0)?;
        Ok(edges)
    }

    async fn query_edges_for_note(
        &self,
        table: &str,
        note_id: &RecordId,
    ) -> Result<Vec<NoteEdgeRow>> {
        let query = format!(
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at \
             FROM {table} WHERE (in = $note_id OR out = $note_id) \
             AND {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION}"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        Ok(edges)
    }
}

// ==========================================
// RESULT TYPES
// ==========================================

/// Decision made before a file import starts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceImportAction {
    Created,
    Updated,
    Unchanged,
}

/// Source state returned by the staged file-import API.
#[derive(Debug, Clone)]
pub struct SourceImportPlan {
    pub source: Source,
    pub action: SourceImportAction,
    /// Deletions recovered before returning an unchanged import decision.
    /// Other actions defer cleanup until their import successfully promotes.
    pub cleanup: SourceDeleteSummary,
}

/// Deterministic cascade counts used by dry-run and completed import output.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceDeleteSummary {
    pub notes: u64,
    pub mentions: u64,
    pub note_edges: u64,
    /// Pending, accepting, or accepted proposals transitioned to `superseded`.
    pub proposals: u64,
    pub note_conversation_provenance: u64,
    pub note_message_provenance: u64,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct SourceDeleteCount {
    kind: String,
    #[serde(default)]
    count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SearchResult {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub note_type: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MessageSearchResult {
    pub id: RecordId,
    pub conversation_id: RecordId,
    pub conversation_uuid: String,
    pub message_index: i64,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct ConversationSearchResult {
    pub id: RecordId,
    pub uuid: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub source_uri: Option<String>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

impl FusionRecord for SearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

impl FusionRecord for MessageSearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

impl FusionRecord for ConversationSearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
pub struct RelatedNotes {
    #[serde(default)]
    pub supporting: Vec<Note>,
    #[serde(default)]
    pub supported_by: Vec<Note>,
    #[serde(default)]
    pub contradicting: Vec<Note>,
    #[serde(default)]
    pub contradicted_by: Vec<Note>,
    #[serde(default)]
    pub related: Vec<Note>,
    #[serde(default)]
    pub related_from: Vec<Note>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SimilarNote {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
pub struct DbStats {
    #[serde(default)]
    pub note_count: i64,
    #[serde(default)]
    pub entity_count: i64,
    #[serde(default)]
    pub source_count: i64,
    #[serde(default)]
    pub conversation_count: i64,
    #[serde(default)]
    pub message_count: i64,
    #[serde(default)]
    pub mention_count: i64,
    #[serde(default)]
    pub note_conversation_link_count: i64,
    #[serde(default)]
    pub note_message_link_count: i64,
    #[serde(default)]
    pub edge_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;
    use graphrag_core::{EntityType, SourceType};

    async fn begin_markdown(repo: &Repository, content: &str, force: bool) -> SourceImportPlan {
        repo.begin_file_import(
            SourceType::Markdown,
            "alpha.md".into(),
            "file:///notes/alpha.md".into(),
            content.into(),
            format!("sha256:{content}"),
            force,
        )
        .await
        .unwrap()
    }

    async fn create_provenance_records(repo: &Repository) -> (RecordId, RecordId) {
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }

        let conversation: Option<IdRow> = repo
            .db
            .query(
                "CREATE conversation SET uuid = $uuid, title = 'test conversation', summary = NONE, source_uri = NONE, account_uuid = NONE, metadata = {}, summary_embedding = NONE, created_at = time::now(), updated_at = time::now() RETURN AFTER",
            )
            .bind(("uuid", "test-conversation"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let conversation_id = conversation.unwrap().id;
        let message: Option<IdRow> = repo
            .db
            .query(
                "CREATE message SET message_key = $message_key, message_uuid = NONE, conversation_id = $conversation_id, conversation_uuid = $conversation_uuid, message_index = 0, role = 'human', content = 'test message', embedding = NONE, content_blocks = [], attachments = [], files = [], has_files = false, created_at = NONE, updated_at = NONE RETURN AFTER",
            )
            .bind(("message_key", "test-message"))
            .bind(("conversation_id", conversation_id.clone()))
            .bind(("conversation_uuid", "test-conversation"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        (conversation_id, message.unwrap().id)
    }

    #[tokio::test]
    async fn note_list_source_uri_filter_projects_linked_source_uri() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first_source = repo
            .create_source(Source::from_file("first.md", SourceType::Markdown).unwrap())
            .await
            .unwrap();
        let second_source = repo
            .create_source(Source::from_file("second.md", SourceType::Markdown).unwrap())
            .await
            .unwrap();
        let first = repo
            .create_note(Note::new("first").with_source(first_source.id.clone().unwrap()))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_source(second_source.id.clone().unwrap()))
            .await
            .unwrap();

        let first_uri = first_source.uri.as_deref().unwrap();
        let notes = repo
            .list_notes_filtered(10, &[], Some(first_uri))
            .await
            .unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, first.id.unwrap());
        assert_eq!(notes[0].source_uri.as_deref(), Some(first_uri));
    }

    #[test]
    fn graph_entity_query_normalization_keeps_unicode_terms_and_guards_prefixes() {
        assert_eq!(graph_query_normalize("Where is Atlas?"), "where is atlas");
        assert_eq!(graph_query_normalize("東京？"), "東京");
        assert_eq!(
            graph_prefix_terms("what changed in atla"),
            vec!["atla".to_string()]
        );
        assert!(graph_prefix_terms("ai").is_empty());
    }

    #[tokio::test]
    async fn graph_entity_lookup_normalizes_internal_punctuation_for_names_and_aliases() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut canonical = Entity::new("GPT-4", EntityType::Technology);
        canonical.metadata = serde_json::json!({});
        let canonical = repo.upsert_entity(canonical).await.unwrap();
        let mut aliased = Entity::new("Model reference", EntityType::Technology);
        aliased.metadata = serde_json::json!({"aliases": ["GPT-4"]});
        let aliased = repo.upsert_entity(aliased).await.unwrap();

        let matches = repo
            .find_graph_entities("Where is GPT-4?", 10)
            .await
            .unwrap();
        assert!(matches
            .iter()
            .any(|entity| entity.id == *canonical.id.as_ref().unwrap()));
        assert!(matches
            .iter()
            .any(|entity| entity.id == *aliased.id.as_ref().unwrap()));
    }

    #[tokio::test]
    async fn duplicate_entity_upsert_merges_aliases_without_losing_metadata() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut original = Entity::new("Atlas service", EntityType::Project);
        original.metadata = serde_json::json!({"aliases": ["Atlas"]});
        let original = repo.upsert_entity(original).await.unwrap();

        let mut update = Entity::new("Atlas Service", EntityType::Project);
        update.metadata = serde_json::json!({"aliases": ["Atlas", "Atlas v2"]});
        let updated = repo.upsert_entity(update).await.unwrap();

        assert_eq!(updated.id, original.id);
        assert_eq!(
            updated.metadata["aliases"],
            serde_json::json!(["Atlas", "Atlas v2"])
        );

        let original_matches = repo.find_graph_entities("Atlas", 1).await.unwrap();
        assert_eq!(original_matches.len(), 1);
        assert_eq!(original_matches[0].id, *original.id.as_ref().unwrap());
        let matches = repo.find_graph_entities("Atlas v2", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *original.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn graph_prefix_lookup_prefers_the_likely_entity_fragment_over_context_words() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut zeta = Entity::new("Zeta archive", EntityType::Project);
        zeta.metadata = serde_json::json!({});
        let zeta = repo.upsert_entity(zeta).await.unwrap();
        let mut deployed = Entity::new("Deployed controller", EntityType::Project);
        deployed.metadata = serde_json::json!({});
        repo.upsert_entity(deployed).await.unwrap();

        // Neither multi-word canonical name is contained in the sentence, so
        // this uses prefix recovery. The four-character entity fragment must
        // win before the eight-character context word at a one-entity cap.
        let matches = repo.find_graph_entities("zeta deployed", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *zeta.id.as_ref().unwrap());

        let mut atla = Entity::new("Atlas archive", EntityType::Project);
        atla.metadata = serde_json::json!({});
        repo.upsert_entity(atla).await.unwrap();
        let atla_matches = repo
            .find_graph_entities("atla deployment", 1)
            .await
            .unwrap();
        assert_eq!(atla_matches.len(), 1);
        assert_eq!(atla_matches[0].name, "Atlas archive");
    }

    #[tokio::test]
    async fn graph_entity_lookup_prioritizes_whole_query_names_and_aliases_over_phrases() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut short_name = Entity::new("New", EntityType::Project);
        short_name.metadata = serde_json::json!({});
        repo.upsert_entity(short_name).await.unwrap();
        let mut exact_name = Entity::new("New York", EntityType::Project);
        exact_name.metadata = serde_json::json!({});
        let exact_name = repo.upsert_entity(exact_name).await.unwrap();

        let matches = repo.find_graph_entities("New York", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_name.id.as_ref().unwrap());

        let matches = repo
            .find_graph_entities("status New York today", 1)
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_name.id.as_ref().unwrap());

        let mut short_alias = Entity::new("Big", EntityType::Project);
        short_alias.metadata = serde_json::json!({});
        repo.upsert_entity(short_alias).await.unwrap();
        let mut exact_alias = Entity::new("New York City", EntityType::Project);
        exact_alias.metadata = serde_json::json!({"aliases": ["Big Apple"]});
        let exact_alias = repo.upsert_entity(exact_alias).await.unwrap();

        let matches = repo.find_graph_entities("Big Apple", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_alias.id.as_ref().unwrap());

        let matches = repo
            .find_graph_entities("status Big Apple today", 1)
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_alias.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn graph_note_seed_query_preserves_ranked_entity_coverage_under_cap() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut atlas = Entity::new("Atlas", EntityType::Project);
        atlas.metadata = serde_json::json!({});
        let atlas = repo.upsert_entity(atlas).await.unwrap();
        let mut beacon = Entity::new("Beacon", EntityType::Project);
        beacon.metadata = serde_json::json!({});
        let beacon = repo.upsert_entity(beacon).await.unwrap();

        for id in ["atlas_a", "atlas_b", "beacon_a"] {
            repo.db
                .query(format!(
                    "CREATE note:{id} SET note_type = 'raw', content = $content, embedding = NONE, tags = [], created_at = time::now(), updated_at = time::now()"
                ))
                .bind(("content", id.to_string()))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for id in ["atlas_a", "atlas_b"] {
            repo.link_note_to_entity(&RecordId::new("note", id), atlas.id.as_ref().unwrap())
                .await
                .unwrap();
        }
        repo.link_note_to_entity(
            &RecordId::new("note", "beacon_a"),
            beacon.id.as_ref().unwrap(),
        )
        .await
        .unwrap();

        let seeds = repo
            .graph_notes_for_entities(
                &[
                    atlas.id.as_ref().unwrap().clone(),
                    beacon.id.as_ref().unwrap().clone(),
                ],
                1,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0].entity_id, *atlas.id.as_ref().unwrap());
        assert_eq!(seeds[1].entity_id, *beacon.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn source_lifecycle_is_idempotent_and_preserves_manual_notes() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        assert_eq!(first.action, SourceImportAction::Created);
        let source_id = first.source.id.as_ref().unwrap().clone();
        let derived = repo
            .create_note(
                Note::new("derived first")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual association").with_source(source_id.clone()))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let unchanged = begin_markdown(&repo, "first", false).await;
        assert_eq!(unchanged.action, SourceImportAction::Unchanged);
        assert_eq!(unchanged.source.generation, 1);

        let mut changed = begin_markdown(&repo, "second", false).await;
        assert_eq!(changed.action, SourceImportAction::Updated);
        assert_eq!(changed.source.generation, 2);
        let current = repo
            .create_note(
                Note::new("derived second")
                    .with_source(source_id.clone())
                    .with_source_generation(changed.source.generation),
            )
            .await
            .unwrap();
        let cleanup = repo
            .complete_file_import(&mut changed.source)
            .await
            .unwrap();
        assert_eq!(cleanup.notes, 1);
        assert!(repo
            .get_note(&record_id_to_string(derived.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(current.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
        assert!(repo
            .get_note(&record_id_to_string(manual.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());

        let mut failed = begin_markdown(&repo, "third", true).await;
        let partial = repo
            .create_note(
                Note::new("partial")
                    .with_source(source_id.clone())
                    .with_source_generation(failed.source.generation),
            )
            .await
            .unwrap();
        repo.fail_file_import(&mut failed.source, "embedding unavailable")
            .await
            .unwrap();
        let stored = repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.status, SourceIngestionStatus::Failed);
        assert_eq!(stored.successful_generation, 2);
        assert!(repo
            .get_note(&record_id_to_string(partial.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(current.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn retrieval_hides_unpromoted_source_generations_after_interruption() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        repo.create_note(
            Note::new("visible generation one")
                .with_embedding(vec![0.0; 1024])
                .with_source(source_id.clone())
                .with_source_generation(first.source.generation),
        )
        .await
        .unwrap();

        // The first process can be interrupted before promotion. Its staged
        // note must not be returned by either retrieval path after restart.
        assert!(repo
            .fulltext_search("visible generation", 10)
            .await
            .unwrap()
            .is_empty());
        assert!(repo
            .vector_search(vec![0.0; 1024], 10)
            .await
            .unwrap()
            .is_empty());

        repo.complete_file_import(&mut first.source).await.unwrap();
        assert_eq!(
            repo.fulltext_search("visible generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            repo.vector_search(vec![0.0; 1024], 10).await.unwrap().len(),
            1
        );

        let second = begin_markdown(&repo, "second", false).await;
        assert_eq!(second.source.generation, 2);
        repo.create_note(
            Note::new("pending generation two")
                .with_embedding(vec![0.0; 1024])
                .with_source(source_id)
                .with_source_generation(second.source.generation),
        )
        .await
        .unwrap();
        assert!(repo
            .fulltext_search("pending generation", 10)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            repo.fulltext_search("visible generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn related_notes_hide_staged_source_generations() {
        let repo = Repository::new(init_memory().await.unwrap());
        let anchor = repo.create_note(Note::new("manual anchor")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        let anchor_id = anchor.id.as_ref().unwrap();
        let staged_id = staged.id.as_ref().unwrap();
        // Cover both graph directions for every relationship projection.
        for edge_type in [
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::RelatedTo,
        ] {
            repo.create_edge(anchor_id, staged_id, edge_type.clone(), None)
                .await
                .unwrap();
            repo.create_edge(staged_id, anchor_id, edge_type, None)
                .await
                .unwrap();
        }

        let related = repo.get_related_notes(anchor_id).await.unwrap();
        assert!(related.supporting.is_empty());
        assert!(related.supported_by.is_empty());
        assert!(related.contradicting.is_empty());
        assert!(related.contradicted_by.is_empty());
        assert!(related.related.is_empty());
        assert!(related.related_from.is_empty());
    }

    #[tokio::test]
    async fn orphan_notes_ignore_hidden_source_generation_neighbors() {
        let repo = Repository::new(init_memory().await.unwrap());
        let anchor = repo.create_note(Note::new("manual anchor")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        let anchor_id = anchor.id.as_ref().unwrap();
        let staged_id = staged.id.as_ref().unwrap();
        // A staged source generation is invisible in every graph direction,
        // so it cannot prevent a visible manual note from being an orphan.
        for edge_type in [
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::RelatedTo,
        ] {
            repo.create_edge(anchor_id, staged_id, edge_type.clone(), None)
                .await
                .unwrap();
            repo.create_edge(staged_id, anchor_id, edge_type, None)
                .await
                .unwrap();
        }

        let orphans = repo.find_orphan_notes().await.unwrap();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].id.as_ref(), anchor.id.as_ref());
    }

    #[tokio::test]
    async fn note_edge_lists_hide_edges_with_hidden_source_generation_endpoints() {
        let repo = Repository::new(init_memory().await.unwrap());
        let manual_left = repo.create_note(Note::new("manual left")).await.unwrap();
        let manual_right = repo.create_note(Note::new("manual right")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        repo.create_edge(
            manual_left.id.as_ref().unwrap(),
            manual_right.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();
        repo.create_edge(
            manual_left.id.as_ref().unwrap(),
            staged.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();

        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
        assert_eq!(
            repo.get_note_edges(&record_id_to_string(manual_left.id.as_ref().unwrap()))
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(repo
            .get_note_edges(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn source_owned_notes_are_created_and_updated_with_ownership_intact() {
        let repo = Repository::new(init_memory().await.unwrap());
        let plan = begin_markdown(&repo, "content", false).await;
        let source_id = plan.source.id.as_ref().unwrap().clone();
        let created = repo
            .create_note(
                Note::new("owned content")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        // Creation writes both ownership fields in the single CREATE command;
        // there is no unowned persisted state to leak if import work stops.
        assert_eq!(created.source_id.as_ref(), Some(&source_id));
        assert_eq!(created.source_generation, Some(plan.source.generation));

        let mut edited = created.clone();
        edited.content = "edited content".into();
        // Callers that do not repeat source ownership must not accidentally
        // detach a source-owned note during a content update.
        edited.source_id = None;
        edited.source_generation = None;
        let updated = repo
            .update_note(&record_id_to_string(created.id.as_ref().unwrap()), edited)
            .await
            .unwrap();
        assert_eq!(updated.source_id.as_ref(), Some(&source_id));
        assert_eq!(updated.source_generation, Some(plan.source.generation));
    }

    #[tokio::test]
    async fn update_note_rebuilds_markdown_search_content_without_stale_terms() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut markdown_note = Note::new("obsolete-search-token is removed");
        markdown_note.chunk_heading_path = vec!["Roadmap".into()];
        markdown_note.search_content = Some("Roadmap\n\nobsolete-search-token is removed".into());
        let created = repo.create_note(markdown_note).await.unwrap();
        let created_id = created.id.as_ref().unwrap().clone();

        let mut edited = created.clone();
        edited.content = "current-search-token is indexed".into();
        // Model a caller changing only `content`; this stale field used to
        // keep removed body terms in the highest-weight FTS column.
        edited.search_content = created.search_content.clone();
        let updated = repo
            .update_note(&record_id_to_string(&created_id), edited)
            .await
            .unwrap();

        assert_eq!(
            updated.search_content.as_deref(),
            Some("Roadmap\n\ncurrent-search-token is indexed")
        );
        assert!(repo
            .fulltext_search("obsolete-search-token", 10)
            .await
            .unwrap()
            .iter()
            .all(|result| result.id != created_id));
        let current = repo
            .fulltext_search("current-search-token", 10)
            .await
            .unwrap();
        assert!(current
            .iter()
            .any(|result| { result.id == created_id && result.fts_score.is_some() }));
    }

    #[tokio::test]
    async fn update_note_preserves_custom_search_aliases_and_replacements() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut note = Note::new("original body text");
        note.search_content = Some("legacyalias".into());
        let created = repo.create_note(note).await.unwrap();
        let note_id = created.id.as_ref().unwrap().clone();

        // A metadata-only update must not discard an intentional alias just
        // because it does not equal the default body-derived search value.
        let mut metadata_only = created.clone();
        metadata_only.title = Some("Reference note".into());
        let metadata_updated = repo
            .update_note(&record_id_to_string(&note_id), metadata_only)
            .await
            .unwrap();
        assert_eq!(
            metadata_updated.search_content.as_deref(),
            Some("legacyalias")
        );
        assert_eq!(metadata_updated.title.as_deref(), Some("Reference note"));
        assert!(repo
            .fulltext_search("legacyalias", 10)
            .await
            .unwrap()
            .iter()
            .any(|result| result.id == note_id));

        // When callers intentionally replace search text alongside a content
        // edit, retain that replacement instead of overriding it with the
        // generic body-derived value.
        let mut content_edit = metadata_updated;
        content_edit.content = "replacement body text".into();
        content_edit.search_content = Some("replacementalias".into());
        let replaced = repo
            .update_note(&record_id_to_string(&note_id), content_edit)
            .await
            .unwrap();
        assert_eq!(replaced.search_content.as_deref(), Some("replacementalias"));
        assert!(repo
            .fulltext_search("legacyalias", 10)
            .await
            .unwrap()
            .iter()
            .all(|result| result.id != note_id));
        assert!(repo
            .fulltext_search("replacementalias", 10)
            .await
            .unwrap()
            .iter()
            .any(|result| result.id == note_id));
    }

    #[tokio::test]
    async fn atomic_note_update_replaces_mentions_with_the_new_content() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo.create_note(Note::new("old body")).await.unwrap();
        let note_id = record_id_to_string(note.id.as_ref().unwrap());
        let mut old_entity = Entity::new("Old Entity", EntityType::Concept);
        old_entity.metadata = serde_json::json!({});
        let old_entity = repo.upsert_entity(old_entity).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), old_entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let mut replacement = note.clone();
        replacement.content = "new body".into();
        let mut new_entity = Entity::new("New Entity", EntityType::Concept);
        new_entity.metadata = serde_json::json!({});
        let updated = repo
            .update_note_and_replace_entities(&note_id, replacement, vec![new_entity])
            .await
            .unwrap();

        assert_eq!(updated.content, "new body");
        let linked = repo.get_entities_for_note(&note_id).await.unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "New Entity");
    }

    #[tokio::test]
    async fn atomic_note_update_rolls_back_before_content_change_on_entity_error() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo.create_note(Note::new("old body")).await.unwrap();
        let note_id = record_id_to_string(note.id.as_ref().unwrap());
        let mut old_entity = Entity::new("Prior Entity", EntityType::Concept);
        old_entity.metadata = serde_json::json!({});
        let old_entity = repo.upsert_entity(old_entity).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), old_entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let mut replacement = note.clone();
        replacement.content = "new body must not persist".into();
        let mut malformed = Entity::new("Malformed Entity", EntityType::Concept);
        malformed.metadata = serde_json::json!("not an object");
        assert!(repo
            .update_note_and_replace_entities(&note_id, replacement, vec![malformed])
            .await
            .is_err());

        assert_eq!(
            repo.get_note(&note_id).await.unwrap().unwrap().content,
            "old body"
        );
        let linked = repo.get_entities_for_note(&note_id).await.unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Prior Entity");
    }

    #[tokio::test]
    async fn promotion_selects_the_new_generation_before_old_cleanup() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let first_note = repo
            .create_note(
                Note::new("first generation")
                    .with_embedding(vec![1.0; 1024])
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let old_generation_partner = repo
            .create_note(
                Note::new("first generation partner")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_generation_proposal = repo
            .upsert_gardener_proposal(
                first_note.id.as_ref().unwrap(),
                old_generation_partner.id.as_ref().unwrap(),
                0.9,
                "old generation appears related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let old_generation_proposal_id = old_generation_proposal.id.unwrap();
        let old_generation_accepted_partner = repo
            .create_note(
                Note::new("first generation accepted partner")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_generation_accepted_proposal = repo
            .upsert_gardener_proposal(
                first_note.id.as_ref().unwrap(),
                old_generation_accepted_partner.id.as_ref().unwrap(),
                0.9,
                "old generation accepted relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let old_generation_accepted_proposal_id = old_generation_accepted_proposal.id.unwrap();
        let old_generation_accepted_edge_id = repo
            .accept_edge_proposal(
                &old_generation_accepted_proposal_id,
                Some("reviewer".into()),
                Some("approved before reimport".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let second_note = repo
            .create_note(
                Note::new("second generation")
                    .with_embedding(vec![1.0; 1024])
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        // Simulate a process stopping after durable promotion but before the
        // best-effort destructive cleanup. The new complete generation is
        // immediately searchable; the old one is merely hidden/recoverable.
        repo.promote_file_import(&mut second.source).await.unwrap();
        assert_eq!(
            second.source.successful_generation,
            second.source.generation
        );
        assert_eq!(
            repo.get_edge_proposal(&old_generation_proposal_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
        assert_eq!(
            repo.fulltext_search("second generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(repo
            .fulltext_search("first generation", 10)
            .await
            .unwrap()
            .is_empty());
        // Other unfiltered scans must honor the same visibility rule while
        // cleanup is deferred after an interruption.
        assert_eq!(repo.list_notes(10).await.unwrap().len(), 1);
        assert_eq!(repo.find_orphan_notes().await.unwrap().len(), 1);
        assert_eq!(repo.get_notes_page(10, 0).await.unwrap().len(), 1);
        let second_key = record_id_to_string(second_note.id.as_ref().unwrap())
            .strip_prefix("note:")
            .unwrap()
            .to_string();
        assert!(repo
            .find_similar_notes(&second_key, vec![1.0; 1024], 0.0, 10)
            .await
            .unwrap()
            .is_empty());

        // The unchanged-hash path doubles as durable recovery: it retries
        // cleanup instead of leaving hidden old generations forever.
        let retry = begin_markdown(&repo, "second", false).await;
        assert_eq!(retry.action, SourceImportAction::Unchanged);
        assert_eq!(retry.cleanup.notes, 3);
        assert_eq!(retry.cleanup.proposals, 1);
        assert!(repo
            .get_note(&record_id_to_string(first_note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        let accepted_proposal = repo
            .get_edge_proposal(&old_generation_accepted_proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(accepted_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted_proposal.resulting_edge_id, None);
        assert!(!repo
            .note_edge_exists(&old_generation_accepted_edge_id)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn promotion_serializes_hidden_generation_retirement_with_acceptance() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "old generation race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();

        let second = begin_markdown(&repo, "second", false).await;
        // Queue promotion ahead of acceptance while the shared lock is held.
        // Tokio's mutex queues waiters fairly, so acceptance must observe the
        // proposal retirement performed at the visibility boundary.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let promotion_repo = repo.clone();
        let mut second_source = second.source;
        let promotion =
            tokio::spawn(
                async move { promotion_repo.promote_file_import(&mut second_source).await },
            );
        tokio::task::yield_now().await;
        let acceptance_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let acceptance = tokio::spawn(async move {
            acceptance_repo
                .accept_edge_proposal(&accepting_id, Some("reviewer".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        promotion.await.unwrap().unwrap();
        assert!(acceptance.await.unwrap().is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn completion_serializes_proposal_retarget_before_old_edge_undo() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "retarget race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let old_edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let new_left = repo
            .create_note(
                Note::new("second generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let new_right = repo
            .create_note(
                Note::new("second generation right")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.copy_note_dependents_to_successors(&[
            (
                old_left.id.as_ref().unwrap().clone(),
                new_left.id.as_ref().unwrap().clone(),
                true,
            ),
            (
                old_right.id.as_ref().unwrap().clone(),
                new_right.id.as_ref().unwrap().clone(),
                true,
            ),
        ])
        .await
        .unwrap();

        // Queue complete before undo while the lifecycle lock is held. The
        // completion transition must promote, retarget the accepted proposal
        // to the staged edge, and retire the old edge before undo may act.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let completion_repo = repo.clone();
        let completion = tokio::spawn(async move {
            completion_repo
                .complete_file_import(&mut second.source)
                .await
        });
        tokio::task::yield_now().await;
        let undo_repo = repo.clone();
        let undo_edge_id = old_edge_id.clone();
        let undo = tokio::spawn(async move {
            undo_repo
                .undo_edge(&undo_edge_id, Some("concurrent undo".into()))
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        completion.await.unwrap().unwrap();
        assert!(!undo.await.unwrap().unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Accepted);
        let replacement_edge_id = proposal.resulting_edge_id.unwrap();
        assert_ne!(replacement_edge_id, old_edge_id);
        assert!(repo.note_edge_exists(&replacement_edge_id).await.unwrap());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_graph_writes_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // Queue reconciliation before a manual graph write. The shared lock
        // makes the write observe the finished cleanup, where its old endpoint
        // is absent, instead of allowing it to succeed and be silently swept
        // away after the dependent snapshot.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let mutation_repo = repo.clone();
        let mutation_old_id = old.id.as_ref().unwrap().clone();
        let mutation_manual_id = manual.id.as_ref().unwrap().clone();
        let mutation = tokio::spawn(async move {
            mutation_repo
                .create_edge(
                    &mutation_old_id,
                    &mutation_manual_id,
                    EdgeType::Supports,
                    Some(0.9),
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(mutation.await.unwrap().is_err());
        assert!(repo
            .get_note(&record_id_to_string(old.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_mentions_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let mut entity = Entity::new("Concurrent Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // Queue reconciliation before the mention write. The write must see
        // that the old endpoint is gone after cleanup rather than succeeding
        // in the snapshot/cleanup window and being silently discarded.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let mention_repo = repo.clone();
        let old_note_id = old.id.as_ref().unwrap().clone();
        let entity_id = entity.id.as_ref().unwrap().clone();
        let mention = tokio::spawn(async move {
            mention_repo
                .link_note_to_entity(&old_note_id, &entity_id)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(mention.await.unwrap().is_err());
        assert!(repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn reconciliation_snapshots_a_complete_batched_entity_extraction() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let entities = ["First Extracted Entity", "Second Extracted Entity"]
            .into_iter()
            .map(|name| {
                let mut entity = Entity::new(name, EntityType::Concept);
                entity.metadata = serde_json::json!({});
                entity
            })
            .collect();

        // Queue the entire extraction result before reconciliation. The
        // shared lifecycle lock makes reconciliation snapshot both links,
        // rather than seeing an arbitrary prefix between per-entity writes.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let extraction_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let extraction = tokio::spawn(async move {
            extraction_repo
                .upsert_entities_and_link_note(&old_id, entities)
                .await
        });
        tokio::task::yield_now().await;
        let reconciliation_repo = repo.clone();
        let reconcile_old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(
                    &mut second.source,
                    &[(reconcile_old_id, replacement_id, true)],
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert_eq!(extraction.await.unwrap().unwrap(), 2);
        reconciliation.await.unwrap().unwrap();
        let entities = repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(entities.len(), 2);
        assert!(entities
            .iter()
            .any(|entity| entity.name == "First Extracted Entity"));
        assert!(entities
            .iter()
            .any(|entity| entity.name == "Second Extracted Entity"));
    }

    #[tokio::test]
    async fn failed_entity_link_batch_rolls_back_only_its_partial_mentions() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("entity batch rollback target"))
            .await
            .unwrap();
        let mut existing = Entity::new("Preexisting link survives", EntityType::Concept);
        existing.metadata = serde_json::json!({});
        let existing = repo.upsert_entity(existing).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), existing.id.as_ref().unwrap())
            .await
            .unwrap();
        let mut valid = Entity::new("Link must be rolled back", EntityType::Concept);
        valid.metadata = serde_json::json!({});
        let mut invalid = Entity::new("Entity write fails after first link", EntityType::Concept);
        // The entity schema requires an object or NONE. This reliably fails
        // the second upsert after the first link has been staged.
        invalid.metadata = serde_json::json!("not an object");

        assert!(repo
            .upsert_entities_and_link_note(note.id.as_ref().unwrap(), vec![valid, invalid])
            .await
            .is_err());
        let linked = repo
            .get_entities_for_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Preexisting link survives");
    }

    #[tokio::test]
    async fn failed_entity_replacement_preserves_the_prior_complete_mention_set() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("entity replacement rollback target"))
            .await
            .unwrap();
        let mut existing = Entity::new("Prior extraction survives", EntityType::Concept);
        existing.metadata = serde_json::json!({});
        let existing = repo.upsert_entity(existing).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), existing.id.as_ref().unwrap())
            .await
            .unwrap();
        let mut invalid = Entity::new("Malformed replacement", EntityType::Concept);
        invalid.metadata = serde_json::json!("not an object");

        assert!(repo
            .replace_note_entities(note.id.as_ref().unwrap(), vec![invalid])
            .await
            .is_err());
        let linked = repo
            .get_entities_for_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Prior extraction survives");
    }

    #[tokio::test]
    async fn reconciliation_snapshots_a_complete_replaced_entity_set() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let mut stale = Entity::new("Stale extraction", EntityType::Concept);
        stale.metadata = serde_json::json!({});
        let stale = repo.upsert_entity(stale).await.unwrap();
        repo.link_note_to_entity(old.id.as_ref().unwrap(), stale.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let entities = ["Fresh extraction one", "Fresh extraction two"]
            .into_iter()
            .map(|name| {
                let mut entity = Entity::new(name, EntityType::Concept);
                entity.metadata = serde_json::json!({});
                entity
            })
            .collect();

        // Inference has completed before this point. Queue the atomic mention
        // replacement ahead of reconciliation to prove the source transition
        // copies the full fresh set, never a transient cleared set.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let replacement_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let refresh = tokio::spawn(async move {
            replacement_repo
                .replace_note_entities(&old_id, entities)
                .await
        });
        tokio::task::yield_now().await;
        let reconciliation_repo = repo.clone();
        let reconcile_old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(
                    &mut second.source,
                    &[(reconcile_old_id, replacement_id, true)],
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert_eq!(refresh.await.unwrap().unwrap(), 2);
        reconciliation.await.unwrap().unwrap();
        let linked = repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 2);
        assert!(linked
            .iter()
            .any(|entity| entity.name == "Fresh extraction one"));
        assert!(linked
            .iter()
            .any(|entity| entity.name == "Fresh extraction two"));
    }

    #[tokio::test]
    async fn reconciliation_serializes_provenance_writes_and_copies_them_without_deadlock() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let (conversation_id, message_id) = create_provenance_records(&repo).await;
        repo.link_note_to_conversation(old.id.as_ref().unwrap(), &conversation_id)
            .await
            .unwrap();
        repo.link_note_to_message(old.id.as_ref().unwrap(), &message_id)
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // This also exercises the lock-free internal helpers used while the
        // reconciliation task already owns the lifecycle lock.
        repo.reconcile_file_import(
            &mut second.source,
            &[(
                old.id.as_ref().unwrap().clone(),
                replacement.id.as_ref().unwrap().clone(),
                true,
            )],
        )
        .await
        .unwrap();
        assert!(repo
            .conversation_has_note_links(&conversation_id)
            .await
            .unwrap());
        let copied_messages: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(copied_messages, vec![message_id]);

        // The now-hidden old endpoint must reject later provenance writes;
        // this prevents a link from being inserted after the copy snapshot
        // and then lost during cleanup.
        assert!(repo
            .link_note_to_conversation(old.id.as_ref().unwrap(), &conversation_id)
            .await
            .is_err());
        assert!(repo
            .link_note_to_message(old.id.as_ref().unwrap(), &copied_messages[0])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_provenance_writes_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let (conversation_id, message_id) = create_provenance_records(&repo).await;

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let conversation_repo = repo.clone();
        let conversation_note_id = old.id.as_ref().unwrap().clone();
        let conversation_link = tokio::spawn(async move {
            conversation_repo
                .link_note_to_conversation(&conversation_note_id, &conversation_id)
                .await
        });
        tokio::task::yield_now().await;
        let message_repo = repo.clone();
        let message_note_id = old.id.as_ref().unwrap().clone();
        let message_link = tokio::spawn(async move {
            message_repo
                .link_note_to_message(&message_note_id, &message_id)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(conversation_link.await.unwrap().is_err());
        assert!(message_link.await.unwrap().is_err());
        let copied_conversations: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_conversation WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let copied_messages: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(copied_conversations.is_empty());
        assert!(copied_messages.is_empty());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_mention_removals_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let mut entity = Entity::new("Copied Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(old.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let removal_repo = repo.clone();
        let old_note_id = old.id.as_ref().unwrap().clone();
        let removal =
            tokio::spawn(async move { removal_repo.delete_mentions_for_note(&old_note_id).await });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(removal.await.unwrap().is_err());
        assert_eq!(
            repo.get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn graph_writes_reject_hidden_generations_but_allow_current_pending() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("old generation")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let current_pending = repo
            .create_note(
                Note::new("current pending generation")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.create_edge(
            current_pending.id.as_ref().unwrap(),
            manual.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.9),
        )
        .await
        .unwrap();

        // Simulate a crash after durable promotion and before physical old
        // generation cleanup. The old note remains stored but is hidden.
        repo.promote_file_import(&mut second.source).await.unwrap();
        assert!(repo
            .create_edge(
                old.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                EdgeType::RelatedTo,
                Some(0.9),
            )
            .await
            .is_err());
        repo.create_edge(
            current_pending.id.as_ref().unwrap(),
            manual.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.9),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn recovery_drops_copied_edge_when_its_proposal_was_undone() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "undo before retarget".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let old_edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let new_left = repo
            .create_note(
                Note::new("second generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let new_right = repo
            .create_note(
                Note::new("second generation right")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.copy_note_dependents_to_successors(&[
            (
                old_left.id.as_ref().unwrap().clone(),
                new_left.id.as_ref().unwrap().clone(),
                true,
            ),
            (
                old_right.id.as_ref().unwrap().clone(),
                new_right.id.as_ref().unwrap().clone(),
                true,
            ),
        ])
        .await
        .unwrap();
        let copied_edges: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE id FROM related_to WHERE in = $note OR out = $note")
            .bind(("note", new_left.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let copied_edge_id = copied_edges.into_iter().next().unwrap();

        // Reconstruct a crash window from older callers that copied staged
        // dependents before completion: the original edge was undone before
        // the copied edge could retarget the proposal audit.
        assert!(repo
            .undo_edge(&old_edge_id, Some("undone before retarget".into()))
            .await
            .unwrap());
        repo.complete_file_import(&mut second.source).await.unwrap();

        assert!(!repo.note_edge_exists(&copied_edge_id).await.unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn post_promotion_failure_keeps_new_generation_for_recovery() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut import = begin_markdown(&repo, "durably promoted", false).await;
        let source_id = import.source.id.as_ref().unwrap().clone();
        let staged = repo
            .create_note(
                Note::new("new generation remains visible")
                    .with_source(source_id.clone())
                    .with_source_generation(import.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                staged.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                0.9,
                "force a post-promotion retarget failure".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let (edge_id, _) = repo
            .create_audited_edge(
                staged.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                EdgeType::RelatedTo,
                Some(0.9),
                Some("staged proposal-backed edge"),
                "test",
                Some(&proposal_id),
                false,
            )
            .await
            .unwrap();

        // The proposal is deliberately still pending, so retargeting fails
        // after `replace_source` durably promoted this generation.
        assert!(repo.complete_file_import(&mut import.source).await.is_err());
        let stored = repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.status, SourceIngestionStatus::Ready);
        assert_eq!(stored.successful_generation, stored.generation);
        assert!(repo
            .get_note(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());

        // Repair the injected inconsistency, then exercise the unchanged-hash
        // recovery path. It retries retargeting/cleanup without deleting the
        // already promoted current generation.
        repo.db
            .query(
                "UPDATE $proposal SET status = 'accepted', resulting_edge_id = $edge, updated_at = time::now()",
            )
            .bind(("proposal", proposal_id.clone()))
            .bind(("edge", edge_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let recovered = begin_markdown(&repo, "durably promoted", false).await;
        assert_eq!(recovered.action, SourceImportAction::Unchanged);
        assert!(repo.note_edge_exists(&edge_id).await.unwrap());
        assert!(repo
            .get_note(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn interrupted_promotion_cannot_resume_an_acceptance_for_hidden_notes() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "interrupted promotion race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        // Persist an acceptance claim, then simulate a different process
        // being interrupted after promotion updates source visibility but
        // before it can retire old-generation proposals.
        assert!(repo
            .claim_pending_proposal(
                &proposal_id,
                ProposedEdgeStatus::Accepting,
                Some("reviewer".into()),
                Some("approved before interruption".into()),
                Some(true),
            )
            .await
            .unwrap());
        let mut second = begin_markdown(&repo, "second", false).await;
        second.source.successful_generation = second.source.generation;
        second.source.status = SourceIngestionStatus::Ready;
        second.source.last_error = None;
        second.source.updated_at = chrono::Utc::now();
        second.source.last_ingested_at = Some(second.source.updated_at);
        repo.replace_source(&second.source).await.unwrap();

        // Resuming must treat now-hidden endpoints as stale, rather than
        // materializing an edge during the interruption window.
        assert!(repo
            .accept_edge_proposal(&proposal_id, Some("retry".into()), None, true)
            .await
            .is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("proposal endpoint is no longer visible")
        );
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());

        // The normal unchanged import recovery then deletes the hidden old
        // generation without finding a dangling accepted edge/proposal.
        let recovery = begin_markdown(&repo, "second", false).await;
        assert_eq!(recovery.action, SourceImportAction::Unchanged);
        assert_eq!(recovery.cleanup.notes, 2);
        assert!(repo
            .get_note(&record_id_to_string(old_left.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn source_delete_preview_matches_confirmed_derived_cascade() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut plan = begin_markdown(&repo, "content", false).await;
        let source_id = plan.source.id.as_ref().unwrap().clone();
        // Garden acceptance only operates on notes in the active source
        // generation, so promote this source before creating its fixture
        // relationships.
        repo.complete_file_import(&mut plan.source).await.unwrap();
        let derived = repo
            .create_note(
                Note::new("derived")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();
        let derived_second = repo
            .create_note(
                Note::new("derived second")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();
        let unrelated = repo.create_note(Note::new("manual")).await.unwrap();
        // This internal edge is reachable through two source-owned notes but
        // must count once in the exact dry-run/delete summary.
        repo.create_edge(
            derived.id.as_ref().unwrap(),
            derived_second.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.5),
        )
        .await
        .unwrap();
        repo.create_edge(
            derived.id.as_ref().unwrap(),
            unrelated.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.5),
        )
        .await
        .unwrap();
        let proposal = repo
            .upsert_gardener_proposal(
                derived.id.as_ref().unwrap(),
                unrelated.id.as_ref().unwrap(),
                0.9,
                "source-derived note looks related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let accepted_proposal = repo
            .upsert_gardener_proposal(
                derived_second.id.as_ref().unwrap(),
                unrelated.id.as_ref().unwrap(),
                0.9,
                "accepted source-derived note looks related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let accepted_proposal_id = accepted_proposal.id.unwrap();
        let accepted_edge_id = repo
            .accept_edge_proposal(
                &accepted_proposal_id,
                Some("reviewer".into()),
                Some("approved before source removal".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();
        let mut retained_entity = Entity::new("Retained entity", EntityType::Concept);
        retained_entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(retained_entity).await.unwrap();
        repo.link_note_to_entity(derived.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let preview = repo.preview_source_delete(&plan.source).await.unwrap();
        assert_eq!(preview.notes, 2);
        assert_eq!(preview.mentions, 1);
        assert_eq!(preview.note_edges, 3);
        assert_eq!(preview.proposals, 2);
        let confirmed = repo.delete_source(&plan.source).await.unwrap();
        assert_eq!(confirmed, preview);
        assert!(repo
            .get_note(&record_id_to_string(derived.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(unrelated.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
        assert!(repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .is_none());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("proposal endpoint removed by source lifecycle")
        );
        let accepted_proposal = repo
            .get_edge_proposal(&accepted_proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(accepted_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted_proposal.resulting_edge_id, None);
        assert!(!repo.note_edge_exists(&accepted_edge_id).await.unwrap());
        // Source cleanup must retire the pending proposal before policy batch
        // acceptance sees it, rather than failing on its missing endpoint.
        assert_eq!(
            repo.accept_gardener_proposals_above(0.8, Some("policy".into()))
                .await
                .unwrap(),
            0
        );
        let mut retained_entity = Entity::new("Retained entity", EntityType::Concept);
        retained_entity.metadata = serde_json::json!({});
        assert!(repo.upsert_entity(retained_entity).await.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_get_note() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let note = Note::new("Test content").with_title("Test Title");
        let created = repo.create_note(note).await.unwrap();

        assert!(created.id.is_some());
        assert_eq!(created.content, "Test content");
    }

    #[tokio::test]
    async fn test_list_notes() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        // Create a few notes
        for i in 0..3 {
            let note = Note::new(format!("Content {}", i));
            repo.create_note(note).await.unwrap();
        }

        let notes = repo.list_notes(10).await.unwrap();
        assert_eq!(notes.len(), 3);
    }

    async fn two_notes(repo: &Repository) -> (RecordId, RecordId) {
        let first = repo
            .create_note(Note::new("first"))
            .await
            .unwrap()
            .id
            .unwrap();
        let second = repo
            .create_note(Note::new("second"))
            .await
            .unwrap()
            .id
            .unwrap();
        (first, second)
    }

    #[tokio::test]
    async fn gardener_proposals_are_canonical_and_idempotent() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;

        let original = repo
            .upsert_gardener_proposal(
                &second,
                &first,
                0.81,
                "similar notes".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let updated = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.93,
                "newer similarity".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();

        assert_eq!(original.id, updated.id);
        assert_eq!(updated.confidence, 0.93);
        assert_eq!(updated.reason, "newer similarity");
        assert_eq!(updated.status, ProposedEdgeStatus::Pending);
        assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
        assert!(record_id_to_string(&updated.from_id) < record_id_to_string(&updated.to_id));
        let proposal_id = updated.id.as_ref().unwrap();
        assert_eq!(
            parse_record_id(&record_id_to_string(proposal_id), Some("proposed_edge")).unwrap(),
            *proposal_id
        );
    }

    #[tokio::test]
    async fn proposal_refresh_preserves_terminal_decision_audit() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.8,
                "original scan".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let rejected = repo
            .reject_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("not appropriate".into()),
            )
            .await
            .unwrap();
        let refreshed = repo
            .upsert_gardener_proposal(
                &second,
                &first,
                0.99,
                "later scan must not overwrite the decision".into(),
                Some("new-test".into()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(refreshed.id, Some(proposal_id));
        assert_eq!(refreshed.status, ProposedEdgeStatus::Rejected);
        assert_eq!(refreshed.reason, "original scan");
        assert_eq!(refreshed.reviewer, rejected.reviewer);
        assert_eq!(refreshed.action_reason, rejected.action_reason);
    }

    #[tokio::test]
    async fn concurrent_proposal_upserts_reload_the_unique_index_winner() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let left_repo = repo.clone();
        let right_repo = repo.clone();
        let left_from = first.clone();
        let left_to = second.clone();
        let right_from = second;
        let right_to = first;

        let (left, right) = tokio::join!(
            left_repo.upsert_gardener_proposal(
                &left_from,
                &left_to,
                0.81,
                "left scan".into(),
                Some("test".into()),
                None,
            ),
            right_repo.upsert_gardener_proposal(
                &right_from,
                &right_to,
                0.93,
                "right scan".into(),
                Some("test".into()),
                None,
            ),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.id, right.id);
        assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn interrupted_acceptance_claims_are_recoverable_by_retry_and_batch() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let retry = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "retry after interruption".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let retry_id = retry.id.unwrap();
        // Simulate a process stopping after its durable acceptance claim but
        // before edge creation/finalization.
        repo.db
            .query(
                "UPDATE $id SET status = 'accepting', reviewer = 'first reviewer', action_reason = 'first decision', acceptance_is_manual = true, reviewed_at = time::now()",
            )
            .bind(("id", retry_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let recovered = repo
            .accept_edge_proposal(
                &retry_id,
                Some("retry reviewer".into()),
                Some("retry decision".into()),
                false,
            )
            .await
            .unwrap();
        assert_eq!(recovered.status, ProposedEdgeStatus::Accepted);
        assert_eq!(recovered.action_reason.as_deref(), Some("first decision"));
        let retry_edge_id = recovered.resulting_edge_id.unwrap();
        assert!(repo.note_edge_exists(&retry_edge_id).await.unwrap());
        assert!(repo.list_note_edges(10).await.unwrap()[0].is_manual);

        let third = repo
            .create_note(Note::new("third"))
            .await
            .unwrap()
            .id
            .unwrap();
        let batch = repo
            .upsert_gardener_proposal(
                &first,
                &third,
                0.9,
                "batch recovery".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let batch_id = batch.id.unwrap();
        // Simulate the legacy poisoned shape produced before recovery support:
        // accepted without a resulting edge id. Batch completion must repair it.
        repo.db
            .query(
                "UPDATE $id SET status = 'accepted', reviewer = 'policy', action_reason = 'policy decision', acceptance_is_manual = false, resulting_edge_id = NONE, reviewed_at = time::now()",
            )
            .bind(("id", batch_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        assert_eq!(
            repo.accept_gardener_proposals_above(0.8, Some("policy retry".into()))
                .await
                .unwrap(),
            1
        );
        let batch = repo.get_edge_proposal(&batch_id).await.unwrap().unwrap();
        assert_eq!(batch.status, ProposedEdgeStatus::Accepted);
        assert!(batch.resulting_edge_id.is_some());
        assert_eq!(batch.action_reason.as_deref(), Some("policy decision"));
    }

    #[tokio::test]
    async fn concurrent_accepts_share_one_stable_edge_and_completion() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "same proposal".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let left_repo = repo.clone();
        let right_repo = repo.clone();
        let left_id = proposal_id.clone();
        let right_id = proposal_id.clone();

        let (left, right) = tokio::join!(
            left_repo.accept_edge_proposal(
                &left_id,
                Some("left reviewer".into()),
                Some("left acceptance".into()),
                true,
            ),
            right_repo.accept_edge_proposal(
                &right_id,
                Some("right reviewer".into()),
                Some("right acceptance".into()),
                false,
            ),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.status, ProposedEdgeStatus::Accepted);
        assert_eq!(left.resulting_edge_id, right.resulting_edge_id);
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Accepted);
        assert!(proposal.resulting_edge_id.is_some());
    }

    #[tokio::test]
    async fn acceptance_never_adopts_an_independent_manual_edge() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        repo.create_edge(&first, &second, EdgeType::RelatedTo, Some(0.7))
            .await
            .unwrap();
        let manual_edge = repo.list_note_edges(10).await.unwrap().pop().unwrap();
        assert_eq!(manual_edge.provenance.as_deref(), Some("manual_api"));

        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "would duplicate manual edge".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        assert!(repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("equivalent edge already materialized independently")
        );
        assert!(repo.note_edge_exists(&manual_edge.id).await.unwrap());
    }

    #[tokio::test]
    async fn endpoint_deletion_and_acceptance_leave_no_dangling_edge() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (removed, retained) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &removed,
                &retained,
                0.9,
                "race with deletion".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();

        // Queue both operations behind the shared lifecycle lock, then let
        // them race for it. Either ordering is valid, but the final graph may
        // not retain an edge to the removed endpoint.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let accepting_repo = repo.clone();
        let deleting_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let removed_id = record_id_to_string(&removed);
        let acceptance = async move {
            accepting_repo
                .accept_edge_proposal(&accepting_id, Some("reviewer".into()), None, true)
                .await
        };
        let deletion = async move { deleting_repo.delete_note(&removed_id).await };
        drop(guard);
        let (_acceptance, deletion) = tokio::join!(acceptance, deletion);
        deletion.unwrap();

        assert!(repo
            .get_note(&record_id_to_string(&removed))
            .await
            .unwrap()
            .is_none());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn proposal_accept_reject_and_undo_are_auditable_and_idempotent() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let accepted = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "semantic overlap".into(),
                Some("test".into()),
                Some("fixture".into()),
            )
            .await
            .unwrap();
        let accepted_id = accepted.id.unwrap();
        let accepted = repo
            .accept_edge_proposal(
                &accepted_id,
                Some("reviewer".into()),
                Some("looks related".into()),
                true,
            )
            .await
            .unwrap();
        assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
        let accepted_reviewed_at = accepted.reviewed_at;
        assert_eq!(accepted.action_reason.as_deref(), Some("looks related"));
        let edge_id = accepted.resulting_edge_id.clone().unwrap();
        assert_eq!(
            repo.accept_edge_proposal(&accepted_id, None, None, true)
                .await
                .unwrap()
                .resulting_edge_id,
            Some(edge_id.clone())
        );

        let edge = repo.list_note_edges(10).await.unwrap().pop().unwrap();
        assert_eq!(edge.id, edge_id);
        assert_eq!(edge.reason.as_deref(), Some("semantic overlap"));
        assert_eq!(edge.provenance.as_deref(), Some("gardener-similarity"));
        assert!(edge.is_manual);
        assert!(repo.note_edge_exists(&edge_id).await.unwrap());

        assert!(repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        assert!(!repo.note_edge_exists(&edge_id).await.unwrap());
        assert!(!repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        let undone_proposal = repo.get_edge_proposal(&accepted_id).await.unwrap().unwrap();
        assert_eq!(undone_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(undone_proposal.resulting_edge_id, None);
        assert_eq!(
            undone_proposal.action_reason.as_deref(),
            Some("looks related")
        );
        assert_eq!(undone_proposal.reviewed_at, accepted_reviewed_at);
        assert_eq!(
            undone_proposal.supersession_reason.as_deref(),
            Some("reversed")
        );
        assert!(undone_proposal.superseded_at.is_some());

        let rejected = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "same pair after undo stays terminal".into(),
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(rejected.status, ProposedEdgeStatus::Superseded);
    }

    #[tokio::test]
    async fn undo_repairs_a_proposal_after_a_prior_edge_only_delete() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "accepted then interrupted undo".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let edge_id = repo
            .accept_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("accepted".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();
        // Simulate an interruption after undo's physical deletion but before
        // it could supersede the proposal record.
        repo.db
            .query("DELETE $id")
            .bind(("id", edge_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();

        assert!(repo
            .undo_edge(&edge_id, Some("recovered undo".into()))
            .await
            .unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("recovered undo")
        );
        assert!(!repo.undo_edge(&edge_id, None).await.unwrap());
    }

    #[tokio::test]
    async fn undo_serializes_with_acceptance_completion() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal_id = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "undo and retry race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        // Queue undo before a concurrent acceptance retry. Without the shared
        // lock, the retry can read `accepted` before undo retires its audit
        // record and race finalization against edge deletion.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let undo_repo = repo.clone();
        let undo_id = edge_id.clone();
        let undo = tokio::spawn(async move { undo_repo.undo_edge(&undo_id, None).await });
        tokio::task::yield_now().await;
        let acceptance_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let acceptance = tokio::spawn(async move {
            acceptance_repo
                .accept_edge_proposal(&accepting_id, Some("retry".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert!(undo.await.unwrap().unwrap());
        assert!(acceptance.await.unwrap().is_err());
        assert!(!repo.note_edge_exists(&edge_id).await.unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn independently_constructed_repositories_share_lifecycle_serialization() {
        let db = init_memory().await.unwrap();
        let accepting_repo = Repository::new(db.clone());
        let undoing_repo = Repository::new(db);
        assert!(Arc::ptr_eq(
            &accepting_repo.proposal_acceptance_lock,
            &undoing_repo.proposal_acceptance_lock
        ));
        let (first, second) = two_notes(&accepting_repo).await;
        let proposal_id = accepting_repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "independent repository race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let edge_id = accepting_repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        // Queue the competing operations through separate Repository values.
        // Undo wins the shared lifecycle lock, so a later acceptance retry
        // must see a superseded proposal rather than recreate/finalize it.
        let guard = accepting_repo.proposal_acceptance_lock.lock().await;
        let undo_repo = undoing_repo.clone();
        let undo_id = edge_id.clone();
        let undo = tokio::spawn(async move { undo_repo.undo_edge(&undo_id, None).await });
        tokio::task::yield_now().await;
        let retry_repo = accepting_repo.clone();
        let retry_id = proposal_id.clone();
        let retry = tokio::spawn(async move {
            retry_repo
                .accept_edge_proposal(&retry_id, Some("retry".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert!(undo.await.unwrap().unwrap());
        assert!(retry.await.unwrap().is_err());
        assert!(!undoing_repo.note_edge_exists(&edge_id).await.unwrap());
        let proposal = undoing_repo
            .get_edge_proposal(&proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn independent_memory_stores_do_not_share_lifecycle_serialization() {
        let first_repo = Repository::new(init_memory().await.unwrap());
        let second_repo = Repository::new(init_memory().await.unwrap());
        assert!(!Arc::ptr_eq(
            &first_repo.proposal_acceptance_lock,
            &second_repo.proposal_acceptance_lock
        ));

        // Holding a lifecycle transition for one logical store must not block
        // a repository backed by a separately initialized in-memory store.
        let first_guard = first_repo.proposal_acceptance_lock.lock().await;
        let second_guard = second_repo
            .proposal_acceptance_lock
            .try_lock()
            .expect("independent store lifecycle lock is not held");
        drop(second_guard);
        drop(first_guard);
    }

    #[tokio::test]
    async fn proposal_reject_is_idempotent_and_stale_endpoints_are_superseded() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(&first, &second, 0.8, "similar".into(), None, None)
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let rejected = repo
            .reject_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("not useful".into()),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status, ProposedEdgeStatus::Rejected);
        assert_eq!(
            repo.reject_edge_proposal(&proposal_id, None, None)
                .await
                .unwrap()
                .status,
            ProposedEdgeStatus::Rejected
        );

        let (third, fourth) = two_notes(&repo).await;
        let stale = repo
            .upsert_gardener_proposal(&third, &fourth, 0.8, "similar".into(), None, None)
            .await
            .unwrap();
        let stale_id = stale.id.unwrap();
        let _: Option<Note> = repo.db.delete(fourth).await.unwrap();
        assert!(repo
            .accept_edge_proposal(&stale_id, None, None, true)
            .await
            .is_err());
        assert_eq!(
            repo.get_edge_proposal(&stale_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
    }

    #[tokio::test]
    async fn delete_note_retires_proposals_and_their_accepted_edges() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (removed, pending_target) = two_notes(&repo).await;
        let accepted_target = repo
            .create_note(Note::new("accepted target"))
            .await
            .unwrap()
            .id
            .unwrap();
        let pending = repo
            .upsert_gardener_proposal(
                &removed,
                &pending_target,
                0.8,
                "pending relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let pending_id = pending.id.unwrap();
        let accepted = repo
            .upsert_gardener_proposal(
                &removed,
                &accepted_target,
                0.9,
                "accepted relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let accepted_id = accepted.id.unwrap();
        let accepted_edge_id = repo
            .accept_edge_proposal(
                &accepted_id,
                Some("reviewer".into()),
                Some("approved".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let removed_id = record_id_to_string(&removed);
        let preview = repo.preview_note_delete(&removed_id).await.unwrap();
        assert_eq!(preview.notes, 1);
        assert_eq!(preview.note_edges, 1);
        assert_eq!(preview.proposals, 2);
        let deleted = repo.delete_note_with_summary(&removed_id).await.unwrap();
        assert_eq!(deleted, preview);

        assert!(repo.get_note(&removed_id).await.unwrap().is_none());
        assert_eq!(
            repo.get_edge_proposal(&pending_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
        let accepted = repo.get_edge_proposal(&accepted_id).await.unwrap().unwrap();
        assert_eq!(accepted.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted.resulting_edge_id, None);
        assert!(!repo.note_edge_exists(&accepted_edge_id).await.unwrap());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn competing_accept_and_reject_claim_exactly_one_terminal_decision() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "similarly scoped notes".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();

        let (accept, reject) = tokio::join!(
            repo.accept_edge_proposal(
                &proposal_id,
                Some("acceptor".into()),
                Some("accept decision".into()),
                true,
            ),
            repo.reject_edge_proposal(
                &proposal_id,
                Some("rejector".into()),
                Some("reject decision".into()),
            ),
        );
        assert_ne!(accept.is_ok(), reject.is_ok());

        let final_proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        match final_proposal.status {
            ProposedEdgeStatus::Accepted => {
                assert!(final_proposal.resulting_edge_id.is_some());
                assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
            }
            ProposedEdgeStatus::Rejected => {
                assert!(final_proposal.resulting_edge_id.is_none());
                assert!(repo.list_note_edges(10).await.unwrap().is_empty());
            }
            status => panic!("unexpected terminal status: {status}"),
        }
    }

    #[tokio::test]
    async fn bulk_gardener_acceptance_pages_every_match_and_propagates_failures() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let third = repo
            .create_note(Note::new("third"))
            .await
            .unwrap()
            .id
            .unwrap();
        for (from, to) in [(&first, &second), (&first, &third), (&second, &third)] {
            repo.upsert_gardener_proposal(from, to, 0.9, "similar".into(), None, None)
                .await
                .unwrap();
        }
        assert_eq!(
            repo.accept_gardener_proposals_above_in_pages(
                0.8,
                Some("cli batch acceptance".into()),
                "reviewed as a related note".into(),
                true,
                1,
            )
            .await
            .unwrap(),
            3
        );
        assert_eq!(
            repo.list_edge_proposals(Some(ProposedEdgeStatus::Accepted), 10)
                .await
                .unwrap()
                .len(),
            3
        );
        let accepted = repo
            .list_edge_proposals(Some(ProposedEdgeStatus::Accepted), 10)
            .await
            .unwrap();
        assert!(accepted.iter().all(|proposal| {
            proposal.action_reason.as_deref() == Some("reviewed as a related note")
        }));
        assert!(repo
            .list_note_edges(10)
            .await
            .unwrap()
            .iter()
            .all(|edge| edge.is_manual));

        let failing_repo = Repository::new(init_memory().await.unwrap());
        let (from, stale_endpoint) = two_notes(&failing_repo).await;
        failing_repo
            .upsert_gardener_proposal(&from, &stale_endpoint, 0.9, "similar".into(), None, None)
            .await
            .unwrap();
        let _: Option<Note> = failing_repo.db.delete(stale_endpoint).await.unwrap();
        assert!(failing_repo
            .accept_gardener_proposals_above_in_pages(
                0.8,
                None,
                "automatic policy".into(),
                false,
                1,
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn self_edges_and_reverse_symmetric_duplicates_are_rejected() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        assert!(repo
            .upsert_gardener_proposal(&first, &first, 0.8, "self".into(), None, None)
            .await
            .is_err());

        repo.create_edge(&first, &second, EdgeType::RelatedTo, Some(0.8))
            .await
            .unwrap();
        repo.create_edge(&second, &first, EdgeType::RelatedTo, Some(0.8))
            .await
            .unwrap();
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);

        repo.create_edge(&first, &second, EdgeType::Supports, Some(0.8))
            .await
            .unwrap();
        repo.create_edge(&second, &first, EdgeType::Supports, Some(0.8))
            .await
            .unwrap();
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn similar_note_search_excludes_only_the_query_note() {
        let repo = Repository::new(init_memory().await.unwrap());
        let embedding = vec![1.0; 1024];
        let first = repo
            .create_note(Note::new("first").with_embedding(embedding.clone()))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_embedding(embedding.clone()))
            .await
            .unwrap();
        let similar = repo
            .find_similar_notes(
                &record_id_to_string(first.id.as_ref().unwrap()),
                embedding,
                0.7,
                5,
            )
            .await
            .unwrap();
        assert_eq!(similar.len(), 1);
    }

    #[tokio::test]
    async fn get_entities_for_note_uses_a_bound_record_id() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let note = repo
            .create_note(Note::new("Entity-linked note"))
            .await
            .unwrap();
        let note_id = note.id.unwrap();
        let mut entity = Entity::new("SurrealDB", EntityType::Technology);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        let entity_id = entity.id.unwrap();
        repo.link_note_to_entity(&note_id, &entity_id)
            .await
            .unwrap();

        let note_key = record_id_to_string(&note_id)
            .strip_prefix("note:")
            .unwrap()
            .to_string();

        for note_reference in [note_key.clone(), format!("note:{note_key}")] {
            let entities = repo.get_entities_for_note(&note_reference).await.unwrap();

            assert_eq!(entities.len(), 1);
            assert_eq!(entities[0].id.as_ref(), Some(&entity_id));
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_small_limit_keeps_relevant_note() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let rust_embedding = vec![1.0_f32; 1024];
        let distractor_embedding = vec![0.05_f32; 1024];

        let rust_note = Note::new("Rust is memory-safe and fast")
            .with_title("Rust note")
            .with_embedding(rust_embedding.clone());
        repo.create_note(rust_note).await.unwrap();

        for i in 0..60 {
            let note = Note::new(format!("Distractor content {}", i))
                .with_title(format!("Distractor {}", i))
                .with_embedding(distractor_embedding.clone());
            repo.create_note(note).await.unwrap();
        }

        let results = repo
            .hybrid_search_notes("Rust note", rust_embedding, 3, None, None)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title.as_deref(), Some("Rust note"));
    }

    #[tokio::test]
    async fn candidate_cutoffs_break_equal_component_scores_by_record_id() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        let embedding = vec![1.0_f32; 1024];
        for id in ["zulu", "alpha"] {
            let note = Note::new("identical deterministic candidate")
                .with_title("identical deterministic candidate")
                .with_embedding(embedding.clone());
            let _: Option<Note> = db.create(("note", id)).content(note).await.unwrap();
        }

        let vector = repo
            .vector_search_notes(embedding, 1, None, None)
            .await
            .unwrap();
        let fulltext = repo
            .fulltext_search_notes("identical deterministic candidate", 1, None, None)
            .await
            .unwrap();

        assert_eq!(record_id_to_string(&vector[0].id), "note:alpha");
        assert_eq!(record_id_to_string(&fulltext[0].id), "note:alpha");
    }

    #[tokio::test]
    async fn processing_job_checkpoint_cancel_and_resume_are_durable() {
        let repo = Repository::new(init_memory().await.unwrap());
        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::Embedding,
                Some("source:7/2".into()),
                3,
                Some("missing_embeddings".into()),
                vec!["note:one".into(), "note:two".into(), "note:three".into()],
            )
            .await
            .unwrap();
        let id = job.id.clone().unwrap();
        assert_eq!(job.scope.as_deref(), Some("missing_embeddings"));
        assert_eq!(job.item_ids.len(), 3);
        let updated = repo
            .update_processing_job(
                &id,
                ProcessingJobUpdate {
                    completed_count: Some(1),
                    checkpoint: Some(Some("note:one".into())),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.completed_count, 1);
        assert_eq!(updated.checkpoint.as_deref(), Some("note:one"));
        let cancelled = repo.cancel_processing_job(&id).await.unwrap();
        assert_eq!(cancelled.status, ProcessingJobStatus::Cancelled.as_str());
        let resumed = repo.resume_processing_job(&id).await.unwrap();
        assert_eq!(resumed.status, ProcessingJobStatus::Running.as_str());
        assert_eq!(repo.list_processing_jobs(10).await.unwrap().len(), 1);
    }
}
