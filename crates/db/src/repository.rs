//! Repository pattern for database operations

use crate::{
    compatibility::{
        check_embedding_compatibility, record_embedding_metadata, CompatibilityState,
        EmbeddingIdentity, ExtractionIdentity,
    },
    fusion::{self, FusionConfig, FusionEvidence, FusionRecord},
    DbConnection, DbError, Result,
};
use chrono::{DateTime, Utc};
use graphrag_core::{
    record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, Note, ProposedEdge,
    ProposedEdgeStatus, Source, SourceIngestionStatus, SourceType,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

fn count_to_i64(count: u64) -> Result<i64> {
    i64::try_from(count)
        .map_err(|_| DbError::QueryFailed("processing count exceeds database integer range".into()))
}

/// The coarse operation of a durable local processing job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingJobType {
    Embedding,
    EntityExtraction,
}

impl ProcessingJobType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::EntityExtraction => "entity_extraction",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "embedding" => Some(Self::Embedding),
            "entity_extraction" => Some(Self::EntityExtraction),
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

    /// Create a local, resumable unit of inference work. `source_generation`
    /// is an opaque durable identifier so jobs remain valid even when their
    /// source record is later reloaded by another process.
    pub async fn create_processing_job(
        &self,
        job_type: ProcessingJobType,
        source_generation: Option<String>,
        total_count: u64,
    ) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "CREATE processing_job SET job_type = $job_type, source_generation = $source_generation, \
                 status = 'running', total_count = $total_count, completed_count = 0, failed_count = 0, \
                 checkpoint = NONE, last_error = NONE, created_at = time::now(), updated_at = time::now(), \
                 finished_at = NONE RETURN AFTER",
            )
            .bind(("job_type", job_type.as_str()))
            .bind(("source_generation", source_generation))
            .bind(("total_count", count_to_i64(total_count)?))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::QueryFailed("create_processing_job".into()))
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
                    source_generation = $source_generation, tags = $tags, \
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

    /// Update a note
    #[instrument(skip(self, note))]
    pub async fn update_note(&self, id: &str, note: Note) -> Result<Note> {
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let updated: Option<Note> = self
            .db
            .query(
                "UPDATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, tags = $tags, \
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
            .bind(("created_at", note.created_at.to_rfc3339()))
            .bind(("updated_at", note.updated_at.to_rfc3339()))
            .await?
            .take(0)?;

        updated.ok_or_else(|| DbError::NotFound("note".into(), id.into()))
    }

    /// Delete a note
    #[instrument(skip(self))]
    pub async fn delete_note(&self, id: &str) -> Result<()> {
        // Serialize endpoint removal with proposal acceptance. Without this,
        // deletion could run after acceptance checks existence but before the
        // accepted edge write, leaving a dangling endpoint reference.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note_id = RecordId::new("note", raw_id);
        self.supersede_proposals_for_removed_notes(std::slice::from_ref(&note_id))
            .await?;
        self.delete_notes_and_dependents(std::slice::from_ref(&note_id))
            .await?;
        Ok(())
    }

    /// List recent notes (basic fields only, for CLI)
    #[instrument(skip(self))]
    pub async fn list_notes(&self, limit: usize) -> Result<Vec<SearchResult>> {
        let mut notes: Vec<SearchResult> = self
            .db
            .query(format!("SELECT * FROM note WHERE {VISIBLE_NOTE_CONDITION}"))
            .await?
            .take(0)?;

        // Sort by creation time descending and apply limit in Rust to avoid
        // SurrealDB multi-result `take` issues and deserialization problems
        // with full `Note` records.
        notes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
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
                    (search::score(0) * 0.7 + search::score(1) * 0.3) AS fts_score
                FROM note
                WHERE (content @0@ $query OR title @1@ $query)
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
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = $reason, resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting' AND resulting_edge_id IS NONE",
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
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'equivalent edge already materialized independently', resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting' AND resulting_edge_id IS NONE",
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
                    embedding = $embedding
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
        self.db
            .query("DELETE mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?;

        Ok(())
    }

    /// Get entities linked to a note
    #[instrument(skip(self))]
    pub async fn get_entities_for_note(&self, note_id: &str) -> Result<Vec<Entity>> {
        let raw = if note_id.starts_with("note:") {
            note_id["note:".len()..].to_string()
        } else {
            note_id.to_string()
        };
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

        let raw = if note_id.starts_with("note:") {
            note_id["note:".len()..].to_string()
        } else {
            note_id.to_string()
        };
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
                WHERE in = type::thing("note", $note_id)
                  AND out IN (
                    SELECT VALUE id
                    FROM entity
                    WHERE canonical_name CONTAINS $entity_query
                  )
                GROUP ALL
            "#,
            )
            .bind(("note_id", raw))
            .bind(("entity_query", normalized))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        Ok(count > 0)
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
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        let summary = self
            .source_delete_summary(&source_id, Some(source.generation), true)
            .await?;
        self.promote_file_import(source).await?;
        // Do this only after durable promotion. A failure here can leave old
        // records behind, but cannot leave the corpus with no visible complete
        // generation; visibility selects `successful_generation`.
        self.delete_source_notes(&source_id, Some(source.generation), true)
            .await?;
        Ok(summary)
    }

    async fn promote_file_import(&self, source: &mut Source) -> Result<()> {
        // Promotion changes which source generation is visible. Keep that
        // transition and retirement of proposals for the newly hidden notes
        // atomic with respect to proposal acceptance in this repository.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        source.successful_generation = source.generation;
        source.status = SourceIngestionStatus::Ready;
        source.last_error = None;
        source.updated_at = chrono::Utc::now();
        source.last_ingested_at = Some(source.updated_at);
        self.replace_source(source).await?;
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
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        self.delete_source_notes(source_id, Some(source.successful_generation), true)
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
                .bind(("note", note_id))
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
    pub confidence: Option<f32>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub provenance: Option<String>,
    #[serde(default)]
    pub is_manual: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
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
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, confidence, reason, provenance, is_manual, created_at \
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
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, confidence, reason, provenance, is_manual, created_at \
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

        repo.delete_note(&record_id_to_string(&removed))
            .await
            .unwrap();

        assert!(repo
            .get_note(&record_id_to_string(&removed))
            .await
            .unwrap()
            .is_none());
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
            .create_processing_job(ProcessingJobType::Embedding, Some("source:7/2".into()), 3)
            .await
            .unwrap();
        let id = job.id.clone().unwrap();
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
