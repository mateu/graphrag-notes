//! Repository pattern for database operations

use crate::{
    compatibility::{
        check_embedding_compatibility, record_embedding_metadata, CompatibilityState,
        EmbeddingIdentity, ExtractionIdentity,
    },
    fusion::{self, FusionConfig, FusionEvidence, FusionRecord},
    DbConnection, DbError, Result,
};
use graphrag_core::{
    record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, Note, Source,
    ProposedEdge, ProposedEdgeStatus, SourceIngestionStatus, SourceType,
};
use serde::{Deserialize, Serialize};
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;
use tracing::instrument;

/// Repository for all database operations
#[derive(Clone)]
pub struct Repository {
    db: DbConnection,
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
        Self { db }
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
        let _: Option<Note> = self.db.delete(("note", id)).await?;
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
                self.db
                    .query(
                        "UPDATE $id SET confidence = $confidence, reason = $reason, generator = $generator, generator_version = $generator_version, model = $model, updated_at = time::now()",
                    )
                    .bind(("id", existing.id.clone().expect("stored proposal has id")))
                    .bind(("confidence", draft.confidence))
                    .bind(("reason", draft.reason))
                    .bind(("generator", draft.generator))
                    .bind(("generator_version", draft.generator_version))
                    .bind(("model", draft.model))
                    .await?
                    .check()?;
                return self
                    .get_edge_proposal(&existing.id.expect("stored proposal has id"))
                    .await?
                    .ok_or_else(|| DbError::QueryFailed("updated proposal disappeared".into()));
            }
            return Ok(existing);
        }

        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }
        let created: Vec<IdRow> = self
            .db
            .query(
                "INSERT INTO proposed_edge (dedupe_key, in, out, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at) VALUES ($dedupe_key, $from, $to, $edge_type, $confidence, $reason, $generator, $generator_version, $model, 'pending', time::now(), time::now()) RETURN id",
            )
            .bind(("dedupe_key", dedupe_key))
            .bind(("from", draft.from_id))
            .bind(("to", draft.to_id))
            .bind(("edge_type", draft.edge_type.to_string()))
            .bind(("confidence", draft.confidence))
            .bind(("reason", draft.reason))
            .bind(("generator", draft.generator))
            .bind(("generator_version", draft.generator_version))
            .bind(("model", draft.model))
            .await?
            .take(0)?;
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
        let proposal = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        if proposal.status == ProposedEdgeStatus::Accepted {
            return Ok(proposal);
        }
        if proposal.status != ProposedEdgeStatus::Pending {
            return Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                proposal.status
            )));
        }
        if !self.note_exists(&proposal.from_id).await? || !self.note_exists(&proposal.to_id).await?
        {
            self.mark_proposal_stale(id, "proposal endpoint no longer exists")
                .await?;
            return Err(DbError::QueryFailed(format!(
                "proposal {} is stale: an endpoint no longer exists",
                record_id_to_string(id)
            )));
        }
        let edge_id = self
            .create_audited_edge(
                &proposal.from_id,
                &proposal.to_id,
                proposal.edge_type.clone(),
                Some(proposal.confidence),
                Some(&proposal.reason),
                &proposal.generator,
                Some(id),
                is_manual,
            )
            .await?;
        self.db.query("UPDATE $id SET status = 'accepted', reviewed_at = time::now(), reviewer = $reviewer, action_reason = $action_reason, resulting_edge_id = $edge_id, updated_at = time::now()")
            .bind(("id", id.clone())).bind(("reviewer", reviewer)).bind(("action_reason", action_reason)).bind(("edge_id", edge_id)).await?.check()?;
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
        self.db.query("UPDATE $id SET status = 'rejected', reviewed_at = time::now(), reviewer = $reviewer, action_reason = $action_reason, updated_at = time::now()")
            .bind(("id", id.clone())).bind(("reviewer", reviewer)).bind(("action_reason", action_reason)).await?.check()?;
        self.get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("rejected proposal disappeared".into()))
    }

    /// Accept all pending similarity proposals at or above a configured threshold.
    /// This is intentionally restricted to canonical `related_to` proposals.
    #[instrument(skip(self))]
    pub async fn accept_gardener_proposals_above(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
    ) -> Result<usize> {
        let proposals = self
            .list_edge_proposals(Some(ProposedEdgeStatus::Pending), 10_000)
            .await?;
        let mut accepted = 0;
        for proposal in proposals.into_iter().filter(|proposal| {
            proposal.edge_type == EdgeType::RelatedTo
                && proposal.generator == "gardener-similarity"
                && proposal.confidence >= min_confidence
        }) {
            let id = proposal.id.expect("stored proposal has id");
            if self
                .accept_edge_proposal(
                    &id,
                    reviewer.clone(),
                    Some("configured gardener auto-apply policy".into()),
                    false,
                )
                .await
                .is_ok()
            {
                accepted += 1;
            }
        }
        Ok(accepted)
    }

    /// Delete an accepted edge and mark its source proposal superseded. This is
    /// idempotent for a proposal that was already undone.
    #[instrument(skip(self))]
    pub async fn undo_edge(
        &self,
        edge_id: &RecordId,
        action_reason: Option<String>,
    ) -> Result<bool> {
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
        if existing.is_none() {
            return Ok(false);
        }
        self.db
            .query("DELETE $id")
            .bind(("id", edge_id.clone()))
            .await?
            .check()?;
        self.db.query("UPDATE proposed_edge SET status = 'superseded', reviewed_at = time::now(), action_reason = $reason, updated_at = time::now() WHERE resulting_edge_id = $id")
            .bind(("id", edge_id.clone())).bind(("reason", action_reason.unwrap_or_else(|| "accepted edge undone".into()))).await?.check()?;
        Ok(true)
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

    async fn note_exists(&self, id: &RecordId) -> Result<bool> {
        let existing: Option<Note> = self.db.select(id.clone()).await?;
        Ok(existing.is_some())
    }

    async fn mark_proposal_stale(&self, id: &RecordId, reason: &str) -> Result<()> {
        self.db.query("UPDATE $id SET status = 'superseded', reviewed_at = time::now(), action_reason = $reason, updated_at = time::now()")
            .bind(("id", id.clone())).bind(("reason", reason.to_string())).await?.check()?;
        Ok(())
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
    ) -> Result<RecordId> {
        validate_note_edge(from_id, to_id, &edge_type)?;
        let mut from_id = from_id.clone();
        let mut to_id = to_id.clone();
        canonicalize_note_edge(&mut from_id, &mut to_id, &edge_type);
        let table = note_edge_table(&edge_type)?;
        let dedupe_key = edge_dedupe_key(&from_id, &to_id, &edge_type);
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }
        let existing: Option<IdRow> = self
            .db
            .query(format!(
                "SELECT id FROM {table} WHERE dedupe_key = $dedupe_key LIMIT 1"
            ))
            .bind(("dedupe_key", dedupe_key.clone()))
            .await?
            .take(0)?;
        if let Some(existing) = existing {
            return Ok(existing.id);
        }
        let created: Vec<IdRow> = self.db.query(format!("INSERT INTO {table} (in, out, confidence, reason, provenance, proposal_id, is_manual, dedupe_key, created_at) VALUES ($from, $to, $confidence, $reason, $provenance, $proposal_id, $is_manual, $dedupe_key, time::now()) RETURN id"))
            .bind(("from", from_id)).bind(("to", to_id)).bind(("confidence", confidence.map(|value| value.clamp(0.0, 1.0))))
            .bind(("reason", reason.map(str::to_owned))).bind(("provenance", provenance.to_string())).bind(("proposal_id", proposal_id.cloned()))
            .bind(("is_manual", is_manual)).bind(("dedupe_key", dedupe_key)).await?.take(0)?;
        created
            .into_iter()
            .next()
            .map(|row| row.id)
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
        source.successful_generation = source.generation;
        source.status = SourceIngestionStatus::Ready;
        source.last_error = None;
        source.updated_at = chrono::Utc::now();
        source.last_ingested_at = Some(source.updated_at);
        self.replace_source(source).await
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
        let summary = self
            .source_delete_summary(source_id, generation, older_than_generation)
            .await?;
        let notes = self
            .source_owned_note_ids(source_id, generation, older_than_generation)
            .await?;
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
                .bind(("note", note_id))
                .await?
                .check()?;
        }
        Ok(summary)
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
    resulting_edge_id: Option<RecordId>,
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
            resulting_edge_id: self.resulting_edge_id,
        })
    }
}

fn proposal_select_sql(where_clause: &str) -> String {
    format!(
        "SELECT id, dedupe_key, in AS from_id, out AS to_id, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at, reviewed_at, reviewer, action_reason, resulting_edge_id FROM proposed_edge {where_clause}"
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
        assert_eq!(retry.cleanup.notes, 1);
        assert!(repo
            .get_note(&record_id_to_string(first_note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn source_delete_preview_matches_confirmed_derived_cascade() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut plan = begin_markdown(&repo, "content", false).await;
        let source_id = plan.source.id.as_ref().unwrap().clone();
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
        let mut retained_entity = Entity::new("Retained entity", EntityType::Concept);
        retained_entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(retained_entity).await.unwrap();
        repo.link_note_to_entity(derived.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.complete_file_import(&mut plan.source).await.unwrap();

        let preview = repo.preview_source_delete(&plan.source).await.unwrap();
        assert_eq!(preview.notes, 2);
        assert_eq!(preview.mentions, 1);
        assert_eq!(preview.note_edges, 2);
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

        assert!(repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        assert!(!repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        assert_eq!(
            repo.get_edge_proposal(&accepted_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );

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
}
