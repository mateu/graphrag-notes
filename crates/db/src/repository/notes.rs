//! Note CRUD and note/message/conversation retrieval ownership.
//!
//! Query text, ordering, filters, and fusion calls are moved verbatim from
//! the façade. Source lifecycle and graph mutations remain in their domains.

use super::*;

impl Repository {
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

    /// Atomically create a manual note and its complete mention set. A
    /// detached copy is never visible without its extraction result; if a
    /// mention write fails, the note creation rolls back as well, so retrying
    /// cannot leave duplicate manual copies behind.
    #[instrument(skip(self, note, entities))]
    pub async fn create_note_and_replace_entities(
        &self,
        note: Note,
        entities: Vec<Entity>,
    ) -> Result<Note> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let entity_ids = self.replacement_entity_ids(entities).await?;
        let note_id = RecordId::new("note", Uuid::new_v4().to_string());
        let mut response = self
            .db
            .query(
                "BEGIN TRANSACTION; \
                 CREATE $id SET \
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
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at; \
                 FOR $entity_id IN $entity_ids { CREATE mentions SET in = $id, out = $entity_id; }; \
                 COMMIT TRANSACTION;",
            )
            .bind(("id", note_id.clone()))
            .bind(("note_type", serde_json::to_value(&note.note_type).map_err(|error| DbError::QueryFailed(error.to_string()))?))
            .bind(("title", note.title.clone()))
            .bind(("content", note.content.clone()))
            .bind(("embedding", (!note.embedding.is_empty()).then_some(note.embedding.clone())))
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
            .bind(("search_content", note.search_content.clone()))
            .bind(("tags", note.tags.clone()))
            .bind(("created_at", note.created_at.to_rfc3339()))
            .bind(("updated_at", note.updated_at.to_rfc3339()))
            .bind(("entity_ids", entity_ids))
            .await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            return Err(DbError::QueryFailed(format!(
                "atomic note-and-mention create failed: {}",
                errors
                    .into_iter()
                    .map(|(statement, error)| format!("statement {statement}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
        self.get_note(&record_id_to_string(&note_id))
            .await?
            .ok_or_else(|| DbError::CreateFailed("atomic note-and-mention create".into()))
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
}
