//! Note CRUD and note/message/conversation retrieval ownership.
//!
//! Query text, ordering, filters, and fusion calls are moved verbatim from
//! the façade. Source lifecycle and graph mutations remain in their domains.

use super::*;

#[cfg(test)]
#[path = "notes_edit_guard_tests.rs"]
mod edit_guard_tests;

#[cfg(test)]
#[path = "title_ranking_tests.rs"]
mod title_ranking_tests;

const EDITOR_REVISION_CONFLICT: &str = "graphrag-note-editor-revision-conflict";

/// Compare the persisted editor-opening snapshot inside the same transaction
/// that changes note/entity/mention records. Timestamps alone cannot detect imports
/// or other writers which preserve `updated_at`; every persisted Note field
/// participates. Defaulted fields normalize older records to their domain
/// representation, while missing/hidden rows always fail the guard. Guarded
/// in-place updates additionally require manual ownership; provenance links
/// added during editor/provider work must not turn a manual edit into an
/// imported-note overwrite. Detach only checks the opening note snapshot.
pub(super) fn editor_snapshot_guard(require_manual: bool) -> String {
    let ownership = if require_manual {
        "AND source_generation IS NONE \
         AND array::len((SELECT VALUE id FROM note_from_conversation WHERE in = $editor_source LIMIT 1)) = 0 \
         AND array::len((SELECT VALUE id FROM note_from_message WHERE in = $editor_source LIMIT 1)) = 0 "
    } else {
        ""
    };
    format!(
        "LET $editor_matches = (SELECT VALUE id FROM note WHERE id = $editor_source \
         AND {VISIBLE_NOTE_CONDITION} \
         {ownership}\
         AND note_type = $editor_expected.note_type \
         AND title = $editor_expected.title AND content = $editor_expected.content \
         AND (embedding ?? []) = ($editor_expected.embedding ?? []) \
         AND source_id = $editor_expected.source_id \
         AND source_generation = $editor_expected.source_generation \
         AND chunk_key = $editor_expected.chunk_key \
         AND chunk_location_key = $editor_expected.chunk_location_key \
         AND extraction_scope = $editor_expected.extraction_scope \
         AND chunk_ordinal = $editor_expected.chunk_ordinal \
         AND (chunk_heading_path ?? []) = $editor_expected.chunk_heading_path \
         AND source_start_line = $editor_expected.source_start_line \
         AND source_end_line = $editor_expected.source_end_line \
         AND source_start_byte = $editor_expected.source_start_byte \
         AND source_end_byte = $editor_expected.source_end_byte \
         AND chunk_overlap_from = $editor_expected.chunk_overlap_from \
         AND chunk_overlap_chars = $editor_expected.chunk_overlap_chars \
         AND (split_fenced_code ?? false) = $editor_expected.split_fenced_code \
         AND content_hash = $editor_expected.content_hash \
         AND search_content = $editor_expected.search_content \
         AND (tags ?? []) = $editor_expected.tags \
         AND created_at = $editor_expected.created_at \
         AND updated_at = $editor_expected.updated_at LIMIT 1); \
         IF array::len($editor_matches) != 1 {{ THROW '{EDITOR_REVISION_CONFLICT}'; }}; "
    )
}

fn editor_source_id(expected: &Note) -> Result<RecordId> {
    expected
        .id
        .as_ref()
        .filter(|id| id.table.as_str() == "note")
        .cloned()
        .ok_or_else(|| DbError::NoteRevisionConflict("(missing note ID)".into()))
}

/// Keep the entity upsert semantics aligned with `Repository::upsert_entity`:
/// explicit identities identify rows, existing type/creation time survive, and
/// legacy metadata aliases merge distinctly, while extracted metadata reflects
/// the current result. Per-mention evidence owns aliases used by graph search.
/// Running these writes after the snapshot
/// guard and inside the note transaction prevents failed edits from changing
/// shared entities or leaving unused rows behind. Resolve IDs after all
/// upserts so repeated identities create only one mention.
pub(super) fn replacement_entities_transaction() -> &'static str {
    "FOR $entity IN $replacement_entities { \
        INSERT INTO entity (entity_type, name, canonical_name, identity_key, embedding, metadata, created_at) \
        VALUES ($entity.entity_type, $entity.name, $entity.canonical_name, $entity.identity_key ?? string::concat('legacy:', $entity.canonical_name), \
                $entity.embedding ?? [], $entity.metadata, time::now()) \
        ON DUPLICATE KEY UPDATE \
            name = $entity.name, embedding = $entity.embedding ?? [], \
            metadata = IF string::starts_with($entity.identity_key ?? '', 'extracted-v1:') THEN \
                object::extend(object::extend(metadata ?? {}, $entity.metadata ?? {}), \
                    { aliases: array::slice($entity.metadata.aliases ?? [], 0, 8) }) \
            ELSE object::extend( \
                object::extend(metadata ?? {}, $entity.metadata ?? {}), \
                { extraction: IF $entity.metadata.extraction = NONE THEN metadata.extraction ELSE object::extend($entity.metadata.extraction, { \
                    mention_spellings: array::distinct(array::concat(metadata.extraction.mention_spellings ?? [], $entity.metadata.extraction.mention_spellings ?? [])), \
                    alias_spellings: array::distinct(array::concat(metadata.extraction.alias_spellings ?? [], $entity.metadata.extraction.alias_spellings ?? [])), \
                    reported_types: array::distinct(array::concat(metadata.extraction.reported_types ?? [], $entity.metadata.extraction.reported_types ?? [])) \
                }) END, aliases: array::distinct(array::concat( \
                    metadata.aliases ?? [], $entity.metadata.aliases ?? [] \
                )) } \
            ) END; \
     }; \
     LET $entity_ids = (SELECT VALUE id FROM entity \
                       WHERE identity_key IN $replacement_entity_names); "
}

/// Store the current extraction evidence on its owning mention. Shared source
/// entities may have different valid aliases in different chunks; replacing or
/// deleting one chunk must not preserve its aliases or erase another's.
pub(super) fn replacement_mentions_transaction(note_variable: &str) -> String {
    format!(
        "FOR $entity_id IN $entity_ids {{ \
            LET $evidence = array::filter($replacement_entities, |$entity| \
                ($entity.identity_key ?? string::concat('legacy:', $entity.canonical_name)) = $entity_id.identity_key)[0].metadata; \
            LET $current_evidence = IF $evidence = NONE THEN NONE ELSE object::extend($evidence, \
                    {{ aliases: array::slice($evidence.aliases ?? [], 0, 8) }}) END; \
            IF array::len((SELECT VALUE id FROM mentions WHERE in = {note_variable} AND out = $entity_id LIMIT 1)) = 0 {{ \
                CREATE mentions SET in = {note_variable}, out = $entity_id, metadata = $current_evidence; \
            }} ELSE {{ \
                UPDATE mentions SET metadata = $current_evidence WHERE in = {note_variable} AND out = $entity_id; \
            }}; \
         }}; "
    )
}

fn check_note_mutation_errors(
    errors: HashMap<usize, surrealdb::Error>,
    operation: &str,
    expected: Option<&Note>,
) -> Result<()> {
    if errors.is_empty() {
        return Ok(());
    }
    if let Some(expected) = expected {
        if errors.values().any(|error| {
            // The SDK preserves the engine's THROW display prefix rather
            // than always marking it as a public `Thrown` error. Match only
            // our fixed SQL sentinel; callers still receive a typed error.
            (error
                .message()
                .strip_prefix("An error occurred: ")
                .unwrap_or(error.message())
                == EDITOR_REVISION_CONFLICT)
                || matches!(
                    error.details(),
                    surrealdb_types::ErrorDetails::Query(Some(
                        surrealdb_types::QueryError::TransactionConflict
                    ))
                )
        }) {
            return Err(DbError::NoteRevisionConflict(record_id_to_string(
                &editor_source_id(expected)?,
            )));
        }
    }
    Err(DbError::QueryFailed(format!(
        "atomic note-and-mention {operation} failed: {}",
        errors
            .into_iter()
            .map(|(statement, error)| format!("statement {statement}: {error}"))
            .collect::<Vec<_>>()
            .join("; ")
    )))
}

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
                    extraction_scope = $extraction_scope, \
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
            .bind(("extraction_scope", note.extraction_scope.clone()))
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

    /// Atomically create a manual note, upsert its extracted entities, and
    /// write its complete mention set. Failed note/entity/mention writes roll
    /// back together, so a retry cannot leave partial extraction records or
    /// duplicate manual copies behind.
    #[instrument(skip(self, note, entities))]
    pub async fn create_note_and_replace_entities(
        &self,
        note: Note,
        entities: Vec<Entity>,
    ) -> Result<Note> {
        self.create_note_and_replace_entities_guarded(note, entities, None, None)
            .await
    }

    /// Create a chat-derived note and its conversation ownership together.
    /// A note is never visible without the relationship that distinguishes it
    /// from a detached manual copy, even if later extraction or message linking
    /// fails. Source-generation and source-lifecycle semantics stay unchanged.
    #[instrument(skip(self, note))]
    pub async fn create_chat_note(&self, note: Note, conversation_id: &RecordId) -> Result<Note> {
        self.create_note_and_replace_entities_guarded(note, Vec::new(), None, Some(conversation_id))
            .await
    }

    /// Create a detached manual copy only if its editor-opening source note
    /// remains visible and exactly unchanged. The guard and note/mention
    /// creation share a transaction, so conflicts leave no detached copy.
    #[instrument(skip(self, note, entities, expected))]
    pub async fn create_note_and_replace_entities_if_unchanged(
        &self,
        note: Note,
        entities: Vec<Entity>,
        expected: &Note,
    ) -> Result<Note> {
        self.create_note_and_replace_entities_guarded(note, entities, Some(expected), None)
            .await
    }

    async fn create_note_and_replace_entities_guarded(
        &self,
        note: Note,
        entities: Vec<Entity>,
        expected: Option<&Note>,
        conversation_id: Option<&RecordId>,
    ) -> Result<Note> {
        let editor_source = expected.map(editor_source_id).transpose()?;
        let guard = expected
            .map(|_| editor_snapshot_guard(false))
            .unwrap_or_default();
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let replacement_entities = replacement_entities_transaction();
        let replacement_mentions = replacement_mentions_transaction("$id");
        let entity_names: Vec<String> = entities
            .iter()
            .map(Entity::effective_identity_key)
            .collect();
        let note_id = note
            .id
            .clone()
            .unwrap_or_else(|| RecordId::new("note", Uuid::new_v4().to_string()));
        if note_id.table.as_str() != "note" || note_id.key.is_range() {
            return Err(DbError::QueryFailed(
                "atomic note creation requires one note ID".into(),
            ));
        }
        let mut response = self
            .db
            .query(format!(
                "BEGIN TRANSACTION; {guard}{replacement_entities}\
                 IF $chat_conversation != NONE {{ \
                    CREATE note_from_conversation SET in = $id, out = $chat_conversation; \
                 }}; \
                 CREATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, source_id = $source_id, \
                    source_generation = $source_generation, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    extraction_scope = $extraction_scope, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at; \
                 {replacement_mentions}\
                 COMMIT TRANSACTION;"
            ))
            .bind(("editor_source", editor_source))
            .bind(("editor_expected", expected.cloned()))
            .bind(("chat_conversation", conversation_id.cloned()))
            .bind(("id", note_id.clone()))
            .bind(("note_type", serde_json::to_value(&note.note_type).map_err(|error| DbError::QueryFailed(error.to_string()))?))
            .bind(("title", note.title.clone()))
            .bind(("content", note.content.clone()))
            .bind(("embedding", (!note.embedding.is_empty()).then_some(note.embedding.clone())))
            .bind(("source_id", note.source_id.clone()))
            .bind(("source_generation", note.source_generation.map(|generation| generation as i64)))
            .bind(("chunk_key", note.chunk_key.clone()))
            .bind(("chunk_location_key", note.chunk_location_key.clone()))
            .bind(("extraction_scope", note.extraction_scope.clone()))
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
            .bind(("replacement_entities", entities))
            .bind(("replacement_entity_names", entity_names))
            .await?;
        check_note_mutation_errors(response.take_errors(), "create", expected)?;
        // The final response slots are CREATE, the mention loop, and COMMIT.
        // Read the authoritative, schema-normalized CREATE result only after
        // every statement (including COMMIT) succeeds. A separate read here
        // could fail after persistence and incorrectly encourage a retry that
        // creates another manual note or detached copy.
        let created_index = response.num_statements().checked_sub(3).ok_or_else(|| {
            DbError::CreateFailed("missing atomic note-and-mention create result".into())
        })?;
        let created: Option<Note> = response.take(created_index)?;
        created.ok_or_else(|| DbError::CreateFailed("atomic note-and-mention create".into()))
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

    /// Source generations and chat provenance relationships identify imported
    /// notes. Source IDs/types and tags alone do not: detached manual copies
    /// intentionally retain those fields without generation or chat links.
    pub async fn note_requires_detach(&self, note: &Note) -> Result<bool> {
        if note.source_generation.is_some() {
            return Ok(true);
        }
        match &note.id {
            Some(id) => self.note_has_chat_provenance(id).await,
            None => Ok(false),
        }
    }

    /// Update a note
    #[instrument(skip(self, note))]
    pub async fn update_note(&self, id: &str, note: Note) -> Result<Note> {
        let _lifecycle_guard = self.proposal_acceptance_lock.lock().await;
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let existing = self
            .get_note(raw_id)
            .await?
            .ok_or_else(|| DbError::NotFound("note".into(), id.into()))?;
        let search_content = search_content_for_note_update(&existing, &note);
        let invalidation =
            super::remote_endpoint_proposals::invalidate_reviewed_endpoint_sql("[$id]");
        let mut response = self
            .db
            .query(format!(
                "BEGIN TRANSACTION; {invalidation} UPDATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    extraction_scope = IF $extraction_scope = NONE THEN extraction_scope ELSE $extraction_scope END, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    source_id = IF $source_id = NONE THEN source_id ELSE $source_id END, \
                    source_generation = IF $source_generation = NONE THEN source_generation ELSE $source_generation END, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at \
                 RETURN AFTER; COMMIT TRANSACTION;",
            ))
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
            .bind(("extraction_scope", note.extraction_scope.clone()))
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
            .await?;
        response
            .take_errors()
            .into_iter()
            .next()
            .map_or(Ok(()), |(_, error)| Err(DbError::Surreal(error)))?;
        let index = response
            .num_statements()
            .checked_sub(2)
            .ok_or_else(|| DbError::QueryFailed("missing note update result".into()))?;
        let updated: Option<Note> = response.take(index)?;

        updated.ok_or_else(|| DbError::NotFound("note".into(), id.into()))
    }

    /// Atomically replace a note's searchable payload, extracted entities, and
    /// complete mention set. Any failure rolls back shared entity changes as
    /// well as note content and mentions.
    #[instrument(skip(self, note, entities))]
    pub async fn update_note_and_replace_entities(
        &self,
        id: &str,
        note: Note,
        entities: Vec<Entity>,
    ) -> Result<Note> {
        self.update_note_and_replace_entities_guarded(id, note, entities, None)
            .await
    }

    /// Replace an editor draft and its mention set only while the persisted
    /// manual note is still visible and matches the editor-opening snapshot.
    /// Generation ownership or either chat-provenance relationship rejects the
    /// update as a NoteRevisionConflict, including links added during provider
    /// work. A deleted note cannot be recreated by the subsequent UPDATE.
    #[instrument(skip(self, note, entities, expected))]
    pub async fn update_note_and_replace_entities_if_unchanged(
        &self,
        id: &str,
        note: Note,
        entities: Vec<Entity>,
        expected: &Note,
    ) -> Result<Note> {
        self.update_note_and_replace_entities_guarded(id, note, entities, Some(expected))
            .await
    }

    async fn update_note_and_replace_entities_guarded(
        &self,
        id: &str,
        note: Note,
        entities: Vec<Entity>,
        expected: Option<&Note>,
    ) -> Result<Note> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let raw_id = id.strip_prefix("note:").unwrap_or(id);
        let note_id = RecordId::new("note", raw_id);
        let (existing, guard, editor_source) = if let Some(expected) = expected {
            let editor_source = editor_source_id(expected)?;
            if editor_source != note_id {
                return Err(DbError::NoteRevisionConflict(id.into()));
            }
            (
                expected.clone(),
                editor_snapshot_guard(true),
                Some(editor_source),
            )
        } else {
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
            (existing, String::new(), None)
        };
        let invalidation =
            super::remote_endpoint_proposals::invalidate_reviewed_endpoint_sql("[$id]");
        let replacement_entities = replacement_entities_transaction();
        let replacement_mentions = replacement_mentions_transaction("$id");
        let entity_names: Vec<String> = entities
            .iter()
            .map(Entity::effective_identity_key)
            .collect();
        let search_content = search_content_for_note_update(&existing, &note);

        let mut response = self
            .db
            .query(format!(
                "BEGIN TRANSACTION; {guard}{invalidation}{replacement_entities}\
                 UPDATE $id SET \
                    note_type = $note_type, title = $title, content = $content, \
                    embedding = $embedding, chunk_key = $chunk_key, \
                    chunk_location_key = $chunk_location_key, chunk_ordinal = $chunk_ordinal, \
                    extraction_scope = IF $extraction_scope = NONE THEN extraction_scope ELSE $extraction_scope END, \
                    chunk_heading_path = $chunk_heading_path, source_start_line = $source_start_line, \
                    source_end_line = $source_end_line, source_start_byte = $source_start_byte, \
                    source_end_byte = $source_end_byte, chunk_overlap_from = $chunk_overlap_from, \
                    chunk_overlap_chars = $chunk_overlap_chars, split_fenced_code = $split_fenced_code, \
                    content_hash = $content_hash, \
                    search_content = IF $search_content = NONE THEN $content ELSE $search_content END, tags = $tags, \
                    source_id = IF $source_id = NONE THEN source_id ELSE $source_id END, \
                    source_generation = IF $source_generation = NONE THEN source_generation ELSE $source_generation END, \
                    created_at = <datetime>$created_at, updated_at = <datetime>$updated_at RETURN AFTER; \
                 DELETE mentions WHERE in = $id; \
                 {replacement_mentions}\
                 COMMIT TRANSACTION;"
            ))
            .bind(("editor_source", editor_source))
            .bind(("editor_expected", expected.cloned()))
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
            .bind(("extraction_scope", note.extraction_scope.clone()))
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
            .bind(("replacement_entities", entities))
            .bind(("replacement_entity_names", entity_names))
            .await?;
        check_note_mutation_errors(response.take_errors(), "update", expected)?;
        // The final response slots are UPDATE AFTER, mention deletion, the
        // replacement mention loop, and COMMIT. Once all statements succeed,
        // return the saved row without a fallible post-commit read that could
        // report an already persisted edit as a failure.
        let updated_index = response.num_statements().checked_sub(4).ok_or_else(|| {
            DbError::QueryFailed("missing atomic note-and-mention update result".into())
        })?;
        let updated: Option<Note> = response.take(updated_index)?;
        updated.ok_or_else(|| DbError::NotFound("note".into(), id.into()))
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
        let _lifecycle_guard = self.proposal_acceptance_lock.lock().await;
        let invalidation =
            super::remote_endpoint_proposals::invalidate_reviewed_endpoint_sql("[$id]");
        self.db
            .query(format!(
                "BEGIN TRANSACTION; {invalidation} \
                 UPDATE note SET embedding = $embedding, updated_at = time::now() WHERE id = $id; \
                 COMMIT TRANSACTION;"
            ))
            .bind(("id", id.clone()))
            .bind(("embedding", embedding))
            .await?
            .check()?;

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
        self.hybrid_search_notes_with_fusion_and_lexical_candidates(
            query_text, embedding, limit, since, source_uri, fusion, 0,
        )
        .await
        .map(|(results, _)| results)
    }

    /// Retain a bounded lexical prefix for graph seeds without repeating the
    /// full-text query. The fusion channel keeps its original candidate limit,
    /// even when the separately bounded graph prefix is larger.
    #[instrument(skip(self, embedding, fusion))]
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search_notes_with_fusion_and_lexical_candidates(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
        fusion: &FusionConfig,
        lexical_limit: usize,
    ) -> Result<(Vec<SearchResult>, Vec<SearchResult>)> {
        let candidate_limit = fusion.candidate_limit(limit);
        let lexical_limit = lexical_limit.min(200);

        let started = std::time::Instant::now();
        let vec_results = self
            .vector_search_notes(
                embedding.clone(),
                candidate_limit,
                since,
                source_uri.clone(),
            )
            .await?;
        tracing::debug!(
            phase = "note_vector",
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            count = vec_results.len(),
            "Retrieval phase completed"
        );

        let started = std::time::Instant::now();
        let mut fts_results = self
            .fulltext_search_notes(
                query_text,
                candidate_limit.max(lexical_limit),
                since,
                source_uri,
            )
            .await?;
        tracing::debug!(
            phase = "note_fulltext",
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            count = fts_results.len(),
            "Retrieval phase completed"
        );

        let lexical_candidates = fts_results.iter().take(lexical_limit).cloned().collect();
        fts_results.truncate(candidate_limit);
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
            existing.exact_title_match |= incoming.exact_title_match;
        });
        if results.len() > limit {
            results.truncate(limit);
        }
        Ok((results, lexical_candidates))
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
        // Case folding and outer whitespace are the only equivalences here:
        // punctuation and internal whitespace can distinguish note titles.
        let normalized_title = query.trim().to_lowercase();
        // Score/filter the exact mirrored lexical population before hydrating
        // the bounded primary rows. A single SQL statement holds one engine
        // snapshot for candidates, source visibility and canonical contents.
        // The physical native keys mirror note IDs, preserving all tie types.
        let results: Vec<SearchResult> = self
            .db
            .query(
                r#"
                SELECT
                    record_id AS id,
                    record_id.title AS title,
                    record_id.content AS content,
                    record_id.note_type AS note_type,
                    record_id.tags AS tags,
                    record_id.created_at AS created_at,
                    record_id.source_id.uri AS source_uri,
                    exact_title_match,
                    fts_score
                FROM (
                    SELECT id, record_id,
                        ($normalized_title != '' AND string::lowercase(string::trim(title ?? '')) = $normalized_title) AS exact_title_match,
                        (search::score(0) * 0.7 + search::score(1) * 0.2 + search::score(2) * 0.1) AS fts_score
                    FROM note_search
                    WHERE (search_content @0@ $query OR content @1@ $query OR title @2@ $query)
                      AND ($since = NONE OR created_at >= <datetime>$since)
                      AND ($source_uri = NONE OR source_id.uri = $source_uri)
                      AND (
                        source_id IS NONE
                        OR source_generation IS NONE
                        OR source_generation = source_id.successful_generation
                      )
                    ORDER BY exact_title_match DESC, fts_score DESC, id ASC
                    LIMIT $limit
                )
                ORDER BY exact_title_match DESC, fts_score DESC, id ASC
            "#,
            )
            .bind(("query", query.to_string()))
            .bind(("normalized_title", normalized_title))
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
