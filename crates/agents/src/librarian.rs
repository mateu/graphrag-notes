//! Librarian Agent - Ingests content and creates notes

use crate::{Result, TeiClient, TgiClient};
use graphrag_core::{
    ChatConversation, ChatExport, ChatMessage, Entity, EntityType, MessageRole, Note, NoteType,
    Source, SourceType,
};
use graphrag_db::Repository;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument};

const DEFAULT_PROGRESS_EVERY: usize = 10;
const DEFAULT_PROGRESS_EVERY_SECS: u64 = 5;
const DEFAULT_EXTRACT_MAX_CHARS: usize = 8000;

fn skip_entity_extraction() -> bool {
    std::env::var("SKIP_ENTITY_EXTRACTION")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn extract_log_each() -> bool {
    std::env::var("EXTRACT_LOG_EACH")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn extract_max_chars() -> usize {
    std::env::var("EXTRACT_MAX_CHARS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EXTRACT_MAX_CHARS)
}

fn truncate_for_extraction(text: &str) -> String {
    let max_chars = extract_max_chars();
    if max_chars == 0 {
        return text.to_string();
    }

    let mut iter = text.chars();
    let mut collected = String::new();
    for _ in 0..max_chars {
        if let Some(ch) = iter.next() {
            collected.push(ch);
        } else {
            return text.to_string();
        }
    }

    collected.push_str("\n\n[truncated]");
    collected
}

/// The Librarian agent handles content ingestion
pub struct LibrarianAgent {
    repo: Repository,
    tei: TeiClient,
    tgi: TgiClient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatImportMode {
    Qa,
    Message,
    Hybrid,
}

#[derive(Debug, Default)]
struct QaExtractionResult {
    pairs: Vec<QaPair>,
    dropped_short_question: usize,
    dropped_short_answer: usize,
    assistant_without_human: usize,
    trailing_unpaired_human: usize,
}

#[derive(Debug, Clone)]
struct QaPair {
    question: String,
    answer: String,
    human_idx: usize,
    assistant_idx: usize,
}

#[derive(Debug, Default)]
struct ConversationImportOutcome {
    notes_created: usize,
    notes_from_qa: usize,
    notes_from_messages: usize,
    notes_from_summaries: usize,
    notes_from_fallback: usize,
    qa_pairs_created: usize,
    qa_pairs_dropped_short_question: usize,
    qa_pairs_dropped_short_answer: usize,
    assistant_without_human: usize,
    trailing_unpaired_human: usize,
}

#[derive(Debug, Default)]
struct MessageSignal {
    block_count: usize,
    tool_block_count: usize,
    citation_count: usize,
    has_tooling: bool,
    has_files: bool,
    has_citations: bool,
}

impl LibrarianAgent {
    /// Create a new Librarian agent
    pub fn new(repo: Repository, tei: TeiClient, tgi: TgiClient) -> Self {
        Self { repo, tei, tgi }
    }

    /// Ingest raw text content and create a note
    #[instrument(skip(self, content))]
    pub async fn ingest_text<C>(
        &self,
        content: C,
        title: Option<String>,
        tags: Vec<String>,
    ) -> Result<Note>
    where
        C: Into<String> + std::fmt::Debug,
    {
        let content = content.into();
        info!("Ingesting text content ({} chars)", content.len());

        // Create source record
        let source = Source::manual()
            .with_title(title.clone().unwrap_or_else(|| "Manual note".into()))
            .with_content(content.clone());
        let source = self.repo.create_source(source).await?;
        let source_id = source.id.clone();

        // Generate embedding
        debug!("Generating embedding...");
        let embedding = self.tei.embed(&content, false).await?;

        // Determine title
        let note_title = if let Some(t) = title {
            Some(t)
        } else {
            // Fallback heuristic: first 3-5 words, max 48 characters
            content.lines().find(|l| !l.trim().is_empty()).map(|l| {
                let l = l.trim();
                let words: Vec<&str> = l.split_whitespace().collect();
                let mut title = words.iter().take(5).cloned().collect::<Vec<_>>().join(" ");
                if title.chars().count() > 48 {
                    title = title.chars().take(48).collect();
                }
                title
            })
        };

        // Create the note
        let mut note = Note::new(content)
            .with_type(NoteType::Raw)
            .with_embedding(embedding)
            .with_tags(tags);

        if let Some(t) = note_title {
            note = note.with_title(t);
        }

        if let Some(sid) = source_id {
            note = note.with_source(sid.to_string());
        }

        let note = self.repo.create_note(note).await?;

        info!("Created note with id: {:?}", note.id);

        // Extract and link entities (best effort)
        if let Err(e) = self.extract_and_link_entities(&note).await {
            debug!("Entity extraction failed (non-fatal): {}", e);
        }

        Ok(note)
    }

    /// Ingest from a markdown file
    #[instrument(skip(self))]
    pub async fn ingest_markdown<C>(&self, path: &str, content: C) -> Result<Vec<Note>>
    where
        C: Into<String> + std::fmt::Debug,
    {
        let content = content.into();
        info!("Ingesting markdown from: {}", path);

        // Create source
        let source = Source::from_file(path, SourceType::Markdown).with_content(content.clone());
        let source = self.repo.create_source(source).await?;
        let source_id = source.id.as_ref().map(|id| id.to_string());

        // For MVP, treat the whole file as one note
        // Future: parse markdown and create atomic notes
        let notes = self.chunk_and_create_notes(&content, source_id).await?;

        info!("Created {} notes from markdown", notes.len());

        Ok(notes)
    }

    /// Process notes that don't have embeddings yet
    #[instrument(skip(self))]
    pub async fn process_pending_embeddings(&self) -> Result<usize> {
        let notes = self.repo.get_notes_without_embeddings().await?;

        if notes.is_empty() {
            return Ok(0);
        }

        info!("Processing {} notes without embeddings", notes.len());

        // Batch embed for efficiency
        let texts: Vec<String> = notes.iter().map(|n| n.content.clone()).collect();
        let embeddings = self.tei.embed_batch(&texts, false).await?;

        // Update each note
        for (note, embedding) in notes.iter().zip(embeddings.into_iter()) {
            if let Some(ref id) = note.id {
                self.repo.update_note_embedding(id, embedding).await?;
            }
        }

        Ok(notes.len())
    }

    /// Extract entities from a note and link them
    async fn extract_and_link_entities(&self, note: &Note) -> Result<()> {
        self.extract_and_link_entities_inner(note, false).await
    }

    /// Extract entities from a note regardless of skip flag
    async fn extract_and_link_entities_force(&self, note: &Note) -> Result<()> {
        self.extract_and_link_entities_inner(note, true).await
    }

    async fn extract_and_link_entities_inner(&self, note: &Note, force: bool) -> Result<()> {
        if !force && skip_entity_extraction() {
            return Ok(());
        }

        let text = truncate_for_extraction(&note.content);
        let extraction = self.tgi.extract(&text).await?;
        let entities = extraction.entities;
        let extracted_count = entities.len();
        let mut linked_count = 0usize;

        for extracted in entities {
            // Map string type to EntityType
            let entity_type = match extracted
                .entity_type
                .as_deref()
                .unwrap_or("concept")
                .to_lowercase()
                .as_str()
            {
                "person" | "per" => EntityType::Person,
                "organization" | "org" => EntityType::Organization,
                "location" | "loc" | "gpe" => EntityType::Location,
                "date" | "time" => EntityType::Date,
                _ => EntityType::Concept,
            };

            let entity = Entity::new(&extracted.name, entity_type);
            let entity = self.repo.upsert_entity(entity).await?;

            // Link note to entity
            if let (Some(note_id), Some(entity_id)) = (&note.id, &entity.id) {
                self.repo.link_note_to_entity(note_id, entity_id).await?;
                linked_count += 1;
            }
        }

        if extracted_count > 0 || linked_count > 0 {
            debug!(
                "Entity extraction summary: extracted={} linked={} note_id={:?}",
                extracted_count, linked_count, note.id
            );
        }

        Ok(())
    }

    /// Extract entities for notes missing entity links
    #[instrument(skip(self))]
    pub async fn extract_entities_for_notes(&self, limit: usize) -> Result<usize> {
        let notes = self.repo.get_notes_without_entities(limit).await?;
        if notes.is_empty() {
            return Ok(0);
        }

        let progress_every = std::env::var("EXTRACT_PROGRESS_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY);
        let progress_every_secs = std::env::var("EXTRACT_PROGRESS_EVERY_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY_SECS);
        let log_each = extract_log_each();

        let mut processed = 0usize;
        let total = notes.len();
        let start = Instant::now();
        let mut last_progress = Instant::now();

        for (index, note) in notes.into_iter().enumerate() {
            let note_id = note
                .id
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            let note_len = note.content.len();
            if log_each {
                info!(
                    "Entity extraction start: {}/{} note_id={} chars={}",
                    index + 1,
                    total,
                    note_id,
                    note_len
                );
            }

            let note_start = Instant::now();
            match self.extract_and_link_entities_force(&note).await {
                Ok(()) => {
                    processed += 1;
                    if log_each {
                        info!(
                            "Entity extraction done: {}/{} note_id={} elapsed={:.2}s",
                            index + 1,
                            total,
                            note_id,
                            note_start.elapsed().as_secs_f32()
                        );
                    }
                }
                Err(e) => {
                    if log_each {
                        info!(
                            "Entity extraction failed: {}/{} note_id={} elapsed={:.2}s error={}",
                            index + 1,
                            total,
                            note_id,
                            note_start.elapsed().as_secs_f32(),
                            e
                        );
                    } else {
                        debug!("Entity extraction failed (non-fatal): {}", e);
                    }
                }
            }

            if processed % progress_every == 0
                || last_progress.elapsed() >= Duration::from_secs(progress_every_secs)
            {
                let elapsed = start.elapsed().as_secs_f32().max(0.001);
                let rate = processed as f32 / elapsed;
                let remaining = total.saturating_sub(processed);
                let eta_secs = if rate > 0.0 {
                    (remaining as f32 / rate).round() as u64
                } else {
                    0
                };
                info!(
                    "Entity extraction progress: {}/{} notes (rate: {:.2}/s, eta: {}s)",
                    processed, total, rate, eta_secs
                );
                last_progress = Instant::now();
            }
        }

        Ok(processed)
    }

    /// Extract entities for all notes (optionally clearing existing mentions first)
    #[instrument(skip(self))]
    pub async fn extract_entities_for_all_notes(
        &self,
        limit: usize,
        force_clear: bool,
    ) -> Result<usize> {
        let progress_every = std::env::var("EXTRACT_PROGRESS_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY);
        let progress_every_secs = std::env::var("EXTRACT_PROGRESS_EVERY_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY_SECS);
        let log_each = extract_log_each();

        let mut processed = 0usize;
        let start = Instant::now();
        let mut last_progress = Instant::now();
        let mut offset = 0usize;

        loop {
            let notes = self.repo.get_notes_page(limit, offset).await?;
            if notes.is_empty() {
                break;
            }

            let total = notes.len();
            for (index, note) in notes.into_iter().enumerate() {
                let note_id = note
                    .id
                    .as_ref()
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let note_len = note.content.len();
                if log_each {
                    info!(
                        "Entity extraction start: {}/{} note_id={} chars={}",
                        index + 1,
                        total,
                        note_id,
                        note_len
                    );
                }

                if force_clear {
                    if let Some(ref id) = note.id {
                        self.repo.delete_mentions_for_note(id).await?;
                    }
                }

                let note_start = Instant::now();
                match self.extract_and_link_entities_force(&note).await {
                    Ok(()) => {
                        processed += 1;
                        if log_each {
                            info!(
                                "Entity extraction done: {}/{} note_id={} elapsed={:.2}s",
                                index + 1,
                                total,
                                note_id,
                                note_start.elapsed().as_secs_f32()
                            );
                        }
                    }
                    Err(e) => {
                        if log_each {
                            info!(
                                "Entity extraction failed: {}/{} note_id={} elapsed={:.2}s error={}",
                                index + 1,
                                total,
                                note_id,
                                note_start.elapsed().as_secs_f32(),
                                e
                            );
                        } else {
                            debug!("Entity extraction failed (non-fatal): {}", e);
                        }
                    }
                }

                if processed % progress_every == 0
                    || last_progress.elapsed() >= Duration::from_secs(progress_every_secs)
                {
                    let elapsed = start.elapsed().as_secs_f32().max(0.001);
                    let rate = processed as f32 / elapsed;
                    let eta_secs = 0;
                    info!(
                        "Entity extraction progress: {} processed (rate: {:.2}/s, eta: {}s)",
                        processed, rate, eta_secs
                    );
                    last_progress = Instant::now();
                }
            }

            offset += limit;
        }

        Ok(processed)
    }

    /// Extract entities for explicit note ids
    #[instrument(skip(self, note_ids))]
    pub async fn extract_entities_for_note_ids(
        &self,
        note_ids: &[String],
        force_clear: bool,
    ) -> Result<usize> {
        if note_ids.is_empty() {
            return Ok(0);
        }

        let log_each = extract_log_each();
        let mut processed = 0usize;

        for (index, note_id_raw) in note_ids.iter().enumerate() {
            let key = note_id_raw.strip_prefix("note:").unwrap_or(note_id_raw);
            let maybe_note = self.repo.get_note(key).await?;
            let Some(note) = maybe_note else {
                debug!("Note not found for extraction: {}", note_id_raw);
                continue;
            };

            let note_id = note
                .id
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            let note_len = note.content.len();
            if log_each {
                info!(
                    "Entity extraction start: {}/{} note_id={} chars={}",
                    index + 1,
                    note_ids.len(),
                    note_id,
                    note_len
                );
            }

            if force_clear {
                if let Some(ref id) = note.id {
                    self.repo.delete_mentions_for_note(id).await?;
                }
            }

            match self.extract_and_link_entities_force(&note).await {
                Ok(()) => {
                    processed += 1;
                    if log_each {
                        info!(
                            "Entity extraction done: {}/{} note_id={}",
                            index + 1,
                            note_ids.len(),
                            note_id
                        );
                    }
                }
                Err(e) => {
                    debug!("Entity extraction failed for {}: {}", note_id_raw, e);
                }
            }
        }

        Ok(processed)
    }

    /// Chunk content and create notes
    async fn chunk_and_create_notes(
        &self,
        content: &str,
        source_id: Option<String>,
    ) -> Result<Vec<Note>> {
        // Simple chunking: split by double newlines (paragraphs)
        // Future: smarter chunking with overlap
        let chunks: Vec<&str> = content
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 20) // Skip very short chunks
            .collect();

        if chunks.is_empty() {
            // Treat whole content as one note
            let embedding = self.tei.embed(content, false).await?;
            let mut note = Note::new(content)
                .with_type(NoteType::Raw)
                .with_embedding(embedding);

            if let Some(ref sid) = source_id {
                note = note.with_source(sid.clone());
            }

            let note = self.repo.create_note(note).await?;
            return Ok(vec![note]);
        }

        // Generate embeddings in batch
        let texts: Vec<String> = chunks.iter().map(|s| s.to_string()).collect();
        let embeddings = self.tei.embed_batch(&texts, false).await?;

        let mut notes = Vec::new();

        for (chunk, embedding) in chunks.iter().zip(embeddings.into_iter()) {
            let mut note = Note::new(*chunk)
                .with_type(NoteType::Raw)
                .with_embedding(embedding);

            if let Some(ref sid) = source_id {
                note = note.with_source(sid.clone());
            }

            let note = self.repo.create_note(note).await?;
            notes.push(note);
        }

        Ok(notes)
    }

    /// Ingest a chat export (e.g., from Claude Desktop)
    ///
    /// Creates notes from chat conversations. Each Q&A pair (human message followed
    /// by assistant response) becomes a single note for better semantic search.
    #[instrument(skip(self, export))]
    pub async fn ingest_chat_export(
        &self,
        export: ChatExport,
        source_uri: Option<String>,
        mode: ChatImportMode,
    ) -> Result<ChatImportResult> {
        let total = export.conversation_count();
        let progress_every = std::env::var("IMPORT_PROGRESS_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY);
        let progress_every_secs = std::env::var("IMPORT_PROGRESS_EVERY_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROGRESS_EVERY_SECS);
        let mut last_progress = Instant::now();

        info!(
            "Ingesting chat export with {} conversations (mode: {:?})",
            total, mode
        );

        let mut result = ChatImportResult {
            conversations_total: total,
            ..Default::default()
        };
        let mut processed = 0usize;

        for conversation in export.conversations {
            if conversation.messages.is_empty() {
                result.conversations_without_messages += 1;
                if !conversation.summary.is_empty() {
                    result.conversations_summary_only += 1;
                }
            } else {
                result.conversations_with_messages += 1;
            }
            result.messages_total += conversation.messages.len();

            match self
                .ingest_conversation(&conversation, source_uri.clone(), mode)
                .await
            {
                Ok(outcome) => {
                    result.conversations_imported += 1;
                    result.notes_created += outcome.notes_created;
                    result.notes_from_qa += outcome.notes_from_qa;
                    result.notes_from_messages += outcome.notes_from_messages;
                    result.notes_from_summaries += outcome.notes_from_summaries;
                    result.notes_from_fallback += outcome.notes_from_fallback;
                    result.qa_pairs_created += outcome.qa_pairs_created;
                    result.qa_pairs_dropped_short_question +=
                        outcome.qa_pairs_dropped_short_question;
                    result.qa_pairs_dropped_short_answer += outcome.qa_pairs_dropped_short_answer;
                    result.assistant_without_human += outcome.assistant_without_human;
                    result.trailing_unpaired_human += outcome.trailing_unpaired_human;
                }
                Err(e) => {
                    debug!("Failed to import conversation: {}", e);
                    result.conversations_failed += 1;
                    result.errors.push(format!(
                        "Conversation '{}': {}",
                        conversation.display_title(),
                        e
                    ));
                }
            }

            processed += 1;
            if processed % progress_every == 0
                || last_progress.elapsed() >= Duration::from_secs(progress_every_secs)
            {
                info!(
                    "Import progress: {}/{} conversations ({} ok, {} failed, {} notes)",
                    processed,
                    total,
                    result.conversations_imported,
                    result.conversations_failed,
                    result.notes_created
                );
                last_progress = Instant::now();
            }
        }

        info!(
            "Chat import complete: {} conversations, {} notes created",
            result.conversations_imported, result.notes_created
        );

        Ok(result)
    }

    /// Ingest a single conversation
    #[instrument(skip(self, conversation))]
    async fn ingest_conversation(
        &self,
        conversation: &ChatConversation,
        source_uri: Option<String>,
        mode: ChatImportMode,
    ) -> Result<ConversationImportOutcome> {
        let title = conversation.display_title();
        info!("Ingesting conversation: {}", title);

        // Create source record for this conversation
        let mut source = Source::chat_export(&title, source_uri);

        // Add conversation metadata
        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "conversation_id".into(),
            serde_json::json!(&conversation.uuid),
        );
        metadata.insert(
            "created_at".into(),
            serde_json::json!(&conversation.created_at),
        );
        if !conversation.summary.is_empty() {
            metadata.insert("summary".into(), serde_json::json!(&conversation.summary));
        }
        source = source.with_metadata(serde_json::Value::Object(metadata));

        let source = self.repo.create_source(source).await?;
        let source_id = source.id.as_ref().map(|id| id.to_string());
        let mut outcome = ConversationImportOutcome::default();

        if !conversation.summary.is_empty() {
            self.create_summary_note(conversation, source_id.clone())
                .await?;
            outcome.notes_created += 1;
            outcome.notes_from_summaries += 1;
        }

        let qa = self.extract_qa_pairs_with_diagnostics(&conversation.messages);
        outcome.qa_pairs_created = qa.pairs.len();
        outcome.qa_pairs_dropped_short_question = qa.dropped_short_question;
        outcome.qa_pairs_dropped_short_answer = qa.dropped_short_answer;
        outcome.assistant_without_human = qa.assistant_without_human;
        outcome.trailing_unpaired_human = qa.trailing_unpaired_human;

        match mode {
            ChatImportMode::Qa => {
                if qa.pairs.is_empty() {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self.chunk_and_create_notes(&markdown, source_id).await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                } else {
                    let created = self.create_qa_notes(&qa.pairs, source_id, &title).await?;
                    outcome.notes_created += created;
                    outcome.notes_from_qa += created;
                }
            }
            ChatImportMode::Message => {
                if conversation.messages.is_empty() {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self.chunk_and_create_notes(&markdown, source_id).await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                } else {
                    let selection: Vec<usize> = (0..conversation.messages.len()).collect();
                    let created = self
                        .create_message_notes(
                            conversation,
                            &conversation.messages,
                            &selection,
                            source_id,
                        )
                        .await?;
                    outcome.notes_created += created;
                    outcome.notes_from_messages += created;
                }
            }
            ChatImportMode::Hybrid => {
                if !qa.pairs.is_empty() {
                    let created = self
                        .create_qa_notes(&qa.pairs, source_id.clone(), &title)
                        .await?;
                    outcome.notes_created += created;
                    outcome.notes_from_qa += created;
                }

                let mut selected_messages = Vec::new();
                let paired_indices: HashSet<usize> = qa
                    .pairs
                    .iter()
                    .flat_map(|pair| [pair.human_idx, pair.assistant_idx])
                    .collect();
                for (idx, msg) in conversation.messages.iter().enumerate() {
                    let include =
                        !paired_indices.contains(&idx) || Self::message_has_extra_signal(msg);
                    if include {
                        selected_messages.push(idx);
                    }
                }

                if !selected_messages.is_empty() {
                    let created = self
                        .create_message_notes(
                            conversation,
                            &conversation.messages,
                            &selected_messages,
                            source_id.clone(),
                        )
                        .await?;
                    outcome.notes_created += created;
                    outcome.notes_from_messages += created;
                }

                if outcome.notes_from_qa == 0 && outcome.notes_from_messages == 0 {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self.chunk_and_create_notes(&markdown, source_id).await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                }
            }
        }

        info!(
            "Created {} notes from conversation '{}'",
            outcome.notes_created, title
        );

        Ok(outcome)
    }

    async fn create_qa_notes(
        &self,
        qa_pairs: &[QaPair],
        source_id: Option<String>,
        conversation_title: &str,
    ) -> Result<usize> {
        let mut texts_to_embed: Vec<String> = Vec::new();
        let mut note_builders: Vec<(String, Option<String>)> = Vec::new();

        for (idx, qa) in qa_pairs.iter().enumerate() {
            // Format the Q&A as a note
            let content = format!("**Question:** {}\n\n**Answer:** {}", qa.question, qa.answer);

            // Generate a title from the question
            let note_title = self.generate_qa_title(&qa.question, idx + 1);

            texts_to_embed.push(content.clone());
            note_builders.push((content, Some(note_title)));
        }

        // Batch embed all Q&A pairs
        let embeddings = self.tei.embed_batch(&texts_to_embed, false).await?;

        // Create notes
        let mut created = 0usize;
        for ((content, title), embedding) in note_builders.into_iter().zip(embeddings.into_iter()) {
            let mut note = Note::new(&content)
                .with_type(NoteType::Synthesis)
                .with_embedding(embedding)
                .with_tags(vec![
                    "chat-export".into(),
                    "qa".into(),
                    format!("conversation:{}", Self::slugify_tag(conversation_title)),
                ]);

            if let Some(t) = title {
                note = note.with_title(t);
            }

            if let Some(ref sid) = source_id {
                note = note.with_source(sid.clone());
            }

            let note = self.repo.create_note(note).await?;

            // Extract and link entities (best effort)
            if let Err(e) = self.extract_and_link_entities(&note).await {
                debug!("Entity extraction failed (non-fatal): {}", e);
            }

            created += 1;
        }

        Ok(created)
    }

    async fn create_summary_note(
        &self,
        conversation: &ChatConversation,
        source_id: Option<String>,
    ) -> Result<()> {
        let summary_title = format!("Summary: {}", conversation.display_title());
        let summary_content = format!(
            "**Conversation:** {}\n\n{}",
            conversation.display_title(),
            conversation.summary
        );
        let embedding = self.tei.embed(&summary_content, false).await?;
        let mut note = Note::new(summary_content)
            .with_type(NoteType::Synthesis)
            .with_title(summary_title)
            .with_embedding(embedding)
            .with_tags(vec!["chat-export".into(), "summary".into()]);

        if let Some(sid) = source_id {
            note = note.with_source(sid);
        }

        let note = self.repo.create_note(note).await?;
        if let Err(e) = self.extract_and_link_entities(&note).await {
            debug!("Entity extraction failed (non-fatal): {}", e);
        }

        Ok(())
    }

    async fn create_message_notes(
        &self,
        conversation: &ChatConversation,
        messages: &[ChatMessage],
        indices: &[usize],
        source_id: Option<String>,
    ) -> Result<usize> {
        if indices.is_empty() {
            return Ok(0);
        }

        let mut texts_to_embed = Vec::with_capacity(indices.len());
        let mut builders = Vec::with_capacity(indices.len());

        for idx in indices {
            if let Some(message) = messages.get(*idx) {
                let role_label = Self::role_label(&message.role);
                let mut tags = vec![
                    "chat-export".to_string(),
                    "message".to_string(),
                    format!("role:{}", role_label.to_ascii_lowercase()),
                    format!(
                        "conversation:{}",
                        Self::slugify_tag(&conversation.display_title())
                    ),
                ];
                let signal = Self::message_signal(message);
                if signal.has_tooling {
                    tags.push("tooling".into());
                }
                if signal.has_files {
                    tags.push("has-files".into());
                }
                if signal.has_citations {
                    tags.push("has-citations".into());
                }

                let text = if message.content.trim().is_empty() {
                    "[No text body. See structured content metadata below.]".to_string()
                } else {
                    message.content.clone()
                };
                let content = format!(
                    "**Conversation:** {}\n\
                     **Message #:** {}\n\
                     **Role:** {}\n\
                     **Created At:** {}\n\
                     **Structured Blocks:** {} (tools: {}, citations: {})\n\
                     **Files:** {}\n\n{}",
                    conversation.display_title(),
                    idx + 1,
                    role_label,
                    message
                        .created_at
                        .as_ref()
                        .map(|ts| ts.to_rfc3339())
                        .unwrap_or_else(|| "unknown".into()),
                    signal.block_count,
                    signal.tool_block_count,
                    signal.citation_count,
                    if message.files.is_empty() {
                        "(none)".into()
                    } else {
                        message.files.len().to_string()
                    },
                    text
                );
                let title = format!("{} message #{}", role_label, idx + 1);
                texts_to_embed.push(content.clone());
                builders.push((content, title, tags));
            }
        }

        let embeddings = self.tei.embed_batch(&texts_to_embed, false).await?;
        let mut created = 0usize;

        for ((content, title, tags), embedding) in builders.into_iter().zip(embeddings.into_iter())
        {
            let mut note = Note::new(content)
                .with_type(NoteType::Raw)
                .with_title(title)
                .with_embedding(embedding)
                .with_tags(tags);

            if let Some(sid) = source_id.as_ref() {
                note = note.with_source(sid.clone());
            }

            let note = self.repo.create_note(note).await?;
            if let Err(e) = self.extract_and_link_entities(&note).await {
                debug!("Entity extraction failed (non-fatal): {}", e);
            }
            created += 1;
        }

        Ok(created)
    }

    /// Extract Q&A pairs from a list of messages and collect diagnostics
    fn extract_qa_pairs_with_diagnostics(&self, messages: &[ChatMessage]) -> QaExtractionResult {
        let mut result = QaExtractionResult::default();
        let mut current_human: Option<(usize, &str)> = None;

        for (idx, msg) in messages.iter().enumerate() {
            match msg.role {
                MessageRole::Human => {
                    current_human = Some((idx, &msg.content));
                }
                MessageRole::Assistant => {
                    if let Some((human_idx, question)) = current_human.take() {
                        // Only include if both question and answer are substantial
                        if question.len() > 10 && msg.content.len() > 20 {
                            result.pairs.push(QaPair {
                                question: question.to_string(),
                                answer: msg.content.clone(),
                                human_idx,
                                assistant_idx: idx,
                            });
                        } else if question.len() <= 10 {
                            result.dropped_short_question += 1;
                        } else {
                            result.dropped_short_answer += 1;
                        }
                    } else {
                        result.assistant_without_human += 1;
                    }
                }
                MessageRole::System => {
                    // Skip system messages for Q&A extraction
                }
            }
        }

        if current_human.is_some() {
            result.trailing_unpaired_human = 1;
        }

        result
    }

    fn role_label(role: &MessageRole) -> &'static str {
        match role {
            MessageRole::Human => "Human",
            MessageRole::Assistant => "Assistant",
            MessageRole::System => "System",
        }
    }

    fn slugify_tag(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        let mut last_dash = false;
        for ch in value.chars() {
            let ch = ch.to_ascii_lowercase();
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
                last_dash = false;
            } else if !last_dash {
                out.push('-');
                last_dash = true;
            }
        }
        out.trim_matches('-').to_string()
    }

    fn message_has_extra_signal(message: &ChatMessage) -> bool {
        let signal = Self::message_signal(message);
        signal.has_tooling
            || signal.has_files
            || signal.has_citations
            || message.content.trim().is_empty()
    }

    fn message_signal(message: &ChatMessage) -> MessageSignal {
        let mut signal = MessageSignal::default();
        if !message.files.is_empty() {
            signal.has_files = true;
        }
        if let Some(blocks) = message.content_blocks.as_array() {
            signal.block_count = blocks.len();
            for block in blocks {
                if let Some(t) = block.get("type").and_then(|value| value.as_str()) {
                    match t {
                        "tool_use" | "tool_result" | "token_budget" => {
                            signal.has_tooling = true;
                            signal.tool_block_count += 1;
                        }
                        _ => {}
                    }
                }
                if let Some(citations) = block.get("citations").and_then(|value| value.as_array()) {
                    if !citations.is_empty() {
                        signal.has_citations = true;
                        signal.citation_count += citations.len();
                    }
                }
            }
        }
        signal
    }

    /// Generate a title for a Q&A note
    fn generate_qa_title(&self, question: &str, index: usize) -> String {
        // Take first line
        let first_line = question.lines().next().unwrap_or(question);

        // Remove common prefixes like "Can you", "Please", etc.
        let cleaned = first_line
            .trim_start_matches("Can you ")
            .trim_start_matches("Could you ")
            .trim_start_matches("Please ")
            .trim_start_matches("I want to ")
            .trim_start_matches("I'd like to ")
            .trim_start_matches("Help me ");

        if cleaned.is_empty() {
            return format!("Q&A #{}", index);
        }

        // Truncate at word boundary near 48 chars
        let truncated = Self::truncate_at_word_boundary(cleaned, 48);
        if truncated.len() < cleaned.len() {
            format!("{}...", truncated)
        } else {
            truncated.to_string()
        }
    }

    /// Truncate a string at the nearest word boundary at or after the target length
    fn truncate_at_word_boundary(s: &str, target: usize) -> &str {
        if s.chars().count() <= target {
            return s;
        }

        // Find word boundaries (spaces) and pick the one closest to target
        let mut last_space = 0;
        let mut char_count = 0;

        for (byte_idx, c) in s.char_indices() {
            char_count += 1;
            if c.is_whitespace() {
                if char_count > target {
                    // We've passed target, use the last space we found
                    break;
                }
                last_space = byte_idx;
            }
        }

        if last_space == 0 {
            // No space found before target, just return up to target chars
            s.char_indices()
                .nth(target)
                .map(|(idx, _)| &s[..idx])
                .unwrap_or(s)
        } else {
            &s[..last_space]
        }
    }
}

/// Result of importing chat conversations
#[derive(Debug, Default)]
pub struct ChatImportResult {
    /// Number of conversations in the parsed export payload
    pub conversations_total: usize,
    /// Number of conversations successfully imported
    pub conversations_imported: usize,
    /// Number of conversations that failed to import
    pub conversations_failed: usize,
    /// Number of conversations containing at least one message
    pub conversations_with_messages: usize,
    /// Number of conversations containing zero messages
    pub conversations_without_messages: usize,
    /// Number of zero-message conversations that still include a summary
    pub conversations_summary_only: usize,
    /// Total messages observed in the input export
    pub messages_total: usize,
    /// Total number of notes created
    pub notes_created: usize,
    /// Number of notes created from Q&A synthesis
    pub notes_from_qa: usize,
    /// Number of notes created directly from messages
    pub notes_from_messages: usize,
    /// Number of notes created from conversation summaries
    pub notes_from_summaries: usize,
    /// Number of notes created by markdown fallback chunking
    pub notes_from_fallback: usize,
    /// Q&A pairs created
    pub qa_pairs_created: usize,
    /// Q&A candidates dropped due to short question
    pub qa_pairs_dropped_short_question: usize,
    /// Q&A candidates dropped due to short answer
    pub qa_pairs_dropped_short_answer: usize,
    /// Assistant turns without a pending human question
    pub assistant_without_human: usize,
    /// Conversations ending with an unpaired human turn
    pub trailing_unpaired_human: usize,
    /// Error messages for failed imports
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
    // See tests/integration_test.rs
}
