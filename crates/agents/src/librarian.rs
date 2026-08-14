//! Librarian Agent - Ingests content and creates notes

use crate::{
    chunking::{Chunk, Chunker, ChunkingConfig, MarkdownChunker},
    classify_retry,
    inference::validate_embedding_dim,
    Result, RetryClassification, SharedEmbedder, SharedEntityExtractor,
};
use graphrag_core::{
    normalize_file_uri, normalized_content_hash, record_id_to_string, ChatConversation, ChatExport,
    ChatMessage, Entity, EntityType, MessageRole, Note, NoteType, Source, SourceType,
};
use graphrag_db::compatibility::{EmbeddingIdentity, ExtractionIdentity};
use graphrag_db::{
    ProcessingJobStatus, ProcessingJobType, ProcessingJobUpdate, Repository, SourceDeleteSummary,
    SourceImportAction,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use surrealdb::types::RecordId;
use tracing::{debug, info, instrument};

const DEFAULT_PROGRESS_EVERY: usize = 10;
const DEFAULT_PROGRESS_EVERY_SECS: u64 = 5;
const DEFAULT_EXTRACT_MAX_CHARS: usize = 8000;
const DEFAULT_MIN_CHUNK_SIZE: usize = 20;
const DEFAULT_MAX_CHUNK_SIZE: usize = usize::MAX;
const DEFAULT_ENTITY_JOB_PAGE_SIZE: usize = 100;
const EMBEDDING_JOB_WINDOW: usize = 32;
const DEFAULT_TARGET_CHUNK_SIZE: usize = 500;

/// Runtime controls for librarian ingestion and extraction.
///
/// The CLI constructs this from the resolved application configuration so the
/// agent never needs to read process environment variables directly. Defaults
/// retain the library's historical behavior for programmatic callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibrarianRuntimeConfig {
    pub min_chunk_size: usize,
    /// Target Markdown chunk size in Unicode scalar values (characters).
    pub target_chunk_size: usize,
    pub max_chunk_size: usize,
    /// Tail characters copied into a following Markdown chunk when the hard
    /// maximum permits it.
    pub chunk_overlap: usize,
    pub skip_entity_extraction: bool,
    pub extract_log_each: bool,
    /// A value of zero keeps the full note content.
    pub extract_max_chars: usize,
    pub extract_progress_every: usize,
    pub extract_progress_every_secs: u64,
    pub import_progress_every: usize,
    pub import_progress_every_secs: u64,
}

impl Default for LibrarianRuntimeConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: DEFAULT_MIN_CHUNK_SIZE,
            target_chunk_size: DEFAULT_TARGET_CHUNK_SIZE,
            max_chunk_size: DEFAULT_MAX_CHUNK_SIZE,
            chunk_overlap: 0,
            skip_entity_extraction: false,
            extract_log_each: false,
            extract_max_chars: DEFAULT_EXTRACT_MAX_CHARS,
            extract_progress_every: DEFAULT_PROGRESS_EVERY,
            extract_progress_every_secs: DEFAULT_PROGRESS_EVERY_SECS,
            import_progress_every: DEFAULT_PROGRESS_EVERY,
            import_progress_every_secs: DEFAULT_PROGRESS_EVERY_SECS,
        }
    }
}

fn ensure_batch_length(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(crate::AgentError::Processing(format!(
        "Embedding provider returned {actual} embeddings for {expected} inputs"
    )))
}

fn truncate_for_extraction(text: &str, max_chars: usize) -> String {
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

fn chunk_content(content: &str, min_chunk_size: usize, max_chunk_size: usize) -> Vec<String> {
    content
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .filter(|paragraph| paragraph.chars().count() >= min_chunk_size)
        .flat_map(|paragraph| {
            if max_chunk_size == usize::MAX || paragraph.chars().count() <= max_chunk_size {
                return vec![paragraph.to_string()];
            }

            let chars: Vec<char> = paragraph.chars().collect();
            let mut chunks = Vec::new();
            let mut start = 0;
            while start < chars.len() {
                let remaining = chars.len() - start;
                let take = if min_chunk_size <= max_chunk_size
                    && remaining > max_chunk_size
                    && remaining - max_chunk_size < min_chunk_size
                {
                    remaining - min_chunk_size
                } else {
                    remaining.min(max_chunk_size)
                };
                chunks.push(chars[start..start + take].iter().collect());
                start += take;
            }
            chunks
        })
        .collect()
}

fn note_from_markdown_chunk(chunk: &Chunk, embedding: Vec<f32>) -> Note {
    Note::new(chunk.content.clone())
        .with_type(NoteType::Raw)
        .with_embedding(embedding)
        .with_chunk_metadata(
            chunk.key.clone(),
            chunk.location_key.clone(),
            chunk.ordinal,
            chunk.heading_path.clone(),
            chunk.start_line,
            chunk.end_line,
            chunk.start_byte,
            chunk.end_byte,
            chunk.overlap_from.clone(),
            chunk.overlap_chars,
            chunk.split_fenced_code,
            chunk.content_hash.clone(),
            chunk.search_text.clone(),
        )
}

/// Content and heading context used to align two generations of a Markdown
/// source. In particular, this deliberately excludes `chunk_ordinal` and
/// `chunk_location_key`: inserting or removing a chunk changes those values
/// for every following chunk and must never redirect its dependents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MarkdownChunkAnchor {
    heading_path: Vec<String>,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MarkdownChunkMatch {
    existing_index: usize,
    staged_index: usize,
    exact_content: bool,
}

fn markdown_chunk_anchors_from_notes(notes: &[Note]) -> Vec<MarkdownChunkAnchor> {
    notes
        .iter()
        .map(|note| MarkdownChunkAnchor {
            heading_path: note.chunk_heading_path.clone(),
            content: note.content.clone(),
        })
        .collect()
}

fn markdown_chunk_anchors_from_chunks(chunks: &[Chunk]) -> Vec<MarkdownChunkAnchor> {
    chunks
        .iter()
        .map(|chunk| MarkdownChunkAnchor {
            heading_path: chunk.heading_path.clone(),
            content: chunk.content.clone(),
        })
        .collect()
}

/// Align Markdown generations using exact, content-aware anchors in document
/// order. This monotonic alignment remains stable when chunks are inserted or
/// removed before an unchanged chunk. A changed chunk is a structural
/// successor only when it is the sole unmatched chunk between two exact
/// anchors in the same heading context; ambiguous boundary/replacement cases
/// intentionally receive no successor.
fn align_markdown_chunk_sequences(
    existing: &[MarkdownChunkAnchor],
    staged: &[MarkdownChunkAnchor],
) -> Vec<MarkdownChunkMatch> {
    let mut staged_positions = HashMap::<MarkdownChunkAnchor, VecDeque<usize>>::new();
    for (index, anchor) in staged.iter().cloned().enumerate() {
        staged_positions.entry(anchor).or_default().push_back(index);
    }

    let mut exact = Vec::new();
    let mut last_staged_index = None;
    for (existing_index, anchor) in existing.iter().enumerate() {
        let Some(candidates) = staged_positions.get_mut(anchor) else {
            continue;
        };
        while candidates
            .front()
            .is_some_and(|index| last_staged_index.is_some_and(|last| *index <= last))
        {
            candidates.pop_front();
        }
        let Some(staged_index) = candidates.pop_front() else {
            continue;
        };
        last_staged_index = Some(staged_index);
        exact.push(MarkdownChunkMatch {
            existing_index,
            staged_index,
            exact_content: true,
        });
    }

    let mut matches = exact.clone();
    for anchors in exact.windows(2) {
        let [before, after] = anchors else {
            continue;
        };
        if after.existing_index != before.existing_index + 2
            || after.staged_index != before.staged_index + 2
        {
            continue;
        }
        let existing_index = before.existing_index + 1;
        let staged_index = before.staged_index + 1;
        let heading_path = &existing[existing_index].heading_path;
        if heading_path == &staged[staged_index].heading_path
            && heading_path == &existing[before.existing_index].heading_path
            && heading_path == &existing[after.existing_index].heading_path
            && heading_path == &staged[before.staged_index].heading_path
            && heading_path == &staged[after.staged_index].heading_path
        {
            matches.push(MarkdownChunkMatch {
                existing_index,
                staged_index,
                exact_content: false,
            });
        }
    }
    matches.sort_unstable_by_key(|matched| matched.staged_index);
    matches
}

/// Pair each previously successful Markdown chunk with its staged successor.
/// The boolean marks a byte-for-byte content match. Dependents that describe
/// extracted content (notably entity mentions) are copied only for those
/// exact matches; graph/provenance links can also follow a safely anchored
/// local edit.
fn markdown_chunk_successors(
    existing: &[Note],
    staged: &[Note],
) -> Vec<(RecordId, RecordId, bool)> {
    let existing_anchors = markdown_chunk_anchors_from_notes(existing);
    let staged_anchors = markdown_chunk_anchors_from_notes(staged);
    align_markdown_chunk_sequences(&existing_anchors, &staged_anchors)
        .into_iter()
        .filter_map(|matched| {
            Some((
                existing.get(matched.existing_index)?.id.clone()?,
                staged.get(matched.staged_index)?.id.clone()?,
                matched.exact_content,
            ))
        })
        .collect()
}

/// The Librarian agent handles content ingestion
pub struct LibrarianAgent {
    repo: Repository,
    embedder: SharedEmbedder,
    extractor: SharedEntityExtractor,
    runtime: LibrarianRuntimeConfig,
    cancellation_requested: Arc<AtomicBool>,
}

/// Stable, machine-readable result of a Markdown import. Counts describe the
/// lifecycle transition; `notes` contains only notes created in this attempt.
#[derive(Debug, Clone)]
pub struct MarkdownImportResult {
    pub source_id: String,
    pub source_uri: String,
    pub generation: u64,
    pub action: SourceImportAction,
    pub created: u64,
    pub unchanged: u64,
    pub updated: u64,
    pub deleted: u64,
    pub failed: u64,
    pub cleanup: SourceDeleteSummary,
    pub notes: Vec<Note>,
}

/// Stable outcome of a durable, resumable inference pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingRunResult {
    /// Empty when the requested scope had no work, so no durable job exists.
    pub job_id: String,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: bool,
}

fn no_processing_work() -> ProcessingRunResult {
    ProcessingRunResult {
        job_id: String::new(),
        completed: 0,
        failed: 0,
        cancelled: false,
    }
}

/// Cancellation before durable work is created still needs to reach callers:
/// command-line entrypoints must not report a successful zero-item run after
/// Ctrl-C was observed while resolving its scope.
fn cancelled_before_processing_work() -> ProcessingRunResult {
    ProcessingRunResult {
        cancelled: true,
        ..no_processing_work()
    }
}

fn entity_job_force_clear(scope: Option<&str>) -> bool {
    scope
        .and_then(|scope| {
            scope
                .split([';', ':'])
                .find_map(|part| part.strip_prefix("force="))
        })
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(false)
}

fn entity_job_page_size(scope: Option<&str>, fallback: usize) -> usize {
    scope
        .and_then(|scope| {
            scope
                .split(';')
                .find_map(|part| part.split_once("page_size=").map(|(_, value)| value))
        })
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&size| size > 0)
        .unwrap_or_else(|| fallback.max(1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatImportMode {
    Qa,
    Message,
    Hybrid,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatIngestOptions {
    pub persist_notes: bool,
    pub skip_notes_if_linked: bool,
}

impl Default for ChatIngestOptions {
    fn default() -> Self {
        Self {
            persist_notes: true,
            skip_notes_if_linked: false,
        }
    }
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
    conversation_records_upserted: usize,
    message_records_upserted: usize,
    note_conversation_links_created: usize,
    note_message_links_created: usize,
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

#[derive(Debug, Default)]
struct NoteCreationStats {
    notes_created: usize,
    note_conversation_links_created: usize,
    note_message_links_created: usize,
}

impl LibrarianAgent {
    /// Create a new Librarian agent
    pub fn new(
        repo: Repository,
        embedder: SharedEmbedder,
        extractor: SharedEntityExtractor,
    ) -> Self {
        Self {
            repo,
            embedder,
            extractor,
            runtime: LibrarianRuntimeConfig::default(),
            cancellation_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Supply resolved runtime controls for this agent instance.
    pub fn with_runtime_config(mut self, runtime: LibrarianRuntimeConfig) -> Self {
        self.runtime = runtime;
        self
    }

    /// The CLI installs this flag from its Ctrl-C handler. Workers check it
    /// only between atomic database item mutations, preserving an in-flight
    /// provider call and its source-generation invariants.
    pub fn with_cancellation_flag(mut self, cancellation_requested: Arc<AtomicBool>) -> Self {
        self.cancellation_requested = cancellation_requested;
        self
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed(text, false).await?;
        validate_embedding_dim(embedding.len())?;
        self.record_embedding_compatibility(embedding.len()).await?;
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embeddings = self.embedder.embed_batch(texts, false).await?;
        ensure_batch_length(texts.len(), embeddings.len())?;
        for embedding in &embeddings {
            validate_embedding_dim(embedding.len())?;
        }
        if let Some(embedding) = embeddings.first() {
            self.record_embedding_compatibility(embedding.len()).await?;
        }
        Ok(embeddings)
    }

    /// Record the identity only after the provider has successfully returned a
    /// vector.  This prevents an unavailable provider from creating metadata
    /// and makes both single and batch ingestion paths enforce the same guard.
    async fn record_embedding_compatibility(&self, dimension: usize) -> Result<()> {
        let embedding = self.embedder.capabilities();
        let extraction = self.extractor.capabilities();
        self.repo
            .record_embedding_metadata(
                &EmbeddingIdentity::new(embedding.provider, embedding.model, dimension),
                Some(&ExtractionIdentity::new(
                    extraction.provider,
                    extraction.model,
                )),
            )
            .await?;
        Ok(())
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
        let embedding = self.embed_text(&content).await?;

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
            note = note.with_source(sid);
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
        Ok(self
            .ingest_markdown_with_options(path, content, false)
            .await?
            .notes)
    }

    /// Import a Markdown file idempotently. The content hash is SHA-256 after
    /// newline normalization; an unchanged ready source is a no-op unless
    /// `force` is supplied. Refreshes stage a new generation before old notes
    /// are removed, so failed work leaves the prior generation searchable.
    #[instrument(skip(self, content))]
    pub async fn ingest_markdown_with_options<C>(
        &self,
        path: &str,
        content: C,
        force: bool,
    ) -> Result<MarkdownImportResult>
    where
        C: Into<String> + std::fmt::Debug,
    {
        let content = content.into();
        let normalized_uri = normalize_file_uri(path)
            .map_err(|error| crate::AgentError::Processing(error.to_string()))?;
        info!("Ingesting markdown from: {}", normalized_uri);
        let plan = self
            .repo
            .begin_file_import(
                SourceType::Markdown,
                path.to_string(),
                normalized_uri.clone(),
                content.clone(),
                normalized_content_hash(&content),
                force,
            )
            .await?;

        if plan.action == SourceImportAction::Unchanged {
            let cleanup = plan.cleanup;
            return Ok(MarkdownImportResult {
                source_id: plan
                    .source
                    .id
                    .as_ref()
                    .map(record_id_to_string)
                    .unwrap_or_default(),
                source_uri: normalized_uri,
                generation: plan.source.successful_generation,
                action: plan.action,
                created: 0,
                unchanged: 1,
                updated: 0,
                deleted: cleanup.notes,
                failed: 0,
                cleanup,
                notes: Vec::new(),
            });
        }

        let mut source = plan.source;
        let source_id = source.id.clone();
        let generation = source.generation;
        let existing_chunks = match source_id.as_ref() {
            Some(source_id) => self.repo.get_source_chunks(source_id).await?,
            None => Vec::new(),
        };
        let notes = match self
            .chunk_and_create_markdown_notes(
                &content,
                source_id,
                Some(generation),
                &existing_chunks,
            )
            .await
        {
            Ok(notes) => notes,
            Err(error) => {
                let message = error.to_string();
                self.repo.fail_file_import(&mut source, &message).await?;
                return Err(error);
            }
        };
        let successors = markdown_chunk_successors(&existing_chunks, &notes);
        if let Err(error) = self
            .repo
            .copy_note_dependents_to_successors(&successors)
            .await
        {
            let message = error.to_string();
            self.repo.fail_file_import(&mut source, &message).await?;
            return Err(error.into());
        }
        let cleanup = self.repo.complete_file_import(&mut source).await?;
        let created = u64::from(plan.action == SourceImportAction::Created) * notes.len() as u64;
        let updated = u64::from(plan.action == SourceImportAction::Updated) * notes.len() as u64;
        Ok(MarkdownImportResult {
            source_id: source
                .id
                .as_ref()
                .map(record_id_to_string)
                .unwrap_or_default(),
            source_uri: normalized_uri,
            generation,
            action: plan.action,
            created,
            unchanged: 0,
            updated,
            deleted: cleanup.notes,
            failed: 0,
            cleanup,
            notes,
        })
    }

    /// Reimport a file source using its stored normalized file URI. Sources
    /// with non-file URIs are intentionally rejected rather than guessing how
    /// to retrieve remote content.
    #[instrument(skip(self))]
    pub async fn reimport_markdown_source(
        &self,
        id_or_uri: &str,
        force: bool,
    ) -> Result<MarkdownImportResult> {
        let source = self.repo.get_source(id_or_uri).await?.ok_or_else(|| {
            crate::AgentError::Processing(format!("source not found: {id_or_uri}"))
        })?;
        let uri = source.normalized_uri.or(source.uri).ok_or_else(|| {
            crate::AgentError::Processing("source has no reimportable URI".into())
        })?;
        let path = file_uri_to_path(&uri)?;
        let content = std::fs::read_to_string(&path).map_err(|error| {
            crate::AgentError::Processing(format!("failed to read {}: {error}", path.display()))
        })?;
        let path = path.to_str().ok_or_else(|| {
            crate::AgentError::Processing(format!(
                "source path is not valid UTF-8 and cannot be reimported: {}",
                path.display()
            ))
        })?;
        self.ingest_markdown_with_options(path, content, force)
            .await
    }

    /// Process notes without embeddings through bounded windows. Every note is
    /// updated before its checkpoint advances; rerunning after an interruption
    /// therefore reconciles from the `embedding IS NONE` predicate instead of
    /// creating duplicate records.
    #[instrument(skip(self))]
    pub async fn process_pending_embeddings(&self) -> Result<usize> {
        let result = self.process_pending_embeddings_job(None).await?;
        Ok(usize::try_from(result.completed).unwrap_or(usize::MAX))
    }

    /// Resume a persisted embedding or entity-extraction pass. Work selection
    /// remains idempotent: pending embeddings and unlinked notes are queried
    /// fresh rather than replaying a stale in-memory queue.
    pub async fn resume_processing_job(&self, job_id: &str) -> Result<ProcessingRunResult> {
        let job = self.repo.get_processing_job(job_id).await?.ok_or_else(|| {
            crate::AgentError::Processing(format!("processing job not found: {job_id}"))
        })?;
        let id = job.id.clone().ok_or_else(|| {
            crate::AgentError::Processing(format!("processing job has no record id: {job_id}"))
        })?;
        // Validate the durable contract before changing a failed/cancelled
        // job back to `running`. This prevents malformed legacy/publicly
        // created jobs from becoming wedged in a state the worker cannot run.
        let job_type = job.job_type_enum().ok_or_else(|| {
            crate::AgentError::Processing(format!(
                "unsupported processing job type: {}",
                job.job_type
            ))
        })?;
        if job.item_ids.is_empty() {
            return Err(crate::AgentError::Processing(
                "processing job has no persisted item set; create a new job instead of resuming an ambiguous legacy job".into(),
            ));
        }
        // Failed entity passes retry their full durable scope, while a
        // cleanly cancelled pass continues after its checkpoint. Capture this
        // before the repository transition clears neither status nor counts.
        let resume_from_checkpoint = job_type == ProcessingJobType::EntityExtraction
            && job.status == ProcessingJobStatus::Cancelled.as_str()
            && job.failed_count == 0;
        let resumed = self.repo.resume_processing_job(&id).await?;
        match job_type {
            ProcessingJobType::Embedding => {
                self.process_pending_embeddings_job(Some(resumed)).await
            }
            ProcessingJobType::EntityExtraction => {
                self.extract_entities_for_notes_job(
                    100,
                    Some(resumed),
                    true,
                    resume_from_checkpoint,
                )
                .await
            }
        }
    }

    async fn process_pending_embeddings_job(
        &self,
        existing: Option<graphrag_db::ProcessingJob>,
    ) -> Result<ProcessingRunResult> {
        let (job, resumed) = match existing {
            Some(job) => {
                if job.item_ids.is_empty() {
                    return Err(crate::AgentError::Processing(
                        "embedding job has no persisted item set; create a new job instead of resuming an ambiguous legacy job".into(),
                    ));
                }
                (job, true)
            }
            None => {
                let Some(item_ids) = self
                    .pending_embedding_note_ids(EMBEDDING_JOB_WINDOW)
                    .await?
                else {
                    return Ok(no_processing_work());
                };
                if item_ids.is_empty() {
                    return Ok(no_processing_work());
                }
                let job = self
                    .repo
                    .create_processing_job_with_scope(
                        ProcessingJobType::Embedding,
                        None,
                        item_ids.len() as u64,
                        Some("missing_embeddings".to_string()),
                        item_ids,
                    )
                    .await?;
                (job, false)
            }
        };
        let job_id = job.id.clone().ok_or_else(|| {
            crate::AgentError::Processing("created processing job has no record id".into())
        })?;
        let mut completed = if resumed {
            0
        } else {
            u64::try_from(job.completed_count.max(0)).unwrap_or(0)
        };
        let mut failed = if resumed {
            0
        } else {
            u64::try_from(job.failed_count.max(0)).unwrap_or(0)
        };
        if resumed {
            self.repo
                .update_processing_job(
                    &job_id,
                    ProcessingJobUpdate {
                        completed_count: Some(0),
                        failed_count: Some(0),
                        last_error: Some(None),
                        ..Default::default()
                    },
                )
                .await?;
        }
        let mut item_ids = VecDeque::from(job.item_ids);
        // Keep the first per-item failure for the terminal job diagnostic
        // while allowing later windows to process their independently valid
        // items.
        let mut first_error = None;

        loop {
            if self.processing_job_cancelled(&job_id).await? {
                return Ok(ProcessingRunResult {
                    job_id: record_id_to_string(&job_id),
                    completed,
                    failed,
                    cancelled: true,
                });
            }
            let completed_before_reconciliation = completed;
            let failed_before_reconciliation = failed;
            let mut notes = Vec::with_capacity(EMBEDDING_JOB_WINDOW);
            let mut reconciled_items = 0;
            while notes.len() < EMBEDDING_JOB_WINDOW && reconciled_items < EMBEDDING_JOB_WINDOW {
                // A long prefix of already embedded or deleted IDs must not
                // defer Ctrl-C until the entire prefix has been scanned.
                // Persist any reconciliation completed so far before safely
                // returning the durable cancellation result.
                if self.processing_job_cancelled(&job_id).await? {
                    if completed != completed_before_reconciliation
                        || failed != failed_before_reconciliation
                    {
                        self.repo
                            .update_processing_job(
                                &job_id,
                                ProcessingJobUpdate {
                                    completed_count: Some(completed),
                                    failed_count: Some(failed),
                                    ..Default::default()
                                },
                            )
                            .await?;
                    }
                    return Ok(ProcessingRunResult {
                        job_id: record_id_to_string(&job_id),
                        completed,
                        failed,
                        cancelled: true,
                    });
                }
                let Some(item_id) = item_ids.pop_front() else {
                    break;
                };
                reconciled_items += 1;
                match self.repo.get_note(&item_id).await? {
                    Some(note) if note.has_embedding() => completed += 1,
                    Some(note) => notes.push(note),
                    None => completed += 1,
                }
            }
            // Reconciliation can consume a whole resume window when its
            // selected notes were embedded or deleted by an earlier attempt.
            // Persist those durable counts before either checking Ctrl-C or
            // starting the next inference request.
            if completed != completed_before_reconciliation
                || failed != failed_before_reconciliation
            {
                self.repo
                    .update_processing_job(
                        &job_id,
                        ProcessingJobUpdate {
                            completed_count: Some(completed),
                            failed_count: Some(failed),
                            ..Default::default()
                        },
                    )
                    .await?;
            }
            if self.processing_job_cancelled(&job_id).await? {
                return Ok(ProcessingRunResult {
                    job_id: record_id_to_string(&job_id),
                    completed,
                    failed,
                    cancelled: true,
                });
            }
            if notes.is_empty() {
                if !item_ids.is_empty() {
                    continue;
                }
                self.repo
                    .update_processing_job(
                        &job_id,
                        ProcessingJobUpdate {
                            status: Some(if failed == 0 {
                                ProcessingJobStatus::Completed
                            } else {
                                ProcessingJobStatus::Failed
                            }),
                            completed_count: Some(completed),
                            failed_count: Some(failed),
                            last_error: Some(first_error.clone()),
                            finish: true,
                            ..Default::default()
                        },
                    )
                    .await?;
                if failed > 0 {
                    return Err(crate::AgentError::Processing(first_error.unwrap_or_else(
                        || "one or more embedding items failed".to_string(),
                    )));
                }
                return Ok(ProcessingRunResult {
                    job_id: record_id_to_string(&job_id),
                    completed,
                    failed,
                    cancelled: false,
                });
            }

            let texts = notes
                .iter()
                .map(|note| note.content.clone())
                .collect::<Vec<_>>();
            match self.embed_batch(&texts).await {
                Ok(embeddings) => {
                    for (note, embedding) in notes.iter().zip(embeddings) {
                        if self.processing_job_cancelled(&job_id).await? {
                            return Ok(ProcessingRunResult {
                                job_id: record_id_to_string(&job_id),
                                completed,
                                failed,
                                cancelled: true,
                            });
                        }
                        self.checkpoint_embedding_outcome(
                            &job_id,
                            note,
                            Ok(embedding),
                            &mut completed,
                            &mut failed,
                            &mut first_error,
                        )
                        .await?;
                        if self.processing_job_cancelled(&job_id).await? {
                            return Ok(ProcessingRunResult {
                                job_id: record_id_to_string(&job_id),
                                completed,
                                failed,
                                cancelled: true,
                            });
                        }
                    }
                }
                // A rejected batch never proves that every item is invalid:
                // providers can reject a whole request because of one bad
                // input or a batch-only constraint. Fall back once per item
                // so valid notes are retained and the durable job records
                // only the actual per-item failures.
                Err(batch_error) => {
                    debug!("Embedding batch failed; falling back to per-item processing: {batch_error}");
                    for note in &notes {
                        // A cancellation that arrives while a slow fallback
                        // request is in flight still lets that request finish,
                        // then checkpoints its outcome before preventing the
                        // next request in this bounded window.
                        if self.processing_job_cancelled(&job_id).await? {
                            return Ok(ProcessingRunResult {
                                job_id: record_id_to_string(&job_id),
                                completed,
                                failed,
                                cancelled: true,
                            });
                        }
                        let outcome = self.embed_text(&note.content).await;
                        self.checkpoint_embedding_outcome(
                            &job_id,
                            note,
                            outcome,
                            &mut completed,
                            &mut failed,
                            &mut first_error,
                        )
                        .await?;
                        if self.processing_job_cancelled(&job_id).await? {
                            return Ok(ProcessingRunResult {
                                job_id: record_id_to_string(&job_id),
                                completed,
                                failed,
                                cancelled: true,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Snapshot pending embedding IDs in bounded pages before creating a
    /// durable job. A cancellation during this pre-job phase returns `None`,
    /// intentionally leaving no empty job to resume.
    async fn pending_embedding_note_ids(&self, page_size: usize) -> Result<Option<Vec<String>>> {
        if page_size == 0 || self.cancellation_requested.load(Ordering::Acquire) {
            return Ok(None);
        }
        let mut item_ids = Vec::new();
        let mut offset = 0;
        loop {
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(None);
            }
            let notes = self
                .repo
                .get_notes_without_embeddings_page(page_size, offset)
                .await?;
            // Ctrl-C can arrive while a page query is in flight. Do not retain
            // it or create a durable job after that request.
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(None);
            }
            if notes.is_empty() {
                break;
            }
            let count = notes.len();
            item_ids.extend(
                notes
                    .into_iter()
                    .filter_map(|note| note.id.as_ref().map(record_id_to_string)),
            );
            offset += count;
        }
        Ok(Some(item_ids))
    }

    /// Persist one embedding outcome before asking whether cancellation should
    /// stop a fallback window. This keeps successful work resumable even when
    /// a Ctrl-C arrives during a slow individual provider request.
    async fn checkpoint_embedding_outcome(
        &self,
        job_id: &RecordId,
        note: &Note,
        outcome: Result<Vec<f32>>,
        completed: &mut u64,
        failed: &mut u64,
        first_error: &mut Option<String>,
    ) -> Result<()> {
        let checkpoint = note.id.as_ref().map(record_id_to_string);
        match outcome {
            Ok(embedding) => {
                if let Some(id) = &note.id {
                    self.repo.update_note_embedding(id, embedding).await?;
                    *completed += 1;
                }
                self.repo
                    .update_processing_job(
                        job_id,
                        ProcessingJobUpdate {
                            completed_count: Some(*completed),
                            failed_count: Some(*failed),
                            checkpoint: Some(checkpoint),
                            last_error: Some(first_error.clone()),
                            ..Default::default()
                        },
                    )
                    .await?;
            }
            Err(error) => {
                *failed += 1;
                if first_error.is_none() {
                    *first_error = Some(error.to_string());
                }
                self.repo
                    .update_processing_job(
                        job_id,
                        ProcessingJobUpdate {
                            completed_count: Some(*completed),
                            failed_count: Some(*failed),
                            checkpoint: Some(checkpoint),
                            last_error: Some(Some(error.to_string())),
                            ..Default::default()
                        },
                    )
                    .await?;
            }
        }
        Ok(())
    }

    async fn processing_job_cancelled(&self, id: &RecordId) -> Result<bool> {
        if self.cancellation_requested.load(Ordering::Acquire) {
            self.repo.cancel_processing_job(id).await?;
            return Ok(true);
        }
        let current = self
            .repo
            .get_processing_job(&record_id_to_string(id))
            .await?
            .ok_or_else(|| crate::AgentError::Processing("processing job disappeared".into()))?;
        Ok(current.status == ProcessingJobStatus::Cancelled.as_str())
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
        if !force && self.runtime.skip_entity_extraction {
            return Ok(());
        }

        let text = truncate_for_extraction(&note.content, self.runtime.extract_max_chars);
        let extraction = self.extractor.extract(&text).await?;
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

            let mut entity = Entity::new(&extracted.name, entity_type);
            // The persisted entity schema requires an object (or NONE) for
            // metadata. Extraction supplies no metadata, so use an empty
            // object instead of `Entity::new`'s JSON null default.
            entity.metadata = serde_json::json!({});
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
        let result = self.extract_entities_for_notes_result(limit).await?;
        Ok(usize::try_from(result.completed).unwrap_or(usize::MAX))
    }

    /// Extract entities for notes missing entity links and retain the durable
    /// run state so callers can distinguish cancellation from zero work.
    #[instrument(skip(self))]
    pub async fn extract_entities_for_notes_result(
        &self,
        limit: usize,
    ) -> Result<ProcessingRunResult> {
        self.extract_entities_for_notes_job(limit, None, false, false)
            .await
    }

    async fn extract_entities_for_notes_job(
        &self,
        limit: usize,
        existing: Option<graphrag_db::ProcessingJob>,
        resumed: bool,
        resume_from_checkpoint: bool,
    ) -> Result<ProcessingRunResult> {
        let (job, page_size, force_clear) = match existing {
            Some(job) => {
                if job.item_ids.is_empty() {
                    return Err(crate::AgentError::Processing(
                        "entity extraction job has no persisted item set; create a new job instead of resuming an ambiguous legacy job".into(),
                    ));
                }
                let page_size =
                    entity_job_page_size(job.scope.as_deref(), DEFAULT_ENTITY_JOB_PAGE_SIZE);
                let force_clear = entity_job_force_clear(job.scope.as_deref());
                (job, page_size, force_clear)
            }
            None => {
                let notes = self.repo.get_notes_without_entities(limit).await?;
                let item_ids = notes
                    .iter()
                    .filter_map(|note| note.id.as_ref().map(record_id_to_string))
                    .collect::<Vec<_>>();
                if item_ids.is_empty() {
                    return Ok(no_processing_work());
                }
                let job = self
                    .repo
                    .create_processing_job_with_scope(
                        ProcessingJobType::EntityExtraction,
                        None,
                        item_ids.len() as u64,
                        Some(format!("missing_entities:limit={limit}")),
                        item_ids,
                    )
                    .await?;
                (job, limit.max(1), false)
            }
        };
        let job_id = job.id.clone().ok_or_else(|| {
            crate::AgentError::Processing("created processing job has no record id".into())
        })?;
        let item_ids = job.item_ids.clone();
        // A cleanly cancelled entity job has checkpointed each committed item.
        // Continue after that durable checkpoint so `--force` never clears or
        // relinks an already completed note on resume. A failed job instead
        // retries its complete persisted scope after provider repair.
        let checkpoint_index = resume_from_checkpoint
            .then(|| {
                job.checkpoint.as_ref().and_then(|checkpoint| {
                    item_ids.iter().position(|item_id| item_id == checkpoint)
                })
            })
            .flatten();
        let start_index = checkpoint_index.map_or(0, |index| index + 1);
        let restore_checkpoint_counts = checkpoint_index.is_some();
        let mut completed = if restore_checkpoint_counts {
            u64::try_from(job.completed_count.max(0)).unwrap_or(0)
        } else if resumed {
            0
        } else {
            u64::try_from(job.completed_count.max(0)).unwrap_or(0)
        };
        let mut failed = if restore_checkpoint_counts {
            u64::try_from(job.failed_count.max(0)).unwrap_or(0)
        } else if resumed {
            0
        } else {
            u64::try_from(job.failed_count.max(0)).unwrap_or(0)
        };
        if resumed {
            self.repo
                .update_processing_job(
                    &job_id,
                    ProcessingJobUpdate {
                        // A checkpoint resume carries forward the durable
                        // completed prefix. Persist it before looking for a
                        // new cancellation so a second Ctrl-C cannot erase
                        // work that is intentionally skipped on retry.
                        completed_count: Some(completed),
                        failed_count: Some(failed),
                        // A failed job retries its complete item set. Clear
                        // the old checkpoint before checking cancellation so
                        // an immediate second stop cannot later mistake that
                        // stale cursor for a completed prefix.
                        checkpoint: (!resume_from_checkpoint).then_some(None),
                        last_error: Some(None),
                        ..Default::default()
                    },
                )
                .await?;
        }

        let progress_every = self.runtime.extract_progress_every;
        let progress_every_secs = self.runtime.extract_progress_every_secs;
        let log_each = self.runtime.extract_log_each;

        let total = item_ids.len();
        let start = Instant::now();
        let mut last_progress = Instant::now();

        let mut first_error = None;
        for (window_index, window) in item_ids[start_index..].chunks(page_size).enumerate() {
            // Item IDs are durable, but note payloads are loaded only for the
            // current bounded window. This keeps `extract-entities --all`
            // memory bounded and observes cancellation before fetching the
            // next page of notes.
            if self.processing_job_cancelled(&job_id).await? {
                return Ok(ProcessingRunResult {
                    job_id: record_id_to_string(&job_id),
                    completed,
                    failed,
                    cancelled: true,
                });
            }
            for (item_index, item_id) in window.iter().enumerate() {
                // A provider call can request cancellation while this page is
                // in flight. Check again after each checkpointed item rather
                // than starting the next provider call in the same page.
                if self.processing_job_cancelled(&job_id).await? {
                    return Ok(ProcessingRunResult {
                        job_id: record_id_to_string(&job_id),
                        completed,
                        failed,
                        cancelled: true,
                    });
                }
                let index = start_index + window_index * page_size + item_index;
                let Some(note) = self.repo.get_note(item_id).await? else {
                    // A deleted item is terminally skipped: it cannot be
                    // extracted later, and retrying it would keep an
                    // otherwise healthy durable job permanently failed.
                    completed += 1;
                    debug!("Selected note was deleted before processing: {item_id}");
                    self.repo
                        .update_processing_job(
                            &job_id,
                            ProcessingJobUpdate {
                                completed_count: Some(completed),
                                failed_count: Some(failed),
                                checkpoint: Some(Some(item_id.clone())),
                                last_error: Some(first_error.clone()),
                                ..Default::default()
                            },
                        )
                        .await?;
                    continue;
                };
                let note_id = note
                    .id
                    .as_ref()
                    .map(record_id_to_string)
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
                if force_clear {
                    if let Some(id) = &note.id {
                        self.repo.delete_mentions_for_note(id).await?;
                    }
                }
                let item_error = match self.extract_and_link_entities_force(&note).await {
                    Ok(()) => {
                        completed += 1;
                        if log_each {
                            info!(
                                "Entity extraction done: {}/{} note_id={} elapsed={:.2}s",
                                index + 1,
                                total,
                                note_id,
                                note_start.elapsed().as_secs_f32()
                            );
                        }
                        None
                    }
                    Err(e) => {
                        failed += 1;
                        if first_error.is_none() {
                            first_error = Some(e.to_string());
                        }
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
                        Some(e.to_string())
                    }
                };

                self.repo
                    .update_processing_job(
                        &job_id,
                        ProcessingJobUpdate {
                            completed_count: Some(completed),
                            failed_count: Some(failed),
                            checkpoint: Some(note.id.as_ref().map(record_id_to_string)),
                            last_error: Some(item_error.or_else(|| first_error.clone())),
                            ..Default::default()
                        },
                    )
                    .await?;

                if completed as usize % progress_every == 0
                    || last_progress.elapsed() >= Duration::from_secs(progress_every_secs)
                {
                    let elapsed = start.elapsed().as_secs_f32().max(0.001);
                    let rate = completed as f32 / elapsed;
                    let remaining = total.saturating_sub(completed as usize);
                    let eta_secs = if rate > 0.0 {
                        (remaining as f32 / rate).round() as u64
                    } else {
                        0
                    };
                    info!(
                        "Entity extraction progress: {}/{} notes (rate: {:.2}/s, eta: {}s)",
                        completed, total, rate, eta_secs
                    );
                    last_progress = Instant::now();
                }
            }
        }

        self.repo
            .update_processing_job(
                &job_id,
                ProcessingJobUpdate {
                    status: Some(if failed == 0 {
                        ProcessingJobStatus::Completed
                    } else {
                        ProcessingJobStatus::Failed
                    }),
                    completed_count: Some(completed),
                    failed_count: Some(failed),
                    finish: true,
                    ..Default::default()
                },
            )
            .await?;
        // Match embedding jobs: a persisted terminal failure is still an
        // invocation failure. The durable row above retains the detailed
        // failed count and first item diagnostic for `jobs show`/retry.
        if failed > 0 {
            return Err(crate::AgentError::Processing(first_error.unwrap_or_else(
                || "one or more entity extraction items failed".to_string(),
            )));
        }
        Ok(ProcessingRunResult {
            job_id: record_id_to_string(&job_id),
            completed,
            failed,
            cancelled: false,
        })
    }

    async fn create_entity_extraction_job(
        &self,
        item_ids: Vec<String>,
        scope: String,
    ) -> Result<ProcessingRunResult> {
        if item_ids.is_empty() {
            return Ok(no_processing_work());
        }
        let job = self
            .repo
            .create_processing_job_with_scope(
                ProcessingJobType::EntityExtraction,
                None,
                item_ids.len() as u64,
                Some(scope),
                item_ids,
            )
            .await?;
        // Reuse the persisted-item executor while keeping newly created jobs
        // distinct from a real checkpoint resume.
        self.extract_entities_for_notes_job(0, Some(job), false, false)
            .await
    }

    /// Return the durable item set for `extract-entities --all`. A
    /// cancellation before the job exists returns `None`: no work has been
    /// committed yet, so there is intentionally no empty/cancelled job to
    /// resume.
    async fn all_note_ids_for_entity_extraction(
        &self,
        page_size: usize,
    ) -> Result<Option<Vec<String>>> {
        if page_size == 0 || self.cancellation_requested.load(Ordering::Acquire) {
            return Ok(None);
        }
        let mut item_ids = Vec::new();
        let mut offset = 0;
        loop {
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(None);
            }
            let notes = self.repo.get_notes_page(page_size, offset).await?;
            // The query may have been in flight when Ctrl-C arrived. Check
            // before retaining this page or creating a durable job.
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(None);
            }
            if notes.is_empty() {
                break;
            }
            let count = notes.len();
            item_ids.extend(
                notes
                    .into_iter()
                    .filter_map(|note| note.id.as_ref().map(record_id_to_string)),
            );
            offset += count;
        }
        Ok(Some(item_ids))
    }

    /// Extract entities for all notes (optionally clearing existing mentions first)
    #[instrument(skip(self))]
    pub async fn extract_entities_for_all_notes(
        &self,
        limit: usize,
        force_clear: bool,
    ) -> Result<usize> {
        let result = self
            .extract_entities_for_all_notes_result(limit, force_clear)
            .await?;
        Ok(usize::try_from(result.completed).unwrap_or(usize::MAX))
    }

    /// Extract entities for every note while retaining durable cancellation
    /// state for callers such as the CLI.
    #[instrument(skip(self))]
    pub async fn extract_entities_for_all_notes_result(
        &self,
        limit: usize,
        force_clear: bool,
    ) -> Result<ProcessingRunResult> {
        let Some(item_ids) = self.all_note_ids_for_entity_extraction(limit).await? else {
            return Ok(if self.cancellation_requested.load(Ordering::Acquire) {
                cancelled_before_processing_work()
            } else {
                no_processing_work()
            });
        };
        self.create_entity_extraction_job(
            item_ids,
            format!("all_notes:page_size={limit};force={force_clear}"),
        )
        .await
    }

    /// Extract entities for explicit note ids
    #[instrument(skip(self, note_ids))]
    pub async fn extract_entities_for_note_ids(
        &self,
        note_ids: &[String],
        force_clear: bool,
    ) -> Result<usize> {
        let result = self
            .extract_entities_for_note_ids_result(note_ids, force_clear)
            .await?;
        Ok(usize::try_from(result.completed).unwrap_or(usize::MAX))
    }

    /// Extract entities for explicit note ids while retaining durable
    /// cancellation state for callers such as the CLI.
    #[instrument(skip(self, note_ids))]
    pub async fn extract_entities_for_note_ids_result(
        &self,
        note_ids: &[String],
        force_clear: bool,
    ) -> Result<ProcessingRunResult> {
        // Resolving explicit IDs happens before a durable job exists. A
        // cancellation in this phase intentionally creates no empty job to
        // resume, matching `extract-entities --all` snapshot semantics.
        if self.cancellation_requested.load(Ordering::Acquire) {
            return Ok(cancelled_before_processing_work());
        }
        let mut item_ids = Vec::new();
        for note_id_raw in note_ids {
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(cancelled_before_processing_work());
            }
            let key = note_id_raw.strip_prefix("note:").unwrap_or(note_id_raw);
            let maybe_note = self.repo.get_note(key).await?;
            // A lookup may have been in flight while Ctrl-C arrived. Do not
            // retain its result or create a durable job after cancellation.
            if self.cancellation_requested.load(Ordering::Acquire) {
                return Ok(cancelled_before_processing_work());
            }
            let Some(note) = maybe_note else {
                debug!("Note not found for extraction: {}", note_id_raw);
                continue;
            };
            if let Some(id) = note.id.as_ref().map(record_id_to_string) {
                if !item_ids.contains(&id) {
                    item_ids.push(id);
                }
            }
        }
        self.create_entity_extraction_job(item_ids, format!("note_ids:force={force_clear}"))
            .await
    }

    /// Chunk content and create notes
    async fn chunk_and_create_notes(
        &self,
        content: &str,
        source_id: Option<RecordId>,
        source_generation: Option<u64>,
    ) -> Result<Vec<Note>> {
        // Split by paragraphs and apply the resolved size limits. Long
        // paragraphs are bounded by characters to keep embedding requests
        // predictable without changing the retrieval algorithm.
        let chunks = chunk_content(
            content,
            self.runtime.min_chunk_size,
            self.runtime.max_chunk_size,
        );

        if chunks.is_empty() {
            // Treat whole content as one note
            let embedding = self.embed_text(content).await?;
            let mut note = Note::new(content)
                .with_type(NoteType::Raw)
                .with_embedding(embedding);

            if let Some(ref sid) = source_id {
                note = note.with_source(sid.clone());
            }
            if let Some(generation) = source_generation {
                note = note.with_source_generation(generation);
            }

            let note = self.repo.create_note(note).await?;
            return Ok(vec![note]);
        }

        // Generate embeddings in batch
        let embeddings = self.embed_batch(&chunks).await?;

        let mut notes = Vec::new();

        for (chunk, embedding) in chunks.iter().zip(embeddings.into_iter()) {
            let mut note = Note::new(chunk.clone())
                .with_type(NoteType::Raw)
                .with_embedding(embedding);

            if let Some(ref sid) = source_id {
                note = note.with_source(sid.clone());
            }
            if let Some(generation) = source_generation {
                note = note.with_source_generation(generation);
            }

            let note = self.repo.create_note(note).await?;
            notes.push(note);
        }

        Ok(notes)
    }

    /// Chunk Markdown with structural metadata, then reconcile it with the
    /// previous successful source generation. Exact key/hash matches reuse
    /// embeddings; each pending generation is written copy-on-write so a
    /// failed refresh cannot mutate or hide the last successful corpus.
    /// `complete_file_import` then promotes and safely cascades the prior
    /// generation.
    async fn chunk_and_create_markdown_notes(
        &self,
        content: &str,
        source_id: Option<RecordId>,
        source_generation: Option<u64>,
        existing_chunks: &[Note],
    ) -> Result<Vec<Note>> {
        let Some(source_id) = source_id else {
            return self
                .chunk_and_create_notes(content, None, source_generation)
                .await;
        };
        let chunker = MarkdownChunker::new(ChunkingConfig {
            min_size: self.runtime.min_chunk_size,
            target_size: self.runtime.target_chunk_size,
            max_size: self.runtime.max_chunk_size,
            overlap_size: self.runtime.chunk_overlap,
        })
        .map_err(|error| crate::AgentError::Processing(error.to_string()))?;
        let source_identity = record_id_to_string(&source_id);
        let chunks = chunker
            .chunk(&source_identity, content)
            .map_err(|error| crate::AgentError::Processing(error.to_string()))?;

        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let existing_anchors = markdown_chunk_anchors_from_notes(existing_chunks);
        let staged_anchors = markdown_chunk_anchors_from_chunks(&chunks);
        let matches_by_staged = align_markdown_chunk_sequences(&existing_anchors, &staged_anchors)
            .into_iter()
            .map(|matched| (matched.staged_index, matched))
            .collect::<HashMap<_, _>>();

        // Generate every required embedding before mutating any prior record.
        // This keeps a provider failure from erasing the old visible source
        // generation during reconciliation.
        let changed = chunks
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                !matches_by_staged
                    .get(index)
                    .is_some_and(|matched| matched.exact_content)
            })
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();
        let embedding_inputs = changed
            .iter()
            .map(|chunk| chunk.search_text.clone())
            .collect::<Vec<_>>();
        let embeddings = if embedding_inputs.is_empty() {
            Vec::new()
        } else {
            self.embed_batch(&embedding_inputs).await?
        };
        let mut embedding_by_key = changed
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| (chunk.key.clone(), embedding))
            .collect::<HashMap<_, _>>();

        let mut notes = Vec::with_capacity(chunks.len());
        for (staged_index, chunk) in chunks.into_iter().enumerate() {
            let matched_existing = matches_by_staged
                .get(&staged_index)
                .and_then(|matched| existing_chunks.get(matched.existing_index));
            let is_exact_match = matches_by_staged
                .get(&staged_index)
                .is_some_and(|matched| matched.exact_content);
            let embedding = if !is_exact_match {
                embedding_by_key.remove(&chunk.key).ok_or_else(|| {
                    crate::AgentError::Processing(
                        "missing prepared Markdown chunk embedding".into(),
                    )
                })?
            } else {
                matched_existing
                    .expect("exact match has an existing chunk")
                    .embedding
                    .clone()
            };
            let mut note =
                note_from_markdown_chunk(&chunk, embedding).with_source(source_id.clone());
            if let Some(generation) = source_generation {
                note = note.with_source_generation(generation);
            }
            // Keep the creation time for a structural successor. Even though
            // staged generations use a fresh record ID for crash-safe
            // copy-on-write, creation ordering and `since` search semantics
            // remain stable for unchanged/reconciled chunks.
            if let Some(existing) = matched_existing {
                note.created_at = existing.created_at;
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
        self.ingest_chat_export_with_options(export, source_uri, mode, ChatIngestOptions::default())
            .await
    }

    #[instrument(skip(self, export))]
    pub async fn ingest_chat_export_with_options(
        &self,
        export: ChatExport,
        source_uri: Option<String>,
        mode: ChatImportMode,
        options: ChatIngestOptions,
    ) -> Result<ChatImportResult> {
        let total = export.conversation_count();
        let progress_every = self.runtime.import_progress_every;
        let progress_every_secs = self.runtime.import_progress_every_secs;
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
                .ingest_conversation(&conversation, source_uri.clone(), mode, options)
                .await
            {
                Ok(outcome) => {
                    result.conversations_imported += 1;
                    result.notes_created += outcome.notes_created;
                    result.notes_from_qa += outcome.notes_from_qa;
                    result.notes_from_messages += outcome.notes_from_messages;
                    result.notes_from_summaries += outcome.notes_from_summaries;
                    result.notes_from_fallback += outcome.notes_from_fallback;
                    result.conversation_records_upserted += outcome.conversation_records_upserted;
                    result.message_records_upserted += outcome.message_records_upserted;
                    result.note_conversation_links_created +=
                        outcome.note_conversation_links_created;
                    result.note_message_links_created += outcome.note_message_links_created;
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

    /// Backfill conversation/message records without generating derived notes.
    #[instrument(skip(self, export))]
    pub async fn backfill_chat_export_records(
        &self,
        export: ChatExport,
        source_uri: Option<String>,
    ) -> Result<ChatImportResult> {
        self.ingest_chat_export_with_options(
            export,
            source_uri,
            ChatImportMode::Qa,
            ChatIngestOptions {
                persist_notes: false,
                skip_notes_if_linked: true,
            },
        )
        .await
    }

    /// Preview import/backfill counts without writing to the database.
    pub fn preview_chat_export(
        export: &ChatExport,
        mode: ChatImportMode,
        include_notes: bool,
    ) -> ChatImportPreview {
        let mut preview = ChatImportPreview {
            conversations_total: export.conversation_count(),
            messages_total: export.total_messages(),
            ..Default::default()
        };

        for conversation in &export.conversations {
            if conversation.messages.is_empty() {
                preview.conversations_without_messages += 1;
                if !conversation.summary.is_empty() {
                    preview.summary_only_conversations += 1;
                }
            } else {
                preview.conversations_with_messages += 1;
            }

            if !conversation.summary.is_empty() {
                preview.summary_notes += 1;
            }

            let qa = Self::extract_qa_pairs_with_diagnostics(&conversation.messages);
            preview.qa_pairs += qa.pairs.len();
            preview.qa_pairs_dropped_short_question += qa.dropped_short_question;
            preview.qa_pairs_dropped_short_answer += qa.dropped_short_answer;
            preview.assistant_without_human += qa.assistant_without_human;
            preview.trailing_unpaired_human += qa.trailing_unpaired_human;

            if !include_notes {
                continue;
            }

            let fallback_chunks = Self::estimate_fallback_chunks(conversation);
            match mode {
                ChatImportMode::Qa => {
                    if qa.pairs.is_empty() {
                        preview.notes_from_fallback += fallback_chunks;
                    } else {
                        preview.notes_from_qa += qa.pairs.len();
                    }
                }
                ChatImportMode::Message => {
                    if conversation.messages.is_empty() {
                        preview.notes_from_fallback += fallback_chunks;
                    } else {
                        preview.notes_from_messages += conversation.messages.len();
                    }
                }
                ChatImportMode::Hybrid => {
                    preview.notes_from_qa += qa.pairs.len();
                    let paired_indices: HashSet<usize> = qa
                        .pairs
                        .iter()
                        .flat_map(|pair| [pair.human_idx, pair.assistant_idx])
                        .collect();
                    let selected = conversation
                        .messages
                        .iter()
                        .enumerate()
                        .filter(|(idx, msg)| {
                            !paired_indices.contains(idx) || Self::message_has_extra_signal(msg)
                        })
                        .count();
                    if qa.pairs.is_empty() && selected == 0 {
                        preview.notes_from_fallback += fallback_chunks;
                    } else {
                        preview.notes_from_messages += selected;
                    }
                }
            }
        }

        if include_notes {
            preview.notes_from_summaries = preview.summary_notes;
            preview.estimated_notes = preview.notes_from_qa
                + preview.notes_from_messages
                + preview.notes_from_summaries
                + preview.notes_from_fallback;
        }

        preview
    }

    fn estimate_fallback_chunks(conversation: &ChatConversation) -> usize {
        conversation
            .to_markdown()
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 20)
            .count()
    }

    /// Ingest a single conversation
    #[instrument(skip(self, conversation))]
    async fn ingest_conversation(
        &self,
        conversation: &ChatConversation,
        source_uri: Option<String>,
        mode: ChatImportMode,
        options: ChatIngestOptions,
    ) -> Result<ConversationImportOutcome> {
        let title = conversation.display_title();
        info!("Ingesting conversation: {}", title);
        let source_uri_for_source = source_uri.clone();

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
        let metadata_value = serde_json::Value::Object(metadata);
        let mut outcome = ConversationImportOutcome::default();

        let summary_embedding = if conversation.summary.is_empty() {
            None
        } else {
            let summary_text = format!(
                "{}\n\n{}",
                conversation.display_title(),
                conversation.summary
            );
            Some(self.embed_text(&summary_text).await?)
        };

        let conversation_record_id = self
            .repo
            .upsert_conversation(conversation, source_uri, metadata_value, summary_embedding)
            .await?;
        outcome.conversation_records_upserted = 1;

        let (message_embeddings, expected_embeddings) = if conversation.messages.is_empty() {
            (Vec::new(), 0)
        } else {
            let message_texts: Vec<String> = conversation
                .messages
                .iter()
                .map(Self::message_text_for_embedding)
                .collect();
            let expected = message_texts.len();
            (self.embed_batch(&message_texts).await?, expected)
        };
        ensure_batch_length(expected_embeddings, message_embeddings.len())?;

        let mut message_record_ids = Vec::with_capacity(conversation.messages.len());
        for (idx, message) in conversation.messages.iter().enumerate() {
            let embedding = message_embeddings.get(idx).cloned();
            let message_id = self
                .repo
                .upsert_message(
                    &conversation_record_id,
                    &conversation.uuid,
                    idx,
                    message,
                    embedding,
                )
                .await?;
            message_record_ids.push(message_id);
        }
        outcome.message_records_upserted = message_record_ids.len();

        if !options.persist_notes {
            return Ok(outcome);
        }

        if options.skip_notes_if_linked
            && self
                .repo
                .conversation_has_note_links(&conversation_record_id)
                .await?
        {
            return Ok(outcome);
        }

        // Create source record for this conversation only when generating notes.
        let mut source = Source::chat_export(&title, source_uri_for_source);
        source = source.with_metadata(serde_json::json!({
            "conversation_id": &conversation.uuid,
            "created_at": &conversation.created_at,
            "summary": &conversation.summary,
        }));
        let source = self.repo.create_source(source).await?;
        let source_id = source.id.clone();

        if !conversation.summary.is_empty() {
            let summary_stats = self
                .create_summary_note(conversation, source_id.clone(), &conversation_record_id)
                .await?;
            outcome.notes_created += summary_stats.notes_created;
            outcome.notes_from_summaries += summary_stats.notes_created;
            outcome.note_conversation_links_created +=
                summary_stats.note_conversation_links_created;
            outcome.note_message_links_created += summary_stats.note_message_links_created;
        }

        let qa = Self::extract_qa_pairs_with_diagnostics(&conversation.messages);
        outcome.qa_pairs_created = qa.pairs.len();
        outcome.qa_pairs_dropped_short_question = qa.dropped_short_question;
        outcome.qa_pairs_dropped_short_answer = qa.dropped_short_answer;
        outcome.assistant_without_human = qa.assistant_without_human;
        outcome.trailing_unpaired_human = qa.trailing_unpaired_human;

        match mode {
            ChatImportMode::Qa => {
                if qa.pairs.is_empty() {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self
                        .chunk_and_create_notes(&markdown, source_id, None)
                        .await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                    outcome.note_conversation_links_created += self
                        .link_notes_to_conversation(&fallback_notes, &conversation_record_id)
                        .await?;
                } else {
                    let stats = self
                        .create_qa_notes(
                            &qa.pairs,
                            source_id,
                            &title,
                            &conversation_record_id,
                            &message_record_ids,
                        )
                        .await?;
                    outcome.notes_created += stats.notes_created;
                    outcome.notes_from_qa += stats.notes_created;
                    outcome.note_conversation_links_created +=
                        stats.note_conversation_links_created;
                    outcome.note_message_links_created += stats.note_message_links_created;
                }
            }
            ChatImportMode::Message => {
                if conversation.messages.is_empty() {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self
                        .chunk_and_create_notes(&markdown, source_id, None)
                        .await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                    outcome.note_conversation_links_created += self
                        .link_notes_to_conversation(&fallback_notes, &conversation_record_id)
                        .await?;
                } else {
                    let selection: Vec<usize> = (0..conversation.messages.len()).collect();
                    let stats = self
                        .create_message_notes(
                            conversation,
                            &conversation.messages,
                            &selection,
                            source_id,
                            &conversation_record_id,
                            &message_record_ids,
                        )
                        .await?;
                    outcome.notes_created += stats.notes_created;
                    outcome.notes_from_messages += stats.notes_created;
                    outcome.note_conversation_links_created +=
                        stats.note_conversation_links_created;
                    outcome.note_message_links_created += stats.note_message_links_created;
                }
            }
            ChatImportMode::Hybrid => {
                if !qa.pairs.is_empty() {
                    let stats = self
                        .create_qa_notes(
                            &qa.pairs,
                            source_id.clone(),
                            &title,
                            &conversation_record_id,
                            &message_record_ids,
                        )
                        .await?;
                    outcome.notes_created += stats.notes_created;
                    outcome.notes_from_qa += stats.notes_created;
                    outcome.note_conversation_links_created +=
                        stats.note_conversation_links_created;
                    outcome.note_message_links_created += stats.note_message_links_created;
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
                            &conversation_record_id,
                            &message_record_ids,
                        )
                        .await?;
                    outcome.notes_created += created.notes_created;
                    outcome.notes_from_messages += created.notes_created;
                    outcome.note_conversation_links_created +=
                        created.note_conversation_links_created;
                    outcome.note_message_links_created += created.note_message_links_created;
                }

                if outcome.notes_from_qa == 0 && outcome.notes_from_messages == 0 {
                    let markdown = conversation.to_markdown();
                    let fallback_notes = self
                        .chunk_and_create_notes(&markdown, source_id, None)
                        .await?;
                    outcome.notes_created += fallback_notes.len();
                    outcome.notes_from_fallback += fallback_notes.len();
                    outcome.note_conversation_links_created += self
                        .link_notes_to_conversation(&fallback_notes, &conversation_record_id)
                        .await?;
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
        source_id: Option<RecordId>,
        conversation_title: &str,
        conversation_record_id: &surrealdb::types::RecordId,
        message_record_ids: &[surrealdb::types::RecordId],
    ) -> Result<NoteCreationStats> {
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
        let embeddings = self.embed_batch(&texts_to_embed).await?;

        // Create notes
        let mut stats = NoteCreationStats::default();
        for (idx, ((content, title), embedding)) in note_builders
            .into_iter()
            .zip(embeddings.into_iter())
            .enumerate()
        {
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

            stats.notes_created += 1;
            stats.note_conversation_links_created += self
                .link_note_to_conversation(&note, conversation_record_id)
                .await?;

            if let Some(message_id) = message_record_ids.get(qa_pairs[idx].human_idx) {
                stats.note_message_links_created +=
                    self.link_note_to_message(&note, message_id).await?;
            }
            if let Some(message_id) = message_record_ids.get(qa_pairs[idx].assistant_idx) {
                stats.note_message_links_created +=
                    self.link_note_to_message(&note, message_id).await?;
            }
        }

        Ok(stats)
    }

    async fn create_summary_note(
        &self,
        conversation: &ChatConversation,
        source_id: Option<RecordId>,
        conversation_record_id: &surrealdb::types::RecordId,
    ) -> Result<NoteCreationStats> {
        let summary_title = format!("Summary: {}", conversation.display_title());
        let summary_content = format!(
            "**Conversation:** {}\n\n{}",
            conversation.display_title(),
            conversation.summary
        );
        let embedding = self.embed_text(&summary_content).await?;
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

        let mut stats = NoteCreationStats {
            notes_created: 1,
            ..Default::default()
        };
        stats.note_conversation_links_created += self
            .link_note_to_conversation(&note, conversation_record_id)
            .await?;

        Ok(stats)
    }

    async fn create_message_notes(
        &self,
        conversation: &ChatConversation,
        messages: &[ChatMessage],
        indices: &[usize],
        source_id: Option<RecordId>,
        conversation_record_id: &surrealdb::types::RecordId,
        message_record_ids: &[surrealdb::types::RecordId],
    ) -> Result<NoteCreationStats> {
        if indices.is_empty() {
            return Ok(NoteCreationStats::default());
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
                builders.push((content, title, tags, *idx));
            }
        }

        let embeddings = self.embed_batch(&texts_to_embed).await?;
        let mut stats = NoteCreationStats::default();

        for ((content, title, tags, message_idx), embedding) in
            builders.into_iter().zip(embeddings.into_iter())
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
            stats.notes_created += 1;
            stats.note_conversation_links_created += self
                .link_note_to_conversation(&note, conversation_record_id)
                .await?;
            if let Some(message_id) = message_record_ids.get(message_idx) {
                stats.note_message_links_created +=
                    self.link_note_to_message(&note, message_id).await?;
            }
        }

        Ok(stats)
    }

    async fn link_notes_to_conversation(
        &self,
        notes: &[Note],
        conversation_record_id: &surrealdb::types::RecordId,
    ) -> Result<usize> {
        let mut created = 0usize;
        for note in notes {
            created += self
                .link_note_to_conversation(note, conversation_record_id)
                .await?;
        }
        Ok(created)
    }

    async fn link_note_to_conversation(
        &self,
        note: &Note,
        conversation_record_id: &surrealdb::types::RecordId,
    ) -> Result<usize> {
        if let Some(note_id) = &note.id {
            let linked = self
                .repo
                .link_note_to_conversation(note_id, conversation_record_id)
                .await?;
            return Ok(if linked { 1 } else { 0 });
        }
        Ok(0)
    }

    async fn link_note_to_message(
        &self,
        note: &Note,
        message_record_id: &surrealdb::types::RecordId,
    ) -> Result<usize> {
        if let Some(note_id) = &note.id {
            let linked = self
                .repo
                .link_note_to_message(note_id, message_record_id)
                .await?;
            return Ok(if linked { 1 } else { 0 });
        }
        Ok(0)
    }

    /// Extract Q&A pairs from a list of messages and collect diagnostics
    fn extract_qa_pairs_with_diagnostics(messages: &[ChatMessage]) -> QaExtractionResult {
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

    fn message_text_for_embedding(message: &ChatMessage) -> String {
        if !message.content.trim().is_empty() {
            return message.content.clone();
        }

        if let Some(blocks) = message.content_blocks.as_array() {
            let mut parts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    if !text.trim().is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
            if !parts.is_empty() {
                return parts.join("\n\n");
            }
        }

        "[empty message]".to_string()
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

fn file_uri_to_path(uri: &str) -> Result<PathBuf> {
    let path = decode_file_uri(uri, cfg!(windows))?;
    Ok(PathBuf::from(path))
}

/// Decode GraphRAG's normalized local-file URI format. Windows drive letters
/// need special handling: `file:///C:/notes.md` maps to `C:/notes.md`, while
/// Unix absolute paths keep their leading slash. The helper is parameterized
/// for deterministic cross-platform tests.
fn decode_file_uri(uri: &str, windows: bool) -> Result<String> {
    let path = uri.strip_prefix("file://").ok_or_else(|| {
        crate::AgentError::Processing(format!("source is not a local file: {uri}"))
    })?;
    if path.is_empty() {
        return Err(crate::AgentError::Processing(
            "source file URI has no path".into(),
        ));
    }

    if windows {
        let bytes = path.as_bytes();
        if bytes.len() >= 3
            && bytes[0] == b'/'
            && bytes[1].is_ascii_alphabetic()
            && bytes[2] == b':'
        {
            return Ok(path[1..].to_string());
        }
        // `file://server/share/path` is the URI form for a Windows UNC path.
        // Reconstruct its two leading separators before handing it to PathBuf.
        if !path.starts_with('/') {
            return Ok(format!("//{path}"));
        }
    }
    Ok(path.to_string())
}

/// Result of importing chat conversations
#[derive(Debug, Default)]
pub struct ChatImportPreview {
    pub conversations_total: usize,
    pub conversations_with_messages: usize,
    pub conversations_without_messages: usize,
    pub summary_only_conversations: usize,
    pub messages_total: usize,
    pub summary_notes: usize,
    pub qa_pairs: usize,
    pub qa_pairs_dropped_short_question: usize,
    pub qa_pairs_dropped_short_answer: usize,
    pub assistant_without_human: usize,
    pub trailing_unpaired_human: usize,
    pub notes_from_qa: usize,
    pub notes_from_messages: usize,
    pub notes_from_summaries: usize,
    pub notes_from_fallback: usize,
    pub estimated_notes: usize,
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
    /// Conversation records upserted into the conversation table
    pub conversation_records_upserted: usize,
    /// Message records upserted into the message table
    pub message_records_upserted: usize,
    /// Provenance links created from notes to conversations
    pub note_conversation_links_created: usize,
    /// Provenance links created from notes to messages
    pub note_message_links_created: usize,
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
    use super::{
        chunk_content, decode_file_uri, entity_job_force_clear, no_processing_work,
        truncate_for_extraction, LibrarianAgent, LibrarianRuntimeConfig, EMBEDDING_JOB_WINDOW,
    };
    use crate::{
        DeterministicEmbedder, EntityExtraction, ExtractedEntity, FixtureEntityExtractor,
        InferenceCapabilities,
    };
    use graphrag_core::{
        normalized_content_hash, record_id_to_string, EdgeType, Entity, EntityType, Note,
    };
    use graphrag_db::{
        compatibility::{EmbeddingIdentity, ExtractionIdentity},
        init_memory, ProcessingJobStatus, ProcessingJobType, ProcessingJobUpdate, Repository,
        SourceImportAction,
    };
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Clone)]
    struct CancellingFallbackEmbedder {
        fallback_calls: Arc<AtomicUsize>,
        cancellation_requested: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl crate::Embedder for CancellingFallbackEmbedder {
        async fn embed(&self, _text: &str, _is_query: bool) -> crate::Result<Vec<f32>> {
            self.fallback_calls.fetch_add(1, Ordering::SeqCst);
            // Model the first Ctrl-C arriving during this slow fallback
            // request. The caller must checkpoint this result and stop before
            // starting the next individual request.
            self.cancellation_requested.store(true, Ordering::Release);
            Ok(vec![0.0; 1024])
        }

        async fn embed_batch(
            &self,
            _texts: &[String],
            _is_query: bool,
        ) -> crate::Result<Vec<Vec<f32>>> {
            Err(crate::AgentError::InferenceService("batch timeout".into()))
        }

        async fn health(&self) -> crate::Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "cancelling-fixture".into(),
                model: "fixture".into(),
                endpoint: "offline://cancelling-fallback".into(),
                known_dimension: Some(1024),
                cache_identity: "cancelling-fallback-v1".into(),
            }
        }
    }

    #[derive(Clone)]
    struct CancellingBatchEmbedder {
        cancellation_requested: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl crate::Embedder for CancellingBatchEmbedder {
        async fn embed(&self, _text: &str, _is_query: bool) -> crate::Result<Vec<f32>> {
            Ok(vec![0.0; 1024])
        }

        async fn embed_batch(
            &self,
            texts: &[String],
            _is_query: bool,
        ) -> crate::Result<Vec<Vec<f32>>> {
            // Model Ctrl-C arriving while the next batch is in flight, after
            // resume reconciliation has consumed already-finished items.
            self.cancellation_requested.store(true, Ordering::Release);
            Ok(vec![vec![0.0; 1024]; texts.len()])
        }

        async fn health(&self) -> crate::Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "cancelling-batch-fixture".into(),
                model: "fixture".into(),
                endpoint: "offline://cancelling-batch".into(),
                known_dimension: Some(1024),
                cache_identity: "cancelling-batch-v1".into(),
            }
        }
    }

    #[derive(Clone)]
    struct PermanentBatchFailureEmbedder;

    #[async_trait::async_trait]
    impl crate::Embedder for PermanentBatchFailureEmbedder {
        async fn embed(&self, text: &str, _is_query: bool) -> crate::Result<Vec<f32>> {
            if text == "permanently invalid embedding input" {
                return Err(crate::AgentError::InferenceService(
                    "invalid embedding input".into(),
                ));
            }
            Ok(vec![0.0; 1024])
        }

        async fn embed_batch(
            &self,
            _texts: &[String],
            _is_query: bool,
        ) -> crate::Result<Vec<Vec<f32>>> {
            Err(crate::AgentError::InferenceService(
                "invalid batch request".into(),
            ))
        }

        async fn health(&self) -> crate::Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "permanent-batch-failure-fixture".into(),
                model: "fixture".into(),
                endpoint: "offline://permanent-batch-failure".into(),
                known_dimension: Some(1024),
                cache_identity: "permanent-batch-failure-v1".into(),
            }
        }
    }

    #[derive(Clone)]
    struct CancellingEntityExtractor {
        calls: Arc<AtomicUsize>,
        cancellation_requested: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl crate::EntityExtractor for CancellingEntityExtractor {
        async fn extract(&self, _text: &str) -> crate::Result<crate::EntityExtraction> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // The worker must not fetch or process the next persisted page
            // once a request asks it to stop after this item.
            self.cancellation_requested.store(true, Ordering::Release);
            Ok(crate::EntityExtraction {
                entities: Vec::new(),
                relationships: Vec::new(),
            })
        }

        async fn health(&self) -> crate::Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "cancelling-entity-fixture".into(),
                model: "fixture".into(),
                endpoint: "offline://cancelling-entity".into(),
                known_dimension: None,
                cache_identity: "cancelling-entity-v1".into(),
            }
        }
    }

    #[test]
    fn runtime_config_defaults_preserve_library_behavior() {
        let config = LibrarianRuntimeConfig::default();
        assert_eq!(config.min_chunk_size, 20);
        assert_eq!(config.max_chunk_size, usize::MAX);
        assert!(!config.skip_entity_extraction);
    }

    #[test]
    fn entity_job_force_policy_parses_all_and_explicit_scopes() {
        assert!(entity_job_force_clear(Some(
            "all_notes:page_size=100;force=true"
        )));
        assert!(entity_job_force_clear(Some("note_ids:force=true")));
        assert!(!entity_job_force_clear(Some(
            "all_notes:page_size=100;force=false"
        )));
        assert!(!entity_job_force_clear(Some("note_ids:force=invalid")));
        assert!(!entity_job_force_clear(None));
    }

    #[tokio::test]
    async fn explicit_force_scope_clears_mentions_on_initial_run_and_resume() {
        let repo = Repository::new(init_memory().await.unwrap());
        let initial = repo
            .create_note(Note::new("initial explicit force target"))
            .await
            .unwrap();
        let resumed = repo
            .create_note(Note::new("resumed explicit force target"))
            .await
            .unwrap();
        let mut entity = Entity::new("preexisting entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        for note in [&initial, &resumed] {
            repo.link_note_to_entity(note.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
                .await
                .unwrap();
        }
        let initial_id = graphrag_core::record_id_to_string(initial.id.as_ref().unwrap());
        let resumed_id = graphrag_core::record_id_to_string(resumed.id.as_ref().unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );
        assert_eq!(
            librarian
                .extract_entities_for_note_ids(&[initial_id.clone()], true)
                .await
                .unwrap(),
            1
        );
        assert!(repo
            .get_entities_for_note(&initial_id)
            .await
            .unwrap()
            .is_empty());

        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::EntityExtraction,
                None,
                1,
                Some("note_ids:force=true".into()),
                vec![resumed_id.clone()],
            )
            .await
            .unwrap();
        let job_id = graphrag_core::record_id_to_string(job.id.as_ref().unwrap());
        repo.cancel_processing_job(job.id.as_ref().unwrap())
            .await
            .unwrap();
        assert_eq!(
            librarian
                .resume_processing_job(&job_id)
                .await
                .unwrap()
                .completed,
            1
        );
        assert!(repo
            .get_entities_for_note(&resumed_id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn empty_processing_scopes_return_zero_work_without_persisting_jobs() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );

        assert_eq!(
            librarian
                .process_pending_embeddings_job(None)
                .await
                .unwrap(),
            no_processing_work()
        );
        assert_eq!(
            librarian
                .extract_entities_for_notes_job(100, None, false, false)
                .await
                .unwrap(),
            no_processing_work()
        );
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn resume_rejects_empty_public_job_without_transitioning_to_running() {
        let repo = Repository::new(init_memory().await.unwrap());
        // The public constructor predates persisted item scopes and can still
        // produce a cancelled job with no resumable item set.
        let job = repo
            .create_processing_job(ProcessingJobType::Embedding, None, 0)
            .await
            .unwrap();
        let job_id = graphrag_core::record_id_to_string(job.id.as_ref().unwrap());
        repo.cancel_processing_job(job.id.as_ref().unwrap())
            .await
            .unwrap();

        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );
        let error = librarian.resume_processing_job(&job_id).await.unwrap_err();
        assert!(error.to_string().contains("no persisted item set"));

        let unchanged = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(
            unchanged.status,
            ProcessingJobStatus::Cancelled.as_str(),
            "invalid jobs must remain resumable/cancellable rather than being wedged in running"
        );
    }

    #[tokio::test]
    async fn pending_embedding_snapshot_pages_ids_and_stops_before_job_on_cancellation() {
        let repo = Repository::new(init_memory().await.unwrap());
        for content in [
            "first pending embedding",
            "second pending embedding",
            "third pending embedding",
        ] {
            repo.create_note(Note::new(content)).await.unwrap();
        }
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );
        // A one-item page forces a multi-page snapshot without materializing
        // all notes in the initial pending query.
        assert_eq!(
            librarian
                .pending_embedding_note_ids(1)
                .await
                .unwrap()
                .unwrap()
                .len(),
            3
        );

        let cancelled = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(Arc::new(AtomicBool::new(true)));
        assert_eq!(cancelled.process_pending_embeddings().await.unwrap(), 0);
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
    }

    #[test]
    fn chunking_honors_resolved_minimum_and_maximum_sizes() {
        assert_eq!(
            chunk_content("abcdefghijk", 2, 5),
            vec!["abcde", "fghi", "jk"]
        );
        assert_eq!(chunk_content("abcdef", 2, 5), vec!["abcd", "ef"]);
        let content = "tiny\n\nabcdefghijkl";
        assert_eq!(chunk_content(content, 11, 20), vec!["abcdefghijkl"]);
    }

    #[test]
    fn extraction_truncation_uses_runtime_limit() {
        assert_eq!(truncate_for_extraction("abcdef", 0), "abcdef");
        assert_eq!(truncate_for_extraction("abcdef", 3), "abc\n\n[truncated]");
    }

    #[test]
    fn decodes_file_uri_drive_letters_without_host_platform_assumptions() {
        assert_eq!(
            decode_file_uri("file:///C:/notes/alpha.md", true).unwrap(),
            "C:/notes/alpha.md"
        );
        assert_eq!(
            decode_file_uri("file:///Users/hunter/notes.md", false).unwrap(),
            "/Users/hunter/notes.md"
        );
        assert_eq!(
            decode_file_uri("file://server/share/notes.md", true).unwrap(),
            "//server/share/notes.md"
        );
    }

    #[tokio::test]
    async fn unchanged_markdown_import_reports_recovered_cleanup() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );
        let content = "enough markdown content to create a stable source note";
        let first = librarian
            .ingest_markdown_with_options("cleanup-recovery.md", content, false)
            .await
            .unwrap();
        let source = repo.get_source(&first.source_uri).await.unwrap().unwrap();
        repo.create_note(
            Note::new("stale hidden generation")
                .with_source(source.id.unwrap())
                .with_source_generation(0),
        )
        .await
        .unwrap();

        let retry = librarian
            .ingest_markdown_with_options("cleanup-recovery.md", content, false)
            .await
            .unwrap();
        assert_eq!(retry.action, SourceImportAction::Unchanged);
        assert_eq!(retry.deleted, 1);
        assert_eq!(retry.cleanup.notes, 1);
        assert_eq!(retry.cleanup.note_edges, 0);
        assert_eq!(retry.cleanup.note_conversation_provenance, 0);
        assert_eq!(retry.cleanup.note_message_provenance, 0);
    }

    #[tokio::test]
    async fn entity_resume_reuses_its_persisted_scope_and_reconciles_prior_failures() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first = repo
            .create_note(Note::new("first pending entity extraction"))
            .await
            .unwrap();
        let second = repo
            .create_note(Note::new("second pending entity extraction"))
            .await
            .unwrap();
        let failed_run = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default().fail_next_requests(1, "timeout")),
        );
        assert!(failed_run.extract_entities_for_notes(2).await.is_err());
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        let job_id = job.id.as_ref().map(record_id_to_string).unwrap();
        assert_eq!(job.status, ProcessingJobStatus::Failed.as_str());
        assert!(
            job.last_error
                .as_deref()
                .is_some_and(|error| error.contains("timeout")),
            "a later successful item must not erase the job's failure diagnostic"
        );
        assert_eq!(job.scope.as_deref(), Some("missing_entities:limit=2"));
        assert_eq!(job.item_ids.len(), 2);
        assert!(job
            .item_ids
            .iter()
            .any(|id| id == &record_id_to_string(first.id.as_ref().unwrap())));
        assert!(job
            .item_ids
            .iter()
            .any(|id| id == &record_id_to_string(second.id.as_ref().unwrap())));

        // A failed job retries its entire durable scope. If that retry is
        // immediately cancelled, its stale terminal checkpoint must already
        // be cleared so the next resume cannot skip the failed prior scope.
        let cancelled_retry = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(Arc::new(AtomicBool::new(true)))
        .resume_processing_job(&job_id)
        .await
        .unwrap();
        assert!(cancelled_retry.cancelled);
        assert_eq!(cancelled_retry.completed, 0);
        let cancelled_job = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(
            cancelled_job.status,
            ProcessingJobStatus::Cancelled.as_str()
        );
        assert_eq!(cancelled_job.completed_count, 0);
        assert_eq!(cancelled_job.failed_count, 0);
        assert!(cancelled_job.checkpoint.is_none());

        let resumed = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .resume_processing_job(&job_id)
        .await
        .unwrap();
        assert_eq!(resumed.completed, 2);
        assert_eq!(resumed.failed, 0);
        assert!(!resumed.cancelled);
        assert_eq!(
            repo.get_processing_job(&job_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProcessingJobStatus::Completed.as_str()
        );
    }

    #[tokio::test]
    async fn in_process_cancellation_is_observed_between_job_items() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("pending entity extraction"))
            .await
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(true));
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(cancelled);
        assert_eq!(librarian.extract_entities_for_notes(1).await.unwrap(), 0);
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.status, ProcessingJobStatus::Cancelled.as_str());
    }

    #[tokio::test]
    async fn all_entity_extraction_stops_between_persisted_id_items() {
        let repo = Repository::new(init_memory().await.unwrap());
        for content in ["first page", "second page", "third page"] {
            repo.create_note(Note::new(content)).await.unwrap();
        }
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(CancellingEntityExtractor {
                calls: calls.clone(),
                cancellation_requested: cancellation_requested.clone(),
            }),
        )
        .with_cancellation_flag(cancellation_requested.clone());

        // One persisted page contains all three notes. The first extraction
        // requests cancellation, so the in-page check must prevent later
        // notes from reaching the provider.
        assert_eq!(
            librarian
                .extract_entities_for_all_notes(3, true)
                .await
                .unwrap(),
            1
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.item_ids.len(), 3);
        assert_eq!(job.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(job.completed_count, 1);
        assert_eq!(job.failed_count, 0);
        let job_id = graphrag_core::record_id_to_string(job.id.as_ref().unwrap());

        // Resume keeps the completed prefix out of the force-mode executor:
        // exactly the two remaining note IDs reach the provider.
        let resumed = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(CancellingEntityExtractor {
                calls: calls.clone(),
                cancellation_requested,
            }),
        )
        .with_cancellation_flag(Arc::new(AtomicBool::new(false)))
        .resume_processing_job(&job_id)
        .await
        .unwrap();
        assert_eq!(resumed.completed, 3);
        assert_eq!(resumed.failed, 0);
        assert!(!resumed.cancelled);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            repo.get_processing_job(&job_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProcessingJobStatus::Completed.as_str()
        );
    }

    #[tokio::test]
    async fn entity_resume_skips_deleted_persisted_items() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first = repo
            .create_note(Note::new("first extracted"))
            .await
            .unwrap();
        let deleted = repo
            .create_note(Note::new("deleted queued item"))
            .await
            .unwrap();
        let remaining = repo
            .create_note(Note::new("remaining queued item"))
            .await
            .unwrap();
        let first_id = graphrag_core::record_id_to_string(first.id.as_ref().unwrap());
        let deleted_id = graphrag_core::record_id_to_string(deleted.id.as_ref().unwrap());
        let remaining_id = graphrag_core::record_id_to_string(remaining.id.as_ref().unwrap());
        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::EntityExtraction,
                None,
                3,
                Some("all_notes:page_size=3;force=false".into()),
                vec![first_id.clone(), deleted_id.clone(), remaining_id.clone()],
            )
            .await
            .unwrap();
        let job_id = graphrag_core::record_id_to_string(job.id.as_ref().unwrap());
        repo.update_processing_job(
            job.id.as_ref().unwrap(),
            ProcessingJobUpdate {
                completed_count: Some(1),
                checkpoint: Some(Some(first_id.clone())),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        repo.cancel_processing_job(job.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.delete_note(&deleted_id).await.unwrap();

        // A second cancellation immediately after resume must retain the
        // restored completed prefix before the worker can inspect another
        // item. A later resume then reconciles from that same checkpoint.
        let immediately_cancelled = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(Arc::new(AtomicBool::new(true)))
        .resume_processing_job(&job_id)
        .await
        .unwrap();
        assert!(immediately_cancelled.cancelled);
        assert_eq!(immediately_cancelled.completed, 1);
        let checkpointed = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(checkpointed.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(checkpointed.completed_count, 1);
        assert_eq!(checkpointed.checkpoint.as_deref(), Some(first_id.as_str()));

        let result = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .resume_processing_job(&job_id)
        .await
        .unwrap();
        assert_eq!(result.completed, 3);
        assert_eq!(result.failed, 0);
        let job = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(job.status, ProcessingJobStatus::Completed.as_str());
        assert_eq!(job.completed_count, 3);
        assert_eq!(job.failed_count, 0);
        assert_eq!(job.checkpoint.as_deref(), Some(remaining_id.as_str()));
    }

    #[tokio::test]
    async fn cancellation_stops_all_and_explicit_entity_modes_before_work() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("pending extraction"))
            .await
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(true));
        let all = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(cancelled.clone());
        let all_result = all
            .extract_entities_for_all_notes_result(10, false)
            .await
            .unwrap();
        assert!(all_result.cancelled);
        assert_eq!(all_result.completed, 0);
        // Cancellation during the initial `--all` ID snapshot happens before
        // a durable job exists, so it must leave no empty job behind.
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
        let explicit = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(cancelled);
        let explicit_result = explicit
            .extract_entities_for_note_ids_result(
                &[graphrag_core::record_id_to_string(
                    note.id.as_ref().unwrap(),
                )],
                false,
            )
            .await
            .unwrap();
        assert!(explicit_result.cancelled);
        assert_eq!(explicit_result.completed, 0);
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn all_and_explicit_entity_modes_persist_resumable_job_scopes() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first = repo
            .create_note(Note::new("first durable entity selection"))
            .await
            .unwrap();
        let second = repo
            .create_note(Note::new("second durable entity selection"))
            .await
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let all_calls = Arc::new(AtomicUsize::new(0));
        let all = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(CancellingEntityExtractor {
                calls: all_calls,
                cancellation_requested: cancelled.clone(),
            }),
        )
        .with_cancellation_flag(cancelled.clone());
        assert_eq!(
            all.extract_entities_for_all_notes(1, true).await.unwrap(),
            1
        );

        let all_job = repo
            .list_processing_jobs(10)
            .await
            .unwrap()
            .into_iter()
            .find(|job| job.scope.as_deref() == Some("all_notes:page_size=1;force=true"))
            .unwrap();
        assert_eq!(all_job.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(all_job.item_ids.len(), 2);
        let all_job_id = graphrag_core::record_id_to_string(all_job.id.as_ref().unwrap());
        let resumed_all = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .resume_processing_job(&all_job_id)
        .await
        .unwrap();
        assert_eq!(resumed_all.completed, 2);
        assert!(!resumed_all.cancelled);

        let explicit_cancelled = Arc::new(AtomicBool::new(false));
        let explicit = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(CancellingEntityExtractor {
                calls: Arc::new(AtomicUsize::new(0)),
                cancellation_requested: explicit_cancelled.clone(),
            }),
        )
        .with_cancellation_flag(explicit_cancelled);
        let first_id = graphrag_core::record_id_to_string(first.id.as_ref().unwrap());
        let second_id = graphrag_core::record_id_to_string(second.id.as_ref().unwrap());
        assert_eq!(
            explicit
                .extract_entities_for_note_ids(&[first_id.clone(), second_id.clone()], false)
                .await
                .unwrap(),
            1
        );
        let explicit_job = repo
            .list_processing_jobs(10)
            .await
            .unwrap()
            .into_iter()
            .find(|job| job.scope.as_deref() == Some("note_ids:force=false"))
            .unwrap();
        assert_eq!(explicit_job.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(explicit_job.item_ids, vec![first_id, second_id]);
        let explicit_job_id = graphrag_core::record_id_to_string(explicit_job.id.as_ref().unwrap());
        let resumed_explicit = LibrarianAgent::new(
            repo,
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .resume_processing_job(&explicit_job_id)
        .await
        .unwrap();
        assert_eq!(resumed_explicit.completed, 2);
        assert!(!resumed_explicit.cancelled);
    }

    #[tokio::test]
    async fn embedding_fallback_keeps_failure_diagnostic_after_later_success() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("first embedding"))
            .await
            .unwrap();
        repo.create_note(Note::new("second embedding"))
            .await
            .unwrap();
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().fail_next_requests(2, "timeout")),
            Arc::new(FixtureEntityExtractor::default()),
        );
        assert!(librarian.process_pending_embeddings().await.is_err());
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.status, ProcessingJobStatus::Failed.as_str());
        assert_eq!(job.failed_count, 1);
        assert!(
            job.last_error
                .as_deref()
                .is_some_and(|error| error.contains("timeout")),
            "the success after a fallback failure must not clear diagnostics"
        );
    }

    #[tokio::test]
    async fn embedding_batch_response_cardinality_is_validated_before_fallback() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo,
            Arc::new(DeterministicEmbedder::default().with_batch_length_mismatch()),
            Arc::new(FixtureEntityExtractor::default()),
        );

        let error = librarian
            .embed_batch(&["only batch input".to_string()])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("embeddings for 1 inputs"));
    }

    #[tokio::test]
    async fn permanent_batch_failure_isolated_to_the_bad_embedding_item() {
        let repo = Repository::new(init_memory().await.unwrap());
        let valid_first = repo
            .create_note(Note::new("first valid embedding input"))
            .await
            .unwrap();
        let invalid = repo
            .create_note(Note::new("permanently invalid embedding input"))
            .await
            .unwrap();
        let valid_last = repo
            .create_note(Note::new("last valid embedding input"))
            .await
            .unwrap();
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(PermanentBatchFailureEmbedder),
            Arc::new(FixtureEntityExtractor::default()),
        );

        assert!(librarian.process_pending_embeddings().await.is_err());
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.status, ProcessingJobStatus::Failed.as_str());
        assert_eq!(job.completed_count, 2);
        assert_eq!(job.failed_count, 1);
        assert!(job
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("invalid embedding input")));

        for valid in [valid_first, valid_last] {
            let note = repo
                .get_note(&graphrag_core::record_id_to_string(
                    valid.id.as_ref().unwrap(),
                ))
                .await
                .unwrap()
                .unwrap();
            assert!(!note.embedding.is_empty());
        }
        let invalid = repo
            .get_note(&graphrag_core::record_id_to_string(
                invalid.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        assert!(invalid.embedding.is_empty());
    }

    #[tokio::test]
    async fn embedding_failure_does_not_stop_later_durable_windows() {
        let repo = Repository::new(init_memory().await.unwrap());
        let invalid = repo
            .create_note(Note::new("permanently invalid embedding input"))
            .await
            .unwrap();
        let mut item_ids = vec![graphrag_core::record_id_to_string(
            invalid.id.as_ref().unwrap(),
        )];
        let mut later_note_id = None;
        for index in 0..33 {
            let note = repo
                .create_note(Note::new(format!("valid embedding input {index}")))
                .await
                .unwrap();
            item_ids.push(graphrag_core::record_id_to_string(
                note.id.as_ref().unwrap(),
            ));
            if index == 32 {
                later_note_id = note.id.as_ref().map(graphrag_core::record_id_to_string);
            }
        }
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(PermanentBatchFailureEmbedder),
            Arc::new(FixtureEntityExtractor::default()),
        );
        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::Embedding,
                None,
                item_ids.len() as u64,
                Some("durable-window-regression".into()),
                item_ids,
            )
            .await
            .unwrap();

        assert!(librarian
            .process_pending_embeddings_job(Some(job))
            .await
            .is_err());
        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.status, ProcessingJobStatus::Failed.as_str());
        assert_eq!(job.completed_count, 33);
        assert_eq!(job.failed_count, 1);
        assert!(job
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("invalid embedding input")));
        let invalid = repo
            .get_note(&graphrag_core::record_id_to_string(
                invalid.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        assert!(invalid.embedding.is_empty());
        let later = repo
            .get_note(&later_note_id.expect("last note id"))
            .await
            .unwrap()
            .unwrap();
        assert!(!later.embedding.is_empty());
    }

    #[tokio::test]
    async fn embedding_fallback_checkpoints_then_stops_on_cancellation() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first = repo
            .create_note(Note::new("first fallback embedding"))
            .await
            .unwrap();
        let second = repo
            .create_note(Note::new("second fallback embedding"))
            .await
            .unwrap();
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let fallback_calls = Arc::new(AtomicUsize::new(0));
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(CancellingFallbackEmbedder {
                fallback_calls: fallback_calls.clone(),
                cancellation_requested: cancellation_requested.clone(),
            }),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(cancellation_requested);

        let result = librarian
            .process_pending_embeddings_job(None)
            .await
            .unwrap();
        assert!(result.cancelled);
        assert_eq!(result.completed, 1);
        assert_eq!(result.failed, 0);
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 1);

        let job = repo.list_processing_jobs(1).await.unwrap().remove(0);
        assert_eq!(job.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(job.completed_count, 1);
        assert!(job.checkpoint.is_some());
        let first = repo
            .get_note(&graphrag_core::record_id_to_string(
                first.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        let second = repo
            .get_note(&graphrag_core::record_id_to_string(
                second.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        let embedded = [first, second]
            .into_iter()
            .filter(|note| !note.embedding.is_empty())
            .count();
        assert_eq!(embedded, 1);
    }

    #[tokio::test]
    async fn embedding_resume_persists_reconciliation_before_next_batch_cancellation() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(
            &EmbeddingIdentity::new("cancelling-batch-fixture", "fixture", 1024),
            Some(&ExtractionIdentity::new("fixture-test", "fixture")),
        )
        .await
        .unwrap();
        // A resumed job may have a long prefix that was completed by the
        // interrupted attempt. Keep that prefix exactly one reconciliation
        // window so the following deleted/pending items exercise the next
        // bounded pass rather than a single unbounded scan.
        let mut completed_ids = Vec::with_capacity(EMBEDDING_JOB_WINDOW);
        for index in 0..EMBEDDING_JOB_WINDOW {
            let embedded = repo
                .create_note(
                    Note::new(format!("already embedded {index}")).with_embedding(vec![0.0; 1024]),
                )
                .await
                .unwrap();
            completed_ids.push(graphrag_core::record_id_to_string(
                embedded.id.as_ref().unwrap(),
            ));
        }
        let deleted = repo
            .create_note(Note::new("deleted before resume"))
            .await
            .unwrap();
        let pending = repo
            .create_note(Note::new("pending when cancellation arrives"))
            .await
            .unwrap();
        let deleted_id = graphrag_core::record_id_to_string(deleted.id.as_ref().unwrap());
        repo.delete_note(&deleted_id).await.unwrap();
        let job = repo
            .create_processing_job_with_scope(
                graphrag_db::ProcessingJobType::Embedding,
                None,
                (EMBEDDING_JOB_WINDOW + 2) as u64,
                Some("resume-fixture".into()),
                completed_ids
                    .into_iter()
                    .chain([
                        deleted_id,
                        graphrag_core::record_id_to_string(pending.id.as_ref().unwrap()),
                    ])
                    .collect(),
            )
            .await
            .unwrap();
        let job_id = graphrag_core::record_id_to_string(job.id.as_ref().unwrap());
        repo.cancel_processing_job(job.id.as_ref().unwrap())
            .await
            .unwrap();
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(CancellingBatchEmbedder {
                cancellation_requested: cancellation_requested.clone(),
            }),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_cancellation_flag(cancellation_requested);

        let result = librarian.resume_processing_job(&job_id).await.unwrap();
        assert!(result.cancelled);
        assert_eq!(result.completed, (EMBEDDING_JOB_WINDOW + 1) as u64);
        assert_eq!(result.failed, 0);

        let job = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(job.status, ProcessingJobStatus::Cancelled.as_str());
        assert_eq!(job.completed_count, (EMBEDDING_JOB_WINDOW + 1) as i64);
        assert_eq!(job.failed_count, 0);
        assert!(repo
            .get_note(&graphrag_core::record_id_to_string(
                pending.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .unwrap()
            .embedding
            .is_empty());
    }

    #[tokio::test]
    async fn markdown_reconciliation_preserves_unchanged_chunk_identity_and_provenance() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_runtime_config(LibrarianRuntimeConfig {
            min_chunk_size: 10,
            target_chunk_size: 60,
            max_chunk_size: 100,
            chunk_overlap: 0,
            ..Default::default()
        });
        let first_content = "# Planning\n\nFirst independently stable paragraph has useful words.\n\nSecond independently stable paragraph has useful words.";
        let first = librarian
            .ingest_markdown_with_options("chunk-reconcile.md", first_content, false)
            .await
            .unwrap();
        assert_eq!(first.notes.len(), 2);
        let stable_key_before = first.notes[1].chunk_key.clone();
        let stable_embedding_before = first.notes[1].embedding.clone();
        let stable_created_at = first.notes[1].created_at;
        assert_eq!(first.notes[1].chunk_heading_path, vec!["Planning"]);
        assert!(first.notes[1].source_start_byte.is_some());
        assert!(first.notes[1]
            .search_content
            .as_deref()
            .is_some_and(|text| text.starts_with("Planning\n\n")));

        let second_content = "# Planning\n\nFirst independently changed paragraph has useful words.\n\nSecond independently stable paragraph has useful words.";
        let second = librarian
            .ingest_markdown_with_options("chunk-reconcile.md", second_content, false)
            .await
            .unwrap();
        assert_eq!(second.notes.len(), 2);
        assert_eq!(second.action, SourceImportAction::Updated);
        assert_eq!(stable_key_before, second.notes[1].chunk_key);
        assert_eq!(stable_embedding_before, second.notes[1].embedding);
        assert_eq!(stable_created_at, second.notes[1].created_at);
        assert_ne!(first.notes[0].id, second.notes[0].id);
        assert_ne!(first.notes[0].content_hash, second.notes[0].content_hash);
        assert_ne!(first.notes[0].embedding, second.notes[0].embedding);
        assert_eq!(repo.fulltext_search("Planning", 10).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn failed_markdown_reconciliation_keeps_the_successful_generation_visible() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_runtime_config(LibrarianRuntimeConfig {
            min_chunk_size: 10,
            target_chunk_size: 60,
            max_chunk_size: 100,
            ..Default::default()
        });
        let original = "# Plan\n\nFirst stable paragraph has enough content.\n\nSecond stable paragraph has enough content.";
        let first = librarian
            .ingest_markdown_with_options("copy-on-write.md", original, false)
            .await
            .unwrap();
        let source_id = first.notes[0].source_id.clone().unwrap();
        let successful = repo.get_source_chunks(&source_id).await.unwrap();
        let successful_ids = successful
            .iter()
            .map(|note| note.id.clone().unwrap())
            .collect::<Vec<_>>();

        let changed = "# Plan\n\nFirst changed paragraph has enough content.\n\nSecond stable paragraph has enough content.";
        let mut pending = repo
            .begin_file_import(
                graphrag_core::SourceType::Markdown,
                "copy-on-write.md".into(),
                first.source_uri.clone(),
                changed.into(),
                normalized_content_hash(changed),
                false,
            )
            .await
            .unwrap();
        let staged = librarian
            .chunk_and_create_markdown_notes(
                changed,
                pending.source.id.clone(),
                Some(pending.source.generation),
                &successful,
            )
            .await
            .unwrap();
        assert!(staged
            .iter()
            .all(|note| !successful_ids.contains(note.id.as_ref().unwrap())));

        repo.fail_file_import(&mut pending.source, "simulated promotion failure")
            .await
            .unwrap();
        let restored = repo.get_source_chunks(&source_id).await.unwrap();
        assert_eq!(
            restored
                .iter()
                .map(|note| note.id.clone().unwrap())
                .collect::<Vec<_>>(),
            successful_ids
        );
        assert_eq!(
            repo.fulltext_search("First stable", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(repo
            .fulltext_search("First changed", 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn markdown_reconciliation_copies_only_safe_dependents_for_changed_chunks() {
        let repo = Repository::new(init_memory().await.unwrap());
        let replacement = "Middle replacement paragraph has enough content.";
        let extractor = FixtureEntityExtractor::default().with_fixture(
            replacement,
            EntityExtraction {
                entities: vec![ExtractedEntity {
                    name: "Replacement Entity".into(),
                    entity_type: Some("concept".into()),
                }],
                relationships: Vec::new(),
            },
        );
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(extractor),
        )
        .with_runtime_config(LibrarianRuntimeConfig {
            min_chunk_size: 10,
            target_chunk_size: 60,
            max_chunk_size: 100,
            ..Default::default()
        });
        let original = "# Plan\n\nBefore stable paragraph has enough content.\n\nMiddle original paragraph has enough content.\n\nAfter stable paragraph has enough content.";
        let first = librarian
            .ingest_markdown_with_options("dependent-copy.md", original, false)
            .await
            .unwrap();
        assert_eq!(first.notes.len(), 3);
        let middle_old = first.notes[1].id.as_ref().unwrap().clone();
        let after_old = first.notes[2].id.as_ref().unwrap().clone();
        let mut entity = Entity::new("Planning", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(&middle_old, entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(&middle_old, &after_old, EdgeType::Supports, Some(0.9))
            .await
            .unwrap();

        let changed = format!(
            "# Plan\n\nBefore stable paragraph has enough content.\n\n{replacement}\n\nAfter stable paragraph has enough content."
        );
        let second = librarian
            .ingest_markdown_with_options("dependent-copy.md", &changed, false)
            .await
            .unwrap();
        assert_eq!(second.notes[1].content, replacement);
        let middle_new = second.notes[1].id.as_ref().unwrap();
        let after_new = second.notes[2].id.as_ref().unwrap();
        assert!(repo
            .get_note(&record_id_to_string(&middle_old))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_entities_for_note(&record_id_to_string(middle_new))
            .await
            .unwrap()
            .is_empty());
        assert!(repo
            .get_notes_without_entities(10)
            .await
            .unwrap()
            .iter()
            .any(|note| note.id.as_ref() == Some(middle_new)));
        assert_eq!(
            librarian
                .extract_entities_for_note_ids(&[record_id_to_string(middle_new)], false)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            repo.get_entities_for_note(&record_id_to_string(middle_new))
                .await
                .unwrap()
                .iter()
                .map(|entity| entity.canonical_name.as_str())
                .collect::<Vec<_>>(),
            vec!["replacement entity"]
        );
        let edges = repo
            .get_note_edges(&record_id_to_string(middle_new))
            .await
            .unwrap();
        assert!(edges.iter().any(|edge| {
            edge.edge_type == EdgeType::Supports.to_string()
                && edge.in_id == *middle_new
                && edge.out_id == *after_new
                && edge.is_manual
        }));
    }

    #[tokio::test]
    async fn markdown_reconciliation_aligns_unchanged_chunks_across_insertions_and_removals() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_runtime_config(LibrarianRuntimeConfig {
            min_chunk_size: 10,
            target_chunk_size: 60,
            max_chunk_size: 100,
            ..Default::default()
        });
        let first_chunk = "First stable paragraph has enough content.";
        let second_chunk = "Second stable paragraph has enough content.";
        let original = format!("# Plan\n\n{first_chunk}\n\n{second_chunk}");
        let first = librarian
            .ingest_markdown_with_options("sequence-alignment.md", &original, false)
            .await
            .unwrap();
        assert_eq!(first.notes.len(), 2);
        let first_old = first.notes[0].id.as_ref().unwrap().clone();
        let second_old = first.notes[1].id.as_ref().unwrap().clone();
        let mut entity = Entity::new("First", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(&first_old, entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(&first_old, &second_old, EdgeType::Supports, Some(0.9))
            .await
            .unwrap();

        let inserted = format!(
            "# Plan\n\nInserted unrelated paragraph has enough content.\n\n{first_chunk}\n\n{second_chunk}"
        );
        let after_insert = librarian
            .ingest_markdown_with_options("sequence-alignment.md", &inserted, false)
            .await
            .unwrap();
        let inserted_note = after_insert
            .notes
            .iter()
            .find(|note| note.content == "Inserted unrelated paragraph has enough content.")
            .unwrap();
        let first_after_insert = after_insert
            .notes
            .iter()
            .find(|note| note.content == first_chunk)
            .unwrap();
        let second_after_insert = after_insert
            .notes
            .iter()
            .find(|note| note.content == second_chunk)
            .unwrap();
        assert!(repo
            .get_entities_for_note(&record_id_to_string(inserted_note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            repo.get_entities_for_note(&record_id_to_string(
                first_after_insert.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .len(),
            1
        );
        assert!(repo
            .get_note_edges(&record_id_to_string(
                first_after_insert.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .iter()
            .any(|edge| {
                edge.edge_type == EdgeType::Supports.to_string()
                    && edge.in_id == *first_after_insert.id.as_ref().unwrap()
                    && edge.out_id == *second_after_insert.id.as_ref().unwrap()
            }));

        let after_removal = librarian
            .ingest_markdown_with_options("sequence-alignment.md", &original, false)
            .await
            .unwrap();
        let first_after_removal = after_removal
            .notes
            .iter()
            .find(|note| note.content == first_chunk)
            .unwrap();
        let second_after_removal = after_removal
            .notes
            .iter()
            .find(|note| note.content == second_chunk)
            .unwrap();
        assert_eq!(
            repo.get_entities_for_note(&record_id_to_string(
                first_after_removal.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .len(),
            1
        );
        assert!(repo
            .get_note_edges(&record_id_to_string(
                first_after_removal.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .iter()
            .any(|edge| {
                edge.edge_type == EdgeType::Supports.to_string()
                    && edge.in_id == *first_after_removal.id.as_ref().unwrap()
                    && edge.out_id == *second_after_removal.id.as_ref().unwrap()
            }));
    }

    #[tokio::test]
    async fn oversized_fenced_code_split_marker_is_persisted() {
        let repo = Repository::new(init_memory().await.unwrap());
        let librarian = LibrarianAgent::new(
            repo,
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        )
        .with_runtime_config(LibrarianRuntimeConfig {
            min_chunk_size: 5,
            target_chunk_size: 10,
            max_chunk_size: 16,
            ..Default::default()
        });
        let imported = librarian
            .ingest_markdown_with_options(
                "split-fence.md",
                "```text\n0123456789012345678901234567890123456789\n```",
                false,
            )
            .await
            .unwrap();
        assert!(!imported.notes.is_empty());
        assert!(imported.notes.iter().all(|note| note.split_fenced_code));
    }
}
