//! Librarian Agent - Ingests content and creates notes

use crate::{MlClient, Result};
use graphrag_core::{
    ChatConversation, ChatExport, ChatMessage, Entity, EntityType, MessageRole, Note, NoteType,
    Source, SourceType,
};
use graphrag_db::Repository;
use tracing::{debug, info, instrument};

/// The Librarian agent handles content ingestion
pub struct LibrarianAgent {
    repo: Repository,
    ml: MlClient,
}

impl LibrarianAgent {
    /// Create a new Librarian agent
    pub fn new(repo: Repository, ml: MlClient) -> Self {
        Self { repo, ml }
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
        let embedding = self.ml.embed_one(&content).await?;
        
        // Determine title
        let note_title = if let Some(t) = title {
            Some(t)
        } else {
            // Try AI generation
            if let Ok(Some(gen_title)) = self.ml.generate_title(&content).await {
                Some(gen_title)
            } else {
                // Fallback heuristic: first 3-5 words, max 48 characters
                content
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| {
                        let l = l.trim();
                        let words: Vec<&str> = l.split_whitespace().collect();
                        let mut title = words
                            .iter()
                            .take(5)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(" ");
                        if title.chars().count() > 48 {
                            title = title.chars().take(48).collect();
                        }
                        title
                    })
            }
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
    pub async fn ingest_markdown<C>(
        &self,
        path: &str,
        content: C,
    ) -> Result<Vec<Note>>
    where
        C: Into<String> + std::fmt::Debug,
    {
        let content = content.into();
        info!("Ingesting markdown from: {}", path);
        
        // Create source
        let source = Source::from_file(path, SourceType::Markdown)
            .with_content(content.clone());
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
        let embeddings = self.ml.embed(texts).await?;
        
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
        let entities = self.ml.extract_entities(&note.content).await?;
        
        for extracted in entities {
            // Map string type to EntityType
            let entity_type = match extracted.entity_type.to_lowercase().as_str() {
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
            }
        }
        
        Ok(())
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
            let embedding = self.ml.embed_one(content).await?;
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
        let embeddings = self.ml.embed(texts).await?;

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
    ) -> Result<ChatImportResult> {
        info!(
            "Ingesting chat export with {} conversations",
            export.conversation_count()
        );

        let mut result = ChatImportResult::default();

        for conversation in export.conversations {
            match self
                .ingest_conversation(&conversation, source_uri.clone())
                .await
            {
                Ok(notes) => {
                    result.conversations_imported += 1;
                    result.notes_created += notes.len();
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
        }

        info!(
            "Chat import complete: {} conversations, {} notes created",
            result.conversations_imported, result.notes_created
        );

        Ok(result)
    }

    /// Ingest a single conversation
    #[instrument(skip(self, conversation))]
    pub async fn ingest_conversation(
        &self,
        conversation: &ChatConversation,
        source_uri: Option<String>,
    ) -> Result<Vec<Note>> {
        let title = conversation.display_title();
        info!("Ingesting conversation: {}", title);

        // Create source record for this conversation
        let mut source = Source::chat_export(&title, source_uri);

        // Add conversation metadata
        let mut metadata = serde_json::Map::new();
        metadata.insert("conversation_id".into(), serde_json::json!(&conversation.uuid));
        metadata.insert("created_at".into(), serde_json::json!(&conversation.created_at));
        if !conversation.summary.is_empty() {
            metadata.insert("summary".into(), serde_json::json!(&conversation.summary));
        }
        source = source.with_metadata(serde_json::Value::Object(metadata));

        let source = self.repo.create_source(source).await?;
        let source_id = source.id.as_ref().map(|id| id.to_string());

        // Extract Q&A pairs from messages
        let qa_pairs = self.extract_qa_pairs(&conversation.messages);

        if qa_pairs.is_empty() {
            // No Q&A pairs, import the whole conversation as markdown
            let markdown = conversation.to_markdown();
            return self.chunk_and_create_notes(&markdown, source_id).await;
        }

        // Create notes from Q&A pairs
        let mut notes = Vec::new();
        let mut texts_to_embed: Vec<String> = Vec::new();
        let mut note_builders: Vec<(String, Option<String>)> = Vec::new();

        for (idx, (question, answer)) in qa_pairs.iter().enumerate() {
            // Format the Q&A as a note
            let content = format!("**Question:** {}\n\n**Answer:** {}", question, answer);

            // Generate a title from the question
            let note_title = self.generate_qa_title(question, idx + 1);

            texts_to_embed.push(content.clone());
            note_builders.push((content, Some(note_title)));
        }

        // Batch embed all Q&A pairs
        let embeddings = self.ml.embed(texts_to_embed).await?;

        // Create notes
        for ((content, title), embedding) in note_builders.into_iter().zip(embeddings.into_iter()) {
            let mut note = Note::new(&content)
                .with_type(NoteType::Synthesis)
                .with_embedding(embedding)
                .with_tags(vec!["chat-export".into(), "qa".into()]);

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

            notes.push(note);
        }

        info!(
            "Created {} notes from conversation '{}'",
            notes.len(),
            title
        );

        Ok(notes)
    }

    /// Extract Q&A pairs from a list of messages
    fn extract_qa_pairs(&self, messages: &[ChatMessage]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let mut current_human: Option<&str> = None;

        for msg in messages {
            match msg.role {
                MessageRole::Human => {
                    current_human = Some(&msg.content);
                }
                MessageRole::Assistant => {
                    if let Some(question) = current_human.take() {
                        // Only include if both question and answer are substantial
                        if question.len() > 10 && msg.content.len() > 20 {
                            pairs.push((question.to_string(), msg.content.clone()));
                        }
                    }
                }
                MessageRole::System => {
                    // Skip system messages for Q&A extraction
                }
            }
        }

        pairs
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
    /// Number of conversations successfully imported
    pub conversations_imported: usize,
    /// Number of conversations that failed to import
    pub conversations_failed: usize,
    /// Total number of notes created
    pub notes_created: usize,
    /// Error messages for failed imports
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
    // See tests/integration_test.rs
}
