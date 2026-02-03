//! Librarian Agent - Ingests content and creates notes

use crate::{MlClient, Result};
use graphrag_core::{Note, NoteType, Source, SourceType, Entity, EntityType};
use graphrag_db::Repository;
use tracing::{info, debug, instrument};

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
}

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
    // See tests/integration_test.rs
}
