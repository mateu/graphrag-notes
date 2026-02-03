//! Note types - the atomic units of knowledge

use chrono::{DateTime, Utc};
use surrealdb::RecordId;
use serde::{Deserialize, Serialize};

/// The type/classification of a note
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NoteType {
    /// A factual claim that can be verified
    Claim,
    /// A definition of a term or concept
    Definition,
    /// An observation or data point
    Observation,
    /// A question to be answered
    Question,
    /// A synthesis of multiple notes
    Synthesis,
    /// Raw/unprocessed note
    Raw,
}

impl Default for NoteType {
    fn default() -> Self {
        Self::Raw
    }
}

/// An atomic note - the fundamental unit of knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    /// Unique identifier (maps to SurrealDB record ID)
    pub id: Option<RecordId>,
    
    /// The type of this note
    pub note_type: NoteType,
    
    /// Title or summary (optional)
    pub title: Option<String>,
    
    /// The actual content
    pub content: String,
    
    /// Vector embedding (1536 dimensions for OpenAI, 384 for MiniLM)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,
    
    /// Source this note was derived from
    pub source_id: Option<String>,
    
    /// Extracted entities mentioned in this note
    #[serde(default)]
    pub entity_ids: Vec<String>,
    
    /// User-defined tags
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// When this note was created
    #[serde(skip_serializing)]
    pub created_at: DateTime<Utc>,
    
    /// When this note was last modified
    #[serde(skip_serializing)]
    pub updated_at: DateTime<Utc>,
}

impl Note {
    /// Create a new note with content
    pub fn new(content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: None,
            note_type: NoteType::Raw,
            title: None,
            content: content.into(),
            embedding: Vec::new(),
            source_id: None,
            entity_ids: Vec::new(),
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Builder pattern: set note type
    pub fn with_type(mut self, note_type: NoteType) -> Self {
        self.note_type = note_type;
        self
    }
    
    /// Builder pattern: set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    
    /// Builder pattern: set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
    
    /// Builder pattern: set source
    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }
    
    /// Builder pattern: add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Check if note has an embedding
    pub fn has_embedding(&self) -> bool {
        !self.embedding.is_empty()
    }
}

/// A note with additional context from graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicNote {
    pub note: Note,
    /// Notes that support this one
    #[serde(default)]
    pub supporting: Vec<Note>,
    /// Notes that contradict this one
    #[serde(default)]
    pub contradicting: Vec<Note>,
    /// Notes this one was derived from
    #[serde(default)]
    pub derived_from: Vec<Note>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_note_creation() {
        let note = Note::new("Test content")
            .with_type(NoteType::Claim)
            .with_title("Test Title")
            .with_tags(vec!["test".into(), "example".into()]);
        
        assert_eq!(note.content, "Test content");
        assert_eq!(note.note_type, NoteType::Claim);
        assert_eq!(note.title, Some("Test Title".into()));
        assert_eq!(note.tags.len(), 2);
        assert!(!note.has_embedding());
    }
    
    #[test]
    fn test_note_with_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let note = Note::new("Test").with_embedding(embedding.clone());
        
        assert!(note.has_embedding());
        assert_eq!(note.embedding, embedding);
    }
}
