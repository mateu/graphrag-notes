//! Note types - the atomic units of knowledge

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;

/// The type/classification of a note
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, SurrealValue)]
#[serde(rename_all = "snake_case")]
#[surreal(crate = "surrealdb_types")]
#[surreal(untagged, lowercase)]
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
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct Note {
    /// Unique identifier (maps to SurrealDB record ID)
    pub id: Option<RecordId>,

    /// The type of this note
    pub note_type: NoteType,

    /// Title or summary (optional)
    pub title: Option<String>,

    /// The actual content
    pub content: String,

    /// Vector embedding (current Rust path uses 1024-dimensional embeddings)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[surreal(default, skip_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,

    /// Source this note was derived from
    pub source_id: Option<RecordId>,

    /// Source generation that produced this derived note. `None` is reserved
    /// for manual/legacy records and is never removed by source refreshes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_generation: Option<u64>,

    /// Stable identity of a structure-aware source chunk. Manual and legacy
    /// notes leave this unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_key: Option<String>,

    /// Structural location independent of content, used to reconcile a small
    /// edit with the existing note record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_location_key: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_ordinal: Option<u64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunk_heading_path: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_line: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end_line: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_byte: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end_byte: Option<u64>,

    /// Key of the predecessor from which the display content copied its tail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_overlap_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_overlap_chars: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,

    /// Search/embedding text. For Markdown this prefixes heading context while
    /// preserving `content` as the exact displayed source text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_content: Option<String>,

    /// Extracted entities mentioned in this note
    ///
    /// This remains in-memory/app-facing only; persisted note↔entity links live
    /// in the `mentions` edge table rather than on the note record itself.
    #[serde(default, skip_serializing)]
    #[surreal(skip)]
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
            source_generation: None,
            chunk_key: None,
            chunk_location_key: None,
            chunk_ordinal: None,
            chunk_heading_path: Vec::new(),
            source_start_line: None,
            source_end_line: None,
            source_start_byte: None,
            source_end_byte: None,
            chunk_overlap_from: None,
            chunk_overlap_chars: None,
            content_hash: None,
            search_content: None,
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
    pub fn with_source(mut self, source_id: RecordId) -> Self {
        self.source_id = Some(source_id);
        self
    }

    /// Mark this note as generated by a specific source generation.
    pub fn with_source_generation(mut self, generation: u64) -> Self {
        self.source_generation = Some(generation);
        self
    }

    /// Attach deterministic Markdown chunk provenance.
    pub fn with_chunk_metadata(
        mut self,
        chunk_key: String,
        location_key: String,
        ordinal: usize,
        heading_path: Vec<String>,
        start_line: usize,
        end_line: usize,
        start_byte: usize,
        end_byte: usize,
        overlap_from: Option<String>,
        overlap_chars: usize,
        content_hash: String,
        search_content: String,
    ) -> Self {
        self.chunk_key = Some(chunk_key);
        self.chunk_location_key = Some(location_key);
        self.chunk_ordinal = Some(ordinal as u64);
        self.chunk_heading_path = heading_path;
        self.source_start_line = Some(start_line as u64);
        self.source_end_line = Some(end_line as u64);
        self.source_start_byte = Some(start_byte as u64);
        self.source_end_byte = Some(end_byte as u64);
        self.chunk_overlap_from = overlap_from;
        self.chunk_overlap_chars = (overlap_chars > 0).then_some(overlap_chars as u64);
        self.content_hash = Some(content_hash);
        self.search_content = Some(search_content);
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
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
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
