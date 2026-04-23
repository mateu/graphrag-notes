//! Core domain types for GraphRAG Notes
//!
//! This crate defines the fundamental data structures used throughout
//! the application: Notes, Entities, Sources, and their relationships.

pub mod chat_export;
pub mod edge;
pub mod entity;
pub mod error;
pub mod note;
pub mod source;

pub use chat_export::{ChatConversation, ChatExport, ChatMessage, MessageRole};
pub use edge::{Edge, EdgeType};
pub use entity::{Entity, EntityType};
pub use error::{CoreError, Result};
pub use note::{AtomicNote, Note, NoteType};
pub use source::{Source, SourceType};

/// Format a SurrealDB [`RecordId`] as a canonical `table:key` string.
///
/// Used for human-readable output in logs, error messages, and CLI output.
pub fn record_id_to_string(id: &surrealdb::types::RecordId) -> String {
    match &id.key {
        surrealdb::types::RecordIdKey::String(s) => format!("{}:{}", id.table, s),
        surrealdb::types::RecordIdKey::Number(n) => format!("{}:{}", id.table, n),
        surrealdb::types::RecordIdKey::Uuid(u) => format!("{}:{}", id.table, u),
        surrealdb::types::RecordIdKey::Array(a) => format!("{}:{:?}", id.table, a),
        surrealdb::types::RecordIdKey::Object(o) => format!("{}:{:?}", id.table, o),
        surrealdb::types::RecordIdKey::Range(r) => format!("{}:{:?}", id.table, r),
    }
}
