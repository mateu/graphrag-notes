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
