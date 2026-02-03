//! Core domain types for GraphRAG Notes
//! 
//! This crate defines the fundamental data structures used throughout
//! the application: Notes, Entities, Sources, and their relationships.

pub mod note;
pub mod entity;
pub mod source;
pub mod edge;
pub mod error;

pub use note::{Note, NoteType, AtomicNote};
pub use entity::{Entity, EntityType};
pub use source::{Source, SourceType};
pub use edge::{Edge, EdgeType};
pub use error::{CoreError, Result};
