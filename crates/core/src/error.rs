//! Error types for the core domain

use thiserror::Error;

/// Core domain errors
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Note not found: {0}")]
    NoteNotFound(String),
    
    #[error("Entity not found: {0}")]
    EntityNotFound(String),
    
    #[error("Source not found: {0}")]
    SourceNotFound(String),
    
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type alias for core operations
pub type Result<T> = std::result::Result<T, CoreError>;
