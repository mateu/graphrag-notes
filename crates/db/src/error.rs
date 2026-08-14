//! Database error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DbError {
    #[error("Database connection error: {0}")]
    Connection(String),

    #[error("Record not found: {0} with id {1}")]
    NotFound(String, String),

    #[error("Failed to create {0}")]
    CreateFailed(String),

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Schema initialization failed: {0}")]
    SchemaInit(String),

    #[error(
        "Database schema migration {version} is newer than this binary supports (latest: {latest})"
    )]
    UnsupportedSchemaVersion { version: u32, latest: u32 },

    #[error("Database schema migration history is inconsistent: {0}")]
    MigrationHistory(String),

    #[error(
        "embedding compatibility check failed: database uses {stored_provider}/{stored_model} ({stored_dimension} dimensions), but the active provider is {active_provider}/{active_model} ({active_dimension} dimensions). Reindex with: graphrag reindex --all"
    )]
    EmbeddingCompatibility {
        stored_provider: String,
        stored_model: String,
        stored_dimension: usize,
        active_provider: String,
        active_model: String,
        active_dimension: usize,
    },

    #[error(
        "database contains {vector_records} legacy vector-bearing records without embedding metadata. Reindex with: graphrag reindex --all"
    )]
    LegacyEmbeddingMetadata { vector_records: usize },

    #[error("Database schema migration {version} ({name}) failed: {reason}")]
    MigrationFailed {
        version: u32,
        name: String,
        reason: String,
    },

    #[error("SurrealDB error: {0}")]
    Surreal(#[from] surrealdb::Error),
}

pub type Result<T> = std::result::Result<T, DbError>;
