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
