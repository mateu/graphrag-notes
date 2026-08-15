//! Agent error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Database error: {0}")]
    Database(#[from] graphrag_db::DbError),

    #[error("Inference service error: {0}")]
    InferenceService(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Processing error: {0}")]
    Processing(String),

    /// A durable processing job reached its persisted failed terminal state
    /// after completing some of its scoped items. Keeping this distinct from
    /// ordinary processing errors lets CLI callers report the documented
    /// partial-failure outcome without guessing from an error message.
    #[error(
        "Durable processing job {job_id} partially failed after {completed} completed and {failed} failed items: {message}"
    )]
    DurablePartialFailure {
        job_id: String,
        completed: u64,
        failed: u64,
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, AgentError>;
