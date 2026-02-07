//! Agent error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Database error: {0}")]
    Database(#[from] graphrag_db::DbError),

    #[error("ML worker error: {0}")]
    MlWorker(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Processing error: {0}")]
    Processing(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
