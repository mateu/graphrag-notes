use graphrag_agents::AgentError;
use graphrag_db::DbError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error("{0}")]
    Validation(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    RevisionConflict(String),
    #[error("{0}")]
    ProviderUnavailable(String),
    #[error("Action cancelled before persistence; retain the draft and retry when ready.")]
    Cancelled,
    #[error("{0}")]
    Internal(String),
}

impl From<DbError> for ApplicationError {
    fn from(error: DbError) -> Self {
        match error {
            DbError::NotFound(..) => Self::NotFound(error.to_string()),
            DbError::NoteRevisionConflict(_) => Self::RevisionConflict(error.to_string()),
            _ => Self::Internal(error.to_string()),
        }
    }
}

impl From<AgentError> for ApplicationError {
    fn from(error: AgentError) -> Self {
        match error {
            AgentError::Database(error) => error.into(),
            AgentError::NotFound(message) => Self::NotFound(message),
            AgentError::Cancelled => Self::Cancelled,
            AgentError::InferenceService(_) | AgentError::Http(_) => {
                Self::ProviderUnavailable(error.to_string())
            }
            _ => Self::Internal(error.to_string()),
        }
    }
}

pub type ApplicationResult<T> = Result<T, ApplicationError>;
