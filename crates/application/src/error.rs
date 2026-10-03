use graphrag_agents::AgentError;
use graphrag_db::DbError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
    #[error("{0}")]
    Compatibility(String),
    #[error("{0}")]
    ServiceUnreachable(String),
    #[error("Action cancelled at a safe boundary; retain any draft and inspect the durable outcome before retrying.")]
    Cancelled,
    #[error("{0}")]
    Internal(String),
}

impl From<DbError> for ApplicationError {
    fn from(error: DbError) -> Self {
        match error {
            DbError::NotFound(..) => Self::NotFound(error.to_string()),
            DbError::RemoteJobCancelled(_) => Self::Cancelled,
            DbError::RemoteJobOwnershipLost(_)
            | DbError::RemoteJobSourceConflict(_)
            | DbError::NoteRevisionConflict(_)
            | DbError::MutationRevisionConflict(_) => Self::RevisionConflict(error.to_string()),
            DbError::RemoteRequestConflict { .. } => Self::RevisionConflict(error.to_string()),
            DbError::InvalidRemoteRequest(_) | DbError::InvalidMutationRequest(_) => {
                Self::Validation(error.to_string())
            }
            DbError::EmbeddingCompatibility { .. }
            | DbError::LegacyEmbeddingMetadata { .. }
            | DbError::UnsupportedSchemaVersion { .. } => Self::Compatibility(error.to_string()),
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApplicationFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl ApplicationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Validation(_) => "validation",
            Self::NotFound(_) => "not_found",
            Self::RevisionConflict(_) => "conflict",
            Self::ProviderUnavailable(_) => "provider_unavailable",
            Self::Compatibility(_) => "compatibility",
            Self::ServiceUnreachable(_) => "service_unreachable",
            Self::Cancelled => "cancelled",
            Self::Internal(_) => "internal",
        }
    }

    pub fn failure(&self) -> ApplicationFailure {
        ApplicationFailure {
            code: self.code().into(),
            message: self.to_string(),
            retryable: matches!(
                self,
                Self::ProviderUnavailable(_) | Self::ServiceUnreachable(_)
            ),
        }
    }
}
