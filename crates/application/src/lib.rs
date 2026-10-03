//! Shared local operations for CLI, terminal, and future service adapters.
//! Storage ownership and client interaction remain outside this boundary.

mod contracts;
mod embedded;
mod error;
mod inference;
mod remote_contracts;
mod remote_operations;

pub use contracts::*;
pub use embedded::{
    decide_proposal, proposal_card, proposal_revision, validate_neighbors, validate_record_id,
    validate_record_revision, EmbeddedApplication,
};
pub use error::{ApplicationError, ApplicationFailure, ApplicationResult};
pub use remote_contracts::*;
pub use remote_operations::remote_capture_fingerprint;

use async_trait::async_trait;
use graphrag_core::{Note, ProposedEdgeStatus};
use graphrag_db::{repository::DbStats, RecordInspection};

#[async_trait]
pub trait ApplicationOperations: Send + Sync {
    async fn search(
        &self,
        request: SearchRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<Vec<RecordSummary>>;
    async fn recent(&self, limit: usize) -> ApplicationResult<Vec<RecordSummary>>;
    async fn inspect(
        &self,
        reference: RecordRef,
        neighbors: usize,
    ) -> ApplicationResult<RecordInspection>;
    async fn capture(
        &self,
        request: CaptureRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<Note>;
    async fn source_status(&self, limit: usize) -> ApplicationResult<Vec<SourceStatus>>;
    async fn proposals(
        &self,
        status: Option<ProposedEdgeStatus>,
        limit: usize,
    ) -> ApplicationResult<Vec<ProposalCard>>;
    async fn proposal(&self, id: &str) -> ApplicationResult<ProposalCard>;
    async fn decide_proposal(
        &self,
        request: ProposalDecisionRequest,
    ) -> ApplicationResult<ProposalCard>;
    async fn stats(&self) -> ApplicationResult<DbStats>;
}

/// Shared remote foundation; transports authenticate the caller before invoking it.
#[async_trait]
pub trait RemoteApplicationOperations: ApplicationOperations {
    async fn build_context(
        &self,
        request: BuildContextRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<ContextResponse>;
    async fn capture_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteCaptureRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<RemoteCaptureResponse>;
}
