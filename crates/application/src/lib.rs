//! Shared local operations for CLI, terminal, and future service adapters.
//! Storage ownership and client interaction remain outside this boundary.

mod contracts;
mod embedded;
mod error;
mod inference;
mod remote_contracts;
mod remote_endpoint_proposal_contracts;
mod remote_endpoint_proposals;
mod remote_enrichment;
mod remote_job_contracts;
mod remote_jobs;
pub use remote_enrichment::*;
mod remote_mutation_contracts;
mod remote_mutations;
mod remote_operations;
mod remote_status;

pub use contracts::*;
pub use embedded::{
    decide_proposal, proposal_card, proposal_revision, validate_neighbors, validate_record_id,
    validate_record_revision, EmbeddedApplication,
};
pub use error::{ApplicationError, ApplicationFailure, ApplicationResult};
pub use remote_contracts::*;
pub use remote_endpoint_proposal_contracts::*;
pub use remote_endpoint_proposals::remote_endpoint_proposal_fingerprint;
pub use remote_job_contracts::*;
pub use remote_mutation_contracts::*;
pub use remote_mutations::remote_mutation_fingerprint;
pub use remote_operations::remote_capture_fingerprint;
pub use remote_status::*;

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
    /// Read-only, bounded owner-side evidence. Never initialize storage or probe inference.
    async fn service_readiness(&self) -> ApplicationResult<ApplicationReadiness> {
        Ok(ApplicationReadiness::default())
    }
    async fn note_snapshot(&self, reference: RecordRef) -> ApplicationResult<RemoteNoteSnapshot> {
        let _ = reference;
        Err(ApplicationError::Validation(
            "Remote note editing is unavailable.".into(),
        ))
    }
    async fn edit_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteEditRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<RemoteMutationResponse> {
        let _ = (caller, request, cancellation);
        Err(ApplicationError::Validation(
            "Remote note editing is unavailable.".into(),
        ))
    }
    async fn delete_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteDeleteRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        let _ = (caller, request);
        Err(ApplicationError::Validation(
            "Remote note deletion is unavailable.".into(),
        ))
    }
    async fn decide_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteDecisionRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        let _ = (caller, request);
        Err(ApplicationError::Validation(
            "Remote proposal decisions are unavailable.".into(),
        ))
    }

    async fn propose_endpoint_remote(
        &self,
        _caller: CallerIdentity,
        _request: RemoteEndpointProposalRequest,
    ) -> ApplicationResult<RemoteEndpointProposalResponse> {
        Err(ApplicationError::Compatibility(
            "Remote endpoint proposals are unavailable in this adapter".into(),
        ))
    }

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
    async fn prepare_source_enrichment(
        &self,
        _caller: CallerIdentity,
        _request: PrepareSourceEnrichment,
    ) -> ApplicationResult<ReviewedEnrichmentReceipt> {
        Err(ApplicationError::Compatibility(
            "Enrichment unavailable".into(),
        ))
    }
    async fn execute_source_enrichment(
        &self,
        _caller: CallerIdentity,
        _request: ExecuteSourceEnrichment,
        _rollback: bool,
    ) -> ApplicationResult<UploadAdmission> {
        Err(ApplicationError::Compatibility(
            "Enrichment unavailable".into(),
        ))
    }
    async fn read_source_enrichment_plan(
        &self,
        _caller: CallerIdentity,
        _id: &str,
    ) -> ApplicationResult<ReviewedEnrichmentReceipt> {
        Err(ApplicationError::Compatibility(
            "Enrichment unavailable".into(),
        ))
    }
    async fn upload_source(
        &self,
        _caller: CallerIdentity,
        _request: UploadSourceRequest,
    ) -> ApplicationResult<UploadAdmission> {
        Err(ApplicationError::Compatibility(
            "Uploaded jobs are unavailable in this adapter".into(),
        ))
    }
    /// Authenticated transport lookup; the unscoped sibling is trusted-internal only.
    async fn get_owned_uploaded_source(
        &self,
        caller: CallerIdentity,
        id: &str,
    ) -> ApplicationResult<UploadedSource> {
        let source = self.get_uploaded_source(id).await?;
        if source.instance_id != caller.instance_id {
            return Err(ApplicationError::NotFound(
                "This instance has no uploaded source with that ID".into(),
            ));
        }
        Ok(source)
    }
    async fn get_uploaded_source(&self, _id: &str) -> ApplicationResult<UploadedSource> {
        Err(ApplicationError::Compatibility(
            "Uploaded sources are unavailable in this adapter".into(),
        ))
    }
    async fn lookup_uploaded_source(
        &self,
        _caller: CallerIdentity,
        _key: &str,
    ) -> ApplicationResult<UploadedSource> {
        Err(ApplicationError::Compatibility(
            "Uploaded source lookup is unavailable in this adapter".into(),
        ))
    }
    async fn delete_uploaded_source(
        &self,
        _caller: CallerIdentity,
        _request: DeleteUploadedSourceRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        Err(ApplicationError::Compatibility(
            "Uploaded source retirement is unavailable in this adapter".into(),
        ))
    }
    async fn get_remote_job(
        &self,
        _caller: CallerIdentity,
        _id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        Err(ApplicationError::Compatibility(
            "Uploaded jobs are unavailable in this adapter".into(),
        ))
    }
    async fn list_remote_jobs(
        &self,
        _caller: CallerIdentity,
        _limit: usize,
    ) -> ApplicationResult<RemoteJobList> {
        Err(ApplicationError::Compatibility(
            "Uploaded jobs are unavailable in this adapter".into(),
        ))
    }
    async fn cancel_remote_job(
        &self,
        _caller: CallerIdentity,
        _id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        Err(ApplicationError::Compatibility(
            "Uploaded jobs are unavailable in this adapter".into(),
        ))
    }
    async fn resume_remote_job(
        &self,
        _caller: CallerIdentity,
        _id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        Err(ApplicationError::Compatibility(
            "Uploaded jobs are unavailable in this adapter".into(),
        ))
    }
    async fn reconcile_remote_jobs(&self, _epoch: &str) -> ApplicationResult<()> {
        Ok(())
    }
    async fn claim_remote_job(
        &self,
        _epoch: &str,
        _worker: &str,
    ) -> ApplicationResult<Option<RemoteJobExecution>> {
        Ok(None)
    }
    async fn remote_job_cancel_requested(
        &self,
        _execution: &RemoteJobExecution,
    ) -> ApplicationResult<bool> {
        Ok(false)
    }
    async fn execute_remote_job(
        &self,
        _execution: RemoteJobExecution,
        _cancel: ActionCancellation,
    ) -> ApplicationResult<()> {
        Ok(())
    }
    async fn interrupt_remote_job(&self, _execution: RemoteJobExecution) -> ApplicationResult<()> {
        Ok(())
    }
    /// Confirm a failed worker's durable terminal outcome or loss of ownership.
    /// Errors retain the worker lease for retry; storage failures are not success.
    async fn recover_remote_job(
        &self,
        execution: RemoteJobExecution,
        _error_code: String,
    ) -> ApplicationResult<()> {
        self.interrupt_remote_job(execution).await
    }
}
