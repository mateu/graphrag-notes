//! Opt-in, read-only Alchemist synthesis over accepted relations.
//!
//! `synthesize_notes` requires the dedicated `alchemist` capability, which
//! grants no other tool, plus `read`, because the synthesis quotes note text.
//! The tool never writes; its output is not stored and is not evidence.
use crate::{
    tools::{application_failure, definition, failure, parse, success, ToolService},
    Capability, Principal,
};
use graphrag_application::{
    ActionCancellation, ApplicationError, SynthesisError, SynthesisRequest, SynthesisResponse,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, Tool};
use tokio_util::sync::CancellationToken;

pub(crate) const TOOL: &str = "synthesize_notes";

/// Listing and calling need both capabilities.
pub(crate) fn allows(principal: &Principal) -> bool {
    principal.allows(Capability::Alchemist) && principal.allows(Capability::Read)
}

pub(crate) fn catalog() -> (Capability, Tool) {
    (
        Capability::Alchemist,
        definition::<SynthesisRequest, SynthesisResponse>(
            TOOL,
            "Read-only, citation-verified synthesis of what the notes collectively say. Supply exactly one of query (reuses build_context retrieval) or seed (a canonical note:ID; its neighbourhood over accepted edges, at most 2 hops). Every returned claim cites a verbatim span of a chunk at its pinned revision; unverifiable claims are dropped and counted (or rejected under citation_policy=strict). Conflicts are surfaced, never resolved; strength is a lineage-deduplicated origin count, never a probability. Nothing is stored, proposed or written, and the output is not evidence. Disabled unless the server opts in.",
            true,
        ),
    )
}

fn synthesis_failure(error: SynthesisError) -> CallToolResult {
    match error {
        SynthesisError::Disabled => failure(
            "alchemist_disabled",
            "Synthesis is disabled on this server; the owner must opt in with [alchemist] enabled = true.",
            false,
        ),
        SynthesisError::DeadlineExceeded { .. } => failure(
            "deadline_exceeded",
            "Synthesis exceeded its deadline; nothing was returned or stored. Retry later or with smaller budgets.",
            true,
        ),
        SynthesisError::Unverifiable { .. } => failure(
            "unverifiable_synthesis",
            "The draft could not be verified against its cited chunks under the active citation policy; nothing was returned or stored.",
            true,
        ),
        SynthesisError::Application(ApplicationError::Validation(_)) => failure(
            "invalid_input",
            "Supply exactly one of query (1–1024 characters) or seed (a canonical note:ID), with budgets between 1 and the server's configured ceilings.",
            false,
        ),
        SynthesisError::Application(ApplicationError::NotFound(_)) => failure(
            "not_found",
            "The seed is not a visible note. Search again to refresh its ID.",
            false,
        ),
        SynthesisError::Application(error) => application_failure(error),
    }
}

pub(crate) async fn call(
    service: &ToolService,
    request: CallToolRequestParams,
    principal: Principal,
    client_cancelled: CancellationToken,
) -> CallToolResult {
    // The catalog filter already requires both; keep the check local too.
    if !allows(&principal) {
        return failure(
            "forbidden",
            "This instance token does not grant the required capability.",
            false,
        );
    }
    let input: SynthesisRequest = match parse(request.arguments) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let cancellation = ActionCancellation::new();
    let result = tokio::select! {
        result = service.application.synthesize_notes(input, cancellation.clone()) => result,
        _ = client_cancelled.cancelled() => {
            cancellation.cancel();
            Err(ApplicationError::Cancelled.into())
        }
    };
    match result {
        Ok(response) => success(response),
        Err(error) => synthesis_failure(error),
    }
}
