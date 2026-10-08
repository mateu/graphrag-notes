//! Separately authorized creation of reviewed endpoint proposals.
use crate::{
    tools::{application_failure, definition, failure, parse, success, ToolService},
    Capability, Principal,
};
use graphrag_application::{
    CallerIdentity, EndpointRelationship, RemoteEndpointProposalRequest,
    RemoteEndpointProposalResponse,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, Tool, ToolAnnotations};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct EvidenceInput {
    #[schemars(length(min = 1, max = 512))]
    id: String,
    #[schemars(length(min = 64, max = 64))]
    revision: String,
    #[schemars(length(min = 1, max = 1024))]
    quote: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ProposalInput {
    #[schemars(length(min = 1, max = 128))]
    request_id: String,
    from: EvidenceInput,
    to: EvidenceInput,
    relationship: EndpointRelationship,
    #[schemars(length(min = 1, max = 2048))]
    rationale: String,
    confirmed: bool,
}

impl From<ProposalInput> for RemoteEndpointProposalRequest {
    fn from(value: ProposalInput) -> Self {
        Self {
            request_id: value.request_id,
            from: graphrag_application::EndpointProposalEvidence {
                id: value.from.id,
                revision: value.from.revision,
                quote: value.from.quote,
            },
            to: graphrag_application::EndpointProposalEvidence {
                id: value.to.id,
                revision: value.to.revision,
                quote: value.to.quote,
            },
            relationship: value.relationship,
            rationale: value.rationale,
            confirmed: value.confirmed,
        }
    }
}

pub(crate) fn catalog() -> (Capability, Tool) {
    (Capability::Propose, definition::<ProposalInput, RemoteEndpointProposalResponse>("propose_endpoint_relationship", "Create exactly one pending, explicitly reviewed relationship proposal for two current canonical note snapshots. Supply each 64-character revision and a bounded exact quote found in that current note, select the relationship deliberately (similarity never supplies causality), explain the linkage, and set confirmed=true. Preserve an identical request_id and payload for safe replay; this capability cannot accept, reject, or scan.", false).with_annotations(ToolAnnotations::new().read_only(false).destructive(false).idempotent(true).open_world(false)))
}

pub(crate) async fn call(
    service: &ToolService,
    request: CallToolRequestParams,
    principal: Principal,
) -> CallToolResult {
    let input: ProposalInput = match parse(request.arguments) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let Ok(permit) = service.write_gate.clone().try_acquire_owned() else {
        return failure(
            "busy",
            "Write capacity is busy; retain the identical request_id and payload for retry.",
            true,
        );
    };
    let Some(guard) = service.writes.start() else {
        return failure(
            "service_unavailable",
            "The service is shutting down; retain the identical request_id and payload for retry.",
            true,
        );
    };
    let application = Arc::clone(&service.application);
    match tokio::spawn(async move {
        let (_permit, _guard) = (permit, guard);
        application
            .propose_endpoint_remote(
                CallerIdentity {
                    instance_id: principal.instance_id,
                },
                input.into(),
            )
            .await
    })
    .await
    {
        Ok(Ok(value)) => success(value),
        Ok(Err(error)) => application_failure(error),
        Err(_) => failure(
            "internal",
            "Proposal outcome is uncertain. Retain the identical request_id and payload for replay; refresh both snapshots after an authoritative result.",
            true,
        ),
    }
}
