//! Readable manual-note snapshots and separately authorized mutation tools.
use crate::{
    tools::{application_failure, definition, failure, parse, success, ToolService},
    Capability, Principal,
};
use graphrag_application::{
    ActionCancellation, CallerIdentity, ProposalAction, RecordRef, RemoteDecisionRequest,
    RemoteDeleteRequest, RemoteEditRequest, RemoteMutationResponse, RemoteNoteSnapshot,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, Tool, ToolAnnotations};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SnapshotInput {
    #[schemars(length(min = 1, max = 512))]
    id: String,
    #[schemars(length(min = 64, max = 64))]
    revision: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ProposalInput {
    #[schemars(length(min = 1, max = 512))]
    id: String,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ProposalList {
    status: Option<String>,
    #[serde(default = "default_limit")]
    #[schemars(range(min = 1, max = 200))]
    limit: usize,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct ProposalEndpointOutput {
    id: String,
    available: bool,
    title: Option<String>,
    excerpt: Option<String>,
    revision: Option<String>,
    provenance: Option<serde_json::Value>,
    warnings: Vec<String>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ProposalStatusOutput {
    Pending,
    Accepting,
    Accepted,
    Rejected,
    Superseded,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct ProposalOutput {
    id: String,
    status: ProposalStatusOutput,
    edge_type: String,
    confidence: f32,
    reason: String,
    generator: String,
    generator_version: Option<String>,
    model: Option<String>,
    updated_at: String,
    reviewed_at: Option<String>,
    reviewer: Option<String>,
    action_reason: Option<String>,
    acceptance_is_manual: Option<bool>,
    supersession_reason: Option<String>,
    resulting_edge_id: Option<String>,
    from: ProposalEndpointOutput,
    to: ProposalEndpointOutput,
    accept_allowed: bool,
    accept_blocked_reason: Option<String>,
    revision: String,
}
fn encode<T: serde::Serialize, O: serde::de::DeserializeOwned + serde::Serialize>(
    value: T,
) -> CallToolResult {
    match serde_json::to_value(value).and_then(serde_json::from_value::<O>) {
        Ok(value) => success(value),
        Err(_) => failure(
            "internal",
            "Cannot encode proposal cards; contact the service owner.",
            false,
        ),
    }
}
fn default_limit() -> usize {
    10
}

fn read_id_valid(id: &str) -> bool {
    !id.is_empty() && id.len() <= 512 && id.trim() == id && !id.chars().any(char::is_control)
}
pub(crate) fn allows_catalog(principal: &Principal, capability: Capability, name: &str) -> bool {
    if name == "decide_proposal" {
        [Capability::Accept, Capability::Reject, Capability::Undo]
            .into_iter()
            .any(|c| principal.allows(c))
    } else {
        principal.allows(capability)
    }
}
pub(crate) fn catalog() -> Vec<(Capability, Tool)> {
    vec![
        (Capability::Read,definition::<SnapshotInput,RemoteNoteSnapshot>("get_note","Read a manual-note edit snapshot with tags and ownership. Imported file/chat notes cannot be edited or deleted remotely.",true)),
        (Capability::Read,definition::<ProposalList,Vec<ProposalOutput>>("list_proposals","List bounded proposal cards with revisions, endpoint evidence and allowed actions.",true)),
        (Capability::Read,definition::<ProposalInput,ProposalOutput>("get_proposal","Read current proposal and endpoint revisions before an explicitly confirmed decision.",true)),
        (Capability::Edit,definition::<RemoteEditRequest,RemoteMutationResponse>("edit_note","Apply an explicit patch to a manual note using its revision. Content limit is 65536 UTF-8 bytes, title 512 characters, tags 32x64 characters. Preserve identical request_id/payload after an uncertain outcome; changed content uses server providers.",false)),
        (Capability::Delete,definition::<RemoteDeleteRequest,RemoteMutationResponse>("delete_note","Delete a manual note and its graph/provenance links only with confirmed=true and its current revision. Preserve the identical request_id/payload for safe replay.",false).with_annotations(ToolAnnotations::new().read_only(false).destructive(true).idempotent(true).open_world(false))),
        (Capability::Accept,definition::<RemoteDecisionRequest,RemoteMutationResponse>("decide_proposal","Explicitly confirm accept, reject or undo with the current proposal revision. Each action requires its independently granted capability. Authentication supplies reviewer identity; reason limit 2048 characters. Durable replay returns the original bounded outcome.",false).with_annotations(ToolAnnotations::new().read_only(false).destructive(true).idempotent(true).open_world(false))),
    ]
}
pub(crate) async fn call(
    service: &ToolService,
    request: CallToolRequestParams,
    principal: Principal,
) -> CallToolResult {
    match request.name.as_ref() {
        "get_note" => {
            let input: SnapshotInput = match parse(request.arguments) {
                Ok(v) => v,
                Err(e) => return e,
            };
            if !read_id_valid(&input.id)
                || input.revision.as_ref().is_some_and(|revision| {
                    revision.len() != 64
                        || !revision
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
            {
                return failure("invalid_input","Use a canonical record ID of at most 512 UTF-8 bytes and a 64-character revision from its snapshot.",false);
            }
            match service
                .application
                .note_snapshot(RecordRef {
                    id: input.id,
                    revision: input.revision,
                })
                .await
            {
                Ok(v) => success(v),
                Err(e) => application_failure(e),
            }
        }
        "get_proposal" => {
            let input: ProposalInput = match parse(request.arguments) {
                Ok(v) => v,
                Err(e) => return e,
            };
            if !read_id_valid(&input.id) {
                return failure(
                    "invalid_input",
                    "Use a canonical proposed_edge:ID of at most 512 UTF-8 bytes.",
                    false,
                );
            }
            match service.application.proposal(&input.id).await {
                Ok(v) => encode::<_, ProposalOutput>(v),
                Err(e) => application_failure(e),
            }
        }
        "list_proposals" => {
            let input: ProposalList = match parse(request.arguments) {
                Ok(v) => v,
                Err(e) => return e,
            };
            if !(1..=200).contains(&input.limit) {
                return failure(
                    "invalid_input",
                    "Proposal limit must be between 1 and 200.",
                    false,
                );
            }
            let status = match input.status.as_deref() {
                None => None,
                Some(value) => match serde_json::from_value(serde_json::json!(value)) {
                    Ok(value) => Some(value),
                    Err(_) => return failure("invalid_input", "Unknown proposal status.", false),
                },
            };
            match service.application.proposals(status, input.limit).await {
                Ok(v) => encode::<_, Vec<ProposalOutput>>(v),
                Err(e) => application_failure(e),
            }
        }
        name => {
            enum Mutation {
                Edit(RemoteEditRequest),
                Delete(RemoteDeleteRequest),
                Decision(RemoteDecisionRequest),
            }
            let mutation = match name {
                "edit_note" => match parse(request.arguments) {
                    Ok(v) => Mutation::Edit(v),
                    Err(e) => return e,
                },
                "delete_note" => match parse(request.arguments) {
                    Ok(v) => Mutation::Delete(v),
                    Err(e) => return e,
                },
                "decide_proposal" => {
                    let input: RemoteDecisionRequest = match parse(request.arguments) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let required = match input.action {
                        ProposalAction::Accept => Capability::Accept,
                        ProposalAction::Reject => Capability::Reject,
                        ProposalAction::Undo => Capability::Undo,
                    };
                    if !principal.allows(required) {
                        return failure(
                            "forbidden",
                            "This token does not grant the requested proposal action.",
                            false,
                        );
                    }
                    Mutation::Decision(input)
                }
                _ => return failure("invalid_input", "Unknown tool; refresh tools/list.", false),
            };
            let Ok(permit) = service.write_gate.clone().try_acquire_owned() else {
                return failure("busy","Write capacity is busy; retain the identical request_id and payload for retry.",true);
            };
            let Some(guard) = service.writes.start() else {
                return failure("service_unavailable","The service is shutting down; retain the identical request_id and payload for retry.",true);
            };
            let application = Arc::clone(&service.application);
            let task = tokio::spawn(async move {
                let (_permit, _guard) = (permit, guard);
                let caller = CallerIdentity {
                    instance_id: principal.instance_id,
                };
                match mutation {
                    Mutation::Edit(request) => {
                        application
                            .edit_remote(caller, request, ActionCancellation::new())
                            .await
                    }
                    Mutation::Delete(request) => application.delete_remote(caller, request).await,
                    Mutation::Decision(request) => application.decide_remote(caller, request).await,
                }
            });
            match task.await {Ok(Ok(v))=>success(v),Ok(Err(e))=>application_failure(e),Err(_)=>failure("internal","Mutation outcome is uncertain. Retain the identical request_id and payload for replay; refresh the snapshot after an authoritative result.",true)}
        }
    }
}
