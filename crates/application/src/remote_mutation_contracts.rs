//! Explicit bounded wire operations. Authentication supplies the actor;
//! clients supply the opening revision and a durable request identity.
use crate::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Immutable request-journal encoding, independent of local API contracts.
/// Changing this or v1 payload semantics requires an explicit replay migration.
pub const REMOTE_MUTATION_PAYLOAD_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemoteNoteSnapshot {
    pub id: String,
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    pub title: Option<String>,
    pub content: String,
    pub tags: Vec<String>,
    pub editable: bool,
    pub blocked_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteNotePatch {
    /// Replacement content: nonempty, at most 65536 UTF-8 bytes (not characters).
    #[schemars(length(min = 1, max = 65536))]
    pub content: Option<String>,
    /// Set the title; use clear_title to remove it.
    #[schemars(length(max = 512))]
    pub title: Option<String>,
    #[serde(default)]
    pub clear_title: bool,
    /// Replace all tags: at most 32 nonempty tags, each at most 64 characters.
    #[schemars(length(max = 32))]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteEditRequest {
    /// Durable identity: 1–128 characters and at most 256 UTF-8 bytes, no controls/whitespace.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    pub id: String,
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    pub patch: RemoteNotePatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteDeleteRequest {
    /// Durable identity: 1–128 characters and at most 256 UTF-8 bytes, no controls/whitespace.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    pub id: String,
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteDecisionRequest {
    /// Durable identity: 1–128 characters and at most 256 UTF-8 bytes, no controls/whitespace.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    pub id: String,
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    pub action: ProposalAction,
    #[schemars(length(max = 2048))]
    pub reason: Option<String>,
    pub confirmed: bool,
}

/// Exact original bounded outcome survives subsequent edits/deletion. Refresh
/// get_note/get_proposal for current content and a new revision; replay never
/// executes another effect or pretends its snapshot describes current state.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RemoteMutationOutcome {
    pub id: String,
    pub operation: String,
    pub previous_revision: String,
    pub actor: String,
    pub status: String,
    pub resulting_edge_id: Option<String>,
    pub cascade: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RemoteMutationResponse {
    /// Durable identity: 1–128 characters and at most 256 UTF-8 bytes, no controls/whitespace.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    pub replayed: bool,
    pub outcome: RemoteMutationOutcome,
}
