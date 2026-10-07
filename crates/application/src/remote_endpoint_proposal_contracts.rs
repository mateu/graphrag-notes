//! Explicitly reviewed, quote-grounded remote edge-proposal contracts.
use graphrag_core::EdgeType;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Versioned with the semantic request payload used for idempotency.
pub const REMOTE_ENDPOINT_PROPOSAL_PAYLOAD_VERSION: u32 = 1;
pub const MAX_ENDPOINT_QUOTE_BYTES: usize = 2048;
pub const MAX_ENDPOINT_QUOTE_CHARS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EndpointProposalEvidence {
    /// Canonical `note:...` ID selected by the caller.
    pub id: String,
    /// Exact current revision obtained from a note snapshot.
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    /// A bounded, exact nonempty substring of the note's current content.
    #[schemars(length(min = 1, max = 1024))]
    pub quote: String,
}

/// This is a caller-selected semantic claim, not a label inferred from textual
/// similarity. `related_to` is appropriate for a non-causal association.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EndpointRelationship {
    Supports,
    Contradicts,
    DerivedFrom,
    RelatedTo,
}

impl From<EndpointRelationship> for EdgeType {
    fn from(value: EndpointRelationship) -> Self {
        match value {
            EndpointRelationship::Supports => Self::Supports,
            EndpointRelationship::Contradicts => Self::Contradicts,
            EndpointRelationship::DerivedFrom => Self::DerivedFrom,
            EndpointRelationship::RelatedTo => Self::RelatedTo,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteEndpointProposalRequest {
    /// Durable identity: reuse exactly for uncertain/retried writes.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    pub from: EndpointProposalEvidence,
    pub to: EndpointProposalEvidence,
    pub relationship: EndpointRelationship,
    /// Caller-written explanation linking the two supplied quotations.
    #[schemars(length(min = 1, max = 2048))]
    pub rationale: String,
    /// Required after the caller has reviewed both exact revisions and quotes.
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RemoteEndpointProposalOutcome {
    pub id: String,
    pub operation: String,
    pub actor: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RemoteEndpointProposalResponse {
    pub request_id: String,
    pub replayed: bool,
    pub outcome: RemoteEndpointProposalOutcome,
}
