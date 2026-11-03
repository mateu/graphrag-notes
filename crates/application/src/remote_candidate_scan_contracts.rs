//! Bounded, review-only relationship candidate scan contracts.
//!
//! A scan nominates cross-source `related_to` candidates and, outside a dry
//! run, records them as pending proposals for explicit review. It never
//! accepts, rejects or refreshes a proposal and never writes an accepted edge.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Version of the scan request/receipt contract.
pub const RELATIONSHIP_CANDIDATE_SCAN_CONTRACT: u32 = 1;
pub const MAX_SCAN_CANDIDATES: usize = 50;
pub const MAX_SCAN_ENDPOINTS: usize = 200;
pub const MIN_SCAN_DEADLINE_MS: u64 = 100;
pub const MAX_SCAN_DEADLINE_MS: u64 = 60_000;
pub const MAX_SCAN_SCOPE_IDS: usize = 50;
pub const MAX_SCAN_PER_ENDPOINT: usize = 50;

/// Which notes are examined as anchors.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelationshipCandidateScope {
    /// These canonical `note:` IDs (missing or embedding-free notes are skipped).
    Notes {
        #[schemars(length(min = 1, max = 50))]
        ids: Vec<String>,
    },
    /// Every visible note owned by these canonical `source:` IDs.
    Sources {
        #[schemars(length(min = 1, max = 50))]
        ids: Vec<String>,
    },
    /// Notes without an accepted relation of any type in either direction.
    Orphans {},
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RelationshipCandidateScanRequest {
    pub scope: RelationshipCandidateScope,
    /// Restrict the other endpoint to these canonical `source:` IDs; empty
    /// means any other source.
    #[serde(default)]
    #[schemars(length(max = 50))]
    pub counterpart_source_ids: Vec<String>,
    /// Resume after this canonical `note:` ID, from a previous receipt's
    /// `next_after_note_id`. Anchors are examined in record-ID order.
    #[serde(default)]
    pub after_note_id: Option<String>,
    /// Required. Most candidates returned (and persisted outside a dry run).
    #[schemars(range(min = 1, max = 50))]
    pub max_candidates: usize,
    /// Required. Most anchor notes examined.
    #[schemars(range(min = 1, max = 200))]
    pub max_endpoints: usize,
    /// Required. Wall-clock budget in milliseconds.
    #[schemars(range(min = 100, max = 60000))]
    pub deadline_ms: u64,
    /// Optional; can only raise the operator's similarity floor.
    #[serde(default)]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub min_similarity: Option<f32>,
    /// Optional; can only lower the operator's per-endpoint candidate cap.
    #[serde(default)]
    #[schemars(range(min = 1, max = 50))]
    pub max_per_endpoint: Option<usize>,
    /// Required. `true` changes nothing; `false` records pending proposals only.
    pub dry_run: bool,
}

/// Every resolved budget, echoed so callers can verify what actually ran.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateScanLimits {
    pub max_candidates: usize,
    pub max_endpoints: usize,
    pub deadline_ms: u64,
}

/// The effective, operator-owned precision policy after request tightening.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RelationshipCandidatePolicy {
    pub relationship: String,
    pub generator: String,
    pub generator_version: String,
    pub similarity_floor: f32,
    pub neighbors_per_endpoint: usize,
    pub max_per_endpoint: usize,
    /// Shared names mentioned by more notes than this are generic.
    pub max_entity_note_frequency: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipCandidateStopReason {
    Completed,
    EndpointBudget,
    CandidateBudget,
    Deadline,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipCandidateSignal {
    ExplicitCrossReference,
    SharedSpecificEntity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipCrossReferenceKind {
    RecordId,
    SourceUri,
    Wikilink,
    FileName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipCrossReferenceDirection {
    FromMentionsTo,
    ToMentionsFrom,
    Both,
}

/// Why nominated pairs were not selected. Same-source neighbours are
/// excluded before nomination and are not counted.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateRejections {
    pub same_origin: usize,
    pub duplicate_content: usize,
    pub no_second_signal: usize,
    pub generic_entities_only: usize,
    pub already_connected: usize,
    pub already_proposed: usize,
    pub prior_decision: usize,
    pub endpoint_cap: usize,
    pub over_budget: usize,
    /// An endpoint was deleted or its source generation retired before the
    /// pending proposal could be recorded.
    #[serde(default)]
    pub vanished: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateEndpoint {
    pub id: String,
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateEntity {
    pub name: String,
    /// Distinct notes whose mention evidence carries this name.
    pub note_frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateCrossReference {
    pub kind: RelationshipCrossReferenceKind,
    pub direction: RelationshipCrossReferenceDirection,
}

/// The pending proposal recorded for a candidate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateProposal {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RelationshipCandidateOutput {
    pub from: RelationshipCandidateEndpoint,
    pub to: RelationshipCandidateEndpoint,
    pub relationship: String,
    pub similarity: f32,
    pub signals: Vec<RelationshipCandidateSignal>,
    /// Rarest first, at most eight.
    pub shared_entities: Vec<RelationshipCandidateEntity>,
    pub cross_reference: Option<RelationshipCandidateCrossReference>,
    /// `null` in a dry run, or when the deadline stopped persistence first.
    pub proposal: Option<RelationshipCandidateProposal>,
}

/// Why a dry run returned no continuation cursor.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RelationshipCandidateDryRunCut {
    /// Candidates that passed every gate but were cut by `max_candidates`
    /// (equal to `rejections.over_budget`).
    pub candidates_cut: usize,
    /// How to reach them instead of paging.
    pub note: String,
}

/// Receipt for one scan: what was requested, what actually ran and why it
/// stopped. Zero candidates is a valid abstention.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RelationshipCandidateScanResponse {
    pub contract: u32,
    pub actor: String,
    pub dry_run: bool,
    pub limits: RelationshipCandidateScanLimits,
    pub policy: RelationshipCandidatePolicy,
    pub scope: RelationshipCandidateScope,
    pub counterpart_source_ids: Vec<String>,
    pub after_note_id: Option<String>,
    pub stop_reason: RelationshipCandidateStopReason,
    pub endpoints_examined: usize,
    /// Every anchor in scope after the cursor was examined and every
    /// candidate found was reported.
    pub scope_exhausted: bool,
    /// Pass as `after_note_id` to continue unless `scope_exhausted`. It never
    /// advances past candidates cut by `max_candidates` or the deadline: a
    /// persisting scan then returns its incoming cursor, so repeating it
    /// revisits them (recorded pairs are then skipped), and `null` with
    /// `scope_exhausted = false` means repeat from the start. A dry run
    /// records nothing, so when it cut candidates this is `null` and
    /// `dry_run_cut` is set: there is no continuation.
    pub next_after_note_id: Option<String>,
    /// Set only when a dry run cut candidates at `max_candidates`. Paging
    /// stops here: a repeated dry run returns the same candidates. Narrow the
    /// scope, raise `max_candidates` (at most 50), or run with
    /// `dry_run = false` so the recorded pending proposals let later scans
    /// progress.
    #[serde(default)]
    pub dry_run_cut: Option<RelationshipCandidateDryRunCut>,
    /// Distinct cross-source pairs at or above the similarity floor.
    pub nominations: usize,
    pub rejections: RelationshipCandidateRejections,
    /// Shared names beyond the per-scan frequency budget; never a signal.
    pub unevaluated_names: usize,
    pub elapsed_ms: u64,
    pub embedding_model: Option<String>,
    pub proposals_created: usize,
    pub candidates: Vec<RelationshipCandidateOutput>,
}
