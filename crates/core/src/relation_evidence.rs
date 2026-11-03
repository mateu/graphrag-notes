//! Versioned evidence assessments for typed note-relation proposals (#205).
//!
//! The proposer (a reviewer or a model worker) states its claims: relation,
//! direction, evidence mode, supporting quotes, an optional explicit link,
//! declared lineage, an alternative explanation and the claim's scope. The
//! functions here check those claims against the pinned endpoint snapshots and
//! add facts the proposer cannot assert: span offsets and origin groups.
//!
//! An assessment is not a probability. `calibration` is always `none`, and no
//! field feeds ranking, traversal or context packing.

use crate::EdgeType;
use serde::{Deserialize, Serialize};

/// Version of the stored assessment object. A new version needs a migration
/// because the storage `ASSERT`s pin this value.
pub const EVIDENCE_ASSESSMENT_VERSION: u32 = 1;
/// Supporting spans per endpoint in addition to the request's primary quote.
pub const MAX_ADDITIONAL_SUPPORT_SPANS: usize = 3;
/// Bounds shared by quotes, spans and free-text fields.
pub const MAX_EVIDENCE_TEXT_CHARS: usize = 1024;
pub const MAX_EVIDENCE_TEXT_BYTES: usize = 2048;
/// Bound for generator, version and model labels.
pub const MAX_EVIDENCE_LABEL_CHARS: usize = 128;
/// Shorter references or identifiers are too generic to name an endpoint.
pub const MIN_LINK_REFERENCE_CHARS: usize = 3;

/// How the quoted text supports the asserted relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum EvidenceMode {
    /// The quoted text states the relation directly.
    ExplicitStatement,
    /// The relation follows from the quoted text by a stated derivation.
    Derivation,
    /// A suggestive comparison. Kept pending; never accepted.
    Analogy,
    /// An untested conjecture. Kept pending; never accepted.
    Hypothesis,
}

impl EvidenceMode {
    /// Only explicit statements and derivations may enter the accepted graph.
    pub const fn admits_accepted_graph(self) -> bool {
        matches!(self, Self::ExplicitStatement | Self::Derivation)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitStatement => "explicit_statement",
            Self::Derivation => "derivation",
            Self::Analogy => "analogy",
            Self::Hypothesis => "hypothesis",
        }
    }
}

/// How to read the relation between the request's `from` and `to` notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RelationDirection {
    /// "from <relation> to": `from supports to` means the from passage is
    /// evidence for the to claim; `from derived_from to` means from was
    /// derived from to.
    FromTo,
    /// Order carries no meaning (`related_to`).
    Symmetric,
}

impl RelationDirection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FromTo => "from_to",
            Self::Symmetric => "symmetric",
        }
    }
}

/// The proposer's lineage claim. Deterministic checks take precedence: a
/// declared `independent` that contradicts them is rejected, while
/// `same_origin` records a dependency the store cannot detect (a summary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DeclaredLineage {
    #[default]
    Unknown,
    Independent,
    SameOrigin,
}

impl DeclaredLineage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Independent => "independent",
            Self::SameOrigin => "same_origin",
        }
    }
}

/// Endpoint role in the request, before any symmetric canonical ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum EndpointRole {
    From,
    To,
}

impl EndpointRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::From => "from",
            Self::To => "to",
        }
    }
}

/// Probability calibration status. Only `none` exists: no assessment is a
/// calibrated probability, and none may be presented as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Calibration {
    #[default]
    #[serde(rename = "none")]
    Uncalibrated,
}

/// Score from the generator that nominated the pair. It is never written to
/// `confidence` and is never treated as evidence strength.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CandidateScore {
    /// Candidate generator, e.g. `gardener-similarity`.
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 128)))]
    pub generator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 128)))]
    pub generator_version: Option<String>,
    /// Generator score in [0, 1], such as cosine similarity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(range(min = 0.0, max = 1.0)))]
    pub similarity: Option<f64>,
}

/// Tool or model that typed the proposal, as declared by the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ProposerDeclaration {
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 128)))]
    pub generator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 128)))]
    pub generator_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 128)))]
    pub model: Option<String>,
}

/// An explicit textual link from one endpoint to the other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LinkEvidenceRequest {
    /// Endpoint whose current content contains the link.
    pub endpoint: EndpointRole,
    /// Verbatim passage in that endpoint.
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 1024)))]
    pub quote: String,
    /// Verbatim part of `quote` naming the other endpoint: its canonical
    /// note ID, title, source title, source URI or source file name.
    #[cfg_attr(feature = "json-schema", schemars(length(min = 3, max = 1024)))]
    pub reference: String,
}

/// Typed evidence supplied with an endpoint proposal. It is optional: a
/// request without it keeps the legacy payload and replay behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct EvidenceAssessmentRequest {
    pub evidence_mode: EvidenceMode,
    /// Must match the relation: `from_to` for supports, contradicts and
    /// derived_from; `symmetric` for related_to.
    pub direction: RelationDirection,
    #[serde(default)]
    pub lineage: DeclaredLineage,
    /// Additional verbatim spans from the `from` note. The primary quote is
    /// always recorded as the first span.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "json-schema", schemars(length(max = 3)))]
    pub from_support: Vec<String>,
    /// Additional verbatim spans from the `to` note.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "json-schema", schemars(length(max = 3)))]
    pub to_support: Vec<String>,
    /// Required for an accepted-graph `related_to`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<LinkEvidenceRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<CandidateScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposer: Option<ProposerDeclaration>,
    /// Competing reading the proposer considered and why it was set aside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 1024)))]
    pub alternative_explanation: Option<String>,
    /// Where or when the relation holds (version, incident, period).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "json-schema", schemars(length(min = 1, max = 1024)))]
    pub scope: Option<String>,
}

/// Per-relation rules. #183's new relations add rows here together with
/// their own edge tables and migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationPolicy {
    pub relation: &'static str,
    pub direction: RelationDirection,
    /// Modes that may be accepted. Analogy and hypothesis are always
    /// proposable but never accepted.
    pub accepted_modes: &'static [EvidenceMode],
    /// Whether both endpoints may share an origin (same source or copy).
    pub same_origin_allowed: bool,
    /// Whether an accepted-graph claim needs explicit link evidence.
    pub requires_link: bool,
}

const EVIDENCED: &[EvidenceMode] = &[EvidenceMode::ExplicitStatement, EvidenceMode::Derivation];

pub fn relation_policy(relation: &EdgeType) -> Option<RelationPolicy> {
    let (direction, accepted_modes, same_origin_allowed, requires_link) = match relation {
        EdgeType::Supports => (RelationDirection::FromTo, EVIDENCED, false, false),
        EdgeType::Contradicts => (RelationDirection::FromTo, EVIDENCED, true, false),
        EdgeType::DerivedFrom => (RelationDirection::FromTo, EVIDENCED, true, false),
        EdgeType::RelatedTo => (
            RelationDirection::Symmetric,
            &[EvidenceMode::ExplicitStatement][..],
            true,
            true,
        ),
        EdgeType::References | EdgeType::Mentions | EdgeType::TaggedWith => return None,
    };
    Some(RelationPolicy {
        relation: match relation {
            EdgeType::Supports => "supports",
            EdgeType::Contradicts => "contradicts",
            EdgeType::DerivedFrom => "derived_from",
            _ => "related_to",
        },
        direction,
        accepted_modes,
        same_origin_allowed,
        requires_link,
    })
}

fn relation_for_name(relation: &str) -> Option<EdgeType> {
    Some(match relation {
        "supports" => EdgeType::Supports,
        "contradicts" => EdgeType::Contradicts,
        "derived_from" => EdgeType::DerivedFrom,
        "related_to" => EdgeType::RelatedTo,
        _ => return None,
    })
}

/// One verbatim span with offsets computed by the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SupportSpan {
    pub quote: String,
    /// UTF-8 byte offsets of the first occurrence in the pinned content.
    pub byte_start: u64,
    pub byte_end: u64,
    /// Number of non-overlapping occurrences. Above 1 the location is ambiguous.
    pub occurrences: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct EndpointEvidence {
    pub role: EndpointRole,
    /// Canonical note ID, so canonical symmetric ordering cannot mislabel it.
    pub note: String,
    /// Origin group: the note's `source:` record, or the note itself for a
    /// manual note without a source.
    pub origin: String,
    pub support: Vec<SupportSpan>,
}

/// Which identifier of the target endpoint a link reference matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum LinkReferenceKind {
    NoteId,
    NoteTitle,
    SourceTitle,
    SourceUri,
    SourceFileName,
}

impl LinkReferenceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoteId => "note_id",
            Self::NoteTitle => "note_title",
            Self::SourceTitle => "source_title",
            Self::SourceUri => "source_uri",
            Self::SourceFileName => "source_file_name",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LinkEvidence {
    /// Note containing the link.
    pub note: String,
    /// Note the link names.
    pub target: String,
    pub quote: String,
    pub reference: String,
    pub matched: LinkReferenceKind,
    pub byte_start: u64,
    pub byte_end: u64,
}

/// Why the two endpoints were or were not grouped under one origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum LineageBasis {
    DistinctOrigins,
    SameSource,
    IdenticalSourceContent,
    DeclaredSameOrigin,
}

impl LineageBasis {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DistinctOrigins => "distinct_origins",
            Self::SameSource => "same_source",
            Self::IdenticalSourceContent => "identical_source_content",
            Self::DeclaredSameOrigin => "declared_same_origin",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LineageAssessment {
    pub declared: DeclaredLineage,
    /// True when the endpoints are not independent corroboration.
    pub same_origin: bool,
    pub basis: LineageBasis,
}

/// Who proposed: authenticated actor plus the caller's declared tooling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ProposerRecord {
    /// Server-derived identity, e.g. `mcp:<instance>`.
    pub actor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Stored, immutable v1 assessment. The reviewer decision stays in the
/// proposal lifecycle (`status`, `reviewer`, `action_reason`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct EvidenceAssessment {
    pub version: u32,
    pub calibration: Calibration,
    pub relation: String,
    pub direction: RelationDirection,
    pub evidence_mode: EvidenceMode,
    pub proposer: ProposerRecord,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<CandidateScore>,
    /// Exactly two entries in request order (`from`, then `to`).
    pub endpoints: Vec<EndpointEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<LinkEvidence>,
    pub lineage: LineageAssessment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternative_explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// A rule the proposer's claims broke. Messages tell the caller what to fix.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AssessmentViolation {
    #[error("relation {0} has no typed evidence policy")]
    UnsupportedRelation(String),
    #[error("direction {found} does not match {relation}; use {expected}")]
    Direction {
        relation: &'static str,
        expected: &'static str,
        found: &'static str,
    },
    #[error("{relation} cannot be accepted on {mode} evidence; use a permitted mode, or analogy/hypothesis to keep a speculative claim pending")]
    ModeNotAdmitted {
        relation: &'static str,
        mode: &'static str,
    },
    #[error("related_to requires explicit link evidence: a verbatim quote in one endpoint whose reference names the other endpoint; similarity or a shared topic is not a link")]
    LinkRequired,
    #[error("{0} must be trimmed, nonempty, without NUL, and at most 1024 characters / 2048 UTF-8 bytes")]
    TextBounds(&'static str),
    #[error(
        "{0} must be trimmed, nonempty, without control characters, and at most 128 characters"
    )]
    LabelBounds(&'static str),
    #[error("at most 3 additional support spans are allowed for the {0} endpoint")]
    TooManySpans(&'static str),
    #[error("duplicate support span for the {0} endpoint")]
    DuplicateSpan(&'static str),
    #[error("a {0} support span is not verbatim in that note at its pinned revision")]
    SpanNotVerbatim(&'static str),
    #[error("candidate similarity must be a finite number between 0 and 1")]
    CandidateSimilarity,
    #[error("the link quote is not verbatim in the {0} note at its pinned revision")]
    LinkQuoteNotVerbatim(&'static str),
    #[error(
        "the link reference must be a verbatim part of the link quote and at least 3 characters"
    )]
    LinkReferenceNotInQuote,
    #[error("the link reference does not name the other endpoint (its note ID, title, source title, source URI or file name)")]
    LinkReferenceUnmatched,
    #[error("the link must be quoted from one assessed endpoint and name the other")]
    LinkEndpoints,
    #[error("declared independent lineage contradicts the deterministic finding ({0})")]
    DeclaredIndependentContradicted(&'static str),
    #[error("same-source or same-lineage endpoints cannot be supports: copies and summaries of one origin are not independent corroboration; propose derived_from instead")]
    SameOriginSupports,
    #[error("unsupported evidence assessment version {0}")]
    Version(u32),
    #[error("malformed evidence assessment: {0}")]
    Malformed(String),
    #[error(
        "the stored evidence assessment does not match its endpoints at their pinned revisions"
    )]
    Unverified,
}

fn check_text(value: &str, field: &'static str) -> Result<(), AssessmentViolation> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.contains('\0')
        || value.len() > MAX_EVIDENCE_TEXT_BYTES
        || value.chars().count() > MAX_EVIDENCE_TEXT_CHARS
    {
        return Err(AssessmentViolation::TextBounds(field));
    }
    Ok(())
}

fn check_label(value: &str, field: &'static str) -> Result<(), AssessmentViolation> {
    if value.is_empty()
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.chars().count() > MAX_EVIDENCE_LABEL_CHARS
    {
        return Err(AssessmentViolation::LabelBounds(field));
    }
    Ok(())
}

fn check_policy(
    policy: RelationPolicy,
    direction: RelationDirection,
    mode: EvidenceMode,
) -> Result<(), AssessmentViolation> {
    if direction != policy.direction {
        return Err(AssessmentViolation::Direction {
            relation: policy.relation,
            expected: policy.direction.as_str(),
            found: direction.as_str(),
        });
    }
    if mode.admits_accepted_graph() && !policy.accepted_modes.contains(&mode) {
        return Err(AssessmentViolation::ModeNotAdmitted {
            relation: policy.relation,
            mode: mode.as_str(),
        });
    }
    Ok(())
}

/// Checks that need no storage. Run before persistence and again inside
/// [`assess`].
pub fn validate_request(
    relation: &EdgeType,
    request: &EvidenceAssessmentRequest,
) -> Result<RelationPolicy, AssessmentViolation> {
    let policy = relation_policy(relation)
        .ok_or_else(|| AssessmentViolation::UnsupportedRelation(relation.to_string()))?;
    check_policy(policy, request.direction, request.evidence_mode)?;
    if policy.requires_link
        && request.evidence_mode.admits_accepted_graph()
        && request.link.is_none()
    {
        return Err(AssessmentViolation::LinkRequired);
    }
    for (role, spans) in [
        (EndpointRole::From, &request.from_support),
        (EndpointRole::To, &request.to_support),
    ] {
        if spans.len() > MAX_ADDITIONAL_SUPPORT_SPANS {
            return Err(AssessmentViolation::TooManySpans(role.as_str()));
        }
        for span in spans {
            check_text(span, "support span")?;
        }
    }
    if let Some(link) = &request.link {
        check_text(&link.quote, "link quote")?;
        check_text(&link.reference, "link reference")?;
        if link.reference.chars().count() < MIN_LINK_REFERENCE_CHARS
            || !link.quote.contains(&link.reference)
        {
            return Err(AssessmentViolation::LinkReferenceNotInQuote);
        }
    }
    if let Some(candidate) = &request.candidate {
        check_label(&candidate.generator, "candidate generator")?;
        if let Some(version) = &candidate.generator_version {
            check_label(version, "candidate generator_version")?;
        }
        if candidate
            .similarity
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(AssessmentViolation::CandidateSimilarity);
        }
    }
    if let Some(proposer) = &request.proposer {
        check_label(&proposer.generator, "proposer generator")?;
        for (value, field) in [
            (&proposer.generator_version, "proposer generator_version"),
            (&proposer.model, "proposer model"),
        ] {
            if let Some(value) = value {
                check_label(value, field)?;
            }
        }
    }
    if let Some(value) = &request.alternative_explanation {
        check_text(value, "alternative_explanation")?;
    }
    if let Some(value) = &request.scope {
        check_text(value, "scope")?;
    }
    Ok(policy)
}

/// Source facts pinned with an endpoint snapshot.
#[derive(Debug, Clone, Copy)]
pub struct SourceSnapshot<'a> {
    /// Canonical `source:` record ID.
    pub id: &'a str,
    pub title: Option<&'a str>,
    pub uri: Option<&'a str>,
    pub content_hash: Option<&'a str>,
}

/// One endpoint at its pinned revision, as fenced by the proposal transaction.
#[derive(Debug, Clone, Copy)]
pub struct EndpointSnapshot<'a> {
    /// Canonical `note:` record ID as supplied in the request.
    pub note_id: &'a str,
    pub content: &'a str,
    /// The request's primary quote for this endpoint.
    pub primary_quote: &'a str,
    pub title: Option<&'a str>,
    pub source: Option<SourceSnapshot<'a>>,
}

impl EndpointSnapshot<'_> {
    fn origin(&self) -> String {
        self.source
            .map(|source| source.id.to_string())
            .unwrap_or_else(|| self.note_id.to_string())
    }

    fn content_hash(&self) -> Option<&str> {
        self.source
            .and_then(|source| source.content_hash)
            .filter(|hash| !hash.trim().is_empty())
    }

    /// Identifiers a link reference may name, in reporting priority order.
    fn identifiers(&self) -> Vec<(LinkReferenceKind, &str)> {
        let mut identifiers = vec![(LinkReferenceKind::NoteId, self.note_id)];
        if let Some(title) = self.title {
            identifiers.push((LinkReferenceKind::NoteTitle, title));
        }
        if let Some(source) = self.source {
            if let Some(title) = source.title {
                identifiers.push((LinkReferenceKind::SourceTitle, title));
            }
            if let Some(uri) = source.uri {
                identifiers.push((LinkReferenceKind::SourceUri, uri));
                if let Some(name) = uri.rsplit(['/', '\\']).next() {
                    identifiers.push((LinkReferenceKind::SourceFileName, name));
                    if let Some((stem, _)) = name.rsplit_once('.') {
                        identifiers.push((LinkReferenceKind::SourceFileName, stem));
                    }
                }
            }
        }
        identifiers
            .into_iter()
            .filter(|(_, value)| value.trim().chars().count() >= MIN_LINK_REFERENCE_CHARS)
            .collect()
    }
}

fn locate(content: &str, quote: &str) -> Option<SupportSpan> {
    let start = content.find(quote)?;
    Some(SupportSpan {
        quote: quote.to_string(),
        byte_start: start as u64,
        byte_end: (start + quote.len()) as u64,
        occurrences: content.matches(quote).count() as u64,
    })
}

/// Whether stored offsets span exactly `quote`'s bytes.
fn located(quote: &str, start: u64, end: u64) -> bool {
    end.checked_sub(start) == u64::try_from(quote.len()).ok()
}

/// Whether a stored lineage is one [`assess`] can produce for endpoints whose
/// recorded origins are (or are not) the same.
fn lineage_consistent(lineage: LineageAssessment, shared_origin: bool) -> bool {
    let basis_follows = match lineage.basis {
        LineageBasis::SameSource => shared_origin,
        LineageBasis::IdenticalSourceContent => !shared_origin,
        LineageBasis::DeclaredSameOrigin => {
            !shared_origin && lineage.declared == DeclaredLineage::SameOrigin
        }
        LineageBasis::DistinctOrigins => {
            !shared_origin && lineage.declared != DeclaredLineage::SameOrigin
        }
    };
    basis_follows
        && lineage.same_origin == (lineage.basis != LineageBasis::DistinctOrigins)
        && (lineage.declared != DeclaredLineage::Independent || !lineage.same_origin)
}

const TWO_ENDPOINTS: &str = "assessment must describe two distinct endpoints with support";

fn normalized(value: &str) -> String {
    value.trim().to_lowercase()
}

/// Check the claims against both pinned endpoints and build the stored
/// assessment. `actor` is the authenticated proposer identity.
pub fn assess(
    relation: &EdgeType,
    request: &EvidenceAssessmentRequest,
    actor: &str,
    from: &EndpointSnapshot<'_>,
    to: &EndpointSnapshot<'_>,
) -> Result<EvidenceAssessment, AssessmentViolation> {
    let policy = validate_request(relation, request)?;
    let mut endpoints = Vec::with_capacity(2);
    for (role, snapshot, additional) in [
        (EndpointRole::From, from, &request.from_support),
        (EndpointRole::To, to, &request.to_support),
    ] {
        let mut support = Vec::with_capacity(1 + additional.len());
        for quote in
            std::iter::once(snapshot.primary_quote).chain(additional.iter().map(String::as_str))
        {
            if support.iter().any(|span: &SupportSpan| span.quote == quote) {
                return Err(AssessmentViolation::DuplicateSpan(role.as_str()));
            }
            support.push(
                locate(snapshot.content, quote)
                    .ok_or(AssessmentViolation::SpanNotVerbatim(role.as_str()))?,
            );
        }
        endpoints.push(EndpointEvidence {
            role,
            note: snapshot.note_id.to_string(),
            origin: snapshot.origin(),
            support,
        });
    }

    let link = match &request.link {
        None => None,
        Some(link) => {
            let (holder, target) = match link.endpoint {
                EndpointRole::From => (from, to),
                EndpointRole::To => (to, from),
            };
            let span = locate(holder.content, &link.quote).ok_or(
                AssessmentViolation::LinkQuoteNotVerbatim(link.endpoint.as_str()),
            )?;
            let reference = normalized(&link.reference);
            let matched = target
                .identifiers()
                .into_iter()
                .find(|(_, identifier)| normalized(identifier) == reference)
                .map(|(kind, _)| kind)
                .ok_or(AssessmentViolation::LinkReferenceUnmatched)?;
            Some(LinkEvidence {
                note: holder.note_id.to_string(),
                target: target.note_id.to_string(),
                quote: link.quote.clone(),
                reference: link.reference.clone(),
                matched,
                byte_start: span.byte_start,
                byte_end: span.byte_end,
            })
        }
    };

    let computed = if endpoints[0].origin == endpoints[1].origin {
        Some((LineageBasis::SameSource, "same source"))
    } else if from.content_hash().is_some() && from.content_hash() == to.content_hash() {
        Some((
            LineageBasis::IdenticalSourceContent,
            "identical source content",
        ))
    } else {
        None
    };
    let lineage = match (request.lineage, computed) {
        (DeclaredLineage::Independent, Some((_, finding))) => {
            return Err(AssessmentViolation::DeclaredIndependentContradicted(
                finding,
            ))
        }
        (declared, Some((basis, _))) => LineageAssessment {
            declared,
            same_origin: true,
            basis,
        },
        (DeclaredLineage::SameOrigin, None) => LineageAssessment {
            declared: DeclaredLineage::SameOrigin,
            same_origin: true,
            basis: LineageBasis::DeclaredSameOrigin,
        },
        (declared, None) => LineageAssessment {
            declared,
            same_origin: false,
            basis: LineageBasis::DistinctOrigins,
        },
    };
    if lineage.same_origin && !policy.same_origin_allowed {
        return Err(AssessmentViolation::SameOriginSupports);
    }

    Ok(EvidenceAssessment {
        version: EVIDENCE_ASSESSMENT_VERSION,
        calibration: Calibration::Uncalibrated,
        relation: policy.relation.to_string(),
        direction: request.direction,
        evidence_mode: request.evidence_mode,
        proposer: ProposerRecord {
            actor: actor.to_string(),
            generator: request.proposer.as_ref().map(|p| p.generator.clone()),
            generator_version: request
                .proposer
                .as_ref()
                .and_then(|p| p.generator_version.clone()),
            model: request.proposer.as_ref().and_then(|p| p.model.clone()),
        },
        candidate: request.candidate.clone(),
        endpoints,
        link,
        lineage,
        alternative_explanation: request.alternative_explanation.clone(),
        scope: request.scope.clone(),
    })
}

impl EvidenceAssessment {
    /// Parse a stored object and re-check every invariant that needs no
    /// endpoint content: the relation's direction, mode and link gates, the
    /// request bounds, located and distinct spans, a link quoted from one
    /// endpoint that names the other, and a lineage that follows from the
    /// recorded origins. Anything unparseable, of another version, or
    /// inconsistent fails closed. [`Self::verify`] adds the content checks.
    pub fn from_stored(value: &serde_json::Value) -> Result<Self, AssessmentViolation> {
        if let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) {
            if version != u64::from(EVIDENCE_ASSESSMENT_VERSION) {
                return Err(AssessmentViolation::Version(
                    u32::try_from(version).unwrap_or(u32::MAX),
                ));
            }
        }
        let assessment: Self = serde_json::from_value(value.clone())
            .map_err(|error| AssessmentViolation::Malformed(error.to_string()))?;
        if assessment.version != EVIDENCE_ASSESSMENT_VERSION {
            return Err(AssessmentViolation::Version(assessment.version));
        }
        let relation = relation_for_name(&assessment.relation)
            .ok_or_else(|| AssessmentViolation::UnsupportedRelation(assessment.relation.clone()))?;
        let [from, to] = assessment.endpoints.as_slice() else {
            return Err(AssessmentViolation::Malformed(TWO_ENDPOINTS.into()));
        };
        if from.role != EndpointRole::From || to.role != EndpointRole::To || from.note == to.note {
            return Err(AssessmentViolation::Malformed(TWO_ENDPOINTS.into()));
        }
        for endpoint in [from, to] {
            if endpoint.support.is_empty() {
                return Err(AssessmentViolation::Malformed(TWO_ENDPOINTS.into()));
            }
            for (index, span) in endpoint.support.iter().enumerate() {
                check_text(&span.quote, "support span")?;
                if endpoint.support[..index]
                    .iter()
                    .any(|earlier| earlier.quote == span.quote)
                {
                    return Err(AssessmentViolation::DuplicateSpan(endpoint.role.as_str()));
                }
                if span.occurrences == 0 || !located(&span.quote, span.byte_start, span.byte_end) {
                    return Err(AssessmentViolation::Malformed(
                        "support span offsets do not locate its quote".into(),
                    ));
                }
            }
        }
        let policy = validate_request(&relation, &assessment.claims()?)?;
        if assessment
            .link
            .as_ref()
            .is_some_and(|link| !located(&link.quote, link.byte_start, link.byte_end))
        {
            return Err(AssessmentViolation::Malformed(
                "link offsets do not locate its quote".into(),
            ));
        }
        if assessment.lineage.same_origin && !policy.same_origin_allowed {
            return Err(AssessmentViolation::SameOriginSupports);
        }
        if !lineage_consistent(assessment.lineage, from.origin == to.origin) {
            return Err(AssessmentViolation::Malformed(
                "lineage does not follow from the endpoint origins".into(),
            ));
        }
        Ok(assessment)
    }

    /// The proposer's claims as a request, so stored evidence can be checked
    /// again. A link must be quoted from one endpoint and name the other.
    fn claims(&self) -> Result<EvidenceAssessmentRequest, AssessmentViolation> {
        let [from, to] = self.endpoints.as_slice() else {
            return Err(AssessmentViolation::Malformed(TWO_ENDPOINTS.into()));
        };
        let link = match &self.link {
            None => None,
            Some(link) => {
                let pair = (link.note.as_str(), link.target.as_str());
                let endpoint = if pair == (from.note.as_str(), to.note.as_str()) {
                    EndpointRole::From
                } else if pair == (to.note.as_str(), from.note.as_str()) {
                    EndpointRole::To
                } else {
                    return Err(AssessmentViolation::LinkEndpoints);
                };
                Some(LinkEvidenceRequest {
                    endpoint,
                    quote: link.quote.clone(),
                    reference: link.reference.clone(),
                })
            }
        };
        let proposer = match &self.proposer {
            ProposerRecord {
                generator: Some(generator),
                generator_version,
                model,
                ..
            } => Some(ProposerDeclaration {
                generator: generator.clone(),
                generator_version: generator_version.clone(),
                model: model.clone(),
            }),
            ProposerRecord {
                generator_version: None,
                model: None,
                ..
            } => None,
            _ => {
                return Err(AssessmentViolation::Malformed(
                    "proposer tooling is recorded without a generator".into(),
                ))
            }
        };
        let additional = |endpoint: &EndpointEvidence| {
            endpoint
                .support
                .iter()
                .skip(1)
                .map(|span| span.quote.clone())
                .collect()
        };
        Ok(EvidenceAssessmentRequest {
            evidence_mode: self.evidence_mode,
            direction: self.direction,
            lineage: self.lineage.declared,
            from_support: additional(from),
            to_support: additional(to),
            link,
            candidate: self.candidate.clone(),
            proposer,
            alternative_explanation: self.alternative_explanation.clone(),
            scope: self.scope.clone(),
        })
    }

    /// Analogy and hypothesis assessments stay pending.
    pub fn admits_accepted_graph(&self) -> bool {
        self.evidence_mode.admits_accepted_graph()
    }

    /// Re-run [`assess`] on the stored claims against both endpoints at their
    /// pinned revisions, in request order, and require exactly the stored
    /// object: relation, direction and mode gates, every span offset and
    /// occurrence count, origin groups, the link holder, target, verbatim
    /// quote and matched identifier, and the lineage finding. Acceptance runs
    /// this, so restored or tampered evidence fails closed.
    pub fn verify(
        &self,
        relation: &EdgeType,
        from: &EndpointSnapshot<'_>,
        to: &EndpointSnapshot<'_>,
    ) -> Result<(), AssessmentViolation> {
        let derived = assess(relation, &self.claims()?, &self.proposer.actor, from, to)?;
        if derived == *self {
            Ok(())
        } else {
            Err(AssessmentViolation::Unverified)
        }
    }
}

/// Reporting status of a relation's evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Assessed,
    /// Legacy or untyped relation: no evidence assessment exists.
    #[default]
    Unassessed,
    /// A stored assessment failed to parse or validate; acceptance fails closed.
    Invalid,
}

/// What a legacy stored `confidence` value actually is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum LegacyWeightMeaning {
    /// 1.0 recorded for an explicitly reviewed endpoint assertion.
    ReviewConvention,
    /// Gardener's cosine similarity between endpoint embeddings.
    CandidateSimilarity,
    /// A stored weight of unrecorded meaning.
    #[default]
    Unspecified,
    /// No stored weight; traversal uses 1.0.
    Absent,
}

/// Who recorded the legacy weight: a proposal's generator or an edge's
/// provenance. Only Gardener and the reviewed-endpoint family have a known
/// meaning.
pub const REVIEWED_ENDPOINT_GENERATOR: &str = "remote-reviewed-endpoint";
pub const GARDENER_SIMILARITY_GENERATOR: &str = "gardener-similarity";

/// Widen a stored `f32` through its shortest decimal form, so every encoder
/// reports `0.9` rather than `0.8999999761581421`.
fn widen(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(f64::from(value))
}

/// Compact, report-only summary for cards, edges and explanation paths.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct EvidenceSummary {
    pub status: EvidenceStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_mode: Option<EvidenceMode>,
    #[serde(default)]
    pub calibration: Calibration,
    /// Generator score, never confidence: the assessment's candidate score,
    /// or a Gardener proposal's legacy cosine similarity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_similarity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub same_origin: Option<bool>,
    /// Stored legacy `confidence`, kept readable, never a probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_weight: Option<f64>,
    #[serde(default)]
    pub legacy_weight_meaning: LegacyWeightMeaning,
}

impl EvidenceSummary {
    /// `recorded_by` is a proposal generator or an edge provenance.
    pub fn describe(
        assessment: Option<&serde_json::Value>,
        recorded_by: Option<&str>,
        weight: Option<f32>,
    ) -> Self {
        let legacy_weight_meaning = match (weight, recorded_by) {
            (None, _) => LegacyWeightMeaning::Absent,
            (Some(_), Some(REVIEWED_ENDPOINT_GENERATOR)) => LegacyWeightMeaning::ReviewConvention,
            (Some(_), Some(GARDENER_SIMILARITY_GENERATOR)) => {
                LegacyWeightMeaning::CandidateSimilarity
            }
            (Some(_), _) => LegacyWeightMeaning::Unspecified,
        };
        let weight = weight.map(widen);
        let legacy_similarity = (legacy_weight_meaning == LegacyWeightMeaning::CandidateSimilarity)
            .then_some(weight)
            .flatten();
        let mut summary = Self {
            legacy_weight: weight,
            legacy_weight_meaning,
            candidate_similarity: legacy_similarity,
            ..Self::default()
        };
        if let Some(value) = assessment {
            match EvidenceAssessment::from_stored(value) {
                Ok(assessment) => {
                    summary.status = EvidenceStatus::Assessed;
                    summary.evidence_mode = Some(assessment.evidence_mode);
                    summary.same_origin = Some(assessment.lineage.same_origin);
                    summary.candidate_similarity = assessment
                        .candidate
                        .and_then(|candidate| candidate.similarity)
                        .or(summary.candidate_similarity);
                }
                Err(_) => summary.status = EvidenceStatus::Invalid,
            }
        }
        summary
    }

    /// One-line human label. Never renders a weight as a percentage.
    pub fn label(&self) -> String {
        let candidate = self
            .candidate_similarity
            .map(|value| format!("; candidate similarity {value:.2} (generator score)"))
            .unwrap_or_default();
        match self.status {
            EvidenceStatus::Assessed => format!(
                "assessed: {}; {}{candidate}; uncalibrated",
                self.evidence_mode.map(EvidenceMode::as_str).unwrap_or("unknown"),
                if self.same_origin == Some(true) {
                    "same origin"
                } else {
                    "independent origins"
                },
            ),
            EvidenceStatus::Invalid => {
                "invalid assessment; acceptance fails closed".to_string()
            }
            EvidenceStatus::Unassessed => match (self.legacy_weight_meaning, self.legacy_weight) {
                (LegacyWeightMeaning::ReviewConvention, Some(weight)) => format!(
                    "unassessed; stored weight {weight:.2} is a review convention, not a probability"
                ),
                (LegacyWeightMeaning::CandidateSimilarity, Some(weight)) => format!(
                    "unassessed; candidate similarity {weight:.2} (generator score, not a probability)"
                ),
                (_, Some(weight)) => {
                    format!("unassessed; legacy weight {weight:.2}, not a probability")
                }
                (_, None) => "unassessed; no stored weight".to_string(),
            },
        }
    }

    /// Short token for compact path renderings.
    pub fn token(&self) -> &'static str {
        match self.status {
            EvidenceStatus::Assessed => self
                .evidence_mode
                .map(EvidenceMode::as_str)
                .unwrap_or("assessed"),
            EvidenceStatus::Unassessed => "unassessed",
            EvidenceStatus::Invalid => "invalid",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: EvidenceMode, direction: RelationDirection) -> EvidenceAssessmentRequest {
        EvidenceAssessmentRequest {
            evidence_mode: mode,
            direction,
            lineage: DeclaredLineage::Unknown,
            from_support: vec![],
            to_support: vec![],
            link: None,
            candidate: None,
            proposer: None,
            alternative_explanation: None,
            scope: None,
        }
    }

    fn snapshot<'a>(
        note_id: &'a str,
        content: &'a str,
        quote: &'a str,
        source: Option<SourceSnapshot<'a>>,
    ) -> EndpointSnapshot<'a> {
        EndpointSnapshot {
            note_id,
            content,
            primary_quote: quote,
            title: None,
            source,
        }
    }

    fn source<'a>(id: &'a str, uri: &'a str, hash: &'a str) -> SourceSnapshot<'a> {
        SourceSnapshot {
            id,
            title: None,
            uri: Some(uri),
            content_hash: Some(hash),
        }
    }

    const A: &str = "Rollout gate: the canary must pass before release. See plan-b for timing.";
    const B: &str = "Canary results passed on the pinned build.";

    #[test]
    fn explicit_support_records_offsets_origins_and_no_calibration() {
        let mut typed = request(EvidenceMode::ExplicitStatement, RelationDirection::FromTo);
        typed.to_support = vec!["See plan-b for timing.".into()];
        typed.candidate = Some(CandidateScore {
            generator: "gardener-similarity".into(),
            generator_version: None,
            similarity: Some(0.81),
        });
        typed.scope = Some("Pinned build only".into());
        let from = snapshot(
            "note:b",
            B,
            "Canary results passed",
            Some(source("source:results", "file:///n/results.md", "h1")),
        );
        let to = snapshot(
            "note:a",
            A,
            "the canary must pass",
            Some(source("source:plan", "file:///n/plan.md", "h2")),
        );
        let assessment = assess(&EdgeType::Supports, &typed, "mcp:reviewer", &from, &to).unwrap();
        assert_eq!(assessment.version, 1);
        assert_eq!(assessment.calibration, Calibration::Uncalibrated);
        assert_eq!(assessment.relation, "supports");
        assert_eq!(assessment.endpoints[0].note, "note:b");
        assert_eq!(assessment.endpoints[0].origin, "source:results");
        assert_eq!(assessment.endpoints[0].support[0].byte_start, 0);
        assert_eq!(assessment.endpoints[0].support.len(), 1);
        assert_eq!(assessment.endpoints[1].support.len(), 2);
        assert_eq!(
            assessment.endpoints[1].support[0].quote,
            "the canary must pass"
        );
        assert_eq!(assessment.endpoints[1].support[1].occurrences, 1);
        assert_eq!(
            &A[assessment.endpoints[1].support[0].byte_start as usize
                ..assessment.endpoints[1].support[0].byte_end as usize],
            "the canary must pass"
        );
        assert!(!assessment.lineage.same_origin);
        assert_eq!(assessment.lineage.basis, LineageBasis::DistinctOrigins);
        assert_eq!(
            assessment.candidate.as_ref().unwrap().similarity,
            Some(0.81)
        );
        assert_eq!(assessment.verify(&EdgeType::Supports, &from, &to), Ok(()));
        assert_eq!(
            assessment.verify(
                &EdgeType::Supports,
                &from,
                &EndpointSnapshot {
                    content: "changed",
                    ..to
                }
            ),
            Err(AssessmentViolation::SpanNotVerbatim("to"))
        );
        let stored = serde_json::to_value(&assessment).unwrap();
        assert_eq!(stored["calibration"], "none");
        assert_eq!(
            EvidenceAssessment::from_stored(&stored).unwrap(),
            assessment
        );
        let summary =
            EvidenceSummary::describe(Some(&stored), Some(REVIEWED_ENDPOINT_GENERATOR), Some(1.0));
        assert_eq!(summary.status, EvidenceStatus::Assessed);
        assert_eq!(summary.candidate_similarity, Some(0.81));
        assert!(!summary.label().contains('%'));
        assert!(summary.label().contains("uncalibrated"));
    }

    #[test]
    fn spans_must_be_verbatim_distinct_and_bounded() {
        let from = snapshot("note:b", B, "Canary results passed", None);
        let to = snapshot("note:a", A, "the canary must pass", None);
        let mut typed = request(EvidenceMode::Derivation, RelationDirection::FromTo);
        typed.to_support = vec!["not in the note".into()];
        assert_eq!(
            assess(&EdgeType::DerivedFrom, &typed, "mcp:r", &from, &to),
            Err(AssessmentViolation::SpanNotVerbatim("to"))
        );
        typed.to_support = vec!["the canary must pass".into()];
        assert_eq!(
            assess(&EdgeType::DerivedFrom, &typed, "mcp:r", &from, &to),
            Err(AssessmentViolation::DuplicateSpan("to"))
        );
        typed.to_support = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        assert_eq!(
            validate_request(&EdgeType::DerivedFrom, &typed),
            Err(AssessmentViolation::TooManySpans("to"))
        );
        typed.to_support = vec![" padded".into()];
        assert_eq!(
            validate_request(&EdgeType::DerivedFrom, &typed),
            Err(AssessmentViolation::TextBounds("support span"))
        );
        typed.to_support = vec![];
        typed.scope = Some("x".repeat(MAX_EVIDENCE_TEXT_CHARS + 1));
        assert_eq!(
            validate_request(&EdgeType::DerivedFrom, &typed),
            Err(AssessmentViolation::TextBounds("scope"))
        );
    }

    #[test]
    fn direction_and_mode_follow_the_relation_policy() {
        for (relation, direction) in [
            (EdgeType::Supports, RelationDirection::Symmetric),
            (EdgeType::RelatedTo, RelationDirection::FromTo),
        ] {
            assert!(matches!(
                validate_request(
                    &relation,
                    &request(EvidenceMode::ExplicitStatement, direction)
                ),
                Err(AssessmentViolation::Direction { .. })
            ));
        }
        // related_to admits only explicit statements into the accepted graph.
        let mut derivation = request(EvidenceMode::Derivation, RelationDirection::Symmetric);
        derivation.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::From,
            quote: "See plan-b".into(),
            reference: "plan-b".into(),
        });
        assert!(matches!(
            validate_request(&EdgeType::RelatedTo, &derivation),
            Err(AssessmentViolation::ModeNotAdmitted { .. })
        ));
        assert!(matches!(
            validate_request(
                &EdgeType::Mentions,
                &request(EvidenceMode::ExplicitStatement, RelationDirection::FromTo)
            ),
            Err(AssessmentViolation::UnsupportedRelation(_))
        ));
        // Pending-only modes are proposable for every relation.
        for mode in [EvidenceMode::Analogy, EvidenceMode::Hypothesis] {
            assert!(!mode.admits_accepted_graph());
            validate_request(
                &EdgeType::RelatedTo,
                &request(mode, RelationDirection::Symmetric),
            )
            .unwrap();
            validate_request(
                &EdgeType::Supports,
                &request(mode, RelationDirection::FromTo),
            )
            .unwrap();
        }
    }

    #[test]
    fn related_to_requires_a_link_naming_the_other_endpoint() {
        let from = EndpointSnapshot {
            title: Some("Rollout gate"),
            ..snapshot("note:a", A, "the canary must pass", None)
        };
        let to = EndpointSnapshot {
            title: None,
            ..snapshot(
                "note:b",
                B,
                "Canary results passed",
                Some(source("source:b", "file:///notes/plan-b.md", "h2")),
            )
        };
        let mut typed = request(
            EvidenceMode::ExplicitStatement,
            RelationDirection::Symmetric,
        );
        assert_eq!(
            validate_request(&EdgeType::RelatedTo, &typed),
            Err(AssessmentViolation::LinkRequired)
        );
        typed.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::From,
            quote: "See plan-b for timing.".into(),
            reference: "plan-b".into(),
        });
        let assessment = assess(&EdgeType::RelatedTo, &typed, "mcp:r", &from, &to).unwrap();
        let link = assessment.link.as_ref().unwrap();
        assert_eq!(link.matched, LinkReferenceKind::SourceFileName);
        assert_eq!(
            (link.note.as_str(), link.target.as_str()),
            ("note:a", "note:b")
        );
        assert_eq!(assessment.verify(&EdgeType::RelatedTo, &from, &to), Ok(()));
        // A shared topic word is not a reference to the other endpoint.
        typed.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::From,
            quote: "the canary must pass".into(),
            reference: "canary".into(),
        });
        assert_eq!(
            assess(&EdgeType::RelatedTo, &typed, "mcp:r", &from, &to),
            Err(AssessmentViolation::LinkReferenceUnmatched)
        );
        typed.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::To,
            quote: "See plan-b".into(),
            reference: "plan-b".into(),
        });
        assert_eq!(
            assess(&EdgeType::RelatedTo, &typed, "mcp:r", &from, &to),
            Err(AssessmentViolation::LinkQuoteNotVerbatim("to"))
        );
        typed.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::From,
            quote: "See plan-b".into(),
            reference: "plan-c".into(),
        });
        assert_eq!(
            validate_request(&EdgeType::RelatedTo, &typed),
            Err(AssessmentViolation::LinkReferenceNotInQuote)
        );
        // Hypotheses may stay link-less because they never enter the graph.
        typed.evidence_mode = EvidenceMode::Hypothesis;
        typed.link = None;
        let pending = assess(&EdgeType::RelatedTo, &typed, "mcp:r", &from, &to).unwrap();
        assert!(!pending.admits_accepted_graph());
    }

    #[test]
    fn copies_and_summaries_of_one_origin_cannot_support_each_other() {
        let shared = source("source:doc", "file:///doc.md", "same");
        let from = snapshot("note:a", A, "the canary must pass", Some(shared));
        let to = snapshot("note:b", B, "Canary results passed", Some(shared));
        let supports = request(EvidenceMode::ExplicitStatement, RelationDirection::FromTo);
        assert_eq!(
            assess(&EdgeType::Supports, &supports, "mcp:r", &from, &to),
            Err(AssessmentViolation::SameOriginSupports)
        );
        let derived = assess(&EdgeType::DerivedFrom, &supports, "mcp:r", &from, &to).unwrap();
        assert!(derived.lineage.same_origin);
        assert_eq!(derived.lineage.basis, LineageBasis::SameSource);
        // Identical content under another URI is a copy, not corroboration.
        let copy_from = snapshot(
            "note:a",
            A,
            "the canary must pass",
            Some(source("source:one", "file:///one.md", "same")),
        );
        let copy_to = snapshot(
            "note:b",
            B,
            "Canary results passed",
            Some(source("source:two", "file:///two.md", "same")),
        );
        assert_eq!(
            assess(
                &EdgeType::Supports,
                &supports,
                "mcp:r",
                &copy_from,
                &copy_to
            ),
            Err(AssessmentViolation::SameOriginSupports)
        );
        let mut independent = supports.clone();
        independent.lineage = DeclaredLineage::Independent;
        assert_eq!(
            assess(
                &EdgeType::DerivedFrom,
                &independent,
                "mcp:r",
                &copy_from,
                &copy_to
            ),
            Err(AssessmentViolation::DeclaredIndependentContradicted(
                "identical source content"
            ))
        );
        // A summary the store cannot detect is declared by the proposer.
        let distinct_from = snapshot("note:a", A, "the canary must pass", None);
        let distinct_to = snapshot("note:b", B, "Canary results passed", None);
        let mut declared = supports.clone();
        declared.lineage = DeclaredLineage::SameOrigin;
        assert_eq!(
            assess(
                &EdgeType::Supports,
                &declared,
                "mcp:r",
                &distinct_from,
                &distinct_to
            ),
            Err(AssessmentViolation::SameOriginSupports)
        );
        let summary = assess(
            &EdgeType::DerivedFrom,
            &declared,
            "mcp:r",
            &distinct_from,
            &distinct_to,
        )
        .unwrap();
        assert_eq!(summary.lineage.basis, LineageBasis::DeclaredSameOrigin);
    }

    #[test]
    fn candidate_scores_and_labels_are_bounded() {
        let mut typed = request(EvidenceMode::Derivation, RelationDirection::FromTo);
        for similarity in [f64::NAN, -0.1, 1.5] {
            typed.candidate = Some(CandidateScore {
                generator: "gardener-similarity".into(),
                generator_version: None,
                similarity: Some(similarity),
            });
            assert_eq!(
                validate_request(&EdgeType::DerivedFrom, &typed),
                Err(AssessmentViolation::CandidateSimilarity)
            );
        }
        typed.candidate = None;
        typed.proposer = Some(ProposerDeclaration {
            generator: "typing\nworker".into(),
            generator_version: None,
            model: None,
        });
        assert_eq!(
            validate_request(&EdgeType::DerivedFrom, &typed),
            Err(AssessmentViolation::LabelBounds("proposer generator"))
        );
    }

    #[test]
    fn stored_assessments_fail_closed_on_tampering_or_unknown_versions() {
        let from = snapshot("note:a", A, "the canary must pass", None);
        let to = snapshot("note:b", B, "Canary results passed", None);
        let assessment = assess(
            &EdgeType::Supports,
            &request(EvidenceMode::ExplicitStatement, RelationDirection::FromTo),
            "mcp:r",
            &from,
            &to,
        )
        .unwrap();
        let stored = serde_json::to_value(&assessment).unwrap();
        let mut future = stored.clone();
        future["version"] = 2.into();
        assert_eq!(
            EvidenceAssessment::from_stored(&future),
            Err(AssessmentViolation::Version(2))
        );
        let mut same_origin = stored.clone();
        same_origin["lineage"]["same_origin"] = true.into();
        assert_eq!(
            EvidenceAssessment::from_stored(&same_origin),
            Err(AssessmentViolation::SameOriginSupports)
        );
        let mut probability = stored.clone();
        probability["calibration"] = "platt".into();
        assert!(EvidenceAssessment::from_stored(&probability).is_err());
        let mut unknown = stored.clone();
        unknown["confidence"] = 0.9.into();
        assert!(EvidenceAssessment::from_stored(&unknown).is_err());
        let summary = EvidenceSummary::describe(Some(&unknown), None, Some(1.0));
        assert_eq!(summary.status, EvidenceStatus::Invalid);
        assert_eq!(summary.token(), "invalid");
    }

    #[test]
    fn legacy_weights_are_unassessed_and_never_percentages() {
        let reviewed =
            EvidenceSummary::describe(None, Some(REVIEWED_ENDPOINT_GENERATOR), Some(1.0));
        assert_eq!(reviewed.status, EvidenceStatus::Unassessed);
        assert_eq!(
            reviewed.legacy_weight_meaning,
            LegacyWeightMeaning::ReviewConvention
        );
        assert_eq!(reviewed.candidate_similarity, None);
        assert_eq!(
            reviewed.label(),
            "unassessed; stored weight 1.00 is a review convention, not a probability"
        );
        let gardener =
            EvidenceSummary::describe(None, Some(GARDENER_SIMILARITY_GENERATOR), Some(0.95));
        assert_eq!(
            gardener.legacy_weight_meaning,
            LegacyWeightMeaning::CandidateSimilarity
        );
        // Stored f32 weights report their shortest decimal, identically in
        // every encoder (JSON envelopes widen f32 otherwise).
        assert_eq!(gardener.candidate_similarity, Some(0.95));
        assert_eq!(gardener.legacy_weight, Some(0.95));
        assert_eq!(
            serde_json::to_string(&gardener.legacy_weight).unwrap(),
            "0.95"
        );
        assert!(gardener.label().contains("candidate similarity 0.95"));
        let absent = EvidenceSummary::describe(None, Some("manual_api"), None);
        assert_eq!(absent.legacy_weight_meaning, LegacyWeightMeaning::Absent);
        assert_eq!(absent.label(), "unassessed; no stored weight");
        let unknown = EvidenceSummary::describe(None, Some("manual"), Some(0.9));
        assert_eq!(
            unknown.legacy_weight_meaning,
            LegacyWeightMeaning::Unspecified
        );
        for summary in [reviewed, gardener, absent, unknown] {
            assert!(!summary.label().contains('%'));
            assert_eq!(summary.token(), "unassessed");
            assert_eq!(
                serde_json::to_value(&summary).unwrap()["calibration"],
                "none"
            );
        }
        assert_eq!(
            EvidenceSummary::default().status,
            EvidenceStatus::Unassessed
        );
    }

    #[test]
    fn assessment_request_rejects_unknown_fields_and_defaults_lineage() {
        let parsed: EvidenceAssessmentRequest = serde_json::from_value(serde_json::json!({
            "evidence_mode": "derivation", "direction": "from_to"
        }))
        .unwrap();
        assert_eq!(parsed.lineage, DeclaredLineage::Unknown);
        assert!(
            serde_json::from_value::<EvidenceAssessmentRequest>(serde_json::json!({
                "evidence_mode": "derivation", "direction": "from_to", "confidence": 0.9
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<EvidenceAssessmentRequest>(serde_json::json!({
                "evidence_mode": "certain", "direction": "from_to"
            }))
            .is_err()
        );
    }

    fn linked_pair() -> (EndpointSnapshot<'static>, EndpointSnapshot<'static>) {
        let from = EndpointSnapshot {
            title: Some("Rollout gate"),
            ..snapshot("note:a", A, "the canary must pass", None)
        };
        let to = snapshot(
            "note:b",
            B,
            "Canary results passed",
            Some(source("source:b", "file:///notes/plan-b.md", "h2")),
        );
        (from, to)
    }

    fn linked_assessment(
        from: &EndpointSnapshot<'_>,
        to: &EndpointSnapshot<'_>,
    ) -> EvidenceAssessment {
        let mut typed = request(
            EvidenceMode::ExplicitStatement,
            RelationDirection::Symmetric,
        );
        typed.link = Some(LinkEvidenceRequest {
            endpoint: EndpointRole::From,
            quote: "See plan-b for timing.".into(),
            reference: "plan-b".into(),
        });
        assess(&EdgeType::RelatedTo, &typed, "mcp:r", from, to).unwrap()
    }

    #[test]
    fn stored_links_must_be_quoted_from_one_endpoint_and_name_the_other() {
        let (from, to) = linked_pair();
        let stored = serde_json::to_value(linked_assessment(&from, &to)).unwrap();
        assert!(EvidenceAssessment::from_stored(&stored).is_ok());
        for (field, value) in [
            ("note", "note:c"),
            ("target", "note:c"),
            ("target", "note:a"),
        ] {
            let mut tampered = stored.clone();
            tampered["link"][field] = value.into();
            assert_eq!(
                EvidenceAssessment::from_stored(&tampered),
                Err(AssessmentViolation::LinkEndpoints),
                "{field} = {value}"
            );
        }
        let mut swapped = stored.clone();
        swapped["link"]["note"] = "note:b".into();
        swapped["link"]["target"] = "note:a".into();
        assert!(EvidenceAssessment::from_stored(&swapped).is_ok());
        let mut unquoted = stored.clone();
        unquoted["link"]["reference"] = "plan-c".into();
        assert_eq!(
            EvidenceAssessment::from_stored(&unquoted),
            Err(AssessmentViolation::LinkReferenceNotInQuote)
        );
        let mut shifted = stored.clone();
        shifted["link"]["byte_end"] = 3.into();
        assert!(matches!(
            EvidenceAssessment::from_stored(&shifted),
            Err(AssessmentViolation::Malformed(_))
        ));
        let mut unlinked = stored.clone();
        unlinked["link"] = serde_json::Value::Null;
        assert_eq!(
            EvidenceAssessment::from_stored(&unlinked),
            Err(AssessmentViolation::LinkRequired)
        );
    }

    #[test]
    fn stored_spans_and_lineage_must_be_internally_consistent() {
        let shared = source("source:doc", "file:///doc.md", "same");
        let from = snapshot("note:a", A, "the canary must pass", Some(shared));
        let to = snapshot("note:b", B, "Canary results passed", Some(shared));
        let derived = assess(
            &EdgeType::DerivedFrom,
            &request(EvidenceMode::Derivation, RelationDirection::FromTo),
            "mcp:r",
            &from,
            &to,
        )
        .unwrap();
        let stored = serde_json::to_value(&derived).unwrap();
        assert!(EvidenceAssessment::from_stored(&stored).is_ok());
        let mut cases: Vec<(&str, serde_json::Value)> = Vec::new();
        let mut independent = stored.clone();
        independent["lineage"] = serde_json::json!({"declared": "unknown", "same_origin": false, "basis": "distinct_origins"});
        cases.push(("one origin recorded as distinct", independent));
        let mut declared = stored.clone();
        declared["lineage"]["declared"] = "independent".into();
        cases.push(("declared independent but same origin", declared));
        let mut unlocated = stored.clone();
        unlocated["endpoints"][0]["support"][0]["byte_end"] = 1.into();
        cases.push(("span offsets", unlocated));
        let mut never = stored.clone();
        never["endpoints"][0]["support"][0]["occurrences"] = 0.into();
        cases.push(("zero occurrences", never));
        let mut padded = stored.clone();
        padded["endpoints"][1]["support"][0]["quote"] = " Canary results passed".into();
        cases.push(("unbounded quote", padded));
        let mut anonymous = stored.clone();
        anonymous["proposer"]["model"] = "fictional-model".into();
        cases.push(("tooling without a generator", anonymous));
        for (case, tampered) in cases {
            assert!(
                EvidenceAssessment::from_stored(&tampered).is_err(),
                "{case}"
            );
        }
    }

    #[test]
    fn verification_rederives_the_whole_assessment_from_the_pinned_endpoints() {
        let (from, to) = linked_pair();
        let assessment = linked_assessment(&from, &to);
        assert_eq!(assessment.verify(&EdgeType::RelatedTo, &from, &to), Ok(()));
        // The link quote must still be verbatim in its holder...
        let edited = EndpointSnapshot {
            content: "Rollout gate: the canary must pass before release.",
            ..from
        };
        assert_eq!(
            assessment.verify(&EdgeType::RelatedTo, &edited, &to),
            Err(AssessmentViolation::LinkQuoteNotVerbatim("from"))
        );
        // ...and still name the opposite endpoint.
        let renamed = EndpointSnapshot {
            source: Some(source("source:b", "file:///notes/plan-c.md", "h2")),
            ..to
        };
        assert_eq!(
            assessment.verify(&EdgeType::RelatedTo, &from, &renamed),
            Err(AssessmentViolation::LinkReferenceUnmatched)
        );
        // Request order and the relation are part of the evidence.
        assert!(assessment.verify(&EdgeType::RelatedTo, &to, &from).is_err());
        assert!(assessment
            .verify(&EdgeType::Contradicts, &from, &to)
            .is_err());
        // Facts that still parse but are not what the server derives.
        let mut occurrences = assessment.clone();
        occurrences.endpoints[0].support[0].occurrences = 2;
        let mut matched = assessment.clone();
        matched.link.as_mut().unwrap().matched = LinkReferenceKind::NoteId;
        let mut origin = assessment.clone();
        origin.endpoints[1].origin = "source:elsewhere".into();
        for tampered in [occurrences, matched, origin] {
            assert_eq!(
                tampered.verify(&EdgeType::RelatedTo, &from, &to),
                Err(AssessmentViolation::Unverified)
            );
        }
        // Lineage is recomputed: a pair that shares a source cannot be
        // accepted as supports, whatever the stored lineage says.
        let supports = assess(
            &EdgeType::Supports,
            &request(EvidenceMode::ExplicitStatement, RelationDirection::FromTo),
            "mcp:r",
            &snapshot("note:a", A, "the canary must pass", None),
            &snapshot("note:b", B, "Canary results passed", None),
        )
        .unwrap();
        let shared = source("source:doc", "file:///doc.md", "same");
        assert_eq!(
            supports.verify(
                &EdgeType::Supports,
                &snapshot("note:a", A, "the canary must pass", Some(shared)),
                &snapshot("note:b", B, "Canary results passed", Some(shared)),
            ),
            Err(AssessmentViolation::SameOriginSupports)
        );
    }
}
