//! Alchemist: read-only, citation-verified synthesis over reviewed relations.
//!
//! The Alchemist explains what the notes collectively say. It reads current
//! chunks and **accepted** note edges only, asks the configured inference
//! provider for a structured draft, and keeps nothing that the Rust verifier
//! cannot pin to a verbatim span of a cited chunk at its pinned revision.
//!
//! Boundaries:
//! - It never creates notes, edges or proposals. Storage is reached only
//!   through [`AlchemistReads`], a read-only facade, plus the existing
//!   `build_context` retrieval path in query mode.
//! - Its output is a regenerable view. It is not stored, is not evidence and
//!   never feeds retrieval.
//! - Evidence strength is wording derived from lineage-deduplicated origin
//!   counts. It is never a probability.

mod gather;
mod generator;
mod lineage;
mod verify;

#[cfg(test)]
mod tests;

pub use generator::{
    draft_json_schema, ConfiguredSynthesisGenerator, FixtureSynthesisGenerator, GenerationRequest,
    SharedSynthesisGenerator, SynthesisGenerator,
};

use crate::search::{AugmentOptions, GraphMode, SearchAgent, SearchScope};
use crate::AgentError;
use async_trait::async_trait;
use graphrag_core::record_id_to_string;
use graphrag_db::repository::NoteEdgeRow;
use graphrag_db::{DbError, RecordInspection, Repository};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use surrealdb::types::RecordId;

/// Wire contract identity echoed by every receipt.
pub const SYNTHESIS_CONTRACT: &str = "alchemist-synthesis-v1";
/// Seed neighbourhoods never expand beyond two accepted-edge hops.
pub const MAX_SEED_HOPS: usize = 2;
/// Hard ceilings that operator configuration cannot exceed.
pub const MAX_CHUNKS_CAP: usize = 32;
pub const MAX_CONTEXT_TOKENS_CAP: usize = 32_768;
pub const MAX_OUTPUT_TOKENS_CAP: usize = 8_192;
pub const MAX_DEADLINE_MS_CAP: u64 = 300_000;
/// Shorter spans would let almost any claim "verify" against common words.
pub const MIN_QUOTE_CHARS: usize = 12;
/// The only relations the Alchemist reads. Proposal tables are never read.
pub const ACCEPTED_EDGE_TYPES: [&str; 4] =
    ["supports", "contradicts", "derived_from", "related_to"];
/// Accepted edges carry no #205 evidence assessment in this build.
pub const EVIDENCE_UNASSESSED: &str = "unassessed";

/// What happens to a claim whose citations do not verify.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CitationPolicy {
    /// Drop uncited or unverifiable claims and count them in the receipt.
    #[default]
    Drop,
    /// Fail the whole request if any citation or claim is rejected.
    Strict,
}

impl CitationPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Drop => "drop",
            Self::Strict => "strict",
        }
    }
}

/// Resolved hard budgets. The deadline covers retrieval, generation and
/// verification together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SynthesisBudgets {
    pub max_chunks: usize,
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub deadline_ms: u64,
    /// Seed mode only; always at most [`MAX_SEED_HOPS`].
    pub max_hops: usize,
}

impl Default for SynthesisBudgets {
    fn default() -> Self {
        Self {
            max_chunks: 8,
            max_context_tokens: 4_000,
            max_output_tokens: 1_024,
            deadline_ms: 60_000,
            max_hops: MAX_SEED_HOPS,
        }
    }
}

impl SynthesisBudgets {
    /// Reject budgets outside the hard ceilings, including zero values.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=MAX_CHUNKS_CAP).contains(&self.max_chunks)
            || !(1..=MAX_CONTEXT_TOKENS_CAP).contains(&self.max_context_tokens)
            || !(1..=MAX_OUTPUT_TOKENS_CAP).contains(&self.max_output_tokens)
            || !(1..=MAX_DEADLINE_MS_CAP).contains(&self.deadline_ms)
            || !(1..=MAX_SEED_HOPS).contains(&self.max_hops)
        {
            return Err(format!(
                "alchemist budgets require max_chunks 1..={MAX_CHUNKS_CAP}, max_context_tokens 1..={MAX_CONTEXT_TOKENS_CAP}, max_output_tokens 1..={MAX_OUTPUT_TOKENS_CAP}, deadline_ms 1..={MAX_DEADLINE_MS_CAP} and max_hops 1..={MAX_SEED_HOPS}"
            ));
        }
        Ok(())
    }
}

/// Exactly one synthesis input.
#[derive(Debug, Clone)]
pub enum SynthesisInput {
    /// Reuse `build_context` retrieval, including accepted-edge graph paths.
    Query {
        query: String,
        scope: SearchScope,
        graph: GraphMode,
    },
    /// Expand a canonical `note:` seed over accepted edges, hops ≤ 2.
    Seed { id: String },
}

#[derive(Debug, Clone)]
pub struct SynthesisPlan {
    pub input: SynthesisInput,
    pub budgets: SynthesisBudgets,
    pub policy: CitationPolicy,
}

/// A verbatim span of a chunk, pinned to the revision the model was shown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Citation {
    /// Prompt label, such as `C2`.
    pub label: String,
    pub chunk_id: String,
    pub revision: String,
    /// Exact text found in the chunk at `revision`.
    pub span: String,
    /// UTF-8 byte offsets of `span` in the pinned chunk content.
    pub start: usize,
    pub end: usize,
    pub origin: String,
}

/// Deterministic wording from lineage-deduplicated origin counts. These are
/// counts, never probabilities or calibrated confidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceStrength {
    pub independent_origins: usize,
    pub cited_chunks: usize,
    /// An accepted contradicts edge or a conflicting claim touches this
    /// claim's evidence.
    pub contested: bool,
    pub wording: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Claim {
    pub text: String,
    /// Always at least one verified citation.
    pub citations: Vec<Citation>,
    pub strength: EvidenceStrength,
}

/// An accepted edge whose endpoints are both evidence chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RelationStep {
    pub edge_id: String,
    pub edge_type: String,
    pub from_chunk: String,
    pub to_chunk: String,
    pub from_origin: String,
    pub to_origin: String,
    /// `derived_from` inside one lineage: a copy or summary, not corroboration.
    pub lineage_collapsed: bool,
    /// Report-only evidence label. Legacy edges are `unassessed`.
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ChainStep {
    pub edge_id: String,
    pub edge_type: String,
    /// Traversal direction relative to the edge (`outbound` or `inbound`).
    pub direction: String,
    /// Origin of the edge's own `from` endpoint (the asserting side).
    pub from_origin: String,
    pub to_origin: String,
    pub evidence: String,
}

/// An accepted path expressed between origin groups. `derived_from` steps
/// inside one lineage are collapsed, so a summary and its origin count once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RelationChain {
    pub target_chunk: String,
    pub origins: Vec<String>,
    pub steps: Vec<ChainStep>,
    pub collapsed_lineage_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    AcceptedContradictsEdge,
    ConflictingClaims,
}

/// A surfaced disagreement. It is deliberately never resolved: there is no
/// winner, verdict or resolution field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Conflict {
    pub kind: ConflictKind,
    /// Model-written subject of a conflicting-claims entry.
    pub topic: Option<String>,
    pub edge_id: Option<String>,
    pub chunks: Vec<String>,
    /// Indexes into `claims`.
    pub claims: Vec<usize>,
}

/// Claims ordered by the creation timestamp of their earliest dated cited
/// note. These dates describe the notes, not necessarily the events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TimelineEntry {
    pub note_date: String,
    pub claim: usize,
    pub chunk_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GapKind {
    NoEvidence,
    UncitedEvidence,
    DroppedClaims,
    SingleOriginClaims,
    NeighbourhoodTruncated,
    UnpinnedEvidence,
    NoAcceptedRelations,
    UnassessedRelations,
}

/// Deterministic coverage gaps, independent of model prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Gap {
    pub kind: GapKind,
    pub detail: String,
    pub chunks: Vec<String>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LineageReason {
    DerivedFrom,
    SameSource,
    SameConversation,
    IdenticalContent,
}

/// Chunks that are copies, summaries or parts of one origin. They are not
/// independent corroboration of each other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OriginGroup {
    pub origin: String,
    pub chunks: Vec<String>,
    pub reasons: Vec<LineageReason>,
}

/// One pinned chunk the synthesis was allowed to cite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceRef {
    pub label: String,
    pub chunk_id: String,
    pub revision: String,
    pub title: Option<String>,
    pub note_date: Option<String>,
    pub source_uri: Option<String>,
    pub origin: String,
}

/// Structured, read-only synthesis. Nothing here is stored or reused as
/// evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Synthesis {
    pub claims: Vec<Claim>,
    pub relation_chains: Vec<RelationChain>,
    pub conflicts: Vec<Conflict>,
    pub timeline: Vec<TimelineEntry>,
    /// Model-generated and unverified; never evidence.
    pub open_questions: Vec<String>,
    pub gaps: Vec<Gap>,
    pub lineage: Vec<OriginGroup>,
    pub relations: Vec<RelationStep>,
    pub evidence: Vec<EvidenceRef>,
}

/// Counts from the fail-closed citation verifier.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct VerificationReport {
    pub claims_proposed: usize,
    pub claims_kept: usize,
    pub claims_dropped: usize,
    pub uncited_claims: usize,
    pub citations_checked: usize,
    pub citations_kept: usize,
    pub citations_rejected: usize,
    /// Rejected citation counts by reason.
    pub rejections: BTreeMap<String, usize>,
    pub conflicts_proposed: usize,
    pub conflicts_dropped: usize,
    pub open_questions_dropped: usize,
    /// Cited chunks re-read after generation to confirm their pinned revision.
    pub chunks_rechecked: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SynthesisUsage {
    pub mode: String,
    pub chunks_considered: usize,
    pub chunks_used: usize,
    pub context_tokens: usize,
    pub relations_considered: usize,
    pub neighbourhood_truncated: bool,
    pub generator_called: bool,
    pub provider: String,
    pub model: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SynthesisOutcome {
    pub synthesis: Synthesis,
    pub verification: VerificationReport,
    pub usage: SynthesisUsage,
}

#[derive(Debug, thiserror::Error)]
pub enum AlchemistError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    NotFound(String),
    /// The draft could not be verified under the active policy.
    #[error("{reason}")]
    Unverifiable {
        reason: String,
        report: Box<VerificationReport>,
    },
    #[error(transparent)]
    Agent(#[from] AgentError),
}

impl From<DbError> for AlchemistError {
    fn from(error: DbError) -> Self {
        Self::Agent(AgentError::Database(error))
    }
}

/// Visible-note metadata used for dates and display.
#[derive(Debug, Clone, Default)]
pub struct NoteMeta {
    pub title: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source_uri: Option<String>,
}

/// The only storage operations the Alchemist can perform. Every method is a
/// read; there is deliberately no way to create or change a record here.
#[async_trait]
pub trait AlchemistReads: Send + Sync {
    /// Exact current content and revision of a canonical record ID.
    async fn pinned_record(&self, id: &str) -> graphrag_db::Result<RecordInspection>;
    /// Accepted note edges of the four persisted relation tables, both
    /// directions, at most `per_note_limit` per note and table.
    async fn accepted_edges(
        &self,
        note_ids: &[RecordId],
        per_note_limit: usize,
        exclude: &HashMap<String, Vec<RecordId>>,
    ) -> graphrag_db::Result<Vec<NoteEdgeRow>>;
    /// Visible (not staged, hidden or superseded) notes among `note_ids`.
    async fn visible_notes(
        &self,
        note_ids: &[RecordId],
    ) -> graphrag_db::Result<HashMap<String, NoteMeta>>;
}

#[async_trait]
impl AlchemistReads for Repository {
    async fn pinned_record(&self, id: &str) -> graphrag_db::Result<RecordInspection> {
        self.inspect_record(id, 0).await
    }

    async fn accepted_edges(
        &self,
        note_ids: &[RecordId],
        per_note_limit: usize,
        exclude: &HashMap<String, Vec<RecordId>>,
    ) -> graphrag_db::Result<Vec<NoteEdgeRow>> {
        let edge_types = ACCEPTED_EDGE_TYPES.map(String::from);
        self.graph_note_edges_excluding_visited(
            note_ids,
            &edge_types,
            per_note_limit,
            true,
            true,
            0.0,
            None,
            None,
            exclude,
        )
        .await
    }

    async fn visible_notes(
        &self,
        note_ids: &[RecordId],
    ) -> graphrag_db::Result<HashMap<String, NoteMeta>> {
        Ok(self
            .graph_notes_by_ids(note_ids, None, None)
            .await?
            .into_iter()
            .map(|note| {
                (
                    record_id_to_string(&note.id),
                    NoteMeta {
                        title: note.title,
                        created_at: Some(note.created_at),
                        source_uri: note.source_uri,
                    },
                )
            })
            .collect())
    }
}

/// Run one bounded, read-only synthesis. Callers enforce the deadline by
/// dropping this future; nothing is persisted, so dropping is always safe.
pub async fn synthesize(
    reads: &dyn AlchemistReads,
    search: &SearchAgent,
    generator: &dyn SynthesisGenerator,
    augment: &AugmentOptions,
    plan: &SynthesisPlan,
) -> Result<SynthesisOutcome, AlchemistError> {
    plan.budgets.validate().map_err(AlchemistError::Invalid)?;
    let started = Instant::now();
    let mut evidence = match &plan.input {
        SynthesisInput::Query {
            query,
            scope,
            graph,
        } => gather::query(reads, search, query, *scope, *graph, &plan.budgets, augment).await?,
        SynthesisInput::Seed { id } => gather::seed(reads, id, &plan.budgets, augment).await?,
    };
    evidence.relations = gather::relations_among(reads, &evidence.chunks).await?;
    let lineage = lineage::origin_groups(&evidence.chunks, &evidence.relations);
    let capabilities = generator.capabilities();
    let mut usage = SynthesisUsage {
        mode: match plan.input {
            SynthesisInput::Query { .. } => "query",
            SynthesisInput::Seed { .. } => "seed",
        }
        .into(),
        chunks_considered: evidence.considered,
        chunks_used: evidence.chunks.len(),
        context_tokens: evidence.context_tokens,
        relations_considered: evidence.relations.len(),
        neighbourhood_truncated: evidence.truncated,
        generator_called: false,
        provider: capabilities.provider,
        model: capabilities.model,
        elapsed_ms: 0,
    };
    let verified = if evidence.chunks.is_empty() {
        // Nothing to cite: never ask a model to synthesize from no evidence.
        verify::Verified::default()
    } else {
        let request = generator::build_request(plan, &evidence, &lineage);
        usage.generator_called = true;
        let raw = generator.generate(&request).await?;
        let draft = verify::parse_draft(&raw)?;
        let rechecked = gather::recheck_cited(reads, &mut evidence, &draft).await?;
        verify::verify(&draft, &evidence.chunks, plan.policy, rechecked)?
    };
    usage.elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let verification = verified.report.clone();
    Ok(SynthesisOutcome {
        synthesis: assemble(&evidence, &lineage, verified),
        verification,
        usage,
    })
}

fn chain_steps(
    path: &[gather::PathStep],
    origin_of: &HashMap<&str, &str>,
) -> (Vec<ChainStep>, usize) {
    let origin = |id: &str| {
        origin_of
            .get(id)
            .map_or_else(|| format!("outside:{id}"), |origin| (*origin).to_string())
    };
    let mut steps = Vec::new();
    let mut collapsed = 0;
    for step in path {
        // `from_id`/`to_id` follow traversal; recover the edge's own endpoints.
        let (from, to) = if step.direction == "inbound" {
            (&step.to_id, &step.from_id)
        } else {
            (&step.from_id, &step.to_id)
        };
        let (from_origin, to_origin) = (origin(from), origin(to));
        if step.edge_type == "derived_from" && from_origin == to_origin {
            collapsed += 1;
            continue;
        }
        steps.push(ChainStep {
            edge_id: step.edge_id.clone(),
            edge_type: step.edge_type.clone(),
            direction: step.direction.clone(),
            from_origin,
            to_origin,
            evidence: EVIDENCE_UNASSESSED.into(),
        });
    }
    (steps, collapsed)
}

fn assemble(
    evidence: &gather::EvidenceSet,
    lineage: &lineage::Lineage,
    verified: verify::Verified,
) -> Synthesis {
    let chunks = &evidence.chunks;
    let origin_of: HashMap<&str, &str> = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (chunk.id.as_str(), lineage.origin_label(index)))
        .collect();
    let index_of: HashMap<&str, usize> = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (chunk.id.as_str(), index))
        .collect();

    let relations: Vec<RelationStep> = evidence
        .relations
        .iter()
        .map(|edge| {
            let (from, to) = (
                record_id_to_string(&edge.in_id),
                record_id_to_string(&edge.out_id),
            );
            let from_origin = origin_of.get(from.as_str()).copied().unwrap_or_default();
            let to_origin = origin_of.get(to.as_str()).copied().unwrap_or_default();
            RelationStep {
                edge_id: record_id_to_string(&edge.id),
                edge_type: edge.edge_type.clone(),
                lineage_collapsed: edge.edge_type == "derived_from" && from_origin == to_origin,
                from_origin: from_origin.into(),
                to_origin: to_origin.into(),
                from_chunk: from,
                to_chunk: to,
                evidence: EVIDENCE_UNASSESSED.into(),
            }
        })
        .collect();

    // Chunks touched by an accepted contradicts edge among the evidence.
    let contradicted: std::collections::HashSet<&str> = relations
        .iter()
        .filter(|relation| relation.edge_type == "contradicts")
        .flat_map(|relation| [relation.from_chunk.as_str(), relation.to_chunk.as_str()])
        .collect();

    let mut conflicts = Vec::new();
    for relation in relations.iter().filter(|r| r.edge_type == "contradicts") {
        let touching = verified
            .claims
            .iter()
            .enumerate()
            .filter(|(_, claim)| {
                claim.citations.iter().any(|citation| {
                    citation.chunk_id == relation.from_chunk
                        || citation.chunk_id == relation.to_chunk
                })
            })
            .map(|(index, _)| index)
            .collect();
        conflicts.push(Conflict {
            kind: ConflictKind::AcceptedContradictsEdge,
            topic: None,
            edge_id: Some(relation.edge_id.clone()),
            chunks: vec![relation.from_chunk.clone(), relation.to_chunk.clone()],
            claims: touching,
        });
    }
    let mut conflicting_claims = std::collections::HashSet::new();
    for (topic, claim_indexes) in &verified.conflicts {
        let mut conflict_chunks: Vec<String> = claim_indexes
            .iter()
            .flat_map(|index| verified.claims[*index].citations.iter())
            .map(|citation| citation.chunk_id.clone())
            .collect();
        conflict_chunks.sort_by_key(|id| index_of.get(id.as_str()).copied());
        conflict_chunks.dedup();
        conflicting_claims.extend(claim_indexes.iter().copied());
        conflicts.push(Conflict {
            kind: ConflictKind::ConflictingClaims,
            topic: Some(topic.clone()),
            edge_id: None,
            chunks: conflict_chunks,
            claims: claim_indexes.clone(),
        });
    }

    let claims: Vec<Claim> = verified
        .claims
        .into_iter()
        .enumerate()
        .map(|(index, claim)| {
            let citations: Vec<Citation> = claim
                .citations
                .into_iter()
                .map(|citation| Citation {
                    origin: origin_of
                        .get(citation.chunk_id.as_str())
                        .copied()
                        .unwrap_or_default()
                        .into(),
                    label: citation.label,
                    chunk_id: citation.chunk_id,
                    revision: citation.revision,
                    span: citation.span,
                    start: citation.start,
                    end: citation.end,
                })
                .collect();
            let contested = conflicting_claims.contains(&index)
                || citations
                    .iter()
                    .any(|citation| contradicted.contains(citation.chunk_id.as_str()));
            let strength = lineage::strength(&citations, contested);
            Claim {
                text: claim.text,
                citations,
                strength,
            }
        })
        .collect();

    let mut relation_chains = Vec::new();
    for chunk in chunks.iter().filter(|chunk| !chunk.path.is_empty()) {
        let (steps, collapsed_lineage_steps) = chain_steps(&chunk.path, &origin_of);
        if steps.is_empty() {
            // A pure lineage path is fully described by `lineage`.
            continue;
        }
        let mut origins = Vec::new();
        for step in &steps {
            for origin in [&step.from_origin, &step.to_origin] {
                if !origins.contains(origin) {
                    origins.push(origin.clone());
                }
            }
        }
        relation_chains.push(RelationChain {
            target_chunk: chunk.id.clone(),
            origins,
            steps,
            collapsed_lineage_steps,
        });
    }

    let mut timeline: Vec<TimelineEntry> = claims
        .iter()
        .enumerate()
        .filter_map(|(index, claim)| {
            claim
                .citations
                .iter()
                .filter_map(|citation| {
                    let chunk = &chunks[*index_of.get(citation.chunk_id.as_str())?];
                    Some((chunk.created_at?, chunk.id.clone()))
                })
                .min()
                .map(|(date, chunk_id)| TimelineEntry {
                    note_date: date.to_rfc3339(),
                    claim: index,
                    chunk_id,
                })
        })
        .collect();
    timeline.sort_by(|a, b| a.note_date.cmp(&b.note_date).then(a.claim.cmp(&b.claim)));

    let gaps = gaps(evidence, &claims, &verified.report, &relations);
    let evidence_refs = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| EvidenceRef {
            label: chunk.label.clone(),
            chunk_id: chunk.id.clone(),
            revision: chunk.revision.clone(),
            title: chunk.title.clone(),
            note_date: chunk.created_at.map(|date| date.to_rfc3339()),
            source_uri: chunk.source_uri.clone(),
            origin: lineage.origin_label(index).into(),
        })
        .collect();
    Synthesis {
        claims,
        relation_chains,
        conflicts,
        timeline,
        open_questions: verified.open_questions,
        gaps,
        lineage: lineage.groups.clone(),
        relations,
        evidence: evidence_refs,
    }
}

fn gaps(
    evidence: &gather::EvidenceSet,
    claims: &[Claim],
    report: &VerificationReport,
    relations: &[RelationStep],
) -> Vec<Gap> {
    let mut gaps = Vec::new();
    let mut gap = |kind, detail: String, chunks: Vec<String>| {
        gaps.push(Gap {
            kind,
            detail,
            chunks,
        })
    };
    if evidence.chunks.is_empty() {
        gap(
            GapKind::NoEvidence,
            "No visible evidence was found; nothing was synthesized.".into(),
            Vec::new(),
        );
    }
    let cited: std::collections::HashSet<&str> = claims
        .iter()
        .flat_map(|claim| {
            claim
                .citations
                .iter()
                .map(|citation| citation.chunk_id.as_str())
        })
        .collect();
    let uncited: Vec<String> = evidence
        .chunks
        .iter()
        .filter(|chunk| !cited.contains(chunk.id.as_str()))
        .map(|chunk| chunk.id.clone())
        .collect();
    if !uncited.is_empty() && !claims.is_empty() {
        gap(
            GapKind::UncitedEvidence,
            format!("{} evidence chunk(s) support no kept claim.", uncited.len()),
            uncited,
        );
    }
    if report.claims_dropped > 0 {
        gap(
            GapKind::DroppedClaims,
            format!(
                "{} claim(s) were dropped because no citation verified ({} uncited).",
                report.claims_dropped, report.uncited_claims
            ),
            Vec::new(),
        );
    }
    let single = claims
        .iter()
        .filter(|claim| claim.strength.independent_origins == 1)
        .count();
    if single > 0 {
        gap(
            GapKind::SingleOriginClaims,
            format!("{single} claim(s) rest on a single origin after lineage de-duplication."),
            Vec::new(),
        );
    }
    if evidence.truncated {
        gap(
            GapKind::NeighbourhoodTruncated,
            "Budgets cut the evidence set; more accepted neighbours or chunks were available."
                .into(),
            Vec::new(),
        );
    }
    if !evidence.unpinned.is_empty() {
        gap(
            GapKind::UnpinnedEvidence,
            "Some retrieved records disappeared before their revision could be pinned.".into(),
            evidence.unpinned.clone(),
        );
    }
    if !evidence.chunks.is_empty() {
        if relations.is_empty() {
            gap(
                GapKind::NoAcceptedRelations,
                "No accepted edge connects the evidence chunks.".into(),
                Vec::new(),
            );
        } else {
            gap(
                GapKind::UnassessedRelations,
                "Accepted edges carry no evidence assessment yet; they are shown as unassessed."
                    .into(),
                Vec::new(),
            );
        }
    }
    gaps
}
