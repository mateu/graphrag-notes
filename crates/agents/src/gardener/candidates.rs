//! High-precision, bounded relationship candidate generation.
//!
//! Embedding similarity only *nominates* a pair. A candidate must also come
//! from different origins (same-source notes are excluded before the
//! neighbour limit), carry a second independent signal (a shared specific
//! entity or an explicit cross-reference), survive existing-link and
//! prior-decision checks, and fit per-endpoint and global budgets. Zero
//! candidates is a valid, reported abstention. Persistence creates pending
//! proposals only; nothing here accepts an edge.

use crate::Result;
use graphrag_core::{record_id_to_string, EdgeType, ProposedEdge};
use graphrag_db::repository::{
    CandidateAnchorScope, CandidateExistingLinks, CandidateNote, CandidateProposalOutcome,
    EdgeProposalDraft,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::time::{Duration, Instant};
use surrealdb::types::RecordId;

/// Generator recorded on candidate proposals. Distinct from the legacy
/// `gardener-similarity` family, so the configured auto-apply policy and
/// batch acceptance never select these review-only proposals. They carry no
/// evidence assessment; review surfaces report them as unassessed with the
/// stored weight read as candidate similarity.
pub const CANDIDATE_GENERATOR: &str =
    graphrag_core::relation_evidence::GARDENER_CANDIDATE_GENERATOR;
/// Version of the candidate reason/feature contract.
pub const CANDIDATE_CONTRACT: u32 = 1;
/// Most distinct shared names whose frequency one scan evaluates. Names past
/// this bound are counted as unevaluated and never become a signal.
pub const MAX_FREQUENCY_NAMES: usize = 256;
/// Frequency probes read at most `(cutoff + 1) * FACTOR` indexed rows.
const FREQUENCY_PROBE_FACTOR: usize = 4;
/// Shared entities recorded per candidate, rarest first.
const MAX_RECORDED_ENTITIES: usize = 8;
const MAX_RECORDED_NAME_CHARS: usize = 128;
/// Share of the deadline available to nomination; the rest is reserved for
/// evidence reads and persistence so a busy scan can still finish its work.
const NOMINATION_SHARE: f64 = 0.6;

/// Operator-owned precision policy. Requests may only tighten it.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePolicy {
    /// Minimum cosine similarity that nominates a pair.
    pub similarity_floor: f32,
    /// Cross-source neighbours read per examined endpoint.
    pub neighbors_per_endpoint: usize,
    /// Most candidates any one note may appear in per scan.
    pub max_per_endpoint: usize,
    /// A shared name mentioned by more notes than this is generic and is
    /// not evidence of a specific relationship.
    pub max_entity_note_frequency: usize,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            similarity_floor: 0.80,
            neighbors_per_endpoint: 8,
            max_per_endpoint: 2,
            max_entity_note_frequency: 10,
        }
    }
}

/// Which notes a scan examines as anchors.
#[derive(Debug, Clone)]
pub enum CandidateScope {
    Notes(Vec<RecordId>),
    Sources(Vec<RecordId>),
    Orphans,
}

#[derive(Debug, Clone)]
pub struct CandidateScanRequest {
    pub scope: CandidateScope,
    /// Restrict the other endpoint to these sources.
    pub counterpart_sources: Option<Vec<RecordId>>,
    /// Resume after this anchor (exclusive), in record-ID order.
    pub after: Option<RecordId>,
    pub max_candidates: usize,
    pub max_endpoints: usize,
    pub deadline: Duration,
    /// Can only raise the policy's similarity floor.
    pub min_similarity: Option<f32>,
    /// Can only lower the policy's per-endpoint cap.
    pub max_per_endpoint: Option<usize>,
    pub dry_run: bool,
    /// Audit actor recorded in persisted candidate features.
    pub requested_by: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateStopReason {
    Completed,
    EndpointBudget,
    CandidateBudget,
    Deadline,
}

impl CandidateStopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::EndpointBudget => "endpoint_budget",
            Self::CandidateBudget => "candidate_budget",
            Self::Deadline => "deadline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateSignal {
    ExplicitCrossReference,
    SharedSpecificEntity,
}

impl CandidateSignal {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitCrossReference => "explicit_cross_reference",
            Self::SharedSpecificEntity => "shared_specific_entity",
        }
    }
}

/// Strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CrossReferenceKind {
    RecordId,
    SourceUri,
    Wikilink,
    FileName,
}

impl CrossReferenceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RecordId => "record_id",
            Self::SourceUri => "source_uri",
            Self::Wikilink => "wikilink",
            Self::FileName => "file_name",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossReferenceDirection {
    FromMentionsTo,
    ToMentionsFrom,
    Both,
}

impl CrossReferenceDirection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FromMentionsTo => "from_mentions_to",
            Self::ToMentionsFrom => "to_mentions_from",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossReferenceEvidence {
    pub direction: CrossReferenceDirection,
    pub kind: CrossReferenceKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedEntityEvidence {
    pub name: String,
    /// Distinct notes whose mention evidence carries this name.
    pub note_frequency: usize,
}

/// One cross-source `related_to` candidate, in canonical endpoint order.
#[derive(Debug, Clone)]
pub struct RelationshipCandidate {
    pub from: RecordId,
    pub to: RecordId,
    pub from_source: Option<RecordId>,
    pub to_source: Option<RecordId>,
    pub similarity: f32,
    pub signals: Vec<CandidateSignal>,
    pub shared_entities: Vec<SharedEntityEvidence>,
    pub cross_reference: Option<CrossReferenceEvidence>,
    /// The persisted proposal, for a non-dry run that reached it.
    pub proposal: Option<ProposedEdge>,
}

/// Why nominated pairs were not selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateRejections {
    /// Both notes share chat-conversation provenance. (Same-source notes are
    /// excluded before nomination and are not counted here.)
    pub same_origin: usize,
    pub duplicate_content: usize,
    pub no_second_signal: usize,
    pub generic_entities_only: usize,
    pub already_connected: usize,
    /// A pending or accepting proposal of any kind already joins the pair.
    pub already_proposed: usize,
    /// A rejected, superseded or accepted proposal already joins the pair;
    /// prior decisions are never resurrected.
    pub prior_decision: usize,
    pub endpoint_cap: usize,
    pub over_budget: usize,
}

#[derive(Debug, Clone)]
pub struct CandidateScanReport {
    /// Effective policy after request tightening.
    pub policy: CandidatePolicy,
    pub candidates: Vec<RelationshipCandidate>,
    pub endpoints_examined: usize,
    /// True when every anchor in scope after the cursor was examined.
    pub scope_exhausted: bool,
    /// Resume cursor when anchors remain. Never set for a dry run that cut
    /// candidates; see [`CandidateScanReport::dry_run_cut`].
    pub next_after: Option<RecordId>,
    /// Distinct cross-source pairs at or above the similarity floor.
    pub nominations: usize,
    pub rejections: CandidateRejections,
    /// Shared names past [`MAX_FREQUENCY_NAMES`], never used as a signal.
    pub unevaluated_names: usize,
    pub stop_reason: CandidateStopReason,
    pub elapsed: Duration,
    pub dry_run: bool,
    pub embedding_model: Option<String>,
}

impl CandidateScanReport {
    /// Candidates a dry run cut at `max_candidates`. A dry run records
    /// nothing for a later scan to skip, so repeating it from any cursor
    /// would return the same page forever: such a report carries no resume
    /// cursor. Narrowing the scope, raising the budget, or a persisting run
    /// (whose pending proposals later scans skip) reaches the cut candidates.
    pub fn dry_run_cut(&self) -> Option<usize> {
        let cut = self.rejections.over_budget;
        (self.dry_run && cut > 0).then_some(cut)
    }
}

impl CandidatePolicy {
    /// Apply request overrides that may only make the policy stricter.
    pub fn tightened(&self, min_similarity: Option<f32>, max_per_endpoint: Option<usize>) -> Self {
        let mut policy = self.clone();
        if let Some(floor) = min_similarity.filter(|floor| floor.is_finite()) {
            policy.similarity_floor = policy.similarity_floor.max(floor.clamp(0.0, 1.0));
        }
        if let Some(cap) = max_per_endpoint.filter(|cap| *cap > 0) {
            policy.max_per_endpoint = policy.max_per_endpoint.min(cap);
        }
        policy
    }
}

struct Pair {
    from: CandidateNote,
    to: CandidateNote,
    similarity: f32,
}

struct Gated {
    pair: Pair,
    signals: Vec<CandidateSignal>,
    shared_entities: Vec<SharedEntityEvidence>,
    cross_reference: Option<CrossReferenceEvidence>,
}

fn key(id: &RecordId) -> String {
    record_id_to_string(id)
}

/// Await a bounded read; `None` means the deadline passed first.
async fn before<T>(
    deadline: Instant,
    future: impl Future<Output = graphrag_db::Result<T>>,
) -> Result<Option<T>> {
    match tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), future).await {
        Ok(result) => Ok(Some(result?)),
        Err(_) => Ok(None),
    }
}

/// Dates and number-only names describe co-occurrence, not a relationship.
fn eligible_name(name: &str, kind: Option<&str>) -> bool {
    kind != Some("date") && name.chars().any(char::is_alphabetic)
}

impl super::GardenerAgent {
    /// Generate high-precision `related_to` candidates within explicit
    /// budgets. A dry run is fully read-only; otherwise only pending
    /// proposals are created through the existing proposal lifecycle.
    pub async fn scan_candidates(
        &self,
        policy: &CandidatePolicy,
        request: CandidateScanRequest,
    ) -> Result<CandidateScanReport> {
        let started = Instant::now();
        let policy = policy.tightened(request.min_similarity, request.max_per_endpoint);
        let deadline = started + request.deadline;
        let nomination_deadline =
            started + Duration::from_secs_f64(request.deadline.as_secs_f64() * NOMINATION_SHARE);
        let mut report = CandidateScanReport {
            policy: policy.clone(),
            candidates: Vec::new(),
            endpoints_examined: 0,
            scope_exhausted: false,
            next_after: request.after.clone(),
            nominations: 0,
            rejections: CandidateRejections::default(),
            unevaluated_names: 0,
            stop_reason: CandidateStopReason::Deadline,
            elapsed: Duration::ZERO,
            dry_run: request.dry_run,
            embedding_model: None,
        };
        let finish = |mut report: CandidateScanReport| -> Result<CandidateScanReport> {
            report.elapsed = started.elapsed();
            Ok(report)
        };
        if request.max_candidates == 0 || request.max_endpoints == 0 {
            return Err(crate::AgentError::Processing(
                "candidate scans require positive candidate and endpoint budgets".into(),
            ));
        }

        let repo = &self.repo;
        let Some(metadata) = before(deadline, repo.portable_embedding_metadata()).await? else {
            return finish(report);
        };
        report.embedding_model = metadata.map(|metadata| metadata.embedding.model);

        // 1. Anchors, in record-ID order after the cursor.
        let scope = match &request.scope {
            CandidateScope::Notes(ids) => CandidateAnchorScope::Notes(ids),
            CandidateScope::Sources(ids) => CandidateAnchorScope::Sources(ids),
            CandidateScope::Orphans => CandidateAnchorScope::Orphans,
        };
        let Some(mut anchors) = before(
            nomination_deadline,
            repo.candidate_anchor_notes(
                scope,
                request.after.as_ref(),
                request.max_endpoints.saturating_add(1),
            ),
        )
        .await?
        else {
            return finish(report);
        };
        let more_in_scope = anchors.len() > request.max_endpoints;
        anchors.truncate(request.max_endpoints);

        // 2. Nominate cross-source neighbours above the floor.
        let mut pairs = BTreeMap::<(String, String), Pair>::new();
        let mut time_limited = false;
        for anchor in &anchors {
            if Instant::now() >= nomination_deadline {
                time_limited = true;
                break;
            }
            let Some(neighbours) = before(
                nomination_deadline,
                repo.candidate_similar_notes(
                    anchor,
                    policy.similarity_floor,
                    policy.neighbors_per_endpoint,
                    request.counterpart_sources.as_deref(),
                ),
            )
            .await?
            else {
                time_limited = true;
                break;
            };
            report.endpoints_examined += 1;
            for neighbour in neighbours {
                let similarity = neighbour.similarity.unwrap_or_default();
                let mut from = anchor.clone();
                from.embedding = Vec::new();
                let mut to = neighbour;
                if key(&to.id) < key(&from.id) {
                    std::mem::swap(&mut from, &mut to);
                }
                pairs.entry((key(&from.id), key(&to.id))).or_insert(Pair {
                    from,
                    to,
                    similarity,
                });
            }
        }
        report.nominations = pairs.len();
        let examined = report.endpoints_examined;
        report.scope_exhausted = !more_in_scope && examined == anchors.len();

        // 3–5. Evidence, signals and existing-link exclusions. A read that
        // misses the deadline abstains without advancing the cursor.
        let Some(mut gated) = self
            .gate_pairs(&policy, pairs, deadline, &mut report)
            .await?
        else {
            report.scope_exhausted = false;
            return finish(report);
        };

        // 6. Rank, then apply per-endpoint and global budgets.
        gated.sort_by(|left, right| {
            right
                .cross_reference
                .is_some()
                .cmp(&left.cross_reference.is_some())
                .then_with(|| right.shared_entities.len().cmp(&left.shared_entities.len()))
                .then_with(|| right.pair.similarity.total_cmp(&left.pair.similarity))
                .then_with(|| key(&left.pair.from.id).cmp(&key(&right.pair.from.id)))
                .then_with(|| key(&left.pair.to.id).cmp(&key(&right.pair.to.id)))
        });
        let mut per_endpoint = HashMap::<String, usize>::new();
        for item in gated {
            let (from, to) = (key(&item.pair.from.id), key(&item.pair.to.id));
            if per_endpoint.get(&from).copied().unwrap_or(0) >= policy.max_per_endpoint
                || per_endpoint.get(&to).copied().unwrap_or(0) >= policy.max_per_endpoint
            {
                report.rejections.endpoint_cap += 1;
                continue;
            }
            if report.candidates.len() >= request.max_candidates {
                report.rejections.over_budget += 1;
                continue;
            }
            *per_endpoint.entry(from).or_default() += 1;
            *per_endpoint.entry(to).or_default() += 1;
            report.candidates.push(RelationshipCandidate {
                from: item.pair.from.id,
                to: item.pair.to.id,
                from_source: item.pair.from.source_id,
                to_source: item.pair.to.source_id,
                similarity: item.pair.similarity,
                signals: item.signals,
                shared_entities: item.shared_entities,
                cross_reference: item.cross_reference,
                proposal: None,
            });
        }

        report.stop_reason = if time_limited {
            CandidateStopReason::Deadline
        } else if report.rejections.over_budget > 0 {
            CandidateStopReason::CandidateBudget
        } else if more_in_scope {
            CandidateStopReason::EndpointBudget
        } else {
            CandidateStopReason::Completed
        };

        // 7. Pending-only persistence through the existing lifecycle. Each
        // pair is re-checked under the proposal lifecycle lock, in the same
        // critical section as its insert, so a relation or proposal written
        // concurrently since the gate read is skipped and counted, never
        // refreshed. The deadline bounds the wait for that lock; once it is
        // held the check and write run to completion. Candidates the deadline
        // cuts off stay in the report without a proposal.
        let mut persistence_cut = false;
        if !request.dry_run {
            let selected = std::mem::take(&mut report.candidates);
            let mut remaining = selected.into_iter();
            while let Some(mut candidate) = remaining.next() {
                let draft = EdgeProposalDraft {
                    from_id: candidate.from.clone(),
                    to_id: candidate.to.clone(),
                    edge_type: EdgeType::RelatedTo,
                    confidence: candidate.similarity,
                    reason: candidate_reason(&candidate, &report, &request.requested_by),
                    generator: CANDIDATE_GENERATOR.into(),
                    generator_version: Some(CANDIDATE_CONTRACT.to_string()),
                    model: report.embedding_model.clone(),
                };
                let outcome = if Instant::now() < deadline {
                    repo.record_candidate_proposal_before(
                        draft,
                        tokio::time::Instant::from_std(deadline),
                    )
                    .await?
                } else {
                    None
                };
                let Some(outcome) = outcome else {
                    persistence_cut = true;
                    report.candidates.push(candidate);
                    report.candidates.extend(remaining);
                    break;
                };
                match outcome {
                    CandidateProposalOutcome::Created(proposal) => {
                        candidate.proposal = Some(*proposal);
                        report.candidates.push(candidate);
                    }
                    CandidateProposalOutcome::Existing(links) => {
                        let links = ExistingLinks::from(links);
                        let pair = unordered(&candidate.from, &candidate.to);
                        let rejected = &mut report.rejections;
                        if links.accepted.contains(&pair) {
                            rejected.already_connected += 1;
                        } else if links.decided.contains(&pair) && !links.open.contains(&pair) {
                            rejected.prior_decision += 1;
                        } else {
                            rejected.already_proposed += 1;
                        }
                    }
                }
            }
        }
        if persistence_cut {
            report.stop_reason = CandidateStopReason::Deadline;
        }
        let (next_after, scope_exhausted) = resume_cursor(
            request.after.as_ref(),
            examined.checked_sub(1).map(|index| &anchors[index].id),
            report.scope_exhausted,
            persistence_cut || report.rejections.over_budget > 0,
            request.dry_run,
        );
        report.next_after = next_after;
        report.scope_exhausted = scope_exhausted;
        finish(report)
    }

    /// Apply origin, duplicate, second-signal and existing-link gates.
    /// Returns `None` when a bounded read missed the deadline.
    async fn gate_pairs(
        &self,
        policy: &CandidatePolicy,
        pairs: BTreeMap<(String, String), Pair>,
        deadline: Instant,
        report: &mut CandidateScanReport,
    ) -> Result<Option<Vec<Gated>>> {
        if pairs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let repo = &self.repo;
        let mut notes = BTreeMap::<String, RecordId>::new();
        for pair in pairs.values() {
            notes.insert(key(&pair.from.id), pair.from.id.clone());
            notes.insert(key(&pair.to.id), pair.to.id.clone());
        }
        let note_ids = notes.into_values().collect::<Vec<_>>();

        let Some(provenance) = before(deadline, repo.graph_note_provenance_ids(&note_ids)).await?
        else {
            return Ok(None);
        };
        let Some(mentions) = before(deadline, repo.candidate_note_mentions(&note_ids)).await?
        else {
            return Ok(None);
        };
        let mut names = HashMap::<String, BTreeSet<String>>::new();
        for mention in mentions {
            if let Some(name) = mention
                .name
                .filter(|name| eligible_name(name, mention.kind.as_deref()))
            {
                names.entry(key(&mention.note)).or_default().insert(name);
            }
        }

        // Origin and duplicate gates, then shared names per surviving pair.
        let mut survivors = Vec::new();
        let mut shared_names = BTreeSet::<String>::new();
        for pair in pairs.into_values() {
            let from_origin = provenance.get(&key(&pair.from.id));
            let to_origin = provenance.get(&key(&pair.to.id));
            if from_origin.is_some_and(|from| {
                to_origin.is_some_and(|to| from.iter().any(|id| to.contains(id)))
            }) {
                report.rejections.same_origin += 1;
                continue;
            }
            let same_hash = pair
                .from
                .content_hash
                .as_ref()
                .is_some_and(|hash| Some(hash) == pair.to.content_hash.as_ref());
            if same_hash || pair.from.content.trim() == pair.to.content.trim() {
                report.rejections.duplicate_content += 1;
                continue;
            }
            let shared = match (names.get(&key(&pair.from.id)), names.get(&key(&pair.to.id))) {
                (Some(from), Some(to)) => from.intersection(to).cloned().collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            shared_names.extend(shared.iter().cloned());
            survivors.push((pair, shared));
        }

        // Bounded document frequency for every distinct shared name.
        let evaluated = shared_names
            .iter()
            .take(MAX_FREQUENCY_NAMES)
            .cloned()
            .collect::<Vec<_>>();
        report.unevaluated_names = shared_names.len().saturating_sub(evaluated.len());
        let probe = policy
            .max_entity_note_frequency
            .saturating_add(1)
            .saturating_mul(FREQUENCY_PROBE_FACTOR);
        let Some(frequencies) =
            before(deadline, repo.candidate_name_frequencies(&evaluated, probe)).await?
        else {
            return Ok(None);
        };
        let specific = evaluated
            .into_iter()
            .zip(frequencies)
            .filter(|(_, frequency)| {
                !frequency.truncated
                    && frequency.notes >= 2
                    && frequency.notes <= policy.max_entity_note_frequency
            })
            .map(|(name, frequency)| (name, frequency.notes))
            .collect::<HashMap<_, _>>();

        let mut signalled = Vec::new();
        for (pair, shared) in survivors {
            let mut entities = shared
                .iter()
                .filter_map(|name| {
                    specific.get(name).map(|frequency| SharedEntityEvidence {
                        name: name.chars().take(MAX_RECORDED_NAME_CHARS).collect(),
                        note_frequency: *frequency,
                    })
                })
                .collect::<Vec<_>>();
            entities.sort_by(|left, right| {
                left.note_frequency
                    .cmp(&right.note_frequency)
                    .then_with(|| left.name.cmp(&right.name))
            });
            let specific_count = entities.len();
            entities.truncate(MAX_RECORDED_ENTITIES);
            let cross_reference = cross_reference(&pair.from, &pair.to);
            let mut signals = Vec::new();
            if cross_reference.is_some() {
                signals.push(CandidateSignal::ExplicitCrossReference);
            }
            if specific_count > 0 {
                signals.push(CandidateSignal::SharedSpecificEntity);
            }
            if signals.is_empty() {
                if shared.is_empty() {
                    report.rejections.no_second_signal += 1;
                } else {
                    report.rejections.generic_entities_only += 1;
                }
                continue;
            }
            signalled.push(Gated {
                pair,
                signals,
                shared_entities: entities,
                cross_reference,
            });
        }
        if signalled.is_empty() {
            return Ok(Some(signalled));
        }

        // Existing accepted relations and proposals between the endpoints.
        let mut endpoints = BTreeMap::<String, RecordId>::new();
        for item in &signalled {
            endpoints.insert(key(&item.pair.from.id), item.pair.from.id.clone());
            endpoints.insert(key(&item.pair.to.id), item.pair.to.id.clone());
        }
        let endpoints = endpoints.into_values().collect::<Vec<_>>();
        let Some(links) = before(deadline, repo.candidate_existing_links(&endpoints)).await? else {
            return Ok(None);
        };
        let links = ExistingLinks::from(links);
        let mut gated = Vec::with_capacity(signalled.len());
        for item in signalled {
            let pair = unordered(&item.pair.from.id, &item.pair.to.id);
            if links.accepted.contains(&pair) {
                report.rejections.already_connected += 1;
            } else if links.open.contains(&pair) {
                report.rejections.already_proposed += 1;
            } else if links.decided.contains(&pair) {
                report.rejections.prior_decision += 1;
            } else {
                gated.push(item);
            }
        }
        Ok(Some(gated))
    }
}

/// Where the next page starts, and whether the scope is exhausted. The cursor
/// never advances past anchors whose candidates were not all reported: when
/// the candidate budget or the deadline cut candidates (`unresolved`), a
/// persisting scan repeats from the incoming cursor, and pairs already
/// recorded are then skipped as already proposed. A dry run records nothing,
/// so a repeat would return the same page forever: when it cut candidates
/// there is no cursor at all and paging stops.
fn resume_cursor(
    incoming: Option<&RecordId>,
    last_examined: Option<&RecordId>,
    scope_exhausted: bool,
    unresolved: bool,
    dry_run: bool,
) -> (Option<RecordId>, bool) {
    if unresolved && dry_run {
        (None, false)
    } else if unresolved {
        (incoming.cloned(), false)
    } else if scope_exhausted {
        (None, true)
    } else {
        (last_examined.or(incoming).cloned(), false)
    }
}

fn unordered(left: &RecordId, right: &RecordId) -> (String, String) {
    let (left, right) = (key(left), key(right));
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

struct ExistingLinks {
    accepted: BTreeSet<(String, String)>,
    open: BTreeSet<(String, String)>,
    decided: BTreeSet<(String, String)>,
}

impl From<CandidateExistingLinks> for ExistingLinks {
    fn from(links: CandidateExistingLinks) -> Self {
        let mut existing = Self {
            accepted: BTreeSet::new(),
            open: BTreeSet::new(),
            decided: BTreeSet::new(),
        };
        for (from, to) in &links.accepted {
            existing.accepted.insert(unordered(from, to));
        }
        for proposal in &links.proposals {
            let pair = unordered(&proposal.from_id, &proposal.to_id);
            if matches!(proposal.status.as_str(), "pending" | "accepting") {
                existing.open.insert(pair);
            } else {
                existing.decided.insert(pair);
            }
        }
        existing
    }
}

/// Lowercased file name and stem for a `file://` source URI.
fn file_name(uri: &str) -> Option<(String, String)> {
    let path = uri.strip_prefix("file://")?;
    let name = path.rsplit('/').next()?.to_lowercase();
    let stem = match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() && !extension.is_empty() => stem.to_owned(),
        _ => return None,
    };
    Some((name, stem))
}

fn word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

/// Starts of the whole-token occurrences of `needle` in `haystack`: not glued
/// to a preceding or following identifier character (a trailing sentence
/// period is fine).
fn token_starts<'a>(haystack: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    let bytes = haystack.as_bytes();
    let mut offset = 0;
    std::iter::from_fn(move || {
        if needle.is_empty() {
            return None;
        }
        while let Some(found) = haystack.get(offset..)?.find(needle) {
            let start = offset + found;
            let end = start + needle.len();
            // Step one character so overlapping occurrences are still examined.
            offset = start + haystack[start..].chars().next().map_or(1, char::len_utf8);
            let before_ok = start == 0 || !word_byte(bytes[start - 1]);
            let after_ok = match bytes.get(end) {
                None => true,
                Some(b'.') => bytes
                    .get(end + 1)
                    .is_none_or(|next| !next.is_ascii_alphanumeric()),
                Some(next) => !word_byte(*next),
            };
            if before_ok && after_ok {
                return Some(start);
            }
        }
        None
    })
}

/// `needle` occurs in `haystack` as a whole token.
fn contains_token(haystack: &str, needle: &str) -> bool {
    token_starts(haystack, needle).next().is_some()
}

/// `uri` occurs in `text` as a complete URI: not glued to a preceding
/// identifier character and not continued by more of a URL. A longer path
/// (`/child`), a query, an extension or an escape names a different
/// resource; only a `#fragment`, or one trailing `/` or punctuation mark
/// that ends the URI, may follow it.
fn contains_uri(text: &str, uri: &str) -> bool {
    if uri.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let ends_here = |byte: Option<&u8>| {
        byte.is_none_or(|byte| byte.is_ascii_whitespace() || b")]}>\"'`|<".contains(byte))
    };
    let mut offset = 0;
    while let Some(found) = text[offset..].find(uri) {
        let start = offset + found;
        let end = start + uri.len();
        let before_ok = start == 0 || !word_byte(bytes[start - 1]);
        let after_ok = match bytes.get(end) {
            Some(b'#') => true,
            Some(byte) if b".,;:!?/".contains(byte) => ends_here(bytes.get(end + 1)),
            next => ends_here(next),
        };
        if before_ok && after_ok {
            return true;
        }
        offset = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Components of a written path, outermost first: a `scheme://` prefix is
/// dropped, `/` and `\` separate, and empty and `.` segments are removed.
/// `..` and a URL host stay, so they only ever match a directory of that
/// exact name.
fn path_components(path: &str) -> Vec<String> {
    let path = path.rsplit_once("://").map_or(path, |(_, rest)| rest);
    path.split(['/', '\\'])
        .filter(|segment| !segment.is_empty() && *segment != ".")
        .map(str::to_owned)
        .collect()
}

/// Lowercased directories of a `file://` source URI's path, outermost first.
fn file_directories(uri: &str) -> Vec<String> {
    let mut directories = uri
        .strip_prefix("file://")
        .map(|path| path_components(&path.to_lowercase()))
        .unwrap_or_default();
    directories.pop();
    directories
}

/// Wikilinks in lowercased `text` (`[[dir/stem|alias]]`, `[[stem#heading]]`)
/// as path segments, the last without a Markdown extension, plus the text
/// outside the links.
fn wikilinks(text: &str) -> (Vec<Vec<String>>, String) {
    let mut links = Vec::new();
    let mut outside = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("[[") {
        let Some(close) = rest[open + 2..].find("]]") else {
            break;
        };
        let inner = &rest[open + 2..open + 2 + close];
        let end = open + 2 + close + 2;
        if inner.len() > 512 {
            outside.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        outside.push_str(&rest[..open]);
        outside.push(' ');
        rest = &rest[end..];
        let target = inner.split(['|', '#']).next().unwrap_or_default().trim();
        if target.ends_with(['/', '\\']) {
            continue;
        }
        let mut segments = path_components(target);
        let Some(last) = segments.pop() else {
            continue;
        };
        let stem = last.strip_suffix(".md").unwrap_or(&last);
        if stem.is_empty() {
            continue;
        }
        segments.push(stem.to_owned());
        links.push(segments);
    }
    outside.push_str(rest);
    (links, outside)
}

/// The one rule for a reference written with a path: its directories must
/// be the file's trailing directories, so `project-a/design` names
/// `.../project-a/design.md`, never `.../project-b/design.md`. An
/// unqualified reference (no directories) falls back to the bare name.
fn names_trailing_directories(qualifiers: &[String], directories: &[String]) -> bool {
    qualifiers.len() <= directories.len()
        && qualifiers
            .iter()
            .rev()
            .zip(directories.iter().rev())
            .all(|(qualifier, directory)| qualifier == directory)
}

/// A wikilink names a file when its last segment is the file's stem and its
/// directories pass [`names_trailing_directories`].
fn wikilink_names(link: &[String], directories: &[String], stem: &str) -> bool {
    let Some((last, qualifiers)) = link.split_last() else {
        return false;
    };
    last == stem && names_trailing_directories(qualifiers, directories)
}

/// Directories written immediately before `text[start..]`, as in
/// `project-a/design.md`, `[design](../a/design.md)` or
/// `file:///vault/a/design.md`; empty for an unqualified mention.
fn written_directories(text: &str, start: usize) -> Vec<String> {
    let before = &text[..start];
    if !before.ends_with(['/', '\\']) {
        return Vec::new();
    }
    let token_start = before
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace() || "()[]{}<>\"'`|,;=*".contains(*c))
        .map_or(0, |(index, c)| index + c.len_utf8());
    path_components(&before[token_start..])
}

/// `name` occurs in lowercased `text` as a whole token whose written
/// directories, if any, pass [`names_trailing_directories`]: a file URI or
/// relative link to another folder's file of the same name never counts.
fn mentions_file(text: &str, name: &str, directories: &[String]) -> bool {
    token_starts(text, name)
        .any(|start| names_trailing_directories(&written_directories(text, start), directories))
}

/// The strongest explicit reference from `text` to `target`. File-name forms
/// are skipped when both sources share a file name (the reference would be
/// ambiguous with a self-reference). Every reference written with a path (a
/// qualified wikilink, a file URI, a relative link) must name the target's
/// trailing directories; only an unqualified name falls back to the file
/// name or wikilink stem. A wikilink counts only through its own target,
/// never as a bare file-name mention.
fn reference_kind(
    text: &str,
    target: &CandidateNote,
    own_file: Option<&(String, String)>,
) -> Option<CrossReferenceKind> {
    if contains_token(text, &key(&target.id)) {
        return Some(CrossReferenceKind::RecordId);
    }
    let uri = target.source_uri.as_deref()?;
    if uri.len() >= 12 && contains_uri(text, uri) {
        return Some(CrossReferenceKind::SourceUri);
    }
    let (name, stem) = file_name(uri)?;
    if own_file.is_some_and(|(own, _)| *own == name) || stem.chars().count() < 3 {
        return None;
    }
    let (links, outside) = wikilinks(&text.to_lowercase());
    let directories = file_directories(uri);
    if links
        .iter()
        .any(|link| wikilink_names(link, &directories, &stem))
    {
        return Some(CrossReferenceKind::Wikilink);
    }
    if mentions_file(&outside, &name, &directories) {
        return Some(CrossReferenceKind::FileName);
    }
    None
}

fn cross_reference(from: &CandidateNote, to: &CandidateNote) -> Option<CrossReferenceEvidence> {
    let from_file = from.source_uri.as_deref().and_then(file_name);
    let to_file = to.source_uri.as_deref().and_then(file_name);
    let forward = reference_kind(&from.content, to, from_file.as_ref());
    let backward = reference_kind(&to.content, from, to_file.as_ref());
    let (direction, kind) = match (forward, backward) {
        (Some(forward), Some(backward)) => (CrossReferenceDirection::Both, forward.min(backward)),
        (Some(kind), None) => (CrossReferenceDirection::FromMentionsTo, kind),
        (None, Some(kind)) => (CrossReferenceDirection::ToMentionsFrom, kind),
        (None, None) => return None,
    };
    Some(CrossReferenceEvidence { direction, kind })
}

/// Versioned candidate features stored in the proposal's existing `reason`.
fn candidate_reason(
    candidate: &RelationshipCandidate,
    report: &CandidateScanReport,
    requested_by: &str,
) -> String {
    let mut summary = format!(
        "Cross-source candidate for review: cosine {:.3}",
        candidate.similarity
    );
    if let Some(reference) = candidate.cross_reference {
        summary.push_str(&format!(
            "; explicit {} reference ({})",
            reference.kind.as_str(),
            reference.direction.as_str()
        ));
    }
    if !candidate.shared_entities.is_empty() {
        let names = candidate
            .shared_entities
            .iter()
            .map(|entity| format!("{} ({} notes)", entity.name, entity.note_frequency))
            .collect::<Vec<_>>()
            .join(", ");
        summary.push_str(&format!("; shared specific entities: {names}"));
    }
    let policy = &report.policy;
    serde_json::json!({
        "contract": CANDIDATE_CONTRACT,
        "generator": CANDIDATE_GENERATOR,
        "summary": summary,
        "similarity": candidate.similarity,
        "signals": candidate.signals.iter().map(|signal| signal.as_str()).collect::<Vec<_>>(),
        "shared_entities": candidate.shared_entities.iter().map(|entity| serde_json::json!({
            "name": entity.name,
            "note_frequency": entity.note_frequency,
        })).collect::<Vec<_>>(),
        "cross_reference": candidate.cross_reference.map(|reference| serde_json::json!({
            "kind": reference.kind.as_str(),
            "direction": reference.direction.as_str(),
        })),
        "from": {"id": key(&candidate.from), "source": candidate.from_source.as_ref().map(key)},
        "to": {"id": key(&candidate.to), "source": candidate.to_source.as_ref().map(key)},
        "policy": {
            "similarity_floor": policy.similarity_floor,
            "neighbors_per_endpoint": policy.neighbors_per_endpoint,
            "max_per_endpoint": policy.max_per_endpoint,
            "max_entity_note_frequency": policy.max_entity_note_frequency,
        },
        "embedding_model": report.embedding_model,
        "requested_by": requested_by,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(id: &str, uri: Option<&str>, content: &str) -> CandidateNote {
        CandidateNote {
            id: RecordId::new("note", id),
            source_id: None,
            source_uri: uri.map(str::to_owned),
            content: content.into(),
            content_hash: None,
            embedding: Vec::new(),
            similarity: None,
        }
    }

    #[test]
    fn explicit_cross_references_require_whole_tokens() {
        let target = note("target", Some("file:///vault/daily/2026-08-04.md"), "x");
        for (text, expected) in [
            (
                "See note:target for the fixture.",
                Some(CrossReferenceKind::RecordId),
            ),
            (
                "Copied from file:///vault/daily/2026-08-04.md today",
                Some(CrossReferenceKind::SourceUri),
            ),
            (
                "Follow-up to [[daily/2026-08-04|Tuesday]].",
                Some(CrossReferenceKind::Wikilink),
            ),
            (
                "Follow-up to [[2026-08-04#Decisions]]",
                Some(CrossReferenceKind::Wikilink),
            ),
            (
                "Recorded in 2026-08-04.md.",
                Some(CrossReferenceKind::FileName),
            ),
            (
                "Recorded in (daily/2026-08-04.md)",
                Some(CrossReferenceKind::FileName),
            ),
            ("note:targeted is a different record", None),
            ("Recorded in 2026-08-04.md.bak", None),
            ("Recorded in old-2026-08-04.md", None),
            ("The 2026-08-04 meeting", None),
            ("[[2026-08-045]]", None),
        ] {
            assert_eq!(reference_kind(text, &target, None), expected, "{text}");
        }
        // The same file name in both sources is ambiguous with a self-reference.
        let own = file_name("file:///other/2026-08-04.md");
        assert_eq!(
            reference_kind("Recorded in 2026-08-04.md", &target, own.as_ref()),
            None
        );
        // Short stems/names never count.
        let short = note("short", Some("file:///vault/ab.md"), "x");
        assert_eq!(reference_kind("[[ab]] and ab.md", &short, None), None);
    }

    #[test]
    fn source_uris_must_match_completely() {
        let target = note("target", Some("https://example.test/docs"), "x");
        for (text, expected) in [
            ("Background: https://example.test/docs", true),
            ("See https://example.test/docs.", true),
            ("Is it https://example.test/docs?", true),
            ("See (https://example.test/docs), then", true),
            ("[docs](https://example.test/docs)", true),
            ("<https://example.test/docs>", true),
            ("https://example.test/docs#install covers it", true),
            ("The page https://example.test/docs/ again", true),
            // Longer URLs name different resources.
            ("https://example.test/docs/unrelated", false),
            ("https://example.test/docs/unrelated/", false),
            ("https://example.test/docs?page=2", false),
            ("https://example.test/docs.html", false),
            ("https://example.test/docs-v2", false),
            ("https://example.test/docs%20old", false),
            ("https://example.test/docs:8080", false),
            ("xhttps://example.test/docs", false),
        ] {
            assert_eq!(
                reference_kind(text, &target, None),
                expected.then_some(CrossReferenceKind::SourceUri),
                "{text}"
            );
        }
        let file = note("file", Some("file:///vault/daily/2026-08-04.md"), "x");
        assert_eq!(
            reference_kind("file:///vault/daily/2026-08-04.md.bak", &file, None),
            None
        );
    }

    #[test]
    fn written_paths_must_name_the_targets_trailing_directories() {
        use CrossReferenceKind::{FileName, SourceUri, Wikilink};
        let target = note("target", Some("file:///vault/project-b/design.md"), "x");
        for (text, expected) in [
            // Wikilinks: an unqualified stem falls back; a path must match.
            ("[[design]]", Some(Wikilink)),
            ("[[project-b/design]]", Some(Wikilink)),
            ("[[vault/project-b/design.md|Design]]", Some(Wikilink)),
            ("[[/vault/./project-b/design#Goals]]", Some(Wikilink)),
            ("[[project-a/design]]", None),
            ("[[archive/vault/project-b/design]]", None),
            ("[[../project-b/design]]", None),
            ("[[project-b/]]", None),
            // File URIs: the exact URI, never another folder's file.
            ("file:///vault/project-b/design.md", Some(SourceUri)),
            ("<file:///vault/project-b/design.md#goals>", Some(SourceUri)),
            ("file:///vault/project-a/design.md", None),
            ("file:///archive/vault/project-b/design.md", None),
            ("https://example.test/project-b/design.md", None),
            // Relative Markdown links and bare paths.
            ("[design](design.md)", Some(FileName)),
            ("[design](./design.md)", Some(FileName)),
            ("[design](project-b/design.md)", Some(FileName)),
            ("[design](vault/project-b/design.md)", Some(FileName)),
            ("See project-b\\design.md.", Some(FileName)),
            ("[design](project-a/design.md)", None),
            ("[design](../project-a/design.md)", None),
            // `..` cannot be resolved here, so it never verifies a path.
            ("[design](../project-b/design.md)", None),
            ("See project-a/design.md and project-a\\design.md", None),
            // One qualifying mention suffices; a wikilink is never a bare name.
            ("project-a/design.md, then design.md", Some(FileName)),
            (
                "[[project-a/design]] and project-b/design.md",
                Some(FileName),
            ),
            ("[[project-a/design.md]]", None),
        ] {
            assert_eq!(reference_kind(text, &target, None), expected, "{text}");
        }
    }

    #[test]
    fn qualified_wikilinks_must_name_the_source_directories() {
        let target = note("target", Some("file:///vault/daily/2026-08-04.md"), "x");
        for (text, expected) in [
            ("[[daily/2026-08-04]]", Some(CrossReferenceKind::Wikilink)),
            (
                "[[vault/daily/2026-08-04|x]]",
                Some(CrossReferenceKind::Wikilink),
            ),
            (
                "[[/vault/./daily/2026-08-04]]",
                Some(CrossReferenceKind::Wikilink),
            ),
            (
                "[[Daily/2026-08-04.md#Decisions]]",
                Some(CrossReferenceKind::Wikilink),
            ),
            ("[[weekly/2026-08-04]]", None),
            ("[[other/daily/2026-08-04]]", None),
            ("[[root/vault/daily/2026-08-04]]", None),
            // A qualified wikilink never falls back to its bare file name.
            ("[[weekly/2026-08-04.md]]", None),
            ("[[elsewhere|see 2026-08-04.md]]", None),
            // Plain file-name mentions outside wikilinks are unchanged.
            (
                "[[weekly/2026-08-04]] and 2026-08-04.md",
                Some(CrossReferenceKind::FileName),
            ),
        ] {
            assert_eq!(reference_kind(text, &target, None), expected, "{text}");
        }

        let index = note(
            "index",
            Some("file:///vault/index.md"),
            "Design notes live in [[project-a/design]].",
        );
        let design_a = note("design-a", Some("file:///vault/project-a/design.md"), "A");
        let design_b = note("design-b", Some("file:///vault/project-b/design.md"), "B");
        assert_eq!(
            cross_reference(&index, &design_a),
            Some(CrossReferenceEvidence {
                direction: CrossReferenceDirection::FromMentionsTo,
                kind: CrossReferenceKind::Wikilink,
            })
        );
        assert_eq!(cross_reference(&index, &design_b), None);
    }

    #[test]
    fn cross_reference_direction_and_strongest_kind() {
        let left = note(
            "left",
            Some("file:///vault/left-file.md"),
            "Builds on [[right-file]].",
        );
        let right = note(
            "right",
            Some("file:///vault/right-file.md"),
            "Supersedes note:left.",
        );
        assert_eq!(
            cross_reference(&left, &right),
            Some(CrossReferenceEvidence {
                direction: CrossReferenceDirection::Both,
                kind: CrossReferenceKind::RecordId,
            })
        );
        let silent = note("silent", Some("file:///vault/silent.md"), "Unrelated text");
        let cites_silent = note("cites", Some("file:///vault/cites.md"), "See silent.md");
        assert_eq!(
            cross_reference(&silent, &cites_silent),
            Some(CrossReferenceEvidence {
                direction: CrossReferenceDirection::ToMentionsFrom,
                kind: CrossReferenceKind::FileName,
            })
        );
        assert_eq!(cross_reference(&silent, &right), None);
        assert_eq!(cross_reference(&silent, &note("x", None, "y")), None);
    }

    #[test]
    fn dates_and_number_only_names_are_never_signals() {
        assert!(eligible_name("aurora ledger", Some("project")));
        assert!(eligible_name("migration 0087", Some("other")));
        assert!(!eligible_name("august 4", Some("date")));
        assert!(!eligible_name("2026", Some("other")));
        assert!(!eligible_name("10.5%", None));
    }

    #[test]
    fn the_cursor_never_skips_unresolved_candidates() {
        let incoming = RecordId::new("note", "alpha");
        let last = RecordId::new("note", "mike");
        for dry_run in [false, true] {
            // Nothing cut: dry and persisting runs advance alike.
            assert_eq!(
                resume_cursor(Some(&incoming), Some(&last), false, false, dry_run),
                (Some(last.clone()), false)
            );
            assert_eq!(
                resume_cursor(Some(&incoming), Some(&last), true, false, dry_run),
                (None, true)
            );
            assert_eq!(
                resume_cursor(Some(&incoming), None, false, false, dry_run),
                (Some(incoming.clone()), false)
            );
        }
        // Budget- or deadline-cut candidates in a persisting run: repeat from
        // the incoming cursor, even when every anchor was examined.
        assert_eq!(
            resume_cursor(Some(&incoming), Some(&last), true, true, false),
            (Some(incoming.clone()), false)
        );
        assert_eq!(
            resume_cursor(None, Some(&last), false, true, false),
            (None, false)
        );
        // A dry run that cut candidates has no continuation at all.
        assert_eq!(
            resume_cursor(Some(&incoming), Some(&last), true, true, true),
            (None, false)
        );
        assert_eq!(
            resume_cursor(Some(&incoming), Some(&last), false, true, true),
            (None, false)
        );
    }

    #[test]
    fn requests_can_only_tighten_the_policy() {
        let policy = CandidatePolicy::default();
        let looser = policy.tightened(Some(0.5), Some(10));
        assert_eq!(looser, policy);
        let stricter = policy.tightened(Some(0.9), Some(1));
        assert_eq!(stricter.similarity_floor, 0.9);
        assert_eq!(stricter.max_per_endpoint, 1);
        assert_eq!(policy.tightened(Some(f32::NAN), Some(0)), policy);
    }
}
