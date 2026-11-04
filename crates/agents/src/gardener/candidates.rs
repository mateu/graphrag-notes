//! High-precision, bounded relationship candidate generation.
//!
//! Embedding similarity only *nominates* a pair. A candidate must also come
//! from different origins (same-source notes are excluded before the
//! neighbour limit), carry a second independent signal (a shared specific
//! entity or an explicit cross-reference), survive existing-link and
//! prior-decision checks, and fit per-endpoint and global budgets. Zero
//! candidates is a valid, reported abstention. Persistence creates pending
//! proposals only; nothing here accepts an edge.
//!
//! Opt-in quality gates (#230), all off unless the operator configures them:
//! a near-duplicate ceiling and shared-span check, machine-dump anchors and
//! counterparts, lineage-aware entity specificity (one origin document, or a
//! recurring document family), a minimum number of specific entities, and an
//! explicit textual-cue second signal. Batched nomination reads the corpus
//! once per group of anchors instead of once per anchor, with identical
//! results. Three further opt-in tightenings: a relative-date cue must
//! resolve to its counterpart (or come with a specific shared entity),
//! recurring timestamped status entries never pair without an explicit
//! reference, and one endpoint never pairs with two identical counterparts.

pub use super::candidate_quality::TextualCueKind;
use super::candidate_quality::{
    lineage_family, machine_line_share, relative_date_resolves, shared_span_share,
    textual_cue_checked, timestamped_entry_template, CueDirection,
};
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
    /// Cross-source neighbours not yet joined to an examined endpoint (by an
    /// accepted relation or any proposal) read per endpoint. Joined ones are
    /// read separately, up to as many again, and only reported.
    pub neighbors_per_endpoint: usize,
    /// Most candidates any one note may appear in per scan.
    pub max_per_endpoint: usize,
    /// A shared name mentioned by more notes than this is generic and is
    /// not evidence of a specific relationship.
    pub max_entity_note_frequency: usize,
    /// Pairs at or above this cosine are near-duplicates (copies, overlapping
    /// chunks, repeated output), never related-candidates. `None`: off.
    pub near_duplicate_ceiling: Option<f32>,
    /// Pairs whose longest shared run of words covers at least this share of
    /// the shorter note are near-duplicates. `None`: off.
    pub max_shared_span: Option<f32>,
    /// Notes whose non-blank lines are at least this share machine output
    /// (JSON/YAML, tables, stack frames) neither anchor nor qualify as a
    /// counterpart. `None`: off.
    pub max_machine_line_share: Option<f32>,
    /// Lineage-aware specificity. When set, two notes of one origin document
    /// are never a pair, and a shared name is specific only if at most this
    /// many origin documents of the pair's document families (dated or
    /// numbered runs of one recurring log, heartbeat or job) mention it.
    /// `None`: off.
    pub max_lineage_documents: Option<usize>,
    /// Without an explicit cross-reference or textual cue, a pair needs at
    /// least this many specific shared entities. `None`: one, as before.
    pub min_specific_entities: Option<usize>,
    /// Explicit textual cues between dated documents ("yesterday's …",
    /// "compared with <date>", an earlier question answered later) count as a
    /// second signal, ranked with cross-references.
    pub textual_cues: bool,
    /// A "yesterday" cue counts only when it resolves to the counterpart (a
    /// sentence naming none of the counterpart's other times and sharing a
    /// topic span with it) or the pair also shares a specific entity.
    /// Requires `textual_cues`. `false`: any "yesterday" cues, as before.
    pub resolve_relative_dates: bool,
    /// Recurring timestamped status entries: a note whose clock-stamped
    /// entry heading recurs, digit-collapsed, in at least this many distinct
    /// entries of its source (a heartbeat, watchdog or status log) pairs only
    /// through an explicit cross-reference. `None`: off.
    pub periodic_status_entries: Option<usize>,
    /// One endpoint is paired with at most one of several counterparts whose
    /// content is identical. `false`: off.
    pub dedupe_counterparts: bool,
    /// Anchors whose neighbours one exact similarity pass ranks together.
    /// `1` runs one similarity query per anchor, as before.
    pub similarity_batch_anchors: usize,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            similarity_floor: 0.80,
            neighbors_per_endpoint: 8,
            max_per_endpoint: 2,
            max_entity_note_frequency: 10,
            near_duplicate_ceiling: None,
            max_shared_span: None,
            max_machine_line_share: None,
            max_lineage_documents: None,
            min_specific_entities: None,
            textual_cues: false,
            resolve_relative_dates: false,
            periodic_status_entries: None,
            dedupe_counterparts: false,
            similarity_batch_anchors: 1,
        }
    }
}

impl CandidatePolicy {
    /// Any opt-in quality gate or cue is enabled.
    pub fn quality_gates_enabled(&self) -> bool {
        self.near_duplicate_ceiling.is_some()
            || self.max_shared_span.is_some()
            || self.max_machine_line_share.is_some()
            || self.max_lineage_documents.is_some()
            || self.min_specific_entities.is_some()
            || self.textual_cues
            || self.resolve_relative_dates
            || self.periodic_status_entries.is_some()
            || self.dedupe_counterparts
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
    TextualCue,
    SharedSpecificEntity,
}

impl CandidateSignal {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitCrossReference => "explicit_cross_reference",
            Self::TextualCue => "textual_cue",
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

/// An explicit textual cue between the endpoints. `direction` names the note
/// that carries the cue (the later note pointing back, or the answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextualCueEvidence {
    pub kind: TextualCueKind,
    pub direction: CrossReferenceDirection,
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
    pub textual_cue: Option<TextualCueEvidence>,
    /// The persisted proposal, for a non-dry run that reached it.
    pub proposal: Option<ProposedEdge>,
}

/// Why nominated pairs were not selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateRejections {
    /// Both notes come from one chat conversation, directly or through any
    /// of its messages. Same-source and same-conversation neighbours are
    /// excluded before nomination and are not counted here, so this only
    /// counts a shared origin recorded after the neighbour read.
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
    /// An endpoint was deleted or its source generation retired after
    /// nomination, before the proposal could be recorded.
    pub vanished: usize,
    /// Both notes come from one origin document (for example two uploaded
    /// parts of one file). Counted only with lineage specificity on.
    pub same_document: usize,
    /// A counterpart is mostly machine output.
    pub machine_dump: usize,
    /// Above the near-duplicate ceiling, or a long shared span of words.
    pub near_duplicate: usize,
    /// Shared names are specific corpus-wide but recur across the pair's
    /// document lineage (templated logs, heartbeat or job output).
    pub lineage_template: usize,
    /// Fewer specific shared entities than the policy requires, and no
    /// explicit cross-reference or textual cue.
    pub insufficient_specific_entities: usize,
    /// The only second signal was a "yesterday" cue that did not resolve to
    /// the counterpart, without a specific shared entity.
    pub unresolved_relative_date: usize,
    /// An endpoint is a recurring timestamped status entry and the pair has
    /// no explicit cross-reference.
    pub periodic_status: usize,
    /// An endpoint was already selected with a counterpart of identical
    /// content.
    pub duplicate_counterpart: usize,
}

#[derive(Debug, Clone)]
pub struct CandidateScanReport {
    /// Effective policy after request tightening.
    pub policy: CandidatePolicy,
    pub candidates: Vec<RelationshipCandidate>,
    pub endpoints_examined: usize,
    /// Examined anchors skipped as mostly machine output; included in
    /// `endpoints_examined`.
    pub machine_dump_anchors: usize,
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
    textual_cue: Option<TextualCueEvidence>,
}

impl Gated {
    /// An explicit pointer between the notes: ranked before entity overlap.
    fn explicit(&self) -> bool {
        self.cross_reference.is_some() || self.textual_cue.is_some()
    }
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
            machine_dump_anchors: 0,
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

        // 2. Nominate cross-source neighbours above the floor, one exact
        // similarity pass per group of anchors (one anchor per pass unless
        // batching is configured). Machine-dump anchors are examined and
        // skipped without a similarity pass.
        let mut pairs = BTreeMap::<(String, String), Pair>::new();
        let mut time_limited = false;
        let group = policy.similarity_batch_anchors.max(1);
        for chunk in anchors.chunks(group) {
            if Instant::now() >= nomination_deadline {
                time_limited = true;
                break;
            }
            let live = chunk
                .iter()
                .filter(|anchor| {
                    policy
                        .max_machine_line_share
                        .is_none_or(|max| machine_line_share(&anchor.content) < max)
                })
                .collect::<Vec<_>>();
            let neighbours = if live.is_empty() {
                Some(Vec::new())
            } else if group == 1 {
                before(
                    nomination_deadline,
                    repo.candidate_similar_notes(
                        live[0],
                        policy.similarity_floor,
                        policy.neighbors_per_endpoint,
                        request.counterpart_sources.as_deref(),
                    ),
                )
                .await?
                .map(|neighbours| vec![neighbours])
            } else {
                let batch = live
                    .iter()
                    .map(|anchor| (*anchor).clone())
                    .collect::<Vec<_>>();
                before(
                    nomination_deadline,
                    repo.candidate_similar_notes_batch(
                        &batch,
                        policy.similarity_floor,
                        policy.neighbors_per_endpoint,
                        request.counterpart_sources.as_deref(),
                    ),
                )
                .await?
            };
            let Some(neighbours) = neighbours else {
                time_limited = true;
                break;
            };
            report.endpoints_examined += chunk.len();
            report.machine_dump_anchors += chunk.len() - live.len();
            for (anchor, neighbours) in live.into_iter().zip(neighbours) {
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
                .explicit()
                .cmp(&left.explicit())
                .then_with(|| right.shared_entities.len().cmp(&left.shared_entities.len()))
                .then_with(|| right.pair.similarity.total_cmp(&left.pair.similarity))
                .then_with(|| key(&left.pair.from.id).cmp(&key(&right.pair.from.id)))
                .then_with(|| key(&left.pair.to.id).cmp(&key(&right.pair.to.id)))
        });
        let mut per_endpoint = HashMap::<String, usize>::new();
        // (endpoint, counterpart content) of selected candidates.
        let mut counterparts = BTreeSet::<(String, String)>::new();
        for item in gated {
            let (from, to) = (key(&item.pair.from.id), key(&item.pair.to.id));
            let sides = policy.dedupe_counterparts.then(|| {
                [
                    (from.clone(), item.pair.to.content.trim().to_owned()),
                    (to.clone(), item.pair.from.content.trim().to_owned()),
                ]
            });
            if sides
                .as_ref()
                .is_some_and(|sides| sides.iter().any(|side| counterparts.contains(side)))
            {
                report.rejections.duplicate_counterpart += 1;
                continue;
            }
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
            counterparts.extend(sides.into_iter().flatten());
            report.candidates.push(RelationshipCandidate {
                from: item.pair.from.id,
                to: item.pair.to.id,
                from_source: item.pair.from.source_id,
                to_source: item.pair.to.source_id,
                similarity: item.pair.similarity,
                signals: item.signals,
                shared_entities: item.shared_entities,
                cross_reference: item.cross_reference,
                textual_cue: item.textual_cue,
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
                    CandidateProposalOutcome::Vanished => report.rejections.vanished += 1,
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

        let Some(provenance) = before(deadline, repo.candidate_note_origins(&note_ids)).await?
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
            // Text gates are synchronous: a gate pass that outlives the
            // deadline abstains without advancing the cursor, like a read.
            if Instant::now() >= deadline {
                return Ok(None);
            }
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
            if let Some(reason) = quality_rejection(policy, &pair) {
                *reason.counter(&mut report.rejections) += 1;
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
        // Globally specific names, with the notes that mention them when
        // lineage specificity needs to know where they recur.
        let specific = if policy.max_lineage_documents.is_some() {
            let Some(evidence) =
                before(deadline, repo.candidate_name_notes(&evaluated, probe)).await?
            else {
                return Ok(None);
            };
            evaluated
                .into_iter()
                .zip(evidence)
                .filter(|(_, evidence)| {
                    !evidence.truncated
                        && evidence.notes.len() >= 2
                        && evidence.notes.len() <= policy.max_entity_note_frequency
                })
                .map(|(name, evidence)| (name, (evidence.notes.len(), evidence.notes)))
                .collect::<HashMap<_, _>>()
        } else {
            let Some(frequencies) =
                before(deadline, repo.candidate_name_frequencies(&evaluated, probe)).await?
            else {
                return Ok(None);
            };
            evaluated
                .into_iter()
                .zip(frequencies)
                .filter(|(_, frequency)| {
                    !frequency.truncated
                        && frequency.notes >= 2
                        && frequency.notes <= policy.max_entity_note_frequency
                })
                .map(|(name, frequency)| (name, (frequency.notes, Vec::new())))
                .collect::<HashMap<_, _>>()
        };
        // Origin documents of every note mentioning a specific name, keyed
        // by native note ID (distinct records may share a display form).
        #[allow(clippy::mutable_key_type)]
        let origins = if policy.max_lineage_documents.is_some() {
            let mentioning = specific
                .values()
                .flat_map(|(_, notes)| notes.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let Some(origins) =
                before(deadline, repo.candidate_note_origin_uris(&mentioning)).await?
            else {
                return Ok(None);
            };
            origins
        } else {
            BTreeMap::new()
        };

        // Recurring timestamped entry templates per source, for the notes
        // whose own entry heading is clock-stamped.
        #[allow(clippy::mutable_key_type)]
        let periodic_templates = if let Some(min_entries) = policy.periodic_status_entries {
            let sources = survivors
                .iter()
                .flat_map(|(pair, _)| [&pair.from, &pair.to])
                .filter(|note| timestamped_entry_template(&note.chunk_heading_path).is_some())
                .filter_map(|note| note.source_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let Some(paths) =
                before(deadline, repo.candidate_source_heading_paths(&sources)).await?
            else {
                return Ok(None);
            };
            paths
                .into_iter()
                .map(|(source, paths)| (source, recurring_entry_templates(&paths, min_entries)))
                .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };
        let periodic = |note: &CandidateNote| {
            note.source_id
                .as_ref()
                .and_then(|source| periodic_templates.get(source))
                .is_some_and(|recurring| {
                    timestamped_entry_template(&note.chunk_heading_path)
                        .is_some_and(|template| recurring.contains(&template))
                })
        };

        let mut signalled = Vec::new();
        for (pair, shared) in survivors {
            if Instant::now() >= deadline {
                return Ok(None);
            }
            let globally_specific = shared.iter().any(|name| specific.contains_key(name));
            let families = policy.max_lineage_documents.map(|_| {
                [&pair.from, &pair.to]
                    .into_iter()
                    .filter_map(|note| note.origin_uri.as_deref().map(lineage_family))
                    .collect::<BTreeSet<_>>()
            });
            let mut entities = shared
                .iter()
                .filter_map(|name| {
                    let (frequency, notes) = specific.get(name)?;
                    if let (Some(max), Some(families)) = (policy.max_lineage_documents, &families) {
                        // Generic within the pair's lineage: mentioned by
                        // more origin documents of its families than allowed.
                        let documents = notes
                            .iter()
                            .filter_map(|note| origins.get(note))
                            .filter(|origin| families.contains(&lineage_family(origin)))
                            .collect::<BTreeSet<_>>();
                        if documents.len() > max {
                            return None;
                        }
                    }
                    Some(SharedEntityEvidence {
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
            if policy.periodic_status_entries.is_some()
                && cross_reference.is_none()
                && (periodic(&pair.from) || periodic(&pair.to))
            {
                report.rejections.periodic_status += 1;
                continue;
            }
            let cue = policy.textual_cues.then(|| {
                textual_cue_checked(
                    &pair.from.content,
                    pair.from.origin_uri.as_deref(),
                    &pair.to.content,
                    pair.to.origin_uri.as_deref(),
                    |direction| {
                        let (cue_note, other) = match direction {
                            CueDirection::FromMentionsTo => (&pair.from, &pair.to),
                            CueDirection::ToMentionsFrom => (&pair.to, &pair.from),
                        };
                        !policy.resolve_relative_dates
                            || specific_count > 0
                            || relative_date_resolves(
                                &cue_note.content,
                                &other.content,
                                &other.chunk_heading_path,
                            )
                    },
                )
            });
            let declined_relative_date =
                cue.is_some_and(|cue| cue.declined_relative_date.is_some());
            let textual_cue =
                cue.and_then(|cue| cue.cue)
                    .map(|(kind, direction)| TextualCueEvidence {
                        kind,
                        direction: match direction {
                            CueDirection::FromMentionsTo => CrossReferenceDirection::FromMentionsTo,
                            CueDirection::ToMentionsFrom => CrossReferenceDirection::ToMentionsFrom,
                        },
                    });
            let mut signals = Vec::new();
            if cross_reference.is_some() {
                signals.push(CandidateSignal::ExplicitCrossReference);
            }
            if textual_cue.is_some() {
                signals.push(CandidateSignal::TextualCue);
            }
            if specific_count > 0 {
                signals.push(CandidateSignal::SharedSpecificEntity);
            }
            if signals.is_empty() {
                if declined_relative_date {
                    report.rejections.unresolved_relative_date += 1;
                } else if shared.is_empty() {
                    report.rejections.no_second_signal += 1;
                } else if globally_specific && policy.max_lineage_documents.is_some() {
                    report.rejections.lineage_template += 1;
                } else {
                    report.rejections.generic_entities_only += 1;
                }
                continue;
            }
            let explicit = cross_reference.is_some() || textual_cue.is_some();
            if !explicit && specific_count < policy.min_specific_entities.unwrap_or(1) {
                report.rejections.insufficient_specific_entities += 1;
                continue;
            }
            signalled.push(Gated {
                pair,
                signals,
                shared_entities: entities,
                cross_reference,
                textual_cue,
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

/// Clock-stamped entry templates that recur in at least `min_entries`
/// entries of one source. `paths` are the source's heading paths in chunk
/// order. A run of consecutive chunks with one path is one entry (one long
/// section split into chunks), and the same heading under another path or
/// after another section is another entry.
fn recurring_entry_templates(paths: &[Vec<String>], min_entries: usize) -> BTreeSet<String> {
    let mut entries = HashMap::<String, usize>::new();
    let mut previous: Option<&Vec<String>> = None;
    for path in paths {
        if previous != Some(path) {
            if let Some(template) = timestamped_entry_template(path) {
                *entries.entry(template).or_default() += 1;
            }
        }
        previous = Some(path);
    }
    entries
        .into_iter()
        .filter(|(_, count)| *count >= min_entries)
        .map(|(template, _)| template)
        .collect()
}

/// An opt-in quality rejection, checked after the origin and exact-duplicate
/// gates and before any entity evidence is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityRejection {
    SameDocument,
    MachineDump,
    NearDuplicate,
}

impl QualityRejection {
    fn counter(self, rejections: &mut CandidateRejections) -> &mut usize {
        match self {
            Self::SameDocument => &mut rejections.same_document,
            Self::MachineDump => &mut rejections.machine_dump,
            Self::NearDuplicate => &mut rejections.near_duplicate,
        }
    }
}

fn quality_rejection(policy: &CandidatePolicy, pair: &Pair) -> Option<QualityRejection> {
    if policy.max_lineage_documents.is_some()
        && pair
            .from
            .origin_uri
            .as_ref()
            .is_some_and(|origin| Some(origin) == pair.to.origin_uri.as_ref())
    {
        return Some(QualityRejection::SameDocument);
    }
    if let Some(max) = policy.max_machine_line_share {
        if machine_line_share(&pair.from.content) >= max
            || machine_line_share(&pair.to.content) >= max
        {
            return Some(QualityRejection::MachineDump);
        }
    }
    if policy
        .near_duplicate_ceiling
        .is_some_and(|ceiling| pair.similarity >= ceiling)
        || policy
            .max_shared_span
            .is_some_and(|max| shared_span_share(&pair.from.content, &pair.to.content) >= max)
    {
        return Some(QualityRejection::NearDuplicate);
    }
    None
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
    if let Some(cue) = candidate.textual_cue {
        summary.push_str(&format!(
            "; textual {} cue ({})",
            cue.kind.as_str(),
            cue.direction.as_str()
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
    let mut reason = serde_json::json!({
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
    });
    // Legacy policies record exactly the legacy features; the opt-in gates
    // and cue are echoed only when they were in force.
    if policy.quality_gates_enabled() {
        reason["policy"]["quality"] = quality_policy_json(policy);
        reason["textual_cue"] = candidate
            .textual_cue
            .map(|cue| {
                serde_json::json!({
                    "kind": cue.kind.as_str(),
                    "direction": cue.direction.as_str(),
                })
            })
            .unwrap_or(serde_json::Value::Null);
    }
    reason.to_string()
}

/// The opt-in quality policy, as persisted in candidate features.
fn quality_policy_json(policy: &CandidatePolicy) -> serde_json::Value {
    let mut quality = serde_json::json!({
        "near_duplicate_ceiling": policy.near_duplicate_ceiling,
        "max_shared_span": policy.max_shared_span,
        "max_machine_line_share": policy.max_machine_line_share,
        "max_lineage_documents": policy.max_lineage_documents,
        "min_specific_entities": policy.min_specific_entities,
        "textual_cues": policy.textual_cues,
    });
    // The cue-precision tightenings are recorded only when in force, so
    // features persisted under the earlier quality policy are unchanged.
    if policy.resolve_relative_dates {
        quality["resolve_relative_dates"] = true.into();
    }
    if let Some(entries) = policy.periodic_status_entries {
        quality["periodic_status_entries"] = entries.into();
    }
    if policy.dedupe_counterparts {
        quality["dedupe_counterparts"] = true.into();
    }
    quality
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
            origin_uri: uri.map(str::to_owned),
            chunk_heading_path: Vec::new(),
        }
    }

    #[test]
    fn recurring_entries_count_sections_not_heading_strings() {
        let path = |parts: &[&str]| {
            parts
                .iter()
                .map(|part| (*part).to_owned())
                .collect::<Vec<_>>()
        };
        let beat = "09:00 — Heartbeat recheck";
        // Three daily sections with one identical entry heading: three entries.
        let daily = [
            path(&["Day one", beat]),
            path(&["Day two", beat]),
            path(&["Day three", beat]),
        ];
        assert_eq!(
            recurring_entry_templates(&daily, 3),
            BTreeSet::from(["#:# — heartbeat recheck".to_owned()])
        );
        // The same heading repeated under one path, separated by another
        // section, is also several entries.
        let repeated = [
            path(&[beat]),
            path(&["Notes"]),
            path(&[beat]),
            path(&["Notes"]),
            path(&[beat]),
        ];
        assert_eq!(recurring_entry_templates(&repeated, 3).len(), 1);
        // One long section split into three chunks is one entry.
        let split = [path(&[beat]), path(&[beat]), path(&[beat])];
        assert!(recurring_entry_templates(&split, 2).is_empty());
        // Headings without a clock time never count.
        let plain = [path(&["Status"]), path(&["Other"]), path(&["Status"])];
        assert!(recurring_entry_templates(&plain, 2).is_empty());
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
