//! Fictional fixtures for the high-precision candidate scan.
use super::candidates::*;
use super::GardenerAgent;
use graphrag_core::{
    record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, EntityType, Note,
    ProposedEdgeStatus, Source, SourceType,
};
use graphrag_db::{init_memory, Repository};
use std::time::Duration;
use surrealdb::types::RecordId;

/// Unit vector whose cosine with axis 0 is `cosine`; the remainder lies on a
/// private `axis`, so notes on different axes are mutually orthogonal there.
fn vector(cosine: f32, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = cosine;
    vector[axis] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

fn entity(name: &str) -> Entity {
    let mut entity = Entity::new(name, EntityType::Concept);
    entity.metadata = serde_json::json!({});
    entity
}

struct Corpus {
    repo: Repository,
}

impl Corpus {
    async fn new() -> Self {
        Self {
            repo: Repository::new(init_memory().await.unwrap()),
        }
    }

    async fn source(&self, file: &str) -> RecordId {
        self.repo
            .create_source(
                Source::from_file(format!("/fixtures/vault/{file}"), SourceType::Markdown).unwrap(),
            )
            .await
            .unwrap()
            .id
            .unwrap()
    }

    /// A note in `source` with an embedding and the named entity mentions.
    async fn note(
        &self,
        source: Option<&RecordId>,
        content: &str,
        embedding: Vec<f32>,
        entities: &[&str],
    ) -> RecordId {
        let mut note = Note::new(content).with_embedding(embedding);
        if let Some(source) = source {
            note = note.with_source(source.clone());
        }
        let id = self.repo.create_note(note).await.unwrap().id.unwrap();
        if !entities.is_empty() {
            self.repo
                .upsert_entities_and_link_note(
                    &id,
                    entities.iter().map(|name| entity(name)).collect(),
                )
                .await
                .unwrap();
        }
        id
    }

    /// Mention-only notes that make `name` common across the corpus.
    async fn make_generic(&self, name: &str, notes: usize) {
        for index in 0..notes {
            let id = self
                .repo
                .create_note(Note::new(format!("Fictional filler {name} {index}")))
                .await
                .unwrap()
                .id
                .unwrap();
            self.repo
                .upsert_entities_and_link_note(&id, vec![entity(name)])
                .await
                .unwrap();
        }
    }

    async fn scan(&self, request: CandidateScanRequest) -> CandidateScanReport {
        GardenerAgent::new(self.repo.clone())
            .scan_candidates(&CandidatePolicy::default(), request)
            .await
            .unwrap()
    }

    async fn proposal_count(&self) -> usize {
        self.repo
            .list_edge_proposals(None, 200)
            .await
            .unwrap()
            .len()
    }

    async fn accepted_edge_count(&self) -> usize {
        self.repo.list_note_edges(200).await.unwrap().len()
    }
}

fn request(scope: CandidateScope, dry_run: bool) -> CandidateScanRequest {
    CandidateScanRequest {
        scope,
        counterpart_sources: None,
        after: None,
        max_candidates: 10,
        max_endpoints: 25,
        deadline: Duration::from_secs(30),
        min_similarity: None,
        max_per_endpoint: None,
        dry_run,
        requested_by: "mcp:fixture-gardener".into(),
    }
}

fn pairs(report: &CandidateScanReport) -> Vec<(String, String)> {
    report
        .candidates
        .iter()
        .map(|candidate| {
            (
                record_id_to_string(&candidate.from),
                record_id_to_string(&candidate.to),
            )
        })
        .collect()
}

fn unordered(left: &RecordId, right: &RecordId) -> (String, String) {
    let (left, right) = (record_id_to_string(left), record_id_to_string(right));
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

#[tokio::test]
async fn same_source_neighbours_never_become_candidates() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    let rare = ["Kestrel ledger"];
    let anchor = corpus
        .note(
            Some(&alpha),
            "Alpha chunk one on the Kestrel ledger",
            vector(1.0, 1),
            &rare,
        )
        .await;
    // The adjacent chunk is identical in direction and shares the rare entity.
    let adjacent = corpus
        .note(
            Some(&alpha),
            "Alpha chunk two on the Kestrel ledger",
            vector(1.0, 2),
            &rare,
        )
        .await;
    let other = corpus
        .note(
            Some(&beta),
            "Beta follow-up on the Kestrel ledger",
            vector(0.9, 3),
            &rare,
        )
        .await;

    let report = corpus
        .scan(request(CandidateScope::Sources(vec![alpha.clone()]), true))
        .await;
    assert_eq!(report.endpoints_examined, 2);
    // Equal similarity: canonical record-ID order breaks the tie.
    let mut expected = vec![unordered(&anchor, &other), unordered(&adjacent, &other)];
    expected.sort();
    let selected = pairs(&report);
    assert_eq!(selected, expected);
    assert!(!selected.contains(&unordered(&anchor, &adjacent)));
    for candidate in &report.candidates {
        assert_ne!(candidate.from_source, candidate.to_source);
        assert_eq!(
            candidate.signals,
            vec![CandidateSignal::SharedSpecificEntity]
        );
        assert_eq!(candidate.shared_entities[0].name, "kestrel ledger");
        assert_eq!(candidate.shared_entities[0].note_frequency, 3);
    }
}

#[tokio::test]
async fn generic_shared_entities_are_not_evidence_but_a_rare_one_is() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    corpus.make_generic("Rust", 11).await;
    let left = corpus
        .note(
            Some(&alpha),
            "Alpha Rust build notes",
            vector(1.0, 1),
            &["Rust"],
        )
        .await;
    let right = corpus
        .note(
            Some(&beta),
            "Beta Rust release notes",
            vector(0.92, 2),
            &["Rust"],
        )
        .await;

    let report = corpus
        .scan(request(CandidateScope::Notes(vec![left.clone()]), true))
        .await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.nominations, 1);
    assert_eq!(report.rejections.generic_entities_only, 1);
    assert_eq!(report.stop_reason, CandidateStopReason::Completed);

    // A rare shared entity is specific evidence; the generic one is not recorded.
    corpus
        .repo
        .upsert_entities_and_link_note(&left, vec![entity("Heron cutover 0087")])
        .await
        .unwrap();
    corpus
        .repo
        .upsert_entities_and_link_note(&right, vec![entity("Heron cutover 0087")])
        .await
        .unwrap();
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![left.clone()]), true))
        .await;
    assert_eq!(pairs(&report), vec![unordered(&left, &right)]);
    let names = report.candidates[0]
        .shared_entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["heron cutover 0087"]);
}

#[tokio::test]
async fn abstains_with_zero_candidates_when_no_pair_has_a_second_signal() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    let gamma = corpus.source("gamma.md").await;
    let anchor = corpus
        .note(
            Some(&alpha),
            "Alpha observation",
            vector(1.0, 1),
            &["Plover"],
        )
        .await;
    corpus
        .note(
            Some(&beta),
            "Beta observation",
            vector(0.95, 2),
            &["Avocet"],
        )
        .await;
    // Below the default floor: never nominated.
    corpus
        .note(
            Some(&gamma),
            "Gamma observation",
            vector(0.7, 3),
            &["Plover"],
        )
        .await;

    let report = corpus
        .scan(request(CandidateScope::Notes(vec![anchor]), false))
        .await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.nominations, 1);
    assert_eq!(report.rejections.no_second_signal, 1);
    assert_eq!(report.stop_reason, CandidateStopReason::Completed);
    assert!(report.scope_exhausted);
    assert_eq!(corpus.proposal_count().await, 0);
}

async fn hub(corpus: &Corpus) -> (RecordId, Vec<RecordId>) {
    let hub_source = corpus.source("hub.md").await;
    let hub = corpus
        .note(
            Some(&hub_source),
            "Hub index",
            vector(1.0, 1),
            &["Marker one", "Marker two", "Marker three", "Marker four"],
        )
        .await;
    let mut spokes = Vec::new();
    for (index, (name, cosine)) in [
        ("Marker one", 0.95),
        ("Marker two", 0.93),
        ("Marker three", 0.91),
        ("Marker four", 0.89),
    ]
    .into_iter()
    .enumerate()
    {
        let source = corpus.source(&format!("spoke-{index}.md")).await;
        spokes.push(
            corpus
                .note(
                    Some(&source),
                    &format!("Spoke {index}"),
                    vector(cosine, index + 2),
                    &[name],
                )
                .await,
        );
    }
    (hub, spokes)
}

#[tokio::test]
async fn per_endpoint_and_global_budgets_are_enforced_in_deterministic_order() {
    let corpus = Corpus::new().await;
    let (hub, spokes) = hub(&corpus).await;
    let scope = || CandidateScope::Notes(vec![hub.clone()]);

    let report = corpus.scan(request(scope(), true)).await;
    assert_eq!(report.nominations, 4);
    // Default policy: a note appears in at most two candidates, best first.
    assert_eq!(
        pairs(&report),
        vec![unordered(&hub, &spokes[0]), unordered(&hub, &spokes[1])]
    );
    assert_eq!(report.rejections.endpoint_cap, 2);
    assert_eq!(report.stop_reason, CandidateStopReason::Completed);
    let again = corpus.scan(request(scope(), true)).await;
    assert_eq!(pairs(&again), pairs(&report));

    let mut single = request(scope(), true);
    single.max_candidates = 1;
    let report = corpus.scan(single).await;
    assert_eq!(pairs(&report), vec![unordered(&hub, &spokes[0])]);
    assert_eq!(report.stop_reason, CandidateStopReason::CandidateBudget);
    assert!(report.rejections.over_budget > 0);

    // A request can tighten, never loosen, the operator policy.
    let mut tightened = request(scope(), true);
    tightened.max_per_endpoint = Some(1);
    tightened.min_similarity = Some(0.94);
    let report = corpus.scan(tightened).await;
    assert_eq!(report.policy.max_per_endpoint, 1);
    assert_eq!(report.policy.similarity_floor, 0.94);
    assert_eq!(pairs(&report), vec![unordered(&hub, &spokes[0])]);
    let mut loosened = request(scope(), true);
    loosened.max_per_endpoint = Some(50);
    loosened.min_similarity = Some(0.1);
    assert_eq!(
        corpus.scan(loosened).await.policy,
        CandidatePolicy::default()
    );
}

#[tokio::test]
async fn endpoint_budget_reports_a_resumable_cursor() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let mut anchors = Vec::new();
    for index in 0..3 {
        anchors.push(
            corpus
                .note(
                    Some(&alpha),
                    &format!("Alpha {index}"),
                    vector(1.0, index + 1),
                    &[],
                )
                .await,
        );
    }
    anchors.sort_by_key(record_id_to_string);
    let mut first = request(CandidateScope::Sources(vec![alpha.clone()]), true);
    first.max_endpoints = 2;
    let report = corpus.scan(first).await;
    assert_eq!(report.endpoints_examined, 2);
    assert!(!report.scope_exhausted);
    assert_eq!(report.next_after.as_ref(), Some(&anchors[1]));
    assert_eq!(report.stop_reason, CandidateStopReason::EndpointBudget);

    let mut resumed = request(CandidateScope::Sources(vec![alpha]), true);
    resumed.max_endpoints = 2;
    resumed.after = report.next_after;
    let report = corpus.scan(resumed).await;
    assert_eq!(report.endpoints_examined, 1);
    assert!(report.scope_exhausted);
    assert_eq!(report.next_after, None);
    assert_eq!(report.stop_reason, CandidateStopReason::Completed);
}

#[tokio::test]
async fn an_exhausted_deadline_stops_without_persisting() {
    let corpus = Corpus::new().await;
    let (hub, _) = hub(&corpus).await;
    let mut expired = request(CandidateScope::Notes(vec![hub]), false);
    expired.deadline = Duration::ZERO;
    let report = corpus.scan(expired).await;
    assert_eq!(report.stop_reason, CandidateStopReason::Deadline);
    assert!(report.candidates.is_empty());
    assert_eq!(report.endpoints_examined, 0);
    assert!(!report.scope_exhausted);
    assert_eq!(report.next_after, None);
    assert_eq!(corpus.proposal_count().await, 0);
}

#[tokio::test]
async fn dry_runs_do_not_mutate_and_persistence_is_pending_only() {
    let corpus = Corpus::new().await;
    let (hub, spokes) = hub(&corpus).await;
    let scope = || CandidateScope::Notes(vec![hub.clone()]);

    let preview = corpus.scan(request(scope(), true)).await;
    assert_eq!(preview.candidates.len(), 2);
    assert!(preview.candidates.iter().all(|c| c.proposal.is_none()));
    assert_eq!(corpus.proposal_count().await, 0);
    assert_eq!(corpus.accepted_edge_count().await, 0);

    // Even an agent configured for automatic acceptance only proposes.
    let report = GardenerAgent::new(corpus.repo.clone())
        .with_auto_apply_policy(true, 0.0)
        .scan_candidates(&CandidatePolicy::default(), request(scope(), false))
        .await
        .unwrap();
    assert_eq!(pairs(&report), pairs(&preview));
    let proposals = corpus.repo.list_edge_proposals(None, 200).await.unwrap();
    assert_eq!(proposals.len(), 2);
    for proposal in &proposals {
        assert_eq!(proposal.status, ProposedEdgeStatus::Pending);
        assert_eq!(proposal.edge_type, EdgeType::RelatedTo);
        assert_eq!(proposal.generator, CANDIDATE_GENERATOR);
        assert_eq!(proposal.generator_version.as_deref(), Some("1"));
        let reason: serde_json::Value = serde_json::from_str(&proposal.reason).unwrap();
        assert_eq!(reason["contract"], 1);
        assert_eq!(reason["signals"][0], "shared_specific_entity");
        assert_eq!(reason["shared_entities"][0]["note_frequency"], 2);
        assert_eq!(reason["requested_by"], "mcp:fixture-gardener");
        assert_eq!(reason["policy"]["max_entity_note_frequency"], 10);
        assert!(
            (reason["similarity"].as_f64().unwrap() - f64::from(proposal.confidence)).abs() < 1e-6
        );
    }
    assert!(report.candidates.iter().all(|c| c
        .proposal
        .as_ref()
        .is_some_and(|p| p.status == ProposedEdgeStatus::Pending)));
    assert_eq!(corpus.accepted_edge_count().await, 0);
    // The configured auto-apply selector never picks review-only candidates.
    assert_eq!(
        corpus
            .repo
            .accept_gardener_proposals_above(0.0, Some("fixture auto-apply".into()))
            .await
            .unwrap(),
        0
    );
    assert_eq!(corpus.accepted_edge_count().await, 0);

    // A re-run skips pending pairs instead of refreshing them and surfaces
    // the next-best candidates; existing proposals are untouched.
    let rerun = corpus.scan(request(scope(), true)).await;
    assert_eq!(rerun.rejections.already_proposed, 2);
    assert_eq!(
        pairs(&rerun),
        vec![unordered(&hub, &spokes[2]), unordered(&hub, &spokes[3])]
    );
    let by_id = |mut proposals: Vec<graphrag_core::ProposedEdge>| {
        proposals.sort_by_key(|proposal| record_id_to_string(proposal.id.as_ref().unwrap()));
        proposals
    };
    let before = by_id(proposals);
    let after = by_id(corpus.repo.list_edge_proposals(None, 200).await.unwrap());
    assert_eq!(after.len(), 2);
    for (before, after) in before.iter().zip(&after) {
        assert_eq!(before.updated_at, after.updated_at);
        assert_eq!(before.reason, after.reason);
        assert_eq!(after.status, ProposedEdgeStatus::Pending);
    }
}

#[tokio::test]
async fn connected_pairs_and_prior_decisions_are_never_resurrected() {
    let corpus = Corpus::new().await;
    let (hub, spokes) = hub(&corpus).await;
    corpus
        .repo
        .create_edge(&spokes[0], &hub, EdgeType::DerivedFrom, None)
        .await
        .unwrap();
    let rejected = corpus
        .repo
        .upsert_gardener_proposal(&hub, &spokes[1], 0.93, "fixture".into(), None, None)
        .await
        .unwrap();
    corpus
        .repo
        .reject_edge_proposal(
            rejected.id.as_ref().unwrap(),
            Some("fixture reviewer".into()),
            None,
        )
        .await
        .unwrap();

    let report = corpus
        .scan(request(CandidateScope::Notes(vec![hub.clone()]), false))
        .await;
    assert_eq!(report.rejections.already_connected, 1);
    assert_eq!(report.rejections.prior_decision, 1);
    assert_eq!(
        pairs(&report),
        vec![unordered(&hub, &spokes[2]), unordered(&hub, &spokes[3])]
    );
    let still_rejected = corpus
        .repo
        .get_edge_proposal(rejected.id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(still_rejected.status, ProposedEdgeStatus::Rejected);
    assert_eq!(still_rejected.reason, "fixture");
    assert_eq!(corpus.accepted_edge_count().await, 1);
}

#[tokio::test]
async fn explicit_cross_reference_is_a_second_signal() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let plan = corpus.source("beta-plan.md").await;
    let left = corpus
        .note(
            Some(&alpha),
            "Follow-up to [[beta-plan#Rollout]] decisions.",
            vector(1.0, 1),
            &[],
        )
        .await;
    let right = corpus
        .note(
            Some(&plan),
            "Rollout plan with staged checkpoints.",
            vector(0.9, 2),
            &[],
        )
        .await;
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![left.clone()]), true))
        .await;
    assert_eq!(pairs(&report), vec![unordered(&left, &right)]);
    let candidate = &report.candidates[0];
    assert_eq!(
        candidate.signals,
        vec![CandidateSignal::ExplicitCrossReference]
    );
    let reference = candidate.cross_reference.unwrap();
    assert_eq!(reference.kind, CrossReferenceKind::Wikilink);
    let expected = if record_id_to_string(&left) < record_id_to_string(&right) {
        CrossReferenceDirection::FromMentionsTo
    } else {
        CrossReferenceDirection::ToMentionsFrom
    };
    assert_eq!(reference.direction, expected);
}

#[tokio::test]
async fn duplicate_content_and_shared_conversations_are_one_origin() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    let gamma = corpus.source("gamma.md").await;
    let rare = ["Sandpiper audit"];
    let anchor = corpus
        .note(
            Some(&alpha),
            "Sandpiper audit summary",
            vector(1.0, 1),
            &rare,
        )
        .await;
    // An identical copy imported from another file is a duplicate, not a relation.
    corpus
        .note(
            Some(&beta),
            "Sandpiper audit summary",
            vector(0.99, 2),
            &rare,
        )
        .await;
    // A note from the same conversation shares provenance with the anchor.
    let sibling = corpus
        .note(
            Some(&gamma),
            "Sandpiper audit follow-up",
            vector(0.95, 3),
            &rare,
        )
        .await;
    let now = chrono::Utc::now();
    let conversation = corpus
        .repo
        .upsert_conversation(
            &ChatConversation {
                uuid: "fixture-conversation".into(),
                name: "Fixture".into(),
                summary: "Fixture".into(),
                created_at: now,
                updated_at: now,
                account: None,
                messages: vec![],
            },
            None,
            serde_json::json!({}),
            None,
        )
        .await
        .unwrap();
    for note in [&anchor, &sibling] {
        corpus
            .repo
            .link_note_to_conversation(note, &conversation)
            .await
            .unwrap();
    }
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![anchor]), true))
        .await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.nominations, 2);
    assert_eq!(report.rejections.duplicate_content, 1);
    assert_eq!(report.rejections.same_origin, 1);
}

#[tokio::test]
async fn shared_dates_are_never_a_second_signal() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    let date = || {
        let mut entity = Entity::new("August 4 2026", EntityType::Date);
        entity.metadata = serde_json::json!({});
        entity
    };
    let left = corpus
        .note(Some(&alpha), "Alpha standup notes", vector(1.0, 1), &[])
        .await;
    let right = corpus
        .note(Some(&beta), "Beta standup notes", vector(0.95, 2), &[])
        .await;
    for id in [&left, &right] {
        corpus
            .repo
            .upsert_entities_and_link_note(id, vec![date()])
            .await
            .unwrap();
    }
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![left]), false))
        .await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.nominations, 1);
    assert_eq!(report.rejections.no_second_signal, 1);
    assert_eq!(corpus.proposal_count().await, 0);
}

#[tokio::test]
async fn the_candidate_budget_holds_the_cursor_until_the_window_is_drained() {
    let corpus = Corpus::new().await;
    let (hub, spokes) = hub(&corpus).await;
    let mut single = request(CandidateScope::Notes(vec![hub.clone()]), false);
    single.max_candidates = 1;
    for (run, spoke) in spokes.iter().enumerate() {
        let report = corpus.scan(single.clone()).await;
        // Repeating drains the window in rank order, one new proposal per run.
        assert_eq!(pairs(&report), vec![unordered(&hub, spoke)], "run {run}");
        let remaining = spokes.len() - run - 1;
        assert_eq!(report.rejections.over_budget, remaining, "run {run}");
        assert_eq!(report.rejections.already_proposed, run, "run {run}");
        if remaining > 0 {
            // Budget-cut candidates stay reachable: neither the cursor nor
            // `scope_exhausted` claims the examined anchor is finished.
            assert_eq!(report.stop_reason, CandidateStopReason::CandidateBudget);
            assert_eq!(report.next_after, None);
            assert!(!report.scope_exhausted, "run {run}");
        } else {
            assert_eq!(report.stop_reason, CandidateStopReason::Completed);
            assert!(report.scope_exhausted);
        }
    }
    assert_eq!(corpus.proposal_count().await, spokes.len());
}

/// Unit vector with `cosine` on the `main` axis and the rest on `private`.
fn topic(main: usize, cosine: f32, private: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[main] = cosine;
    vector[private] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

/// Two anchors on orthogonal topics, in record-ID order: the first has one
/// cross-source counterpart sharing a rare name, the second has two.
async fn two_windows(corpus: &Corpus) -> (RecordId, RecordId) {
    let mut anchors = Vec::new();
    for (axis, name) in [(100, "Quill relay"), (200, "Harbor beacon")] {
        let source = corpus.source(&format!("anchor-{axis}.md")).await;
        let content = format!("Fictional anchor about the {name}");
        let id = corpus
            .note(Some(&source), &content, topic(axis, 1.0, axis + 1), &[name])
            .await;
        anchors.push((id, axis, name));
    }
    anchors.sort_by_key(|(id, _, _)| record_id_to_string(id));
    for (rank, (_, axis, name)) in anchors.iter().enumerate() {
        for (offset, cosine) in [0.95_f32, 0.92].into_iter().take(rank + 1).enumerate() {
            let source = corpus
                .source(&format!("counterpart-{axis}-{offset}.md"))
                .await;
            let content = format!("Fictional counterpart {offset} on the {name}");
            corpus
                .note(
                    Some(&source),
                    &content,
                    topic(*axis, cosine, axis + 2 + offset),
                    &[name],
                )
                .await;
        }
    }
    (anchors[0].0.clone(), anchors[1].0.clone())
}

/// One anchor and one candidate per page.
fn paged(
    first: &RecordId,
    second: &RecordId,
    dry_run: bool,
    after: Option<RecordId>,
) -> CandidateScanRequest {
    let mut paged = request(
        CandidateScope::Notes(vec![first.clone(), second.clone()]),
        dry_run,
    );
    paged.max_candidates = 1;
    paged.max_endpoints = 1;
    paged.after = after;
    paged
}

#[tokio::test]
async fn a_dry_run_that_cuts_candidates_returns_no_cursor_and_counts_the_cut() {
    let corpus = Corpus::new().await;
    let (first, second) = two_windows(&corpus).await;

    // Nothing cut: a dry run advances the cursor like a persisting run.
    let page = corpus.scan(paged(&first, &second, true, None)).await;
    assert_eq!(page.candidates.len(), 1);
    assert_eq!(page.rejections.over_budget, 0);
    assert_eq!(page.dry_run_cut(), None);
    assert_eq!(page.stop_reason, CandidateStopReason::EndpointBudget);
    assert_eq!(page.next_after.as_ref(), Some(&first));
    assert!(!page.scope_exhausted);

    // The next window holds two candidates for a budget of one. A dry run
    // records nothing, so any cursor would return this page forever: there
    // is none, and the cut is counted instead.
    let page = corpus
        .scan(paged(&first, &second, true, page.next_after))
        .await;
    assert_eq!(page.candidates.len(), 1);
    assert_eq!(page.stop_reason, CandidateStopReason::CandidateBudget);
    assert_eq!(page.rejections.over_budget, 1);
    assert_eq!(page.dry_run_cut(), Some(1));
    assert_eq!(page.next_after, None);
    assert!(!page.scope_exhausted);

    // A budget that fits the window cuts nothing and exhausts the scope.
    let mut wider = paged(&first, &second, true, Some(first.clone()));
    wider.max_candidates = 2;
    let page = corpus.scan(wider).await;
    assert_eq!(page.candidates.len(), 2);
    assert_eq!(page.dry_run_cut(), None);
    assert!(page.scope_exhausted);
    assert_eq!(page.next_after, None);
    assert_eq!(corpus.proposal_count().await, 0);
}

#[tokio::test]
async fn persisting_runs_page_as_before_and_record_unassessed_proposals() {
    let corpus = Corpus::new().await;
    let (first, second) = two_windows(&corpus).await;

    let page = corpus.scan(paged(&first, &second, false, None)).await;
    assert_eq!(page.candidates.len(), 1);
    assert_eq!(page.next_after.as_ref(), Some(&first));
    // A persisting run that cut candidates repeats from its incoming cursor;
    // the recorded pending proposal lets the repeat progress.
    let page = corpus
        .scan(paged(&first, &second, false, page.next_after))
        .await;
    assert_eq!(page.rejections.over_budget, 1);
    assert_eq!(page.dry_run_cut(), None);
    assert_eq!(page.next_after.as_ref(), Some(&first));
    assert!(!page.scope_exhausted);
    let page = corpus
        .scan(paged(&first, &second, false, page.next_after))
        .await;
    assert_eq!(page.rejections.already_proposed, 1);
    assert_eq!(page.candidates.len(), 1);
    assert!(page.scope_exhausted);
    assert_eq!(page.next_after, None);

    // Candidate proposals carry no evidence assessment: they read as
    // unassessed candidate similarity and stay acceptable through review.
    let proposals = corpus.repo.list_edge_proposals(None, 200).await.unwrap();
    assert_eq!(proposals.len(), 3);
    for proposal in &proposals {
        assert_eq!(proposal.status, ProposedEdgeStatus::Pending);
        assert!(proposal.evidence_assessment.is_none());
        graphrag_db::repository::ensure_acceptable_evidence(proposal).unwrap();
        let summary = proposal.evidence_summary();
        assert_eq!(
            summary.status,
            graphrag_core::relation_evidence::EvidenceStatus::Unassessed
        );
        assert_eq!(
            summary.legacy_weight_meaning,
            graphrag_core::relation_evidence::LegacyWeightMeaning::CandidateSimilarity
        );
        assert!(summary.candidate_similarity.is_some());
    }
    let accepted = corpus
        .repo
        .accept_edge_proposal(
            proposals[0].id.as_ref().unwrap(),
            Some("fixture reviewer".into()),
            None,
            true,
        )
        .await
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    assert_eq!(corpus.accepted_edge_count().await, 1);
}

/// A fictional conversation with `messages` messages; returns their IDs.
async fn conversation(corpus: &Corpus, uuid: &str, messages: usize) -> Vec<RecordId> {
    let now = chrono::Utc::now();
    let conversation = corpus
        .repo
        .upsert_conversation(
            &ChatConversation {
                uuid: uuid.into(),
                name: "Fixture".into(),
                summary: "Fixture".into(),
                created_at: now,
                updated_at: now,
                account: None,
                messages: vec![],
            },
            None,
            serde_json::json!({}),
            None,
        )
        .await
        .unwrap();
    let mut ids = Vec::new();
    for index in 0..messages {
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "sender": "human",
            "text": format!("Fictional message {index} of {uuid}"),
        }))
        .unwrap();
        ids.push(
            corpus
                .repo
                .upsert_message(&conversation, uuid, index, &message, None)
                .await
                .unwrap(),
        );
    }
    ids
}

#[tokio::test]
async fn notes_from_messages_of_one_conversation_share_an_origin() {
    let corpus = Corpus::new().await;
    let rare = ["Osprey relay"];
    let mut notes = Vec::new();
    for (index, (file, cosine)) in [("alpha.md", 1.0), ("beta.md", 0.95), ("gamma.md", 0.93)]
        .into_iter()
        .enumerate()
    {
        let source = corpus.source(file).await;
        let content = format!("Fictional Osprey relay note {index}");
        notes.push(
            corpus
                .note(Some(&source), &content, vector(cosine, index + 1), &rare)
                .await,
        );
    }
    let shared = conversation(&corpus, "fixture-shared", 2).await;
    let elsewhere = conversation(&corpus, "fixture-elsewhere", 1).await;
    // The anchor and its sibling come from different messages of one
    // conversation; the third note comes from another conversation.
    for (note, message) in [
        (&notes[0], &shared[0]),
        (&notes[1], &shared[1]),
        (&notes[2], &elsewhere[0]),
    ] {
        assert!(corpus
            .repo
            .link_note_to_message(note, message)
            .await
            .unwrap());
    }
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![notes[0].clone()]), true))
        .await;
    assert_eq!(report.nominations, 2);
    assert_eq!(report.rejections.same_origin, 1);
    assert_eq!(pairs(&report), vec![unordered(&notes[0], &notes[2])]);
}

#[tokio::test]
async fn legacy_scans_never_reclassify_candidate_proposals() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("alpha.md").await;
    let beta = corpus.source("beta.md").await;
    let rare = ["Kestrel ledger"];
    let anchor = corpus
        .note(Some(&alpha), "Kestrel ledger plan", vector(1.0, 1), &rare)
        .await;
    corpus
        .note(Some(&beta), "Kestrel ledger review", vector(0.95, 2), &rare)
        .await;
    let report = corpus
        .scan(request(CandidateScope::Notes(vec![anchor]), false))
        .await;
    assert_eq!(report.candidates.len(), 1);
    let before = corpus.repo.list_edge_proposals(None, 10).await.unwrap();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].generator, CANDIDATE_GENERATOR);

    // A legacy maintenance run nominates the same orphan pair with automatic
    // acceptance enabled at any score: the candidate keeps its family and
    // evidence and stays pending for individual review.
    let maintenance = GardenerAgent::new(corpus.repo.clone())
        .with_auto_apply_policy(true, 0.0)
        .run_maintenance()
        .await
        .unwrap();
    assert!(maintenance.suggestions_generated >= 1);
    assert_eq!(maintenance.connections_applied, 0);
    let after = corpus.repo.list_edge_proposals(None, 10).await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].generator, CANDIDATE_GENERATOR);
    assert_eq!(after[0].reason, before[0].reason);
    assert_eq!(after[0].updated_at, before[0].updated_at);
    assert_eq!(after[0].status, ProposedEdgeStatus::Pending);
    assert_eq!(corpus.accepted_edge_count().await, 0);
}
