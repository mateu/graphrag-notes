//! Fictional fixtures for the opt-in cue-precision tightenings (#230):
//! relative-date resolution, recurring timestamped status entries and
//! identical-counterpart deduplication.
use super::candidates::*;
use super::GardenerAgent;
use graphrag_core::{record_id_to_string, Entity, EntityType, Note, Source, SourceType};
use graphrag_db::{init_memory, Repository};
use std::time::Duration;
use surrealdb::types::RecordId;

fn vector(cosine: f32, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = cosine;
    vector[axis] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

/// The quality policy measured by the first #230 validation.
fn quality() -> CandidatePolicy {
    CandidatePolicy {
        near_duplicate_ceiling: Some(0.95),
        max_shared_span: Some(0.5),
        max_machine_line_share: Some(0.6),
        max_lineage_documents: Some(2),
        min_specific_entities: Some(2),
        textual_cues: true,
        ..CandidatePolicy::default()
    }
}

fn precise() -> CandidatePolicy {
    CandidatePolicy {
        resolve_relative_dates: true,
        periodic_status_entries: Some(3),
        dedupe_counterparts: true,
        ..quality()
    }
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

    async fn note(
        &self,
        source: &RecordId,
        heading: &[&str],
        content: &str,
        embedding: Vec<f32>,
        entities: &[&str],
    ) -> RecordId {
        let mut note = Note::new(content)
            .with_embedding(embedding)
            .with_source(source.clone());
        note.chunk_heading_path = heading.iter().map(|part| (*part).to_owned()).collect();
        let id = self.repo.create_note(note).await.unwrap().id.unwrap();
        if !entities.is_empty() {
            let mut linked = Vec::new();
            for name in entities {
                let mut entity = Entity::new(*name, EntityType::Concept);
                entity.metadata = serde_json::json!({});
                linked.push(entity);
            }
            self.repo
                .upsert_entities_and_link_note(&id, linked)
                .await
                .unwrap();
        }
        id
    }

    async fn scan(&self, policy: &CandidatePolicy, anchor: &RecordId) -> CandidateScanReport {
        GardenerAgent::new(self.repo.clone())
            .scan_candidates(
                policy,
                CandidateScanRequest {
                    scope: CandidateScope::Notes(vec![anchor.clone()]),
                    counterpart_sources: None,
                    after: None,
                    max_candidates: 20,
                    max_endpoints: 5,
                    deadline: Duration::from_secs(30),
                    min_similarity: None,
                    max_per_endpoint: None,
                    dry_run: true,
                    requested_by: "mcp:fixture-gardener".into(),
                },
            )
            .await
            .unwrap()
    }
}

fn involves(report: &CandidateScanReport, note: &RecordId) -> bool {
    let note = record_id_to_string(note);
    report.candidates.iter().any(|candidate| {
        record_id_to_string(&candidate.from) == note || record_id_to_string(&candidate.to) == note
    })
}

/// Two days of a recurring heartbeat log. Each day is one dated document
/// whose entries share one clock-stamped heading template. The next
/// morning's entry says "since 22:37 yesterday": a monitoring window, not a
/// pointer at the evening entry it is paired with (stamped 19:07).
async fn heartbeat_days(corpus: &Corpus) -> (RecordId, RecordId) {
    let evening = corpus.source("ops/2026-06-27.md").await;
    let morning = corpus.source("ops/2026-06-28.md").await;
    let anchor = corpus
        .note(
            &evening,
            &["2026-06-27", "19:07 MDT — Heartbeat recheck"],
            "Wake on the plover host. Storefront and dispatcher probes return 200. No drift.",
            vector(1.0, 1),
            &[],
        )
        .await;
    for (index, time) in ["20:07", "21:07", "22:37"].into_iter().enumerate() {
        corpus
            .note(
                &evening,
                &["2026-06-27", &format!("{time} MDT — Heartbeat recheck")],
                &format!("Wake {index}: same posture, standing down."),
                vector(0.1, 10 + index),
                &[],
            )
            .await;
    }
    let follow = corpus
        .note(
            &morning,
            &["2026-06-28", "07:07 MDT — Heartbeat recheck"],
            "Morning wake. Repository head unchanged since 22:37 yesterday. Probes 200.",
            vector(0.9, 2),
            &[],
        )
        .await;
    (anchor, follow)
}

#[tokio::test]
async fn a_heartbeat_window_relative_date_no_longer_nominates_a_pair() {
    let corpus = Corpus::new().await;
    let (anchor, follow) = heartbeat_days(&corpus).await;

    // Legacy quality policy: the bare "yesterday" carries the pair.
    let before = corpus.scan(&quality(), &anchor).await;
    assert_eq!(before.candidates.len(), 1);
    assert!(involves(&before, &follow));
    assert_eq!(
        before.candidates[0].textual_cue.map(|cue| cue.kind),
        Some(TextualCueKind::RelativeDate)
    );

    // The cue names 22:37, not the counterpart's 19:07, and shares no topic
    // span with it: it does not resolve, and no entity backs it.
    let resolving = CandidatePolicy {
        resolve_relative_dates: true,
        ..quality()
    };
    let report = corpus.scan(&resolving, &anchor).await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.rejections.unresolved_relative_date, 1);
    assert_eq!(report.rejections.no_second_signal, 0);

    // Structurally, the anchor is a recurring timestamped status entry
    // (four entries of one heading template in its document).
    let periodic = CandidatePolicy {
        periodic_status_entries: Some(3),
        ..quality()
    };
    let report = corpus.scan(&periodic, &anchor).await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.rejections.periodic_status, 1);
    // A template recurring fewer times than required is not periodic.
    let rare = CandidatePolicy {
        periodic_status_entries: Some(5),
        ..quality()
    };
    assert_eq!(corpus.scan(&rare, &anchor).await.candidates.len(), 1);
}

#[tokio::test]
async fn a_periodic_entry_with_an_explicit_note_reference_still_pairs() {
    let corpus = Corpus::new().await;
    let (anchor, _) = heartbeat_days(&corpus).await;
    let incident = corpus.source("incidents/plover-drift.md").await;
    let referenced = corpus
        .note(
            &incident,
            &["Plover drift"],
            "Root cause of the drift. See [[2026-06-27]] for the heartbeat that caught it.",
            vector(0.88, 3),
            &[],
        )
        .await;
    let report = corpus.scan(&precise(), &anchor).await;
    assert!(involves(&report, &referenced), "{:?}", report.rejections);
    let candidate = report
        .candidates
        .iter()
        .find(|candidate| candidate.cross_reference.is_some())
        .expect("the wikilinked pair");
    assert!(candidate
        .signals
        .contains(&CandidateSignal::ExplicitCrossReference));
    assert_eq!(report.rejections.periodic_status, 1);
}

#[tokio::test]
async fn a_relative_date_that_resolves_to_its_counterpart_is_kept() {
    let corpus = Corpus::new().await;
    let monday = corpus.source("lane/2026-04-27.md").await;
    let tuesday = corpus.source("lane/2026-04-28.md").await;
    let first = corpus
        .note(
            &monday,
            &["2026-04-27", "Maintenance lane — 08:20 MDT"],
            "Gateway running. The service PATH warning is shown and repair is recommended.",
            vector(1.0, 1),
            &[],
        )
        .await;
    let follow = corpus
        .note(
            &tuesday,
            &["2026-04-28", "Maintenance lane — 08:20 MDT"],
            "Gateway running. Yesterday's service PATH warning is not present today.",
            vector(0.88, 2),
            &[],
        )
        .await;
    for policy in [quality(), precise()] {
        let report = corpus.scan(&policy, &first).await;
        assert_eq!(report.candidates.len(), 1, "{:?}", report.rejections);
        assert!(involves(&report, &follow));
        assert_eq!(
            report.candidates[0].textual_cue.map(|cue| cue.kind),
            Some(TextualCueKind::RelativeDate)
        );
    }
}

#[tokio::test]
async fn an_unresolved_relative_date_backed_by_a_specific_entity_is_kept() {
    let corpus = Corpus::new().await;
    let monday = corpus.source("notes/2026-05-04.md").await;
    let tuesday = corpus.source("notes/2026-05-05.md").await;
    let rare = ["Sanderling ledger"];
    let first = corpus
        .note(
            &monday,
            &["2026-05-04"],
            "Opened the quarter close.",
            vector(1.0, 1),
            &rare,
        )
        .await;
    corpus
        .note(
            &tuesday,
            &["2026-05-05"],
            "Nothing changed since yesterday.",
            vector(0.87, 2),
            &rare,
        )
        .await;
    // A third mention keeps the name specific corpus-wide.
    let elsewhere = corpus.source("finance/plan.md").await;
    corpus
        .note(&elsewhere, &[], "Plan.", vector(0.1, 3), &rare)
        .await;
    let report = corpus.scan(&precise(), &first).await;
    assert_eq!(report.candidates.len(), 1, "{:?}", report.rejections);
    let candidate = &report.candidates[0];
    assert_eq!(
        candidate.textual_cue.map(|cue| cue.kind),
        Some(TextualCueKind::RelativeDate)
    );
    assert_eq!(candidate.shared_entities.len(), 1);
    assert_eq!(report.rejections.unresolved_relative_date, 0);
}

#[tokio::test]
async fn identical_counterparts_of_one_endpoint_are_nominated_once() {
    let corpus = Corpus::new().await;
    let monday = corpus.source("log/2026-04-27.md").await;
    let tuesday = corpus.source("log/2026-04-28.md").await;
    let first = corpus
        .note(
            &monday,
            &["2026-04-27"],
            "Gateway status: the service PATH warning is shown.",
            vector(1.0, 1),
            &[],
        )
        .await;
    // One document carrying the same paragraph twice (two chunks).
    let copy = "Gateway status: yesterday's service PATH warning is gone.";
    let a = corpus
        .note(&tuesday, &["2026-04-28"], copy, vector(0.9, 2), &[])
        .await;
    let b = corpus
        .note(
            &tuesday,
            &["2026-04-28", "Repeat"],
            copy,
            vector(0.89, 3),
            &[],
        )
        .await;

    let before = corpus.scan(&quality(), &first).await;
    assert_eq!(before.candidates.len(), 2);
    let after = corpus.scan(&precise(), &first).await;
    assert_eq!(after.candidates.len(), 1);
    assert_eq!(after.rejections.duplicate_counterpart, 1);
    // The better-ranked copy (higher similarity) is the one kept.
    assert!(involves(&after, &a));
    assert!(!involves(&after, &b));
}

#[tokio::test]
async fn legacy_and_earlier_quality_policies_ignore_the_new_options() {
    let corpus = Corpus::new().await;
    let (anchor, _) = heartbeat_days(&corpus).await;
    let legacy = corpus.scan(&CandidatePolicy::default(), &anchor).await;
    let quality = corpus.scan(&quality(), &anchor).await;
    for report in [&legacy, &quality] {
        let rejected = report.rejections;
        assert_eq!(
            (
                rejected.unresolved_relative_date,
                rejected.periodic_status,
                rejected.duplicate_counterpart
            ),
            (0, 0, 0)
        );
        assert!(!report.policy.resolve_relative_dates);
        assert_eq!(report.policy.periodic_status_entries, None);
        assert!(!report.policy.dedupe_counterparts);
    }
    // Legacy has no textual cue: the pair has no second signal at all.
    assert!(legacy.candidates.is_empty());
    assert_eq!(legacy.rejections.no_second_signal, 1);
    assert_eq!(quality.candidates.len(), 1);
}
