//! Fictional fixtures for the opt-in candidate quality gates (#230).
use super::candidates::*;
use super::GardenerAgent;
use graphrag_core::{record_id_to_string, Entity, EntityType, Note, Source, SourceType};
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

/// The policy the #230 validation measured.
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

struct Corpus {
    repo: Repository,
}

impl Corpus {
    async fn new() -> Self {
        Self {
            repo: Repository::new(init_memory().await.unwrap()),
        }
    }

    /// A source for `file` under the fictional vault.
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

    /// One uploaded part of an origin document: its own source, declaring
    /// the document it was split from, like a multi-part memory upload.
    async fn part(&self, document: &str, part: usize) -> RecordId {
        let mut source = Source::from_file(
            format!("/fixtures/uploads/{document}.part-{part}"),
            SourceType::Markdown,
        )
        .unwrap();
        source.metadata = serde_json::json!({
            "remote_upload": {"source": {"uri": format!("vault://ops/{document}")}}
        });
        self.repo.create_source(source).await.unwrap().id.unwrap()
    }

    async fn note(
        &self,
        source: &RecordId,
        content: &str,
        embedding: Vec<f32>,
        entities: &[&str],
    ) -> RecordId {
        let note = Note::new(content)
            .with_embedding(embedding)
            .with_source(source.clone());
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

    async fn scan(&self, policy: &CandidatePolicy, anchors: Vec<RecordId>) -> CandidateScanReport {
        GardenerAgent::new(self.repo.clone())
            .scan_candidates(
                policy,
                CandidateScanRequest {
                    scope: CandidateScope::Notes(anchors),
                    counterpart_sources: None,
                    after: None,
                    max_candidates: 20,
                    max_endpoints: 25,
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

const RUN: &str = "The relay restarted at noon and the queue drained within four minutes once the cache warmed and the audit log rotated cleanly";

#[tokio::test]
async fn a_near_duplicate_pair_is_rejected_only_when_the_gate_is_on() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("2026-03-01-relay.md").await;
    let beta = corpus.source("relay-digest.md").await;
    let rare = ["Kestrel relay", "Heron queue"];
    let anchor = corpus
        .note(
            &alpha,
            &format!("Relay report. {RUN}."),
            vector(1.0, 1),
            &rare,
        )
        .await;
    // A copy of the same run under a different heading: cosine 0.90 (below
    // the ceiling) but it shares one long span of words.
    corpus
        .note(
            &beta,
            &format!("Digest copy: {RUN}; nothing else."),
            vector(0.9, 2),
            &rare,
        )
        .await;
    // A verbatim repeat above the ceiling with different framing words.
    let gamma = corpus.source("relay-mirror.md").await;
    corpus
        .note(
            &gamma,
            "Mirror of the relay summary for the weekly page.",
            vector(0.97, 3),
            &rare,
        )
        .await;

    let legacy = corpus
        .scan(&CandidatePolicy::default(), vec![anchor.clone()])
        .await;
    assert_eq!(legacy.candidates.len(), 2, "legacy keeps both");
    assert_eq!(legacy.rejections.near_duplicate, 0);

    let gated = corpus.scan(&quality(), vec![anchor]).await;
    assert!(gated.candidates.is_empty());
    assert_eq!(gated.nominations, 2);
    assert_eq!(gated.rejections.near_duplicate, 2);
}

#[tokio::test]
async fn a_lineage_template_pair_is_rejected_but_a_cross_lineage_pair_is_kept() {
    let corpus = Corpus::new().await;
    // A recurring heartbeat log: one dated document per day, each repeating
    // the same "specific" names (globally rarer than the generic cutoff).
    let template = ["Plover watchdog", "Avocet probe"];
    let mut days = Vec::new();
    for day in 1..=4 {
        let source = corpus.source(&format!("heartbeat/2026-07-0{day}.md")).await;
        days.push(
            corpus
                .note(
                    &source,
                    &format!("Heartbeat 0{day}: watchdog green, probe green, no change."),
                    vector(0.9 + 0.01 * day as f32, day + 1),
                    &template,
                )
                .await,
        );
    }
    let legacy = corpus
        .scan(&CandidatePolicy::default(), vec![days[0].clone()])
        .await;
    assert!(
        !legacy.candidates.is_empty(),
        "legacy treats the names as specific"
    );

    let gated = corpus.scan(&quality(), vec![days[0].clone()]).await;
    assert!(gated.candidates.is_empty());
    assert_eq!(gated.rejections.lineage_template, gated.nominations);

    // The same names shared by two documents of different lineages remain
    // specific: lineage is computed per document family, not per name alone.
    let design = corpus.source("design/watchdog-plan.md").await;
    let review = corpus.source("reviews/probe-review.md").await;
    let plan = corpus
        .note(
            &design,
            "Plan: retire the watchdog once the probe covers it.",
            vector(1.0, 10),
            &template,
        )
        .await;
    corpus
        .note(
            &review,
            "Review: the probe now covers the watchdog checks.",
            vector(0.88, 11),
            &template,
        )
        .await;
    let kept = corpus.scan(&quality(), vec![plan]).await;
    assert_eq!(kept.candidates.len(), 1);
    assert_eq!(kept.candidates[0].shared_entities.len(), 2);
}

#[tokio::test]
async fn parts_of_one_uploaded_document_are_one_origin_under_lineage_specificity() {
    let corpus = Corpus::new().await;
    let first = corpus.part("memory/2026-07-17.md", 1).await;
    let second = corpus.part("memory/2026-07-17.md", 2).await;
    let rare = ["Tern epic", "Godwit reconciliation"];
    let anchor = corpus
        .note(
            &first,
            "09:41 reconciliation for the tern epic.",
            vector(1.0, 1),
            &rare,
        )
        .await;
    corpus
        .note(
            &second,
            "14:21 reconciliation for the tern epic.",
            vector(0.88, 2),
            &rare,
        )
        .await;
    let legacy = corpus
        .scan(&CandidatePolicy::default(), vec![anchor.clone()])
        .await;
    assert_eq!(legacy.candidates.len(), 1, "distinct sources pair today");
    let gated = corpus.scan(&quality(), vec![anchor]).await;
    assert!(gated.candidates.is_empty());
    assert_eq!(gated.rejections.same_document, 1);
}

#[tokio::test]
async fn a_machine_dump_anchor_is_skipped_and_dump_counterparts_rejected() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("crash-a.md").await;
    let beta = corpus.source("crash-b.md").await;
    let gamma = corpus.source("crash-notes.md").await;
    let frames = |seed: usize| {
        (0..6)
            .map(|frame| {
                format!(
                    "    + {frame} worker_loop  (in libdispatch.dylib) + {}  [0x18983{seed}{frame:03}]",
                    frame * 8
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let rare = ["Lapwing dispatcher", "Snipe worker"];
    let dump = corpus.note(&alpha, &frames(1), vector(1.0, 1), &rare).await;
    corpus.note(&beta, &frames(2), vector(0.9, 2), &rare).await;
    let prose = corpus
        .note(
            &gamma,
            "Crash triage: the snipe worker hung in the lapwing dispatcher after the update.",
            vector(0.88, 3),
            &rare,
        )
        .await;

    let legacy = corpus
        .scan(&CandidatePolicy::default(), vec![dump.clone()])
        .await;
    assert_eq!(legacy.candidates.len(), 2);

    let gated = corpus
        .scan(&quality(), vec![dump.clone(), prose.clone()])
        .await;
    assert_eq!(gated.endpoints_examined, 2);
    assert_eq!(gated.machine_dump_anchors, 1);
    // The prose anchor's dump neighbours are rejected as counterparts.
    assert!(gated.candidates.is_empty());
    assert_eq!(gated.rejections.machine_dump, gated.nominations);
    assert!(gated.nominations >= 1);
}

#[tokio::test]
async fn a_textual_cue_pair_is_kept_and_ranked_with_cross_references() {
    let corpus = Corpus::new().await;
    // One shared specific entity only: below min_specific_entities = 2.
    let rare = ["Curlew gateway"];
    let monday = corpus.source("daily/2026-04-27.md").await;
    let tuesday = corpus.source("daily/2026-04-28.md").await;
    let thursday = corpus.source("daily/2026-04-30.md").await;
    let first = corpus
        .note(
            &monday,
            "Curlew gateway status: PATH warning shown.",
            vector(1.0, 1),
            &rare,
        )
        .await;
    let follow_up = corpus
        .note(
            &tuesday,
            "Curlew gateway status: yesterday's PATH warning is gone.",
            vector(0.85, 2),
            &rare,
        )
        .await;
    let later = corpus
        .note(
            &thursday,
            "Curlew gateway status: still clean.",
            vector(0.86, 3),
            &rare,
        )
        .await;

    let gated = corpus.scan(&quality(), vec![first.clone()]).await;
    assert_eq!(gated.candidates.len(), 1, "only the cue survives");
    let candidate = &gated.candidates[0];
    let expected = {
        let (a, b) = (record_id_to_string(&first), record_id_to_string(&follow_up));
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    };
    assert_eq!(pairs(&gated), vec![expected]);
    assert!(candidate.signals.contains(&CandidateSignal::TextualCue));
    assert_eq!(
        candidate.textual_cue.map(|cue| cue.kind),
        Some(TextualCueKind::RelativeDate)
    );
    // The Thursday note shares only the entity, which three documents of the
    // daily family mention: a lineage template, not evidence.
    assert_eq!(gated.rejections.lineage_template, 1);
    assert!(!pairs(&gated)
        .iter()
        .any(|(a, b)| *a == record_id_to_string(&later) || *b == record_id_to_string(&later)));
    // Without cues the follow-up is the same lineage template.
    let no_cues = CandidatePolicy {
        textual_cues: false,
        ..quality()
    };
    let report = corpus.scan(&no_cues, vec![first]).await;
    assert!(report.candidates.is_empty());
    assert_eq!(report.rejections.lineage_template, 2);
}

#[tokio::test]
async fn legacy_policies_report_identically_and_record_no_quality_features() {
    let corpus = Corpus::new().await;
    let alpha = corpus.source("2026-04-27.md").await;
    let beta = corpus.source("2026-04-28.md").await;
    let rare = ["Whimbrel ledger"];
    let anchor = corpus
        .note(
            &alpha,
            &format!("Ledger close. {RUN}."),
            vector(1.0, 1),
            &rare,
        )
        .await;
    corpus
        .note(
            &beta,
            &format!("Yesterday's ledger close: {RUN}."),
            vector(0.97, 2),
            &rare,
        )
        .await;
    let legacy = corpus
        .scan(&CandidatePolicy::default(), vec![anchor.clone()])
        .await;
    assert_eq!(legacy.candidates.len(), 1);
    assert_eq!(
        legacy.candidates[0].signals,
        vec![CandidateSignal::SharedSpecificEntity]
    );
    assert_eq!(legacy.candidates[0].textual_cue, None);
    assert_eq!(legacy.machine_dump_anchors, 0);
    let rejected = legacy.rejections;
    assert_eq!(
        (
            rejected.same_document,
            rejected.machine_dump,
            rejected.near_duplicate,
            rejected.lineage_template,
            rejected.insufficient_specific_entities
        ),
        (0, 0, 0, 0, 0)
    );

    // Persisted features carry no quality keys under a legacy policy.
    let report = GardenerAgent::new(corpus.repo.clone())
        .scan_candidates(
            &CandidatePolicy::default(),
            CandidateScanRequest {
                scope: CandidateScope::Notes(vec![anchor]),
                counterpart_sources: None,
                after: None,
                max_candidates: 5,
                max_endpoints: 5,
                deadline: Duration::from_secs(30),
                min_similarity: None,
                max_per_endpoint: None,
                dry_run: false,
                requested_by: "mcp:fixture-gardener".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(report.candidates.len(), 1);
    let proposals = corpus.repo.list_edge_proposals(None, 10).await.unwrap();
    let reason: serde_json::Value = serde_json::from_str(&proposals[0].reason).unwrap();
    let mut keys = reason
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    assert_eq!(
        keys,
        [
            "contract",
            "cross_reference",
            "embedding_model",
            "from",
            "generator",
            "policy",
            "requested_by",
            "shared_entities",
            "signals",
            "similarity",
            "summary",
            "to"
        ]
    );
    let mut policy = reason["policy"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    policy.sort();
    assert_eq!(
        policy,
        [
            "max_entity_note_frequency",
            "max_per_endpoint",
            "neighbors_per_endpoint",
            "similarity_floor"
        ]
    );
}

#[tokio::test]
async fn quality_features_are_persisted_when_the_gates_are_on() {
    let corpus = Corpus::new().await;
    let monday = corpus.source("daily/2026-04-27.md").await;
    let tuesday = corpus.source("daily/2026-04-28.md").await;
    let rare = ["Dunlin cutover"];
    let first = corpus
        .note(
            &monday,
            "Dunlin cutover warning recorded.",
            vector(1.0, 1),
            &rare,
        )
        .await;
    let follow_up = corpus
        .note(
            &tuesday,
            "Yesterday's dunlin cutover warning cleared.",
            vector(0.86, 2),
            &rare,
        )
        .await;
    GardenerAgent::new(corpus.repo.clone())
        .scan_candidates(
            &quality(),
            CandidateScanRequest {
                scope: CandidateScope::Notes(vec![first.clone()]),
                counterpart_sources: None,
                after: None,
                max_candidates: 5,
                max_endpoints: 5,
                deadline: Duration::from_secs(30),
                min_similarity: None,
                max_per_endpoint: None,
                dry_run: false,
                requested_by: "mcp:fixture-gardener".into(),
            },
        )
        .await
        .unwrap();
    let proposals = corpus.repo.list_edge_proposals(None, 10).await.unwrap();
    assert_eq!(proposals.len(), 1);
    // The direction names the note carrying the cue: the follow-up.
    let carrier = if record_id_to_string(&follow_up) < record_id_to_string(&first) {
        "from_mentions_to"
    } else {
        "to_mentions_from"
    };
    let reason: serde_json::Value = serde_json::from_str(&proposals[0].reason).unwrap();
    assert_eq!(reason["textual_cue"]["kind"], "relative_date");
    assert_eq!(reason["textual_cue"]["direction"], carrier);
    assert_eq!(reason["policy"]["quality"]["max_lineage_documents"], 2);
    assert_eq!(reason["policy"]["quality"]["textual_cues"], true);
    assert!(reason["signals"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("textual_cue")));
}

#[tokio::test]
async fn batched_similarity_matches_per_anchor_nomination_exactly() {
    let corpus = Corpus::new().await;
    let mut anchors = Vec::new();
    let mut sources = Vec::new();
    for index in 0..6 {
        sources.push(corpus.source(&format!("batch-{index}.md")).await);
    }
    // Every note shares a direction (axis 0) at a distinct strength, so each
    // anchor has several neighbours above the floor, some tied.
    for index in 0..18 {
        let cosine = [1.0, 0.97, 0.95, 0.95, 0.93, 0.9][index % 6];
        let names = [
            format!("Marker {}", index % 5),
            format!("Tag {}", index % 3),
        ];
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        let id = corpus
            .note(
                &sources[index % 6],
                &format!("Batch note {index}"),
                vector(cosine, index + 1),
                &names,
            )
            .await;
        if index % 3 == 0 {
            anchors.push(id);
        }
    }
    let one = corpus
        .scan(&CandidatePolicy::default(), anchors.clone())
        .await;
    for group in [2, 4, 64] {
        let batched = CandidatePolicy {
            similarity_batch_anchors: group,
            ..CandidatePolicy::default()
        };
        let report = corpus.scan(&batched, anchors.clone()).await;
        assert_eq!(report.nominations, one.nominations, "group {group}");
        assert_eq!(report.rejections, one.rejections, "group {group}");
        assert_eq!(pairs(&report), pairs(&one), "group {group}");
        for (left, right) in report.candidates.iter().zip(&one.candidates) {
            assert_eq!(left.similarity.to_bits(), right.similarity.to_bits());
        }
    }
    assert!(one.nominations > 0);
}
