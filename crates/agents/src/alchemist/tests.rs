//! Fictional fixtures only: an orchard-maintenance corpus with reviewed edges.

use super::gather::EvidenceChunk;
use super::*;
use crate::{DeterministicEmbedder, InferenceCapabilities};
use graphrag_core::{EdgeType, Note, Source};
use graphrag_db::compatibility::EmbeddingIdentity;
use graphrag_db::{init_memory, DbConnection};
use serde_json::{json, Value};
use std::sync::Arc;

const LOG: &str = "Irrigation log: the north orchard drip lines were flushed on 3 March and line pressure held at 2.1 bar afterwards.";
const SUMMARY: &str =
    "Weekly summary: drip lines in the north orchard were flushed; pressure stayed near 2.1 bar.";
const PUMP: &str = "Pump inspection: an independent technician measured a steady 2.1 bar at the north orchard manifold on 5 March.";
const NOTEBOOK: &str =
    "Grower notebook: after the flush the north orchard pressure dropped to 1.4 bar by evening.";
const VALVE: &str = "Valve catalogue: the orchard manifold uses brass gate valves rated for 6 bar.";

struct Corpus {
    db: DbConnection,
    repo: Repository,
    log: String,
    summary: String,
    pump: String,
    notebook: String,
    valve: String,
}

async fn note(repo: &Repository, title: &str, content: &str) -> String {
    let note = repo
        .create_note(Note::new(content).with_title(title))
        .await
        .unwrap();
    record_id_to_string(note.id.as_ref().unwrap())
}

async fn edge(repo: &Repository, from: &str, to: &str, edge_type: EdgeType) {
    let parse = |id: &str| graphrag_db::parse_record_id(id, Some("note")).unwrap();
    repo.create_edge(&parse(from), &parse(to), edge_type, None)
        .await
        .unwrap();
}

/// summary -derived_from-> log, pump -supports-> log,
/// notebook -contradicts-> log, valve -related_to- pump.
async fn corpus() -> Corpus {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    repo.record_embedding_metadata(
        &EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
        None,
    )
    .await
    .unwrap();
    let log = note(&repo, "Irrigation log", LOG).await;
    let summary = note(&repo, "Weekly summary", SUMMARY).await;
    let pump = note(&repo, "Pump inspection", PUMP).await;
    let notebook = note(&repo, "Grower notebook", NOTEBOOK).await;
    let valve = note(&repo, "Valve catalogue", VALVE).await;
    edge(&repo, &summary, &log, EdgeType::DerivedFrom).await;
    edge(&repo, &pump, &log, EdgeType::Supports).await;
    edge(&repo, &notebook, &log, EdgeType::Contradicts).await;
    edge(&repo, &valve, &pump, EdgeType::RelatedTo).await;
    Corpus {
        db,
        repo,
        log,
        summary,
        pump,
        notebook,
        valve,
    }
}

fn search(repo: &Repository) -> SearchAgent {
    SearchAgent::new(repo.clone(), Arc::new(DeterministicEmbedder::default()))
}

fn seed_plan(id: &str) -> SynthesisPlan {
    SynthesisPlan {
        input: SynthesisInput::Seed { id: id.into() },
        budgets: SynthesisBudgets::default(),
        policy: CitationPolicy::Drop,
    }
}

fn cite(chunk: &str, quote: &str) -> Value {
    json!({"chunk": chunk, "quote": quote})
}

fn claim(text: &str, citations: Vec<Value>) -> Value {
    json!({"text": text, "citations": citations})
}

fn draft(claims: Vec<Value>, conflicts: Vec<Value>) -> String {
    json!({"claims": claims, "conflicts": conflicts, "open_questions": ["Was the gauge recalibrated?"]})
        .to_string()
}

async fn run(
    corpus: &Corpus,
    generator: &dyn SynthesisGenerator,
    plan: &SynthesisPlan,
) -> Result<SynthesisOutcome, AlchemistError> {
    synthesize(
        &corpus.repo,
        &search(&corpus.repo),
        generator,
        &AugmentOptions::default(),
        plan,
    )
    .await
}

/// Every table's complete contents, for the no-write guarantee.
async fn snapshot(db: &DbConnection) -> BTreeMap<String, String> {
    let info = db
        .query("INFO FOR DB")
        .await
        .unwrap()
        .take::<Option<Value>>(0)
        .unwrap()
        .unwrap();
    let mut tables = BTreeMap::new();
    for table in info["tables"].as_object().unwrap().keys() {
        let rows: Vec<Value> = db
            .query(format!("SELECT * FROM {table} ORDER BY id"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        tables.insert(table.clone(), serde_json::to_string(&rows).unwrap());
    }
    assert!(tables.contains_key("note") && tables.contains_key("supports"));
    tables
}

#[tokio::test]
async fn verifier_keeps_only_verbatim_spans_of_cited_chunks_at_pinned_revisions() {
    let corpus = corpus().await;
    let pinned = corpus.repo.inspect_record(&corpus.log, 0).await.unwrap();
    let response = draft(
        vec![
            claim(
                "The drip lines were flushed and pressure held.",
                vec![cite(
                    &corpus.log,
                    "flushed on 3 March and line pressure held at 2.1 bar",
                )],
            ),
            // Fabricated quote: not in the cited chunk.
            claim(
                "Pressure rose after the flush.",
                vec![cite(
                    &corpus.log,
                    "pressure rose to 3.5 bar after the flush",
                )],
            ),
            // Wrong chunk: the span exists only in the pump inspection.
            claim(
                "A technician measured steady pressure.",
                vec![cite(
                    &corpus.notebook,
                    "an independent technician measured a steady 2.1 bar",
                )],
            ),
            claim(
                "An unknown chunk agrees.",
                vec![cite("C99", "line pressure held at 2.1 bar")],
            ),
            // Stale revision: the draft names a revision that was never pinned.
            claim(
                "Pressure held.",
                vec![
                    json!({"chunk": corpus.log, "revision": "0".repeat(64), "quote": "line pressure held at 2.1 bar"}),
                ],
            ),
            claim("An uncited assertion about valves.", vec![]),
            claim("Too short to check.", vec![cite(&corpus.log, "2.1 bar")]),
        ],
        vec![],
    );
    let generator = FixtureSynthesisGenerator::fixed(response);
    let outcome = run(&corpus, &generator, &seed_plan(&corpus.log))
        .await
        .unwrap();
    let report = &outcome.verification;
    assert_eq!((report.claims_proposed, report.claims_kept), (7, 1));
    assert_eq!((report.claims_dropped, report.uncited_claims), (6, 1));
    assert_eq!(report.citations_rejected, 5);
    assert_eq!(
        report.rejections,
        BTreeMap::from([
            ("quote_not_found".into(), 2),
            ("quote_too_short".into(), 1),
            ("stale_revision".into(), 1),
            ("unknown_chunk".into(), 1),
        ])
    );
    let claims = &outcome.synthesis.claims;
    assert_eq!(claims.len(), 1);
    let citation = &claims[0].citations[0];
    assert_eq!(citation.chunk_id, corpus.log);
    assert_eq!(citation.revision, pinned.revision);
    assert_eq!(&pinned.content[citation.start..citation.end], citation.span);
    assert!(outcome
        .synthesis
        .gaps
        .iter()
        .any(|gap| gap.kind == GapKind::DroppedClaims));
    // The model was shown the pinned excerpt under its label.
    let shown = &generator.requests()[0].prompt;
    assert!(shown.contains(&format!("[C1] {}", corpus.log)) && shown.contains(LOG));
}

/// Edits the cited note while "generating", like a concurrent writer.
struct EditingGenerator {
    repo: Repository,
    id: String,
    response: String,
}

#[async_trait]
impl SynthesisGenerator for EditingGenerator {
    async fn generate(&self, _: &GenerationRequest) -> crate::Result<String> {
        let mut note = self.repo.get_note(&self.id).await?.unwrap();
        note.content = format!("{} Corrected later.", note.content);
        self.repo.update_note(&self.id, note).await?;
        Ok(self.response.clone())
    }
    fn capabilities(&self) -> InferenceCapabilities {
        FixtureSynthesisGenerator::fixed("").capabilities()
    }
}

#[tokio::test]
async fn a_chunk_edited_after_pinning_cannot_be_cited() {
    let corpus = corpus().await;
    let generator = EditingGenerator {
        repo: corpus.repo.clone(),
        id: corpus.log.clone(),
        response: draft(
            vec![
                claim(
                    "Pressure held after the flush.",
                    vec![cite("C1", "line pressure held at 2.1 bar")],
                ),
                claim(
                    "Pressure dropped in the evening.",
                    vec![cite(
                        &corpus.notebook,
                        "pressure dropped to 1.4 bar by evening",
                    )],
                ),
            ],
            vec![],
        ),
    };
    let outcome = run(&corpus, &generator, &seed_plan(&corpus.log))
        .await
        .unwrap();
    assert_eq!(outcome.verification.rejections["stale_revision"], 1);
    assert_eq!(outcome.verification.chunks_rechecked, 2);
    assert_eq!(outcome.synthesis.claims.len(), 1);
    assert_eq!(
        outcome.synthesis.claims[0].citations[0].chunk_id,
        corpus.notebook
    );

    let strict = SynthesisPlan {
        policy: CitationPolicy::Strict,
        ..seed_plan(&corpus.notebook)
    };
    let generator = EditingGenerator {
        id: corpus.notebook.clone(),
        ..generator
    };
    // Seeded at the notebook, both citations (C1 and the ID) now name the
    // edited notebook.
    match run(&corpus, &generator, &strict).await {
        Err(AlchemistError::Unverifiable { report, .. }) => {
            assert_eq!(report.rejections["stale_revision"], 2)
        }
        other => panic!("strict policy must fail closed: {other:?}"),
    }
}

#[tokio::test]
async fn strict_policy_errors_on_any_rejection_and_malformed_output_always_errors() {
    let corpus = corpus().await;
    let strict = SynthesisPlan {
        policy: CitationPolicy::Strict,
        ..seed_plan(&corpus.log)
    };
    let valid = claim(
        "Pressure held after the flush.",
        vec![cite(&corpus.log, "line pressure held at 2.1 bar")],
    );
    let ok = FixtureSynthesisGenerator::fixed(draft(vec![valid.clone()], vec![]));
    assert_eq!(
        run(&corpus, &ok, &strict)
            .await
            .unwrap()
            .synthesis
            .claims
            .len(),
        1
    );
    let uncited = FixtureSynthesisGenerator::fixed(draft(
        vec![valid, claim("Valves failed.", vec![])],
        vec![],
    ));
    assert!(matches!(
        run(&corpus, &uncited, &strict).await,
        Err(AlchemistError::Unverifiable { .. })
    ));
    for malformed in ["not json", "{\"claims\": 3}"] {
        let generator = FixtureSynthesisGenerator::fixed(malformed);
        assert!(matches!(
            run(&corpus, &generator, &seed_plan(&corpus.log)).await,
            Err(AlchemistError::Unverifiable { .. })
        ));
    }
    let fenced = FixtureSynthesisGenerator::fixed(format!(
        "```json\n{}\n```",
        draft(
            vec![claim(
                "Pressure held after the flush.",
                vec![cite("[C1]", "line pressure held at 2.1 bar")]
            )],
            vec![]
        )
    ));
    assert_eq!(
        run(&corpus, &fenced, &strict)
            .await
            .unwrap()
            .synthesis
            .claims
            .len(),
        1
    );
}

#[tokio::test]
async fn lineage_counts_a_summary_and_its_origin_once() {
    let corpus = corpus().await;
    let generator = FixtureSynthesisGenerator::fixed(draft(
        vec![
            claim(
                "The lines were flushed and pressure held near 2.1 bar.",
                vec![
                    cite(&corpus.log, "line pressure held at 2.1 bar afterwards"),
                    cite(&corpus.summary, "pressure stayed near 2.1 bar"),
                ],
            ),
            claim(
                "Two sources report 2.1 bar.",
                vec![
                    cite(&corpus.log, "line pressure held at 2.1 bar afterwards"),
                    cite(
                        &corpus.pump,
                        "measured a steady 2.1 bar at the north orchard manifold",
                    ),
                ],
            ),
        ],
        vec![],
    ));
    // Seed the summary: its path to the pump crosses the derived_from lineage.
    let outcome = run(&corpus, &generator, &seed_plan(&corpus.summary))
        .await
        .unwrap();
    let synthesis = &outcome.synthesis;
    let copied = &synthesis.claims[0].strength;
    assert_eq!((copied.independent_origins, copied.cited_chunks), (1, 2));
    assert!(copied.wording.contains("share one origin") && !copied.wording.contains('%'));
    let independent = &synthesis.claims[1].strength;
    assert_eq!(independent.independent_origins, 2);
    assert!(independent
        .wording
        .contains("2 lineage-independent origins"));

    let group = synthesis
        .lineage
        .iter()
        .find(|group| group.chunks.contains(&corpus.log))
        .unwrap();
    assert_eq!(
        group.chunks,
        vec![corpus.summary.clone(), corpus.log.clone()]
    );
    assert_eq!(group.reasons, vec![LineageReason::DerivedFrom]);
    let lineage_edge = synthesis
        .relations
        .iter()
        .find(|relation| relation.edge_type == "derived_from")
        .unwrap();
    assert!(lineage_edge.lineage_collapsed);
    assert_eq!(lineage_edge.evidence, EVIDENCE_UNASSESSED);

    // summary -derived_from-> log <-supports- pump: one step after collapse.
    let chain = synthesis
        .relation_chains
        .iter()
        .find(|chain| chain.target_chunk == corpus.pump)
        .unwrap();
    assert_eq!(chain.collapsed_lineage_steps, 1);
    assert_eq!(chain.steps.len(), 1);
    assert_eq!(chain.steps[0].edge_type, "supports");
    assert_eq!(chain.steps[0].to_origin, group.origin);
    // The pure lineage path to the origin is not presented as a relation.
    assert!(!synthesis
        .relation_chains
        .iter()
        .any(|chain| chain.target_chunk == corpus.log));
}

fn chunk(id: &str, content: &str) -> EvidenceChunk {
    EvidenceChunk {
        label: String::new(),
        id: id.into(),
        revision: "r".into(),
        content: content.into(),
        text: content.into(),
        title: None,
        created_at: None,
        source_id: None,
        source_uri: None,
        conversation_id: None,
        path: Vec::new(),
        current: true,
    }
}

#[test]
fn same_source_conversation_and_copies_share_an_origin() {
    let mut a = chunk("note:a", "Page one of the field journal.");
    let mut b = chunk("note:b", "Page two of the field journal.");
    a.source_id = Some("source:journal".into());
    b.source_id = Some("source:journal".into());
    let mut c = chunk("message:c", "A chat message about pruning.");
    let mut d = chunk("note:d", "A note derived from that chat.");
    c.conversation_id = Some("conversation:one".into());
    d.conversation_id = Some("conversation:one".into());
    let e = chunk("note:e", "Copied   text about grafting.");
    let f = chunk("note:f", "copied text about GRAFTING.");
    let g = chunk("note:g", "An unrelated independent observation.");
    let lineage = lineage::origin_groups(&[a, b, c, d, e, f, g], &[]);
    let reasons: Vec<_> = lineage
        .groups
        .iter()
        .map(|group| (group.chunks.len(), group.reasons.clone()))
        .collect();
    assert_eq!(
        reasons,
        vec![
            (2, vec![LineageReason::SameSource]),
            (2, vec![LineageReason::SameConversation]),
            (2, vec![LineageReason::IdenticalContent]),
            (1, vec![]),
        ]
    );
}

#[tokio::test]
async fn contradictions_are_surfaced_with_both_sides_and_never_resolved() {
    let corpus = corpus().await;
    let generator = FixtureSynthesisGenerator::fixed(draft(
        vec![
            claim(
                "Pressure held at 2.1 bar.",
                vec![cite(
                    &corpus.log,
                    "line pressure held at 2.1 bar afterwards",
                )],
            ),
            claim(
                "Pressure dropped to 1.4 bar.",
                vec![cite(
                    &corpus.notebook,
                    "pressure dropped to 1.4 bar by evening",
                )],
            ),
            claim(
                "Unverifiable side.",
                vec![cite(&corpus.log, "it fell to zero bar overnight")],
            ),
        ],
        vec![
            json!({"topic": "Pressure after the flush", "claims": [0, 1]}),
            json!({"topic": "Only one surviving side", "claims": [0, 2]}),
        ],
    ));
    let outcome = run(&corpus, &generator, &seed_plan(&corpus.log))
        .await
        .unwrap();
    let conflicts = &outcome.synthesis.conflicts;
    let edge = conflicts
        .iter()
        .find(|conflict| conflict.kind == ConflictKind::AcceptedContradictsEdge)
        .unwrap();
    assert_eq!(
        edge.chunks,
        vec![corpus.notebook.clone(), corpus.log.clone()]
    );
    assert_eq!(edge.claims, vec![0, 1]);
    let claimed = conflicts
        .iter()
        .find(|conflict| conflict.kind == ConflictKind::ConflictingClaims)
        .unwrap();
    assert_eq!(claimed.topic.as_deref(), Some("Pressure after the flush"));
    assert_eq!(claimed.claims, vec![0, 1]);
    assert_eq!(outcome.verification.conflicts_dropped, 1);
    assert!(outcome
        .synthesis
        .claims
        .iter()
        .all(|claim| claim.strength.contested && claim.strength.wording.contains("Contested")));
    // Surfaced, not resolved: no verdict field exists.
    let keys: Vec<String> = serde_json::to_value(claimed)
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(keys, ["chunks", "claims", "edge_id", "kind", "topic"]);
}

#[tokio::test]
async fn seed_budgets_bound_hops_chunks_and_context_tokens() {
    let corpus = corpus().await;
    let generator = FixtureSynthesisGenerator::fixed(draft(vec![], vec![]));
    let ids = |outcome: &SynthesisOutcome| -> Vec<String> {
        outcome
            .synthesis
            .evidence
            .iter()
            .map(|evidence| evidence.chunk_id.clone())
            .collect()
    };

    let two_hops = run(&corpus, &generator, &seed_plan(&corpus.log))
        .await
        .unwrap();
    assert_eq!(
        ids(&two_hops),
        vec![
            corpus.log.clone(),
            corpus.notebook.clone(),
            corpus.summary.clone(),
            corpus.pump.clone(),
            corpus.valve.clone()
        ]
    );
    let mut plan = seed_plan(&corpus.log);
    plan.budgets.max_hops = 1;
    assert!(!ids(&run(&corpus, &generator, &plan).await.unwrap()).contains(&corpus.valve));

    plan.budgets.max_hops = 2;
    plan.budgets.max_chunks = 2;
    let capped = run(&corpus, &generator, &plan).await.unwrap();
    assert_eq!(capped.synthesis.evidence.len(), 2);
    assert!(capped.usage.neighbourhood_truncated);
    assert!(capped
        .synthesis
        .gaps
        .iter()
        .any(|gap| gap.kind == GapKind::NeighbourhoodTruncated));

    plan.budgets.max_chunks = 5;
    plan.budgets.max_context_tokens = 150;
    let tight = run(&corpus, &generator, &plan).await.unwrap();
    assert!(tight.usage.context_tokens <= 150);
    let shown: usize = generator
        .requests()
        .last()
        .unwrap()
        .prompt
        .split("<<<\n")
        .skip(1)
        .map(|block| block.split("\n>>>").next().unwrap().len())
        .sum();
    assert_eq!(shown, tight.usage.context_tokens);

    for invalid in [3, 0] {
        plan.budgets.max_hops = invalid;
        assert!(matches!(
            run(&corpus, &generator, &plan).await,
            Err(AlchemistError::Invalid(_))
        ));
    }
    let bad_seed = seed_plan("message:not-a-note");
    assert!(matches!(
        run(&corpus, &generator, &bad_seed).await,
        Err(AlchemistError::Invalid(_))
    ));
    assert!(matches!(
        run(&corpus, &generator, &seed_plan("note:missing")).await,
        Err(AlchemistError::NotFound(_))
    ));
}

#[tokio::test]
async fn synthesis_never_writes_even_when_the_model_proposes_edges() {
    let corpus = corpus().await;
    let mut response: Value = serde_json::from_str(&draft(
        vec![claim(
            "Pressure held after the flush.",
            vec![cite(&corpus.log, "line pressure held at 2.1 bar")],
        )],
        vec![],
    ))
    .unwrap();
    response["proposals"] = json!([{"from": corpus.pump, "to": corpus.valve, "type": "supports"}]);
    response["edges"] = json!([{"from": corpus.log, "to": corpus.valve}]);
    response["notes"] = json!([{"content": "A synthesized note"}]);
    let generator = FixtureSynthesisGenerator::fixed(response.to_string());
    let before = snapshot(&corpus.db).await;
    let seeded = run(&corpus, &generator, &seed_plan(&corpus.log))
        .await
        .unwrap();
    assert_eq!(seeded.synthesis.claims.len(), 1);
    let query = SynthesisPlan {
        input: SynthesisInput::Query {
            query: "north orchard drip lines pressure".into(),
            scope: SearchScope::Notes,
            graph: GraphMode::On,
        },
        ..seed_plan(&corpus.log)
    };
    let queried = run(&corpus, &generator, &query).await.unwrap();
    assert!(!queried.synthesis.evidence.is_empty());
    assert_eq!(snapshot(&corpus.db).await, before);
}

#[tokio::test]
async fn query_mode_reuses_build_context_and_pins_each_chunk() {
    let corpus = corpus().await;
    let generator = FixtureSynthesisGenerator::fixed(draft(
        vec![claim(
            "The north orchard lines were flushed.",
            vec![cite(&corpus.log, "drip lines were flushed on 3 March")],
        )],
        vec![],
    ));
    let plan = SynthesisPlan {
        input: SynthesisInput::Query {
            query: "north orchard drip lines flushed".into(),
            scope: SearchScope::Notes,
            graph: GraphMode::On,
        },
        ..seed_plan(&corpus.log)
    };
    let outcome = run(&corpus, &generator, &plan).await.unwrap();
    assert_eq!(outcome.usage.mode, "query");
    let evidence = &outcome.synthesis.evidence;
    assert!(evidence.iter().any(|item| item.chunk_id == corpus.log));
    for item in evidence {
        let pinned = corpus.repo.inspect_record(&item.chunk_id, 0).await.unwrap();
        assert_eq!(item.revision, pinned.revision);
    }
    assert_eq!(outcome.synthesis.claims.len(), 1);
    assert_eq!(outcome.synthesis.timeline.len(), 1);
    assert_eq!(outcome.synthesis.timeline[0].chunk_id, corpus.log);
    assert_eq!(
        outcome.synthesis.open_questions,
        vec!["Was the gauge recalibrated?".to_string()]
    );

    let silent = FixtureSynthesisGenerator::fixed(draft(vec![], vec![]));
    let empty = SynthesisPlan {
        input: SynthesisInput::Query {
            query: "zebra quasar".into(),
            scope: SearchScope::Notes,
            graph: GraphMode::Off,
        },
        ..seed_plan(&corpus.log)
    };
    let outcome = run(&corpus, &silent, &empty).await.unwrap();
    assert!(outcome.synthesis.evidence.is_empty() && !outcome.usage.generator_called);
    assert!(silent.requests().is_empty());
    assert_eq!(outcome.synthesis.gaps[0].kind, GapKind::NoEvidence);
}

#[tokio::test]
async fn same_source_chunks_do_not_corroborate_each_other_end_to_end() {
    let corpus = corpus().await;
    let source = corpus
        .repo
        .create_source(Source::manual().with_title("Field journal"))
        .await
        .unwrap();
    let page = |content: &str| {
        Note::new(content)
            .with_title("Field journal")
            .with_source(source.id.clone().unwrap())
    };
    let first = corpus
        .repo
        .create_note(page(
            "Field journal page one: the orchard gauge read 2.1 bar at dawn.",
        ))
        .await
        .unwrap();
    let second = corpus
        .repo
        .create_note(page(
            "Field journal page two: the orchard gauge read 2.1 bar at dusk.",
        ))
        .await
        .unwrap();
    let (first, second) = (
        record_id_to_string(first.id.as_ref().unwrap()),
        record_id_to_string(second.id.as_ref().unwrap()),
    );
    edge(&corpus.repo, &first, &second, EdgeType::Supports).await;
    let generator = FixtureSynthesisGenerator::fixed(draft(
        vec![claim(
            "The gauge read 2.1 bar all day.",
            vec![
                cite(&first, "the orchard gauge read 2.1 bar at dawn"),
                cite(&second, "the orchard gauge read 2.1 bar at dusk"),
            ],
        )],
        vec![],
    ));
    let outcome = run(&corpus, &generator, &seed_plan(&first)).await.unwrap();
    let strength = &outcome.synthesis.claims[0].strength;
    assert_eq!(
        (strength.independent_origins, strength.cited_chunks),
        (1, 2)
    );
    assert_eq!(
        outcome.synthesis.lineage[0].reasons,
        vec![LineageReason::SameSource]
    );
}
