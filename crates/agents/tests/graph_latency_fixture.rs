//! Provider-free graph latency reproduction with fictional, stable records.
//!
//! Run `cargo test -p graphrag-agents --test graph_latency_fixture -- --nocapture`.
//! `GRAPHRAG_GRAPH_LATENCY_ROUNDS=10` collects ten warmed observations;
//! `GRAPHRAG_GRAPH_LATENCY_LARGE=1` adds a 10,000-note population. An optional
//! `GRAPHRAG_GRAPH_LATENCY_SCALES=64,256` selects populations from 64 to 10,000.
//! The default populations are 256 and 2,048 notes.
//! An optional
//! `GRAPHRAG_GRAPH_LATENCY_REPORT` writes JSON to a new path, refusing overwrite.
//! Timings are observations, never pass/fail thresholds. Independently measured
//! repository stages are probes and must not be summed as an end-to-end trace.

use chrono::{Duration, Utc};
use graphrag_agents::{
    DeterministicEmbedder, GraphMode, GraphRetrievalConfig, GraphSearchResults, SearchAgent,
    SearchScope,
};
use graphrag_core::{record_id_to_string, Entity, EntityType, Note, Source};
use graphrag_db::{compatibility::EmbeddingIdentity, init_memory, DbConnection, Repository};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::OpenOptions,
    io::Write,
    sync::Arc,
    time::Instant,
};
use surrealdb::types::RecordId;

const NARROW: &str = "Orion Service";
const BROAD: &str =
    "where do Orion Service, Nebula Project, Quasar Ledger and Vega Gateway connect";
const SCOPED_URI: &str = "fixture://graph-latency/scoped.md";
const FOREIGN_URI: &str = "fixture://graph-latency/foreign.md";
const ENTITY_NAMES: [&str; 4] = [NARROW, "Nebula Project", "Quasar Ledger", "Vega Gateway"];
const VISIBLE: &str = "(source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)";

#[derive(Clone, Copy)]
enum Mentions {
    None,
    Sparse,
    Dense,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    mentions: Mentions,
    edges: bool,
    broad: bool,
    source_filter: bool,
    age_filter: bool,
}

const CASES: [Case; 6] = [
    Case {
        name: "A-no-mentions-empty-edges",
        mentions: Mentions::None,
        edges: false,
        broad: false,
        source_filter: false,
        age_filter: false,
    },
    Case {
        name: "B-sparse-mentions-empty-edges",
        mentions: Mentions::Sparse,
        edges: false,
        broad: false,
        source_filter: false,
        age_filter: false,
    },
    Case {
        name: "C-dense-mentions-empty-edges-broad",
        mentions: Mentions::Dense,
        edges: false,
        broad: true,
        source_filter: false,
        age_filter: false,
    },
    Case {
        name: "D-dense-mentions-empty-edges-source",
        mentions: Mentions::Dense,
        edges: false,
        broad: true,
        source_filter: true,
        age_filter: false,
    },
    Case {
        name: "E-sparse-mentions-accepted-edges",
        mentions: Mentions::Sparse,
        edges: true,
        broad: false,
        source_filter: false,
        age_filter: false,
    },
    Case {
        name: "F-dense-mentions-accepted-edges-source-age",
        mentions: Mentions::Dense,
        edges: true,
        broad: true,
        source_filter: true,
        age_filter: true,
    },
];

#[derive(Serialize)]
struct Observations {
    /// First measured call for this policy, before independent stage probes.
    /// Earlier policies may have warmed shared state; this is not cold-start.
    first_ms: f64,
    warmed_ms: Vec<f64>,
    warmed_median_ms: f64,
    /// Omitted for fewer than ten observations; these are small-sample values.
    warmed_p95_ms: Option<f64>,
}

#[derive(Serialize)]
struct CaseReport {
    case: &'static str,
    notes: usize,
    mentions: usize,
    accepted_edges: usize,
    query: &'static str,
    source_filter: bool,
    since_days: Option<u32>,
    repository_stage_probes_ms: BTreeMap<&'static str, Vec<f64>>,
    search: BTreeMap<&'static str, Observations>,
    snapshots: BTreeMap<&'static str, Value>,
}

struct Fixture {
    db: DbConnection,
    repo: Repository,
    search: SearchAgent,
    note_count: usize,
    anchor: chrono::DateTime<Utc>,
}

fn note_id(index: usize) -> RecordId {
    RecordId::new("note", format!("latency-{index:05}"))
}

fn entity_id(index: usize) -> RecordId {
    RecordId::new("entity", format!("latency-{index}"))
}

fn query_vector() -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = 1.0;
    vector
}

fn note_vector(index: usize, count: usize) -> Vec<f32> {
    let mut vector = query_vector();
    // Distinct cosine scores avoid arbitrary equal-distance KNN candidates.
    let similarity = if index == 1 {
        0.99
    } else {
        0.9 - index as f32 * 0.8 / (count + 1) as f32
    };
    vector[0] = similarity;
    vector[1] = (1.0 - similarity * similarity).sqrt();
    vector
}

fn foreign(index: usize) -> bool {
    index.is_multiple_of(4)
}
fn stale(index: usize) -> bool {
    index.is_multiple_of(17)
}
fn hidden(index: usize) -> bool {
    index.is_multiple_of(23)
}

async fn execute(db: &DbConnection, sql: &str) {
    let mut response = db.query(sql).await.unwrap();
    assert!(
        response.take_errors().is_empty(),
        "fictional fixture SQL failed"
    );
}

impl Fixture {
    async fn new(note_count: usize) -> Self {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        repo.record_embedding_metadata(
            &EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
            None,
        )
        .await
        .unwrap();
        for (key, uri) in [("scoped", SCOPED_URI), ("foreign", FOREIGN_URI)] {
            let mut source = Source::manual();
            source.uri = Some(uri.into());
            source.normalized_uri = Some(uri.into());
            source.generation = 1;
            source.successful_generation = 1;
            let created: Option<Source> = db.create(("source", key)).content(source).await.unwrap();
            assert!(created.is_some());
        }
        for (index, name) in ENTITY_NAMES.iter().enumerate() {
            let mut entity = Entity::new(*name, EntityType::Project);
            entity.metadata = json!({ "aliases": [format!("Fictional alias {index}")] });
            let created: Option<Entity> =
                db.create(entity_id(index)).content(entity).await.unwrap();
            assert!(created.is_some());
        }
        let anchor = Utc::now();
        for start in (1..=note_count).step_by(64) {
            // Typed bulk insertion keeps fixture construction outside timing,
            // while all IDs and vector scores remain stable between versions.
            let notes = (start..=(start + 63).min(note_count)).map(|index| {
                let content = if index <= 12 {
                    format!("Orion Service Nebula Project Quasar Ledger Vega Gateway. Fictional operation {index}.")
                } else { format!("Unrelated fictional background document {index}.") };
                let mut note = Note::new(&content)
                    .with_title(if index == 1 { NARROW.to_string() } else { format!("Fictional heading {index}") })
                    .with_embedding(note_vector(index, note_count))
                    .with_source(RecordId::new("source", if foreign(index) { "foreign" } else { "scoped" }))
                    .with_source_generation(if hidden(index) { 2 } else { 1 });
                note.id = Some(note_id(index));
                note.search_content = Some(content);
                note.created_at = anchor - Duration::days(if stale(index) { 14 } else { 0 });
                note.updated_at = note.created_at;
                note
            }).collect::<Vec<_>>();
            let created: Vec<Note> = db.insert("note").content(notes).await.unwrap();
            assert!(!created.is_empty());
        }
        let search = SearchAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_default_embedding(query_vector())),
        );
        Self {
            db,
            repo,
            search,
            note_count,
            anchor,
        }
    }

    async fn configure(&self, case: Case) -> (usize, usize) {
        execute(&self.db, "DELETE mentions; DELETE supports; DELETE contradicts; DELETE related_to; DELETE derived_from; DELETE proposed_edge;").await;
        let mentioned = match case.mentions {
            Mentions::None => 0,
            Mentions::Sparse => 12,
            Mentions::Dense => self.note_count,
        };
        for start in (1..=mentioned).step_by(64) {
            let mut request = self.db.query((start..=(start + 63).min(mentioned)).map(|index| {
                format!("CREATE mentions:latency_{index} SET in = $note_{index}, out = $entity_{index}, created_at = <datetime>$created;")
            }).collect::<String>()).bind(("created", self.anchor.to_rfc3339()));
            for index in start..=(start + 63).min(mentioned) {
                // Sparse case directly associates the primary hit with the
                // narrow entity; dense associations distribute all four seeds.
                let entity = if index == 1 { 0 } else { (index % 5) % 4 };
                request = request
                    .bind((format!("note_{index}"), note_id(index)))
                    .bind((format!("entity_{index}"), entity_id(entity)));
            }
            assert!(request.await.unwrap().take_errors().is_empty());
        }
        let mut edge_count = 0;
        if case.edges {
            let mut edges = vec![
                ("supports", 1, 2, 0.6),
                ("supports", 1, 4, 0.99),
                ("supports", 1, 17, 0.98),
                ("supports", 1, 23, 1.0),
            ];
            for seed in 1..=12 {
                for (offset, table) in ["supports", "contradicts", "related_to", "derived_from"]
                    .iter()
                    .enumerate()
                {
                    edges.push((*table, seed, 100 + seed * 4 + offset, 0.7));
                }
            }
            for (index, (table, from, to, confidence)) in edges.into_iter().enumerate() {
                if to > self.note_count {
                    continue;
                }
                let mut response = self.db.query(format!("CREATE {table}:latency_{index} SET in = $from, out = $to, confidence = $confidence, is_manual = true, created_at = <datetime>$created;")).bind(("from", note_id(from))).bind(("to", note_id(to))).bind(("confidence", confidence)).bind(("created", self.anchor.to_rfc3339())).await.unwrap();
                assert!(response.take_errors().is_empty());
                edge_count += 1;
            }
            // A high-confidence pending proposal to another eligible note must
            // never become an accepted traversal path.
            let mut response = self.db.query("CREATE proposed_edge:latency_pending SET dedupe_key = 'fictional-pending', in = $from, out = $to, edge_type = 'supports', confidence = 1.0, reason = 'fictional pending proposal', generator = 'fictional-latency-fixture', status = 'pending', created_at = <datetime>$created, updated_at = <datetime>$created;").bind(("from", note_id(1))).bind(("to", note_id(200.min(self.note_count)))).bind(("created", self.anchor.to_rfc3339())).await.unwrap();
            assert!(response.take_errors().is_empty());
        }
        (mentioned, edge_count)
    }

    fn assert_results(&self, case: Case, mode: GraphMode, results: &GraphSearchResults) {
        assert!(results.hits.len() <= 5);
        assert!(results.summary.entities_matched <= 4);
        assert!(results.summary.candidates_considered <= 32);
        let mut ids = HashSet::new();
        for hit in &results.hits {
            assert!(ids.insert(&hit.id), "duplicate canonical result");
            let index = hit
                .id
                .strip_prefix("note:latency-")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            assert!(!hidden(index), "superseded source generation leaked");
            assert!(
                !case.source_filter || !foreign(index),
                "foreign source leaked"
            );
            assert!(!case.age_filter || !stale(index), "old note leaked");
            if let Some(graph) = &hit.graph {
                assert_ne!(mode, GraphMode::Off);
                assert!(graph.hops <= 1);
                assert_eq!(graph.path.len(), graph.hops);
                assert!(graph.score.is_finite());
                assert_eq!(
                    graph.source_uri.as_deref(),
                    Some(if foreign(index) {
                        FOREIGN_URI
                    } else {
                        SCOPED_URI
                    })
                );
                for step in &graph.path {
                    assert!(!step.edge_id.contains("proposed_edge"));
                    assert!(step
                        .edge_id
                        .starts_with(&format!("{}:latency_", step.edge_type)));
                    assert!(step.confidence >= 0.0 && step.confidence <= 1.0);
                    assert!(["supports", "contradicts", "related_to", "derived_from"]
                        .contains(&step.edge_type.as_str()));
                    assert_eq!(step.from_id, graph.seed_note_id);
                    assert_eq!(step.to_id, hit.id);
                }
                assert!(case.edges || graph.hops == 0);
            }
        }
        if matches!(mode, GraphMode::Off) {
            assert_eq!(results.summary.candidates_considered, 0);
        }
        if matches!(case.mentions, Mentions::None) && matches!(mode, GraphMode::Auto) {
            assert_eq!(results.summary.candidates_considered, 0);
        }
    }

    async fn stage_probes(&self, case: Case, rounds: usize) -> BTreeMap<&'static str, Vec<f64>> {
        let query = if case.broad { BROAD } else { NARROW };
        let since = case.age_filter.then_some(self.anchor - Duration::days(1));
        let source = case.source_filter.then(|| SCOPED_URI.to_string());
        let config = GraphRetrievalConfig::default();
        let mut probes = BTreeMap::<&'static str, Vec<f64>>::new();
        for _ in 0..rounds {
            let started = Instant::now();
            let fulltext = self
                .repo
                .fulltext_search_notes(query, 40, since, source.clone())
                .await
                .unwrap();
            probes.entry("keyword").or_default().push(ms(started));
            let started = Instant::now();
            let _vectors = self
                .repo
                .vector_search_notes(query_vector(), 40, since, source.clone())
                .await
                .unwrap();
            probes.entry("vector").or_default().push(ms(started));
            let started = Instant::now();
            let entities = self
                .repo
                .find_graph_entities(query, config.max_seed_entities)
                .await
                .unwrap();
            probes
                .entry("entity_matching")
                .or_default()
                .push(ms(started));
            let started = Instant::now();
            let mut response = self.db.query(format!("SELECT VALUE id FROM note WHERE ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE}")).bind(("since", since.map(|value| value.to_rfc3339()))).bind(("source_uri", source.clone())).await.unwrap();
            let _eligible: Vec<RecordId> = response.take(0).unwrap();
            probes
                .entry("eligibility_standalone_probe")
                .or_default()
                .push(ms(started));
            let started = Instant::now();
            let seeds = self
                .repo
                .graph_notes_for_entities(
                    &entities
                        .iter()
                        .map(|entity| entity.id.clone())
                        .collect::<Vec<_>>(),
                    config.max_seed_notes,
                    since,
                    source.clone(),
                )
                .await
                .unwrap();
            probes
                .entry("mentions_with_eligibility")
                .or_default()
                .push(ms(started));
            assert!(seeds.len() <= entities.len() * config.max_seed_notes);
            for seed in &seeds {
                let index = record_id_to_string(&seed.note_id)
                    .strip_prefix("note:latency-")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                assert!(!hidden(index));
                assert!(!case.source_filter || !foreign(index));
                assert!(!case.age_filter || !stale(index));
            }
            let mut ids = seeds
                .iter()
                .map(|seed| seed.note_id.clone())
                .collect::<Vec<_>>();
            ids.sort_by_key(record_id_to_string);
            ids.dedup();
            ids.truncate(config.max_seed_notes);
            if ids.is_empty() {
                ids.extend(
                    fulltext
                        .iter()
                        .take(config.max_seed_notes)
                        .map(|hit| hit.id.clone()),
                );
            }
            let visited = ids
                .iter()
                .map(|id| (record_id_to_string(id), vec![id.clone()]))
                .collect::<HashMap<_, _>>();
            let started = Instant::now();
            let edges = self
                .repo
                .graph_note_edges_excluding_visited(
                    &ids,
                    &config.allowed_edge_types,
                    config.per_node_fanout,
                    config.allow_outbound,
                    config.allow_inbound,
                    config.min_confidence,
                    since,
                    source.clone(),
                    &visited,
                )
                .await
                .unwrap();
            probes
                .entry("accepted_edges_with_eligibility")
                .or_default()
                .push(ms(started));
            assert!(
                edges.len() <= ids.len() * config.allowed_edge_types.len() * config.per_node_fanout
            );
            if !case.edges {
                assert!(edges.is_empty());
            } else {
                assert!(
                    !edges.is_empty(),
                    "nonempty accepted-edge case did not exercise expansion"
                );
            }
            for edge in &edges {
                for endpoint in [&edge.in_id, &edge.out_id] {
                    let index = record_id_to_string(endpoint)
                        .strip_prefix("note:latency-")
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    assert!(!hidden(index));
                    assert!(!case.source_filter || !foreign(index));
                    assert!(!case.age_filter || !stale(index));
                }
            }
            for edge in edges {
                ids.extend([edge.in_id, edge.out_id]);
            }
            ids.sort_by_key(record_id_to_string);
            ids.dedup();
            ids.truncate(config.candidate_cap);
            let started = Instant::now();
            let records = self
                .repo
                .graph_notes_by_ids(&ids, since, source.clone())
                .await
                .unwrap();
            probes
                .entry("note_hydration")
                .or_default()
                .push(ms(started));
            let started = Instant::now();
            let _provenance = self
                .repo
                .graph_note_provenance_ids(
                    &records
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>(),
                )
                .await
                .unwrap();
            probes
                .entry("provenance_hydration")
                .or_default()
                .push(ms(started));
        }
        probes
    }

    async fn measure(&self, case: Case, rounds: usize) -> CaseReport {
        let (mentions, accepted_edges) = self.configure(case).await;
        let query = if case.broad { BROAD } else { NARROW };
        let source = case.source_filter.then(|| SCOPED_URI.to_string());
        let since_days = case.age_filter.then_some(1);
        let mut search = BTreeMap::new();
        let mut snapshots = BTreeMap::new();
        for (label, mode) in [
            ("off", GraphMode::Off),
            ("auto", GraphMode::Auto),
            ("on", GraphMode::On),
        ] {
            let started = Instant::now();
            let first = self
                .search
                .search_with_scope_graph(
                    query,
                    5,
                    SearchScope::Notes,
                    since_days,
                    source.clone(),
                    mode,
                )
                .await
                .unwrap();
            let first_ms = ms(started);
            self.assert_results(case, mode, &first);
            let expected = snapshot(&first);
            let mut warmed_ms = Vec::new();
            for _ in 0..rounds {
                let started = Instant::now();
                let result = self
                    .search
                    .search_with_scope_graph(
                        query,
                        5,
                        SearchScope::Notes,
                        since_days,
                        source.clone(),
                        mode,
                    )
                    .await
                    .unwrap();
                warmed_ms.push(ms(started));
                self.assert_results(case, mode, &result);
                assert_eq!(
                    snapshot(&result),
                    expected,
                    "ranking/evidence changed between identical warmed requests"
                );
            }
            let mut sorted = warmed_ms.clone();
            sorted.sort_by(f64::total_cmp);
            let middle = sorted.len() / 2;
            let warmed_median_ms = if sorted.len() % 2 == 0 {
                (sorted[middle - 1] + sorted[middle]) / 2.0
            } else {
                sorted[middle]
            };
            let warmed_p95_ms =
                (sorted.len() >= 10).then(|| sorted[(sorted.len() * 95).div_ceil(100) - 1]);
            search.insert(
                label,
                Observations {
                    first_ms,
                    warmed_ms,
                    warmed_median_ms,
                    warmed_p95_ms,
                },
            );
            snapshots.insert(label, expected);
        }
        if matches!(case.mentions, Mentions::None) && !case.edges {
            assert_eq!(
                snapshots["off"]["hits"], snapshots["auto"]["hits"],
                "graph auto must preserve baseline when no useful entity mentions exist"
            );
        }
        let repository_stage_probes_ms = self.stage_probes(case, rounds).await;
        CaseReport {
            case: case.name,
            notes: self.note_count,
            mentions,
            accepted_edges,
            query,
            source_filter: case.source_filter,
            since_days,
            repository_stage_probes_ms,
            search,
            snapshots,
        }
    }
}

fn snapshot(results: &GraphSearchResults) -> Value {
    json!({ "summary": results.summary, "hits": results.hits.iter().map(|hit| json!({ "id": hit.id, "title": hit.title, "score": hit.score, "fusion": hit.fusion, "graph": hit.graph, "score_kind": hit.score_kind })).collect::<Vec<_>>() })
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[tokio::test]
async fn sanitized_graph_latency_fixture() {
    let rounds = std::env::var("GRAPHRAG_GRAPH_LATENCY_ROUNDS")
        .map(|value| value.parse::<usize>().expect("rounds must be an integer"))
        .unwrap_or(3);
    assert!(
        (1..=100).contains(&rounds),
        "rounds must be between 1 and 100"
    );
    let mut scales = match std::env::var("GRAPHRAG_GRAPH_LATENCY_SCALES") {
        Ok(value) => {
            let values = value
                .split(',')
                .map(|part| {
                    part.trim()
                        .parse::<usize>()
                        .expect("scales must be a comma-separated list of integers")
                })
                .collect::<Vec<_>>();
            assert!(
                !values.is_empty() && values.iter().all(|scale| (64..=10_000).contains(scale)),
                "each fixture scale must be between 64 and 10000 notes"
            );
            assert_eq!(
                values.iter().copied().collect::<HashSet<_>>().len(),
                values.len(),
                "fixture scales must not contain duplicates"
            );
            values
        }
        Err(std::env::VarError::NotPresent) => vec![256, 2048],
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("fixture scales must contain Unicode text")
        }
    };
    if std::env::var("GRAPHRAG_GRAPH_LATENCY_LARGE").as_deref() == Ok("1")
        && !scales.contains(&10_000)
    {
        scales.push(10_000);
    }
    let mut cases = Vec::new();
    for scale in &scales {
        let fixture = Fixture::new(*scale).await;
        for case in CASES {
            let report = fixture.measure(case, rounds).await;
            eprintln!("graph latency fixture: completed {} notes={} mentions={} accepted_edges={} warmed_median_ms off={:.2} auto={:.2} on={:.2}", report.case, report.notes, report.mentions, report.accepted_edges, report.search["off"].warmed_median_ms, report.search["auto"].warmed_median_ms, report.search["on"].warmed_median_ms);
            cases.push(report);
        }
        eprintln!(
            "graph latency fixture: completed scale={} cases={} warmed_rounds={}",
            scale,
            CASES.len(),
            rounds
        );
    }
    let report = json!({ "schema_version": 1, "fixture": "fictional-graph-latency-v1", "providers": "deterministic-offline", "scales": scales, "warmed_rounds": rounds, "timing_policy": "first and warmed observations; independent repository probes are not additive phase traces; no timing assertions", "cases": cases });
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(path) = std::env::var_os("GRAPHRAG_GRAPH_LATENCY_REPORT") {
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .expect("report destination must be a new writable file");
        output.write_all(&bytes).unwrap();
        output.write_all(b"\n").unwrap();
    } else {
        eprintln!("{}", String::from_utf8(bytes).unwrap());
    }
}
