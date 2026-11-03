//! #186: accepted-edge support anchored at a directly retrieved note.
//!
//! Fictional fixtures reproduce the shape observed in the fixed graph
//! evaluation: a summary note is the top hybrid hit, its accepted
//! `derived_from` source detail is only a weak lexical/entity match, and
//! vector-only matches fill the build_context over-fetch window. Every
//! assertion uses the evaluation's fixed packing budget unless it states the
//! budget explicitly.
use super::*;
use crate::DeterministicEmbedder;
use graphrag_core::{EdgeType, EntityType, Note};
use graphrag_db::init_memory;
use std::sync::Arc;

const DIMENSIONS: usize = 1024;
const QUERY: &str = "kestrel boundary canary rollout evidence";

fn axis(index: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; DIMENSIONS];
    embedding[index] = 1.0;
    embedding
}

/// A unit vector `angle` radians away from the query direction (axis 0).
fn near_query(angle: f32, axis_index: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; DIMENSIONS];
    embedding[0] = angle.cos();
    embedding[axis_index] = angle.sin();
    embedding
}

fn id(note: &Note) -> String {
    record_id_to_string(note.id.as_ref().unwrap())
}

fn fixed_budget(max_chunks: usize) -> AugmentOptions {
    AugmentOptions {
        max_chunks,
        max_total_tokens: 1000,
        max_chunk_tokens: 400,
        ..Default::default()
    }
}

struct Corpus {
    repo: Repository,
    /// Top hybrid hit: rank 1 by vector, but it covers only two of the five
    /// query terms, like the fixed suite's summary note.
    anchor: Note,
    /// Vector-only matches that fill the over-fetch window in rank order.
    fillers: Vec<Note>,
}

async fn corpus(fillers: usize) -> Corpus {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.record_embedding_metadata(
        &EmbeddingIdentity::new("deterministic-test", "fixture", DIMENSIONS),
        None,
    )
    .await
    .unwrap();
    let anchor = repo
        .create_note(
            Note::new("Kestrel rollout summary for the operations review").with_embedding(axis(0)),
        )
        .await
        .unwrap();
    let mut filler_notes = Vec::new();
    for index in 0..fillers {
        filler_notes.push(
            repo.create_note(
                Note::new(format!("vector filler {index} about unrelated material"))
                    .with_embedding(near_query(0.05 * (index + 1) as f32, index + 1)),
            )
            .await
            .unwrap(),
        );
    }
    Corpus {
        repo,
        anchor,
        fillers: filler_notes,
    }
}

/// A lexical-only note with no embedding: it is a bounded full-text
/// candidate for the query, below every vector-backed window hit.
async fn lexical_note(repo: &Repository, content: &str) -> Note {
    repo.create_note(Note::new(content)).await.unwrap()
}

async fn accept(repo: &Repository, from: &Note, to: &Note, edge_type: EdgeType) {
    repo.create_edge(
        from.id.as_ref().unwrap(),
        to.id.as_ref().unwrap(),
        edge_type,
        Some(1.0),
    )
    .await
    .unwrap();
}

async fn link_entity(repo: &Repository, note: &Note, name: &str) {
    let mut entity = Entity::new(name, EntityType::Project);
    entity.metadata = serde_json::json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(note.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
        .await
        .unwrap();
}

fn agent(repo: &Repository, graph: GraphRetrievalConfig) -> SearchAgent {
    SearchAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default().with_default_embedding(axis(0))),
    )
    .with_graph_config(graph)
}

async fn context(search: &SearchAgent, options: AugmentOptions, mode: GraphMode) -> AugmentContext {
    search
        .build_augmented_context_with_graph_explain(
            QUERY,
            SearchScope::Notes,
            None,
            None,
            None,
            options,
            mode,
        )
        .await
        .unwrap()
}

fn chunk_ids(context: &AugmentContext) -> Vec<String> {
    context
        .chunks
        .iter()
        .map(|chunk| chunk.id.clone())
        .collect()
}

/// The exact reconstruction the fixed evaluation needs: a contiguous accepted
/// path from the graph seed that ends at the returned chunk itself.
fn assert_contiguous_path(
    graph: &GraphEvidence,
    chunk_id: &str,
    seed: &str,
    edge_type: &str,
    direction: &str,
) {
    assert_eq!(graph.seed_note_id, seed);
    assert_eq!(graph.hops, graph.path.len());
    assert_eq!(graph.hops, 1);
    let step = &graph.path[0];
    assert_eq!(step.from_id, seed);
    assert_eq!(step.to_id, chunk_id);
    assert_eq!(step.edge_type, edge_type);
    assert_eq!(step.direction, direction);
    assert!(step.edge_id.starts_with(&format!("{edge_type}:")));
}

#[tokio::test]
async fn entity_seeded_derived_from_support_competes_from_its_retrieved_summary() {
    // The support is a weak lexical match and its own entity seed. Before
    // #186 the hop-0 mention evidence (coverage 3/5) outscored the accepted
    // path from the summary (coverage 2/5 times decay), and the graph-only
    // score could never reach the eight-hit over-fetch window.
    let corpus = corpus(8).await;
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary boundary detail with the exact head and coverage numbers",
    )
    .await;
    link_entity(&corpus.repo, &support, "Kestrel").await;
    accept(
        &corpus.repo,
        &corpus.anchor,
        &support,
        EdgeType::DerivedFrom,
    )
    .await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());
    let (anchor_id, support_id) = (id(&corpus.anchor), id(&support));

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    assert_eq!(
        chunk_ids(&off),
        vec![anchor_id.clone(), id(&corpus.fillers[0])]
    );
    assert_eq!(off.diagnostics.graph_candidates_considered, 0);
    assert_eq!(off.diagnostics.graph_candidates_added, 0);

    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&on), vec![anchor_id.clone(), support_id.clone()]);
    assert!(on.total_tokens <= 1000);
    let support_chunk = &on.chunks[1];
    assert_contiguous_path(
        support_chunk.graph.as_ref().unwrap(),
        &support_id,
        &anchor_id,
        "derived_from",
        "outbound",
    );
    // The summary keeps its direct retrieval score; the inbound path from
    // the support's entity seed is only an annotation on a direct hit.
    let anchor_chunk = &on.chunks[0];
    assert_eq!(
        anchor_chunk.score_kind,
        crate::ScoreKind::ReciprocalRankFusion
    );
    assert_contiguous_path(
        anchor_chunk.graph.as_ref().unwrap(),
        &anchor_id,
        &support_id,
        "derived_from",
        "inbound",
    );
    assert_eq!(on.diagnostics.graph_candidates_selected, 2);
    assert_eq!(on.diagnostics.graph_candidates_added, 1);
    assert_eq!(on.diagnostics.graph_candidates_annotating, 1);

    // `auto` stays entity-seeded: the summary is not a traversal anchor, so
    // the support is not lifted and the path remains an annotation.
    let auto = context(&search, fixed_budget(2), GraphMode::Auto).await;
    assert_eq!(chunk_ids(&auto), chunk_ids(&off));
    assert_eq!(auto.diagnostics.graph_candidates_added, 0);
    assert_eq!(auto.diagnostics.graph_candidates_annotating, 1);
    assert!(auto.diagnostics.graph_candidates_rank_limited >= 1);
}

#[tokio::test]
async fn lexically_corroborated_neighbor_of_a_retrieved_anchor_enters_the_fixed_budget() {
    let corpus = corpus(8).await;
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    accept(&corpus.repo, &corpus.anchor, &support, EdgeType::Supports).await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    assert!(!chunk_ids(&off).contains(&id(&support)));
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&on), vec![id(&corpus.anchor), id(&support)]);
    assert_contiguous_path(
        on.chunks[1].graph.as_ref().unwrap(),
        &id(&support),
        &id(&corpus.anchor),
        "supports",
        "outbound",
    );
    assert_eq!(on.diagnostics.graph_candidates_added, 1);
    assert_eq!(on.diagnostics.graph_candidates_annotating, 0);
}

#[tokio::test]
async fn anchored_support_competes_but_does_not_displace_a_stronger_direct_match() {
    // A second direct match in both channels outranks the decayed neighbor:
    // at two chunks it keeps its slot (no graph quota); at three chunks the
    // support outranks the vector-only fillers and is added.
    let corpus = corpus(12).await;
    let competitor = corpus
        .repo
        .create_note(
            Note::new("Kestrel boundary canary rollout evidence report")
                .with_embedding(near_query(0.01, 900)),
        )
        .await
        .unwrap();
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    accept(
        &corpus.repo,
        &corpus.anchor,
        &support,
        EdgeType::DerivedFrom,
    )
    .await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    let two = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&two), vec![id(&corpus.anchor), id(&competitor)]);
    assert_eq!(two.diagnostics.graph_candidates_added, 0);
    // It reached packing rather than being cut by the over-fetch window.
    assert!(two
        .exclusions
        .iter()
        .any(|explanation| explanation.result_id == id(&support)
            && explanation.inclusion == crate::InclusionReason::DiagnosticLimit
            && explanation.graph.is_some()));
    assert_eq!(two.diagnostics.graph_candidates_rank_limited, 0);

    let three = context(&search, fixed_budget(3), GraphMode::On).await;
    assert_eq!(
        chunk_ids(&three),
        vec![id(&corpus.anchor), id(&competitor), id(&support)]
    );
    assert_eq!(three.diagnostics.graph_candidates_added, 1);
}

#[tokio::test]
async fn hub_neighbors_without_query_evidence_are_not_lifted_by_a_retrieved_anchor() {
    let corpus = corpus(8).await;
    let mut hub_neighbors = Vec::new();
    for index in 0..6 {
        let neighbor = lexical_note(
            &corpus.repo,
            &format!("generic hub neighbor {index} with other material"),
        )
        .await;
        accept(&corpus.repo, &corpus.anchor, &neighbor, EdgeType::RelatedTo).await;
        hub_neighbors.push(id(&neighbor));
    }
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&on), chunk_ids(&off));
    assert_eq!(on.diagnostics.graph_candidates_added, 0);
    assert_eq!(
        on.diagnostics.graph_candidates_rank_limited,
        hub_neighbors.len()
    );

    // The ranked over-fetch window itself is unchanged by the hub.
    let window_off = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::Off)
        .await
        .unwrap();
    let window_on = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::On)
        .await
        .unwrap();
    let ranked = |results: &GraphSearchResults| {
        results
            .hits
            .iter()
            .map(|hit| (hit.id.clone(), hit.score.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(ranked(&window_on), ranked(&window_off));
    assert!(window_on
        .hits
        .iter()
        .all(|hit| !hub_neighbors.contains(&hit.id)));
}

#[tokio::test]
async fn lifted_near_copy_of_the_anchor_is_suppressed_not_added() {
    let corpus = corpus(8).await;
    let copy = lexical_note(
        &corpus.repo,
        "Kestrel rollout summary for the operations review",
    )
    .await;
    accept(&corpus.repo, &corpus.anchor, &copy, EdgeType::DerivedFrom).await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    // Default MMR already demotes a zero-novelty copy below a filler.
    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&on), chunk_ids(&off));
    assert_eq!(on.diagnostics.graph_candidates_added, 0);
    assert!(on
        .exclusions
        .iter()
        .any(|explanation| explanation.result_id == id(&copy) && explanation.graph.is_some()));

    // Even when relevance alone orders packing, the lifted copy is the next
    // candidate and the near-duplicate guard rejects it against its anchor.
    let relevance_only = AugmentOptions {
        novelty_weight: 0.0,
        ..fixed_budget(2)
    };
    let off = context(&search, relevance_only.clone(), GraphMode::Off).await;
    let on = context(&search, relevance_only, GraphMode::On).await;
    assert_eq!(chunk_ids(&on), chunk_ids(&off));
    assert_eq!(on.diagnostics.graph_candidates_added, 0);
    let suppressed = on
        .exclusions
        .iter()
        .find(|explanation| explanation.result_id == id(&copy))
        .expect("the lifted copy reaches packing");
    assert_eq!(suppressed.inclusion, crate::InclusionReason::NearDuplicate);
    assert_eq!(
        suppressed
            .near_duplicate
            .as_ref()
            .unwrap()
            .matching_result_id,
        id(&corpus.anchor)
    );
}

#[tokio::test]
async fn graph_off_is_unchanged_by_an_accepted_edge_that_changes_graph_on() {
    let corpus = corpus(8).await;
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    link_entity(&corpus.repo, &support, "Kestrel").await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());
    let snapshot = |results: &GraphSearchResults| {
        results
            .hits
            .iter()
            .map(|hit| serde_json::to_value(hit.explanation()).unwrap())
            .collect::<Vec<_>>()
    };
    let packed = |context: &AugmentContext| {
        context
            .chunks
            .iter()
            .map(|chunk| {
                (
                    chunk.id.clone(),
                    chunk.score.to_bits(),
                    chunk.snippet.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    let before_window = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::Off)
        .await
        .unwrap();
    let before_context = context(&search, fixed_budget(2), GraphMode::Off).await;

    accept(
        &corpus.repo,
        &corpus.anchor,
        &support,
        EdgeType::DerivedFrom,
    )
    .await;
    let after_window = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::Off)
        .await
        .unwrap();
    let after_context = context(&search, fixed_budget(2), GraphMode::Off).await;
    assert_eq!(snapshot(&after_window), snapshot(&before_window));
    assert_eq!(after_window.summary, GraphRetrievalSummary::default());
    assert_eq!(packed(&after_context), packed(&before_context));
    assert!(after_context
        .chunks
        .iter()
        .all(|chunk| chunk.graph.is_none()));
    // Positive control, not part of the graph-off guarantee: the same edge
    // does add the support once graph retrieval is requested, so the identity
    // above is not vacuous. Only this assertion depends on the #186 policy.
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert!(chunk_ids(&on).contains(&id(&support)));
}

#[tokio::test]
async fn equally_anchored_supports_break_ties_by_record_id_across_runs() {
    let corpus = corpus(8).await;
    let first = lexical_note(&corpus.repo, "Kestrel canary detail alpha").await;
    let second = lexical_note(&corpus.repo, "Kestrel canary detail beta").await;
    accept(&corpus.repo, &corpus.anchor, &second, EdgeType::DerivedFrom).await;
    accept(&corpus.repo, &corpus.anchor, &first, EdgeType::DerivedFrom).await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());
    let mut expected = vec![id(&first), id(&second)];
    expected.sort();

    let mut observed = Vec::new();
    for _ in 0..3 {
        let window = search
            .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        let supports = window
            .hits
            .iter()
            .filter(|hit| expected.contains(&hit.id))
            .collect::<Vec<_>>();
        assert_eq!(supports.len(), 2);
        assert_eq!(supports[0].score.to_bits(), supports[1].score.to_bits());
        observed.push(
            supports
                .iter()
                .map(|hit| hit.id.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert!(observed.iter().all(|order| order == &expected));
}

#[tokio::test]
async fn uncorroborated_bridge_does_not_carry_anchor_strength_to_a_second_hop() {
    let corpus = corpus(8).await;
    let bridge = lexical_note(&corpus.repo, "generic bridge note with other material").await;
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    accept(&corpus.repo, &corpus.anchor, &bridge, EdgeType::RelatedTo).await;
    accept(&corpus.repo, &bridge, &support, EdgeType::DerivedFrom).await;
    let graph = GraphRetrievalConfig {
        max_hops: 2,
        ..Default::default()
    };
    let search = agent(&corpus.repo, graph);

    let window = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::On)
        .await
        .unwrap();
    assert!(window
        .hits
        .iter()
        .all(|hit| hit.id != id(&support) && hit.id != id(&bridge)));
    assert_eq!(window.summary.candidates_rank_limited, 2);
}

#[test]
fn anchored_strength_reaches_only_corroborated_neighbors() {
    let (decay, confidence) = (0.8, 0.5);
    let lexical = 0.4 * decay * confidence;
    // An uncorroborated neighbor keeps the lexical chain even from a strong
    // retrieved anchor.
    assert_eq!(
        graph_transition_strengths(0.4, 1.4, decay, confidence, false),
        (lexical, lexical)
    );
    // A corroborated neighbor inherits the stronger of the two chains.
    assert_eq!(
        graph_transition_strengths(0.4, 1.4, decay, confidence, true),
        (lexical, 1.4 * decay * confidence)
    );
    assert_eq!(
        graph_transition_strengths(0.9, 0.2, decay, confidence, true),
        (0.9 * decay * confidence, 0.9 * decay * confidence)
    );
}
