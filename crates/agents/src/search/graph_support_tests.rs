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

/// An accepted edge that no person reviewed: Gardener's automatic policy
/// accepts a `related_to` similarity proposal (`is_manual = false`).
async fn accept_unreviewed(repo: &Repository, from: &Note, to: &Note) {
    repo.upsert_gardener_proposal(
        from.id.as_ref().unwrap(),
        to.id.as_ref().unwrap(),
        1.0,
        "fixture similarity".into(),
        None,
        None,
    )
    .await
    .unwrap();
    let accepted = repo
        .accept_gardener_proposals_above(0.5, Some("fixture automatic policy".into()))
        .await
        .unwrap();
    assert_eq!(accepted, 1);
}

/// A direct match in both channels (vector rank 2, full-text rank 1) that is
/// slightly weaker than the anchor (vector rank 1, full-text rank 3).
async fn direct_competitor(repo: &Repository) -> Note {
    repo.create_note(
        Note::new("Kestrel boundary canary rollout evidence report")
            .with_embedding(near_query(0.01, 900)),
    )
    .await
    .unwrap()
}

fn excluded(context: &AugmentContext, note: &Note) -> crate::RetrievalExplanation {
    context
        .exclusions
        .iter()
        .find(|explanation| explanation.result_id == id(note))
        .cloned()
        .expect("the candidate reached packing")
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
async fn reviewed_support_competes_at_its_anchor_strength() {
    // #216: a reviewed edge passes the retrieved anchor's strength on without
    // hop decay, so the corroborated support competes with the anchor's
    // weaker direct competitors at the same strength as the anchor. Nothing
    // is reserved: the competitor still reaches packing and loses slot two on
    // MMR, and at three chunks both are packed.
    let corpus = corpus(12).await;
    let competitor = direct_competitor(&corpus.repo).await;
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

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    assert_eq!(chunk_ids(&off), vec![id(&corpus.anchor), id(&competitor)]);
    let two = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&two), vec![id(&corpus.anchor), id(&support)]);
    assert_eq!(two.chunks.len(), off.chunks.len());
    assert!(two.total_tokens <= 1000);
    assert_eq!(two.diagnostics.graph_candidates_added, 1);
    assert_eq!(two.diagnostics.graph_candidates_rank_limited, 0);
    let (anchor_chunk, support_chunk) = (&two.chunks[0], &two.chunks[1]);
    let graph = support_chunk.graph.as_ref().unwrap();
    assert_contiguous_path(
        graph,
        &id(&support),
        &id(&corpus.anchor),
        "derived_from",
        "outbound",
    );
    // The anchor's strength, not a decayed fraction of it, and an explained
    // decay that matches the score.
    assert_eq!(graph.decay, 1.0);
    assert!((support_chunk.score - anchor_chunk.score).abs() <= anchor_chunk.score * 1e-5);
    assert_eq!(
        excluded(&two, &competitor).inclusion,
        crate::InclusionReason::DiagnosticLimit
    );

    let three = context(&search, fixed_budget(3), GraphMode::On).await;
    assert_eq!(
        chunk_ids(&three),
        vec![id(&corpus.anchor), id(&support), id(&competitor)]
    );
    assert_eq!(three.diagnostics.graph_candidates_added, 1);
}

#[tokio::test]
async fn direct_match_stronger_than_the_anchor_keeps_its_slot() {
    // Lexical-only notes covering more query terms push the anchor down the
    // full-text channel, so the competitor now outranks the anchor itself.
    // The undecayed support can reach its anchor's strength but never takes
    // a slot from a direct match stronger than that anchor.
    let corpus = corpus(12).await;
    let competitor = direct_competitor(&corpus.repo).await;
    for index in 0..2 {
        lexical_note(
            &corpus.repo,
            &format!("Kestrel boundary canary evidence ledger {index}"),
        )
        .await;
    }
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

    let window = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::On)
        .await
        .unwrap();
    let position = |note: &Note| {
        window
            .hits
            .iter()
            .position(|hit| hit.id == id(note))
            .unwrap()
    };
    assert_eq!(position(&competitor), 0);
    assert_eq!(position(&corpus.anchor), 1);
    assert!(window.hits[position(&support)].score <= window.hits[0].score);

    let two = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&two)[0], id(&competitor));
    assert_eq!(chunk_ids(&two), vec![id(&competitor), id(&corpus.anchor)]);
    assert_eq!(two.diagnostics.graph_candidates_added, 0);
    assert_eq!(
        excluded(&two, &support).inclusion,
        crate::InclusionReason::DiagnosticLimit
    );
}

#[tokio::test]
async fn unreviewed_neighbor_keeps_the_hop_decay() {
    // The #186 contract still holds for an edge no person reviewed: the
    // decayed neighbor ranks below its anchor and a direct match keeps its
    // slot at two chunks; at three chunks the neighbor is added.
    let corpus = corpus(12).await;
    let competitor = direct_competitor(&corpus.repo).await;
    let support = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    accept_unreviewed(&corpus.repo, &corpus.anchor, &support).await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    let two = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&two), vec![id(&corpus.anchor), id(&competitor)]);
    assert_eq!(two.diagnostics.graph_candidates_added, 0);
    let explanation = excluded(&two, &support);
    assert_eq!(
        explanation.inclusion,
        crate::InclusionReason::DiagnosticLimit
    );
    let graph = explanation.graph.expect("graph-reached support");
    assert_eq!(graph.path[0].edge_type, "related_to");
    assert_eq!(graph.decay, 0.8);

    let three = context(&search, fixed_budget(3), GraphMode::On).await;
    assert_eq!(
        chunk_ids(&three),
        vec![id(&corpus.anchor), id(&competitor), id(&support)]
    );
    assert_eq!(three.diagnostics.graph_candidates_added, 1);
}

#[tokio::test]
async fn derivative_of_a_retrieved_source_keeps_the_hop_decay() {
    // Duplicate-aware: a note derived from the retrieved anchor restates it.
    // Reaching it is an inbound derived_from step, which keeps the decay even
    // over a reviewed edge, so it does not displace the direct match.
    let corpus = corpus(12).await;
    let competitor = direct_competitor(&corpus.repo).await;
    let derivative = lexical_note(
        &corpus.repo,
        "Kestrel canary detail with the exact evidence",
    )
    .await;
    accept(
        &corpus.repo,
        &derivative,
        &corpus.anchor,
        EdgeType::DerivedFrom,
    )
    .await;
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());

    let two = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(chunk_ids(&two), vec![id(&corpus.anchor), id(&competitor)]);
    assert_eq!(two.diagnostics.graph_candidates_added, 0);
    let graph = excluded(&two, &derivative)
        .graph
        .expect("graph-reached derivative");
    assert_contiguous_path(
        &graph,
        &id(&derivative),
        &id(&corpus.anchor),
        "derived_from",
        "inbound",
    );
    assert_eq!(graph.decay, 0.8);
}

#[tokio::test]
async fn reviewed_lift_is_one_hop_from_a_retrieved_note() {
    // The undecayed hop starts only at a retrieved seed. The bridge competes
    // at the anchor's strength; the next reviewed hop decays as before.
    let corpus = corpus(8).await;
    let bridge = lexical_note(&corpus.repo, "Kestrel canary bridge ledger").await;
    let far = lexical_note(&corpus.repo, "Kestrel rollout far evidence").await;
    accept(&corpus.repo, &corpus.anchor, &bridge, EdgeType::Supports).await;
    accept(&corpus.repo, &bridge, &far, EdgeType::Supports).await;
    let graph = GraphRetrievalConfig {
        max_hops: 2,
        ..Default::default()
    };
    let search = agent(&corpus.repo, graph);

    let window = search
        .search_with_scope_graph(QUERY, 8, SearchScope::Notes, None, None, GraphMode::On)
        .await
        .unwrap();
    let hit = |note: &Note| {
        window
            .hits
            .iter()
            .find(|hit| hit.id == id(note))
            .expect("graph-reached note in the window")
    };
    let (anchor, bridge, far) = (hit(&corpus.anchor), hit(&bridge), hit(&far));
    let bridge_graph = bridge.graph.as_ref().unwrap();
    assert_eq!((bridge_graph.hops, bridge_graph.decay), (1, 1.0));
    assert!((bridge.score - anchor.score).abs() <= anchor.score * 1e-5);
    let far_graph = far.graph.as_ref().unwrap();
    assert_eq!((far_graph.hops, far_graph.decay), (2, 0.8));
    assert!((far.score - 0.8 * bridge.score).abs() <= bridge.score * 1e-5);
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
    let strengths = |anchor: f32, reviewed: Option<f32>, corroborated: bool| {
        let strength =
            graph_transition_strengths(0.4, anchor, reviewed, decay, confidence, corroborated);
        (strength.lexical, strength.score, strength.chain)
    };
    // An uncorroborated neighbor keeps the lexical chain even from a strong
    // retrieved anchor, reviewed edge or not.
    assert_eq!(
        strengths(1.4, None, false),
        (lexical, lexical, PathChain::Lexical)
    );
    assert_eq!(
        strengths(1.4, Some(1.4), false),
        (lexical, lexical, PathChain::Lexical)
    );
    // A corroborated neighbor inherits the stronger of the two chains.
    assert_eq!(
        strengths(1.4, None, true),
        (lexical, 1.4 * decay * confidence, PathChain::Anchored)
    );
    let strong = graph_transition_strengths(0.9, 0.2, None, decay, confidence, true);
    assert_eq!(
        (strong.lexical, strong.score, strong.chain),
        (
            0.9 * decay * confidence,
            0.9 * decay * confidence,
            PathChain::Lexical
        )
    );
    // Over a reviewed first hop the anchor's own strength passes undecayed
    // (#216), still times confidence; the lexical chain always decays.
    assert_eq!(
        strengths(1.4, Some(1.4), true),
        (lexical, 1.4 * confidence, PathChain::Reviewed)
    );
}

#[test]
fn reviewed_hops_exclude_unreviewed_edges_and_steps_toward_a_derivative() {
    let edge = |edge_type: &str, is_manual: bool| NoteEdgeRow {
        id: RecordId::new(edge_type, "fixture"),
        edge_type: edge_type.into(),
        in_id: RecordId::new("note", "derived"),
        out_id: RecordId::new("note", "source"),
        proposal_id: None,
        confidence: Some(1.0),
        reason: None,
        provenance: Some("manual".into()),
        is_manual,
        created_at: chrono::Utc::now(),
        evidence_assessment: None,
    };
    for edge_type in ["supports", "contradicts", "related_to", "derived_from"] {
        assert!(
            reviewed_anchor_hop(&edge(edge_type, true), "outbound"),
            "{edge_type}"
        );
        assert!(
            !reviewed_anchor_hop(&edge(edge_type, false), "outbound"),
            "{edge_type}"
        );
        assert!(
            !reviewed_anchor_hop(&edge(edge_type, false), "inbound"),
            "{edge_type}"
        );
    }
    for edge_type in ["supports", "contradicts", "related_to"] {
        assert!(
            reviewed_anchor_hop(&edge(edge_type, true), "inbound"),
            "{edge_type}"
        );
    }
    // Toward a note derived from the retrieved one: a duplicate path.
    assert!(!reviewed_anchor_hop(&edge("derived_from", true), "inbound"));
}

#[test]
fn reviewed_strength_never_calibrates_above_its_anchor() {
    let fusions = [
        FusionConfig::default(),
        FusionConfig {
            strategy: fusion::FusionStrategy::Weighted,
            ..FusionConfig::default()
        },
    ];
    for fusion in &fusions {
        for rank in 1..=200_u16 {
            let score = 0.7 / (60.0 + f32::from(rank)) + 0.3 / (61.0 + f32::from(rank % 7));
            let strength = graph_strength_at_most(score, fusion);
            let calibrated = calibrated_graph_score(strength, fusion);
            // At most the anchor's score, and only rounding steps below it.
            assert!(calibrated <= score, "{score} {calibrated}");
            assert!(
                score - calibrated <= score * 4.0 * f32::EPSILON,
                "{score} {calibrated}"
            );
        }
    }
}

/// Packed chunk IDs, score bits and tokens: a byte-stable context summary.
fn packed(context: &AugmentContext) -> (Vec<(String, u32)>, usize) {
    (
        context
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk.score.to_bits()))
            .collect(),
        context.total_tokens,
    )
}

/// #216 hub negative: a query entity mentioned by many notes contributes
/// only weak hop-0 mention seeds. None of them enters the fixed budget,
/// graph-off is unchanged, and the result is deterministic.
///
/// Known limitation, recorded rather than fixed here: such a hub fills the
/// `max_seed_notes` frontier before hybrid anchors are added, so a reviewed
/// neighbor of a retrieved anchor is not traversed while the hub is matched.
#[tokio::test]
async fn hub_entity_mention_seeds_stay_out_of_the_fixed_budget() {
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
    let off_before = context(&search, fixed_budget(2), GraphMode::Off).await;
    let mut hub_notes = Vec::new();
    for index in 0..40 {
        let note = lexical_note(
            &corpus.repo,
            &format!("release train checklist item {index}"),
        )
        .await;
        link_entity(&corpus.repo, &note, "Canary").await;
        hub_notes.push(id(&note));
    }

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    assert_eq!(packed(&off), packed(&off_before));
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert!(on.total_tokens <= 1000);
    assert!(on.chunks.iter().all(|chunk| !hub_notes.contains(&chunk.id)));
    assert_eq!(on.chunks[0].id, id(&corpus.anchor));
    assert_eq!(on.diagnostics.graph_candidates_added, 0);
    assert!(
        on.diagnostics.graph_candidates_considered
            <= GraphRetrievalConfig::default().max_seed_notes + 1
    );
    let on_again = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(packed(&on_again), packed(&on));
}

/// #216 duplicate negative: many same-spelling entity records (one per
/// extraction) for the query term all mention near-copies of one note. They
/// share the bounded entity page with the reviewed support's entity and
/// never displace it or enter the budget.
#[tokio::test]
async fn duplicate_entity_records_do_not_crowd_out_reviewed_support() {
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
    let mut copies = Vec::new();
    for index in 0..30 {
        let copy = lexical_note(&corpus.repo, "boundary rollout copy of a checklist").await;
        let mut entity = Entity::new("Boundary", EntityType::Project);
        entity.identity_key = Some(format!("extracted-v1:fixture-boundary-{index:02}"));
        entity.metadata = serde_json::json!({});
        let entity = corpus.repo.upsert_entity(entity).await.unwrap();
        corpus
            .repo
            .link_note_to_entity(copy.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        copies.push(id(&copy));
    }
    let search = agent(&corpus.repo, GraphRetrievalConfig::default());
    let (anchor_id, support_id) = (id(&corpus.anchor), id(&support));

    let off = context(&search, fixed_budget(2), GraphMode::Off).await;
    let on = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(
        chunk_ids(&off),
        vec![anchor_id.clone(), id(&corpus.fillers[0])]
    );
    assert_eq!(chunk_ids(&on), vec![anchor_id, support_id]);
    assert!(on.chunks.iter().all(|chunk| !copies.contains(&chunk.id)));
    assert!(on.total_tokens <= 1000);
    // Deterministic across repeated runs, including the tie-heavy entity page.
    let again = context(&search, fixed_budget(2), GraphMode::On).await;
    assert_eq!(packed(&again), packed(&on));
}
