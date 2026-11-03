//! Fictional, deterministic phase expiry and cancellation regressions.
use super::*;
use crate::DeterministicEmbedder;
use graphrag_core::{EdgeType, EntityType, Note};
use graphrag_db::init_memory;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::Notify;

async fn fixture() -> SearchAgent {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.record_embedding_metadata(
        &EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
        None,
    )
    .await
    .unwrap();
    let mut seed = Note::new("Atlas deployment procedure");
    seed.title = Some("Atlas".into());
    seed.embedding = vec![1.0; 1024];
    let seed = repo.create_note(seed).await.unwrap();
    let target = repo
        .create_note(Note::new("Independent verification evidence"))
        .await
        .unwrap();
    let mut entity = Entity::new("Atlas", EntityType::Project);
    entity.metadata = json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(seed.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
        .await
        .unwrap();
    repo.create_edge(
        seed.id.as_ref().unwrap(),
        target.id.as_ref().unwrap(),
        EdgeType::Supports,
        Some(1.0),
    )
    .await
    .unwrap();
    SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
}

async fn search(agent: &SearchAgent, mode: GraphMode) -> GraphSearchResults {
    agent
        .search_with_scope_graph("Atlas", 10, SearchScope::Notes, None, None, mode)
        .await
        .unwrap()
}

fn snapshot(result: &GraphSearchResults) -> Value {
    json!(result
        .hits
        .iter()
        .map(|hit| json!({
            "id": hit.id, "score": hit.score, "explanation": hit.explanation(),
        }))
        .collect::<Vec<_>>())
}

/// Graph budget when the hook sits after entity matching. The pending hook
/// still decides where expiry happens; this only lets the earlier real phases
/// finish first on slow runners, well inside the 3 s concurrency timeout.
const SLOW_RUNNER_BUDGET_MS: u64 = 1_000;

fn slow(agent: &mut SearchAgent, phase: Phase) -> (Arc<Notify>, Arc<AtomicUsize>) {
    let entered = Arc::new(Notify::new());
    let active = Arc::new(AtomicUsize::new(0));
    agent.graph_delay = Some((phase, entered.clone(), active.clone()));
    // The hook remains pending, so expiry is deterministic without relying on
    // database size, CPU speed or a real provider. The budget covers the whole
    // graph future, so a later hook needs room for the real phases before it:
    // on slow CI runners 30 ms expired before hydration was ever reached.
    agent.graph.phase_budget_ms = if phase == Phase::GraphEntityMatching {
        30
    } else {
        SLOW_RUNNER_BUDGET_MS
    };
    (entered, active)
}

#[tokio::test]
async fn expiry_during_entity_matching_returns_exact_graph_off_and_drops_work() {
    expiry_at(Phase::GraphEntityMatching).await;
}

#[tokio::test]
async fn expiry_during_hydration_returns_exact_graph_off_and_drops_work() {
    expiry_at(Phase::GraphNoteHydration).await;
}

async fn expiry_at(phase: Phase) {
    let mut agent = fixture().await;
    let baseline = snapshot(&search(&agent, GraphMode::Off).await);
    let (_, active) = slow(&mut agent, phase);
    let (result, usage) = retrieval_usage::capture(search(&agent, GraphMode::On)).await;
    assert_eq!(result.graph_status, GraphStatus::TimedOut);
    assert_eq!(usage.graph_status, GraphStatus::TimedOut);
    assert_eq!(snapshot(&result), baseline);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert!(usage.phases.contains_key(phase.field()));
    assert!(!usage.phases.contains_key("graph_provenance_ms"));
    if phase == Phase::GraphEntityMatching {
        assert!(!usage.phases.contains_key("graph_note_hydration_ms"));
    }
    agent.graph_delay = None;
    agent.graph.phase_budget_ms = 0;
    assert_eq!(snapshot(&search(&agent, GraphMode::Off).await), baseline);
    let next = search(&agent, GraphMode::On).await;
    assert_eq!(next.graph_status, GraphStatus::Ok);
    assert!(next.hits.iter().any(|hit| hit.graph.is_some()));
}

#[tokio::test]
async fn concurrent_request_during_expiry_is_independent_and_not_locked() {
    let mut agent = fixture().await;
    let baseline = snapshot(&search(&agent, GraphMode::Off).await);
    let (entered, active) = slow(&mut agent, Phase::GraphNoteHydration);
    let graph = retrieval_usage::capture(search(&agent, GraphMode::On));
    let concurrent = async {
        entered.notified().await;
        assert_eq!(active.load(Ordering::SeqCst), 1);
        let (off, usage) = retrieval_usage::capture(search(&agent, GraphMode::Off)).await;
        assert_eq!(snapshot(&off), baseline);
        assert_eq!(usage.graph_status, GraphStatus::Off);
        assert!(!usage.phases.contains_key("graph_entity_matching_ms"));
    };
    let ((result, _), ()) = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        tokio::join!(graph, concurrent)
    })
    .await
    .unwrap();
    assert_eq!(result.graph_status, GraphStatus::TimedOut);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    agent.graph_delay = None;
    assert_eq!(
        search(&agent, GraphMode::On).await.graph_status,
        GraphStatus::Ok
    );
}

#[tokio::test]
async fn client_cancellation_racing_deadline_drops_graph_future_and_next_requests_work() {
    // Exercise cancellation both before and after the deadline becomes ready;
    // the transport's select drops this same in-request retrieval future.
    for cancel_after_ms in [0, 30, 40] {
        let mut agent = fixture().await;
        let baseline = snapshot(&search(&agent, GraphMode::Off).await);
        let (entered, active) = slow(&mut agent, Phase::GraphEntityMatching);
        let cancelled = async {
            entered.notified().await;
            tokio::time::sleep(std::time::Duration::from_millis(cancel_after_ms)).await;
        };
        let result = tokio::select! {
            biased;
            _ = cancelled => None,
            result = search(&agent, GraphMode::On) => Some(result),
        };
        if let Some(result) = result {
            assert_eq!(result.graph_status, GraphStatus::TimedOut);
            assert_eq!(snapshot(&result), baseline);
        }
        assert_eq!(active.load(Ordering::SeqCst), 0);
        agent.graph_delay = None;
        agent.graph.phase_budget_ms = 0;
        assert_eq!(snapshot(&search(&agent, GraphMode::Off).await), baseline);
        assert_eq!(
            search(&agent, GraphMode::On).await.graph_status,
            GraphStatus::Ok
        );
    }
}

#[tokio::test]
async fn completed_graph_and_context_packing_report_executed_phases_only() {
    let mut agent = fixture().await;
    agent.graph.phase_budget_ms = 0;
    let (result, usage) = retrieval_usage::capture(search(&agent, GraphMode::On)).await;
    assert_eq!(result.graph_status, GraphStatus::Ok);
    for field in [
        "embedding_ms",
        "lexical_ms",
        "vector_fusion_ms",
        "graph_query_candidates_ms",
        "graph_entity_matching_ms",
        "graph_mentions_ms",
        "graph_seed_evidence_ms",
        "graph_edge_expansion_ms",
        "graph_note_hydration_ms",
        "graph_provenance_ms",
    ] {
        assert!(usage.phases.contains_key(field), "{field}");
    }
    assert!(!usage.phases.contains_key("context_packing_ms"));
    let (context, usage) = retrieval_usage::capture(agent.build_augmented_context_with_graph(
        "Atlas",
        SearchScope::Notes,
        None,
        None,
        None,
        AugmentOptions::default(),
        GraphMode::On,
    ))
    .await;
    assert_eq!(context.unwrap().diagnostics.graph_status, GraphStatus::Ok);
    assert!(usage.phases.contains_key("context_packing_ms"));
    let (_, usage) = retrieval_usage::capture(agent.keyword_search_with_scope(
        "Atlas",
        10,
        SearchScope::Notes,
        None,
        None,
    ))
    .await;
    assert_eq!(
        usage.phases.keys().copied().collect::<Vec<_>>(),
        ["lexical_ms"]
    );
}

#[tokio::test]
async fn context_timeout_preserves_graph_off_chunks_and_reports_timeout() {
    let mut agent = fixture().await;
    let options = AugmentOptions::default();
    let baseline = agent
        .build_augmented_context_with_graph(
            "Atlas",
            SearchScope::Notes,
            None,
            None,
            None,
            options.clone(),
            GraphMode::Off,
        )
        .await
        .unwrap();
    slow(&mut agent, Phase::GraphNoteHydration);
    let context = agent
        .build_augmented_context_with_graph(
            "Atlas",
            SearchScope::Notes,
            None,
            None,
            None,
            options,
            GraphMode::On,
        )
        .await
        .unwrap();
    assert_eq!(context.diagnostics.graph_status, GraphStatus::TimedOut);
    assert_eq!(
        context.render_prompt_block(),
        baseline.render_prompt_block()
    );
    assert_eq!(
        context
            .chunks
            .iter()
            .map(|hit| (&hit.id, hit.score))
            .collect::<Vec<_>>(),
        baseline
            .chunks
            .iter()
            .map(|hit| (&hit.id, hit.score))
            .collect::<Vec<_>>()
    );
}
