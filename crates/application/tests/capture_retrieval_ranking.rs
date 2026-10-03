//! A saved note remains directly retrievable without rewriting its content or receipt.
use graphrag_agents::{
    DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig, SearchAgent,
    SharedEmbedder,
};
use graphrag_application::{
    ActionCancellation, ApplicationOperations, CallerIdentity, CaptureProvenance,
    EmbeddedApplication, GraphPolicy, RecordRef, RemoteApplicationOperations, RemoteCaptureRequest,
    RemoteCaptureResponse, Scope, SearchMode, SearchRequest,
};
use graphrag_core::{record_id_to_string, Entity, EntityType, Note};
use graphrag_db::{
    compatibility::EmbeddingIdentity,
    fusion::{FusionConfig, FusionStrategy},
    init_memory, Repository,
};
use std::{collections::BTreeMap, sync::Arc};

const INSTANCE: &str = "fictional-direct-note-client";

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[axis] = 1.0;
    vector
}

fn application(repo: &Repository, embedding: SharedEmbedder) -> EmbeddedApplication {
    application_with_fusion(repo, embedding, FusionStrategy::ReciprocalRank)
}

fn application_with_fusion(
    repo: &Repository,
    embedding: SharedEmbedder,
    strategy: FusionStrategy,
) -> EmbeddedApplication {
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()).with_fusion_config(
            FusionConfig {
                strategy,
                ..Default::default()
            },
            1.0,
            1.0,
            1.0,
        ),
        embedding,
        Arc::new(FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}

fn caller() -> CallerIdentity {
    CallerIdentity {
        instance_id: INSTANCE.into(),
    }
}

fn capture(request_id: &str, title: &str, body: &str) -> RemoteCaptureRequest {
    RemoteCaptureRequest {
        request_id: request_id.into(),
        content: body.into(),
        title: Some(title.into()),
        tags: vec!["synthetic-retrieval".into()],
        provenance: Some(CaptureProvenance {
            uri: None,
            label: Some("Fictional direct capture".into()),
            metadata: BTreeMap::from([("interface".into(), "direct-command-test".into())]),
        }),
    }
}

fn search(query: &str, mode: SearchMode, graph: GraphPolicy) -> SearchRequest {
    SearchRequest {
        query: query.into(),
        mode,
        scope: Scope::Notes,
        limit: 5,
        graph,
        since_days: None,
        source_uri: None,
    }
}

async fn assert_capture_unchanged(
    app: &EmbeddedApplication,
    repo: &Repository,
    request: &RemoteCaptureRequest,
    saved: &RemoteCaptureResponse,
    note_count: i64,
) {
    let inspected = app
        .inspect(
            RecordRef {
                id: saved.record.id.clone(),
                revision: Some(saved.record.revision.clone()),
            },
            0,
        )
        .await
        .unwrap();
    assert_eq!(inspected.id, saved.record.id);
    assert_eq!(inspected.revision, saved.record.revision);
    assert_eq!(inspected.title, request.title);
    assert_eq!(inspected.content, request.content);
    assert_eq!(
        serde_json::to_value(&inspected.provenance).unwrap(),
        saved.record.provenance
    );
    assert_eq!(inspected.provenance.instance_id.as_deref(), Some(INSTANCE));
    let replay = app
        .capture_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.request_id, saved.request_id);
    assert_eq!(replay.record, saved.record);
    assert_eq!(repo.get_stats().await.unwrap().note_count, note_count);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 1);
}

#[tokio::test]
async fn exact_manual_title_survives_body_distractors_and_small_result_limits() {
    const TITLE: &str = "Atlas Meridian Setup";
    const BODY: &str = "Read this operational instruction before starting the rehearsal.";
    let repo = Repository::new(init_memory().await.unwrap());
    repo.record_embedding_metadata(
        &EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
        None,
    )
    .await
    .unwrap();
    // Eight stronger body/vector matches would fill a five-result response if
    // exact-title evidence were discarded before the final ranking.
    for index in 0..8 {
        let repetitions = format!("{TITLE} ").repeat(16);
        repo.create_note(
            Note::new(format!("{repetitions}Body-only distractor {index}."))
                .with_title(format!("Different heading {index}"))
                .with_embedding(unit_vector(0)),
        )
        .await
        .unwrap();
    }
    let embedding: SharedEmbedder = Arc::new(
        DeterministicEmbedder::default()
            .with_default_embedding(unit_vector(0))
            .with_embedding(BODY, false, unit_vector(1)),
    );
    let app = application(&repo, embedding);
    let request = capture("manual-title-retrieval", TITLE, BODY);
    let saved = app
        .capture_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert!(!saved.replayed);
    assert_eq!(saved.record.title.as_deref(), Some(TITLE));
    assert_eq!(saved.record.content, BODY);
    assert!(!BODY.to_lowercase().contains(&TITLE.to_lowercase()));
    let captured_before = repo.get_note(&saved.record.id).await.unwrap().unwrap();
    assert_eq!(captured_before.embedding, unit_vector(1));

    for query in [
        TITLE,
        "  Atlas Meridian Setup\t",
        "atlas meridian setup",
        "ATLAS MERIDIAN SETUP",
    ] {
        for (mode, graph) in [
            (SearchMode::Keyword, GraphPolicy::Off),
            (SearchMode::Hybrid, GraphPolicy::Off),
            (SearchMode::Hybrid, GraphPolicy::Auto),
            (SearchMode::Hybrid, GraphPolicy::On),
        ] {
            let results = app
                .search(search(query, mode, graph), ActionCancellation::new())
                .await
                .unwrap();
            assert_eq!(
                results.len(),
                5,
                "query={query:?}, mode={mode:?}, graph={graph:?}"
            );
            let first = &results[0];
            assert_eq!(
                first.id, saved.record.id,
                "query={query:?}, mode={mode:?}, graph={graph:?}"
            );
            assert_eq!(
                first.revision.as_deref(),
                Some(saved.record.revision.as_str())
            );
            assert_eq!(first.title.as_deref(), Some(TITLE));
            assert_eq!(first.content, BODY);
            assert_eq!(
                first.provenance.as_ref().unwrap().instance_id.as_deref(),
                Some(INSTANCE)
            );
        }
    }
    assert_capture_unchanged(&app, &repo, &request, &saved, 9).await;
    assert_eq!(
        repo.get_note(&saved.record.id)
            .await
            .unwrap()
            .unwrap()
            .embedding,
        captured_before.embedding
    );
}

#[tokio::test]
async fn graph_only_seeds_cannot_crowd_out_the_first_direct_hybrid_body_hit() {
    const QUERY: &str = "Atlas Meridian";
    let repo = Repository::new(init_memory().await.unwrap());
    let mut entity = Entity::new(QUERY, EntityType::Project);
    entity.metadata = serde_json::json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    let mut graph_only_ids = Vec::new();
    for index in 0..8 {
        let note = repo
            .create_note(
                Note::new(format!("Unrelated graph seed evidence number {index}."))
                    .with_title(format!("Seed heading {index}")),
            )
            .await
            .unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        graph_only_ids.push(record_id_to_string(note.id.as_ref().unwrap()));
    }
    let embedding: SharedEmbedder =
        Arc::new(DeterministicEmbedder::default().with_default_embedding(unit_vector(0)));
    let app = application(&repo, embedding.clone());
    let request = capture(
        "direct-body-with-graph-competitors",
        "A different operational heading",
        "Atlas Meridian is the direct operational body match.",
    );
    let saved = app
        .capture_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    for strategy in [FusionStrategy::ReciprocalRank, FusionStrategy::Weighted] {
        let app = application_with_fusion(&repo, embedding.clone(), strategy);
        for limit in [5, 20] {
            let mut request = search(QUERY, SearchMode::Hybrid, GraphPolicy::Off);
            request.limit = limit;
            let off = app
                .search(request, ActionCancellation::new())
                .await
                .unwrap();
            // The eight seed notes are absent from both baseline channels.
            assert_eq!(off.len(), 1, "strategy={strategy:?}, limit={limit}");
            assert_eq!(off[0].id, saved.record.id);
            for graph in [GraphPolicy::Auto, GraphPolicy::On] {
                let mut request = search(QUERY, SearchMode::Hybrid, graph);
                request.limit = limit;
                let results = app
                    .search(request, ActionCancellation::new())
                    .await
                    .unwrap();
                assert_eq!(results.len(), limit.min(9));
                assert_eq!(
                    results[0].id, saved.record.id,
                    "strategy={strategy:?}, limit={limit}, graph={graph:?}"
                );
                assert_eq!(
                    results[0].revision.as_deref(),
                    Some(saved.record.revision.as_str())
                );
                assert!(results[1..]
                    .iter()
                    .all(|hit| graph_only_ids.contains(&hit.id)));
                if limit == 20 {
                    assert!(graph_only_ids
                        .iter()
                        .all(|id| results.iter().any(|hit| &hit.id == id)));
                }
            }
        }
        assert_capture_unchanged(&app, &repo, &request, &saved, 9).await;
    }
}
