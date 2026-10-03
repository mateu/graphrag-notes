use async_trait::async_trait;
use graphrag_agents::{
    AugmentOptions, DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor,
    FixtureEntityExtractor, GraphMode, InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent,
    SearchScope, SharedEmbedder, SharedEntityExtractor,
};
use graphrag_application::*;
use graphrag_core::Note;
use graphrag_db::{init_memory, Repository};
use std::collections::BTreeMap;
use std::sync::Arc;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("unexpected embedding")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("unexpected batch embedding")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected provider health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("unexpected provider identity")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("unexpected extraction")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected extraction health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("unexpected extractor identity")
    }
}

fn application(
    repo: &Repository,
    embedding: SharedEmbedder,
    extraction: SharedEntityExtractor,
) -> EmbeddedApplication {
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()),
        embedding,
        extraction,
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}
fn healthy(repo: &Repository) -> EmbeddedApplication {
    application(
        repo,
        Arc::new(DeterministicEmbedder::default()),
        Arc::new(FixtureEntityExtractor::default()),
    )
}
fn offline(repo: &Repository) -> EmbeddedApplication {
    application(repo, Arc::new(NoInference), Arc::new(NoInference))
}
fn caller(instance: &str) -> CallerIdentity {
    CallerIdentity {
        instance_id: instance.into(),
    }
}
fn capture_request() -> RemoteCaptureRequest {
    RemoteCaptureRequest {
        request_id: "a-stable-retry-id".into(),
        content: "sharedcapturelexeme launch rehearsal".into(),
        title: Some("Shared launch note".into()),
        tags: vec!["launch".into()],
        provenance: Some(CaptureProvenance {
            uri: Some("file:///client-only/notes.md".into()),
            label: Some("OpenClaw source".into()),
            metadata: BTreeMap::from([("session".into(), "fictional-session".into())]),
        }),
    }
}
fn context_request() -> BuildContextRequest {
    BuildContextRequest {
        query: "sharedcapturelexeme".into(),
        scope: Scope::Notes,
        graph: GraphPolicy::Off,
        since_days: None,
        source_uri: None,
        entity_filter: None,
        max_chunks: Some(3),
        max_total_tokens: Some(400),
        max_chunk_tokens: Some(100),
    }
}

#[tokio::test]
async fn capture_receipt_replays_offline_without_duplication_and_keeps_wire_provenance() {
    let repo = Repository::new(init_memory().await.unwrap());
    let saved = healthy(&repo)
        .capture_remote(
            caller("openclaw-a"),
            capture_request(),
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    assert!(!saved.replayed);
    let wire = serde_json::to_value(&saved).unwrap();
    assert!(wire["record"].get("embedding").is_none());
    let decoded: RemoteCaptureResponse = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded.record, saved.record);
    let replay = offline(&repo)
        .capture_remote(
            caller("openclaw-a"),
            capture_request(),
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.record, saved.record);
    // Once persistence succeeded, cancellation must not suggest that retrying
    // with a new request ID is safe: the receipt remains authoritative.
    let cancelled = ActionCancellation::new();
    cancelled.cancel();
    let replay = offline(&repo)
        .capture_remote(caller("openclaw-a"), capture_request(), cancelled)
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.record, saved.record);
    let inspected = offline(&repo)
        .inspect(
            RecordRef {
                id: saved.record.id.clone(),
                revision: Some(saved.record.revision.clone()),
            },
            0,
        )
        .await
        .unwrap();
    assert_eq!(
        inspected.provenance.instance_id.as_deref(),
        Some("openclaw-a")
    );
    assert_eq!(
        inspected.provenance.source.as_ref().unwrap()["uri"],
        "file:///client-only/notes.md"
    );
    assert!(inspected
        .provenance
        .source_uri
        .as_deref()
        .unwrap()
        .starts_with("mcp://capture/"));
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 1);
    let mut conflicting = capture_request();
    conflicting.content.push_str(" changed");
    let error = offline(&repo)
        .capture_remote(caller("openclaw-a"), conflicting, ActionCancellation::new())
        .await
        .unwrap_err();
    assert!(matches!(error, ApplicationError::RevisionConflict(_)));
    assert_eq!(error.code(), "conflict");
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
}

#[tokio::test]
async fn simultaneous_retry_is_one_atomic_capture_and_request_ids_are_instance_scoped() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = healthy(&repo);
    let (left, right) = tokio::join!(
        app.capture_remote(
            caller("openclaw-a"),
            capture_request(),
            ActionCancellation::new()
        ),
        app.capture_remote(
            caller("openclaw-a"),
            capture_request(),
            ActionCancellation::new()
        ),
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_eq!(left.record, right.record);
    assert_ne!(left.replayed, right.replayed);
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
    let independent = app
        .capture_remote(
            caller("hermes"),
            capture_request(),
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    assert!(!independent.replayed);
    assert_ne!(independent.record.id, left.record.id);
    assert_eq!(repo.get_stats().await.unwrap().note_count, 2);
}

#[tokio::test]
async fn failed_or_cancelled_remote_capture_creates_no_receipt_and_next_request_can_commit() {
    let repo = Repository::new(init_memory().await.unwrap());
    let unavailable = application(
        &repo,
        Arc::new(DeterministicEmbedder::default().unhealthy()),
        Arc::new(NoInference),
    );
    assert!(matches!(
        unavailable
            .capture_remote(
                caller("openclaw-a"),
                capture_request(),
                ActionCancellation::new()
            )
            .await,
        Err(ApplicationError::ProviderUnavailable(_))
    ));
    let cancel = ActionCancellation::new();
    cancel.cancel();
    assert!(matches!(
        offline(&repo)
            .capture_remote(caller("openclaw-a"), capture_request(), cancel)
            .await,
        Err(ApplicationError::Cancelled)
    ));
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 0);
    let saved = healthy(&repo)
        .capture_remote(
            caller("openclaw-a"),
            capture_request(),
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    assert!(!saved.replayed);
}

#[tokio::test]
async fn zero_context_budgets_and_invalid_remote_bounds_never_touch_inference() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = offline(&repo);
    for field in 0..3 {
        let mut request = context_request();
        match field {
            0 => request.max_chunks = Some(0),
            1 => request.max_total_tokens = Some(0),
            _ => request.max_chunk_tokens = Some(0),
        }
        let result = app
            .build_context(request, ActionCancellation::new())
            .await
            .unwrap();
        assert!(result.chunks.is_empty());
        assert_eq!(result.total_tokens, 0);
        assert!(result.rendered_context.is_empty());
    }
    for field in 0..3 {
        let mut request = context_request();
        match field {
            0 => request.max_chunks = Some(201),
            1 => request.max_total_tokens = Some(32769),
            _ => request.max_chunk_tokens = Some(8193),
        }
        assert!(matches!(
            app.build_context(request, ActionCancellation::new()).await,
            Err(ApplicationError::Validation(_))
        ));
    }
    let mut oversized = capture_request();
    oversized.content = "a".repeat(MAX_REMOTE_CAPTURE_BYTES + 1);
    assert!(matches!(
        app.capture_remote(caller("openclaw-a"), oversized, ActionCancellation::new())
            .await,
        Err(ApplicationError::Validation(_))
    ));
    let mut spoof = serde_json::to_value(capture_request()).unwrap();
    spoof["instance_id"] = "other-caller".into();
    assert!(serde_json::from_value::<RemoteCaptureRequest>(spoof).is_err());
}

#[tokio::test]
async fn source_uri_credentials_are_rejected_before_providers_or_persistence() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = offline(&repo);
    for uri in [
        "https://user:secret@example.test/notes",
        "https://user@example.test/notes",
        "https://example.test/notes?token=secret",
        "https://example.test/notes?%61ccess_token=secret",
        "https://example.test/notes?refreshToken=secret",
        "https://example.test/notes?CLIENT_SECRET=secret",
        "https://example.test/notes?password=secret",
        "https://example.test/notes?api-key=secret",
        "https://example.test/notes?key=secret",
        "https://example.test/notes?authorization=secret",
        "https://example.test/notes?session=secret",
        "https://example.test/notes?sessionid=secret",
        "https://example.test/notes?%73ession_id=secret",
        "https://example.test/notes?Cookie=secret",
        "https://example.test/notes?JWT=secret",
        "https://example.test/notes#access_token=secret",
        "https://example.test/notes#/oauth?token=secret",
        "https://example.test/notes#session_id=secret",
        "https://example.test/notes#/oauth?cookie=secret",
        "https://example.test/notes#jwt=secret",
    ] {
        let mut request = capture_request();
        request.provenance.as_mut().unwrap().uri = Some(uri.into());
        let error = app
            .capture_remote(caller("openclaw-a"), request, ActionCancellation::new())
            .await
            .unwrap_err();
        assert!(matches!(error, ApplicationError::Validation(_)));
        assert!(!error.to_string().contains("secret"));
    }
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 0);
    // Ordinary URL filters and client-only files remain provenance; none are
    // fetched. An unavailable provider is the next boundary for these inputs.
    let unavailable = application(
        &repo,
        Arc::new(DeterministicEmbedder::default().unhealthy()),
        Arc::new(NoInference),
    );
    for uri in [
        "file:///client-only/notes.md",
        "https://example.test/notes?q=ordinary#section-2",
    ] {
        let mut request = capture_request();
        request.provenance.as_mut().unwrap().uri = Some(uri.into());
        assert!(matches!(
            unavailable
                .capture_remote(caller("openclaw-a"), request, ActionCancellation::new())
                .await,
            Err(ApplicationError::ProviderUnavailable(_))
        ));
    }
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 0);
}

#[tokio::test]
async fn provenance_bounds_match_unicode_runtime_limits_before_inference() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut boundary = capture_request();
    boundary.provenance = Some(CaptureProvenance {
        uri: Some(format!("client:{}", "a".repeat(2041))),
        label: Some("é".repeat(512)),
        metadata: (0..32)
            .map(|index| (format!("{index:02}{}", "é".repeat(62)), "é".repeat(1024)))
            .collect(),
    });
    let unavailable = application(
        &repo,
        Arc::new(DeterministicEmbedder::default().unhealthy()),
        Arc::new(NoInference),
    );
    assert!(matches!(
        unavailable
            .capture_remote(
                caller("bounds"),
                boundary.clone(),
                ActionCancellation::new()
            )
            .await,
        Err(ApplicationError::ProviderUnavailable(_))
    ));
    let no_inference = offline(&repo);
    for field in 0..5 {
        let mut request = boundary.clone();
        let provenance = request.provenance.as_mut().unwrap();
        match field {
            0 => provenance.uri.as_mut().unwrap().push('a'),
            1 => provenance.label.as_mut().unwrap().push('é'),
            2 => {
                provenance.metadata.insert("extra".into(), "value".into());
            }
            3 => {
                provenance.metadata = BTreeMap::from([("é".repeat(65), "value".into())]);
            }
            _ => {
                provenance.metadata = BTreeMap::from([("key".into(), "é".repeat(1025))]);
            }
        }
        assert!(matches!(
            no_inference
                .capture_remote(caller("bounds"), request, ActionCancellation::new())
                .await,
            Err(ApplicationError::Validation(_))
        ));
    }
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    assert_eq!(repo.get_stats().await.unwrap().source_count, 0);
}

#[tokio::test]
async fn remote_context_preserves_existing_packing_citations_and_budget() {
    let repo = Repository::new(init_memory().await.unwrap());
    for content in [
        "sharedcapturelexeme launch rehearsal",
        "sharedcapturelexeme documentation readiness",
    ] {
        repo.create_note(Note::new(content)).await.unwrap();
    }
    let options = AugmentOptions {
        max_chunks: 3,
        max_total_tokens: 400,
        max_chunk_tokens: 100,
        ..Default::default()
    };
    let expected = SearchAgent::new(repo.clone(), Arc::new(DeterministicEmbedder::default()))
        .build_augmented_context_with_graph(
            "sharedcapturelexeme",
            SearchScope::Notes,
            None,
            None,
            None,
            options,
            GraphMode::Off,
        )
        .await
        .unwrap();
    let actual = healthy(&repo)
        .build_context(context_request(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(actual.total_tokens, expected.total_tokens);
    assert!(actual.total_tokens <= 400);
    assert_eq!(actual.rendered_context, expected.render_prompt_block());
    assert_eq!(
        actual
            .chunks
            .iter()
            .map(|chunk| (
                &chunk.id,
                &chunk.snippet,
                chunk.citation,
                chunk.rendered_tokens
            ))
            .collect::<Vec<_>>(),
        expected
            .chunks
            .iter()
            .map(|chunk| (
                &chunk.id,
                &chunk.snippet,
                chunk.citation,
                chunk.rendered_tokens
            ))
            .collect::<Vec<_>>()
    );
    assert!(actual
        .chunks
        .iter()
        .all(|chunk| chunk.id.starts_with("note:")));
    let decoded: ContextResponse =
        serde_json::from_value(serde_json::to_value(&actual).unwrap()).unwrap();
    assert_eq!(decoded.total_tokens, actual.total_tokens);
}

#[tokio::test]
async fn excessive_date_filters_fail_before_inference_or_date_arithmetic() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = offline(&repo);
    for mode in [SearchMode::Keyword, SearchMode::Hybrid] {
        let result = app
            .search(
                SearchRequest {
                    query: "synthetic".into(),
                    mode,
                    scope: Scope::All,
                    limit: 10,
                    graph: GraphPolicy::Off,
                    since_days: Some(u32::MAX),
                    source_uri: None,
                },
                ActionCancellation::new(),
            )
            .await;
        assert!(matches!(result, Err(ApplicationError::Validation(_))));
    }
    let mut request = context_request();
    request.since_days = Some(u32::MAX);
    assert!(matches!(
        app.build_context(request, ActionCancellation::new()).await,
        Err(ApplicationError::Validation(_))
    ));
}

#[tokio::test]
async fn credential_aliases_in_query_and_fragment_are_rejected_before_inference() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = offline(&repo);
    for key in [
        "secret",
        "auth",
        "bearer",
        "credential",
        "credentials",
        "private-key",
        "private_key",
        "AWSAccessKeyId",
        "X-Amz-Signature",
        "%73ecret",
        "auth.token",
    ] {
        for location in ["?", "#", "#/oauth?"] {
            let mut request = capture_request();
            request.provenance.as_mut().unwrap().uri = Some(format!(
                "https://example.test/notes{location}{key}=private-value"
            ));
            let error = app
                .capture_remote(
                    caller("credential-tests"),
                    request,
                    ActionCancellation::new(),
                )
                .await
                .unwrap_err();
            assert!(
                matches!(error, ApplicationError::Validation(_)),
                "{location}{key}"
            );
            assert!(!error.to_string().contains("private-value"));
        }
    }
    assert!(repo.list_notes(1).await.unwrap().is_empty());
}

#[test]
fn capture_fingerprint_has_a_frozen_durable_payload_version() {
    assert_eq!(REMOTE_CAPTURE_PAYLOAD_VERSION, 1);
    let fingerprint = remote_capture_fingerprint(&capture_request()).unwrap();
    // Fixture protects persisted receipt identity from unrelated API changes.
    assert_eq!(
        fingerprint,
        "34f1314d19f98e5104f1b5ab4f3170b45b4ff985dfcb5787c1dbbfbd9d3809fa"
    );
}
