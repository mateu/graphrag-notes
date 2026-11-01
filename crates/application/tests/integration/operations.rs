//! Offline operation contracts and safe preparation/mutation boundaries.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, FixtureEntityExtractor,
    InferenceCapabilities, LibrarianAgent, LibrarianRuntimeConfig, SearchAgent, SearchScope,
    SharedEmbedder, SharedEntityExtractor,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, Note, ProposedEdgeStatus, Source, SourceType};
use graphrag_db::{init_memory, Repository};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::Notify;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("offline action embedded")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("offline action batch embedded")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("offline action checked embedding health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("offline action inspected provider metadata")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("offline action extracted")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("offline action checked extraction health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("offline action inspected extraction metadata")
    }
}

fn application(
    repo: &Repository,
    embedder: SharedEmbedder,
    extractor: SharedEntityExtractor,
    skip: bool,
) -> EmbeddedApplication {
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        extractor,
        LibrarianRuntimeConfig {
            skip_entity_extraction: skip,
            ..Default::default()
        },
    )
}
fn offline(repo: &Repository) -> EmbeddedApplication {
    application(repo, Arc::new(NoInference), Arc::new(NoInference), true)
}
fn request(query: &str) -> SearchRequest {
    SearchRequest {
        query: query.into(),
        mode: SearchMode::Keyword,
        scope: Scope::All,
        limit: 20,
        graph: GraphPolicy::Off,
        since_days: None,
        source_uri: None,
    }
}
fn capture() -> CaptureRequest {
    CaptureRequest {
        content: "A private durable capture".into(),
        title: None,
        tags: vec!["private".into()],
    }
}

#[tokio::test]
async fn offline_operations_keep_existing_order_and_never_contact_providers() {
    let repo = Repository::new(init_memory().await.unwrap());
    let left = repo
        .create_note(Note::new("workspacelexeme alpha alpha"))
        .await
        .unwrap();
    let right = repo
        .create_note(Note::new("workspacelexeme beta"))
        .await
        .unwrap();
    repo.create_source(Source::manual()).await.unwrap();
    repo.upsert_gardener_proposal(
        left.id.as_ref().unwrap(),
        right.id.as_ref().unwrap(),
        0.8,
        "Offline context".into(),
        None,
        None,
    )
    .await
    .unwrap();
    let app = offline(&repo);
    let expected = SearchAgent::new(repo.clone(), Arc::new(NoInference))
        .keyword_search_with_scope("workspacelexeme", 20, SearchScope::All, None, None)
        .await
        .unwrap();
    let actual = app
        .search(request("workspacelexeme"), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|row| (&row.id, row.score.unwrap()))
            .collect::<Vec<_>>(),
        expected
            .hits
            .iter()
            .map(|row| (&row.id, row.score))
            .collect::<Vec<_>>()
    );
    assert!(actual.iter().all(|row| row.revision.is_some()));
    let recent = app.recent(20).await.unwrap();
    assert_eq!(recent.len(), 2);
    let row = &actual[0];
    let record = app
        .inspect(
            RecordRef {
                id: row.id.clone(),
                revision: row.revision.clone(),
            },
            0,
        )
        .await
        .unwrap();
    assert_eq!(record.content, row.content);
    assert_eq!(app.source_status(20).await.unwrap().len(), 1);
    let cards = app
        .proposals(Some(ProposedEdgeStatus::Pending), 20)
        .await
        .unwrap();
    assert_eq!(cards.len(), 1);
    assert!(cards[0].accept_allowed);
    assert_eq!(
        app.proposal(&cards[0].id).await.unwrap().revision,
        cards[0].revision
    );
    assert_eq!(app.stats().await.unwrap().note_count, 2);
}

#[tokio::test]
async fn selected_revision_refuses_same_id_edit_and_preserves_current_content() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut note = repo
        .create_note(Note::new("Original selected content"))
        .await
        .unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let app = offline(&repo);
    let old = app.recent(20).await.unwrap().remove(0);
    note.content = "Content edited after selection".into();
    repo.update_note(&id, note).await.unwrap();
    let error = app
        .inspect(
            RecordRef {
                id: old.id,
                revision: old.revision,
            },
            0,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, ApplicationError::RevisionConflict(_)));
    let current = app
        .inspect(RecordRef { id, revision: None }, 0)
        .await
        .unwrap();
    assert_eq!(current.content, "Content edited after selection");
}

#[tokio::test]
async fn selected_generation_does_not_follow_refreshed_successor() {
    let repo = Repository::new(init_memory().await.unwrap());
    let uri = "file:///isolated-workspace-source.md";
    let mut source = repo
        .begin_file_import(
            SourceType::Markdown,
            "source".into(),
            uri.into(),
            "First generated content".into(),
            graphrag_core::normalized_content_hash("First generated content"),
            false,
        )
        .await
        .unwrap()
        .source;
    let first = repo
        .create_note(
            Note::new("First generated content")
                .with_source(source.id.clone().unwrap())
                .with_source_generation(source.generation),
        )
        .await
        .unwrap();
    repo.complete_file_import(&mut source).await.unwrap();
    let app = offline(&repo);
    let selected = app.recent(20).await.unwrap().remove(0);
    let mut next = repo
        .begin_file_import(
            SourceType::Markdown,
            "source".into(),
            uri.into(),
            "Replacement generated content".into(),
            graphrag_core::normalized_content_hash("Replacement generated content"),
            true,
        )
        .await
        .unwrap()
        .source;
    let successor = repo
        .create_note(
            Note::new("Replacement generated content")
                .with_source(next.id.clone().unwrap())
                .with_source_generation(next.generation),
        )
        .await
        .unwrap();
    repo.complete_file_import(&mut next).await.unwrap();
    assert_ne!(first.id, successor.id);
    assert!(matches!(
        app.inspect(
            RecordRef {
                id: selected.id,
                revision: selected.revision
            },
            0
        )
        .await,
        Err(ApplicationError::NotFound(_))
    ));
    let fresh = app.recent(20).await.unwrap();
    assert_eq!(fresh.len(), 1);
    assert_eq!(
        fresh[0].id,
        record_id_to_string(successor.id.as_ref().unwrap())
    );
}

#[tokio::test]
async fn validation_and_pre_cancelled_capture_run_before_provider_preparation() {
    let repo = Repository::new(init_memory().await.unwrap());
    let app = offline(&repo);
    for limit in [0, 201] {
        assert!(matches!(
            app.recent(limit).await,
            Err(ApplicationError::Validation(_))
        ));
        let mut search = request("query");
        search.limit = limit;
        assert!(matches!(
            app.search(search, ActionCancellation::new()).await,
            Err(ApplicationError::Validation(_))
        ));
    }
    assert!(matches!(
        app.inspect(
            RecordRef {
                id: "1".into(),
                revision: None
            },
            0
        )
        .await,
        Err(ApplicationError::Validation(_))
    ));
    assert!(matches!(
        app.inspect(
            RecordRef {
                id: "note:missing".into(),
                revision: None
            },
            21
        )
        .await,
        Err(ApplicationError::Validation(_))
    ));
    assert!(matches!(
        app.search(request(" "), ActionCancellation::new()).await,
        Err(ApplicationError::Validation(_))
    ));
    let cancel = ActionCancellation::new();
    cancel.cancel();
    assert!(matches!(
        app.capture(capture(), cancel).await,
        Err(ApplicationError::Cancelled)
    ));
    assert!(app.recent(20).await.unwrap().is_empty());
}

#[tokio::test]
async fn provider_failure_returns_to_offline_operations() {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.create_note(Note::new("workspacelexeme offline fallback"))
        .await
        .unwrap();
    let app = application(
        &repo,
        Arc::new(DeterministicEmbedder::default().unhealthy()),
        Arc::new(NoInference),
        true,
    );
    let mut hybrid = request("workspacelexeme");
    hybrid.mode = SearchMode::Hybrid;
    assert!(matches!(
        app.search(hybrid, ActionCancellation::new()).await,
        Err(ApplicationError::ProviderUnavailable(_))
    ));
    assert_eq!(
        app.search(request("workspacelexeme"), ActionCancellation::new())
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(app.recent(20).await.unwrap().len(), 1);
}

struct BlockingEmbedder {
    block: AtomicBool,
    entered: Notify,
}
#[async_trait]
impl Embedder for BlockingEmbedder {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        if self.block.swap(false, Ordering::AcqRel) {
            self.entered.notify_one();
            std::future::pending::<()>().await;
        }
        DeterministicEmbedder::default().embed(text, query).await
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        query: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        DeterministicEmbedder::default()
            .embed_batch(texts, query)
            .await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}

#[tokio::test]
async fn cancelled_blocked_embedding_creates_no_note_and_next_capture_works() {
    let repo = Repository::new(init_memory().await.unwrap());
    let embedder = Arc::new(BlockingEmbedder {
        block: AtomicBool::new(true),
        entered: Notify::new(),
    });
    let app = application(
        &repo,
        embedder.clone(),
        Arc::new(FixtureEntityExtractor::default()),
        true,
    );
    let cancel = ActionCancellation::new();
    let cancel_task = async {
        embedder.entered.notified().await;
        cancel.cancel();
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(app.capture(capture(), cancel.clone()), cancel_task)
    })
    .await
    .unwrap();
    assert!(matches!(result, Err(ApplicationError::Cancelled)));
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    let saved = app
        .capture(capture(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(saved.content, "A private durable capture");
    assert!(saved.id.is_some());
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
}

struct BlockingExtractor {
    entered: Notify,
}
#[async_trait]
impl EntityExtractor for BlockingExtractor {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        self.entered.notify_one();
        std::future::pending().await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        FixtureEntityExtractor::default().capabilities()
    }
}

#[tokio::test]
async fn cancelled_blocked_extraction_preserves_empty_note_and_entity_state() {
    let repo = Repository::new(init_memory().await.unwrap());
    let extractor = Arc::new(BlockingExtractor {
        entered: Notify::new(),
    });
    let app = application(
        &repo,
        Arc::new(DeterministicEmbedder::default()),
        extractor.clone(),
        false,
    );
    let cancel = ActionCancellation::new();
    let cancel_task = async {
        extractor.entered.notified().await;
        cancel.cancel();
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(app.capture(capture(), cancel.clone()), cancel_task)
    })
    .await
    .unwrap();
    assert!(matches!(result, Err(ApplicationError::Cancelled)));
    let stats = repo.get_stats().await.unwrap();
    assert_eq!(
        (stats.note_count, stats.entity_count, stats.mention_count),
        (0, 0, 0)
    );
    assert!(app.recent(20).await.unwrap().is_empty());
}

struct CancelAfterEmbedding(Arc<AtomicBool>);
#[async_trait]
impl Embedder for CancelAfterEmbedding {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        let result = DeterministicEmbedder::default().embed(text, query).await;
        self.0.store(true, Ordering::Release);
        result
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        unreachable!()
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}

struct CancelAfterExtraction(Arc<AtomicBool>);
#[async_trait]
impl EntityExtractor for CancelAfterExtraction {
    async fn extract(&self, text: &str) -> graphrag_agents::Result<EntityExtraction> {
        let result = FixtureEntityExtractor::default().extract(text).await;
        self.0.store(true, Ordering::Release);
        result
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        FixtureEntityExtractor::default().capabilities()
    }
}

#[tokio::test]
async fn librarian_checks_cancellation_after_embedding_before_extraction_or_atomic_create() {
    let repo = Repository::new(init_memory().await.unwrap());
    let flag = Arc::new(AtomicBool::new(false));
    // Extract must remain unreachable after the embedder asks to cancel;
    // capabilities are still needed by the existing compatibility contract.
    struct UnusedExtractor;
    #[async_trait]
    impl EntityExtractor for UnusedExtractor {
        async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
            panic!("extracted after cancellation")
        }
        async fn health(&self) -> graphrag_agents::Result<bool> {
            panic!("librarian should not preflight")
        }
        fn capabilities(&self) -> InferenceCapabilities {
            FixtureEntityExtractor::default().capabilities()
        }
    }
    let librarian = LibrarianAgent::new(
        repo.clone(),
        Arc::new(CancelAfterEmbedding(flag.clone())),
        Arc::new(UnusedExtractor),
    )
    .with_cancellation_flag(flag);
    assert!(matches!(
        librarian
            .capture_manual_note("Cancel after embedding".into(), None, vec![])
            .await,
        Err(graphrag_agents::AgentError::Cancelled)
    ));
    let stats = repo.get_stats().await.unwrap();
    assert_eq!(
        (stats.note_count, stats.entity_count, stats.mention_count),
        (0, 0, 0)
    );
}

#[tokio::test]
async fn librarian_checks_cancellation_after_extraction_before_atomic_create() {
    let repo = Repository::new(init_memory().await.unwrap());
    let flag = Arc::new(AtomicBool::new(false));
    let librarian = LibrarianAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default()),
        Arc::new(CancelAfterExtraction(flag.clone())),
    )
    .with_cancellation_flag(flag);
    assert!(matches!(
        librarian
            .capture_manual_note("Cancel after extraction".into(), None, vec![])
            .await,
        Err(graphrag_agents::AgentError::Cancelled)
    ));
    let stats = repo.get_stats().await.unwrap();
    assert_eq!(
        (stats.note_count, stats.entity_count, stats.mention_count),
        (0, 0, 0)
    );
}

async fn pending(repo: &Repository) -> ProposalCard {
    let left = repo
        .create_note(Note::new("Original shared proposal evidence"))
        .await
        .unwrap();
    let right = repo
        .create_note(Note::new("Original related evidence"))
        .await
        .unwrap();
    let proposal = repo
        .upsert_gardener_proposal(
            left.id.as_ref().unwrap(),
            right.id.as_ref().unwrap(),
            0.9,
            "Shared policy".into(),
            None,
            None,
        )
        .await
        .unwrap();
    proposal_card(repo, proposal).await.unwrap()
}
fn decision(card: &ProposalCard, action: ProposalAction) -> ProposalDecisionRequest {
    ProposalDecisionRequest {
        id: card.id.clone(),
        revision: card.revision.clone(),
        action,
        reason: Some("Explicit test review".into()),
        confirmed: true,
        reviewer: "isolated reviewer".into(),
    }
}

#[tokio::test]
async fn shared_proposal_policy_requires_confirmation_and_refuses_stale_endpoint() {
    let repo = Repository::new(init_memory().await.unwrap());
    let shown = pending(&repo).await;
    let mut unconfirmed = decision(&shown, ProposalAction::Accept);
    unconfirmed.confirmed = false;
    assert!(matches!(
        decide_proposal(&repo, unconfirmed).await,
        Err(ApplicationError::Validation(_))
    ));
    let mut note = repo.get_note(&shown.from.id).await.unwrap().unwrap();
    note.content = "Changed after card display".into();
    repo.update_note(&shown.from.id, note).await.unwrap();
    assert!(matches!(
        decide_proposal(&repo, decision(&shown, ProposalAction::Accept)).await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    assert_eq!(repo.get_stats().await.unwrap().edge_count, 0);
    assert_eq!(
        offline(&repo).proposal(&shown.id).await.unwrap().status,
        ProposedEdgeStatus::Pending
    );
}

#[tokio::test]
async fn shared_proposal_policy_preserves_manual_audit_and_recorded_edge_undo() {
    let repo = Repository::new(init_memory().await.unwrap());
    let shown = pending(&repo).await;
    let accepted = decide_proposal(&repo, decision(&shown, ProposalAction::Accept))
        .await
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    assert_eq!(accepted.acceptance_is_manual, Some(true));
    assert_eq!(accepted.reviewer.as_deref(), Some("isolated reviewer"));
    assert!(accepted.resulting_edge_id.is_some());
    assert_eq!(repo.get_stats().await.unwrap().edge_count, 1);
    let undone = decide_proposal(&repo, decision(&accepted, ProposalAction::Undo))
        .await
        .unwrap();
    assert_eq!(undone.status, ProposedEdgeStatus::Superseded);
    assert_eq!(repo.get_stats().await.unwrap().edge_count, 0);
    assert!(matches!(
        decide_proposal(&repo, decision(&undone, ProposalAction::Undo)).await,
        Err(ApplicationError::Validation(_))
    ));
}

#[tokio::test]
async fn proposal_fingerprint_matches_legacy_inbox_and_hidden_endpoint_blocks_acceptance() {
    use sha2::{Digest, Sha256};
    let repo = Repository::new(init_memory().await.unwrap());
    let shown = pending(&repo).await;
    let id = graphrag_db::parse_record_id(&shown.id, Some("proposed_edge")).unwrap();
    let proposal = repo.get_edge_proposal(&id).await.unwrap().unwrap();
    let legacy = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(&proposal, &shown.from.revision, &shown.to.revision,)).unwrap()
        )
    );
    assert_eq!(shown.revision, legacy);
    let mut note = repo.get_note(&shown.from.id).await.unwrap().unwrap();
    let source = repo.create_source(Source::manual()).await.unwrap();
    note.source_id = source.id;
    note.source_generation = Some(42);
    repo.update_note(&shown.from.id, note).await.unwrap();
    let hidden = offline(&repo).proposal(&shown.id).await.unwrap();
    assert!(!hidden.from.available);
    assert!(!hidden.accept_allowed);
    assert!(matches!(
        decide_proposal(&repo, decision(&hidden, ProposalAction::Accept)).await,
        Err(ApplicationError::Validation(_))
    ));
    assert_eq!(repo.get_stats().await.unwrap().edge_count, 0);
}
