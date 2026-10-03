//! Durable jobs are executed by server authority, never by an HTTP cancellation token.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, FixtureEntityExtractor,
    InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent, SharedEmbedder,
    SharedEntityExtractor,
};
use graphrag_application::*;
use graphrag_db::{init_memory, Repository};
use std::{sync::Arc, time::Duration};
use tokio::sync::Notify;

fn caller(instance: &str) -> CallerIdentity {
    CallerIdentity {
        instance_id: instance.into(),
    }
}
fn request(id: &str) -> UploadSourceRequest {
    UploadSourceRequest {
        request_id: id.into(),
        document_key: "daily-document".into(),
        content: "# Shared uploaded notes\n\nuploadlexeme durable workflow".into(),
        title: Some("Uploaded title".into()),
        provenance: Some(CaptureProvenance {
            uri: Some("file:///never-read/client-only.md".into()),
            label: Some("Fictional upload".into()),
            metadata: Default::default(),
        }),
        extract_entities: false,
    }
}
fn app(
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
            min_chunk_size: 1,
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}
fn healthy(repo: &Repository) -> EmbeddedApplication {
    app(
        repo,
        Arc::new(DeterministicEmbedder::default()),
        Arc::new(FixtureEntityExtractor::default()),
    )
}
async fn execute(app: &EmbeddedApplication, epoch: &str) -> RemoteJobExecution {
    let execution = app
        .claim_remote_job(epoch, "fixture-worker")
        .await
        .unwrap()
        .unwrap();
    app.execute_remote_job(execution.clone(), ActionCancellation::new())
        .await
        .unwrap();
    execution
}

#[tokio::test]
async fn admission_retry_is_stable_and_provider_failure_preserves_input_for_resume() {
    let repo = Repository::new(init_memory().await.unwrap());
    let offline = app(
        &repo,
        Arc::new(DeterministicEmbedder::default().unhealthy()),
        Arc::new(FixtureEntityExtractor::default()),
    );
    let admission = offline
        .upload_source(caller("openclaw"), request("upload-1"))
        .await
        .unwrap();
    let replay = offline
        .upload_source(caller("openclaw"), request("upload-1"))
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(admission.job_id, replay.job_id);
    assert_eq!(admission.source_id, replay.source_id);
    let mut changed = request("upload-1");
    changed.content.push_str(" changed");
    assert!(matches!(
        offline.upload_source(caller("openclaw"), changed).await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    let execution = offline
        .claim_remote_job("epoch-a", "worker-a")
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        offline
            .execute_remote_job(execution, ActionCancellation::new())
            .await,
        Err(ApplicationError::ProviderUnavailable(_))
    ));
    let failed = offline
        .get_remote_job(caller("openclaw"), &admission.job_id)
        .await
        .unwrap();
    assert_eq!(failed.status, "failed");
    assert_eq!(failed.phase, "preparing");
    assert!(offline
        .get_remote_job(caller("hermes"), &admission.job_id)
        .await
        .is_err());
    assert!(offline
        .cancel_remote_job(caller("hermes"), &admission.job_id)
        .await
        .is_err());
    let ready = healthy(&repo);
    ready
        .resume_remote_job(caller("openclaw"), &admission.job_id)
        .await
        .unwrap();
    execute(&ready, "epoch-a").await;
    let done = ready
        .get_remote_job(caller("openclaw"), &admission.job_id)
        .await
        .unwrap();
    assert_eq!(done.status, "completed");
    assert!(!done.result.as_ref().unwrap()["note_ids"]
        .as_array()
        .unwrap()
        .is_empty());
    let source = ready
        .get_uploaded_source(&admission.source_id)
        .await
        .unwrap();
    assert_eq!(source.content, request("upload-1").content);
    assert_eq!(
        source.provenance["uri"],
        "file:///never-read/client-only.md"
    );
    let id = done.result.as_ref().unwrap()["note_ids"][0]
        .as_str()
        .unwrap();
    let inspected = ready
        .inspect(
            RecordRef {
                id: id.into(),
                revision: None,
            },
            0,
        )
        .await
        .unwrap();
    assert_eq!(
        inspected.provenance.instance_id.as_deref(),
        Some("openclaw")
    );
}

#[tokio::test]
async fn document_identity_refresh_and_unchanged_runs_keep_safe_generation_boundaries() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = healthy(&repo);
    let first = application
        .upload_source(caller("openclaw"), request("version-a"))
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let old = application
        .get_remote_job(caller("openclaw"), &first.job_id)
        .await
        .unwrap();
    let old_id = old.result.unwrap()["note_ids"][0]
        .as_str()
        .unwrap()
        .to_owned();
    let same = application
        .upload_source(caller("openclaw"), request("version-b"))
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let same_status = application
        .get_remote_job(caller("openclaw"), &same.job_id)
        .await
        .unwrap();
    assert_eq!(same_status.result.unwrap()["action"], "unchanged");
    assert_eq!(first.source_id, same.source_id);
    let mut modified = request("version-c");
    modified.title = Some("Changed supplied title".into());
    modified.content.push_str("\n\nFresh remote content.");
    let changed = application
        .upload_source(caller("openclaw"), modified.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    assert_eq!(changed.source_id, first.source_id);
    let source = application
        .get_uploaded_source(&changed.source_id)
        .await
        .unwrap();
    assert_eq!(source.title, modified.title);
    assert_eq!(source.generation, 2);
    assert_eq!(source.successful_generation, 2);
    assert!(application
        .inspect(
            RecordRef {
                id: old_id,
                revision: None
            },
            0
        )
        .await
        .is_err());
    let independent = application
        .upload_source(caller("hermes"), request("version-a"))
        .await
        .unwrap();
    assert_ne!(independent.source_id, first.source_id);
}

struct BlockedEmbedding {
    started: Notify,
}
#[async_trait]
impl Embedder for BlockedEmbedding {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        self.started.notify_one();
        std::future::pending().await
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        self.started.notify_one();
        std::future::pending().await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}

#[tokio::test]
async fn explicit_cancel_of_blocked_preparation_is_resumable_without_duplicating_notes() {
    let repo = Repository::new(init_memory().await.unwrap());
    let provider = Arc::new(BlockedEmbedding {
        started: Notify::new(),
    });
    let application = Arc::new(app(
        &repo,
        provider.clone(),
        Arc::new(FixtureEntityExtractor::default()),
    ));
    let saved = application
        .upload_source(caller("openclaw"), request("cancel-work"))
        .await
        .unwrap();
    let execution = application
        .claim_remote_job("epoch", "worker")
        .await
        .unwrap()
        .unwrap();
    let cancellation = ActionCancellation::new();
    let flag = cancellation.clone();
    let clone = application.clone();
    let task = tokio::spawn(async move { clone.execute_remote_job(execution, cancellation).await });
    tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
        .await
        .unwrap();
    application
        .cancel_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    flag.cancel();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap(),
        Err(ApplicationError::Cancelled)
    ));
    assert_eq!(
        application
            .get_remote_job(caller("openclaw"), &saved.job_id)
            .await
            .unwrap()
            .status,
        "cancelled"
    );
    assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
    let restarted = healthy(&repo);
    restarted
        .resume_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    execute(&restarted, "epoch-next").await;
    assert_eq!(
        restarted
            .get_remote_job(caller("openclaw"), &saved.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
}

#[tokio::test]
async fn startup_reconciles_only_interrupted_jobs_and_old_worker_cannot_publish() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = healthy(&repo);
    let saved = application
        .upload_source(caller("openclaw"), request("restart-work"))
        .await
        .unwrap();
    let stale = application
        .claim_remote_job("old-service", "worker")
        .await
        .unwrap()
        .unwrap();
    application
        .reconcile_remote_jobs("new-service")
        .await
        .unwrap();
    let job = application
        .get_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    assert_eq!(job.status, "failed");
    assert_eq!(job.error_code.as_deref(), Some("interrupted"));
    assert!(matches!(
        application
            .execute_remote_job(stale, ActionCancellation::new())
            .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    application
        .resume_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    execute(&application, "new-service").await;
    application
        .reconcile_remote_jobs("another-service")
        .await
        .unwrap();
    assert_eq!(
        application
            .get_remote_job(caller("openclaw"), &saved.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
}

#[tokio::test]
async fn invalid_upload_shapes_and_job_kinds_are_rejected_before_admission_or_transition() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = healthy(&repo);
    for content in ["😀".repeat(17000), "embedded\0NUL".into()] {
        let mut input = request("invalid");
        input.content = content;
        assert!(matches!(
            application.upload_source(caller("openclaw"), input).await,
            Err(ApplicationError::Validation(_))
        ));
    }
    let mut spoof = serde_json::to_value(request("spoof")).unwrap();
    spoof["instance_id"] = "other".into();
    assert!(serde_json::from_value::<UploadSourceRequest>(spoof).is_err());
    assert!(application
        .get_remote_job(caller("openclaw"), "embedding:local-job")
        .await
        .is_err());
    assert!(application
        .list_remote_jobs(caller("openclaw"), 101)
        .await
        .is_err());
    assert!(application
        .list_remote_jobs(caller("openclaw"), 100)
        .await
        .unwrap()
        .jobs
        .is_empty());
}

struct FailSecondExtraction {
    calls: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl EntityExtractor for FailSecondExtraction {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            return Err(graphrag_agents::AgentError::InferenceService(
                "injected".into(),
            ));
        }
        Ok(EntityExtraction {
            entities: vec![],
            relationships: vec![],
        })
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        FixtureEntityExtractor::default().capabilities()
    }
}
#[tokio::test]
async fn extraction_failure_resumes_after_committed_item_without_reembedding_or_refresh() {
    let repo = Repository::new(init_memory().await.unwrap());
    let provider = Arc::new(FailSecondExtraction {
        calls: Default::default(),
    });
    let mut runtime = LibrarianRuntimeConfig {
        min_chunk_size: 1,
        target_chunk_size: 60,
        max_chunk_size: 80,
        ..Default::default()
    };
    runtime.skip_entity_extraction = true;
    let embedding: SharedEmbedder = Arc::new(DeterministicEmbedder::default());
    let application = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()),
        embedding.clone(),
        provider.clone(),
        runtime.clone(),
    );
    let mut input = request("extraction-work");
    input.content="# First\n\nFirst paragraph contains a durable snapshot and extraction checkpoint.\n\n# Second\n\nSecond paragraph is separately checkpointed for extraction.".into();
    input.extract_entities = true;
    let saved = application
        .upload_source(caller("openclaw"), input)
        .await
        .unwrap();
    let execution = application
        .claim_remote_job("epoch", "worker")
        .await
        .unwrap()
        .unwrap();
    assert!(application
        .execute_remote_job(execution, ActionCancellation::new())
        .await
        .is_err());
    let failed = application
        .get_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    assert_eq!(failed.status, "failed");
    assert_eq!(failed.phase, "extracting");
    assert_eq!(failed.completed, 1);
    assert!(failed.checkpoint.is_some());
    let source = application
        .get_uploaded_source(&saved.source_id)
        .await
        .unwrap();
    let resumed = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()),
        embedding,
        Arc::new(FixtureEntityExtractor::default()),
        runtime,
    );
    resumed
        .resume_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    execute(&resumed, "epoch").await;
    let done = resumed
        .get_remote_job(caller("openclaw"), &saved.job_id)
        .await
        .unwrap();
    assert_eq!(done.status, "completed");
    assert_eq!(done.completed, done.total);
    assert_eq!(
        resumed
            .get_uploaded_source(&saved.source_id)
            .await
            .unwrap()
            .generation,
        source.generation
    );
}

#[tokio::test]
async fn incompatible_resume_and_excess_chunk_input_fail_before_claim_or_inference() {
    let repo = Repository::new(init_memory().await.unwrap());
    let embedding: SharedEmbedder = Arc::new(DeterministicEmbedder::default().unhealthy());
    let extraction: SharedEntityExtractor = Arc::new(FixtureEntityExtractor::default());
    let original = app(&repo, embedding.clone(), extraction.clone());
    let admission = original
        .upload_source(caller("openclaw"), request("pinned-config"))
        .await
        .unwrap();
    let execution = original
        .claim_remote_job("epoch", "worker")
        .await
        .unwrap()
        .unwrap();
    assert!(original
        .execute_remote_job(execution, ActionCancellation::new())
        .await
        .is_err());
    let changed = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()),
        embedding.clone(),
        extraction.clone(),
        LibrarianRuntimeConfig {
            min_chunk_size: 1,
            target_chunk_size: 20,
            max_chunk_size: 40,
            skip_entity_extraction: true,
            ..Default::default()
        },
    );
    assert!(matches!(
        changed
            .resume_remote_job(caller("openclaw"), &admission.job_id)
            .await,
        Err(ApplicationError::Compatibility(_))
    ));
    assert_eq!(
        original
            .get_remote_job(caller("openclaw"), &admission.job_id)
            .await
            .unwrap()
            .status,
        "failed"
    );
    let bounded = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()),
        embedding,
        extraction,
        LibrarianRuntimeConfig {
            min_chunk_size: 1,
            target_chunk_size: 1,
            max_chunk_size: 1,
            skip_entity_extraction: true,
            ..Default::default()
        },
    );
    let mut large = request("too-many-items");
    large.content = "x".repeat(201);
    assert!(matches!(
        bounded.upload_source(caller("openclaw"), large).await,
        Err(ApplicationError::Validation(_))
    ));
    assert_eq!(
        original
            .list_remote_jobs(caller("openclaw"), 100)
            .await
            .unwrap()
            .jobs
            .len(),
        1
    );
}

#[tokio::test]
async fn unchanged_line_endings_return_latest_input_without_rewriting_chunk_spans() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = healthy(&repo);
    let mut crlf = request("crlf-original");
    crlf.content = "# Shared notes\r\n\r\nExact source span bytes remain stable.\r\n".into();
    let first = application
        .upload_source(caller("openclaw"), crlf.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let before = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    let backing = repo.get_source(&first.source_id).await.unwrap().unwrap();
    let chunks = repo
        .get_source_chunks(backing.id.as_ref().unwrap())
        .await
        .unwrap();
    assert!(!chunks.is_empty());
    assert_eq!(before.content, crlf.content);
    assert_eq!(backing.content.as_deref(), Some(crlf.content.as_str()));
    let mut lf = crlf.clone();
    lf.request_id = "lf-equivalent".into();
    lf.content = lf.content.replace("\r\n", "\n");
    lf.provenance.as_mut().unwrap().label = Some("Latest LF input".into());
    let latest = application
        .upload_source(caller("openclaw"), lf.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let after = application
        .get_uploaded_source(&latest.source_id)
        .await
        .unwrap();
    let status = application
        .get_remote_job(caller("openclaw"), &latest.job_id)
        .await
        .unwrap();
    assert_eq!(status.result.unwrap()["action"], "unchanged");
    assert_eq!(after.content, lf.content);
    assert_eq!(after.provenance["label"], "Latest LF input");
    assert_ne!(after.revision, before.revision);
    assert_eq!(after.generation, before.generation);
    assert_eq!(after.content_hash, before.content_hash);
    let saved = repo.get_source(&latest.source_id).await.unwrap().unwrap();
    assert_eq!(saved.content, backing.content);
    let stable = repo
        .get_source_chunks(saved.id.as_ref().unwrap())
        .await
        .unwrap();
    assert_eq!(stable.len(), chunks.len());
    for (original, current) in chunks.iter().zip(&stable) {
        assert_eq!(current.id, original.id);
        assert_eq!(current.content, original.content);
        assert_eq!(current.source_start_byte, original.source_start_byte);
        assert_eq!(current.source_end_byte, original.source_end_byte);
        let start = current.source_start_byte.unwrap() as usize;
        let end = current.source_end_byte.unwrap() as usize;
        assert_eq!(
            &saved.content.as_ref().unwrap()[start..end],
            &backing.content.as_ref().unwrap()[start..end]
        );
    }
}
