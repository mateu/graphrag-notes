//! Durable jobs are executed by server authority, never by an HTTP cancellation token.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, FixtureEntityExtractor,
    InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent, SharedEmbedder,
    SharedEntityExtractor,
};
use graphrag_application::*;
use graphrag_db::{init_memory, Repository};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
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
        preserve_unchanged: false,
        create_only: false,
        expected_source_revision: None,
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

struct EndpointEmbedder {
    endpoint: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Embedder for EndpointEmbedder {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        DeterministicEmbedder::default().embed(text, query).await
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        query: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        DeterministicEmbedder::default()
            .embed_batch(texts, query)
            .await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        let mut capability = DeterministicEmbedder::default().capabilities();
        capability.endpoint = self.endpoint.clone();
        capability
    }
}

struct EndpointExtractor {
    endpoint: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl EntityExtractor for EndpointExtractor {
    async fn extract(&self, text: &str) -> graphrag_agents::Result<EntityExtraction> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        FixtureEntityExtractor::default().extract(text).await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        let mut capability = FixtureEntityExtractor::default().capabilities();
        capability.endpoint = self.endpoint.clone();
        capability
    }
}

fn endpoint_app(
    repo: &Repository,
    embedding: &str,
    extraction: &str,
) -> (EmbeddedApplication, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let embedding_calls = Arc::new(AtomicUsize::new(0));
    let extraction_calls = Arc::new(AtomicUsize::new(0));
    (
        app(
            repo,
            Arc::new(EndpointEmbedder {
                endpoint: embedding.into(),
                calls: embedding_calls.clone(),
            }),
            Arc::new(EndpointExtractor {
                endpoint: extraction.into(),
                calls: extraction_calls.clone(),
            }),
        ),
        embedding_calls,
        extraction_calls,
    )
}

#[tokio::test]
async fn changed_provider_endpoints_reject_resume_but_preserve_completed_receipt_replay() {
    const EMBEDDING: &str =
        "http://fixture-user:fixture-secret@embedding.invalid?token=fixture-token";
    const EXTRACTION: &str =
        "http://fixture-user:fixture-secret@extraction.invalid?token=fixture-token";
    for (embedding, extraction) in [
        ("http://changed-embedding.invalid", EXTRACTION),
        (EMBEDDING, "http://changed-extraction.invalid"),
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        let (original, _, _) = endpoint_app(&repo, EMBEDDING, EXTRACTION);
        let input = request("endpoint-pinned-upload");
        let admitted = original
            .upload_source(caller("owner"), input.clone())
            .await
            .unwrap();
        let cancelled = original
            .cancel_remote_job(caller("owner"), &admitted.job_id)
            .await
            .unwrap();
        assert_eq!(cancelled.status, "cancelled");
        let (changed, embedding_calls, extraction_calls) =
            endpoint_app(&repo, embedding, extraction);
        assert!(matches!(
            changed
                .resume_remote_job(caller("owner"), &admitted.job_id)
                .await,
            Err(ApplicationError::Compatibility(_))
        ));
        assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
        assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            changed
                .get_remote_job(caller("owner"), &admitted.job_id)
                .await
                .unwrap()
                .status,
            "cancelled"
        );
        let stored = repo
            .get_remote_upload_job("owner", &admitted.job_id)
            .await
            .unwrap()
            .unwrap();
        let snapshot = serde_json::to_string(&stored.input.processing_options).unwrap();
        assert!(!snapshot.contains("fixture-secret"));
        assert!(!snapshot.contains("fixture-token"));
        assert!(!snapshot.contains("embedding.invalid"));
        assert!(!snapshot.contains("extraction.invalid"));
        for role in ["embedding", "extraction"] {
            let digest = stored.input.processing_options[role]["endpoint_identity"]
                .as_str()
                .unwrap();
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        original
            .resume_remote_job(caller("owner"), &admitted.job_id)
            .await
            .unwrap();
        execute(&original, "original-endpoint").await;
        let completed = original
            .get_remote_job(caller("owner"), &admitted.job_id)
            .await
            .unwrap();
        let replay = changed.upload_source(caller("owner"), input).await.unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.job_id, admitted.job_id);
        assert_eq!(replay.source_id, admitted.source_id);
        let replay_status = changed
            .get_remote_job(caller("owner"), &replay.job_id)
            .await
            .unwrap();
        assert_eq!(replay_status.status, "completed");
        assert_eq!(replay_status.result, completed.result);
        assert_eq!(replay_status.generation, completed.generation);
        assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
        assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn queued_legacy_endpoint_snapshot_fails_before_inference_and_requires_explicit_new_input() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let (application, embedding_calls, extraction_calls) = endpoint_app(
        &repo,
        "http://embedding.invalid",
        "http://extraction.invalid",
    );
    let admitted = application
        .upload_source(caller("owner"), request("legacy-endpoint-upload"))
        .await
        .unwrap();
    let job = repo
        .get_remote_upload_job("owner", &admitted.job_id)
        .await
        .unwrap()
        .unwrap();
    let initial_phase = job.phase;
    let mut legacy = job.input.processing_options;
    for role in ["embedding", "extraction"] {
        legacy[role]
            .as_object_mut()
            .unwrap()
            .remove("endpoint_identity");
    }
    db.query("UPDATE $job SET remote_input.processing_options = $legacy")
        .bind((
            "job",
            graphrag_db::parse_record_id(&admitted.job_id, Some("processing_job")).unwrap(),
        ))
        .bind(("legacy", legacy))
        .await
        .unwrap()
        .check()
        .unwrap();
    let execution = application
        .claim_remote_job("new-service", "worker")
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        application
            .execute_remote_job(execution, ActionCancellation::new())
            .await,
        Err(ApplicationError::Compatibility(_))
    ));
    let failed = application
        .get_remote_job(caller("owner"), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(failed.status, "failed");
    assert_eq!(failed.phase, initial_phase);
    assert_eq!(failed.error_code.as_deref(), Some("compatibility"));
    assert_eq!(failed.generation, None);
    assert!(matches!(
        application
            .resume_remote_job(caller("owner"), &admitted.job_id)
            .await,
        Err(ApplicationError::Compatibility(_))
    ));
    assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
    assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
    let replay = application
        .upload_source(caller("owner"), request("legacy-endpoint-upload"))
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.job_id, admitted.job_id);
    let fresh = application
        .upload_source(caller("owner"), request("new-endpoint-upload"))
        .await
        .unwrap();
    assert_ne!(fresh.job_id, admitted.job_id);
    execute(&application, "new-service").await;
    assert_eq!(
        application
            .get_remote_job(caller("owner"), &fresh.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
}

#[tokio::test]
async fn quarantined_malformed_input_remains_readable_only_to_its_owner() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let application = healthy(&repo);
    let admitted = application
        .upload_source(caller("owner"), request("malformed-input-status"))
        .await
        .unwrap();
    db.query("UPDATE $job SET remote_input = {}")
        .bind((
            "job",
            graphrag_db::parse_record_id(&admitted.job_id, Some("processing_job")).unwrap(),
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(application
        .claim_remote_job("service", "worker")
        .await
        .unwrap()
        .is_none());
    let quarantined = application
        .get_remote_job(caller("owner"), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(quarantined.id, admitted.job_id);
    assert_eq!(quarantined.source_id, admitted.source_id);
    assert_eq!(quarantined.status, "failed");
    assert_eq!(quarantined.error_code.as_deref(), Some("validation"));
    let listed = application
        .list_remote_jobs(caller("owner"), 10)
        .await
        .unwrap();
    assert_eq!(listed.jobs.len(), 1);
    assert_eq!(listed.jobs[0].id, quarantined.id);
    assert_eq!(listed.jobs[0].error_code, quarantined.error_code);
    let cancelled = application
        .cancel_remote_job(caller("owner"), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(cancelled.status, "failed");
    assert_eq!(cancelled.error_code, quarantined.error_code);
    assert!(application
        .resume_remote_job(caller("owner"), &admitted.job_id)
        .await
        .is_err());
    assert!(matches!(
        application
            .get_remote_job(caller("foreign"), &admitted.job_id)
            .await,
        Err(ApplicationError::NotFound(_))
    ));
    assert!(matches!(
        application
            .cancel_remote_job(caller("foreign"), &admitted.job_id)
            .await,
        Err(ApplicationError::NotFound(_))
    ));
    assert!(application
        .list_remote_jobs(caller("foreign"), 10)
        .await
        .unwrap()
        .jobs
        .is_empty());
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

#[tokio::test]
async fn reviewed_policy_registration_preserves_enriched_generation_and_fences_drift() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let (application, embedding_calls, extraction_calls) = endpoint_app(
        &repo,
        "http://embedding.invalid",
        "http://extraction.invalid",
    );
    let mut initial = request("enriched-origin");
    initial.extract_entities = true;
    initial.content = "# Atlas\n\nAtlas launch is Monday. Mira leads Atlas.".into();
    let admission = application
        .upload_source(caller("owner"), initial.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    use sha2::{Digest, Sha256};
    let legacy_fingerprint = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                1_u32,
                "uploaded_markdown",
                &initial.document_key,
                &initial.content,
                &initial.title,
                &initial.provenance,
                initial.extract_entities,
            ))
            .unwrap()
        )
    );
    assert_eq!(
        repo.get_remote_upload_job("owner", &admission.job_id)
            .await
            .unwrap()
            .unwrap()
            .input
            .payload_fingerprint,
        legacy_fingerprint
    );
    let before = application
        .get_uploaded_source(&admission.source_id)
        .await
        .unwrap();
    let chunks = repo
        .get_source_chunks(&graphrag_db::parse_record_id(&before.id, Some("source")).unwrap())
        .await
        .unwrap();
    let chunk_ids: Vec<_> = chunks.iter().map(|n| n.id.clone()).collect();
    // Model the historical pilot's enrichment of already uploaded chunks.
    let mut entity = graphrag_core::Entity::new("Atlas", graphrag_core::EntityType::Project);
    entity.metadata = serde_json::json!({});
    let pilot_entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(
        chunks[0].id.as_ref().unwrap(),
        pilot_entity.id.as_ref().unwrap(),
    )
    .await
    .unwrap();
    let mut response = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    let mentions: Vec<serde_json::Value> = response.take(0).unwrap();
    assert!(!mentions.is_empty());
    assert!(before.extract_entities && before.processing_policy_current);
    assert_eq!(before.processing_policy_sha256.len(), 64);
    let lookup = application
        .lookup_uploaded_source(caller("owner"), &initial.document_key)
        .await
        .unwrap();
    assert_eq!(lookup.id, before.id);
    assert!(application
        .lookup_uploaded_source(caller("foreign"), &initial.document_key)
        .await
        .is_err());
    assert!(!application
        .lookup_uploaded_source(caller("owner"), "../invalid\0key")
        .await
        .is_ok());
    embedding_calls.store(0, Ordering::Relaxed);
    extraction_calls.store(0, Ordering::Relaxed);
    let mut registration = initial.clone();
    registration.request_id = "reviewed-register".into();
    registration.preserve_unchanged = true;
    registration
        .provenance
        .as_mut()
        .unwrap()
        .metadata
        .insert("collection_id".into(), "fixture".into());
    let registered = application
        .upload_source(caller("owner"), registration.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let after = application.get_uploaded_source(&before.id).await.unwrap();
    assert_eq!(after.generation, before.generation);
    assert_eq!(
        repo.get_source_chunks(&graphrag_db::parse_record_id(&before.id, Some("source")).unwrap())
            .await
            .unwrap()
            .iter()
            .map(|n| n.id.clone())
            .collect::<Vec<_>>(),
        chunk_ids
    );
    let mut response = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    assert_eq!(
        response.take::<Vec<serde_json::Value>>(0).unwrap(),
        mentions
    );
    assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
    assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
    assert!(
        application
            .upload_source(caller("owner"), registration.clone())
            .await
            .unwrap()
            .replayed
    );
    for (name, drift) in [
        ("body", "changed"),
        ("whitespace", "\r\n"),
        ("extraction", ""),
        ("provenance", ""),
    ] {
        let mut unsafe_registration = registration.clone();
        unsafe_registration.request_id = format!("guard-{name}");
        if name == "provenance" {
            unsafe_registration.provenance.as_mut().unwrap().uri =
                Some("file:///foreign.md".into());
        } else if name == "extraction" {
            unsafe_registration.extract_entities = false;
        } else {
            unsafe_registration.content.push_str(drift);
        }
        let job = application
            .upload_source(caller("owner"), unsafe_registration)
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
        assert_eq!(
            application
                .get_remote_job(caller("owner"), &job.job_id)
                .await
                .unwrap()
                .status,
            "failed"
        );
        let source = application.get_uploaded_source(&before.id).await.unwrap();
        assert_eq!(source.generation, before.generation);
        assert_eq!(source.content, initial.content);
    }
    // Unrelated legacy unguarded replay identities remain stable.
    assert!(
        application
            .upload_source(caller("owner"), initial)
            .await
            .unwrap()
            .replayed
    );
    assert_ne!(registered.job_id, admission.job_id);
}

#[tokio::test]
async fn create_only_negative_lookup_is_fenced_at_generation_start() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let (application, embedding_calls, extraction_calls) = endpoint_app(
        &repo,
        "http://embedding.invalid",
        "http://extraction.invalid",
    );
    // An older queued upload was invisible to get_source when the adapter
    // reviewed absence. It becomes an enriched source before the later claim.
    let mut older = request("older-pending");
    older.extract_entities = true;
    let older_admission = application
        .upload_source(caller("owner"), older.clone())
        .await
        .unwrap();
    assert!(application
        .lookup_uploaded_source(caller("owner"), &older.document_key)
        .await
        .is_err());
    let mut later = request("reviewed-absence");
    later.create_only = true;
    let later_admission = application
        .upload_source(caller("owner"), later.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let before = application
        .get_uploaded_source(&older_admission.source_id)
        .await
        .unwrap();
    let source_id = graphrag_db::parse_record_id(&before.id, Some("source")).unwrap();
    let chunks_before = repo.get_source_chunks(&source_id).await.unwrap();
    let mut entity = graphrag_core::Entity::new("Atlas", graphrag_core::EntityType::Project);
    entity.metadata = serde_json::json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(
        chunks_before[0].id.as_ref().unwrap(),
        entity.id.as_ref().unwrap(),
    )
    .await
    .unwrap();
    let mut query = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    let mentions: Vec<serde_json::Value> = query.take(0).unwrap();
    embedding_calls.store(0, Ordering::Relaxed);
    extraction_calls.store(0, Ordering::Relaxed);
    let execution = application
        .claim_remote_job("epoch", "later-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(application
        .execute_remote_job(execution, ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(
        application
            .get_remote_job(caller("owner"), &later_admission.job_id)
            .await
            .unwrap()
            .status,
        "failed"
    );
    let after = application.get_uploaded_source(&before.id).await.unwrap();
    assert_eq!(after, before);
    assert_eq!(
        repo.get_source_chunks(&source_id)
            .await
            .unwrap()
            .iter()
            .map(|n| n.id.clone())
            .collect::<Vec<_>>(),
        chunks_before
            .iter()
            .map(|n| n.id.clone())
            .collect::<Vec<_>>()
    );
    let mut query = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    assert_eq!(query.take::<Vec<serde_json::Value>>(0).unwrap(), mentions);
    assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
    assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
    assert!(
        application
            .upload_source(caller("owner"), later.clone())
            .await
            .unwrap()
            .replayed
    );
    later.request_id = "invalid-dual-guard".into();
    later.preserve_unchanged = true;
    assert!(application
        .upload_source(caller("owner"), later)
        .await
        .is_err());
    // A distinct genuinely absent source is still admitted and processed.
    let mut fresh = request("new-create-only");
    fresh.document_key = "new-document".into();
    fresh.create_only = true;
    let fresh = application
        .upload_source(caller("owner"), fresh)
        .await
        .unwrap();
    execute(&application, "epoch").await;
    assert_eq!(
        application
            .get_remote_job(caller("owner"), &fresh.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
}

#[tokio::test]
async fn expected_existing_revision_fences_changed_source_after_policy_review() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let (application, embedding_calls, extraction_calls) = endpoint_app(
        &repo,
        "http://embedding.invalid",
        "http://extraction.invalid",
    );
    let initial = application
        .upload_source(caller("owner"), request("legacy-origin"))
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let reviewed = application
        .get_uploaded_source(&initial.source_id)
        .await
        .unwrap();
    let mut older = request("older-queued-foreign-origin");
    older.extract_entities = true;
    older.content = "# Atlas\n\nAtlas unrelated enriched original.".into();
    older.provenance.as_mut().unwrap().uri = Some("file:///foreign.md".into());
    application
        .upload_source(caller("owner"), older)
        .await
        .unwrap();
    let mut changed = request("reviewed-changed-original");
    changed.content.push_str("\nNew indexed version");
    changed.expected_source_revision = Some(reviewed.revision.clone());
    let later = application
        .upload_source(caller("owner"), changed.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let before = application
        .get_uploaded_source(&initial.source_id)
        .await
        .unwrap();
    assert_ne!(before.revision, reviewed.revision);
    let source_id = graphrag_db::parse_record_id(&before.id, Some("source")).unwrap();
    let chunks = repo.get_source_chunks(&source_id).await.unwrap();
    let mut entity = graphrag_core::Entity::new("Atlas", graphrag_core::EntityType::Project);
    entity.metadata = serde_json::json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(chunks[0].id.as_ref().unwrap(), entity.id.as_ref().unwrap())
        .await
        .unwrap();
    let mut query = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    let mentions: Vec<serde_json::Value> = query.take(0).unwrap();
    embedding_calls.store(0, Ordering::Relaxed);
    extraction_calls.store(0, Ordering::Relaxed);
    let execution = application
        .claim_remote_job("epoch", "changed-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(application
        .execute_remote_job(execution, ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(
        application
            .get_remote_job(caller("owner"), &later.job_id)
            .await
            .unwrap()
            .status,
        "failed"
    );
    assert_eq!(
        application
            .get_uploaded_source(&initial.source_id)
            .await
            .unwrap(),
        before
    );
    assert_eq!(
        repo.get_source_chunks(&source_id)
            .await
            .unwrap()
            .iter()
            .map(|n| n.id.clone())
            .collect::<Vec<_>>(),
        chunks.iter().map(|n| n.id.clone()).collect::<Vec<_>>()
    );
    let mut query = db
        .query("SELECT * FROM mentions ORDER BY id")
        .await
        .unwrap();
    assert_eq!(query.take::<Vec<serde_json::Value>>(0).unwrap(), mentions);
    assert_eq!(embedding_calls.load(Ordering::Relaxed), 0);
    assert_eq!(extraction_calls.load(Ordering::Relaxed), 0);
    assert!(
        application
            .upload_source(caller("owner"), changed.clone())
            .await
            .unwrap()
            .replayed
    );
    changed.request_id = "reviewed-current-edit".into();
    changed.expected_source_revision = Some(before.revision);
    let accepted = application
        .upload_source(caller("owner"), changed)
        .await
        .unwrap();
    execute(&application, "epoch").await;
    assert_eq!(
        application
            .get_remote_job(caller("owner"), &accepted.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
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

#[tokio::test]
async fn uploaded_source_retirement_is_scoped_revision_guarded_and_replayable() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = healthy(&repo);
    let mut upload = request("retire-original");
    upload
        .provenance
        .as_mut()
        .unwrap()
        .metadata
        .insert("collection_id".into(), "fixture-memory".into());
    let first = application
        .upload_source(caller("openclaw"), upload.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let view = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    let delete = DeleteUploadedSourceRequest {
        request_id: "retire-reviewed".into(),
        id: first.source_id.clone(),
        revision: view.revision.clone(),
        collection_id: "fixture-memory".into(),
        confirmed: true,
    };
    assert!(matches!(
        application
            .delete_uploaded_source(caller("foreign"), delete.clone())
            .await,
        Err(ApplicationError::Validation(_))
    ));
    let mut wrong = delete.clone();
    wrong.collection_id = "unrelated".into();
    assert!(matches!(
        application
            .delete_uploaded_source(caller("openclaw"), wrong)
            .await,
        Err(ApplicationError::Validation(_))
    ));
    let mut stale = delete.clone();
    stale.revision = "0".repeat(64);
    assert!(matches!(
        application
            .delete_uploaded_source(caller("openclaw"), stale)
            .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    let mut unconfirmed = delete.clone();
    unconfirmed.confirmed = false;
    assert!(application
        .delete_uploaded_source(caller("openclaw"), unconfirmed)
        .await
        .is_err());

    let mut detached = graphrag_core::Note::new("Detached manual note retains its original source");
    detached.source_id = repo.get_source(&first.source_id).await.unwrap().unwrap().id;
    let detached = repo.create_note(detached).await.unwrap();
    let manual = repo
        .create_note(graphrag_core::Note::new("Unrelated manual note"))
        .await
        .unwrap();
    let result = application
        .delete_uploaded_source(caller("openclaw"), delete.clone())
        .await
        .unwrap();
    assert!(!result.replayed);
    assert_eq!(result.outcome.operation, "delete_source");
    assert_eq!(result.outcome.actor, "mcp:openclaw");
    let retained = repo.get_source(&first.source_id).await.unwrap().unwrap();
    assert_eq!(retained.successful_generation, 0);
    assert!(retained.content_hash.is_none());
    assert!(repo
        .get_source_chunks(retained.id.as_ref().unwrap())
        .await
        .unwrap()
        .is_empty());
    assert!(repo
        .get_note(&graphrag_core::record_id_to_string(
            detached.id.as_ref().unwrap()
        ))
        .await
        .unwrap()
        .is_some());
    assert!(repo
        .get_note(&graphrag_core::record_id_to_string(
            manual.id.as_ref().unwrap()
        ))
        .await
        .unwrap()
        .is_some());
    let replay = application
        .delete_uploaded_source(caller("openclaw"), delete.clone())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.outcome, result.outcome);
    let mut reused = delete;
    reused.collection_id = "other".into();
    assert!(matches!(
        application
            .delete_uploaded_source(caller("openclaw"), reused)
            .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    let retired_view = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    assert!(retired_view.retired);
    assert_ne!(retired_view.revision, view.revision);
    let mut stale_draft = upload.clone();
    stale_draft.request_id = "retired-old-draft".into();
    stale_draft.expected_source_revision = Some(view.revision.clone());
    application
        .upload_source(caller("openclaw"), stale_draft)
        .await
        .unwrap();
    let stale = application
        .claim_remote_job("epoch", "stale-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(application
        .execute_remote_job(stale, ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(
        application
            .get_uploaded_source(&first.source_id)
            .await
            .unwrap(),
        retired_view
    );
    let mut recreate = upload;
    recreate.request_id = "deliberate-recreation".into();
    recreate.expected_source_revision = Some(retired_view.revision);
    let recreated = application
        .upload_source(caller("openclaw"), recreate)
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let current = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    assert!(!current.retired);
    assert!(current.generation > view.generation);
    assert_eq!(current.generation, current.successful_generation);
    assert!(repo
        .get_note(&graphrag_core::record_id_to_string(
            detached.id.as_ref().unwrap()
        ))
        .await
        .unwrap()
        .is_some());
    assert!(!repo
        .get_source_chunks(
            &graphrag_db::parse_record_id(&recreated.source_id, Some("source")).unwrap()
        )
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn queued_upload_blocks_retirement_and_receipt_failure_rolls_it_back() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let application = healthy(&repo);
    let mut upload = request("retire-blocked");
    upload
        .provenance
        .as_mut()
        .unwrap()
        .metadata
        .insert("collection_id".into(), "fixture-memory".into());
    let first = application
        .upload_source(caller("openclaw"), upload.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let view = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    let delete = DeleteUploadedSourceRequest {
        request_id: "retire-blocked-request".into(),
        id: first.source_id.clone(),
        revision: view.revision,
        collection_id: "fixture-memory".into(),
        confirmed: true,
    };
    upload.request_id = "queued-new-generation".into();
    upload.content.push_str("\nLater text");
    let pending = application
        .upload_source(caller("openclaw"), upload)
        .await
        .unwrap();
    assert!(matches!(
        application
            .delete_uploaded_source(caller("openclaw"), delete)
            .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    application
        .cancel_remote_job(caller("openclaw"), &pending.job_id)
        .await
        .unwrap();
    let view = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    let delete = DeleteUploadedSourceRequest {
        request_id: "retire-rollback".into(),
        id: first.source_id.clone(),
        revision: view.revision,
        collection_id: "fixture-memory".into(),
        confirmed: true,
    };
    let before = repo
        .get_source_chunks(
            repo.get_source(&first.source_id)
                .await
                .unwrap()
                .unwrap()
                .id
                .as_ref()
                .unwrap(),
        )
        .await
        .unwrap();
    db.query("DEFINE FIELD OVERWRITE result ON remote_mutation_receipt TYPE object ASSERT false;")
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(application
        .delete_uploaded_source(caller("openclaw"), delete)
        .await
        .is_err());
    let after = repo
        .get_source_chunks(
            repo.get_source(&first.source_id)
                .await
                .unwrap()
                .unwrap()
                .id
                .as_ref()
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(after.len(), before.len());
    let pending = application
        .get_remote_job(caller("openclaw"), &pending.job_id)
        .await
        .unwrap();
    assert_eq!(pending.phase, "admitted");
    assert_eq!(pending.error_code, None);
    let receipts: Vec<serde_json::Value> = db
        .query("SELECT * FROM remote_mutation_receipt")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    assert!(receipts.is_empty());
}

#[tokio::test]
async fn retirement_invalidates_dormant_jobs_and_portable_restore_preserves_the_fence() {
    for terminal in ["failed", "cancelled"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let application = healthy(&repo);
        let mut original = request("dormant-original");
        original
            .provenance
            .as_mut()
            .unwrap()
            .metadata
            .insert("collection_id".into(), "fixture-memory".into());
        let first = application
            .upload_source(caller("openclaw"), original.clone())
            .await
            .unwrap();
        execute(&application, "epoch").await;
        let mut replacement = original.clone();
        replacement.request_id = "dormant-replacement".into();
        replacement.content.push_str("\nDormant replacement text");
        let dormant = application
            .upload_source(caller("openclaw"), replacement.clone())
            .await
            .unwrap();
        if terminal == "cancelled" {
            application
                .cancel_remote_job(caller("openclaw"), &dormant.job_id)
                .await
                .unwrap();
        } else {
            let execution = application
                .claim_remote_job("epoch", "fixture-worker")
                .await
                .unwrap()
                .unwrap();
            application
                .recover_remote_job(execution, "provider_unavailable".into())
                .await
                .unwrap();
        }
        let before = application
            .get_remote_job(caller("openclaw"), &dormant.job_id)
            .await
            .unwrap();
        assert_eq!(before.status, terminal);
        assert_eq!(before.generation, None);
        let view = application
            .get_uploaded_source(&first.source_id)
            .await
            .unwrap();
        let delete = DeleteUploadedSourceRequest {
            request_id: "retire-dormant".into(),
            id: first.source_id.clone(),
            revision: view.revision,
            collection_id: "fixture-memory".into(),
            confirmed: true,
        };
        let preflight = repo.remote_upload_resume_preflight_guard().await;
        let retirement_app = healthy(&repo);
        let retirement_request = delete.clone();
        let retirement = tokio::spawn(async move {
            retirement_app
                .delete_uploaded_source(caller("openclaw"), retirement_request)
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !retirement.is_finished(),
            "retirement must wait through resume status/decode/compatibility preflight"
        );
        assert_ne!(
            application
                .get_remote_job(caller("openclaw"), &dormant.job_id)
                .await
                .unwrap()
                .phase,
            "retired"
        );
        drop(preflight);
        tokio::time::timeout(Duration::from_secs(2), retirement)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(repo.get_source(&first.source_id).await.unwrap().is_none());
        let restored_db = init_memory().await.unwrap();
        let restored = Repository::new(restored_db.clone());
        for table in ["processing_job", "remote_mutation_receipt"] {
            for row in repo.portable_records_page(table, 0, 100).await.unwrap() {
                restored.restore_portable_record(table, row).await.unwrap();
            }
        }
        for candidate in [&repo, &restored] {
            let application = healthy(candidate);
            let retired = application
                .get_remote_job(caller("openclaw"), &dormant.job_id)
                .await
                .unwrap();
            assert_eq!(retired.status, terminal);
            assert_eq!(
                retired.cancellation_requested,
                before.cancellation_requested
            );
            assert_eq!(retired.phase, "retired");
            assert_eq!(retired.error_code.as_deref(), Some("source_retired"));
            assert!(matches!(
                application
                    .resume_remote_job(caller("openclaw"), &dormant.job_id)
                    .await,
                Err(ApplicationError::RevisionConflict(_))
            ));
            let (changed, embeddings, extractions) = endpoint_app(
                candidate,
                "http://changed-provider.invalid",
                "http://changed-extractor.invalid",
            );
            let error = changed
                .resume_remote_job(caller("openclaw"), &dormant.job_id)
                .await
                .unwrap_err();
            assert!(matches!(error, ApplicationError::RevisionConflict(_)));
            assert!(error.to_string().contains("retired"));
            assert_eq!(embeddings.load(Ordering::Relaxed), 0);
            assert_eq!(extractions.load(Ordering::Relaxed), 0);
            assert!(candidate
                .claim_remote_upload_job("openclaw", &dormant.job_id, "later", "worker")
                .await
                .is_err());
            assert!(application
                .claim_remote_job("later", "worker")
                .await
                .unwrap()
                .is_none());
            // Original immutable admissions and deletion receipts still replay.
            let replay = application
                .upload_source(caller("openclaw"), replacement.clone())
                .await
                .unwrap();
            assert!(replay.replayed);
            assert_eq!(replay.job_id, dormant.job_id);
            assert!(
                application
                    .delete_uploaded_source(caller("openclaw"), delete.clone())
                    .await
                    .unwrap()
                    .replayed
            );
            assert!(candidate
                .get_source(&first.source_id)
                .await
                .unwrap()
                .is_none());
        }
        // Damaged restored execution input cannot mask permanent retirement
        // with validation/repair guidance or remove owner-scoped status access.
        restored_db
            .query("UPDATE $job SET remote_input={}")
            .bind((
                "job",
                graphrag_db::parse_record_id(&dormant.job_id, Some("processing_job")).unwrap(),
            ))
            .await
            .unwrap()
            .check()
            .unwrap();
        let application = healthy(&restored);
        assert!(matches!(
            application
                .resume_remote_job(caller("openclaw"), &dormant.job_id)
                .await,
            Err(ApplicationError::RevisionConflict(_))
        ));
        assert_eq!(
            application
                .get_remote_job(caller("openclaw"), &dormant.job_id)
                .await
                .unwrap()
                .phase,
            "retired"
        );
    }
}

#[tokio::test]
async fn uploaded_source_retirement_guard_serializes_concurrent_job_admission() {
    let repo = Repository::new(init_memory().await.unwrap());
    let application = Arc::new(healthy(&repo));
    let mut upload = request("source-guard-original");
    upload
        .provenance
        .as_mut()
        .unwrap()
        .metadata
        .insert("collection_id".into(), "fixture-memory".into());
    let first = application
        .upload_source(caller("openclaw"), upload.clone())
        .await
        .unwrap();
    execute(&application, "epoch").await;
    let guard = repo.uploaded_source_mutation_guard().await;
    let view = application
        .get_uploaded_source(&first.source_id)
        .await
        .unwrap();
    let expected = repo.get_source(&first.source_id).await.unwrap().unwrap();
    upload.request_id = "concurrent-later-upload".into();
    upload.content.push_str("\nIntentional later upload");
    let later_app = application.clone();
    let later =
        tokio::spawn(async move { later_app.upload_source(caller("openclaw"), upload).await });
    // An admission started during the reviewed source snapshot cannot insert a
    // phantom queued job before the guarded retirement transaction finishes.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!later.is_finished());
    let payload = serde_json::json!({"id":view.id,"revision":view.revision,"collection_id":"fixture-memory","confirmed":true});
    let input = graphrag_db::RemoteMutationInput {
        instance_id: "openclaw".into(),
        request_id: "guarded-retirement".into(),
        operation: "delete_source".into(),
        payload_fingerprint: remote_mutation_fingerprint("delete_source", &payload).unwrap(),
        payload,
        result: serde_json::json!({"operation":"delete_source"}),
    };
    repo.apply_remote_mutation(
        &guard,
        input,
        graphrag_db::RemoteMutationEffect::DeleteSource {
            expected: Box::new(expected),
        },
    )
    .await
    .unwrap();
    assert!(repo.get_source(&first.source_id).await.unwrap().is_none());
    drop(guard);
    let admission = tokio::time::timeout(Duration::from_secs(2), later)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(admission.source_id, first.source_id);
    execute(&application, "epoch").await;
    let current = application
        .get_uploaded_source(&admission.source_id)
        .await
        .unwrap();
    assert!(current.content.ends_with("Intentional later upload"));
}
