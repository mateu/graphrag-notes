//! Explicit policy migration retains last-good visibility until graph promotion.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, EntityExtraction, EntityExtractor, ExtractedEntity,
    FixtureEntityExtractor, InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_db::{init_memory, parse_record_id, Repository, PORTABLE_TABLES};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use surrealdb::types::ToSql;
use tokio::sync::Notify;
use tracing::Instrument;

fn caller() -> CallerIdentity {
    CallerIdentity {
        instance_id: "fixture-owner".into(),
    }
}
struct Extraction {
    policy: &'static str,
    calls: Arc<AtomicUsize>,
    fail_at: Arc<AtomicUsize>,
}
#[async_trait]
impl EntityExtractor for Extraction {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        let count = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if count == self.fail_at.load(Ordering::SeqCst) {
            return Err(graphrag_agents::AgentError::InferenceService(
                "fictional provider failure".into(),
            ));
        }
        Ok(EntityExtraction {
            entities: vec![ExtractedEntity {
                name: "Atlas".into(),
                entity_type: Some("project".into()),
                aliases: vec![self.policy.into()],
            }],
            relationships: Vec::new(),
        })
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(self.fail_at.load(Ordering::SeqCst) != usize::MAX)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        let mut caps = FixtureEntityExtractor::default().capabilities();
        caps.cache_identity = self.policy.into();
        caps
    }
}
fn application(
    repo: &Repository,
    policy: &'static str,
    calls: Arc<AtomicUsize>,
    fail_at: Arc<AtomicUsize>,
) -> EmbeddedApplication {
    with_extractor(
        repo,
        Arc::new(Extraction {
            policy,
            calls,
            fail_at,
        }),
    )
}
fn with_extractor(repo: &Repository, extractor: Arc<dyn EntityExtractor>) -> EmbeddedApplication {
    with_extractor_target(repo, extractor, 100)
}
fn with_extractor_target(
    repo: &Repository,
    extractor: Arc<dyn EntityExtractor>,
    target: usize,
) -> EmbeddedApplication {
    with_extractor_settings(repo, extractor, target, DeterministicEmbedder::default())
}
fn with_extractor_settings(
    repo: &Repository,
    extractor: Arc<dyn EntityExtractor>,
    target: usize,
    embedder: DeterministicEmbedder,
) -> EmbeddedApplication {
    let embedder = Arc::new(embedder);
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        extractor,
        LibrarianRuntimeConfig {
            min_chunk_size: 1,
            target_chunk_size: target,
            max_chunk_size: 150,
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}
fn request(id: &str) -> UploadSourceRequest {
    UploadSourceRequest {
        request_id: id.into(),
        document_key: "selected-part".into(),
        content: format!(
            "# Atlas first\n\n{}\n\n# Atlas second\n\n{}",
            "Alpha launch. ".repeat(20),
            "Beta launch. ".repeat(20)
        ),
        title: Some("Fictional Atlas".into()),
        provenance: Some(CaptureProvenance {
            uri: Some("file:///fictional/atlas.md".into()),
            label: Some("indexed memory".into()),
            metadata: [
                ("collection_id".into(), "fictional-memory".into()),
                ("host".into(), "fictional-host".into()),
            ]
            .into(),
        }),
        extract_entities: true,
        preserve_unchanged: false,
        create_only: false,
        expected_source_revision: None,
        policy_migration: None,
    }
}
async fn run(app: &EmbeddedApplication) -> ApplicationResult<RemoteJobExecution> {
    let execution = app
        .claim_remote_job("fixture-epoch", "fixture-worker")
        .await?
        .expect("queued job");
    app.execute_remote_job(execution.clone(), ActionCancellation::new())
        .await?;
    Ok(execution)
}
// The opt-in subscriber accepts only static probe fields and uses libtest's
// captured writer. Passing tests stay quiet; failed tests retain the boundary
// events. No engine tracing or raw error/identity fields are enabled here.
fn enable_migration_conflict_probe() {
    if std::env::var("GRAPHRAG_MIGRATION_CONFLICT_PROBE").as_deref() == Ok("1") {
        tracing_subscriber::fmt()
            .with_env_filter("graphrag_migration_probe=info")
            .with_ansi(false)
            .without_time()
            .with_test_writer()
            .try_init()
            .expect("isolated migration probe subscriber");
    }
}
fn probe_fixture_result<T>(
    operation: &'static str,
    stage: &'static str,
    result: &ApplicationResult<T>,
) {
    if std::env::var("GRAPHRAG_MIGRATION_CONFLICT_PROBE").as_deref() == Ok("1") {
        let (outcome, error_class) = match result {
            Ok(_) => ("passed", "none"),
            Err(error) => ("failed", error.code()),
        };
        tracing::info!(target: "graphrag_migration_probe", operation, stage, outcome, error_class);
    }
}
async fn run_probe(
    app: &EmbeddedApplication,
    stage: &'static str,
) -> ApplicationResult<RemoteJobExecution> {
    async {
        let claimed = app
            .claim_remote_job("fixture-epoch", "fixture-worker")
            .await;
        probe_fixture_result("fixture.claim", stage, &claimed);
        let execution = claimed?.expect("queued job");
        let executed = app
            .execute_remote_job(execution.clone(), ActionCancellation::new())
            .await;
        probe_fixture_result("fixture.execute", stage, &executed);
        executed?;
        Ok(execution)
    }
    .instrument(tracing::info_span!(target: "graphrag_migration_probe", "fixture_attempt", stage))
    .await
}
async fn raw(repo: &Repository, table: &str) -> Vec<serde_json::Value> {
    repo.portable_records_page(table, 0, 1000).await.unwrap()
}
fn migration(
    mut input: UploadSourceRequest,
    source: &UploadedSource,
    id: &str,
) -> UploadSourceRequest {
    input.request_id = id.into();
    input.expected_source_revision = Some(source.revision.clone());
    input.policy_migration = Some(SourcePolicyMigration {
        original_policy_sha256: source.processing_policy_sha256.clone(),
        target_policy_sha256: source.configured_processing_policy_sha256.clone(),
        plan_sha256: "a".repeat(64),
    });
    input
}

#[tokio::test]
async fn migration_failure_restore_and_resume_preserve_last_good_and_private_exact_batches() {
    let repo = Repository::new(init_memory().await.unwrap());
    let old = application(
        &repo,
        "legacy-policy",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let original_input = request("initial");
    let original = old
        .upload_source(caller(), original_input.clone())
        .await
        .unwrap();
    run(&old).await.unwrap();
    let source_id = parse_record_id(&original.source_id, Some("source")).unwrap();
    let chunks = repo.get_source_chunks(&source_id).await.unwrap();
    assert!(chunks.len() > 1);
    let mut manual = graphrag_core::Note::new("Fictional detached manual note");
    manual.source_id = Some(source_id.clone());
    let manual = repo.create_note(manual).await.unwrap();
    let manual_id = graphrag_core::record_id_to_string(manual.id.as_ref().unwrap());
    repo.create_edge(
        chunks[0].id.as_ref().unwrap(),
        manual.id.as_ref().unwrap(),
        graphrag_core::EdgeType::RelatedTo,
        None,
    )
    .await
    .unwrap();
    let opening_notes = raw(&repo, "note").await;
    let mut opening_inspections = Vec::new();
    for chunk in &chunks {
        opening_inspections.push(
            repo.inspect_record(
                &graphrag_core::record_id_to_string(chunk.id.as_ref().unwrap()),
                0,
            )
            .await
            .unwrap(),
        );
    }
    let opening_mentions = raw(&repo, "mentions").await;
    let opening_entities = raw(&repo, "entity").await;
    let calls = Arc::new(AtomicUsize::new(0));
    let fail_at = Arc::new(AtomicUsize::new(2));
    let current = application(&repo, "current-policy", calls.clone(), fail_at.clone());
    let before = current
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    assert!(!before.processing_policy_current);
    let input = migration(original_input.clone(), &before, "reviewed-migration");
    let admitted = current
        .upload_source(caller(), input.clone())
        .await
        .unwrap();
    assert_eq!(
        current
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .phase,
        "migration_admitted"
    );
    let result = run(&current).await;
    assert!(result.is_err());
    let failed = current
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(failed.status, "failed");
    assert_eq!(failed.phase, "migration_extracting");
    assert_eq!(failed.completed, 1);
    assert!(failed.result.is_none());
    assert_eq!(raw(&repo, "mentions").await, opening_mentions);
    assert_eq!(raw(&repo, "entity").await, opening_entities);
    let visible = repo.get_source_chunks(&source_id).await.unwrap();
    assert_eq!(
        visible.iter().map(|n| n.id.clone()).collect::<Vec<_>>(),
        chunks.iter().map(|n| n.id.clone()).collect::<Vec<_>>()
    );
    for opening in &opening_notes {
        assert!(raw(&repo, "note").await.contains(opening));
    }
    for opening in &opening_inspections {
        let inspected = repo.inspect_record(&opening.id, 0).await.unwrap();
        assert_eq!(inspected.revision, opening.revision);
        assert_eq!(inspected.content, opening.content);
        assert_eq!(inspected.provenance, opening.provenance);
    }
    assert_eq!(
        current
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap()
            .successful_generation,
        before.successful_generation
    );
    let saved = repo
        .get_remote_upload_job("fixture-owner", &admitted.job_id)
        .await
        .unwrap()
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        saved["policy_migration_stage"]["batches"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let restored = Repository::new(init_memory().await.unwrap());
    for table in PORTABLE_TABLES {
        for record in raw(&repo, table).await {
            restored
                .restore_portable_record(table, record)
                .await
                .unwrap();
        }
    }
    let restored_calls = Arc::new(AtomicUsize::new(0));
    let resumed = application(
        &restored,
        "current-policy",
        restored_calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let replay = resumed
        .upload_source(caller(), input.clone())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.job_id, admitted.job_id);
    assert_eq!(
        restored
            .get_remote_upload_job("fixture-owner", &admitted.job_id)
            .await
            .unwrap()
            .unwrap()
            .result
            .unwrap(),
        saved
    );
    let changed = application(
        &restored,
        "later-target-policy",
        restored_calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    assert!(matches!(
        changed.resume_remote_job(caller(), &admitted.job_id).await,
        Err(ApplicationError::Compatibility(_))
    ));
    assert_eq!(restored_calls.load(Ordering::SeqCst), 0);
    resumed
        .resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    run(&resumed).await.unwrap();
    assert_eq!(restored_calls.load(Ordering::SeqCst), chunks.len() - 1);
    let after = resumed
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    assert!(after.processing_policy_current);
    assert_eq!(after.generation, after.successful_generation);
    assert_eq!(after.generation, before.generation + 1);
    assert_eq!(
        after.processing_policy_sha256,
        before.configured_processing_policy_sha256
    );
    assert_eq!(
        restored
            .get_note(&manual_id)
            .await
            .unwrap()
            .unwrap()
            .content,
        manual.content
    );
    assert_eq!(raw(&restored, "related_to").await.len(), 1);
    let done = resumed
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(done.status, "completed");
    assert_eq!(done.completed as usize, chunks.len());
    assert!(done.result.unwrap().get("policy_migration_stage").is_none());
    let entities = raw(&restored, "entity").await;
    assert!(entities.iter().any(|e| {
        e["metadata"]["aliases"]
            .as_array()
            .is_some_and(|aliases| aliases.contains(&serde_json::json!("current-policy")))
    }));
    let replay = resumed.upload_source(caller(), input).await.unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.job_id, admitted.job_id);
    let mut registration = original_input;
    registration.request_id = "unchanged-registration".into();
    registration.preserve_unchanged = true;
    registration.expected_source_revision = Some(after.revision.clone());
    resumed.upload_source(caller(), registration).await.unwrap();
    let prior_calls = restored_calls.load(Ordering::SeqCst);
    run(&resumed).await.unwrap();
    assert_eq!(restored_calls.load(Ordering::SeqCst), prior_calls);
    assert_eq!(
        resumed
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap()
            .generation,
        after.generation
    );
}

#[tokio::test]
async fn migration_rejects_target_drift_and_aba_source_before_provider_calls() {
    let repo = Repository::new(init_memory().await.unwrap());
    let old = application(
        &repo,
        "legacy-policy",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let original = old
        .upload_source(caller(), request("initial"))
        .await
        .unwrap();
    run(&old).await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let current = application(
        &repo,
        "current-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let before = current
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    let input = migration(request("unused"), &before, "reviewed");
    let changed = application(
        &repo,
        "later-config-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    assert!(matches!(
        changed.upload_source(caller(), input.clone()).await,
        Err(ApplicationError::Compatibility(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let mut overwrite = request("concurrent-before-admission");
    overwrite.content.push_str("\nReviewed newer legacy edit.");
    old.upload_source(caller(), overwrite).await.unwrap();
    run(&old).await.unwrap();
    // A -> B -> A returns exactly the old content, but its later generation
    // and revision must invalidate a reviewed original-generation plan.
    old.upload_source(caller(), request("concurrent-back-to-A"))
        .await
        .unwrap();
    run(&old).await.unwrap();
    let admitted = current.upload_source(caller(), input).await.unwrap();
    let execution = current
        .claim_remote_job("fixture-epoch", "worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(execution.job_id, admitted.job_id);
    assert!(current
        .execute_remote_job(execution, ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let after = current
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    assert_eq!(after.content, request("unused").content);
    assert_eq!(after.generation, before.generation + 2);
    assert_eq!(after.successful_generation, after.generation);
    assert!(current
        .resume_remote_job(caller(), &admitted.job_id)
        .await
        .is_err());
}

struct BlockedExtraction {
    calls: AtomicUsize,
    started: Notify,
}
#[async_trait]
impl EntityExtractor for BlockedExtraction {
    async fn extract(&self, text: &str) -> graphrag_agents::Result<EntityExtraction> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            self.started.notify_one();
            std::future::pending().await
        } else {
            Extraction {
                policy: "current-policy",
                calls: Arc::new(AtomicUsize::new(0)),
                fail_at: Arc::new(AtomicUsize::new(0)),
            }
            .extract(text)
            .await
        }
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        let mut caps = FixtureEntityExtractor::default().capabilities();
        caps.cache_identity = "current-policy".into();
        caps
    }
}

#[tokio::test]
async fn cancelled_and_interrupted_migration_resume_exact_private_checkpoint() {
    for interrupt in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let old = application(
            &repo,
            "legacy-policy",
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let original = old
            .upload_source(caller(), request("original"))
            .await
            .unwrap();
        run(&old).await.unwrap();
        let source_id = parse_record_id(&original.source_id, Some("source")).unwrap();
        let chunks = repo.get_source_chunks(&source_id).await.unwrap();
        let old_mentions = raw(&repo, "mentions").await;
        let old_entities = raw(&repo, "entity").await;
        let old_notes = raw(&repo, "note").await;
        let provider = Arc::new(BlockedExtraction {
            calls: AtomicUsize::new(0),
            started: Notify::new(),
        });
        let current = Arc::new(with_extractor(&repo, provider.clone()));
        let source = current
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap();
        let input = migration(request("unused"), &source, "migration");
        let admitted = current
            .upload_source(caller(), input.clone())
            .await
            .unwrap();
        let execution = current
            .claim_remote_job("old-epoch", "worker")
            .await
            .unwrap()
            .unwrap();
        let cancellation = ActionCancellation::new();
        let flag = cancellation.clone();
        let app = current.clone();
        let executing = execution.clone();
        let task =
            tokio::spawn(async move { app.execute_remote_job(executing, cancellation).await });
        tokio::time::timeout(Duration::from_secs(15), provider.started.notified())
            .await
            .unwrap();
        let before = current
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap();
        assert_eq!(before.completed, 1);
        assert_eq!(before.phase, "migration_extracting");
        let checkpoint = repo
            .get_remote_upload_job("fixture-owner", &admitted.job_id)
            .await
            .unwrap()
            .unwrap()
            .result;
        if interrupt {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            current.reconcile_remote_jobs("new-epoch").await.unwrap();
        } else {
            current
                .cancel_remote_job(caller(), &admitted.job_id)
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
        }
        let settled = current
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap();
        assert_eq!(
            settled.status,
            if interrupt { "failed" } else { "cancelled" }
        );
        assert_eq!(settled.completed, 1);
        assert!(settled.result.is_none());
        assert_eq!(
            repo.get_remote_upload_job("fixture-owner", &admitted.job_id)
                .await
                .unwrap()
                .unwrap()
                .result,
            checkpoint
        );
        assert_eq!(raw(&repo, "mentions").await, old_mentions);
        assert_eq!(raw(&repo, "entity").await, old_entities);
        for note in &old_notes {
            assert!(raw(&repo, "note").await.contains(note));
        }
        assert_eq!(
            repo.get_source_chunks(&source_id)
                .await
                .unwrap()
                .iter()
                .map(|n| n.id.clone())
                .collect::<Vec<_>>(),
            chunks.iter().map(|n| n.id.clone()).collect::<Vec<_>>()
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let restarted = application(
            &repo,
            "current-policy",
            calls.clone(),
            Arc::new(AtomicUsize::new(0)),
        );
        restarted
            .resume_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap();
        assert!(current
            .execute_remote_job(execution, ActionCancellation::new())
            .await
            .is_err());
        run(&restarted).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), chunks.len() - 1);
        let replay = restarted.upload_source(caller(), input).await.unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.job_id, admitted.job_id);
        assert!(
            restarted
                .get_uploaded_source(&original.source_id)
                .await
                .unwrap()
                .processing_policy_current
        );
    }
}

#[tokio::test]
async fn migration_promotion_and_cleanup_faults_replay_without_reextracting_or_exposing_partial_graph(
) {
    enable_migration_conflict_probe();
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let old = application(
        &repo,
        "legacy-policy",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let original = old
        .upload_source(caller(), request("original"))
        .await
        .unwrap();
    run_probe(&old, "seed").await.unwrap();
    let source_id = parse_record_id(&original.source_id, Some("source")).unwrap();
    let chunks = repo.get_source_chunks(&source_id).await.unwrap();
    let old_id = chunks[0].id.as_ref().unwrap().clone();
    let manual = repo
        .create_note(graphrag_core::Note::new("Fictional manual endpoint"))
        .await
        .unwrap()
        .id
        .unwrap();
    let proposal = repo
        .upsert_edge_proposal(graphrag_db::repository::EdgeProposalDraft {
            from_id: old_id.clone(),
            to_id: manual.clone(),
            edge_type: graphrag_core::EdgeType::RelatedTo,
            confidence: 0.8,
            reason: "Reviewed manual relationship".into(),
            generator: "fixture".into(),
            generator_version: None,
            model: None,
        })
        .await
        .unwrap();
    let proposal_id = proposal.id.unwrap();
    let accepted = repo
        .accept_edge_proposal(&proposal_id, Some("fixture-owner".into()), None, true)
        .await
        .unwrap();
    let old_edge = accepted.resulting_edge_id.unwrap();
    let direct_manual = repo
        .create_note(graphrag_core::Note::new("Fictional direct manual endpoint"))
        .await
        .unwrap()
        .id
        .unwrap();
    repo.create_edge(
        &old_id,
        &direct_manual,
        graphrag_core::EdgeType::RelatedTo,
        None,
    )
    .await
    .unwrap();
    let direct_edge = repo
        .get_note_edges(&graphrag_core::record_id_to_string(&old_id))
        .await
        .unwrap()
        .into_iter()
        .find(|edge| edge.in_id == direct_manual || edge.out_id == direct_manual)
        .unwrap()
        .id;
    let old_mentions = raw(&repo, "mentions").await;
    let old_entities = raw(&repo, "entity").await;
    let old_notes = raw(&repo, "note").await;
    let calls = Arc::new(AtomicUsize::new(0));
    let current = application(
        &repo,
        "current-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let source = current
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    let input = migration(request("unused"), &source, "migration");
    let admitted = current
        .upload_source(caller(), input.clone())
        .await
        .unwrap();
    // First stop dependency copying before the atomic visibility transaction.
    db.query(format!(
        "DEFINE FIELD OVERWRITE in ON related_to TYPE record<note> ASSERT $value = {}",
        old_id.to_sql()
    ))
    .await
    .unwrap()
    .check()
    .unwrap();
    assert!(run_probe(&current, "dependency_copy_fault").await.is_err());
    assert_eq!(raw(&repo, "mentions").await, old_mentions);
    assert_eq!(raw(&repo, "entity").await, old_entities);
    for note in &old_notes {
        assert!(raw(&repo, "note").await.contains(note));
    }
    assert_eq!(
        current
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .completed as usize,
        chunks.len()
    );
    db.query("DEFINE FIELD OVERWRITE in ON related_to TYPE record<note>")
        .await
        .unwrap()
        .check()
        .unwrap();
    // Then reject the source's ready update after shared entity writes have
    // been attempted; the whole graph/visibility transaction must roll back.
    db.query("DEFINE FIELD OVERWRITE status ON source TYPE string ASSERT $value != 'ready'")
        .await
        .unwrap()
        .check()
        .unwrap();
    let no_provider_calls = Arc::new(AtomicUsize::new(0));
    let resumed = application(
        &repo,
        "current-policy",
        no_provider_calls.clone(),
        Arc::new(AtomicUsize::new(usize::MAX)),
    );
    resumed
        .resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert!(run_probe(&resumed, "visibility_promotion_fault")
        .await
        .is_err());
    assert_eq!(raw(&repo, "mentions").await, old_mentions);
    assert_eq!(raw(&repo, "entity").await, old_entities);
    assert_eq!(no_provider_calls.load(Ordering::SeqCst), 0);
    let failed = resumed
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(failed.phase, "migration_extracting");
    // The copy succeeded before visibility failed. A subsequent user deletion
    // of the still-live direct edge must remove its stale staged counterpart
    // on retry, rather than publishing a resurrected manual relationship.
    let staged_edges = raw(&repo, "related_to").await;
    assert_eq!(staged_edges.len(), 4);
    db.query("DELETE $edge")
        .bind(("edge", direct_edge))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_eq!(
        resumed
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap()
            .successful_generation,
        source.successful_generation
    );
    db.query("DEFINE FIELD OVERWRITE status ON source TYPE string")
        .await
        .unwrap()
        .check()
        .unwrap();
    // Promotion now commits, but accepted-proposal audit retargeting fails.
    // Its distinct phase survives restart and retries cleanup only.
    db.query(format!("DEFINE FIELD OVERWRITE resulting_edge_id ON proposed_edge TYPE option<record> ASSERT $value = {}", old_edge.to_sql())).await.unwrap().check().unwrap();
    resumed
        .resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert!(run_probe(&resumed, "proposal_audit_fault").await.is_err());
    assert_eq!(no_provider_calls.load(Ordering::SeqCst), 0);
    let promoted = resumed
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(promoted.phase, "migration_promoted");
    let after = resumed
        .get_uploaded_source(&original.source_id)
        .await
        .unwrap();
    assert_eq!(after.generation, source.generation + 1);
    assert_eq!(after.successful_generation, after.generation);
    assert!(after.processing_policy_current);
    db.query("DEFINE FIELD OVERWRITE resulting_edge_id ON proposed_edge TYPE option<record>")
        .await
        .unwrap()
        .check()
        .unwrap();
    resumed
        .resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    run_probe(&resumed, "final_promoted_cleanup").await.unwrap();
    assert_eq!(no_provider_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        resumed
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(raw(&repo, "related_to").await.len(), 1);
    assert!(repo
        .get_note_edges(&graphrag_core::record_id_to_string(&direct_manual))
        .await
        .unwrap()
        .is_empty());
    let retargeted = repo
        .get_edge_proposal(&proposal_id)
        .await
        .unwrap()
        .unwrap()
        .resulting_edge_id
        .unwrap();
    assert_ne!(retargeted, old_edge);
    assert!(repo
        .undo_edge(&retargeted, Some("after migration cleanup".into()))
        .await
        .unwrap());
    assert!(raw(&repo, "related_to").await.is_empty());
    assert!(
        resumed
            .upload_source(caller(), input)
            .await
            .unwrap()
            .replayed
    );
    assert_eq!(
        resumed
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap()
            .generation,
        after.generation
    );
}

#[tokio::test]
async fn migration_wrong_owner_origin_and_ingestion_settings_fail_before_inference() {
    for case in [
        "foreign-owner",
        "foreign-origin",
        "foreign-collection",
        "embedding-change",
        "unreviewed-target",
        "chunk-change",
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        let old = application(
            &repo,
            "legacy-policy",
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let original = old
            .upload_source(caller(), request("original"))
            .await
            .unwrap();
        run(&old).await.unwrap();
        let opening_mentions = raw(&repo, "mentions").await;
        let opening_entities = raw(&repo, "entity").await;
        let calls = Arc::new(AtomicUsize::new(0));
        let current = with_extractor_settings(
            &repo,
            Arc::new(Extraction {
                policy: "current-policy",
                calls: calls.clone(),
                fail_at: Arc::new(AtomicUsize::new(0)),
            }),
            if case == "chunk-change" { 101 } else { 100 },
            if case == "embedding-change" {
                DeterministicEmbedder::default().with_identity("different-provider", "fixture")
            } else {
                DeterministicEmbedder::default()
            },
        );
        // Chunk policy is already incompatible at source inspection, but a
        // hand-crafted reviewed intent must still fail at the owner-side gate.
        let source = current
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap();
        let mut input = migration(request("unused"), &source, "migration");
        let mut owner = caller();
        match case {
            "foreign-owner" => owner.instance_id = "foreign-owner".into(),
            "foreign-origin" => {
                input
                    .provenance
                    .as_mut()
                    .unwrap()
                    .metadata
                    .insert("host".into(), "foreign-host".into());
            }
            "foreign-collection" => {
                input
                    .provenance
                    .as_mut()
                    .unwrap()
                    .metadata
                    .insert("collection_id".into(), "foreign-collection".into());
            }
            "unreviewed-target" => {
                // Changing the reviewed target digest alone cannot rewrite the
                // configured embedding policy, even before job admission.
                input
                    .policy_migration
                    .as_mut()
                    .unwrap()
                    .target_policy_sha256 = "f".repeat(64);
            }
            "chunk-change" | "embedding-change" => {}
            _ => unreachable!(),
        }
        let admitted = current.upload_source(owner, input).await;
        if case == "unreviewed-target" {
            assert!(matches!(admitted, Err(ApplicationError::Compatibility(_))));
        } else {
            admitted.unwrap();
            assert!(run(&current).await.is_err());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let after = current
            .get_uploaded_source(&original.source_id)
            .await
            .unwrap();
        assert_eq!(after.generation, source.generation);
        assert_eq!(after.successful_generation, source.successful_generation);
        assert_eq!(raw(&repo, "mentions").await, opening_mentions);
        assert_eq!(raw(&repo, "entity").await, opening_entities);
    }
}
