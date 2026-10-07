//! Explicit policy migration retains last-good visibility until graph promotion.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, EntityExtraction, EntityExtractor, ExtractedEntity,
    FixtureEntityExtractor, InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_db::{init_memory, Repository};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

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

async fn conversion_plan(
    repo: &Repository,
    app: &EmbeddedApplication,
    source_id: &str,
) -> graphrag_db::SourceEnrichmentPlan {
    let source = app.get_uploaded_source(source_id).await.unwrap();
    let job = repo
        .get_remote_upload_job(
            &caller().instance_id,
            &repo.get_source(source_id).await.unwrap().unwrap().metadata["remote_upload"]["job_id"]
                .as_str()
                .unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    graphrag_db::SourceEnrichmentPlan {
        instance_id: caller().instance_id,
        request_id: "reviewed-conversion".into(),
        source_id: source.id,
        document_key: source.document_key,
        expected_source_revision: source.revision,
        expected_content_sha256: source.content_hash.unwrap(),
        expected_generation: source.generation,
        original_policy_sha256: source.processing_policy_sha256,
        target_processing_options: job.input.processing_options,
        confirmed: true,
    }
}
async fn rows(repo: &Repository, table: &str) -> Vec<serde_json::Value> {
    repo.portable_records_page(table, 0, 1000).await.unwrap()
}
#[tokio::test]
async fn provider_failure_resumes_without_claiming_graph_freshness_and_sync_can_skip() {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicUsize::new(2));
    let app = application(&repo, "fixture-policy", calls.clone(), failure.clone());
    let mut input = request("vector-only");
    input.extract_entities = false;
    let admission = app.upload_source(caller(), input.clone()).await.unwrap();
    run(&app).await.unwrap();
    let before = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let notes = rows(&repo, "note").await;
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    assert!(app
        .enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let failed = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert!(!failed.extract_entities);
    assert_eq!(failed.extraction_policy_current, None);
    assert_eq!(failed.revision, before.revision);
    assert_eq!(failed.generation, before.generation);
    assert!(rows(&repo, "mentions").await.is_empty());
    assert_eq!(rows(&repo, "note").await, notes);
    failure.store(0, Ordering::SeqCst);
    let stage = repo
        .begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    assert_eq!(stage.completed, 1);
    let finished = app
        .enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), finished.total + 1);
    let current = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert!(current.extract_entities);
    assert_eq!(current.extraction_policy_current, Some(true));
    assert!(current.processing_policy_current);
    assert_eq!(current.generation, before.generation);
    assert_eq!(current.successful_generation, before.successful_generation);
    assert_eq!(rows(&repo, "note").await, notes);
    let unchanged_calls = calls.load(Ordering::SeqCst);
    app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), unchanged_calls);
    // Client recognition is read-only: exact unchanged sync does not need another upload.
    assert_eq!(
        current.latest_upload_request_id,
        before.latest_upload_request_id
    );
    assert!(repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .is_ok());
    let rolled_back = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert!(!rolled_back.extract_entities);
    assert_eq!(rolled_back.extraction_policy_current, None);
    assert_ne!(rolled_back.revision, before.revision);
}
#[tokio::test]
async fn owner_policy_and_cancellation_fences_prevent_extraction() {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let app = application(
        &repo,
        "fixture-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut input = request("vector-only");
    input.extract_entities = false;
    let admission = app.upload_source(caller(), input).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    let mut other = caller();
    other.instance_id = "other".into();
    assert!(app
        .enrich_uploaded_source(other, plan.clone(), ActionCancellation::new())
        .await
        .is_err());
    let mut policy_changed = plan.clone();
    policy_changed.target_processing_options["extraction"]["model"] = "different".into();
    assert!(app
        .enrich_uploaded_source(caller(), policy_changed, ActionCancellation::new())
        .await
        .is_err());
    let cancel = ActionCancellation::new();
    cancel.cancel();
    assert!(app
        .enrich_uploaded_source(caller(), plan, cancel)
        .await
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap()
        .metadata
        .get("entity_enrichment_v1")
        .is_none());
}

#[tokio::test]
async fn concurrent_owner_retry_performs_each_extraction_once() {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let app = application(
        &repo,
        "fixture-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut input = request("vector-only");
    input.extract_entities = false;
    let admission = app.upload_source(caller(), input).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    let (left, right) = tokio::join!(
        app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new()),
        app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
    );
    let left = left.unwrap();
    assert_eq!(left, right.unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), left.total);
    assert_eq!(rows(&repo, "mentions").await.len(), left.total);
}

#[tokio::test]
async fn enriched_source_replacement_stages_failed_extraction_and_requires_fresh_relationship_review(
) {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicUsize::new(0));
    let app = application(&repo, "fixture-policy", calls.clone(), failure.clone());
    let mut input = request("vector-only");
    input.extract_entities = false;
    let admission = app.upload_source(caller(), input.clone()).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    let current = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let source_id = graphrag_db::parse_record_id(&admission.source_id, Some("source")).unwrap();
    let chunks = repo.get_source_chunks(&source_id).await.unwrap();
    let detached = repo
        .create_note(graphrag_core::Note::new(
            "Fictional directly reviewed supporting quote",
        ))
        .await
        .unwrap();
    let a = graphrag_core::record_id_to_string(chunks[0].id.as_ref().unwrap());
    let b = graphrag_core::record_id_to_string(detached.id.as_ref().unwrap());
    let ra = repo.inspect_record(&a, 0).await.unwrap();
    let rb = repo.inspect_record(&b, 0).await.unwrap();
    let proposal = app
        .propose_endpoint_remote(
            caller(),
            RemoteEndpointProposalRequest {
                request_id: "before-replacement-review".into(),
                from: EndpointProposalEvidence {
                    id: a,
                    revision: ra.revision,
                    quote: ra.content.chars().take(100).collect(),
                },
                to: EndpointProposalEvidence {
                    id: b,
                    revision: rb.revision,
                    quote: rb.content,
                },
                relationship: EndpointRelationship::Supports,
                rationale: "Narrow explicit quoted support claim".into(),
                confirmed: true,
            },
        )
        .await
        .unwrap();
    let card = app.proposal(&proposal.outcome.id).await.unwrap();
    app.decide_remote(
        caller(),
        RemoteDecisionRequest {
            request_id: "before-replacement-accept".into(),
            id: card.id,
            revision: card.revision,
            action: ProposalAction::Accept,
            reason: None,
            confirmed: true,
        },
    )
    .await
    .unwrap();
    let mentions = rows(&repo, "mentions").await;
    let visible_ids = chunks.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
    let mut replacement = input.clone();
    replacement.request_id = "reviewed-replacement".into();
    replacement.extract_entities = true;
    replacement.expected_source_revision = Some(current.revision);
    replacement.content = replacement
        .content
        .replace("Alpha launch", "Alpha corrected launch");
    failure.store(calls.load(Ordering::SeqCst) + 2, Ordering::SeqCst);
    let replacement_job = app
        .upload_source(caller(), replacement.clone())
        .await
        .unwrap();
    assert!(run(&app).await.is_err());
    let failed = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert_eq!(failed.generation, 2);
    assert_eq!(failed.successful_generation, 1);
    assert_eq!(failed.extraction_policy_current, Some(false));
    assert!(!failed.processing_policy_current);
    assert_eq!(
        repo.get_source_chunks(&source_id)
            .await
            .unwrap()
            .iter()
            .map(|n| n.id.clone())
            .collect::<Vec<_>>(),
        visible_ids
    );
    assert_eq!(rows(&repo, "mentions").await, mentions);
    assert_eq!(rows(&repo, "supports").await.len(), 1);
    let interrupted = app
        .get_remote_job(caller(), &replacement_job.job_id)
        .await
        .unwrap();
    assert_eq!(interrupted.phase, "migration_extracting");
    assert_eq!(interrupted.completed, 1);
    failure.store(0, Ordering::SeqCst);
    app.resume_remote_job(caller(), &replacement_job.job_id)
        .await
        .unwrap();
    run(&app).await.unwrap();
    let after = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert_eq!(after.generation, 2);
    assert_eq!(after.successful_generation, 2);
    assert!(after.processing_policy_current);
    assert_eq!(after.extraction_policy_current, Some(true));
    assert!(rows(&repo, "supports").await.is_empty());
    assert_eq!(
        app.proposal(&proposal.outcome.id).await.unwrap().status,
        graphrag_core::ProposedEdgeStatus::Superseded
    );
    assert!(repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .is_err());
    let rebuilt = repo.get_source_chunks(&source_id).await.unwrap();
    assert!(rebuilt
        .iter()
        .all(|n| n.source_generation == Some(2) && !visible_ids.contains(&n.id)));
    assert_eq!(rows(&repo, "mentions").await.len(), rebuilt.len());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        chunks.len() + rebuilt.len() + 1
    );
    let mut third = replacement;
    third.request_id = "next-reviewed-generation".into();
    third.expected_source_revision = Some(after.revision);
    third.content.push_str("\nFurther fictional correction.");
    app.upload_source(caller(), third).await.unwrap();
    run(&app).await.unwrap();
    let latest = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert_eq!(latest.generation, 3);
    assert_eq!(latest.successful_generation, 3);
    assert_eq!(latest.extraction_policy_current, Some(true));
}

#[tokio::test]
async fn corrupt_promoted_checkpoint_never_reports_current_applied_graph_policy() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let app = application(
        &repo,
        "fixture-policy",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut input = request("vector-only-currency");
    input.extract_entities = false;
    let admission = app.upload_source(caller(), input).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(
        app.get_uploaded_source(&admission.source_id)
            .await
            .unwrap()
            .extraction_policy_current,
        Some(true)
    );
    let saved = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    for case in 0..6 {
        let mut metadata = saved.metadata.clone();
        let stage = &mut metadata["entity_enrichment_v1"];
        match case {
            0 => stage["plan"]["source_id"] = "source:transplanted".into(),
            1 => {
                stage["items"][0]["entities"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("identity_key");
            }
            2 => {
                let e = stage["items"][0]["entities"][0].clone();
                stage["items"][0]["entities"]
                    .as_array_mut()
                    .unwrap()
                    .push(e);
            }
            3 => {
                stage["items"][0]["entities"][0]["metadata"]["extraction"]["scope"] =
                    "source:other:chunk:foreign".into()
            }
            4 => stage["items"][0]["id"] = "note:foreign".into(),
            _ => {
                stage["items"].as_array_mut().unwrap().pop();
            }
        }
        db.query("UPDATE $source SET metadata=$metadata")
            .bind(("source", saved.id.clone()))
            .bind(("metadata", metadata))
            .await
            .unwrap()
            .check()
            .unwrap();
        let current = app.get_uploaded_source(&admission.source_id).await.unwrap();
        assert!(current.extract_entities); // historical applied flag, not freshness
        assert!(!current.processing_policy_current, "case {case}");
        assert_eq!(
            current.extraction_policy_current,
            Some(false),
            "case {case}"
        );
        assert!(app
            .enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
            .await
            .is_err());
    }
}

async fn transport_fixture() -> (
    Repository,
    EmbeddedApplication,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    UploadAdmission,
    ExecuteSourceEnrichment,
) {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicUsize::new(0));
    let app = application(&repo, "transport-policy", calls.clone(), failure.clone());
    let mut upload = request("transport-vector");
    upload.extract_entities = false;
    let source = app.upload_source(caller(), upload).await.unwrap();
    run(&app).await.unwrap();
    let view = app.get_uploaded_source(&source.source_id).await.unwrap();
    let receipt = app
        .prepare_source_enrichment(
            caller(),
            PrepareSourceEnrichment {
                request_id: "transport-review".into(),
                source_id: view.id,
                revision: view.revision,
            },
        )
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(repo
        .get_source(&source.source_id)
        .await
        .unwrap()
        .unwrap()
        .metadata
        .get("entity_enrichment_v1")
        .is_none());
    (
        repo,
        app,
        calls,
        failure,
        source,
        ExecuteSourceEnrichment {
            request_id: "transport-review".into(),
            reviewed: receipt,
            confirmed: true,
            rollback_source_revision: None,
        },
    )
}
#[tokio::test]
async fn reviewed_transport_owner_confirmation_digest_replay_and_atomic_rollback() {
    let (repo, app, calls, _, source, request) = transport_fixture().await;
    let before = rows(&repo, "note").await;
    let mut wrong = request.clone();
    wrong.confirmed = false;
    assert!(app
        .execute_source_enrichment(caller(), wrong, false)
        .await
        .is_err());
    let mut wrong = request.clone();
    wrong.reviewed.target_policy_sha256 = "a".repeat(64);
    assert!(app
        .execute_source_enrichment(caller(), wrong, false)
        .await
        .is_err());
    assert!(app
        .execute_source_enrichment(
            CallerIdentity {
                instance_id: "foreign".into()
            },
            request.clone(),
            false
        )
        .await
        .is_err());
    let admitted = app
        .execute_source_enrichment(caller(), request.clone(), false)
        .await
        .unwrap();
    assert!(
        app.execute_source_enrichment(caller(), request.clone(), false)
            .await
            .unwrap()
            .replayed
    );
    let mut changed = request.clone();
    changed.reviewed.plan["expected_generation"] = 2.into();
    let plan = serde_json::from_value(changed.reviewed.plan.clone()).unwrap();
    changed.reviewed = enrichment_receipt(&plan).unwrap();
    assert!(app
        .execute_source_enrichment(caller(), changed, false)
        .await
        .is_err());
    assert!(app
        .read_source_enrichment_plan(
            CallerIdentity {
                instance_id: "foreign".into()
            },
            &admitted.job_id
        )
        .await
        .is_err());
    assert_eq!(
        app.read_source_enrichment_plan(caller(), &admitted.job_id)
            .await
            .unwrap()
            .plan_sha256,
        request.reviewed.plan_sha256
    );
    run(&app).await.unwrap();
    let completed = app
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(completed.status, "completed");
    let view = app.get_uploaded_source(&source.source_id).await.unwrap();
    assert!(view.extract_entities);
    assert_eq!(view.generation, 1);
    assert_eq!(view.extraction_policy_current, Some(true));
    assert_eq!(rows(&repo, "note").await, before);
    let count = calls.load(Ordering::SeqCst);
    assert!(count > 1);
    assert!(
        app.execute_source_enrichment(caller(), request.clone(), false)
            .await
            .unwrap()
            .replayed
    );
    assert_eq!(calls.load(Ordering::SeqCst), count);
    // Cancellation after atomic publication cannot relabel committed success.
    assert_eq!(
        app.cancel_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
    let mut rollback = request.clone();
    rollback.request_id = "transport-rollback".into();
    rollback.rollback_source_revision = Some(view.revision.clone());
    let rollback_job = app
        .execute_source_enrichment(caller(), rollback.clone(), true)
        .await
        .unwrap();
    run(&app).await.unwrap();
    assert_eq!(
        app.get_remote_job(caller(), &rollback_job.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
    let after = app.get_uploaded_source(&source.source_id).await.unwrap();
    assert!(!after.extract_entities);
    assert_ne!(after.revision, view.revision);
    assert_eq!(after.generation, 1);
    assert_eq!(rows(&repo, "note").await, before);
    assert!(
        app.execute_source_enrichment(caller(), rollback, true)
            .await
            .unwrap()
            .replayed
    );
}
#[tokio::test]
async fn reviewed_transport_partial_failure_restart_requeues_checkpoint_and_fences_old_lease() {
    let (repo, app, calls, failure, source, request) = transport_fixture().await;
    failure.store(2, Ordering::SeqCst);
    let admitted = app
        .execute_source_enrichment(caller(), request, false)
        .await
        .unwrap();
    let old = app
        .claim_remote_job("old-epoch", "old-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(app
        .claim_remote_job("old-epoch", "duplicate")
        .await
        .unwrap()
        .is_none());
    assert!(app
        .execute_remote_job(old.clone(), ActionCancellation::new())
        .await
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(rows(&repo, "mentions").await.is_empty());
    assert!(
        !app.get_uploaded_source(&source.source_id)
            .await
            .unwrap()
            .extract_entities
    );
    app.resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    let interrupted = app
        .claim_remote_job("old-epoch", "interrupted")
        .await
        .unwrap()
        .unwrap();
    // New application/service epoch against persisted datastore, no concurrent old process.
    let restarted = application(&repo, "transport-policy", calls.clone(), failure.clone());
    restarted.reconcile_remote_jobs("new-epoch").await.unwrap();
    let fresh = restarted
        .claim_remote_job("new-epoch", "fresh")
        .await
        .unwrap()
        .unwrap();
    assert!(app
        .execute_remote_job(interrupted, ActionCancellation::new())
        .await
        .is_err());
    failure.store(0, Ordering::SeqCst);
    restarted
        .execute_remote_job(fresh, ActionCancellation::new())
        .await
        .unwrap();
    let final_job = restarted
        .get_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    assert_eq!(final_job.status, "completed");
    assert_eq!(calls.load(Ordering::SeqCst), final_job.total as usize + 1);
    assert_eq!(
        restarted
            .get_uploaded_source(&source.source_id)
            .await
            .unwrap()
            .extraction_policy_current,
        Some(true)
    );
    restarted
        .recover_remote_job(old, "worker_interrupted".into())
        .await
        .unwrap();
    assert_eq!(
        restarted
            .get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
}
#[tokio::test]
async fn reviewed_transport_cancel_before_claim_stale_revision_and_changed_policy_fail_closed() {
    let (repo, app, calls, _, source, request) = transport_fixture().await;
    let mut stale = request.clone();
    stale.reviewed.plan["expected_source_revision"] = "a".repeat(64).into();
    let plan = serde_json::from_value(stale.reviewed.plan.clone()).unwrap();
    stale.reviewed = enrichment_receipt(&plan).unwrap();
    assert!(app
        .execute_source_enrichment(caller(), stale, false)
        .await
        .is_err());
    let admission = app
        .execute_source_enrichment(caller(), request, false)
        .await
        .unwrap();
    assert_eq!(
        app.cancel_remote_job(caller(), &admission.job_id)
            .await
            .unwrap()
            .status,
        "cancelled"
    );
    assert!(app.claim_remote_job("e", "w").await.unwrap().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let changed = application(
        &repo,
        "changed-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    assert!(changed
        .resume_remote_job(caller(), &admission.job_id)
        .await
        .is_err());
    app.resume_remote_job(caller(), &admission.job_id)
        .await
        .unwrap();
    run(&app).await.unwrap();
    assert_eq!(
        app.get_uploaded_source(&source.source_id)
            .await
            .unwrap()
            .extraction_policy_current,
        Some(true)
    );
}

#[tokio::test]
async fn reviewed_transport_running_cancellation_keeps_private_stage_and_shared_keys_conflict() {
    let (repo, app, calls, _, source, request) = transport_fixture().await;
    let admitted = app
        .execute_source_enrichment(caller(), request.clone(), false)
        .await
        .unwrap();
    let execution = app
        .claim_remote_job("cancel-epoch", "cancel-worker")
        .await
        .unwrap()
        .unwrap();
    let lease = graphrag_db::RemoteJobLease {
        job_id: graphrag_db::parse_record_id(&execution.job_id, Some("processing_job")).unwrap(),
        instance_id: execution.instance_id.clone(),
        service_epoch: execution.service_epoch.clone(),
        worker_token: execution.worker_token.clone(),
    };
    let plan = serde_json::from_value(request.reviewed.plan.clone()).unwrap();
    repo.begin_source_entity_enrichment_leased(plan, Some(&lease))
        .await
        .unwrap();
    assert_eq!(
        app.cancel_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "running"
    );
    assert!(app
        .execute_remote_job(execution.clone(), ActionCancellation::new())
        .await
        .is_err());
    app.recover_remote_job(execution, "cancelled".into())
        .await
        .unwrap();
    assert_eq!(
        app.get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "cancelled"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(rows(&repo, "mentions").await.is_empty());
    assert!(
        !app.get_uploaded_source(&source.source_id)
            .await
            .unwrap()
            .extract_entities
    );
    // Exact reviewed endpoint proposal cannot reuse enrichment's shared request key.
    let a = repo
        .create_note(graphrag_core::Note::new("Fictional statement"))
        .await
        .unwrap();
    let b = repo
        .create_note(graphrag_core::Note::new("Fictional supporting evidence"))
        .await
        .unwrap();
    let ai = graphrag_core::record_id_to_string(a.id.as_ref().unwrap());
    let bi = graphrag_core::record_id_to_string(b.id.as_ref().unwrap());
    let ar = repo.inspect_record(&ai, 0).await.unwrap();
    let br = repo.inspect_record(&bi, 0).await.unwrap();
    assert!(app
        .propose_endpoint_remote(
            caller(),
            RemoteEndpointProposalRequest {
                request_id: request.request_id,
                from: EndpointProposalEvidence {
                    id: ai,
                    revision: ar.revision,
                    quote: ar.content
                },
                to: EndpointProposalEvidence {
                    id: bi,
                    revision: br.revision,
                    quote: br.content
                },
                relationship: EndpointRelationship::Supports,
                rationale: "Fictional manual review".into(),
                confirmed: true
            }
        )
        .await
        .is_err());
    assert!(rows(&repo, "proposed_edge").await.is_empty());
    app.resume_remote_job(caller(), &admitted.job_id)
        .await
        .unwrap();
    run(&app).await.unwrap();
    assert_eq!(
        app.get_remote_job(caller(), &admitted.job_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
}
