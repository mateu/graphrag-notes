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
    run_in_epoch(app, "fixture-epoch").await
}
async fn run_in_epoch(
    app: &EmbeddedApplication,
    epoch: &str,
) -> ApplicationResult<RemoteJobExecution> {
    let execution = app
        .claim_remote_job(epoch, "fixture-worker")
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
            repo.get_source(source_id).await.unwrap().unwrap().metadata["remote_upload"]["job_id"]
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
async fn direct_enrichment_unchanged_uploads_preserve_stage_origin_through_rollback() {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let vector_app = application(
        &repo,
        "fixture-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut input = request("direct-origin");
    input.extract_entities = false;
    let admission = vector_app
        .upload_source(caller(), input.clone())
        .await
        .unwrap();
    run(&vector_app).await.unwrap();
    let app = application(
        &repo,
        "conversion-policy",
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let opening = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let reviewed = app
        .prepare_source_enrichment(
            caller(),
            PrepareSourceEnrichment {
                request_id: "reviewed-conversion".into(),
                source_id: opening.id,
                revision: opening.revision,
            },
        )
        .await
        .unwrap();
    let plan: graphrag_db::SourceEnrichmentPlan = serde_json::from_value(reviewed.plan).unwrap();
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    let staged_source = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    let mut staged_unchanged = input.clone();
    staged_unchanged.request_id = "direct-origin-staged-unchanged".into();
    staged_unchanged.preserve_unchanged = true;
    let staged_upload = vector_app
        .upload_source(caller(), staged_unchanged)
        .await
        .unwrap();
    run(&vector_app).await.unwrap();
    assert_eq!(
        app.get_remote_job(caller(), &staged_upload.job_id)
            .await
            .unwrap()
            .result
            .unwrap()["action"],
        "unchanged"
    );
    assert_eq!(
        repo.get_source(&admission.source_id)
            .await
            .unwrap()
            .unwrap()
            .metadata["entity_enrichment_v1"]["prior_origin"],
        staged_source.metadata["entity_enrichment_v1"]["prior_origin"]
    );
    assert_eq!(
        repo.inspect_source_entity_enrichment(&plan)
            .await
            .unwrap()
            .status,
        "staged"
    );
    app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    let promoted = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let source_id = graphrag_db::parse_record_id(&admission.source_id, Some("source")).unwrap();
    let promoted_source = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    let prior_origin = promoted_source.metadata["entity_enrichment_v1"]["prior_origin"].clone();
    let promoted_notes = rows(&repo, "note").await;
    let inference_calls = calls.load(Ordering::SeqCst);

    let mut unchanged = input.clone();
    unchanged.request_id = "direct-origin-unchanged".into();
    unchanged.extract_entities = true;
    unchanged.preserve_unchanged = true;
    unchanged.expected_source_revision = Some(promoted.revision.clone());
    let unchanged_admission = app
        .upload_source(caller(), unchanged.clone())
        .await
        .unwrap();
    run(&app).await.unwrap();
    let unchanged_job = app
        .get_remote_job(caller(), &unchanged_admission.job_id)
        .await
        .unwrap();
    assert_eq!(unchanged_job.result.unwrap()["action"], "unchanged");
    let after_unchanged = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let after_unchanged_source = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after_unchanged_source.id, promoted_source.id);
    assert_eq!(after_unchanged_source.content, promoted_source.content);
    assert_eq!(
        after_unchanged_source.content_hash,
        promoted_source.content_hash
    );
    assert_eq!(
        after_unchanged_source.generation,
        promoted_source.generation
    );
    assert_eq!(
        after_unchanged_source.successful_generation,
        promoted_source.successful_generation
    );
    assert_eq!(
        after_unchanged_source.metadata["remote_upload"]["job_id"],
        unchanged_admission.job_id
    );
    assert_eq!(
        after_unchanged_source.metadata["entity_enrichment_v1"]["prior_origin"],
        prior_origin
    );
    assert_eq!(rows(&repo, "note").await, promoted_notes);
    assert_eq!(calls.load(Ordering::SeqCst), inference_calls);
    assert_eq!(
        after_unchanged.latest_upload_request_id,
        "direct-origin-unchanged"
    );
    assert_eq!(
        after_unchanged.reviewed_enrichment_v1.as_ref().unwrap()["original_request_id"],
        input.request_id
    );
    assert_eq!(
        after_unchanged.reviewed_enrichment_v1.as_ref().unwrap()["operation"],
        "enrich"
    );
    assert!(after_unchanged.processing_policy_current);

    let replay = app.upload_source(caller(), unchanged).await.unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.job_id, unchanged_admission.job_id);
    assert_eq!(calls.load(Ordering::SeqCst), inference_calls);

    let rolled = repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .unwrap();
    assert_eq!(rolled.status, "rolled_back");
    let rolled_back = app.get_uploaded_source(&admission.source_id).await.unwrap();
    assert_eq!(
        rolled_back.reviewed_enrichment_v1.as_ref().unwrap()["operation"],
        "rollback"
    );
    assert!(!rolled_back.extract_entities);
    assert_eq!(rolled_back.extraction_policy_current, None);
    assert_eq!(
        rolled_back.processing_policy_sha256,
        plan.original_policy_sha256
    );
    assert_ne!(
        rolled_back.processing_policy_sha256,
        rolled_back.configured_processing_policy_sha256
    );
    assert_eq!(rows(&repo, "note").await, promoted_notes);

    let mut after_rollback = input.clone();
    after_rollback.request_id = "direct-origin-after-rollback".into();
    after_rollback.preserve_unchanged = true;
    after_rollback.expected_source_revision = Some(rolled_back.revision.clone());
    let after_rollback_admission = vector_app
        .upload_source(caller(), after_rollback)
        .await
        .unwrap();
    run(&vector_app).await.unwrap();
    let after_rollback_job = app
        .get_remote_job(caller(), &after_rollback_admission.job_id)
        .await
        .unwrap();
    assert_eq!(after_rollback_job.result.unwrap()["action"], "unchanged");
    let final_view = app.get_uploaded_source(&admission.source_id).await.unwrap();
    let final_source = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(final_source.id.as_ref(), Some(&source_id));
    assert_eq!(final_source.content, promoted_source.content);
    assert_eq!(final_source.content_hash, promoted_source.content_hash);
    assert_eq!(final_source.generation, promoted_source.generation);
    assert_eq!(
        final_source.successful_generation,
        promoted_source.successful_generation
    );
    assert_eq!(
        final_source.metadata["entity_enrichment_v1"]["prior_origin"],
        prior_origin
    );
    assert_eq!(
        final_view.processing_policy_sha256,
        plan.original_policy_sha256
    );
    assert_eq!(rows(&repo, "note").await, promoted_notes);
    assert_eq!(calls.load(Ordering::SeqCst), inference_calls);
    assert_eq!(
        final_view.latest_upload_request_id,
        "direct-origin-after-rollback"
    );
    assert_eq!(
        final_view.reviewed_enrichment_v1.as_ref().unwrap()["operation"],
        "rollback"
    );
}

#[tokio::test]
async fn malformed_enrichment_lineage_cannot_authorize_unchanged_upload() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let app = application(
        &repo,
        "fixture-policy",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut original = request("malformed-origin-vector");
    original.extract_entities = false;
    let admission = app.upload_source(caller(), original).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &admission.source_id).await;
    app.enrich_uploaded_source(caller(), plan, ActionCancellation::new())
        .await
        .unwrap();
    let saved = repo
        .get_source(&admission.source_id)
        .await
        .unwrap()
        .unwrap();
    for case in 0..3 {
        let mut metadata = saved.metadata.clone();
        match case {
            0 => metadata["entity_enrichment_v1"] = serde_json::json!({}),
            1 => metadata["entity_enrichment_v1"]["plan"]["source_id"] = "source:foreign".into(),
            _ => {
                metadata["entity_enrichment_v1"]["prior_origin"]["job_id"] =
                    "processing_job:foreign".into()
            }
        }
        db.query("UPDATE $source SET metadata=$metadata")
            .bind(("source", saved.id.clone()))
            .bind(("metadata", metadata))
            .await
            .unwrap()
            .check()
            .unwrap();
        let before = rows(&repo, "source").await;
        let unchanged = app
            .upload_source(caller(), request(&format!("malformed-origin-{case}")))
            .await
            .unwrap();
        assert!(run(&app).await.is_err(), "case {case}");
        assert_eq!(rows(&repo, "source").await, before, "case {case}");
        assert_eq!(
            app.get_remote_job(caller(), &unchanged.job_id)
                .await
                .unwrap()
                .status,
            "failed",
            "case {case}"
        );
    }
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

#[tokio::test]
async fn reviewed_transport_rejects_direct_conversion_identity_reuse_for_rollback_without_journal_mutation(
) {
    let repo = Repository::new(init_memory().await.unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let app = application(
        &repo,
        "direct-rollback-policy",
        calls,
        Arc::new(AtomicUsize::new(0)),
    );
    let mut upload = request("direct-rollback-vector");
    upload.extract_entities = false;
    let source = app.upload_source(caller(), upload).await.unwrap();
    run(&app).await.unwrap();
    let plan = conversion_plan(&repo, &app, &source.source_id).await;
    let mut mismatched_conversion = ExecuteSourceEnrichment {
        request_id: "different-conversion-operation".into(),
        reviewed: enrichment_receipt(&plan).unwrap(),
        confirmed: true,
        rollback_source_revision: None,
    };
    assert!(app
        .execute_source_enrichment(caller(), mismatched_conversion.clone(), false)
        .await
        .is_err());
    app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
        .await
        .unwrap();
    let converted = app.get_uploaded_source(&source.source_id).await.unwrap();
    let jobs_before = rows(&repo, "processing_job").await;
    let receipts_before = rows(&repo, "remote_mutation_receipt").await;
    let source_before = rows(&repo, "source").await;
    let original_job = repo
        .get_remote_upload_job(&caller().instance_id, &source.job_id)
        .await
        .unwrap()
        .unwrap();
    for rollback in [false, true] {
        let mut durable = original_job.input.clone();
        durable.request_id = if rollback {
            plan.request_id.clone()
        } else {
            "durable-mismatched-conversion".into()
        };
        durable.extract_entities = true;
        durable.expected_source_revision = Some(plan.expected_source_revision.clone());
        durable.processing_options = plan.target_processing_options.clone();
        durable.enrichment = Some(graphrag_db::RemoteEnrichmentInput {
            plan: plan.clone(),
            rollback,
        });
        assert!(repo.admit_remote_upload(durable).await.is_err());
    }
    mismatched_conversion.request_id = plan.request_id.clone();
    mismatched_conversion.rollback_source_revision = Some(converted.revision.clone());
    assert!(app
        .execute_source_enrichment(caller(), mismatched_conversion, true)
        .await
        .is_err());
    assert_eq!(rows(&repo, "processing_job").await, jobs_before);
    assert_eq!(
        rows(&repo, "remote_mutation_receipt").await,
        receipts_before
    );
    assert_eq!(rows(&repo, "source").await, source_before);

    let rollback = ExecuteSourceEnrichment {
        request_id: "direct-conversion-rollback".into(),
        reviewed: enrichment_receipt(&plan).unwrap(),
        confirmed: true,
        rollback_source_revision: Some(converted.revision),
    };
    let admitted = app
        .execute_source_enrichment(caller(), rollback.clone(), true)
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
    let replay = app
        .execute_source_enrichment(caller(), rollback, true)
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.job_id, admitted.job_id);
    assert_eq!(replay.source_id, admitted.source_id);
}

async fn lineage_fixture(
    state: &str,
) -> (
    Repository,
    EmbeddedApplication,
    Arc<AtomicUsize>,
    UploadAdmission,
    graphrag_db::SourceEnrichmentPlan,
) {
    let (repo, app, calls, _, source, operation) = transport_fixture().await;
    let plan: graphrag_db::SourceEnrichmentPlan =
        serde_json::from_value(operation.reviewed.plan).unwrap();
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    if state != "staged" {
        app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
            .await
            .unwrap();
    }
    if state == "rolled_back" {
        repo.rollback_source_entity_enrichment(&plan, true)
            .await
            .unwrap();
    }
    (repo, app, calls, source, plan)
}

#[tokio::test]
async fn reviewed_unchanged_refresh_preserves_latest_input_and_immutable_lineage() {
    for state in ["staged", "promoted", "rolled_back"] {
        let (repo, app, calls, source, plan) = lineage_fixture(state).await;
        let before = repo.get_source(&source.source_id).await.unwrap().unwrap();
        let notes = rows(&repo, "note").await;
        let calls_before = calls.load(Ordering::SeqCst);
        let mut capture = request("initial");
        capture.extract_entities = state == "promoted";
        for variant in ["line_endings", "host", "collection"] {
            let mut refresh = capture;
            refresh.request_id = format!("{state}-{variant}");
            match variant {
                "line_endings" => refresh.content = refresh.content.replace('\n', "\r\n"),
                "host" => {
                    refresh
                        .provenance
                        .as_mut()
                        .unwrap()
                        .metadata
                        .insert("host".into(), "different-host".into());
                }
                _ => {
                    refresh
                        .provenance
                        .as_mut()
                        .unwrap()
                        .metadata
                        .insert("collection_id".into(), "different-collection".into());
                }
            }
            let expected_content = refresh.content.clone();
            let expected_provenance =
                serde_json::to_value(refresh.provenance.as_ref().unwrap()).unwrap();
            let admitted = app.upload_source(caller(), refresh.clone()).await.unwrap();
            run(&app).await.unwrap();
            assert_eq!(
                app.get_remote_job(caller(), &admitted.job_id)
                    .await
                    .unwrap()
                    .result
                    .unwrap()["action"],
                "unchanged"
            );
            let current = app.get_uploaded_source(&source.source_id).await.unwrap();
            assert_eq!(current.content, expected_content, "{state}/{variant}");
            assert_eq!(current.provenance, expected_provenance, "{state}/{variant}");
            let saved = repo.get_source(&source.source_id).await.unwrap().unwrap();
            assert_eq!(saved.content, before.content);
            assert_eq!(saved.generation, before.generation);
            assert_eq!(
                saved.metadata["entity_enrichment_v1"]["prior_origin"],
                before.metadata["entity_enrichment_v1"]["prior_origin"]
            );
            assert_eq!(
                repo.inspect_source_entity_enrichment(&plan)
                    .await
                    .unwrap()
                    .status,
                state
            );
            assert_eq!(rows(&repo, "note").await, notes, "{state}/{variant}");
            capture = refresh;
        }
        let final_state = if state == "promoted" {
            let current = app.get_uploaded_source(&source.source_id).await.unwrap();
            repo.rollback_source_entity_enrichment(&plan, true)
                .await
                .unwrap();
            let rolled = app.get_uploaded_source(&source.source_id).await.unwrap();
            assert_eq!(rolled.content, current.content);
            assert_eq!(rolled.provenance, current.provenance);
            assert_eq!(
                rolled.latest_upload_request_id,
                current.latest_upload_request_id
            );
            assert_eq!(rolled.processing_policy_sha256, plan.original_policy_sha256);
            assert_eq!(
                rolled.reviewed_enrichment_v1.as_ref().unwrap()["original_request_id"],
                "transport-vector"
            );
            "rolled_back"
        } else {
            state
        };
        let mut exact = request(&format!("{state}-exact"));
        exact.extract_entities = final_state == "promoted";
        let admitted = app.upload_source(caller(), exact).await.unwrap();
        run(&app).await.unwrap();
        assert_eq!(
            app.get_remote_job(caller(), &admitted.job_id)
                .await
                .unwrap()
                .result
                .unwrap()["action"],
            "unchanged"
        );
        assert_eq!(
            repo.inspect_source_entity_enrichment(&plan)
                .await
                .unwrap()
                .status,
            final_state
        );
        assert_eq!(
            app.get_uploaded_source(&source.source_id)
                .await
                .unwrap()
                .latest_upload_request_id,
            format!("{state}-exact")
        );
        assert_eq!(calls.load(Ordering::SeqCst), calls_before);
    }
}

#[tokio::test]
async fn reviewed_unchanged_recovery_is_fenced_across_policy_epoch_transitions() {
    for (opening, successor) in [
        ("staged", "promoted"),
        ("staged", "rolled_back"),
        ("promoted", "rolled_back"),
    ] {
        for interruption in ["cancelled", "interrupted", "restart"] {
            let (repo, app, calls, source, plan) = lineage_fixture(opening).await;
            let mut exact = request(&format!("{opening}-{successor}-{interruption}"));
            exact.extract_entities = opening == "promoted";
            let admitted = app.upload_source(caller(), exact).await.unwrap();
            let execution = app
                .claim_remote_job("epoch", "worker")
                .await
                .unwrap()
                .unwrap();
            let lease = graphrag_db::RemoteJobLease {
                job_id: graphrag_db::parse_record_id(&execution.job_id, Some("processing_job"))
                    .unwrap(),
                instance_id: execution.instance_id.clone(),
                service_epoch: execution.service_epoch.clone(),
                worker_token: execution.worker_token.clone(),
            };
            repo.begin_remote_upload_generation(&lease).await.unwrap();
            if interruption == "cancelled" {
                app.cancel_remote_job(caller(), &admitted.job_id)
                    .await
                    .unwrap();
                repo.finish_remote_upload_job(
                    &lease,
                    graphrag_db::ProcessingJobStatus::Cancelled,
                    None,
                    None,
                )
                .await
                .unwrap();
            } else if interruption == "interrupted" {
                repo.recover_remote_upload_job(&lease, "worker_interrupted")
                    .await
                    .unwrap();
            }
            if opening == "staged" {
                app.enrich_uploaded_source(caller(), plan.clone(), ActionCancellation::new())
                    .await
                    .unwrap();
            }
            if successor == "rolled_back" {
                repo.rollback_source_entity_enrichment(&plan, true)
                    .await
                    .unwrap();
            }
            let calls_after_transition = calls.load(Ordering::SeqCst);
            let current = app.get_uploaded_source(&source.source_id).await.unwrap();
            assert_eq!(
                current.extraction_policy_current,
                (successor == "promoted").then_some(true)
            );
            assert_eq!(
                current.reviewed_enrichment_v1.as_ref().unwrap()["operation"],
                if successor == "promoted" {
                    "enrich"
                } else {
                    "rollback"
                }
            );
            if interruption == "restart" {
                app.reconcile_remote_jobs("new-epoch").await.unwrap();
                assert_eq!(
                    app.get_remote_job(caller(), &admitted.job_id)
                        .await
                        .unwrap()
                        .status,
                    "failed"
                );
            }
            assert!(
                app.resume_remote_job(caller(), &admitted.job_id)
                    .await
                    .is_err(),
                "{opening}/{successor}/{interruption}"
            );
            assert_eq!(
                app.get_uploaded_source(&source.source_id)
                    .await
                    .unwrap()
                    .revision,
                current.revision
            );
            let mut fresh = request(&format!("{opening}-{successor}-{interruption}-fresh"));
            fresh.extract_entities = successor == "promoted";
            fresh.preserve_unchanged = true;
            let fresh_admission = app.upload_source(caller(), fresh).await.unwrap();
            run_in_epoch(
                &app,
                if interruption == "restart" {
                    "new-epoch"
                } else {
                    "fixture-epoch"
                },
            )
            .await
            .unwrap();
            assert_eq!(
                app.get_remote_job(caller(), &fresh_admission.job_id)
                    .await
                    .unwrap()
                    .result
                    .unwrap()["action"],
                "unchanged"
            );
            let fresh_view = app.get_uploaded_source(&source.source_id).await.unwrap();
            assert_eq!(
                fresh_view.extraction_policy_current,
                current.extraction_policy_current
            );
            assert_eq!(
                fresh_view.reviewed_enrichment_v1.as_ref().unwrap()["operation"],
                current.reviewed_enrichment_v1.as_ref().unwrap()["operation"]
            );
            assert_eq!(calls.load(Ordering::SeqCst), calls_after_transition);
        }
    }
}

#[tokio::test]
async fn same_policy_unchanged_resume_preserves_capture_and_redacts_private_checkpoint() {
    for state in ["staged", "promoted", "rolled_back"] {
        for interruption in ["cancelled", "interrupted"] {
            let (repo, app, calls, source, _) = lineage_fixture(state).await;
            let mut capture = request(&format!("{state}-{interruption}-resume"));
            capture.extract_entities = state == "promoted";
            let admitted = app.upload_source(caller(), capture).await.unwrap();
            let execution = app
                .claim_remote_job("epoch", "worker")
                .await
                .unwrap()
                .unwrap();
            let lease = graphrag_db::RemoteJobLease {
                job_id: graphrag_db::parse_record_id(&execution.job_id, Some("processing_job"))
                    .unwrap(),
                instance_id: execution.instance_id.clone(),
                service_epoch: execution.service_epoch.clone(),
                worker_token: execution.worker_token.clone(),
            };
            repo.begin_remote_upload_generation(&lease).await.unwrap();
            if interruption == "cancelled" {
                app.cancel_remote_job(caller(), &admitted.job_id)
                    .await
                    .unwrap();
                repo.finish_remote_upload_job(
                    &lease,
                    graphrag_db::ProcessingJobStatus::Cancelled,
                    None,
                    None,
                )
                .await
                .unwrap();
            } else {
                repo.recover_remote_upload_job(&lease, "worker_interrupted")
                    .await
                    .unwrap();
            }
            let committed = app.get_uploaded_source(&source.source_id).await.unwrap();
            let notes = rows(&repo, "note").await;
            let calls_before = calls.load(Ordering::SeqCst);
            let resumed = app
                .resume_remote_job(caller(), &admitted.job_id)
                .await
                .unwrap();
            assert!(
                resumed.result.is_none(),
                "Successful resume must not expose a private capture checkpoint: {resumed:?}"
            );
            run(&app).await.unwrap();
            let completed = app
                .get_remote_job(caller(), &admitted.job_id)
                .await
                .unwrap();
            assert_eq!(completed.result.as_ref().unwrap()["action"], "unchanged");
            assert_eq!(
                app.get_uploaded_source(&source.source_id).await.unwrap(),
                committed
            );
            assert_eq!(rows(&repo, "note").await, notes);
            assert_eq!(calls.load(Ordering::SeqCst), calls_before);
        }
    }
}
