use super::*;
use crate::init_memory;
use graphrag_core::EntityType;
use std::time::Duration;
use surrealdb_types::ToSql;

fn entity(name: &str) -> Entity {
    let mut entity = Entity::new(name, EntityType::Concept);
    entity.metadata = serde_json::json!({});
    entity
}

fn input(instance: &str, request: &str, body: &str) -> RemoteUploadInput {
    RemoteUploadInput {
        authenticated_instance_id: instance.into(),
        request_id: request.into(),
        payload_fingerprint: format!("{:x}", Sha256::digest(body.as_bytes())),
        document_key: "atlas.md".into(),
        markdown: body.into(),
        title: Some("Atlas".into()),
        source_provenance: serde_json::json!({"uri":"file:///client/atlas.md","label":"original"}),
        extract_entities: false,
        preserve_unchanged: false,
        create_only: false,
        expected_source_revision: None,
        processing_options: serde_json::json!({"runtime":{"target_chunk_chars":1000},"provider":"fixture","model":"fixture","cache_identity":"fixture"}),
    }
}

async fn admit_claim(
    repo: &Repository,
    input: RemoteUploadInput,
    epoch: &str,
    worker: &str,
) -> (RemoteJobAdmission, RemoteJobLease) {
    let instance = input.authenticated_instance_id.clone();
    let admission = repo.admit_remote_upload(input).await.unwrap();
    let id = admission.result["job_id"].as_str().unwrap();
    let job = repo
        .claim_remote_upload_job(&instance, id, epoch, worker)
        .await
        .unwrap();
    (
        admission,
        RemoteJobLease {
            job_id: job.job.id.unwrap(),
            instance_id: instance,
            service_epoch: epoch.into(),
            worker_token: worker.into(),
        },
    )
}

async fn table(repo: &Repository, name: &str) -> Vec<serde_json::Value> {
    repo.db
        .query(format!("SELECT * FROM {name} ORDER BY id"))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

async fn complete(repo: &Repository, lease: &RemoteJobLease, body: &str) -> RemoteUploadJob {
    repo.begin_remote_upload_generation(lease).await.unwrap();
    repo.stage_remote_upload_notes(lease, vec![Note::new(body)])
        .await
        .unwrap();
    repo.reconcile_remote_upload(lease, &[]).await.unwrap();
    repo.finish_remote_upload_job(
        lease,
        ProcessingJobStatus::Completed,
        None,
        Some(serde_json::json!({"status":"completed"})),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn concurrent_admission_is_exact_scoped_and_conflicts_do_not_create_sources() {
    let db = init_memory().await.unwrap();
    let a = Repository::new(db.clone());
    let b = Repository::new(db);
    let request = input("openclaw", "request", "Atlas body");
    let (left, right) = tokio::join!(
        a.admit_remote_upload(request.clone()),
        b.admit_remote_upload(request.clone())
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_ne!(left.replayed, right.replayed);
    assert_eq!(left.result, right.result);
    assert_eq!(table(&a, "processing_job").await.len(), 1);
    assert!(table(&a, "source").await.is_empty());
    assert!(table(&a, "note").await.is_empty());
    let mut changed = request.clone();
    changed.markdown = "Different body".into();
    assert!(matches!(
        a.admit_remote_upload(changed).await,
        Err(DbError::RemoteRequestConflict { .. })
    ));
    assert!(matches!(
        a.find_remote_upload_admission("openclaw", "request", &"b".repeat(64))
            .await,
        Err(DbError::RemoteRequestConflict { .. })
    ));
    let second = a
        .admit_remote_upload(input("hermes", "request", "Atlas body"))
        .await
        .unwrap();
    assert_ne!(second.result["source_uri"], left.result["source_uri"]);
    let id = left.result["job_id"].as_str().unwrap();
    assert!(a
        .get_remote_upload_job("hermes", id)
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        a.cancel_remote_upload_job("hermes", id).await,
        Err(DbError::NotFound(..))
    ));
    assert_eq!(
        a.list_remote_upload_jobs("hermes", 10).await.unwrap().len(),
        1
    );
}

#[tokio::test]
async fn invalid_admission_bounds_are_rejected_without_writes_and_claim_is_single_owner() {
    let db = init_memory().await.unwrap();
    let a = Repository::new(db.clone());
    let b = Repository::new(db);
    let mut oversized = input("a", "large", &"x".repeat(MAX_REMOTE_UPLOAD_BYTES + 1));
    assert!(matches!(
        a.admit_remote_upload(oversized.clone()).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
    oversized.markdown = "Atlas".into();
    oversized.processing_options = serde_json::json!({"data":"x".repeat(MAX_REMOTE_JSON_BYTES)});
    assert!(matches!(
        a.admit_remote_upload(oversized).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
    assert!(table(&a, "processing_job").await.is_empty());
    assert!(table(&a, "source").await.is_empty());
    let admission = a
        .admit_remote_upload(input("a", "claim", "Atlas"))
        .await
        .unwrap();
    let id = admission.result["job_id"].as_str().unwrap();
    let (left, right) = tokio::join!(
        a.claim_remote_upload_job("a", id, "epoch", "left"),
        b.claim_remote_upload_job("a", id, "epoch", "right")
    );
    assert_ne!(left.is_ok(), right.is_ok());
    assert!(a
        .claim_next_remote_upload("epoch", "third")
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        a.list_remote_upload_jobs("a", 201).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
}

#[tokio::test]
async fn cancellation_is_independent_of_transition_gate_and_cannot_stage_after_flag() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (admission, lease) = admit_claim(
        &repo,
        input("openclaw", "cancel", "Atlas"),
        "epoch",
        "worker",
    )
    .await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    let gate = repo.remote_job_transition_lock.lock().await;
    let cancelled = tokio::time::timeout(
        Duration::from_secs(1),
        repo.cancel_remote_upload_job("openclaw", admission.result["job_id"].as_str().unwrap()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(cancelled.job.status, "running");
    assert!(cancelled.cancel_requested);
    drop(gate);
    assert!(matches!(
        repo.stage_remote_upload_notes(&lease, vec![Note::new("Atlas")])
            .await,
        Err(DbError::RemoteJobCancelled(_))
    ));
    let terminal = repo
        .finish_remote_upload_job(
            &lease,
            ProcessingJobStatus::Failed,
            Some("worker_interrupted".into()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(terminal.job.status, "cancelled");
    assert_eq!(terminal.job.last_error.as_deref(), Some("cancelled"));
    assert!(table(&repo, "note").await.is_empty());
    let queued = repo
        .admit_remote_upload(input("openclaw", "queued-cancel", "Other"))
        .await
        .unwrap();
    let cancelled = repo
        .cancel_remote_upload_job("openclaw", queued.result["job_id"].as_str().unwrap())
        .await
        .unwrap();
    assert_eq!(cancelled.job.status, "cancelled");
    assert!(cancelled.job.finished_at.is_some());
    assert!(repo
        .claim_next_remote_upload("epoch", "other-worker")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn restart_fences_old_workers_preserves_checkpoint_and_ignores_current_epoch() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (old, lease) =
        admit_claim(&repo, input("a", "old", "Atlas"), "old-epoch", "old-worker").await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    let staged = repo
        .stage_remote_upload_notes(&lease, vec![Note::new("Atlas")])
        .await
        .unwrap();
    let (_, current) = admit_claim(
        &repo,
        input("b", "current", "Current"),
        "current-epoch",
        "worker",
    )
    .await;
    assert_eq!(
        repo.reconcile_interrupted_remote_uploads("current-epoch")
            .await
            .unwrap(),
        1
    );
    let interrupted = repo
        .get_remote_upload_job("a", old.result["job_id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(interrupted.job.status, "failed");
    assert_eq!(interrupted.job.last_error.as_deref(), Some("interrupted"));
    assert_eq!(interrupted.phase, "staged");
    assert_eq!(interrupted.job.item_ids, staged.job.item_ids);
    assert!(repo.owned_remote_upload_job(&current).await.is_ok());
    repo.resume_remote_upload_job("a", old.result["job_id"].as_str().unwrap())
        .await
        .unwrap();
    let job = repo
        .claim_remote_upload_job(
            "a",
            old.result["job_id"].as_str().unwrap(),
            "current-epoch",
            "new-worker",
        )
        .await
        .unwrap();
    assert_eq!(job.phase, "staged");
    assert!(matches!(
        repo.reconcile_remote_upload(&lease, &[]).await,
        Err(DbError::RemoteJobOwnershipLost(_))
    ));
    assert!(matches!(
        repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Failed, None, None)
            .await,
        Err(DbError::RemoteJobOwnershipLost(_))
    ));
    let new = RemoteJobLease {
        worker_token: "new-worker".into(),
        service_epoch: "current-epoch".into(),
        ..lease
    };
    repo.reconcile_remote_upload(&new, &[]).await.unwrap();
    repo.finish_remote_upload_job(&new, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    assert_eq!(
        repo.reconcile_interrupted_remote_uploads("next-epoch")
            .await
            .unwrap(),
        1
    ); // current job only
    assert_eq!(
        repo.get_remote_upload_job("a", old.result["job_id"].as_str().unwrap())
            .await
            .unwrap()
            .unwrap()
            .job
            .status,
        "completed"
    );
}

#[tokio::test]
async fn chunk_staging_is_atomic_and_hidden_until_promotion() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, lease) = admit_claim(&repo, input("a", "stage", "Atlas body"), "epoch", "worker").await;
    let source = repo.begin_remote_upload_generation(&lease).await.unwrap();
    assert_eq!(
        repo.begin_remote_upload_generation(&lease)
            .await
            .unwrap()
            .generation,
        source.generation
    );
    repo.db
        .query("DEFINE FIELD OVERWRITE content ON note TYPE string ASSERT $value != 'bad chunk'")
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo
        .stage_remote_upload_notes(
            &lease,
            vec![Note::new("Good chunk"), Note::new("bad chunk")]
        )
        .await
        .is_err());
    assert!(table(&repo, "note").await.is_empty());
    assert_eq!(
        repo.owned_remote_upload_job(&lease).await.unwrap().phase,
        "preparing"
    );
    let staged = repo
        .stage_remote_upload_notes(
            &lease,
            vec![Note::new("Atlas part one"), Note::new("Atlas part two")],
        )
        .await
        .unwrap();
    assert_eq!(staged.job.total_count, 2);
    assert_eq!(staged.job.item_ids.len(), 2);
    assert!(repo
        .get_source_chunks(source.id.as_ref().unwrap())
        .await
        .unwrap()
        .is_empty());
    assert_eq!(repo.remote_upload_notes(&lease).await.unwrap().len(), 2);
    for id in &staged.job.item_ids {
        assert!(repo.inspect_record(id, 0).await.is_err());
    }
    repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
    assert_eq!(
        repo.get_source_chunks(source.id.as_ref().unwrap())
            .await
            .unwrap()
            .len(),
        2
    );
    let inspection = repo
        .inspect_record(&staged.job.item_ids[0], 0)
        .await
        .unwrap();
    assert_eq!(inspection.provenance.instance_id.as_deref(), Some("a"));
    assert_eq!(
        inspection.provenance.source.unwrap()["uri"],
        "file:///client/atlas.md"
    );
    assert_eq!(
        repo.owned_remote_upload_job(&lease).await.unwrap().phase,
        "promoted"
    );
}

#[tokio::test]
async fn failed_refresh_preserves_old_content_and_origin_and_new_attempt_invalidates_old_resume() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, original) = admit_claim(
        &repo,
        input("a", "original", "Original Atlas"),
        "epoch",
        "worker1",
    )
    .await;
    let complete = complete(&repo, &original, "Original Atlas").await;
    let id = &complete.job.item_ids[0];
    let original_inspection = repo.inspect_record(id, 0).await.unwrap();
    let mut refresh = input("a", "refresh", "Changed Atlas");
    refresh.source_provenance["label"] = "attempted".into();
    let (admission, lease) = admit_claim(&repo, refresh, "epoch", "worker2").await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    repo.stage_remote_upload_notes(&lease, vec![Note::new("Changed Atlas")])
        .await
        .unwrap();
    assert_eq!(
        repo.inspect_record(id, 0).await.unwrap().provenance,
        original_inspection.provenance
    );
    repo.finish_remote_upload_job(
        &lease,
        ProcessingJobStatus::Failed,
        Some("provider_failed".into()),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        repo.get_source_chunks(complete.source_id.as_ref().unwrap())
            .await
            .unwrap()[0]
            .content,
        "Original Atlas"
    );
    let (_, next) = admit_claim(&repo, input("a", "next", "Next Atlas"), "epoch", "worker3").await;
    let source = repo.begin_remote_upload_generation(&next).await.unwrap();
    assert_eq!(source.generation, 3);
    assert!(matches!(
        repo.resume_remote_upload_job("a", admission.result["job_id"].as_str().unwrap())
            .await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
    repo.stage_remote_upload_notes(&next, vec![Note::new("Next Atlas")])
        .await
        .unwrap();
    repo.reconcile_remote_upload(&next, &[]).await.unwrap();
    assert_eq!(
        repo.get_source_chunks(complete.source_id.as_ref().unwrap())
            .await
            .unwrap()[0]
            .content,
        "Next Atlas"
    );
    assert_eq!(table(&repo, "note").await.len(), 1);
}

#[tokio::test]
async fn unchanged_body_reuses_exact_active_ids_and_has_no_staged_generation() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, first) = admit_claim(
        &repo,
        input("a", "first", "Same\r\nAtlas"),
        "epoch",
        "worker1",
    )
    .await;
    let first = complete(&repo, &first, "Same Atlas").await;
    let (_, next) = admit_claim(&repo, input("a", "next", "Same\nAtlas"), "epoch", "worker2").await;
    let source = repo.begin_remote_upload_generation(&next).await.unwrap();
    let unchanged = repo.owned_remote_upload_job(&next).await.unwrap();
    assert_eq!(unchanged.phase, "promoted");
    assert_eq!(unchanged.job.scope.as_deref(), Some("unchanged"));
    assert_eq!(unchanged.job.item_ids, first.job.item_ids);
    assert_eq!(source.generation, 1);
    assert_eq!(table(&repo, "note").await.len(), 1);
    repo.finish_remote_upload_job(&next, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    let origin = repo
        .get_source(&source.uri.unwrap())
        .await
        .unwrap()
        .unwrap()
        .metadata;
    assert_eq!(origin["remote_upload"]["request_id"], "next");
    assert!(origin.get("remote_upload_pending").is_none());
}

#[tokio::test]
async fn unchanged_extracted_source_finishes_with_no_new_entity_or_note_mutation() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut request = input("a", "first", "Atlas");
    request.extract_entities = true;
    let (_, first) = admit_claim(&repo, request, "epoch", "worker1").await;
    repo.begin_remote_upload_generation(&first).await.unwrap();
    repo.stage_remote_upload_notes(&first, vec![Note::new("Atlas")])
        .await
        .unwrap();
    repo.reconcile_remote_upload(&first, &[]).await.unwrap();
    repo.persist_remote_upload_entities(&first, 0, vec![entity("Atlas")])
        .await
        .unwrap();
    repo.finish_remote_upload_job(&first, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    let notes = table(&repo, "note").await;
    let mentions = table(&repo, "mentions").await;
    let entities = table(&repo, "entity").await;
    let mut same = input("a", "same", "Atlas");
    same.extract_entities = true;
    same.source_provenance["label"] = "new attribution".into();
    let (_, lease) = admit_claim(&repo, same, "epoch", "worker2").await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    let job = repo.owned_remote_upload_job(&lease).await.unwrap();
    assert_eq!(job.job.scope.as_deref(), Some("unchanged"));
    assert_eq!(job.job.completed_count, job.job.total_count);
    repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    assert_eq!(table(&repo, "note").await, notes);
    assert_eq!(table(&repo, "mentions").await, mentions);
    assert_eq!(table(&repo, "entity").await, entities);
}

#[tokio::test]
async fn changed_title_and_processing_options_refresh_even_identical_body() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, first) = admit_claim(&repo, input("a", "first", "Atlas"), "epoch", "worker1").await;
    complete(&repo, &first, "Atlas").await;
    let mut title = input("a", "title", "Atlas");
    title.title = Some("Changed title".into());
    let (_, second) = admit_claim(&repo, title, "epoch", "worker2").await;
    let generation = repo.begin_remote_upload_generation(&second).await.unwrap();
    assert_eq!(generation.generation, 2);
    assert_eq!(
        repo.owned_remote_upload_job(&second).await.unwrap().phase,
        "preparing"
    );
    repo.stage_remote_upload_notes(&second, vec![Note::new("Atlas")])
        .await
        .unwrap();
    repo.reconcile_remote_upload(&second, &[]).await.unwrap();
    repo.finish_remote_upload_job(&second, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    let mut options = input("a", "options", "Atlas");
    options.title = Some("Changed title".into());
    options.processing_options["model"] = "different".into();
    let (_, third) = admit_claim(&repo, options, "epoch", "worker3").await;
    assert_eq!(
        repo.begin_remote_upload_generation(&third)
            .await
            .unwrap()
            .generation,
        3
    );
    assert_eq!(
        repo.owned_remote_upload_job(&third).await.unwrap().phase,
        "preparing"
    );
}

#[tokio::test]
async fn extraction_and_checkpoint_commit_together_and_replayed_item_is_noop() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut request = input("a", "extract", "Atlas");
    request.extract_entities = true;
    let (_, lease) = admit_claim(&repo, request, "epoch", "worker").await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    repo.stage_remote_upload_notes(&lease, vec![Note::new("Atlas")])
        .await
        .unwrap();
    repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
    assert!(matches!(
        repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
            .await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
    repo.db.query("DEFINE FIELD OVERWRITE canonical_name ON entity TYPE string ASSERT $value != 'forbidden'").await.unwrap().check().unwrap();
    assert!(repo
        .persist_remote_upload_entities(&lease, 0, vec![entity("Allowed"), entity("Forbidden")])
        .await
        .is_err());
    assert!(table(&repo, "entity").await.is_empty());
    assert!(table(&repo, "mentions").await.is_empty());
    assert_eq!(
        repo.owned_remote_upload_job(&lease)
            .await
            .unwrap()
            .job
            .completed_count,
        0
    );
    let done = repo
        .persist_remote_upload_entities(&lease, 0, vec![entity("Atlas")])
        .await
        .unwrap();
    assert_eq!(done.job.completed_count, 1);
    assert_eq!(
        done.job.checkpoint.as_deref(),
        Some(done.job.item_ids[0].as_str())
    );
    repo.persist_remote_upload_entities(&lease, 0, vec![entity("Different")])
        .await
        .unwrap();
    assert_eq!(table(&repo, "entity").await.len(), 1);
    assert_eq!(table(&repo, "mentions").await.len(), 1);
    repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
}

#[tokio::test]
async fn reconciliation_retries_copy_and_promotion_failures_without_duplicate_graph_or_lost_undo() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, first) = admit_claim(
        &repo,
        input("a", "first", "First Atlas"),
        "epoch",
        "worker1",
    )
    .await;
    let first = complete(&repo, &first, "Atlas chunk").await;
    let old = parse_record_id(&first.job.item_ids[0], Some("note")).unwrap();
    let manual = repo
        .create_note(Note::new("Manual endpoint"))
        .await
        .unwrap()
        .id
        .unwrap();
    repo.create_edge(&old, &manual, EdgeType::Supports, Some(0.9))
        .await
        .unwrap();
    let proposal = repo
        .upsert_edge_proposal(EdgeProposalDraft {
            from_id: old.clone(),
            to_id: manual.clone(),
            edge_type: EdgeType::RelatedTo,
            confidence: 0.8,
            reason: "user accepted".into(),
            generator: "test".into(),
            generator_version: None,
            model: None,
        })
        .await
        .unwrap();
    let proposal_id = proposal.id.unwrap();
    let accepted = repo
        .accept_edge_proposal(&proposal_id, Some("test".into()), None, true)
        .await
        .unwrap();
    let old_edge = accepted.resulting_edge_id.unwrap();
    let entity = repo.upsert_entity(entity("Atlas")).await.unwrap();
    repo.link_note_to_entity(&old, entity.id.as_ref().unwrap())
        .await
        .unwrap();
    let (_, next) = admit_claim(
        &repo,
        input("a", "next", "Different source wrapper"),
        "epoch",
        "worker2",
    )
    .await;
    repo.begin_remote_upload_generation(&next).await.unwrap();
    let staged = repo
        .stage_remote_upload_notes(&next, vec![Note::new("Atlas chunk")])
        .await
        .unwrap();
    let new = parse_record_id(&staged.job.item_ids[0], Some("note")).unwrap();
    let successors = [(old.clone(), new.clone(), true)];
    assert!(matches!(
        repo.reconcile_remote_upload(&next, &[(manual.clone(), new.clone(), true)])
            .await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
    assert_eq!(table(&repo, "supports").await.len(), 1);
    assert_eq!(table(&repo, "related_to").await.len(), 1);
    repo.db
        .query("DEFINE FIELD OVERWRITE status ON source TYPE string ASSERT $value != 'ready'")
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo
        .reconcile_remote_upload(&next, &successors)
        .await
        .is_err());
    assert_eq!(
        repo.owned_remote_upload_job(&next).await.unwrap().phase,
        "staged"
    );
    assert!(repo
        .inspect_record(&record_id_to_string(&old), 0)
        .await
        .is_ok());
    assert!(repo
        .inspect_record(&record_id_to_string(&new), 0)
        .await
        .is_err());
    assert_eq!(table(&repo, "supports").await.len(), 2);
    assert_eq!(table(&repo, "related_to").await.len(), 2);
    repo.db
        .query("DEFINE FIELD OVERWRITE status ON source TYPE string")
        .await
        .unwrap()
        .check()
        .unwrap();
    repo.db.query(format!("DEFINE FIELD OVERWRITE resulting_edge_id ON proposed_edge TYPE option<record> ASSERT $value = {}",old_edge.to_sql())).await.unwrap().check().unwrap();
    assert!(repo
        .reconcile_remote_upload(&next, &successors)
        .await
        .is_err());
    assert_eq!(
        repo.owned_remote_upload_job(&next).await.unwrap().phase,
        "promoted"
    );
    assert!(repo
        .inspect_record(&record_id_to_string(&old), 0)
        .await
        .is_err());
    assert!(repo
        .inspect_record(&record_id_to_string(&new), 0)
        .await
        .is_ok());
    assert_eq!(table(&repo, "supports").await.len(), 2);
    assert_eq!(table(&repo, "related_to").await.len(), 2);
    repo.db
        .query("DEFINE FIELD OVERWRITE resulting_edge_id ON proposed_edge TYPE option<record>")
        .await
        .unwrap()
        .check()
        .unwrap();
    repo.reconcile_remote_upload(&next, &successors)
        .await
        .unwrap();
    repo.reconcile_remote_upload(&next, &successors)
        .await
        .unwrap();
    assert_eq!(table(&repo, "supports").await.len(), 1);
    assert_eq!(table(&repo, "related_to").await.len(), 1);
    assert_eq!(
        repo.get_entities_for_note(&record_id_to_string(&new))
            .await
            .unwrap()
            .len(),
        1
    );
    let accepted = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    let new_edge = accepted.resulting_edge_id.unwrap();
    assert_ne!(new_edge, old_edge);
    assert!(repo
        .undo_edge(&new_edge, Some("after retry".into()))
        .await
        .unwrap());
    assert_eq!(
        repo.get_edge_proposal(&proposal_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        ProposedEdgeStatus::Superseded
    );
    assert_eq!(table(&repo, "supports").await.len(), 1);
}

#[tokio::test]
async fn portable_jobs_exclude_local_workers_and_restore_exact_admission_with_fresh_ownership() {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.create_processing_job(ProcessingJobType::Embedding, None, 0)
        .await
        .unwrap();
    repo.create_processing_job(ProcessingJobType::EntityExtraction, None, 0)
        .await
        .unwrap();
    let request = input("a", "portable", "Atlas");
    let (admission, lease) = admit_claim(&repo, request.clone(), "old-epoch", "worker").await;
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    repo.stage_remote_upload_notes(&lease, vec![Note::new("Atlas")])
        .await
        .unwrap();
    let restored = Repository::new(init_memory().await.unwrap());
    for name in PORTABLE_TABLES {
        let records = repo.portable_records_page(name, 0, 100).await.unwrap();
        if *name == "processing_job" {
            assert_eq!(records.len(), 1);
        }
        for record in records {
            restored
                .restore_portable_record(name, record)
                .await
                .unwrap();
        }
    }
    let replay = restored.admit_remote_upload(request).await.unwrap();
    assert_eq!(replay.result, admission.result);
    assert!(replay.replayed);
    assert_eq!(table(&restored, "processing_job").await.len(), 1);
    let id = admission.result["job_id"].as_str().unwrap();
    let job = restored
        .get_remote_upload_job("a", id)
        .await
        .unwrap()
        .unwrap();
    assert!(job.service_epoch.is_none());
    assert!(job.worker_token.is_none());
    assert_eq!(job.phase, "staged");
    assert_eq!(
        restored
            .reconcile_interrupted_remote_uploads("new-epoch")
            .await
            .unwrap(),
        1
    );
    restored.resume_remote_upload_job("a", id).await.unwrap();
    let job = restored
        .claim_remote_upload_job("a", id, "new-epoch", "fresh-worker")
        .await
        .unwrap();
    let fresh = RemoteJobLease {
        job_id: job.job.id.unwrap(),
        instance_id: "a".into(),
        service_epoch: "new-epoch".into(),
        worker_token: "fresh-worker".into(),
    };
    assert!(matches!(
        restored.remote_upload_notes(&lease).await,
        Err(DbError::RemoteJobOwnershipLost(_))
    ));
    assert_eq!(restored.remote_upload_notes(&fresh).await.unwrap().len(), 1);
}

#[tokio::test]
async fn generic_local_job_controls_cannot_bypass_remote_owner_or_worker_fences() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, lease) = admit_claim(
        &repo,
        input("openclaw", "generic-bypass", "Original upload"),
        "epoch",
        "worker",
    )
    .await;
    let before = table(&repo, "processing_job").await;
    assert!(repo.cancel_processing_job(&lease.job_id).await.is_err());
    assert!(repo
        .update_processing_job(
            &lease.job_id,
            ProcessingJobUpdate {
                status: Some(ProcessingJobStatus::Failed),
                checkpoint: Some(Some("forged".into())),
                ..Default::default()
            }
        )
        .await
        .is_err());
    repo.update_reindex_item_fingerprints(&lease.job_id, Default::default())
        .await
        .unwrap();
    assert_eq!(table(&repo, "processing_job").await, before);
    repo.finish_remote_upload_job(
        &lease,
        ProcessingJobStatus::Failed,
        Some("interrupted".into()),
        None,
    )
    .await
    .unwrap();
    let failed = table(&repo, "processing_job").await;
    assert!(repo.resume_processing_job(&lease.job_id).await.is_err());
    assert_eq!(table(&repo, "processing_job").await, failed);
}

#[tokio::test]
async fn deleted_and_recreated_source_cannot_accept_old_pinned_generation() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, old) = admit_claim(
        &repo,
        input("openclaw", "old-attempt", "Old upload"),
        "epoch",
        "old-worker",
    )
    .await;
    let first = repo.begin_remote_upload_generation(&old).await.unwrap();
    repo.finish_remote_upload_job(
        &old,
        ProcessingJobStatus::Failed,
        Some("interrupted".into()),
        None,
    )
    .await
    .unwrap();
    repo.delete_source(&first).await.unwrap();
    let (_, new) = admit_claim(
        &repo,
        input("openclaw", "new-attempt", "Replacement upload"),
        "epoch",
        "new-worker",
    )
    .await;
    let replacement = repo.begin_remote_upload_generation(&new).await.unwrap();
    assert_eq!(first.id, replacement.id);
    assert_eq!(first.generation, replacement.generation);
    let sources = table(&repo, "source").await;
    assert!(matches!(
        repo.resume_remote_upload_job("openclaw", &record_id_to_string(&old.job_id))
            .await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
    assert!(repo
        .stage_remote_upload_notes(&old, vec![Note::new("Old upload")])
        .await
        .is_err());
    assert_eq!(table(&repo, "source").await, sources);
    assert!(table(&repo, "note").await.is_empty());
}

#[tokio::test]
async fn same_source_claims_wait_for_terminalization_and_follow_admission_order() {
    for legacy in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut admissions = Vec::new();
        for (request, body) in [
            ("first", "First version"),
            ("second", "Second version"),
            ("third", "Third version"),
        ] {
            admissions.push(
                repo.admit_remote_upload(input("owner", request, body))
                    .await
                    .unwrap(),
            );
        }
        let ids = admissions
            .iter()
            .map(|admission| admission.result["job_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        if legacy {
            repo.db
                .query("UPDATE processing_job UNSET remote_admission_order")
                .await
                .unwrap()
                .check()
                .unwrap();
        } else {
            // Durable order must win even when the admission clock moves back.
            repo.db
                .query("UPDATE $id SET created_at = <datetime>$created")
                .bind(("id", job_id(ids[0]).unwrap()))
                .bind((
                    "created",
                    (Utc::now() + chrono::Duration::seconds(5)).to_rfc3339(),
                ))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        assert!(matches!(
            repo.claim_remote_upload_job("owner", ids[2], "epoch", "out-of-order")
                .await,
            Err(DbError::RemoteJobOwnershipLost(_))
        ));
        let (left, right) = tokio::join!(
            repo.claim_next_remote_upload("epoch", "left"),
            repo.claim_next_remote_upload("epoch", "right")
        );
        let first = match (left.unwrap(), right.unwrap()) {
            (Some(first), None) | (None, Some(first)) => first,
            jobs => panic!("only one document owner may run before generation: {jobs:?}"),
        };
        assert_eq!(first.job.id, Some(job_id(ids[0]).unwrap()));
        assert!(first.source_generation.is_none());
        assert!(table(&repo, "source").await.is_empty());
        let lease = RemoteJobLease {
            job_id: first.job.id.unwrap(),
            instance_id: first.instance_id,
            service_epoch: first.service_epoch.unwrap(),
            worker_token: first.worker_token.unwrap(),
        };
        let source = repo.begin_remote_upload_generation(&lease).await.unwrap();
        repo.stage_remote_upload_notes(&lease, vec![Note::new("First version")])
            .await
            .unwrap();
        repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
        // Promotion does not release the source while extraction/finish owns it.
        assert!(repo
            .claim_next_remote_upload("epoch", "promoted-racer")
            .await
            .unwrap()
            .is_none());
        let foreign = repo
            .admit_remote_upload(input("other-owner", "same-key", "Foreign version"))
            .await
            .unwrap();
        let foreign_job = repo
            .claim_next_remote_upload("epoch", "foreign-worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            foreign_job.job.id,
            Some(job_id(foreign.result["job_id"].as_str().unwrap()).unwrap())
        );
        assert_ne!(foreign_job.source_uri, source.uri.clone().unwrap());
        complete(
            &repo,
            &RemoteJobLease {
                job_id: foreign_job.job.id.unwrap(),
                instance_id: foreign_job.instance_id,
                service_epoch: "epoch".into(),
                worker_token: "foreign-worker".into(),
            },
            "Foreign version",
        )
        .await;
        repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
            .await
            .unwrap();
        for (index, body) in [(1, "Second version"), (2, "Third version")] {
            let job = repo
                .claim_next_remote_upload("epoch", "next-worker")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(job.job.id, Some(job_id(ids[index]).unwrap()));
            let result = complete(
                &repo,
                &RemoteJobLease {
                    job_id: job.job.id.unwrap(),
                    instance_id: job.instance_id,
                    service_epoch: "epoch".into(),
                    worker_token: "next-worker".into(),
                },
                body,
            )
            .await;
            assert_eq!(result.source_generation, Some(index as u64 + 1));
        }
        assert_eq!(
            repo.get_source_chunks(source.id.as_ref().unwrap())
                .await
                .unwrap()[0]
                .content,
            "Third version"
        );
        for id in ids {
            assert_eq!(
                repo.get_remote_upload_job_status("owner", id)
                    .await
                    .unwrap()
                    .unwrap()
                    .job
                    .status,
                "completed"
            );
        }
    }
}

#[tokio::test]
async fn blocked_source_backlog_does_not_starve_an_unrelated_document_or_release_on_cancel_flag() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, first) = admit_claim(
        &repo,
        input("owner", "active", "Active version"),
        "epoch",
        "active-worker",
    )
    .await;
    let mut next_id = None;
    for index in 0..=MAX_REMOTE_CLAIM_SCAN {
        let admission = repo
            .admit_remote_upload(input(
                "owner",
                &format!("blocked-{index}"),
                "Queued version",
            ))
            .await
            .unwrap();
        next_id.get_or_insert_with(|| admission.result["job_id"].as_str().unwrap().to_string());
    }
    let mut unrelated_input = input("owner", "unrelated", "Unrelated version");
    unrelated_input.document_key = "unrelated.md".into();
    let unrelated = repo.admit_remote_upload(unrelated_input).await.unwrap();
    repo.cancel_remote_upload_job_status("owner", &record_id_to_string(&first.job_id))
        .await
        .unwrap();
    let claimed = repo
        .claim_next_remote_upload("epoch", "unrelated-worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        claimed.job.id,
        Some(job_id(unrelated.result["job_id"].as_str().unwrap()).unwrap())
    );
    complete(
        &repo,
        &RemoteJobLease {
            job_id: claimed.job.id.unwrap(),
            instance_id: claimed.instance_id,
            service_epoch: "epoch".into(),
            worker_token: "unrelated-worker".into(),
        },
        "Unrelated version",
    )
    .await;
    // A requested running cancellation still owns the source until settled.
    assert!(repo
        .claim_next_remote_upload("epoch", "cancel-racer")
        .await
        .unwrap()
        .is_none());
    repo.recover_remote_upload_job(&first, "cancelled")
        .await
        .unwrap();
    let next = repo
        .claim_next_remote_upload("epoch", "after-cancel")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        next.job.id,
        Some(job_id(next_id.as_deref().unwrap()).unwrap())
    );
    assert_eq!(
        repo.get_remote_upload_job_status("owner", &record_id_to_string(&first.job_id))
            .await
            .unwrap()
            .unwrap()
            .job
            .status,
        "cancelled"
    );
}

#[tokio::test]
async fn prepared_failed_owner_cannot_resume_into_a_newer_claimed_upload() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (_, older) = admit_claim(
        &repo,
        input("owner", "failed-owner", "Failed version"),
        "epoch",
        "old-worker",
    )
    .await;
    repo.begin_remote_upload_generation(&older).await.unwrap();
    repo.stage_remote_upload_notes(&older, vec![Note::new("Failed version")])
        .await
        .unwrap();
    let failed = repo
        .finish_remote_upload_job(
            &older,
            ProcessingJobStatus::Failed,
            Some("internal".into()),
            None,
        )
        .await
        .unwrap();
    let (_, newer) = admit_claim(
        &repo,
        input("owner", "new-owner", "New version"),
        "epoch",
        "new-worker",
    )
    .await;
    assert!(matches!(
        repo.resume_remote_upload_job("owner", &record_id_to_string(&older.job_id))
            .await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
    let still_failed = repo
        .get_remote_upload_job("owner", &record_id_to_string(&older.job_id))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(still_failed.job.status, "failed");
    assert_eq!(still_failed.job.item_ids, failed.job.item_ids);
    assert_eq!(still_failed.job.last_error, failed.job.last_error);
    assert_eq!(
        complete(&repo, &newer, "New version")
            .await
            .source_generation,
        Some(2)
    );
    assert!(matches!(
        repo.resume_remote_upload_job("owner", &record_id_to_string(&older.job_id))
            .await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
}

#[tokio::test]
async fn mixed_legacy_and_numbered_admissions_have_total_order_after_clock_rollback() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut admissions = Vec::new();
    for (request, body) in [
        ("number-one", "Numbered first"),
        ("number-two", "Numbered second"),
        ("legacy", "Legacy version"),
    ] {
        admissions.push(
            repo.admit_remote_upload(input("owner", request, body))
                .await
                .unwrap(),
        );
    }
    let ids = admissions
        .iter()
        .map(|admission| admission.result["job_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    let base = Utc::now();
    for (index, seconds) in [(0, 3), (1, 1), (2, 2)] {
        repo.db
            .query("UPDATE $id SET created_at = <datetime>$created")
            .bind(("id", job_id(ids[index]).unwrap()))
            .bind((
                "created",
                (base + chrono::Duration::seconds(seconds)).to_rfc3339(),
            ))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    repo.db
        .query("UPDATE $id UNSET remote_admission_order")
        .bind(("id", job_id(ids[2]).unwrap()))
        .await
        .unwrap()
        .check()
        .unwrap();
    // Pairwise timestamp fallback would cycle: number-one precedes number-two,
    // number-two precedes legacy, and legacy precedes number-one.
    for (generation, index, body) in [
        (1, 2, "Legacy version"),
        (2, 0, "Numbered first"),
        (3, 1, "Numbered second"),
    ] {
        let job = repo
            .claim_next_remote_upload("epoch", "worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.job.id, Some(job_id(ids[index]).unwrap()));
        let completed = complete(
            &repo,
            &RemoteJobLease {
                job_id: job.job.id.unwrap(),
                instance_id: job.instance_id,
                service_epoch: "epoch".into(),
                worker_token: "worker".into(),
            },
            body,
        )
        .await;
        assert_eq!(completed.source_generation, Some(generation));
    }
}

#[tokio::test]
async fn equal_legacy_timestamps_use_the_same_id_order_for_claims_and_stale_resume() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut ids = Vec::new();
    for request in ["legacy-alpha", "legacy-beta"] {
        let admission = repo
            .admit_remote_upload(input("owner", request, request))
            .await
            .unwrap();
        ids.push(admission.result["job_id"].as_str().unwrap().to_string());
    }
    ids.sort();
    repo.db.query("UPDATE processing_job SET created_at = <datetime>$created, remote_admission_order = NONE")
        .bind(("created", Utc::now().to_rfc3339())).await.unwrap().check().unwrap();
    repo.cancel_remote_upload_job_status("owner", &ids[0])
        .await
        .unwrap();
    let newest = repo
        .claim_next_remote_upload("epoch", "worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(newest.job.id, Some(job_id(&ids[1]).unwrap()));
    let completed = complete(
        &repo,
        &RemoteJobLease {
            job_id: newest.job.id.unwrap(),
            instance_id: newest.instance_id,
            service_epoch: "epoch".into(),
            worker_token: "worker".into(),
        },
        "Newest version",
    )
    .await;
    assert!(matches!(
        repo.resume_remote_upload_job("owner", &ids[0]).await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
    let source = repo
        .get_source(&completed.source_uri)
        .await
        .unwrap()
        .unwrap();
    repo.delete_source(&source).await.unwrap();
    assert!(matches!(
        repo.resume_remote_upload_job("owner", &ids[0]).await,
        Err(DbError::RemoteJobSourceConflict(_))
    ));
}

#[tokio::test]
async fn older_unprepared_admission_cannot_replace_a_newer_document() {
    for already_claimed in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let old = repo
            .admit_remote_upload(input("openclaw", "older", "Old document"))
            .await
            .unwrap();
        let old_id = old.result["job_id"].as_str().unwrap();
        let old_lease = if already_claimed {
            let row = repo
                .claim_remote_upload_job("openclaw", old_id, "epoch", "old-worker")
                .await
                .unwrap();
            Some(RemoteJobLease {
                job_id: row.job.id.unwrap(),
                instance_id: "openclaw".into(),
                service_epoch: "epoch".into(),
                worker_token: "old-worker".into(),
            })
        } else {
            repo.cancel_remote_upload_job("openclaw", old_id)
                .await
                .unwrap();
            None
        };
        let newer = repo
            .admit_remote_upload(input("openclaw", "newer", "New document"))
            .await
            .unwrap();
        let newer_id = newer.result["job_id"].as_str().unwrap();
        if already_claimed {
            // Pre-fix overlapping claims could be restored from an older
            // journal. Simulate that state directly; current scheduling must
            // never create it, while the generation fence must still reject it.
            repo.db
                .query("UPDATE $id SET status = 'running', remote_service_epoch = 'epoch', remote_worker_token = 'new-worker'")
                .bind(("id", job_id(newer_id).unwrap()))
                .await.unwrap().check().unwrap();
        } else {
            repo.claim_remote_upload_job("openclaw", newer_id, "epoch", "new-worker")
                .await
                .unwrap();
        }
        let newer = RemoteJobLease {
            job_id: job_id(newer_id).unwrap(),
            instance_id: "openclaw".into(),
            service_epoch: "epoch".into(),
            worker_token: "new-worker".into(),
        };
        let saved = complete(&repo, &newer, "New document").await;
        assert_eq!(saved.admission_order, Some(2));
        let sources = table(&repo, "source").await;
        let notes = table(&repo, "note").await;
        if let Some(lease) = old_lease {
            assert!(matches!(
                repo.begin_remote_upload_generation(&lease).await,
                Err(DbError::RemoteJobSourceConflict(_))
            ));
        } else {
            assert!(matches!(
                repo.resume_remote_upload_job("openclaw", old_id).await,
                Err(DbError::RemoteJobSourceConflict(_))
            ));
            // Restored pre-fix journals without an admission counter still
            // compare their original immutable timestamps instead of bypassing.
            repo.db
                .query("UPDATE $id UNSET remote_admission_order")
                .bind(("id", job_id(old_id).unwrap()))
                .await
                .unwrap()
                .check()
                .unwrap();
            assert!(matches!(
                repo.resume_remote_upload_job("openclaw", old_id).await,
                Err(DbError::RemoteJobSourceConflict(_))
            ));
        }
        assert_eq!(table(&repo, "source").await, sources);
        assert_eq!(table(&repo, "note").await, notes);
    }
}

#[tokio::test]
async fn cancellation_accepted_during_lifecycle_wait_wins_terminal_transition() {
    for requested in [ProcessingJobStatus::Failed, ProcessingJobStatus::Completed] {
        let repo = Repository::new(init_memory().await.unwrap());
        let (_, lease) = admit_claim(
            &repo,
            input("openclaw", "cancel-terminal", "Prepared upload"),
            "epoch",
            "worker",
        )
        .await;
        if requested == ProcessingJobStatus::Completed {
            repo.begin_remote_upload_generation(&lease).await.unwrap();
            repo.stage_remote_upload_notes(&lease, vec![Note::new("Prepared upload")])
                .await
                .unwrap();
            repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
        }
        let lifecycle = repo.proposal_acceptance_lock.lock().await;
        let clone = repo.clone();
        let finishing = lease.clone();
        let task = tokio::spawn(async move {
            clone
                .finish_remote_upload_job(
                    &finishing,
                    requested,
                    Some("provider_unavailable".into()),
                    Some(serde_json::json!({"completed":true})),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if repo.remote_job_transition_lock.try_lock().is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // The finalizer has entered its transition phase and cannot pass the
        // held lifecycle gate. Cancellation remains an independent DB action.
        tokio::time::sleep(Duration::from_millis(30)).await;
        let cancelled = repo
            .cancel_remote_upload_job("openclaw", &record_id_to_string(&lease.job_id))
            .await
            .unwrap();
        assert!(cancelled.cancel_requested);
        assert_eq!(cancelled.job.status, "running");
        assert!(!task.is_finished());
        drop(lifecycle);
        let terminal = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(terminal.job.status, "cancelled");
        assert_eq!(terminal.job.last_error.as_deref(), Some("cancelled"));
        assert!(terminal.result.is_none());
        assert!(terminal.service_epoch.is_none());
        assert!(terminal.worker_token.is_none());
    }
}

#[tokio::test]
async fn malformed_queued_input_is_rejected_before_claim_and_repair_needs_no_restart() {
    for malformed in [
        serde_json::json!({"missing_fields":true}),
        serde_json::json!({"semantic_blank":true}),
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        let original = input("owner", "repair-before-claim", "Valid uploaded input");
        let admission = repo.admit_remote_upload(original.clone()).await.unwrap();
        let id = admission.result["job_id"].as_str().unwrap();
        let invalid = if malformed.get("semantic_blank").is_some() {
            let mut invalid = original.clone();
            invalid.markdown = "   ".into();
            serde_json::to_value(invalid).unwrap()
        } else {
            malformed
        };
        repo.db
            .query("UPDATE $id SET remote_input = $input")
            .bind(("id", job_id(id).unwrap()))
            .bind(("input", invalid))
            .await
            .unwrap()
            .check()
            .unwrap();
        let damaged = table(&repo, "processing_job").await;
        assert!(repo
            .claim_remote_upload_job("owner", id, "epoch", "worker")
            .await
            .is_err());
        assert_eq!(table(&repo, "processing_job").await, damaged);
        assert!(repo
            .claim_next_remote_upload("epoch", "worker")
            .await
            .unwrap()
            .is_none());
        let quarantined = table(&repo, "processing_job").await;
        assert_eq!(quarantined[0]["status"], "failed");
        assert_eq!(quarantined[0]["last_error"], "validation");
        assert_eq!(quarantined[0]["remote_input"], damaged[0]["remote_input"]);
        assert!(quarantined[0].get("remote_service_epoch").is_none());
        assert!(quarantined[0].get("remote_worker_token").is_none());
        repo.db
            .query("UPDATE $id SET remote_input = $input")
            .bind(("id", job_id(id).unwrap()))
            .bind(("input", serde_json::to_value(&original).unwrap()))
            .await
            .unwrap()
            .check()
            .unwrap();
        // Repair alone must never silently restart a quarantined admission.
        assert!(repo
            .claim_next_remote_upload("epoch", "worker")
            .await
            .unwrap()
            .is_none());
        repo.resume_remote_upload_job("owner", id).await.unwrap();
        let claimed = repo
            .claim_next_remote_upload("epoch", "worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claimed.job.status, "running");
        assert_eq!(claimed.input, original);
        assert_eq!(claimed.job.id, Some(job_id(id).unwrap()));
        assert_eq!(claimed.worker_token.as_deref(), Some("worker"));
    }
}

#[tokio::test]
async fn malformed_restored_input_is_quarantined_without_blocking_a_healthy_upload() {
    for malformed in [
        serde_json::json!({"missing_fields":true}),
        serde_json::json!({"semantic_blank":true}),
    ] {
        let original = Repository::new(init_memory().await.unwrap());
        let invalid_input = input("owner", "damaged-oldest", "Original admitted input");
        let damaged = original
            .admit_remote_upload(invalid_input.clone())
            .await
            .unwrap();
        let healthy_input = input("other-owner", "healthy-next", "Healthy uploaded input");
        let healthy = original
            .admit_remote_upload(healthy_input.clone())
            .await
            .unwrap();
        let damaged_id = damaged.result["job_id"].as_str().unwrap();
        let healthy_id = healthy.result["job_id"].as_str().unwrap();
        let restored = Repository::new(init_memory().await.unwrap());
        for mut record in table(&original, "processing_job").await {
            if record["remote_request_id"] == "damaged-oldest" {
                record["remote_input"] = if malformed.get("semantic_blank").is_some() {
                    let mut invalid = invalid_input.clone();
                    invalid.markdown = "   ".into();
                    serde_json::to_value(invalid).unwrap()
                } else {
                    malformed.clone()
                };
            }
            restored
                .restore_portable_record("processing_job", record)
                .await
                .unwrap();
        }
        // A failed quarantine write remains a storage error. It cannot be
        // treated as settled input corruption or acquire an untracked lease.
        restored
            .db
            .query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string ASSERT $value != 'failed'")
            .await
            .unwrap()
            .check()
            .unwrap();
        assert!(restored
            .claim_next_remote_upload("restored-epoch", "healthy-worker")
            .await
            .is_err());
        for record in table(&restored, "processing_job").await {
            assert_eq!(record["status"], "queued");
            assert!(record.get("remote_service_epoch").is_none());
            assert!(record.get("remote_worker_token").is_none());
        }
        restored
            .db
            .query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string")
            .await
            .unwrap()
            .check()
            .unwrap();
        let claimed = restored
            .claim_next_remote_upload("restored-epoch", "healthy-worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claimed.job.id, Some(job_id(healthy_id).unwrap()));
        assert_eq!(claimed.input, healthy_input);
        let damaged = table(&restored, "processing_job")
            .await
            .into_iter()
            .find(|record| record["remote_request_id"] == "damaged-oldest")
            .unwrap();
        assert_eq!(damaged["status"], "failed");
        assert_eq!(damaged["last_error"], "validation");
        assert!(damaged.get("remote_service_epoch").is_none());
        assert!(damaged.get("remote_worker_token").is_none());
        let lease = RemoteJobLease {
            job_id: claimed.job.id.unwrap(),
            instance_id: claimed.instance_id,
            service_epoch: "restored-epoch".into(),
            worker_token: "healthy-worker".into(),
        };
        assert_eq!(
            complete(&restored, &lease, "Healthy uploaded chunk")
                .await
                .job
                .status,
            "completed"
        );
        // A different authenticated instance cannot resume the quarantined job.
        assert!(matches!(
            restored
                .resume_remote_upload_job("other-owner", damaged_id)
                .await,
            Err(DbError::NotFound(_, _))
        ));
    }
}

#[tokio::test]
async fn quarantined_input_status_is_readable_without_exposing_other_owners() {
    let original = Repository::new(init_memory().await.unwrap());
    let damaged = original
        .admit_remote_upload(input("owner", "damaged-status", "Damaged admitted input"))
        .await
        .unwrap();
    let healthy = original
        .admit_remote_upload(input("owner", "healthy-status", "Healthy admitted input"))
        .await
        .unwrap();
    let damaged_id = damaged.result["job_id"].as_str().unwrap();
    let healthy_id = healthy.result["job_id"].as_str().unwrap();
    let restored = Repository::new(init_memory().await.unwrap());
    for mut record in table(&original, "processing_job").await {
        if record["remote_request_id"] == "damaged-status" {
            record["remote_input"] = serde_json::json!({"missing_fields":true});
        }
        restored
            .restore_portable_record("processing_job", record)
            .await
            .unwrap();
    }
    let claimed = restored
        .claim_next_remote_upload("restored-epoch", "healthy-worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claimed.job.id, Some(job_id(healthy_id).unwrap()));
    let foreign = restored
        .admit_remote_upload(input(
            "other-owner",
            "foreign-status",
            "Foreign admitted input",
        ))
        .await
        .unwrap();
    let foreign_id = foreign.result["job_id"].as_str().unwrap();
    let damaged = restored
        .get_remote_upload_job_status("owner", damaged_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(damaged.job.status, "failed");
    assert_eq!(damaged.job.last_error.as_deref(), Some("validation"));
    assert!(damaged.job.finished_at.is_some());
    let jobs = restored
        .list_remote_upload_job_statuses("owner", 10)
        .await
        .unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs
        .iter()
        .any(|job| job.job.id == Some(job_id(damaged_id).unwrap())));
    assert!(jobs
        .iter()
        .any(|job| job.job.id == Some(job_id(healthy_id).unwrap())));
    assert!(restored
        .get_remote_upload_job_status("other-owner", damaged_id)
        .await
        .unwrap()
        .is_none());
    assert!(restored
        .get_remote_upload_job_status("owner", foreign_id)
        .await
        .unwrap()
        .is_none());
    let other_jobs = restored
        .list_remote_upload_job_statuses("other-owner", 10)
        .await
        .unwrap();
    assert_eq!(other_jobs.len(), 1);
    assert_eq!(other_jobs[0].job.id, Some(job_id(foreign_id).unwrap()));
    assert!(matches!(
        restored
            .cancel_remote_upload_job_status("other-owner", damaged_id)
            .await,
        Err(DbError::NotFound(_, _))
    ));
    // Cancellation of an already quarantined job is a readable no-op. It
    // neither erases its validation error nor queues an unrepaired input.
    let cancelled = restored
        .cancel_remote_upload_job_status("owner", damaged_id)
        .await
        .unwrap();
    assert_eq!(cancelled.job.status, "failed");
    assert_eq!(cancelled.job.last_error.as_deref(), Some("validation"));
    assert!(!cancelled.cancel_requested);
    assert_eq!(cancelled.job.updated_at, damaged.job.updated_at);
    // The status API never substitutes an executable payload for corruption.
    assert!(matches!(
        restored.get_remote_upload_job("owner", damaged_id).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
    assert!(matches!(
        restored.resume_remote_upload_job("owner", damaged_id).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
    assert_eq!(
        restored
            .get_remote_upload_job_status("owner", damaged_id)
            .await
            .unwrap()
            .unwrap()
            .job
            .status,
        "failed"
    );
    for limit in [0, 201] {
        assert!(matches!(
            restored
                .list_remote_upload_job_statuses("owner", limit)
                .await,
            Err(DbError::InvalidRemoteRequest(_))
        ));
    }
}

#[tokio::test]
async fn malformed_queued_input_can_be_cancelled_without_decoding_it() {
    let repo = Repository::new(init_memory().await.unwrap());
    let admission = repo
        .admit_remote_upload(input("owner", "damaged-cancel", "Admitted input"))
        .await
        .unwrap();
    let id = admission.result["job_id"].as_str().unwrap();
    let damaged_input = serde_json::json!({"missing_fields":true});
    repo.db
        .query("UPDATE $id SET remote_input = $input")
        .bind(("id", job_id(id).unwrap()))
        .bind(("input", damaged_input.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
    let cancelled = repo
        .cancel_remote_upload_job_status("owner", id)
        .await
        .unwrap();
    assert_eq!(cancelled.job.status, "cancelled");
    assert!(cancelled.cancel_requested);
    assert!(cancelled.job.finished_at.is_some());
    assert_eq!(
        table(&repo, "processing_job").await[0]["remote_input"],
        damaged_input
    );
    assert!(repo
        .claim_next_remote_upload("epoch", "worker")
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        repo.resume_remote_upload_job("owner", id).await,
        Err(DbError::InvalidRemoteRequest(_))
    ));
}

#[tokio::test]
async fn minimal_recovery_fence_settles_damaged_post_claim_input_without_restarting() {
    for cancelled in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let original = input("owner", "damaged-after-claim", "Valid input before claim");
        let (_, lease) = admit_claim(&repo, original.clone(), "epoch", "worker").await;
        repo.db
            .query("UPDATE $id SET remote_input = $input, remote_cancel_requested = $cancel")
            .bind(("id", lease.job_id.clone()))
            .bind(("input", serde_json::json!({"damaged_after_claim":true})))
            .bind(("cancel", cancelled))
            .await
            .unwrap()
            .check()
            .unwrap();
        assert!(repo
            .get_remote_upload_job("owner", &record_id_to_string(&lease.job_id))
            .await
            .is_err());
        repo.recover_remote_upload_job(&lease, "internal")
            .await
            .unwrap();
        let rows = table(&repo, "processing_job").await;
        assert_eq!(
            rows[0]["status"],
            if cancelled { "cancelled" } else { "failed" }
        );
        assert_eq!(
            rows[0]["last_error"],
            if cancelled { "cancelled" } else { "internal" }
        );
        assert!(rows[0].get("remote_service_epoch").is_none());
        assert!(rows[0].get("remote_worker_token").is_none());
        assert!(table(&repo, "source").await.is_empty());
        assert!(table(&repo, "note").await.is_empty());
        repo.db
            .query("UPDATE $id SET remote_input = $input")
            .bind(("id", lease.job_id.clone()))
            .bind(("input", serde_json::to_value(original).unwrap()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let resumed = repo
            .resume_remote_upload_job("owner", &record_id_to_string(&lease.job_id))
            .await
            .unwrap();
        assert_eq!(resumed.job.status, "queued");
        assert!(repo
            .claim_next_remote_upload("epoch", "replacement-worker")
            .await
            .unwrap()
            .is_some());
        // The old fence can never terminalize the new owner's running attempt.
        repo.recover_remote_upload_job(&lease, "old-worker-error")
            .await
            .unwrap();
        assert_eq!(
            repo.get_remote_upload_job("owner", &record_id_to_string(&lease.job_id))
                .await
                .unwrap()
                .unwrap()
                .worker_token
                .as_deref(),
            Some("replacement-worker")
        );
    }
}
