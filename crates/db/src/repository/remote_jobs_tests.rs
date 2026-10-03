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
        let (_, newer) = admit_claim(
            &repo,
            input("openclaw", "newer", "New document"),
            "epoch",
            "new-worker",
        )
        .await;
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
