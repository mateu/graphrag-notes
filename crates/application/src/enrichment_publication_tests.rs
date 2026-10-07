use super::*;
use graphrag_core::Note;
use graphrag_db::{init_memory, Repository, SourceEnrichmentPlan};

async fn staged_conversion() -> (Repository, SourceEnrichmentPlan) {
    let repo = Repository::new(init_memory().await.unwrap());
    let processing_options = json!({"runtime":"{}","embedding":{"provider":"fixture","model":"fixture","cache_identity":"fixture","endpoint_identity":"fixture"},"extraction":{"provider":"fixture","model":"fixture","cache_identity":"fixture","endpoint_identity":"fixture"}});
    let input = RemoteUploadInput {
        authenticated_instance_id: "fixture".into(),
        request_id: "original".into(),
        payload_fingerprint: format!("{:x}", Sha256::digest(b"Atlas")),
        document_key: "atlas.md".into(),
        markdown: "Atlas".into(),
        title: Some("Atlas".into()),
        source_provenance: json!({}),
        extract_entities: false,
        preserve_unchanged: false,
        create_only: false,
        expected_source_revision: None,
        policy_migration: None,
        enrichment: None,
        processing_options: processing_options.clone(),
    };
    let admitted = repo.admit_remote_upload(input).await.unwrap();
    let job = repo
        .claim_remote_upload_job(
            "fixture",
            admitted.result["job_id"].as_str().unwrap(),
            "epoch",
            "worker",
        )
        .await
        .unwrap();
    let lease = RemoteJobLease {
        job_id: job.job.id.unwrap(),
        instance_id: "fixture".into(),
        service_epoch: "epoch".into(),
        worker_token: "worker".into(),
    };
    repo.begin_remote_upload_generation(&lease).await.unwrap();
    repo.stage_remote_upload_notes(&lease, vec![Note::new("Atlas")])
        .await
        .unwrap();
    repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
    repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    let source = repo
        .get_source(admitted.result["source_uri"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    let plan = SourceEnrichmentPlan {
        instance_id: "fixture".into(),
        request_id: "reviewed".into(),
        source_id: record_id_to_string(source.id.as_ref().unwrap()),
        document_key: "atlas.md".into(),
        expected_source_revision: graphrag_db::uploaded_source_revision(&source, "Atlas").unwrap(),
        expected_content_sha256: source.content_hash.unwrap(),
        expected_generation: source.generation,
        original_policy_sha256: graphrag_db::uploaded_processing_policy_sha256(&processing_options)
            .unwrap(),
        target_processing_options: processing_options,
        confirmed: true,
    };
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    repo.checkpoint_source_entity_enrichment(&plan, 0, vec![])
        .await
        .unwrap();
    (repo, plan)
}

#[tokio::test]
async fn cancellation_after_final_preflight_returns_actual_committed_policy_epoch() {
    let (repo, plan) = staged_conversion().await;
    let cancellation = ActionCancellation::new();
    let committed = promote_after_final_preflight(
        || {
            assert!(!cancellation.is_cancelled());
            Ok(())
        },
        || {
            // Runs only AFTER the final preflight. Deterministic cancellation
            // now precedes entry into the real datastore transaction.
            cancellation.cancel();
            async { Ok(repo.promote_source_entity_enrichment(&plan).await?) }
        },
    )
    .await
    .unwrap();
    assert!(cancellation.is_cancelled());
    assert_eq!(committed.status, "promoted");
    let source = repo.get_source(&plan.source_id).await.unwrap().unwrap();
    assert_eq!(
        source.metadata["entity_enrichment_v1"]["status"],
        "promoted"
    );
    assert_eq!(source.metadata["graph_policy_revision"], 1);
    assert_eq!(source.metadata["remote_upload"]["extract_entities"], true);
    assert!(repo
        .source_entity_enrichment_current(&source)
        .await
        .unwrap());
    assert_eq!(
        repo.promote_source_entity_enrichment(&plan).await.unwrap(),
        committed
    );
}

#[tokio::test]
async fn cancellation_before_publication_preflight_never_enters_transaction() {
    let (repo, plan) = staged_conversion().await;
    let cancellation = ActionCancellation::new();
    cancellation.cancel();
    let entered = std::sync::atomic::AtomicBool::new(false);
    let result = promote_after_final_preflight(
        || {
            if cancellation.is_cancelled() {
                Err(ApplicationError::Cancelled)
            } else {
                Ok(())
            }
        },
        || {
            entered.store(true, std::sync::atomic::Ordering::SeqCst);
            async { Ok(repo.promote_source_entity_enrichment(&plan).await?) }
        },
    )
    .await;
    assert!(matches!(result, Err(ApplicationError::Cancelled)));
    assert!(!entered.load(std::sync::atomic::Ordering::SeqCst));
    let source = repo.get_source(&plan.source_id).await.unwrap().unwrap();
    assert_eq!(source.metadata["entity_enrichment_v1"]["status"], "staged");
    assert_eq!(source.metadata["remote_upload"]["extract_entities"], false);
    assert!(source.metadata["graph_policy_revision"].is_null());
}
