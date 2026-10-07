use super::*;
use crate::init_memory;
use graphrag_core::EntityType;
use sha2::{Digest, Sha256};

fn input(request: &str, body: &str) -> RemoteUploadInput {
    RemoteUploadInput {
        authenticated_instance_id: "fixture".into(),
        request_id: request.into(),
        payload_fingerprint: format!("{:x}", Sha256::digest(body.as_bytes())),
        document_key: "atlas.md".into(),
        markdown: body.into(),
        title: Some("Atlas".into()),
        source_provenance: serde_json::json!({"metadata":{"collection_id":"fictional"}}),
        extract_entities: false,
        preserve_unchanged: false,
        create_only: false,
        expected_source_revision: None,
        policy_migration: None,
        processing_options: serde_json::json!({"runtime":"{}","embedding":{"provider":"fixture","model":"fixture","cache_identity":"fixture","endpoint_identity":"fixture"},"extraction":{"provider":"fixture","model":"fixture","cache_identity":"fixture","endpoint_identity":"fixture"}}),
    }
}
async fn upload(repo: &Repository, input: RemoteUploadInput, bodies: &[&str]) -> Source {
    let extraction = input.extract_entities;
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
    repo.stage_remote_upload_notes(&lease, bodies.iter().map(|body| Note::new(*body)).collect())
        .await
        .unwrap();
    let staged = repo.owned_remote_upload_job(&lease).await.unwrap();
    if staged.phase == "migration_staged" {
        repo.prepare_policy_migration_extraction(&lease, &[])
            .await
            .unwrap();
        let chunks = repo.remote_upload_notes(&lease).await.unwrap();
        for index in 0..chunks.len() {
            repo.checkpoint_policy_migration_entities(&lease, index, vec![])
                .await
                .unwrap();
        }
        repo.promote_policy_migration(&lease, &[]).await.unwrap();
    } else {
        repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
    }
    if extraction && staged.phase != "migration_staged" {
        let chunks = repo.remote_upload_notes(&lease).await.unwrap();
        for index in 0..chunks.len() {
            repo.persist_remote_upload_entities(&lease, index, vec![])
                .await
                .unwrap();
        }
    }
    repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, None)
        .await
        .unwrap();
    repo.get_source(admitted.result["source_uri"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap()
}
fn plan(source: &Source, body: &str) -> SourceEnrichmentPlan {
    SourceEnrichmentPlan {
        instance_id: "fixture".into(),
        request_id: "reviewed-1".into(),
        source_id: record_id_to_string(source.id.as_ref().unwrap()),
        document_key: "atlas.md".into(),
        expected_source_revision: uploaded_source_revision(source, body).unwrap(),
        expected_content_sha256: source.content_hash.clone().unwrap(),
        expected_generation: source.generation,
        original_policy_sha256: uploaded_processing_policy_sha256(
            &source.metadata["remote_upload"]["processing_options"],
        )
        .unwrap(),
        target_processing_options: source.metadata["remote_upload"]["processing_options"].clone(),
        confirmed: true,
    }
}
async fn rows(repo: &Repository, table: &str) -> Vec<serde_json::Value> {
    repo.db
        .query(format!("SELECT * FROM {table} ORDER BY id"))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}
fn entity(note: &Note, name: &str) -> Entity {
    let mut entity = Entity::new(name, EntityType::Concept);
    let scope = record_id_to_string(note.source_id.as_ref().unwrap());
    let identity =
        serde_json::json!([scope, entity.entity_type, entity.canonical_name]).to_string();
    entity.identity_key = Some(format!(
        "extracted-v1:{}",
        graphrag_core::normalized_content_hash(&identity)
    ));
    entity.metadata = serde_json::json!({"extraction":{"scope":scope}, "aliases":[]});
    entity
}
async fn prepared(repo: &Repository, plan: &SourceEnrichmentPlan) -> Vec<Note> {
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    let notes = repo.source_entity_enrichment_notes(plan).await.unwrap();
    for (index, note) in notes.iter().enumerate() {
        repo.checkpoint_source_entity_enrichment(plan, index, vec![entity(note, "Atlas")])
            .await
            .unwrap();
    }
    notes
}
#[tokio::test]
async fn checkpoint_resume_atomic_promotion_preserves_identity_generation_and_vectors() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let source = upload(
        &repo,
        input("original", "Atlas one and two"),
        &["Atlas one", "Atlas two"],
    )
    .await;
    let plan = plan(&source, "Atlas one and two");
    let opening = rows(&repo, "note").await;
    let staged = repo
        .begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    assert_eq!(staged.total, 2);
    let notes = repo.source_entity_enrichment_notes(&plan).await.unwrap();
    assert!(repo.promote_source_entity_enrichment(&plan).await.is_err());
    assert!(repo
        .checkpoint_source_entity_enrichment(&plan, 1, vec![])
        .await
        .is_err());
    let entities = vec![entity(&notes[0], "Atlas")];
    repo.checkpoint_source_entity_enrichment(&plan, 0, entities.clone())
        .await
        .unwrap();
    repo.checkpoint_source_entity_enrichment(&plan, 0, entities)
        .await
        .unwrap();
    assert!(repo
        .checkpoint_source_entity_enrichment(&plan, 0, vec![])
        .await
        .is_err());
    assert!(rows(&repo, "mentions").await.is_empty());
    assert!(rows(&repo, "entity").await.is_empty());
    let saved = repo
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        uploaded_source_revision(&saved, "Atlas one and two").unwrap(),
        plan.expected_source_revision
    );
    assert!(!source_entity_enrichment_current(&saved));
    let resumed = Repository::new(db);
    assert_eq!(
        resumed
            .begin_source_entity_enrichment(plan.clone())
            .await
            .unwrap()
            .completed,
        1
    );
    resumed
        .checkpoint_source_entity_enrichment(&plan, 1, vec![entity(&notes[1], "Atlas")])
        .await
        .unwrap();
    let done = resumed
        .promote_source_entity_enrichment(&plan)
        .await
        .unwrap();
    assert_eq!(done.status, "promoted");
    assert_eq!(
        resumed
            .promote_source_entity_enrichment(&plan)
            .await
            .unwrap(),
        done
    );
    assert_eq!(rows(&resumed, "note").await, opening);
    assert_eq!(rows(&resumed, "mentions").await.len(), 2);
    assert_eq!(rows(&resumed, "entity").await.len(), 1);
    let after = resumed
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.id, source.id);
    assert_eq!(after.generation, source.generation);
    assert_eq!(after.successful_generation, source.successful_generation);
    assert_eq!(after.content_hash, source.content_hash);
    assert!(source_entity_enrichment_current(&after));
    assert_ne!(
        uploaded_source_revision(&after, "Atlas one and two").unwrap(),
        plan.expected_source_revision
    );
}
#[tokio::test]
async fn reviewed_fences_and_overlay_fail_closed_without_effects() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let original = plan(&source, "Atlas");
    for field in 0..8 {
        let mut bad = original.clone();
        match field {
            0 => bad.confirmed = false,
            1 => bad.instance_id = "other".into(),
            2 => bad.document_key = "other".into(),
            3 => bad.expected_generation += 1,
            4 => bad.expected_content_sha256 = format!("sha256:{}", "a".repeat(64)),
            5 => bad.expected_source_revision = "b".repeat(64),
            6 => bad.original_policy_sha256 = "c".repeat(64),
            _ => bad.target_processing_options["embedding"]["model"] = "changed".into(),
        }
        assert!(repo.begin_source_entity_enrichment(bad).await.is_err());
    }
    assert!(repo
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap()
        .metadata
        .get(KEY)
        .is_none());
    let notes = repo
        .get_source_chunks(source.id.as_ref().unwrap())
        .await
        .unwrap();
    let entity = repo
        .upsert_entity({
            let mut e = Entity::new("Overlay", EntityType::Concept);
            e.metadata = serde_json::json!({});
            e
        })
        .await
        .unwrap();
    repo.db
        .query("CREATE mentions SET in=$note,out=$entity")
        .bind(("note", notes[0].id.clone()))
        .bind(("entity", entity.id))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo.begin_source_entity_enrichment(original).await.is_err());
    assert_eq!(rows(&repo, "mentions").await.len(), 1);
}
#[tokio::test]
async fn stale_chunk_and_bad_entities_prevent_complete_publication() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    let notes = repo.source_entity_enrichment_notes(&plan).await.unwrap();
    let mut bad = entity(&notes[0], "Atlas");
    bad.identity_key = Some("forged".into());
    assert!(repo
        .checkpoint_source_entity_enrichment(&plan, 0, vec![bad])
        .await
        .is_err());
    repo.checkpoint_source_entity_enrichment(&plan, 0, vec![entity(&notes[0], "Atlas")])
        .await
        .unwrap();
    repo.db
        .query("UPDATE $note SET content='Changed after checkpoint'")
        .bind(("note", notes[0].id.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo.promote_source_entity_enrichment(&plan).await.is_err());
    assert!(rows(&repo, "mentions").await.is_empty());
    assert!(rows(&repo, "entity").await.is_empty());
}
#[tokio::test]
async fn rollback_requires_confirmation_preserves_content_and_prevents_policy_aba() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    let opening = rows(&repo, "note").await;
    prepared(&repo, &plan).await;
    repo.promote_source_entity_enrichment(&plan).await.unwrap();
    assert!(repo
        .rollback_source_entity_enrichment(&plan, false)
        .await
        .is_err());
    let result = repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .unwrap();
    assert_eq!(result.status, "rolled_back");
    assert_eq!(
        repo.rollback_source_entity_enrichment(&plan, true)
            .await
            .unwrap(),
        result
    );
    assert!(rows(&repo, "mentions").await.is_empty());
    assert_eq!(rows(&repo, "note").await, opening);
    let after = repo
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert!(!source_entity_enrichment_current(&after));
    assert_eq!(
        after.metadata["remote_upload"],
        source.metadata["remote_upload"]
    );
    assert_eq!(after.metadata["graph_policy_revision"], 2);
    assert_ne!(
        uploaded_source_revision(&after, "Atlas").unwrap(),
        plan.expected_source_revision
    );
    assert!(repo.promote_source_entity_enrichment(&plan).await.is_err());
}
#[tokio::test]
async fn rollback_refuses_foreign_mentions_or_relationship_dependents() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    let notes = prepared(&repo, &plan).await;
    repo.promote_source_entity_enrichment(&plan).await.unwrap();
    repo.db
        .query("UPDATE mentions SET metadata={foreign:true}")
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .is_err());
    assert_eq!(rows(&repo, "mentions").await.len(), 1);
    repo.db
        .query("UPDATE mentions SET metadata=$metadata")
        .bind(("metadata", entity(&notes[0], "Atlas").metadata))
        .await
        .unwrap()
        .check()
        .unwrap();
    let other = repo.create_note(Note::new("Other")).await.unwrap();
    repo.create_edge(
        notes[0].id.as_ref().unwrap(),
        other.id.as_ref().unwrap(),
        EdgeType::Supports,
        None,
    )
    .await
    .unwrap();
    assert!(repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .is_err());
}
#[tokio::test]
async fn source_replacement_invalidates_conversion_and_removes_previous_mentions() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    prepared(&repo, &plan).await;
    repo.promote_source_entity_enrichment(&plan).await.unwrap();
    let mut replacement = input("replacement", "New content");
    replacement.extract_entities = true;
    let converted = repo
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    replacement.expected_source_revision =
        Some(uploaded_source_revision(&converted, "Atlas").unwrap());
    let replaced = upload(&repo, replacement, &["New content"]).await;
    assert_eq!(replaced.id, source.id);
    assert_eq!(replaced.generation, 2);
    assert!(!source_entity_enrichment_current(&replaced));
    assert!(replaced.metadata.get(KEY).is_none());
    assert!(rows(&repo, "mentions").await.is_empty());
    assert!(repo.promote_source_entity_enrichment(&plan).await.is_err());
    assert!(repo
        .rollback_source_entity_enrichment(&plan, true)
        .await
        .is_err());
}
#[tokio::test]
async fn concurrent_opening_plans_bind_one_exact_review() {
    let db = init_memory().await.unwrap();
    let a = Repository::new(db.clone());
    let b = Repository::new(db);
    let source = upload(&a, input("original", "Atlas"), &["Atlas"]).await;
    let first = plan(&source, "Atlas");
    let mut second = first.clone();
    second.request_id = "different-review".into();
    let (left, right) = tokio::join!(
        a.begin_source_entity_enrichment(first),
        b.begin_source_entity_enrichment(second)
    );
    assert_ne!(left.is_ok(), right.is_ok());
    assert!(rows(&a, "mentions").await.is_empty());
}

#[tokio::test]
async fn transactional_failure_rolls_back_policy_entity_and_mention_writes() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    let notes = prepared(&repo, &plan).await;
    let source = repo.enrichment_source(&plan).await.unwrap();
    let before = rows(&repo, "source").await;
    let mut metadata = source.metadata.clone();
    let mut staged = stage(&source, &plan).unwrap();
    staged.status = "promoted".into();
    metadata[KEY] = stage_value(&staged).unwrap();
    let effects=format!("LET $note = $notes[0].id; LET $replacement_entities=$entity_batches[0]; LET $replacement_entity_names=$identity_batches[0]; {} {} THROW 'injected promotion failure';",super::super::notes::replacement_entities_transaction(),super::super::notes::replacement_mentions_transaction("$note"));
    assert!(repo
        .write_enrichment(&source, metadata, &effects, notes)
        .await
        .is_err());
    assert_eq!(rows(&repo, "source").await, before);
    assert!(rows(&repo, "entity").await.is_empty());
    assert!(rows(&repo, "mentions").await.is_empty());
    repo.promote_source_entity_enrichment(&plan).await.unwrap();
}
#[tokio::test]
async fn new_upload_cannot_implicitly_downgrade_enrichment() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    prepared(&repo, &plan).await;
    repo.promote_source_entity_enrichment(&plan).await.unwrap();
    let before = rows(&repo, "source").await;
    let mentions = rows(&repo, "mentions").await;
    let request = input("downgrade", "New content");
    let admitted = repo.admit_remote_upload(request).await.unwrap();
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
    assert!(repo.begin_remote_upload_generation(&lease).await.is_err());
    assert_eq!(rows(&repo, "source").await, before);
    assert_eq!(rows(&repo, "mentions").await, mentions);
}

#[tokio::test]
async fn malformed_checkpoint_shapes_never_claim_graph_freshness_or_publish() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = upload(&repo, input("original", "Atlas"), &["Atlas"]).await;
    let plan = plan(&source, "Atlas");
    repo.begin_source_entity_enrichment(plan.clone())
        .await
        .unwrap();
    let saved = repo
        .get_source(source.uri.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    for case in 0..6 {
        let mut metadata = saved.metadata.clone();
        match case {
            0 => metadata[KEY]["items"] = serde_json::json!([]),
            1 => {
                let item = metadata[KEY]["items"][0].clone();
                metadata[KEY]["items"].as_array_mut().unwrap().push(item);
            }
            2 => metadata[KEY]["unexpected"] = true.into(),
            3 => metadata[KEY]["status"] = "invented".into(),
            4 => metadata[KEY]["items"][0]["revision"] = "not-a-revision".into(),
            _ => metadata[KEY]["status"] = "promoted".into(),
        }
        repo.db
            .query("UPDATE $source SET metadata=$metadata")
            .bind(("source", source.id.clone()))
            .bind(("metadata", metadata))
            .await
            .unwrap()
            .check()
            .unwrap();
        let before = rows(&repo, "source").await;
        let now = repo
            .get_source(source.uri.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert!(!source_entity_enrichment_current(&now));
        assert!(repo
            .begin_source_entity_enrichment(plan.clone())
            .await
            .is_err());
        assert!(repo.promote_source_entity_enrichment(&plan).await.is_err());
        assert_eq!(rows(&repo, "source").await, before);
        assert!(rows(&repo, "entity").await.is_empty());
        assert!(rows(&repo, "mentions").await.is_empty());
    }
}
