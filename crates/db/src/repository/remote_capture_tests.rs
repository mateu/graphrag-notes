use super::*;
use crate::init_memory;
use graphrag_core::EntityType;
use surrealdb_types::ToSql;

fn input(instance: &str, request: &str) -> RemoteCaptureInput {
    RemoteCaptureInput {
        authenticated_instance_id: instance.into(),
        request_id: request.into(),
        payload_fingerprint: "a".repeat(64),
        content: "Atlas shared capture body".into(),
        title: Some("Atlas capture".into()),
        tags: vec!["atlas".into(), "shared".into()],
        source_provenance: serde_json::json!({
            "uri": "file:///client-only/atlas.md",
            "label": "Client label",
            "instance_id": "untrusted caller label",
        }),
    }
}

async fn table(repo: &Repository, name: &str) -> Vec<serde_json::Value> {
    repo.db
        .query(format!("SELECT * FROM {name} ORDER BY id"))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

async fn snapshot(repo: &Repository) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for name in [
        "source",
        "note",
        "entity",
        "mentions",
        "remote_capture_receipt",
    ] {
        result.insert(
            name.into(),
            serde_json::Value::Array(table(repo, name).await),
        );
    }
    result.into()
}

#[tokio::test]
async fn capture_commits_exact_vector_free_result_and_authenticated_provenance() {
    let repo = Repository::new(init_memory().await.unwrap());
    let request = input("openclaw-home", "capture-1");
    let result = repo
        .capture_remote_note(request.clone(), Vec::new(), Vec::new())
        .await
        .unwrap();
    assert!(!result.replayed);
    let saved = &result.result;
    assert_eq!(saved["content"], request.content);
    assert_eq!(saved["title"], "Atlas capture");
    assert_eq!(saved["tags"], serde_json::json!(["atlas", "shared"]));
    assert!(saved.get("embedding").is_none());
    assert_eq!(saved["provenance"]["instance_id"], "openclaw-home");
    assert_eq!(saved["provenance"]["source"], request.source_provenance);
    assert!(saved["provenance"]["source_uri"]
        .as_str()
        .unwrap()
        .starts_with("mcp://capture/"));
    assert_eq!(saved.as_object().unwrap().len(), 8);
    DateTime::parse_from_rfc3339(saved["created_at"].as_str().unwrap()).unwrap();
    DateTime::parse_from_rfc3339(saved["updated_at"].as_str().unwrap()).unwrap();
    let inspected = repo
        .inspect_record(saved["id"].as_str().unwrap(), 0)
        .await
        .unwrap();
    assert_eq!(inspected.revision, saved["revision"]);
    assert_eq!(
        serde_json::to_value(inspected.provenance).unwrap(),
        saved["provenance"]
    );
    assert_eq!(table(&repo, "note").await.len(), 1);
    assert_eq!(table(&repo, "source").await.len(), 1);
    assert_eq!(table(&repo, "remote_capture_receipt").await.len(), 1);
}

#[tokio::test]
async fn replay_keeps_original_snapshot_after_note_edit_and_deletion() {
    let repo = Repository::new(init_memory().await.unwrap());
    let request = input("hermes", "capture-2");
    let saved = repo
        .capture_remote_note(request.clone(), Vec::new(), Vec::new())
        .await
        .unwrap();
    let id = saved.result["id"].as_str().unwrap();
    let mut note = repo.get_note(id).await.unwrap().unwrap();
    note.content = "Edited after original response".into();
    repo.update_note(id, note).await.unwrap();
    let receipt = repo
        .find_remote_capture_receipt("hermes", "capture-2", &request.payload_fingerprint)
        .await
        .unwrap()
        .unwrap();
    assert!(receipt.replayed);
    assert_eq!(receipt.result, saved.result);
    repo.db
        .query("DELETE $id")
        .bind(("id", parse_record_id(id, Some("note")).unwrap()))
        .await
        .unwrap()
        .check()
        .unwrap();
    let repeated = repo
        .capture_remote_note(request, Vec::new(), Vec::new())
        .await
        .unwrap();
    assert!(repeated.replayed);
    assert_eq!(repeated.result, saved.result);
    assert!(table(&repo, "note").await.is_empty());
    assert_eq!(table(&repo, "source").await.len(), 1);
}

#[tokio::test]
async fn request_reuse_conflicts_on_fingerprint_and_on_different_payload() {
    let repo = Repository::new(init_memory().await.unwrap());
    let original = input("openclaw-home", "capture-3");
    repo.capture_remote_note(original.clone(), Vec::new(), Vec::new())
        .await
        .unwrap();
    let before = snapshot(&repo).await;
    assert!(matches!(
        repo.find_remote_capture_receipt("openclaw-home", "capture-3", &"b".repeat(64))
            .await,
        Err(DbError::RemoteRequestConflict { .. })
    ));
    for change in ["fingerprint", "body", "title", "tags", "provenance"] {
        let mut altered = original.clone();
        match change {
            "fingerprint" => altered.payload_fingerprint = "b".repeat(64),
            "body" => altered.content.push_str(" changed"),
            "title" => altered.title = None,
            "tags" => altered.tags.push("different".into()),
            _ => altered.source_provenance = serde_json::Value::Null,
        }
        assert!(matches!(
            repo.capture_remote_note(altered, Vec::new(), Vec::new())
                .await,
            Err(DbError::RemoteRequestConflict { .. })
        ));
    }
    assert_eq!(snapshot(&repo).await, before);
}

#[tokio::test]
async fn identical_request_ids_are_scoped_to_authenticated_instances() {
    let repo = Repository::new(init_memory().await.unwrap());
    let left = repo
        .capture_remote_note(
            input("openclaw-home", "same-request"),
            Vec::new(),
            Vec::new(),
        )
        .await
        .unwrap();
    let right = repo
        .capture_remote_note(input("hermes", "same-request"), Vec::new(), Vec::new())
        .await
        .unwrap();
    assert_ne!(left.result["id"], right.result["id"]);
    assert_eq!(table(&repo, "note").await.len(), 2);
    assert!(repo
        .find_remote_capture_receipt("another-instance", "same-request", &"a".repeat(64))
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn concurrent_repositories_on_one_connection_commit_only_one_capture() {
    let db = init_memory().await.unwrap();
    let left = Repository::new(db.clone());
    let right = Repository::new(db);
    let request = input("openclaw-home", "concurrent-request");
    let (a, b) = tokio::join!(
        left.capture_remote_note(request.clone(), Vec::new(), Vec::new()),
        right.capture_remote_note(request, Vec::new(), Vec::new())
    );
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(a.result, b.result);
    assert_ne!(a.replayed, b.replayed);
    assert_eq!(table(&left, "note").await.len(), 1);
    assert_eq!(table(&left, "source").await.len(), 1);
    assert_eq!(table(&left, "remote_capture_receipt").await.len(), 1);
}

#[tokio::test]
async fn source_note_entity_mention_and_receipt_failures_roll_back_every_write() {
    for failed_table in ["source", "note", "entity", "mentions", "receipt"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut old_entity = Entity::new("Existing entity", EntityType::Concept);
        old_entity.metadata = serde_json::json!({"aliases": ["original"]});
        let existing = repo.upsert_entity(old_entity).await.unwrap();
        let baseline = repo
            .capture_remote_note(
                input("hermes", "before-failure"),
                Vec::new(),
                vec![existing.clone()],
            )
            .await
            .unwrap();
        let before = snapshot(&repo).await;
        let constraint = match failed_table {
            "source" => "DEFINE FIELD OVERWRITE metadata ON source TYPE option<object> FLEXIBLE ASSERT $value.remote_capture.request_id != 'will-fail'".into(),
            "note" => "DEFINE FIELD OVERWRITE content ON note TYPE string ASSERT $value != 'rejected capture body'".into(),
            "entity" => "DEFINE FIELD OVERWRITE canonical_name ON entity TYPE string ASSERT $value != 'new orphan'".into(),
            "mentions" => format!("DEFINE FIELD OVERWRITE out ON mentions TYPE record<entity> ASSERT $value != {}", existing.id.as_ref().unwrap().to_sql()),
            _ => "DEFINE FIELD OVERWRITE result ON remote_capture_receipt TYPE object FLEXIBLE ASSERT false".into(),
        };
        repo.db.query(constraint).await.unwrap().check().unwrap();
        let mut request = input("openclaw-home", "will-fail");
        request.content = "rejected capture body".into();
        let mut replacement = existing;
        replacement.name = "CHANGED entity".into();
        replacement.metadata = serde_json::json!({"aliases": ["uncommitted"]});
        let result = repo
            .capture_remote_note(
                request,
                Vec::new(),
                vec![replacement, Entity::new("New orphan", EntityType::Concept)],
            )
            .await;
        assert!(
            result.is_err(),
            "expected {failed_table} constraint to reject capture"
        );
        assert_eq!(
            snapshot(&repo).await,
            before,
            "{failed_table} failure changed persisted data"
        );
        assert_eq!(
            repo.find_remote_capture_receipt("hermes", "before-failure", &"a".repeat(64))
                .await
                .unwrap()
                .unwrap()
                .result,
            baseline.result
        );
        assert!(repo
            .find_remote_capture_receipt("openclaw-home", "will-fail", &"a".repeat(64))
            .await
            .unwrap()
            .is_none());
    }
}

#[tokio::test]
async fn schema_normalization_cannot_commit_an_incorrect_snapshot_revision() {
    for normalization in [
        "DEFINE FIELD OVERWRITE content ON note TYPE string VALUE string::uppercase($value)",
        "DEFINE FIELD OVERWRITE uri ON source TYPE option<string> VALUE string::uppercase($value)",
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.db.query(normalization).await.unwrap().check().unwrap();
        assert!(repo
            .capture_remote_note(input("hermes", "normalize"), Vec::new(), Vec::new())
            .await
            .is_err());
        for name in ["source", "note", "remote_capture_receipt"] {
            assert!(table(&repo, name).await.is_empty());
        }
    }
}

#[tokio::test]
async fn portable_roundtrip_preserves_receipt_provenance_and_original_replay() {
    let repo = Repository::new(init_memory().await.unwrap());
    let request = input("openclaw-home", "portable-request");
    let original = repo
        .capture_remote_note(request.clone(), Vec::new(), Vec::new())
        .await
        .unwrap();
    let restored = Repository::new(init_memory().await.unwrap());
    assert!(PORTABLE_TABLES.contains(&"remote_capture_receipt"));
    for &name in PORTABLE_TABLES {
        for record in repo.portable_records_page(name, 0, 100).await.unwrap() {
            restored
                .restore_portable_record(name, record)
                .await
                .unwrap();
        }
    }
    let replay = restored
        .find_remote_capture_receipt(
            "openclaw-home",
            "portable-request",
            &request.payload_fingerprint,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.result, original.result);
    let replay = restored
        .capture_remote_note(request, Vec::new(), Vec::new())
        .await
        .unwrap();
    assert_eq!(replay.result, original.result);
    assert_eq!(table(&restored, "note").await.len(), 1);
    let inspected = restored
        .inspect_record(original.result["id"].as_str().unwrap(), 0)
        .await
        .unwrap();
    assert_eq!(inspected.revision, original.result["revision"]);
    assert_eq!(
        inspected.provenance.instance_id.as_deref(),
        Some("openclaw-home")
    );
}

#[tokio::test]
async fn invalid_identity_or_content_is_rejected_without_mutation() {
    let repo = Repository::new(init_memory().await.unwrap());
    for invalid in ["instance", "request", "fingerprint", "content", "embedding"] {
        let mut request = input("hermes", "invalid-request");
        let mut embedding = Vec::new();
        match invalid {
            "instance" => request.authenticated_instance_id = "user\nescape".into(),
            "request" => request.request_id.clear(),
            "fingerprint" => request.payload_fingerprint = "not-a-hash".into(),
            "content" => request.content = " \n ".into(),
            _ => embedding.push(f32::NAN),
        }
        assert!(matches!(
            repo.capture_remote_note(request, embedding, Vec::new())
                .await,
            Err(DbError::InvalidRemoteRequest(_))
        ));
    }
    for name in ["source", "note", "remote_capture_receipt"] {
        assert!(table(&repo, name).await.is_empty());
    }
}
