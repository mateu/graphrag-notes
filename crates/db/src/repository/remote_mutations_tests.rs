use super::*;
use crate::init_memory;
use graphrag_core::{EdgeType, EntityType};

fn input(request: &str, operation: &str) -> RemoteMutationInput {
    RemoteMutationInput {
        instance_id: "openclaw-a".into(),
        request_id: request.into(),
        operation: operation.into(),
        payload_fingerprint: "a".repeat(64),
        payload: serde_json::json!({"operation":operation,"request":request}),
        result: serde_json::json!({"operation":operation,"original":request}),
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
async fn edit(repo: &Repository, expected: Note, request: &str) -> Result<RemoteCaptureReceipt> {
    let guard = repo.mutation_guard().await;
    let mut replacement = expected.clone();
    replacement.content = "edited note".into();
    replacement.search_content = Some("edited note".into());
    repo.apply_remote_mutation(
        &guard,
        input(request, "edit"),
        RemoteMutationEffect::Edit {
            expected: Box::new(expected),
            replacement: Box::new(replacement),
            entities: None,
        },
    )
    .await
}

#[tokio::test]
async fn competing_full_snapshot_edits_have_one_effect_and_receipt() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo.create_note(Note::new("original")).await.unwrap();
    let (left, right) = tokio::join!(
        edit(&repo, note.clone(), "left"),
        edit(&repo, note, "right")
    );
    assert_eq!(
        usize::from(left.is_ok()) + usize::from(right.is_ok()),
        1,
        "left={left:?}; right={right:?}"
    );
    assert!(matches!(
        left.as_ref().err().or(right.as_ref().err()),
        Some(DbError::MutationRevisionConflict(_))
    ));
    assert_eq!(rows(&repo, "remote_mutation_receipt").await.len(), 1);
}

#[tokio::test]
async fn same_timestamp_raw_writer_and_new_chat_ownership_abort_without_receipt() {
    for change in [
        "content",
        "embedding",
        "tags",
        "source_generation",
        "message_link",
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        let original = repo.create_note(Note::new("original")).await.unwrap();
        let id = original.id.clone().unwrap();
        match change {
            "content" => {
                repo.db
                    .query("UPDATE $id SET content='concurrent'")
                    .bind(("id", id.clone()))
                    .bind(("vector", vec![1.0_f32; 1024]))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            "embedding" => {
                repo.db
                    .query("UPDATE $id SET embedding=$vector")
                    .bind(("id", id.clone()))
                    .bind(("vector", vec![1.0_f32; 1024]))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            "tags" => {
                repo.db
                    .query("UPDATE $id SET tags=['concurrent']")
                    .bind(("id", id.clone()))
                    .bind(("vector", vec![1.0_f32; 1024]))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            "source_generation" => {
                repo.db
                    .query("UPDATE $id SET source_generation=1")
                    .bind(("id", id.clone()))
                    .bind(("vector", vec![1.0_f32; 1024]))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            _ => {
                repo.db
                    .query("CREATE note_from_message SET in=$id,out=message:synthetic")
                    .bind(("id", id.clone()))
                    .bind(("vector", vec![1.0_f32; 1024]))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
        }
        assert!(
            matches!(
                edit(&repo, original, "blocked").await,
                Err(DbError::MutationRevisionConflict(_))
            ),
            "{change}"
        );
        assert!(rows(&repo, "remote_mutation_receipt").await.is_empty());
    }
}

#[tokio::test]
async fn receipt_schema_failure_rolls_back_note_entities_mentions_and_delete_cascade() {
    for operation in ["edit", "delete"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo.create_note(Note::new("original")).await.unwrap();
        let other = repo.create_note(Note::new("other")).await.unwrap();
        repo.create_edge(
            note.id.as_ref().unwrap(),
            other.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();
        let before = rows(&repo, "note").await;
        repo.db
            .query("DEFINE FIELD OVERWRITE result ON remote_mutation_receipt TYPE object ASSERT false;")
            .await
            .unwrap()
            .check()
            .unwrap();
        let guard = repo.mutation_guard().await;
        let effect = if operation == "edit" {
            let mut replacement = note.clone();
            replacement.content = "new content".into();
            RemoteMutationEffect::Edit {
                expected: Box::new(note),
                replacement: Box::new(replacement),
                entities: Some(vec![Entity::new("Uncommitted", EntityType::Concept)]),
            }
        } else {
            RemoteMutationEffect::Delete {
                expected: Box::new(note),
            }
        };
        assert!(repo
            .apply_remote_mutation(&guard, input("rollback", operation), effect)
            .await
            .is_err());
        assert_eq!(rows(&repo, "note").await, before);
        assert_eq!(rows(&repo, "supports").await.len(), 1);
        assert!(rows(&repo, "entity").await.is_empty());
        assert!(rows(&repo, "mentions").await.is_empty());
        assert!(rows(&repo, "remote_mutation_receipt").await.is_empty());
    }
}

#[tokio::test]
async fn delete_cascade_receipt_replays_after_removal_and_portable_restore() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo.create_note(Note::new("original")).await.unwrap();
    let other = repo.create_note(Note::new("other")).await.unwrap();
    repo.create_edge(
        note.id.as_ref().unwrap(),
        other.id.as_ref().unwrap(),
        EdgeType::Supports,
        None,
    )
    .await
    .unwrap();
    let proposal = repo
        .upsert_gardener_proposal(
            note.id.as_ref().unwrap(),
            other.id.as_ref().unwrap(),
            0.8,
            "proposal".into(),
            None,
            None,
        )
        .await
        .unwrap();
    let request = input("delete", "delete");
    let guard = repo.mutation_guard().await;
    let result = repo
        .apply_remote_mutation(
            &guard,
            request.clone(),
            RemoteMutationEffect::Delete {
                expected: Box::new(note),
            },
        )
        .await
        .unwrap();
    drop(guard);
    assert!(rows(&repo, "supports").await.is_empty());
    assert_eq!(
        repo.get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ProposedEdgeStatus::Superseded
    );
    let replay = repo
        .find_remote_mutation_receipt(&request)
        .await
        .unwrap()
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(result.result, replay.result);
    for change in ["operation", "payload", "fingerprint"] {
        let mut changed = request.clone();
        match change {
            "operation" => changed.operation = "edit".into(),
            "payload" => changed.payload = serde_json::json!({"different":true}),
            _ => changed.payload_fingerprint = "b".repeat(64),
        };
        assert!(matches!(
            repo.find_remote_mutation_receipt(&changed).await,
            Err(DbError::MutationRevisionConflict(_))
        ));
    }
    let restored = Repository::new(init_memory().await.unwrap());
    for record in repo
        .portable_records_page("remote_mutation_receipt", 0, 20)
        .await
        .unwrap()
    {
        restored
            .restore_portable_record("remote_mutation_receipt", record)
            .await
            .unwrap();
    }
    assert_eq!(
        restored
            .find_remote_mutation_receipt(&request)
            .await
            .unwrap()
            .unwrap()
            .result,
        result.result
    );
}

#[tokio::test]
async fn accept_reject_undo_use_trusted_audit_and_atomic_journals() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo.create_note(Note::new("left")).await.unwrap();
    let b = repo.create_note(Note::new("right")).await.unwrap();
    let proposal = repo
        .upsert_gardener_proposal(
            a.id.as_ref().unwrap(),
            b.id.as_ref().unwrap(),
            0.9,
            "related".into(),
            None,
            None,
        )
        .await
        .unwrap();
    let guard = repo.mutation_guard().await;
    repo.apply_remote_mutation(
        &guard,
        input("accept", "accept"),
        RemoteMutationEffect::Decision {
            expected: Box::new(proposal.clone()),
            endpoints: vec![a, b],
            action: "accept".into(),
            reason: Some("reviewed".into()),
        },
    )
    .await
    .unwrap();
    drop(guard);
    let accepted = repo
        .get_edge_proposal(proposal.id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    assert_eq!(accepted.reviewer.as_deref(), Some("mcp:openclaw-a"));
    assert_eq!(rows(&repo, "related_to").await.len(), 1);
    let guard = repo.mutation_guard().await;
    repo.apply_remote_mutation(
        &guard,
        input("undo", "undo"),
        RemoteMutationEffect::Decision {
            expected: Box::new(accepted),
            endpoints: vec![],
            action: "undo".into(),
            reason: None,
        },
    )
    .await
    .unwrap();
    drop(guard);
    let undone = repo
        .get_edge_proposal(proposal.id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(undone.status, ProposedEdgeStatus::Superseded);
    assert_eq!(undone.reviewer.as_deref(), Some("mcp:openclaw-a"));
    assert!(rows(&repo, "related_to").await.is_empty());
    assert_eq!(rows(&repo, "remote_mutation_receipt").await.len(), 2);
}

#[tokio::test]
async fn typed_note_keys_are_bound_exactly_and_ambiguous_public_ids_fail_closed() {
    let uuid = surrealdb_types::Uuid::new_v4();
    for id in [
        RecordId::new("note", "42"),
        RecordId::new("note", 42_i64),
        RecordId::new("note", uuid),
        RecordId::new("note", uuid.to_string()),
        RecordId::new("note", "literal` space:colon"),
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.db.query("CREATE $id SET content='typed manual',tags=['old'],created_at=time::now(),updated_at=time::now()").bind(("id",id.clone())).await.unwrap().check().unwrap();
        let guard = repo.mutation_guard().await;
        let snapshot = repo
            .mutation_note_snapshot(&guard, &record_id_to_string(&id))
            .await
            .unwrap();
        assert_eq!(snapshot.note.id.as_ref(), Some(&id));
        let mut replacement = snapshot.note.clone();
        replacement.tags = vec!["changed".into()];
        repo.apply_remote_mutation(
            &guard,
            input("typed-edit", "edit"),
            RemoteMutationEffect::Edit {
                expected: Box::new(snapshot.note),
                replacement: Box::new(replacement),
                entities: None,
            },
        )
        .await
        .unwrap();
        let saved = repo.db.select::<Option<Note>>(id).await.unwrap().unwrap();
        assert_eq!(saved.tags, ["changed"]);
        assert_eq!(rows(&repo, "note").await.len(), 1);
    }
    let repo = Repository::new(init_memory().await.unwrap());
    for id in [RecordId::new("note", "42"), RecordId::new("note", 42_i64)] {
        repo.db
            .query(
                "CREATE $id SET content='ambiguous',created_at=time::now(),updated_at=time::now()",
            )
            .bind(("id", id))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let guard = repo.mutation_guard().await;
    assert!(repo
        .mutation_note_snapshot(&guard, "note:42")
        .await
        .is_err());
    assert!(rows(&repo, "remote_mutation_receipt").await.is_empty());
}

#[tokio::test]
async fn stale_proposal_full_snapshot_and_independent_edges_abort_without_claim_or_receipt() {
    for change in ["proposal", "endpoint", "independent-edge"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let a = repo.create_note(Note::new("left")).await.unwrap();
        let b = repo.create_note(Note::new("right")).await.unwrap();
        let proposal = repo
            .upsert_gardener_proposal(
                a.id.as_ref().unwrap(),
                b.id.as_ref().unwrap(),
                0.8,
                "evidence".into(),
                None,
                None,
            )
            .await
            .unwrap();
        match change {
            "proposal" => {
                repo.db
                    .query("UPDATE $id SET reason='changed while preserving timestamp'")
                    .bind(("id", proposal.id.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            "endpoint" => {
                repo.db
                    .query("UPDATE $id SET tags=['changed while preserving timestamp']")
                    .bind(("id", a.id.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            _ => {
                repo.create_edge(
                    a.id.as_ref().unwrap(),
                    b.id.as_ref().unwrap(),
                    EdgeType::RelatedTo,
                    None,
                )
                .await
                .unwrap();
            }
        }
        let before = rows(&repo, "proposed_edge").await;
        let edges = rows(&repo, "related_to").await;
        let guard = repo.mutation_guard().await;
        assert!(
            matches!(
                repo.apply_remote_mutation(
                    &guard,
                    input("accept", "accept"),
                    RemoteMutationEffect::Decision {
                        expected: Box::new(proposal),
                        endpoints: vec![a, b],
                        action: "accept".into(),
                        reason: None
                    }
                )
                .await,
                Err(DbError::MutationRevisionConflict(_))
            ),
            "{change}"
        );
        assert_eq!(rows(&repo, "proposed_edge").await, before);
        assert_eq!(rows(&repo, "related_to").await, edges);
        assert!(rows(&repo, "remote_mutation_receipt").await.is_empty());
    }
}

#[tokio::test]
async fn decision_receipt_failure_rolls_back_accept_reject_and_undo_effects() {
    for action in ["accept", "reject", "undo"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let a = repo.create_note(Note::new("left")).await.unwrap();
        let b = repo.create_note(Note::new("right")).await.unwrap();
        let mut proposal = repo
            .upsert_gardener_proposal(
                a.id.as_ref().unwrap(),
                b.id.as_ref().unwrap(),
                0.8,
                "evidence".into(),
                None,
                None,
            )
            .await
            .unwrap();
        if action == "undo" {
            let guard = repo.mutation_guard().await;
            repo.apply_remote_mutation(
                &guard,
                input("accept", "accept"),
                RemoteMutationEffect::Decision {
                    expected: Box::new(proposal.clone()),
                    endpoints: vec![a.clone(), b.clone()],
                    action: "accept".into(),
                    reason: None,
                },
            )
            .await
            .unwrap();
            drop(guard);
            proposal = repo
                .get_edge_proposal(proposal.id.as_ref().unwrap())
                .await
                .unwrap()
                .unwrap();
        }
        repo.db.query("DEFINE FIELD OVERWRITE result ON remote_mutation_receipt TYPE object ASSERT false;").await.unwrap().check().unwrap();
        let before = rows(&repo, "proposed_edge").await;
        let edges = rows(&repo, "related_to").await;
        let receipts = rows(&repo, "remote_mutation_receipt").await;
        let guard = repo.mutation_guard().await;
        assert!(repo
            .apply_remote_mutation(
                &guard,
                input("rollback-decision", action),
                RemoteMutationEffect::Decision {
                    expected: Box::new(proposal),
                    endpoints: vec![a, b],
                    action: action.into(),
                    reason: Some("uncommitted audit".into())
                }
            )
            .await
            .is_err());
        assert_eq!(rows(&repo, "proposed_edge").await, before);
        assert_eq!(rows(&repo, "related_to").await, edges);
        assert_eq!(rows(&repo, "remote_mutation_receipt").await, receipts);
    }
}
