use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, FixtureEntityExtractor,
    InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use std::sync::Arc;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("unexpected embedding")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("unexpected embedding batch")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("unexpected provider identity")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("unexpected extraction")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected extractor health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("unexpected extractor identity")
    }
}
fn application(repo: &Repository, online: bool) -> EmbeddedApplication {
    let embedder: Arc<dyn Embedder> = if online {
        Arc::new(DeterministicEmbedder::default())
    } else {
        Arc::new(NoInference)
    };
    let extractor: Arc<dyn EntityExtractor> = if online {
        Arc::new(FixtureEntityExtractor::default())
    } else {
        Arc::new(NoInference)
    };
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        extractor,
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}
fn caller() -> CallerIdentity {
    CallerIdentity {
        instance_id: "hermes-home".into(),
    }
}
async fn snapshot(app: &EmbeddedApplication, id: &str) -> RemoteNoteSnapshot {
    app.note_snapshot(RecordRef {
        id: id.into(),
        revision: None,
    })
    .await
    .unwrap()
}
fn edit(s: &RemoteNoteSnapshot, request: &str, patch: RemoteNotePatch) -> RemoteEditRequest {
    RemoteEditRequest {
        request_id: request.into(),
        id: s.id.clone(),
        revision: s.revision.clone(),
        patch,
    }
}

#[tokio::test]
async fn metadata_edit_and_confirmed_delete_are_offline_and_replay_after_target_removal() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo
        .create_note(Note::new("Atlas manual content").with_tags(vec!["old".into()]))
        .await
        .unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let app = application(&repo, false);
    let before = snapshot(&app, &id).await;
    assert!(before.editable);
    let request = edit(
        &before,
        "edit-1",
        RemoteNotePatch {
            title: Some("Reviewed title".into()),
            tags: Some(vec!["new".into()]),
            ..Default::default()
        },
    );
    let edited = app
        .edit_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert!(!edited.replayed);
    assert_eq!(edited.outcome.actor, "mcp:hermes-home");
    let after = snapshot(&app, &id).await;
    assert_eq!(after.title.as_deref(), Some("Reviewed title"));
    assert_eq!(after.tags, ["new"]);
    assert_ne!(after.revision, before.revision);
    assert!(matches!(
        app.edit_remote(
            caller(),
            edit(
                &before,
                "stale",
                RemoteNotePatch {
                    clear_title: true,
                    ..Default::default()
                }
            ),
            ActionCancellation::new()
        )
        .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    let mut deletion = RemoteDeleteRequest {
        request_id: "delete-1".into(),
        id: id.clone(),
        revision: after.revision,
        confirmed: false,
    };
    assert!(matches!(
        app.delete_remote(caller(), deletion.clone()).await,
        Err(ApplicationError::Validation(_))
    ));
    deletion.confirmed = true;
    let removed = app.delete_remote(caller(), deletion.clone()).await.unwrap();
    assert_eq!(removed.outcome.cascade.as_ref().unwrap()["notes"], 1);
    assert!(app
        .note_snapshot(RecordRef { id, revision: None })
        .await
        .is_err());
    let replay = app
        .edit_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.outcome, edited.outcome);
    assert!(
        app.delete_remote(caller(), deletion)
            .await
            .unwrap()
            .replayed
    );
    let mut changed = request;
    changed.patch.title = Some("different".into());
    assert!(matches!(
        app.edit_remote(caller(), changed, ActionCancellation::new())
            .await,
        Err(ApplicationError::RevisionConflict(_))
    ));
}

#[tokio::test]
async fn changed_content_prepares_vectors_and_retry_bypasses_unavailable_providers() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo.create_note(Note::new("old body")).await.unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let healthy = application(&repo, true);
    let request = edit(
        &snapshot(&healthy, &id).await,
        "body-1",
        RemoteNotePatch {
            content: Some("Atlas replacement body".into()),
            ..Default::default()
        },
    );
    let result = healthy
        .edit_remote(caller(), request.clone(), ActionCancellation::new())
        .await
        .unwrap();
    assert_eq!(
        repo.get_note(&id).await.unwrap().unwrap().embedding.len(),
        1024
    );
    let replay = application(&repo, false)
        .edit_remote(caller(), request, ActionCancellation::new())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.outcome, result.outcome);
}

#[tokio::test]
async fn imported_or_chat_linked_notes_remain_readable_but_refuse_remote_mutations() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let note = repo.create_note(Note::new("chat-owned")).await.unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    db.query("CREATE note_from_message SET in=$id,out=message:synthetic")
        .bind(("id", note.id))
        .await
        .unwrap()
        .check()
        .unwrap();
    let app = application(&repo, false);
    let view = snapshot(&app, &id).await;
    assert!(!view.editable);
    assert!(view.blocked_reason.is_some());
    assert!(matches!(
        app.edit_remote(
            caller(),
            edit(
                &view,
                "blocked",
                RemoteNotePatch {
                    clear_title: true,
                    ..Default::default()
                }
            ),
            ActionCancellation::new()
        )
        .await,
        Err(ApplicationError::Validation(_))
    ));
    assert!(matches!(
        app.delete_remote(
            caller(),
            RemoteDeleteRequest {
                request_id: "delete".into(),
                id,
                revision: view.revision,
                confirmed: true
            }
        )
        .await,
        Err(ApplicationError::Validation(_))
    ));
}

#[tokio::test]
async fn proposal_reviews_guard_metadata_and_endpoints_and_derive_reviewer_from_caller() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo.create_note(Note::new("left")).await.unwrap();
    let b = repo.create_note(Note::new("right")).await.unwrap();
    let proposal = repo
        .upsert_gardener_proposal(
            a.id.as_ref().unwrap(),
            b.id.as_ref().unwrap(),
            0.8,
            "original evidence".into(),
            None,
            None,
        )
        .await
        .unwrap();
    let id = record_id_to_string(proposal.id.as_ref().unwrap());
    let app = application(&repo, false);
    let card = app.proposal(&id).await.unwrap();
    let stale = RemoteDecisionRequest {
        request_id: "decision-1".into(),
        id: id.clone(),
        revision: card.revision,
        action: ProposalAction::Accept,
        reason: Some("reviewed evidence".into()),
        confirmed: true,
    };
    let mut changed = a.clone();
    changed.tags.push("new".into());
    repo.update_note(&record_id_to_string(a.id.as_ref().unwrap()), changed)
        .await
        .unwrap();
    assert!(matches!(
        app.decide_remote(caller(), stale.clone()).await,
        Err(ApplicationError::RevisionConflict(_))
    ));
    let mut fresh = stale;
    fresh.revision = app.proposal(&id).await.unwrap().revision;
    let accepted = app.decide_remote(caller(), fresh.clone()).await.unwrap();
    assert_eq!(accepted.outcome.status, "accepted");
    assert!(accepted.outcome.resulting_edge_id.is_some());
    let card = app.proposal(&id).await.unwrap();
    assert_eq!(card.reviewer.as_deref(), Some("mcp:hermes-home"));
    let undo = RemoteDecisionRequest {
        request_id: "undo-1".into(),
        id: id.clone(),
        revision: card.revision,
        action: ProposalAction::Undo,
        reason: None,
        confirmed: true,
    };
    app.decide_remote(caller(), undo.clone()).await.unwrap();
    assert!(app.decide_remote(caller(), fresh).await.unwrap().replayed);
    assert!(app.decide_remote(caller(), undo).await.unwrap().replayed);
    assert_eq!(
        app.proposal(&id).await.unwrap().status,
        graphrag_core::ProposedEdgeStatus::Superseded
    );
}

#[test]
fn mutation_payload_v1_fingerprint_is_a_durable_compatibility_contract() {
    let request = RemoteEditRequest {
        request_id: "stable".into(),
        id: "note:manual".into(),
        revision: "a".repeat(64),
        patch: RemoteNotePatch {
            title: Some("Atlas".into()),
            tags: Some(vec!["daily".into()]),
            ..Default::default()
        },
    };
    let payload = serde_json::to_value(request).unwrap();
    assert_eq!(REMOTE_MUTATION_PAYLOAD_VERSION, 1);
    assert_eq!(
        remote_mutation_fingerprint("edit", &payload).unwrap(),
        "246467f82f9ff501edc2f536920e3a8e00f84fda102c84bb7c1358ece02dc44c"
    );
    assert_ne!(
        remote_mutation_fingerprint("delete", &payload).unwrap(),
        remote_mutation_fingerprint("edit", &payload).unwrap()
    );
}
