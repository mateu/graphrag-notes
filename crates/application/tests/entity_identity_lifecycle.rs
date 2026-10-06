//! Fictional regression coverage for final-note evidence identities.
use graphrag_agents::{
    DeterministicEmbedder, EntityExtraction, ExtractedEntity, FixtureEntityExtractor,
    LibrarianAgent, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, EntityType, Note, SourceType};
use graphrag_db::{init_memory, Repository};
use std::{collections::BTreeSet, sync::Arc};

fn extractor() -> Arc<FixtureEntityExtractor> {
    Arc::new(
        FixtureEntityExtractor::default().with_default(EntityExtraction {
            entities: vec![
                ExtractedEntity {
                    name: "Atlas".into(),
                    entity_type: Some("Project".into()),
                    aliases: vec!["Atlas program".into()],
                },
                ExtractedEntity {
                    name: "Rust".into(),
                    entity_type: Some("Technology".into()),
                    aliases: vec!["Rust language".into()],
                },
            ],
            relationships: vec![],
        }),
    )
}
fn librarian(repo: &Repository) -> LibrarianAgent {
    LibrarianAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default()),
        extractor(),
    )
}
async fn identities(repo: &Repository, id: &str) -> BTreeSet<String> {
    repo.get_entities_for_note(id)
        .await
        .unwrap()
        .into_iter()
        .map(|entity| record_id_to_string(entity.id.as_ref().unwrap()))
        .collect()
}

#[tokio::test]
async fn authenticated_capture_and_remote_content_edits_reuse_final_entity_identity() {
    let repo = Repository::new(init_memory().await.unwrap());
    let embedder = Arc::new(DeterministicEmbedder::default());
    let app = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        extractor(),
        LibrarianRuntimeConfig::default(),
    );
    let caller = || CallerIdentity {
        instance_id: "openclaw-fictional".into(),
    };
    let saved = app
        .capture_remote(
            caller(),
            RemoteCaptureRequest {
                request_id: "stable-fictional-capture".into(),
                content: "Atlas uses Rust".into(),
                title: Some("Atlas project".into()),
                tags: vec![],
                provenance: None,
            },
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    let expected =
        Repository::remote_capture_note_id("openclaw-fictional", "stable-fictional-capture")
            .unwrap();
    assert_eq!(saved.record.id, record_id_to_string(&expected));
    let before = identities(&repo, &saved.record.id).await;
    assert_eq!(before.len(), 2);
    librarian(&repo)
        .extract_entities_for_note_ids_result(std::slice::from_ref(&saved.record.id), true)
        .await
        .unwrap();
    assert_eq!(identities(&repo, &saved.record.id).await, before);
    for index in 0..3 {
        let snapshot = app
            .note_snapshot(RecordRef {
                id: saved.record.id.clone(),
                revision: None,
            })
            .await
            .unwrap();
        app.edit_remote(
            caller(),
            RemoteEditRequest {
                request_id: format!("fictional-edit-{index}"),
                id: snapshot.id,
                revision: snapshot.revision,
                patch: RemoteNotePatch {
                    content: Some(format!("Atlas uses Rust revision {index}")),
                    ..Default::default()
                },
            },
            ActionCancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(identities(&repo, &saved.record.id).await, before);
    }
    assert_eq!(
        repo.portable_records_page("entity", 0, 100)
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn local_capture_and_detached_provenance_keep_note_local_identities_on_reprocessing() {
    let repo = Repository::new(init_memory().await.unwrap());
    let agent = librarian(&repo);
    let local = agent
        .capture_manual_note("Atlas uses Rust locally".into(), None, vec![])
        .await
        .unwrap();
    let local_id = record_id_to_string(local.id.as_ref().unwrap());
    let local_entities = identities(&repo, &local_id).await;
    agent
        .extract_entities_for_note_ids_result(std::slice::from_ref(&local_id), true)
        .await
        .unwrap();
    assert_eq!(identities(&repo, &local_id).await, local_entities);

    let mut import = repo
        .begin_file_import(
            SourceType::Markdown,
            "Fictional source".into(),
            "fixture://atlas-source.md".into(),
            "Atlas uses Rust".into(),
            "sha256:fictional".into(),
            false,
        )
        .await
        .unwrap();
    let generated = repo
        .create_note(
            Note::new("Atlas uses Rust in an imported source")
                .with_source(import.source.id.as_ref().unwrap().clone())
                .with_source_generation(import.source.generation),
        )
        .await
        .unwrap();
    repo.complete_file_import(&mut import.source).await.unwrap();
    let generated_id = record_id_to_string(generated.id.as_ref().unwrap());
    agent
        .extract_entities_for_note_ids_result(std::slice::from_ref(&generated_id), true)
        .await
        .unwrap();
    let source_entities = identities(&repo, &generated_id).await;
    let source_technology = repo
        .get_entities_for_note(&generated_id)
        .await
        .unwrap()
        .into_iter()
        .find(|entity| entity.entity_type == EntityType::Technology)
        .unwrap();
    let detached = agent
        .detach_note_to_manual(
            &generated,
            "Atlas uses Rust in a manual copy".into(),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(detached.source_id, generated.source_id);
    assert!(detached.source_generation.is_none());
    let detached_id = record_id_to_string(detached.id.as_ref().unwrap());
    let detached_entities = identities(&repo, &detached_id).await;
    assert!(detached_entities.is_disjoint(&source_entities));
    agent
        .extract_entities_for_note_ids_result(std::slice::from_ref(&detached_id), true)
        .await
        .unwrap();
    assert_eq!(identities(&repo, &detached_id).await, detached_entities);
    agent
        .update_manual_note_content(
            &detached,
            "Atlas uses Rust with manual corrections".into(),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(identities(&repo, &detached_id).await, detached_entities);
    let retained = repo
        .get_entities_for_note(&generated_id)
        .await
        .unwrap()
        .into_iter()
        .find(|entity| entity.entity_type == EntityType::Technology)
        .unwrap();
    assert_eq!(retained.id, source_technology.id);
    assert_eq!(retained.metadata, source_technology.metadata);
}
