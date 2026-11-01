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

fn technology(alias: &str) -> EntityExtraction {
    EntityExtraction {
        entities: vec![ExtractedEntity {
            name: "Rust".into(),
            entity_type: Some("Technology".into()),
            aliases: vec![alias.into()],
        }],
        relationships: vec![],
    }
}
async fn alias_count(repo: &Repository, query: &str, note: &Note) -> usize {
    repo.find_graph_entities_for_search(
        query,
        8,
        std::slice::from_ref(note.id.as_ref().unwrap()),
        None,
        None,
    )
    .await
    .unwrap()
    .len()
}

#[tokio::test]
async fn remote_edits_replace_alias_evidence_without_accumulation_or_identity_growth() {
    let repo = Repository::new(init_memory().await.unwrap());
    let embedder = Arc::new(DeterministicEmbedder::default());
    let mut fixtures = FixtureEntityExtractor::default()
        .with_fixture("Rust initially", technology("obsolete compiler"));
    for index in 0..12 {
        fixtures = fixtures.with_fixture(
            format!("Rust revision {index}"),
            technology(&format!("current compiler {index}")),
        );
    }
    let extractor = Arc::new(fixtures);
    let app = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        extractor.clone(),
        LibrarianRuntimeConfig::default(),
    );
    let caller = || CallerIdentity {
        instance_id: "fictional-alias-owner".into(),
    };
    let saved = app
        .capture_remote(
            caller(),
            RemoteCaptureRequest {
                request_id: "alias-capture".into(),
                content: "Rust initially".into(),
                title: None,
                tags: vec![],
                provenance: None,
            },
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    let note = repo.get_note(&saved.record.id).await.unwrap().unwrap();
    let original = identities(&repo, &saved.record.id).await;
    assert_eq!(alias_count(&repo, "obsolete compiler", &note).await, 1);
    for index in 0..12 {
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
                request_id: format!("alias-edit-{index}"),
                id: snapshot.id,
                revision: snapshot.revision,
                patch: RemoteNotePatch {
                    content: Some(format!("Rust revision {index}")),
                    ..Default::default()
                },
            },
            ActionCancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(alias_count(&repo, "obsolete compiler", &note).await, 0);
        if index > 0 {
            assert_eq!(
                alias_count(&repo, &format!("current compiler {}", index - 1), &note).await,
                0
            );
        }
        assert_eq!(
            alias_count(&repo, &format!("current compiler {index}"), &note).await,
            1
        );
        assert_eq!(identities(&repo, &saved.record.id).await, original);
        let current = repo.get_entities_for_note(&saved.record.id).await.unwrap();
        assert_eq!(current[0].metadata["aliases"].as_array().unwrap().len(), 1);
    }
    let agent = LibrarianAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default()),
        extractor,
    );
    agent
        .extract_entities_for_note_ids_result(std::slice::from_ref(&saved.record.id), true)
        .await
        .unwrap();
    assert_eq!(alias_count(&repo, "current compiler 11", &note).await, 1);
    assert_eq!(
        repo.portable_records_page("entity", 0, 100)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn shared_source_aliases_are_owned_by_current_mentions_and_survive_exact_successors() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let mut import = repo
        .begin_file_import(
            SourceType::Markdown,
            "Fictional aliases".into(),
            "fixture://shared-aliases.md".into(),
            "Rust A\nRust B".into(),
            "sha256:alias-one".into(),
            false,
        )
        .await
        .unwrap();
    let mut notes = Vec::new();
    for text in ["Rust A", "Rust B"] {
        notes.push(
            repo.create_note(
                Note::new(text)
                    .with_source(import.source.id.as_ref().unwrap().clone())
                    .with_source_generation(import.source.generation),
            )
            .await
            .unwrap(),
        );
    }
    repo.complete_file_import(&mut import.source).await.unwrap();
    let initial = LibrarianAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default()),
        Arc::new(
            FixtureEntityExtractor::default()
                .with_fixture("Rust A", technology("amber compiler"))
                .with_fixture("Rust B", technology("violet compiler")),
        ),
    );
    let ids = notes
        .iter()
        .map(|note| record_id_to_string(note.id.as_ref().unwrap()))
        .collect::<Vec<_>>();
    initial
        .extract_entities_for_note_ids_result(&ids, true)
        .await
        .unwrap();
    assert_eq!(
        identities(&repo, &ids[0]).await,
        identities(&repo, &ids[1]).await
    );
    for alias in ["amber compiler", "violet compiler"] {
        assert_eq!(alias_count(&repo, alias, &notes[0]).await, 1);
    }
    let corrected = LibrarianAgent::new(
        repo.clone(),
        Arc::new(DeterministicEmbedder::default()),
        Arc::new(
            FixtureEntityExtractor::default().with_fixture("Rust A", technology("silver compiler")),
        ),
    );
    db.query("DEFINE FIELD OVERWRITE metadata ON mentions TYPE option<object> FLEXIBLE ASSERT $value = NONE OR NOT ($value.aliases CONTAINS 'silver compiler');")
        .await.unwrap().check().unwrap();
    assert!(corrected
        .extract_entities_for_note_ids_result(&ids[..1], true)
        .await
        .is_err());
    assert_eq!(alias_count(&repo, "amber compiler", &notes[0]).await, 1);
    assert_eq!(alias_count(&repo, "violet compiler", &notes[0]).await, 1);
    assert_eq!(alias_count(&repo, "silver compiler", &notes[0]).await, 0);
    db.query("DEFINE FIELD OVERWRITE metadata ON mentions TYPE option<object> FLEXIBLE;")
        .await
        .unwrap()
        .check()
        .unwrap();
    corrected
        .extract_entities_for_note_ids_result(&ids[..1], true)
        .await
        .unwrap();
    assert_eq!(alias_count(&repo, "amber compiler", &notes[0]).await, 0);
    assert_eq!(alias_count(&repo, "silver compiler", &notes[0]).await, 1);
    assert_eq!(alias_count(&repo, "violet compiler", &notes[0]).await, 1);
    assert_eq!(
        repo.get_entities_for_note(&ids[1]).await.unwrap()[0].metadata["aliases"],
        serde_json::json!(["violet compiler"])
    );
    assert!(repo
        .find_graph_entities_for_search(
            "violet compiler",
            8,
            &[],
            None,
            Some("fixture://unrelated.md".into())
        )
        .await
        .unwrap()
        .is_empty());
    assert!(repo
        .find_graph_entities_for_search(
            "violet compiler",
            8,
            &[],
            Some(chrono::Utc::now() + chrono::Duration::days(1)),
            None
        )
        .await
        .unwrap()
        .is_empty());
    let mut next = repo
        .begin_file_import(
            SourceType::Markdown,
            "Fictional aliases".into(),
            "fixture://shared-aliases.md".into(),
            "Rust A\nRust B unchanged".into(),
            "sha256:alias-two".into(),
            true,
        )
        .await
        .unwrap();
    let mut successors = Vec::new();
    for old in &notes {
        successors.push(
            repo.create_note(
                Note::new(&old.content)
                    .with_source(next.source.id.as_ref().unwrap().clone())
                    .with_source_generation(next.source.generation),
            )
            .await
            .unwrap(),
        );
    }
    repo.copy_note_dependents_to_successors(
        &notes
            .iter()
            .zip(&successors)
            .map(|(old, new)| (old.id.clone().unwrap(), new.id.clone().unwrap(), true))
            .collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    repo.complete_file_import(&mut next.source).await.unwrap();
    assert_eq!(
        alias_count(&repo, "silver compiler", &successors[0]).await,
        1
    );
    assert_eq!(
        alias_count(&repo, "violet compiler", &successors[1]).await,
        1
    );
    let restored = Repository::new(init_memory().await.unwrap());
    for table in ["source", "note", "entity", "mentions"] {
        for record in repo.portable_records_page(table, 0, 100).await.unwrap() {
            restored
                .restore_portable_record(table, record)
                .await
                .unwrap();
        }
    }
    assert_eq!(
        alias_count(&restored, "silver compiler", &successors[0]).await,
        1
    );
    assert_eq!(
        alias_count(&restored, "violet compiler", &successors[1]).await,
        1
    );
    assert_eq!(
        alias_count(&restored, "amber compiler", &successors[0]).await,
        0
    );
    repo.delete_note(&record_id_to_string(successors[1].id.as_ref().unwrap()))
        .await
        .unwrap();
    assert_eq!(
        alias_count(&repo, "violet compiler", &successors[0]).await,
        0
    );
    assert_eq!(
        alias_count(&repo, "silver compiler", &successors[0]).await,
        1
    );
    assert_eq!(
        repo.portable_records_page("entity", 0, 100)
            .await
            .unwrap()
            .len(),
        1
    );
}
