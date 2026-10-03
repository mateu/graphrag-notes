use super::*;
use crate::{init_memory, DbConnection};
use graphrag_core::{EntityType, NoteType};
use surrealdb_types::ToSql;

fn entity(name: &str) -> Entity {
    let mut entity = Entity::new(name, EntityType::Concept);
    entity.metadata = serde_json::json!({});
    entity
}

fn id(note: &Note) -> String {
    record_id_to_string(note.id.as_ref().unwrap())
}

async fn initial(repo: &Repository) -> Note {
    let mut original_entity = entity("Original mention").with_embedding(vec![0.25; 1024]);
    original_entity.metadata = serde_json::json!({"aliases": ["original alias"]});
    repo.create_note_and_replace_entities(
        Note::new("original editor body")
            .with_title("Original title")
            .with_tags(vec!["original".into()]),
        vec![original_entity],
    )
    .await
    .unwrap()
}

async fn all_notes(repo: &Repository) -> Vec<Note> {
    repo.db.select("note").await.unwrap()
}

async fn conversation_edge_ids(repo: &Repository) -> Vec<String> {
    let edges: Vec<RecordId> = repo
        .db
        .query("SELECT VALUE id FROM note_from_conversation ORDER BY id")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    edges.iter().map(record_id_to_string).collect()
}

async fn entity_snapshot(repo: &Repository) -> serde_json::Value {
    let mut entities: Vec<Entity> = repo.db.select("entity").await.unwrap();
    entities.sort_by(|left, right| left.canonical_name.cmp(&right.canonical_name));
    serde_json::Value::Array(
        entities
            .into_iter()
            .map(|entity| {
                let mut value = serde_json::to_value(&entity).unwrap();
                // Entity serialization intentionally omits creation time;
                // failed writes must preserve that persisted field too.
                value["created_at"] = serde_json::json!(entity.created_at);
                value
            })
            .collect(),
    )
}

fn changed_original_entity() -> Entity {
    changed_entity("Original mention")
}

fn changed_entity(name: &str) -> Entity {
    let mut incoming = Entity::new(format!(" {} ", name.to_uppercase()), EntityType::Person)
        .with_embedding(vec![0.5; 1024]);
    incoming.metadata = serde_json::json!({
        "aliases": ["new uncommitted alias"],
    });
    incoming
}

async fn mentions(repo: &Repository, note: &Note) -> Vec<String> {
    let mut names = repo
        .get_entities_for_note(&id(note))
        .await
        .unwrap()
        .into_iter()
        .map(|entity| entity.name)
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn assert_snapshot(actual: &Note, expected: &Note) {
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    assert_eq!(actual.created_at, expected.created_at);
    assert_eq!(actual.updated_at, expected.updated_at);
}

fn assert_conflict(result: Result<Note>) {
    assert!(
        matches!(result, Err(DbError::NoteRevisionConflict(_))),
        "expected a typed editor snapshot conflict, got {result:?}"
    );
}

#[tokio::test]
async fn chat_creation_commits_ownership_and_preserves_manual_copy_semantics() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = repo
        .create_source(Source::chat_export("Owned chat", None))
        .await
        .unwrap();
    let chat = repo
        .create_chat_note(
            Note::new("Imported chat body")
                .with_type(NoteType::Synthesis)
                .with_source(source.id.clone().unwrap())
                .with_title("Imported title")
                .with_tags(vec!["chat-export".into(), "summary".into()]),
            &RecordId::new("conversation", "missing-chat-target"),
        )
        .await
        .unwrap();
    assert_eq!(chat.source_generation, None);
    assert!(repo.get_visible_note(&id(&chat)).await.unwrap().is_some());
    assert!(repo
        .note_has_chat_provenance(chat.id.as_ref().unwrap())
        .await
        .unwrap());
    assert!(repo.note_requires_detach(&chat).await.unwrap());
    assert_eq!(conversation_edge_ids(&repo).await.len(), 1);
    assert_eq!(repo.preview_source_delete(&source).await.unwrap().notes, 0);
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&chat),
            chat.clone(),
            Vec::new(),
            &chat,
        )
        .await,
    );

    let copy = Note::new("Manual chat copy")
        .with_source(chat.source_id.clone().unwrap())
        .with_tags(chat.tags.clone());
    let detached = repo
        .create_note_and_replace_entities_if_unchanged(copy, Vec::new(), &chat)
        .await
        .unwrap();
    assert!(!repo.note_requires_detach(&detached).await.unwrap());
    let mut edited = detached.clone();
    edited.content = "Edited manual chat copy".into();
    repo.update_note_and_replace_entities_if_unchanged(
        &id(&detached),
        edited,
        Vec::new(),
        &detached,
    )
    .await
    .unwrap();
    let capture = repo
        .create_note_and_replace_entities(Note::new("Captured manual note"), Vec::new())
        .await
        .unwrap();
    assert!(!repo.note_requires_detach(&capture).await.unwrap());
    assert_eq!(conversation_edge_ids(&repo).await.len(), 1);
    assert_snapshot(&repo.get_note(&id(&chat)).await.unwrap().unwrap(), &chat);
}

#[tokio::test]
async fn chat_creation_rolls_back_ownership_notes_and_entities_on_storage_failures() {
    for failure in ["ownership", "note", "mention"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let original = initial(&repo).await;
        let conversation = RecordId::new("conversation", "new-chat-target");
        repo.link_note_to_conversation(
            original.id.as_ref().unwrap(),
            &RecordId::new("conversation", "existing-chat-target"),
        )
        .await
        .unwrap();
        let blocked = repo
            .upsert_entity(entity("Blocked chat mention"))
            .await
            .unwrap();
        let entities_before = entity_snapshot(&repo).await;
        let edges_before = conversation_edge_ids(&repo).await;
        let constraint = match failure {
            "ownership" => format!(
                "DEFINE FIELD OVERWRITE out ON note_from_conversation TYPE record<conversation> ASSERT $value != {}",
                conversation.to_sql(),
            ),
            "note" => "DEFINE FIELD OVERWRITE content ON note TYPE string ASSERT $value != 'rejected chat body'".into(),
            "mention" => format!(
                "DEFINE FIELD OVERWRITE out ON mentions TYPE record<entity> ASSERT $value != {}",
                blocked.id.as_ref().unwrap().to_sql(),
            ),
            _ => unreachable!(),
        };
        repo.db.query(constraint).await.unwrap().check().unwrap();
        // The shared helper supplies entities here so failures on either side
        // of the ownership/note writes also exercise full entity rollback.
        let result = repo
            .create_note_and_replace_entities_guarded(
                Note::new("rejected chat body"),
                vec![
                    entity("New chat orphan"),
                    changed_original_entity(),
                    changed_entity("Blocked chat mention"),
                ],
                None,
                Some(&conversation),
            )
            .await;
        assert!(
            matches!(result, Err(DbError::QueryFailed(_))),
            "{failure}: {result:?}"
        );
        assert_eq!(entity_snapshot(&repo).await, entities_before, "{failure}");
        assert_eq!(
            conversation_edge_ids(&repo).await,
            edges_before,
            "{failure}"
        );
        assert_eq!(all_notes(&repo).await.len(), 1, "{failure}");
        assert_snapshot(
            &repo.get_note(&id(&original)).await.unwrap().unwrap(),
            &original,
        );
        assert_eq!(mentions(&repo, &original).await, ["Original mention"]);
    }
}

#[tokio::test]
async fn atomic_create_returns_committed_note_when_a_separate_read_would_fail() {
    for guarded in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let opening = initial(&repo).await;
        // CREATE returns schema-normalized values, not merely the input Note.
        // The permission predicate permits the atomic batch (which binds its
        // extracted entities) but deliberately fails any separate note read.
        // This exposes a post-commit SELECT without a repository test hook.
        repo.db
            .query(
                "DEFINE TABLE OVERWRITE note SCHEMAFULL \
                 PERMISSIONS FOR create, update FULL FOR select WHERE \
                    IF $replacement_entities = NONE { THROW 'separate note read unavailable'; } ELSE { true }; \
                 DEFINE FIELD OVERWRITE title ON note TYPE option<string> \
                    VALUE IF $value = NONE THEN NONE ELSE string::uppercase($value) END; \
                 DEFINE TABLE OVERWRITE entity SCHEMAFULL PERMISSIONS FULL; \
                 DEFINE TABLE OVERWRITE mentions SCHEMAFULL PERMISSIONS FULL; \
                 CREATE user:atomic_writer; \
                 DEFINE ACCESS atomic_writer ON DB TYPE RECORD \
                    SIGNIN (SELECT * FROM user:atomic_writer) DURATION FOR SESSION 1h;",
            )
            .await
            .unwrap()
            .check()
            .unwrap();
        let namespace: Option<String> = repo
            .db
            .query("RETURN $session.ns")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        repo.db
            .signin(surrealdb::opt::auth::Record {
                namespace: namespace.unwrap(),
                database: "notes".into(),
                access: "atomic_writer".into(),
                params: (),
            })
            .await
            .unwrap();

        let input = Note::new("Committed manual body")
            .with_title("Mixed case title")
            .with_embedding(vec![0.5; 1024])
            .with_tags(vec!["manual".into()]);
        let created = if guarded {
            repo.create_note_and_replace_entities_if_unchanged(
                input,
                vec![entity("Returned mention")],
                &opening,
            )
            .await
        } else {
            repo.create_note_and_replace_entities(input, vec![entity("Returned mention")])
                .await
        }
        .expect("the committed create must not depend on a separate note read");
        assert_eq!(created.content, "Committed manual body");
        assert_eq!(created.title.as_deref(), Some("MIXED CASE TITLE"));
        assert_eq!(
            created.search_content.as_deref(),
            Some("Committed manual body")
        );
        assert_eq!(mentions(&repo, &created).await, ["Returned mention"]);
        let read_error = repo.get_note(&id(&created)).await.unwrap_err();
        assert!(
            read_error
                .to_string()
                .contains("separate note read unavailable"),
            "{read_error}"
        );

        // Restore the anonymous embedded-owner session and verify the returned
        // row was committed exactly once, including defaults and timestamps.
        repo.db.invalidate().await.unwrap();
        let persisted = repo.get_note(&id(&created)).await.unwrap().unwrap();
        assert_snapshot(&persisted, &created);
        assert_eq!(all_notes(&repo).await.len(), 2);
    }
}

#[tokio::test]
async fn atomic_update_returns_committed_note_when_a_separate_read_would_fail() {
    for guarded in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let opening = initial(&repo).await;
        // Legacy updates legitimately read the old note before their write.
        // Permit that and the bound atomic batch, but deliberately fail reads
        // of the new body outside that batch. This reproduces a read failure
        // only after the update has committed, without a test-only hook.
        repo.db
            .query(
                "DEFINE TABLE OVERWRITE note SCHEMAFULL \
                 PERMISSIONS FOR create, update FULL FOR select WHERE \
                    IF $replacement_entities != NONE OR content = 'original editor body' { true } \
                    ELSE { THROW 'separate updated note read unavailable'; }; \
                 DEFINE FIELD OVERWRITE title ON note TYPE option<string> \
                    VALUE IF $value = NONE THEN NONE ELSE string::uppercase($value) END; \
                 DEFINE TABLE OVERWRITE entity SCHEMAFULL PERMISSIONS FULL; \
                 DEFINE TABLE OVERWRITE mentions SCHEMAFULL PERMISSIONS FULL; \
                 CREATE user:atomic_writer; \
                 DEFINE ACCESS atomic_writer ON DB TYPE RECORD \
                    SIGNIN (SELECT * FROM user:atomic_writer) DURATION FOR SESSION 1h;",
            )
            .await
            .unwrap()
            .check()
            .unwrap();
        let namespace: Option<String> = repo
            .db
            .query("RETURN $session.ns")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        repo.db
            .signin(surrealdb::opt::auth::Record {
                namespace: namespace.unwrap(),
                database: "notes".into(),
                access: "atomic_writer".into(),
                params: (),
            })
            .await
            .unwrap();

        let mut input = opening.clone();
        input.content = "Committed updated body".into();
        input.title = Some("Mixed case update".into());
        input.embedding = vec![0.5; 1024];
        input.tags = vec!["updated".into()];
        input.updated_at = chrono::Utc::now();
        let updated = if guarded {
            repo.update_note_and_replace_entities_if_unchanged(
                &id(&opening),
                input,
                vec![entity("Updated mention")],
                &opening,
            )
            .await
        } else {
            repo.update_note_and_replace_entities(
                &id(&opening),
                input,
                vec![entity("Updated mention")],
            )
            .await
        }
        .expect("the committed update must not depend on a separate note read");
        assert_eq!(updated.id, opening.id);
        assert_eq!(updated.content, "Committed updated body");
        assert_eq!(updated.title.as_deref(), Some("MIXED CASE UPDATE"));
        assert_eq!(
            updated.search_content.as_deref(),
            Some("Committed updated body")
        );
        assert_eq!(updated.tags, ["updated"]);
        assert_eq!(updated.created_at, opening.created_at);
        assert_eq!(mentions(&repo, &updated).await, ["Updated mention"]);
        let read_error = repo.get_note(&id(&updated)).await.unwrap_err();
        assert!(
            read_error
                .to_string()
                .contains("separate updated note read unavailable"),
            "{read_error}"
        );

        repo.db.invalidate().await.unwrap();
        let persisted = repo.get_note(&id(&updated)).await.unwrap().unwrap();
        assert_snapshot(&persisted, &updated);
        assert_eq!(all_notes(&repo).await.len(), 1);
    }
}

#[tokio::test]
async fn chat_provenance_blocks_guarded_updates_but_allows_detached_manual_copies() {
    for message_link in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let source = repo
            .create_source(Source::chat_export(
                "Legacy chat",
                Some("legacy-chat.json".into()),
            ))
            .await
            .unwrap();
        let initial = initial(&repo).await;
        let mut opening = initial.clone();
        opening.source_id = source.id;
        opening.tags.push("chat-export".into());
        let opening = repo.update_note(&id(&opening), opening).await.unwrap();
        assert!(!repo.note_requires_detach(&opening).await.unwrap());
        // Add ownership after the editor snapshot without changing any Note
        // fields. Missing chat targets still identify the imported owner.
        if message_link {
            repo.link_note_to_message(
                opening.id.as_ref().unwrap(),
                &RecordId::new("message", "missing"),
            )
            .await
            .unwrap();
        } else {
            repo.link_note_to_conversation(
                opening.id.as_ref().unwrap(),
                &RecordId::new("conversation", "missing"),
            )
            .await
            .unwrap();
        }
        assert!(repo.note_requires_detach(&opening).await.unwrap());
        let before_entities = entity_snapshot(&repo).await;
        let mut replacement = opening.clone();
        replacement.content = "Must not replace imported chat".into();
        assert_conflict(
            repo.update_note_and_replace_entities_if_unchanged(
                &id(&opening),
                replacement,
                vec![changed_original_entity(), entity("Orphan")],
                &opening,
            )
            .await,
        );
        assert_snapshot(
            &repo.get_note(&id(&opening)).await.unwrap().unwrap(),
            &opening,
        );
        assert_eq!(entity_snapshot(&repo).await, before_entities);
        let mut manual = Note::new("Detached chat").with_tags(opening.tags.clone());
        manual.source_id = opening.source_id.clone();
        let manual = repo
            .create_note_and_replace_entities_if_unchanged(manual, Vec::new(), &opening)
            .await
            .unwrap();
        assert!(!repo.note_requires_detach(&manual).await.unwrap());
        let mut revised = manual.clone();
        revised.content = "Revised detached chat".into();
        let revised = repo
            .update_note_and_replace_entities_if_unchanged(
                &id(&manual),
                revised,
                Vec::new(),
                &manual,
            )
            .await
            .unwrap();
        assert_eq!(revised.source_id, opening.source_id);
        assert_eq!(revised.tags, opening.tags);
        assert_eq!(revised.content, "Revised detached chat");
        assert_snapshot(
            &repo.get_note(&id(&opening)).await.unwrap().unwrap(),
            &opening,
        );
    }
}

#[tokio::test]
async fn current_editor_snapshot_updates_content_and_mentions_together() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let mut replacement = expected.clone();
    replacement.content = "replacement editor body".into();
    replacement.title = Some("Replacement title".into());
    replacement.tags = vec!["replacement".into()];
    replacement.updated_at = Utc::now();
    let updated = repo
        .update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            replacement,
            vec![entity("Replacement mention")],
            &expected,
        )
        .await
        .unwrap();
    assert_eq!(updated.id, expected.id);
    assert_eq!(updated.content, "replacement editor body");
    assert_eq!(updated.title.as_deref(), Some("Replacement title"));
    assert_eq!(updated.tags, vec!["replacement"]);
    assert_eq!(
        updated.search_content.as_deref(),
        Some("replacement editor body")
    );
    assert_eq!(mentions(&repo, &updated).await, vec!["Replacement mention"]);
}

#[tokio::test]
async fn same_timestamp_content_metadata_and_ownership_changes_refuse_editor_writes() {
    for field in [
        "content",
        "title",
        "tags",
        "type",
        "source",
        "search",
        "embedding",
        "lines",
    ] {
        let repo = Repository::new(init_memory().await.unwrap());
        let expected = initial(&repo).await;
        let mut changed = expected.clone();
        match field {
            "content" => changed.content = "a different writer's current body".into(),
            "title" => changed.title = Some("A different title".into()),
            "tags" => changed.tags = vec!["different".into()],
            "type" => changed.note_type = NoteType::Claim,
            "source" => {
                changed.source_id = repo.create_source(Source::manual()).await.unwrap().id;
            }
            "search" => {
                changed.search_content = Some("independently revised search payload".into())
            }
            "embedding" => changed.embedding = vec![1.0; 1024],
            "lines" => changed.source_start_line = Some(17),
            _ => unreachable!(),
        }
        // Deliberately keep both timestamps. Revision guards must not assume
        // that every writer advances a clock field.
        let current = repo.update_note(&id(&expected), changed).await.unwrap();
        assert_eq!(current.updated_at, expected.updated_at);
        let entities_before = entity_snapshot(&repo).await;
        let mut draft = expected.clone();
        draft.content = "stale draft must never overwrite current state".into();
        assert_conflict(
            repo.update_note_and_replace_entities_if_unchanged(
                &id(&expected),
                draft,
                vec![entity("Stale mention"), changed_original_entity()],
                &expected,
            )
            .await,
        );
        let actual = repo.get_note(&id(&expected)).await.unwrap().unwrap();
        assert_snapshot(&actual, &current);
        assert_eq!(mentions(&repo, &actual).await, vec!["Original mention"]);
        assert_eq!(all_notes(&repo).await.len(), 1);
        assert_eq!(entity_snapshot(&repo).await, entities_before);
    }
}

#[tokio::test]
async fn detached_creation_checks_source_snapshot_and_preserves_source_mentions() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let created = repo
        .create_note_and_replace_entities_if_unchanged(
            Note::new("current detached copy"),
            vec![entity("Detached mention")],
            &expected,
        )
        .await
        .unwrap();
    assert_ne!(created.id, expected.id);
    assert_eq!(mentions(&repo, &created).await, vec!["Detached mention"]);
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);

    let mut replacement = expected.clone();
    replacement.content = "source changed with its timestamp preserved".into();
    let current = repo.update_note(&id(&expected), replacement).await.unwrap();
    let entities_before = entity_snapshot(&repo).await;
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("stale detached copy"),
            vec![entity("Stale detached mention"), changed_original_entity()],
            &expected,
        )
        .await,
    );
    assert_eq!(all_notes(&repo).await.len(), 2);
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &current,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
    assert_eq!(entity_snapshot(&repo).await, entities_before);
}

#[tokio::test]
async fn deleted_editor_source_is_never_recreated_or_detached() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    repo.delete_note(&id(&expected)).await.unwrap();
    let entities_before = entity_snapshot(&repo).await;
    let mut replacement = expected.clone();
    replacement.content = "must not recreate deleted note".into();
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            replacement,
            vec![entity("Recreated mention"), changed_original_entity()],
            &expected,
        )
        .await,
    );
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("must not detach deleted note"),
            vec![
                entity("Deleted detached mention"),
                changed_original_entity(),
            ],
            &expected,
        )
        .await,
    );
    assert!(repo.get_note(&id(&expected)).await.unwrap().is_none());
    assert!(all_notes(&repo).await.is_empty());
    assert_eq!(entity_snapshot(&repo).await, entities_before);
}

#[tokio::test]
async fn hidden_or_staged_generation_refuses_editor_writes_and_detachment() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut plan = repo
        .begin_file_import(
            SourceType::Markdown,
            "guard.md".into(),
            "file:///guard.md".into(),
            "original source".into(),
            "guard-hash".into(),
            false,
        )
        .await
        .unwrap();
    let staged = repo
        .create_note_and_replace_entities(
            Note::new("generated editor source")
                .with_source(plan.source.id.as_ref().unwrap().clone())
                .with_source_generation(plan.source.generation),
            vec![entity("Generated mention")],
        )
        .await
        .unwrap();
    let entities_before = entity_snapshot(&repo).await;
    assert!(repo.get_visible_note(&id(&staged)).await.unwrap().is_none());
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("staged draft"),
            vec![
                entity("Staged detached orphan"),
                changed_entity("Generated mention"),
            ],
            &staged,
        )
        .await,
    );
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&staged),
            staged.clone(),
            vec![
                entity("Staged update orphan"),
                changed_entity("Generated mention"),
            ],
            &staged,
        )
        .await,
    );
    assert_eq!(entity_snapshot(&repo).await, entities_before);

    repo.complete_file_import(&mut plan.source).await.unwrap();
    let expected = repo.get_visible_note(&id(&staged)).await.unwrap().unwrap();
    // The note itself is byte-for-byte unchanged, but promotion of another
    // source generation makes it unavailable for a user-facing edit/detach.
    repo.db
        .query("UPDATE $source SET successful_generation = $generation")
        .bind(("source", plan.source.id.clone()))
        .bind(("generation", plan.source.generation as i64 + 1))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("hidden draft"),
            vec![
                entity("Hidden detached orphan"),
                changed_entity("Generated mention"),
            ],
            &expected,
        )
        .await,
    );
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            expected.clone(),
            vec![
                entity("Hidden update orphan"),
                changed_entity("Generated mention"),
            ],
            &expected,
        )
        .await,
    );
    assert_eq!(all_notes(&repo).await.len(), 1);
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Generated mention"]);
    assert_eq!(entity_snapshot(&repo).await, entities_before);
}

#[tokio::test]
async fn guarded_note_and_detached_transactions_roll_back_on_mention_failure() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let blocked = repo.upsert_entity(entity("Blocked mention")).await.unwrap();
    let entities_before = entity_snapshot(&repo).await;
    repo.db
        .query(format!(
            "DEFINE FIELD OVERWRITE out ON mentions TYPE record<entity> ASSERT $value != {}",
            blocked.id.as_ref().unwrap().to_sql(),
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    let mut replacement = expected.clone();
    replacement.content = "must roll back with rejected mention".into();
    let failed_update = repo
        .update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            replacement,
            vec![entity("New rejected mention"), changed_original_entity(), {
                let mut incoming = entity("Blocked mention");
                incoming.metadata = serde_json::json!({"aliases": ["uncommitted blocked alias"]});
                incoming
            }],
            &expected,
        )
        .await;
    assert!(
        matches!(failed_update, Err(DbError::QueryFailed(_))),
        "{failed_update:?}"
    );
    assert!(failed_update.unwrap_err().to_string().contains("mentions"));
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
    assert_eq!(entity_snapshot(&repo).await, entities_before);
    let failed_detach = repo
        .create_note_and_replace_entities_if_unchanged(
            Note::new("must roll back detached note"),
            vec![
                entity("New rejected detached mention"),
                changed_original_entity(),
                {
                    let mut incoming = entity("Blocked mention");
                    incoming.metadata =
                        serde_json::json!({"aliases": ["uncommitted blocked alias"]});
                    incoming
                },
            ],
            &expected,
        )
        .await;
    assert!(
        matches!(failed_detach, Err(DbError::QueryFailed(_))),
        "{failed_detach:?}"
    );
    assert!(failed_detach.unwrap_err().to_string().contains("mentions"));
    assert_eq!(all_notes(&repo).await.len(), 1);
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
    assert_eq!(entity_snapshot(&repo).await, entities_before);
}

#[tokio::test]
async fn atomic_note_entity_schema_failure_rolls_back_every_entity_write() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let entities_before = entity_snapshot(&repo).await;
    for guarded in [true, false] {
        for detached in [true, false] {
            let mut malformed = entity("Malformed metadata");
            malformed.metadata = serde_json::json!("not an object");
            let entities = vec![
                entity("New entity before schema failure"),
                changed_original_entity(),
                malformed,
            ];
            let mut replacement = expected.clone();
            replacement.content = "must not survive entity storage failure".into();
            let result = match (guarded, detached) {
                (true, true) => {
                    repo.create_note_and_replace_entities_if_unchanged(
                        Note::new("must not create a failed detached copy"),
                        entities,
                        &expected,
                    )
                    .await
                }
                (true, false) => {
                    repo.update_note_and_replace_entities_if_unchanged(
                        &id(&expected),
                        replacement,
                        entities,
                        &expected,
                    )
                    .await
                }
                (false, true) => {
                    repo.create_note_and_replace_entities(
                        Note::new("must not create a failed manual capture"),
                        entities,
                    )
                    .await
                }
                (false, false) => {
                    repo.update_note_and_replace_entities(&id(&expected), replacement, entities)
                        .await
                }
            };
            assert!(matches!(result, Err(DbError::QueryFailed(_))), "{result:?}");
            let message = result.unwrap_err().to_string();
            assert!(message.contains("object"), "{message}");
            assert_eq!(entity_snapshot(&repo).await, entities_before);
            assert_eq!(all_notes(&repo).await.len(), 1);
            assert_snapshot(
                &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
                &expected,
            );
            assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
        }
    }
}

#[tokio::test]
async fn guarded_note_storage_failure_rolls_back_entities_before_note_write() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let entities_before = entity_snapshot(&repo).await;
    repo.db
        .query(
            "DEFINE FIELD OVERWRITE content ON note TYPE string ASSERT $value != 'rejected body'",
        )
        .await
        .unwrap()
        .check()
        .unwrap();
    for detached in [false, true] {
        let entities = vec![
            entity("Entity before rejected note"),
            changed_original_entity(),
        ];
        let mut replacement = expected.clone();
        replacement.content = "rejected body".into();
        let result = if detached {
            repo.create_note_and_replace_entities_if_unchanged(
                Note::new("rejected body"),
                entities,
                &expected,
            )
            .await
        } else {
            repo.update_note_and_replace_entities_if_unchanged(
                &id(&expected),
                replacement,
                entities,
                &expected,
            )
            .await
        };
        assert!(matches!(result, Err(DbError::QueryFailed(_))), "{result:?}");
        assert!(result.unwrap_err().to_string().contains("content"));
        assert_eq!(entity_snapshot(&repo).await, entities_before);
        assert_eq!(all_notes(&repo).await.len(), 1);
        assert_snapshot(
            &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
            &expected,
        );
        assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
    }
}

#[tokio::test]
async fn atomic_note_transaction_preserves_entity_upsert_and_duplicate_alias_semantics() {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.db
        .query("DEFINE FIELD metadata.preserved ON entity TYPE option<string>; DEFINE FIELD metadata.updated ON entity TYPE option<string>;")
        .await
        .unwrap()
        .check()
        .unwrap();
    let mut original = entity("Atlas Project").with_embedding(vec![0.25; 1024]);
    original.metadata = serde_json::json!({
        "aliases": ["prior alias", "shared alias"],
        "preserved": "original metadata",
        "updated": "old value",
    });
    let original = repo.upsert_entity(original).await.unwrap();
    let expected = initial(&repo).await;
    let mut first =
        Entity::new("ATLAS PROJECT", EntityType::Organization).with_embedding(vec![0.75; 1024]);
    first.metadata = serde_json::json!({
        "aliases": ["shared alias", "first alias"],
        "updated": "first value",
    });
    let mut second = Entity::new(" Atlas   Project ", EntityType::Person);
    second.metadata = serde_json::json!({
        "aliases": ["second alias", "first alias"],
        "updated": "second value",
    });
    let mut replacement = expected.clone();
    replacement.content = "Atlas project has its updated mentions".into();
    let updated = repo
        .update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            replacement,
            vec![first, second],
            &expected,
        )
        .await
        .unwrap();
    let linked = repo.get_entities_for_note(&id(&updated)).await.unwrap();
    assert_eq!(
        linked.len(),
        1,
        "canonical duplicates must have one mention"
    );
    let stored = &linked[0];
    assert_eq!(stored.id, original.id);
    assert_eq!(stored.name, " Atlas   Project ");
    assert_eq!(stored.canonical_name, "atlas project");
    assert_eq!(stored.entity_type, original.entity_type);
    assert_eq!(stored.created_at, original.created_at);
    assert!(
        stored.embedding.is_empty(),
        "last upsert replaces embedding"
    );
    assert_eq!(stored.metadata["preserved"], "original metadata");
    assert_eq!(stored.metadata["updated"], "second value");
    let mut aliases: Vec<String> =
        serde_json::from_value(stored.metadata["aliases"].clone()).unwrap();
    aliases.sort();
    assert_eq!(
        aliases,
        vec!["first alias", "prior alias", "second alias", "shared alias"]
    );
}

#[tokio::test]
async fn independent_repository_locks_cannot_both_commit_the_same_editor_snapshot() {
    let connection = init_memory().await.unwrap();
    let left = Repository::new(DbConnection::new((*connection).clone()));
    let right = Repository::new(DbConnection::new((*connection).clone()));
    assert!(!Arc::ptr_eq(
        &left.proposal_acceptance_lock,
        &right.proposal_acceptance_lock
    ));
    let expected = initial(&left).await;
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let first_expected = expected.clone();
    let first_barrier = barrier.clone();
    let first = tokio::spawn(async move {
        let mut replacement = first_expected.clone();
        replacement.content = "first concurrent editor".into();
        first_barrier.wait().await;
        left.update_note_and_replace_entities_if_unchanged(
            &id(&first_expected),
            replacement,
            Vec::new(),
            &first_expected,
        )
        .await
    });
    let second_expected = expected.clone();
    let second = tokio::spawn(async move {
        let mut replacement = second_expected.clone();
        replacement.content = "second concurrent editor".into();
        barrier.wait().await;
        right
            .update_note_and_replace_entities_if_unchanged(
                &id(&second_expected),
                replacement,
                Vec::new(),
                &second_expected,
            )
            .await
    });
    let results = [first.await.unwrap(), second.await.unwrap()];
    assert_eq!(
        results.iter().filter(|result| result.is_ok()).count(),
        1,
        "{results:?}"
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(DbError::NoteRevisionConflict(_))))
            .count(),
        1,
        "{results:?}"
    );
    let stored = Repository::new(connection)
        .get_note(&id(&expected))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        stored.content.as_str(),
        "first concurrent editor" | "second concurrent editor"
    ));
}
