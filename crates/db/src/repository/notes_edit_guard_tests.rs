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
    repo.create_note_and_replace_entities(
        Note::new("original editor body")
            .with_title("Original title")
            .with_tags(vec!["original".into()]),
        vec![entity("Original mention")],
    )
    .await
    .unwrap()
}

async fn all_notes(repo: &Repository) -> Vec<Note> {
    repo.db.select("note").await.unwrap()
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
        let mut draft = expected.clone();
        draft.content = "stale draft must never overwrite current state".into();
        assert_conflict(
            repo.update_note_and_replace_entities_if_unchanged(
                &id(&expected),
                draft,
                vec![entity("Stale mention")],
                &expected,
            )
            .await,
        );
        let actual = repo.get_note(&id(&expected)).await.unwrap().unwrap();
        assert_snapshot(&actual, &current);
        assert_eq!(mentions(&repo, &actual).await, vec!["Original mention"]);
        assert_eq!(all_notes(&repo).await.len(), 1);
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
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("stale detached copy"),
            vec![entity("Stale detached mention")],
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
}

#[tokio::test]
async fn deleted_editor_source_is_never_recreated_or_detached() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    repo.delete_note(&id(&expected)).await.unwrap();
    let mut replacement = expected.clone();
    replacement.content = "must not recreate deleted note".into();
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            replacement,
            vec![entity("Recreated mention")],
            &expected,
        )
        .await,
    );
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("must not detach deleted note"),
            Vec::new(),
            &expected,
        )
        .await,
    );
    assert!(repo.get_note(&id(&expected)).await.unwrap().is_none());
    assert!(all_notes(&repo).await.is_empty());
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
    assert!(repo.get_visible_note(&id(&staged)).await.unwrap().is_none());
    assert_conflict(
        repo.create_note_and_replace_entities_if_unchanged(
            Note::new("staged draft"),
            Vec::new(),
            &staged,
        )
        .await,
    );
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&staged),
            staged.clone(),
            Vec::new(),
            &staged,
        )
        .await,
    );

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
            Vec::new(),
            &expected,
        )
        .await,
    );
    assert_conflict(
        repo.update_note_and_replace_entities_if_unchanged(
            &id(&expected),
            expected.clone(),
            Vec::new(),
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
}

#[tokio::test]
async fn guarded_note_and_detached_transactions_roll_back_on_mention_failure() {
    let repo = Repository::new(init_memory().await.unwrap());
    let expected = initial(&repo).await;
    let blocked = repo.upsert_entity(entity("Blocked mention")).await.unwrap();
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
            vec![entity("Blocked mention")],
            &expected,
        )
        .await;
    assert!(
        matches!(failed_update, Err(DbError::QueryFailed(_))),
        "{failed_update:?}"
    );
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
    let failed_detach = repo
        .create_note_and_replace_entities_if_unchanged(
            Note::new("must roll back detached note"),
            vec![entity("Blocked mention")],
            &expected,
        )
        .await;
    assert!(
        matches!(failed_detach, Err(DbError::QueryFailed(_))),
        "{failed_detach:?}"
    );
    assert_eq!(all_notes(&repo).await.len(), 1);
    assert_snapshot(
        &repo.get_note(&id(&expected)).await.unwrap().unwrap(),
        &expected,
    );
    assert_eq!(mentions(&repo, &expected).await, vec!["Original mention"]);
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
