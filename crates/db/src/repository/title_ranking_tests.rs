use super::*;
use crate::{fusion::FusionStrategy, init_memory};

fn query_embedding() -> Vec<f32> {
    let mut embedding = vec![0.0; 1024];
    embedding[0] = 1.0;
    embedding
}

fn id(note: &Note) -> String {
    record_id_to_string(note.id.as_ref().unwrap())
}

#[tokio::test]
async fn exact_title_survives_keyword_and_hybrid_candidate_cutoffs_without_reindexing() {
    let repo = Repository::new(init_memory().await.unwrap());
    for index in 0..80 {
        repo.create_note(
            Note::new("Orion Notebook Orion Notebook")
                .with_title(format!("Orion Notebook discussion {index}"))
                .with_embedding(query_embedding()),
        )
        .await
        .unwrap();
    }
    // Existing title index alone can discover this note. Its display body
    // lacks the query and it has no vector to enter the KNN candidates.
    let named = repo
        .create_note(
            Note::new("Available to the assistant through a shared gateway")
                .with_title("Orion Notebook"),
        )
        .await
        .unwrap();
    let snapshot = repo.get_note(&id(&named)).await.unwrap().unwrap();
    let keyword = repo
        .fulltext_search_notes("Orion Notebook", 5, None, None)
        .await
        .unwrap();
    assert_eq!(keyword.len(), 5);
    assert_eq!(record_id_to_string(&keyword[0].id), id(&named));
    assert!(keyword[0].exact_title_match);

    for strategy in [FusionStrategy::ReciprocalRank, FusionStrategy::Weighted] {
        let results = repo
            .hybrid_search_notes_with_fusion(
                "Orion Notebook",
                query_embedding(),
                5,
                None,
                None,
                &FusionConfig {
                    strategy,
                    candidate_pool_min: 5,
                    candidate_pool_max: 5,
                    ..FusionConfig::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            record_id_to_string(&results[0].id),
            id(&named),
            "{strategy:?}"
        );
        assert!(results[0].fusion.exact_title_match);
        assert_eq!(results[0].fusion.fulltext_rank, Some(1));
        assert!(results[0].fusion.vector_rank.is_none());
    }
    let unchanged = repo.get_note(&id(&named)).await.unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(&unchanged).unwrap(),
        serde_json::to_value(&snapshot).unwrap()
    );
    assert_eq!(unchanged.created_at, snapshot.created_at);
    assert_eq!(unchanged.updated_at, snapshot.updated_at);
}

#[tokio::test]
async fn exact_title_normalization_preserves_punctuation_and_internal_whitespace() {
    let repo = Repository::new(init_memory().await.unwrap());
    for title in [
        "Orion Notebook",
        "  ORION NOTEBOOK  ",
        "Orion  Notebook",
        "Orion-Notebook",
        "Orion Notebook!",
        "Orion Notebook discussion",
    ] {
        repo.create_note(Note::new("Orion Notebook").with_title(title))
            .await
            .unwrap();
    }
    for (query, expected) in [
        (
            "  oRiOn nOtEbOoK  ",
            vec!["  ORION NOTEBOOK  ", "Orion Notebook"],
        ),
        ("Orion  Notebook", vec!["Orion  Notebook"]),
        ("Orion-Notebook", vec!["Orion-Notebook"]),
        ("Orion Notebook!", vec!["Orion Notebook!"]),
    ] {
        let results = repo.fulltext_search(query, 20).await.unwrap();
        let mut exact = results
            .iter()
            .filter(|result| result.exact_title_match)
            .map(|result| result.title.as_deref().unwrap())
            .collect::<Vec<_>>();
        exact.sort();
        assert_eq!(exact, expected, "query {query:?}");
        assert!(results[0].exact_title_match, "query {query:?}");
    }
}

#[tokio::test]
async fn exact_title_candidates_respect_source_time_and_visible_generation_filters() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut source = Source::from_file("selected.md", SourceType::Markdown).unwrap();
    source.generation = 2;
    source.successful_generation = 2;
    source.status = SourceIngestionStatus::Ready;
    let source = repo.create_source(source).await.unwrap();
    let source_id = source.id.clone().unwrap();
    let source_uri = source.uri.clone().unwrap();
    let mut current = Note::new("current eligible body")
        .with_title("Orion Notebook")
        .with_source(source_id.clone())
        .with_source_generation(2);
    current.created_at = Utc::now();
    let current = repo.create_note(current).await.unwrap();
    for generation in [1, 3] {
        repo.create_note(
            Note::new("hidden generation")
                .with_title("Orion Notebook")
                .with_source(source_id.clone())
                .with_source_generation(generation),
        )
        .await
        .unwrap();
    }
    let mut old = Note::new("old eligible body")
        .with_title("Orion Notebook")
        .with_source(source_id)
        .with_source_generation(2);
    old.created_at = Utc::now() - chrono::Duration::days(30);
    repo.create_note(old).await.unwrap();
    repo.create_note(Note::new("other source body").with_title("Orion Notebook"))
        .await
        .unwrap();
    let since = Some(Utc::now() - chrono::Duration::days(1));
    let keyword = repo
        .fulltext_search_notes("Orion Notebook", 20, since, Some(source_uri.clone()))
        .await
        .unwrap();
    assert_eq!(keyword.len(), 1);
    assert_eq!(record_id_to_string(&keyword[0].id), id(&current));
    for strategy in [FusionStrategy::ReciprocalRank, FusionStrategy::Weighted] {
        let hybrid = repo
            .hybrid_search_notes_with_fusion(
                "Orion Notebook",
                query_embedding(),
                5,
                since,
                Some(source_uri.clone()),
                &FusionConfig {
                    strategy,
                    ..FusionConfig::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(hybrid.len(), 1);
        assert_eq!(record_id_to_string(&hybrid[0].id), id(&current));
        assert!(hybrid[0].fusion.exact_title_match);
    }
}
