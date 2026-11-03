use super::*;
use crate::init_memory;
use std::time::Instant;

fn id(note: &Note) -> String {
    record_id_to_string(note.id.as_ref().unwrap())
}

#[tokio::test]
async fn keyword_search_falls_back_to_any_term_bm25_without_displacing_all_term_matches() {
    let repo = Repository::new(init_memory().await.unwrap());
    let all_terms = repo
        .create_note(Note::new("surrealdb 3.3.5 fam deploy"))
        .await
        .unwrap();
    let three_terms = repo
        .create_note(Note::new("surrealdb 3.3.5 fam upgrade notes"))
        .await
        .unwrap();
    let two_terms = repo
        .create_note(Note::new("fam deploy checklist"))
        .await
        .unwrap();
    for index in 0..32 {
        repo.create_note(Note::new(format!("unrelated fixture {index}")))
            .await
            .unwrap();
    }

    let results = repo
        .fulltext_search("surrealdb, 3.3.5 fam deploy", 3)
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(record_id_to_string(&results[0].id), id(&all_terms));
    assert_eq!(record_id_to_string(&results[1].id), id(&three_terms));
    assert_eq!(record_id_to_string(&results[2].id), id(&two_terms));
    assert!(results.iter().all(|result| result.fts_score.is_some()));
    assert!(results[1].fts_score.unwrap() > results[2].fts_score.unwrap());
}

#[tokio::test]
async fn keyword_fallback_does_not_count_substrings_as_extra_terms() {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.create_note(Note::new("memory asynchronous"))
        .await
        .unwrap();

    let results = repo
        .fulltext_search("memory sync recovery", 10)
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn keyword_fallback_matches_ascii_folded_tokens() {
    let repo = Repository::new(init_memory().await.unwrap());
    let accented = repo
        .create_note(Note::new("café résumé migration notes"))
        .await
        .unwrap();

    let results = repo
        .fulltext_search("cafe resume deploy", 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(record_id_to_string(&results[0].id), id(&accented));
}

#[tokio::test]
async fn keyword_fallback_uses_analyzed_terms_when_punctuation_has_no_spaces() {
    let repo = Repository::new(init_memory().await.unwrap());
    let partial_match = repo
        .create_note(Note::new("memory sync incident"))
        .await
        .unwrap();

    let results = repo
        .fulltext_search("memory,sync,recovery", 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(record_id_to_string(&results[0].id), id(&partial_match));
}

#[tokio::test]
async fn keyword_fallback_uses_analyzer_terms_across_letter_digit_transitions() {
    let repo = Repository::new(init_memory().await.unwrap());
    let partial_match = repo
        .create_note(Note::new("bge 3 deployment"))
        .await
        .unwrap();

    let results = repo
        .fulltext_search("bge3 deployment recovery", 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(record_id_to_string(&results[0].id), id(&partial_match));
}

#[tokio::test]
async fn analyzer_call_reports_query_term_latency() {
    let repo = Repository::new(init_memory().await.unwrap());
    let started = Instant::now();
    let terms = repo
        .analyze_lexical_terms("bge3 deployment recovery")
        .await
        .unwrap();
    eprintln!("analyzer call elapsed: {:?}", started.elapsed());

    assert_eq!(terms, ["3", "bge", "deployment", "recovery"]);
}

#[tokio::test]
async fn hybrid_fusion_keeps_all_term_lexical_hits_ahead_of_higher_bm25_fallbacks() {
    let repo = Repository::new(init_memory().await.unwrap());
    let all_terms = repo
        .create_note(Note::new("surrealdb fam deploy production"))
        .await
        .unwrap();
    repo.create_note(Note::new(
        "surrealdb fam surrealdb fam surrealdb fam surrealdb fam surrealdb fam",
    ))
    .await
    .unwrap();

    let results = repo
        .hybrid_search_notes_with_fusion(
            "surrealdb fam deploy production",
            vec![0.0; 1024],
            2,
            None,
            None,
            &FusionConfig {
                candidate_pool_min: 2,
                candidate_pool_max: 2,
                ..FusionConfig::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(record_id_to_string(&results[0].id), id(&all_terms));
    assert_eq!(results[0].fusion.fulltext_rank, Some(1));
}

#[tokio::test]
async fn keyword_fallback_counts_distinct_terms_across_fields_not_repeated_occurrences() {
    let repo = Repository::new(init_memory().await.unwrap());
    let two_terms = repo
        .create_note(Note::new("beta beta").with_title("alpha alpha"))
        .await
        .unwrap();
    repo.create_note(Note::new("alpha alpha").with_title("alpha alpha"))
        .await
        .unwrap();

    // Repetition within the query, within a field, and across fields must not
    // turn a one-distinct-term candidate into an eligible partial match.
    let results = repo
        .fulltext_search("alpha alpha beta gamma", 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(record_id_to_string(&results[0].id), id(&two_terms));
}
