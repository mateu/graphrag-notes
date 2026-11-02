use super::*;
use crate::init_memory;

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
