//! Synthetic regression for index-local full-text document identities.
//!
//! Each full-text index owns a separate document-ID sequence. Building the
//! indexes at different population stages makes their allocations differ even
//! though they ultimately index the same note records. A search::score match
//! reference must resolve the score for that index and that record, rather
//! than reuse an unrelated iterator's internal document ID.
//!
//! The original OR query is observational only: an upstream fix may make it
//! agree with independent queries. The repository correctness assertion must
//! pass with either a repository workaround or a corrected database engine.

use super::*;
use crate::init_memory;
use std::collections::BTreeMap;

const QUERY: &str = "nebula";
const TARGET_KEY: &str = "zz-score-target";

#[derive(Debug, Deserialize, SurrealValue)]
struct ComponentScore {
    id: RecordId,
    score: Option<f32>,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct CombinedScores {
    id: RecordId,
    search_score: Option<f32>,
    content_score: Option<f32>,
    title_score: Option<f32>,
}

async fn execute(db: &DbConnection, sql: &str) {
    let mut response = db.query(sql).await.unwrap();
    assert!(response.take_errors().is_empty(), "fixture SQL failed");
}

async fn insert(db: &DbConnection, key: &str, title: &str, content: &str, search: &str) {
    // Direct typed CREATE includes the vector field, unlike the repository's
    // explicit NONE binding. Supply a valid vector; this fixture only runs FTS.
    let mut embedding = vec![0.0; 1024];
    embedding[0] = 1.0;
    let mut note = Note::new(content)
        .with_title(title)
        .with_embedding(embedding);
    note.search_content = Some(search.into());
    let saved: Option<Note> = db.create(("note", key)).content(note).await.unwrap();
    assert!(saved.is_some());
}

async fn independent_scores(
    db: &DbConnection,
    index: &str,
    field: &str,
    match_ref: usize,
) -> BTreeMap<String, f32> {
    // Index/field/reference come exclusively from constants in this test.
    let sql = format!(
        "SELECT id, search::score({match_ref}) AS score \
         FROM note WITH INDEX {index} WHERE {field} @{match_ref}@ $query"
    );
    let rows: Vec<ComponentScore> = db
        .query(sql)
        .bind(("query", QUERY))
        .await
        .unwrap()
        .take(0)
        .unwrap();
    rows.into_iter()
        .map(|row| {
            (
                record_id_to_string(&row.id),
                row.score.expect("single-index match must have a score"),
            )
        })
        .collect()
}

#[tokio::test]
async fn fulltext_score_uses_each_index_local_record_identity() {
    let db = init_memory().await.unwrap();
    // Rebuild the real note indexes inside this isolated in-memory namespace.
    // Title sees the target first. Content is built after reverse-ID inserts;
    // search_content is built after one further lexically earlier insertion.
    execute(
        &db,
        "REMOVE INDEX idx_note_title ON note; \
         REMOVE INDEX idx_note_content ON note; \
         REMOVE INDEX idx_note_search_content ON note; \
         DEFINE INDEX idx_note_title ON note FIELDS title FULLTEXT ANALYZER ascii BM25;",
    )
    .await;
    insert(
        &db,
        TARGET_KEY,
        "nebula nebula",
        &format!("nebula {}", "bodyfiller ".repeat(96)),
        &format!("nebula {}", "searchfiller ".repeat(24)),
    )
    .await;
    for ordinal in (0..7).rev() {
        insert(
            &db,
            &format!("aa-score-filler-{ordinal}"),
            "unrelated title",
            "unrelated body text",
            "unrelated search text",
        )
        .await;
    }
    execute(
        &db,
        "DEFINE INDEX idx_note_content ON note FIELDS content FULLTEXT ANALYZER ascii BM25;",
    )
    .await;
    insert(
        &db,
        "00-score-late-filler",
        "different title",
        "different body text",
        "different search text",
    )
    .await;
    execute(
        &db,
        "DEFINE INDEX idx_note_search_content ON note FIELDS search_content \
         FULLTEXT ANALYZER ascii BM25;",
    )
    .await;

    let target = format!("note:{TARGET_KEY}");
    let search_scores =
        independent_scores(&db, "idx_note_search_content", "search_content", 0).await;
    let content_scores = independent_scores(&db, "idx_note_content", "content", 1).await;
    let title_scores = independent_scores(&db, "idx_note_title", "title", 2).await;
    // No other synthetic record contains the query. Sufficient nonmatching
    // records ensure Surreal's nonnegative BM25 IDF is positive in every field.
    for scores in [&search_scores, &content_scores, &title_scores] {
        assert_eq!(scores.len(), 1);
        assert!(scores[&target].is_finite() && scores[&target] > 0.0);
    }
    let expected =
        search_scores[&target] * 0.7 + content_scores[&target] * 0.2 + title_scores[&target] * 0.1;

    for predicates in [
        "search_content @0@ $query OR content @1@ $query OR title @2@ $query",
        "title @2@ $query OR content @1@ $query OR search_content @0@ $query",
        "content @1@ $query OR title @2@ $query OR search_content @0@ $query",
    ] {
        let sql = format!(
            "SELECT id, search::score(0) AS search_score, \
             search::score(1) AS content_score, search::score(2) AS title_score \
             FROM note WHERE ({predicates}) ORDER BY id"
        );
        let rows: Vec<CombinedScores> = db
            .query(sql)
            .bind(("query", QUERY))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(record_id_to_string(&rows[0].id), target);
        eprintln!(
            "synthetic original OR scoring: predicates={predicates}; \
             combined={rows:?}; independent_search={}; independent_content={}; \
             independent_title={}; expected_weighted={expected}",
            search_scores[&target], content_scores[&target], title_scores[&target],
        );
        // Read the fields explicitly so future builds do not hide a missing
        // component behind the diagnostic row's derived Debug implementation.
        let _observed = (
            rows[0].search_score,
            rows[0].content_score,
            rows[0].title_score,
        );
    }

    let repo = Repository::new(db);
    let hits = repo
        .fulltext_search_notes(QUERY, 20, None, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(record_id_to_string(&hits[0].id), target);
    let observed = hits[0]
        .fts_score
        .expect("repository match must have a score");
    assert!(
        (observed - expected).abs() <= 1.0e-5 * expected.max(1.0),
        "repository score must agree with independently evaluated field scores: \
         observed={observed}, expected={expected}"
    );
}
