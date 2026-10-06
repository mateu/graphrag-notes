//! Fictional pinned-engine FTS projection experiments; rankings remain exact.
use super::*;
use crate::init_memory;

fn projection() -> &'static str {
    "id, title, content, note_type, tags, created_at, source_id.uri AS source_uri"
}

fn predicates() -> &'static str {
    "(search_content @0@ $query OR content @1@ $query OR title @2@ $query) \
     AND ($since = NONE OR created_at >= <datetime>$since) \
     AND ($source_uri = NONE OR source_id.uri = $source_uri) \
     AND (source_id IS NONE OR source_generation IS NONE \
          OR source_generation = source_id.successful_generation)"
}

fn exact_title() -> &'static str {
    "($normalized_title != '' AND string::lowercase(string::trim(title ?? '')) = $normalized_title)"
}

fn score() -> &'static str {
    "(search::score(0) * 0.7 + search::score(1) * 0.2 + search::score(2) * 0.1)"
}

fn old_query() -> String {
    format!(
        "SELECT {}, {} AS exact_title_match, {} AS fts_score FROM note WHERE {} \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC LIMIT $limit",
        projection(),
        exact_title(),
        score(),
        predicates()
    )
}

fn candidate_query() -> String {
    format!(
        "SELECT id, {} AS exact_title_match, {} AS fts_score FROM note WHERE {} \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC LIMIT $limit",
        exact_title(),
        score(),
        predicates()
    )
}

fn narrow_query() -> String {
    format!(
        "SELECT id, id.title AS title, id.content AS content, id.note_type AS note_type, \
         id.tags AS tags, id.created_at AS created_at, id.source_id.uri AS source_uri, \
         exact_title_match, fts_score FROM ({}) \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
        candidate_query()
    )
}

fn once_hydrated_query() -> String {
    format!(
        "SELECT note.id AS id, note.title AS title, note.content AS content, \
         note.note_type AS note_type, note.tags AS tags, note.created_at AS created_at, \
         note.source_id.uri AS source_uri, exact_title_match, fts_score FROM \
         (SELECT id.* AS note, exact_title_match, fts_score FROM ({})) \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
        candidate_query()
    )
}

async fn query(db: &DbConnection, sql: &str, text: &str, limit: usize) -> Vec<SearchResult> {
    db.query(sql.to_owned())
        .bind(("query", text.to_owned()))
        .bind(("normalized_title", text.trim().to_lowercase()))
        .bind(("limit", limit))
        .bind(("since", Option::<String>::None))
        .bind(("source_uri", Option::<String>::None))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

#[tokio::test]
#[ignore = "opt-in broad/narrow FTS plan and projection experiment; no hardware CI budget"]
async fn fulltext_plan_and_projection_diagnostic() {
    let db = init_memory().await.unwrap();
    for n in 0..256 {
        let mut vector = vec![0.0; 1024];
        vector[0] = 1.0;
        let term = if n % 5 == 0 { "quasar" } else { "unrelated" };
        let note = Note::new(format!(
            "quasar unique{n} {}",
            "fictional body ".repeat(512)
        ))
        .with_title(if n == 255 {
            "quasar".to_owned()
        } else {
            format!("{term} fixture{n}")
        })
        .with_embedding(vector);
        let mut note = note;
        note.search_content = Some(format!(
            "quasar unique{n} {}",
            "fictional searchable text ".repeat(256)
        ));
        let _: Option<Note> = db
            .create(("note", format!("fixture-{n:04}")))
            .content(note)
            .await
            .unwrap();
    }
    let mut runs = Vec::new();
    for text in ["quasar", "unique123", "absent fictional constellation"] {
        for limit in [5, 50] {
            let baseline = query(&db, &old_query(), text, limit).await;
            for (name, sql) in [
                ("baseline", old_query()),
                ("narrow", narrow_query()),
                ("once_hydrated", once_hydrated_query()),
            ] {
                let mut samples = Vec::new();
                let mut exact = true;
                let mut score_bits_exact = true;
                for _ in 0..21 {
                    let started = std::time::Instant::now();
                    let rows = query(&db, &sql, text, limit).await;
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    exact &= serde_json::to_value(&rows).unwrap()
                        == serde_json::to_value(&baseline).unwrap();
                    score_bits_exact &= rows
                        .iter()
                        .map(|row| row.fts_score.map(f32::to_bits))
                        .collect::<Vec<_>>()
                        == baseline
                            .iter()
                            .map(|row| row.fts_score.map(f32::to_bits))
                            .collect::<Vec<_>>();
                }
                let plan: Vec<serde_json::Value> = db
                    .query(format!("{sql} EXPLAIN FULL"))
                    .bind(("query", text.to_owned()))
                    .bind(("normalized_title", text.to_lowercase()))
                    .bind(("limit", limit))
                    .bind(("since", Option::<String>::None))
                    .bind(("source_uri", Option::<String>::None))
                    .await
                    .unwrap()
                    .take(0)
                    .unwrap();
                runs.push(serde_json::json!({"variant":name,"query":text,"limit":limit,"rows":baseline.len(),
                    "full_rows_exact":exact,"score_bits_exact":score_bits_exact,"first_ms":samples[0],
                    "warm_ms":samples[1..],"plan":plan}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":256,"provider_calls":0,
        "engine":"surrealdb-core 3.2.4","runs":runs})
    );
}

// This is a fictional, test-only table. No runtime table, schema version or
// repository query is changed until exactness and optimized timing justify it.
async fn install_lexical_probe(db: &DbConnection) {
    db.query(
        r#"
        DEFINE TABLE lexical_probe SCHEMAFULL;
        DEFINE FIELD record_id ON lexical_probe TYPE record<note>;
        DEFINE FIELD content ON lexical_probe TYPE string;
        DEFINE FIELD search_content ON lexical_probe TYPE option<string>;
        DEFINE FIELD title ON lexical_probe TYPE option<string>;
        DEFINE FIELD created_at ON lexical_probe TYPE datetime;
        DEFINE FIELD source_id ON lexical_probe TYPE option<record<source>>;
        DEFINE FIELD source_generation ON lexical_probe TYPE option<int>;
        DEFINE INDEX lexical_content ON lexical_probe FIELDS content FULLTEXT ANALYZER ascii BM25;
        DEFINE INDEX lexical_search_content ON lexical_probe FIELDS search_content FULLTEXT ANALYZER ascii BM25;
        DEFINE INDEX lexical_title ON lexical_probe FIELDS title FULLTEXT ANALYZER ascii BM25;
        DEFINE FUNCTION fn::materialize_lexical_probe($row: object) {
            UPSERT type::record('lexical_probe', record::id($row.id)) CONTENT {
                record_id: $row.id, content: $row.content,
                search_content: $row.search_content, title: $row.title,
                created_at: $row.created_at, source_id: $row.source_id,
                source_generation: $row.source_generation
            };
            RETURN NONE;
        };
        DEFINE EVENT materialize_lexical_probe ON note WHEN $event != 'DELETE'
            THEN fn::materialize_lexical_probe($after);
        DEFINE EVENT delete_lexical_probe ON note WHEN $event = 'DELETE'
            THEN (DELETE type::record('lexical_probe', record::id($before.id)));
        FOR $id IN (SELECT VALUE id FROM note ORDER BY id ASC) {
            fn::materialize_lexical_probe($id.*);
        };
        "#,
    )
    .await
    .unwrap()
    .check()
    .unwrap();
}

fn lexical_candidate_query() -> String {
    format!(
        "SELECT id, record_id, {} AS exact_title_match, {} AS fts_score \
         FROM lexical_probe WHERE {} \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC LIMIT $limit",
        exact_title(),
        score(),
        predicates()
    )
}

fn lexical_hydration_query(once: bool) -> String {
    if once {
        format!(
            "SELECT note.id AS id, note.title AS title, note.content AS content, \
             note.note_type AS note_type, note.tags AS tags, note.created_at AS created_at, \
             note.source_id.uri AS source_uri, exact_title_match, fts_score FROM \
             (SELECT record_id.* AS note, exact_title_match, fts_score FROM ({})) \
             ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
            lexical_candidate_query()
        )
    } else {
        format!(
            "SELECT record_id AS id, record_id.title AS title, record_id.content AS content, \
             record_id.note_type AS note_type, record_id.tags AS tags, \
             record_id.created_at AS created_at, record_id.source_id.uri AS source_uri, \
             exact_title_match, fts_score FROM ({}) \
             ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
            lexical_candidate_query()
        )
    }
}

async fn filtered_query(
    db: &DbConnection,
    sql: &str,
    text: &str,
    limit: usize,
    since: Option<String>,
    source: Option<String>,
) -> Vec<SearchResult> {
    db.query(sql.to_owned())
        .bind(("query", text.to_owned()))
        .bind(("normalized_title", text.trim().to_lowercase()))
        .bind(("limit", limit))
        .bind(("since", since))
        .bind(("source_uri", source))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

async fn populate_lexical_fixture(db: &DbConnection, count: usize) {
    let mut source = Source::manual().with_content("Fictional indexed original ".repeat(1024));
    source.uri = Some("file:///fictional/lexical-primary".into());
    source.generation = 3;
    source.successful_generation = 2;
    let saved: Option<Source> = db
        .create(("source", "lexical-primary"))
        .content(source)
        .await
        .unwrap();
    let source = saved.unwrap().id.unwrap();
    for n in 0..count {
        let mut vector = vec![0.0; 1024];
        vector[0] = 1.0;
        let mut note = Note::new(format!(
            "quasar unique{n} {}",
            "fictional body ".repeat([0, 512, 2048][n % 3])
        ))
        .with_title(if n == count - 1 {
            "quasar".into()
        } else {
            format!("Fictional quasar {n}")
        })
        .with_embedding(vector);
        note.search_content = (n % 7 != 0).then(|| format!("quasar unique{n} searchable context"));
        note.created_at = chrono::DateTime::parse_from_rfc3339(if n % 2 == 0 {
            "2025-01-01T00:00:00Z"
        } else {
            "2026-01-01T00:00:00Z"
        })
        .unwrap()
        .with_timezone(&chrono::Utc);
        if n % 5 != 0 {
            note.source_id = Some(source.clone());
            note.source_generation = if n % 11 == 0 {
                None
            } else if n % 13 == 0 {
                Some(3)
            } else {
                Some(2)
            };
        }
        if n % 17 == 0 {
            note.embedding.clear();
        }
        db.query("LET $payload = IF array::len($note.embedding) = 0 THEN object::remove($note, 'embedding') ELSE $note END; CREATE $id CONTENT $payload;")
            .bind(("id", RecordId::new("note", format!("lexical-{n:04}"))))
            .bind(("note", note))
            .await.unwrap().check().unwrap();
    }
}

fn assert_lexical_exact(actual: &[SearchResult], expected: &[SearchResult]) {
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    assert_eq!(
        actual
            .iter()
            .map(|row| row.fts_score.map(f32::to_bits))
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|row| row.fts_score.map(f32::to_bits))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn thin_lexical_population_and_hydration_preserve_weighted_scores_and_visibility() {
    let db = init_memory().await.unwrap();
    populate_lexical_fixture(&db, 48).await;
    install_lexical_probe(&db).await;
    // Updates and deletes change both index populations in the same transaction.
    for mutation in [
        "RETURN NONE;",
        "UPDATE note:`lexical-0017` SET title = ' QUASAR ', content = 'quasar altered fictional text', search_content = NONE; DELETE note:`lexical-0018`;",
        "UPDATE source:`lexical-primary` SET successful_generation = 3;",
    ] {
        db.query(mutation).await.unwrap().check().unwrap();
        for text in ["quasar", " QUASAR ", "unique23", "absent fictional constellation"] {
            for limit in [5, 50] {
                for (since, source) in [
                    (None, None),
                    (Some("2025-06-01T00:00:00Z".to_owned()), None),
                    (None, Some("file:///fictional/lexical-primary".to_owned())),
                ] {
                    let baseline = filtered_query(&db, &old_query(), text, limit, since.clone(), source.clone()).await;
                    for once in [false, true] {
                        let rows = filtered_query(&db, &lexical_hydration_query(once), text, limit, since.clone(), source.clone()).await;
                        assert_lexical_exact(&rows, &baseline);
                    }
                }
            }
        }
    }
}

#[tokio::test]
#[ignore = "opt-in thin lexical storage experiment; no hardware CI budget"]
async fn thin_lexical_storage_diagnostic() {
    let backend =
        std::env::var("GRAPHRAG_FTS_DIAGNOSTIC_BACKEND").unwrap_or_else(|_| "memory".into());
    let db = match backend.as_str() {
        "memory" => init_memory().await.unwrap(),
        #[cfg(feature = "rocksdb")]
        "rocksdb" => {
            let path = std::env::var("GRAPHRAG_FTS_DIAGNOSTIC_PATH")
                .expect("owned absent RocksDB diagnostic path required");
            assert!(
                !std::path::Path::new(&path).exists(),
                "refusing an existing diagnostic database"
            );
            crate::init_persistent(path).await.unwrap()
        }
        _ => panic!("unsupported diagnostic backend"),
    };
    let count: usize = std::env::var("GRAPHRAG_FTS_DIAGNOSTIC_RECORDS")
        .unwrap_or_else(|_| "256".into())
        .parse()
        .unwrap();
    assert!((48..=4096).contains(&count));
    populate_lexical_fixture(&db, count).await;
    install_lexical_probe(&db).await;
    let mut runs = Vec::new();
    for text in ["quasar", "unique123", "absent fictional constellation"] {
        for limit in [5, 50] {
            let reference = query(&db, &old_query(), text, limit).await;
            for (name, sql) in [
                ("baseline", old_query()),
                ("thin_direct", lexical_hydration_query(false)),
                ("thin_once", lexical_hydration_query(true)),
            ] {
                let mut samples = Vec::new();
                for _ in 0..21 {
                    let started = std::time::Instant::now();
                    let rows = query(&db, &sql, text, limit).await;
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    assert_lexical_exact(&rows, &reference);
                }
                let plan: Vec<serde_json::Value> = db
                    .query(format!("{sql} EXPLAIN FULL"))
                    .bind(("query", text.to_owned()))
                    .bind(("normalized_title", text.trim().to_lowercase()))
                    .bind(("limit", limit))
                    .bind(("since", Option::<String>::None))
                    .bind(("source_uri", Option::<String>::None))
                    .await
                    .unwrap()
                    .take(0)
                    .unwrap();
                runs.push(serde_json::json!({"variant":name,"limit":limit,"fictional_query":text,
                    "full_rows_and_score_bits_exact":true,"first_measured_ms":samples[0],"warm_ms":samples[1..],"plan":plan}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":count,"backend":backend,"engine":"surrealdb-core 3.2.4", "provider_calls":0,"qualification":"Legacy reference precedes each category; first measured observation is not cold storage.","runs":runs})
    );
}
