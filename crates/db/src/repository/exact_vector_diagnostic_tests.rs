//! Provider-free experiments against the pinned engine, before choosing SQL.

use super::*;
use crate::init_memory;

fn projection() -> &'static str {
    "id, title, content, note_type, tags, created_at, source_id.uri AS source_uri"
}

fn predicates() -> &'static str {
    "($since = NONE OR created_at >= <datetime>$since) \
     AND ($source_uri = NONE OR source_id.uri = $source_uri) \
     AND (source_id IS NONE OR source_generation IS NONE \
          OR source_generation = source_id.successful_generation)"
}

fn old_query(limit: usize) -> String {
    format!(
        "SELECT {}, vector::distance::knn() AS vec_distance FROM note \
         WHERE embedding <|{limit},COSINE|> $embedding AND {} \
         ORDER BY vec_distance ASC, id ASC LIMIT $limit",
        projection(),
        predicates()
    )
}

fn narrow_query(limit: usize) -> String {
    format!(
        "SELECT id, id.title AS title, id.content AS content, id.note_type AS note_type, \
         id.tags AS tags, id.created_at AS created_at, id.source_id.uri AS source_uri, vec_distance \
         FROM (SELECT id, vector::distance::knn() AS vec_distance FROM note \
         WHERE embedding <|{limit},COSINE|> $embedding AND {} \
         ORDER BY vec_distance ASC, id ASC LIMIT $limit) \
         ORDER BY vec_distance ASC, id ASC",
        predicates()
    )
}

fn payload_query(limit: usize) -> String {
    format!(
        "SELECT id, id.* AS payload, id.source_id.uri AS source_uri, vec_distance \
         FROM (SELECT id, vector::distance::knn() AS vec_distance FROM note \
         WHERE embedding <|{limit},COSINE|> $embedding AND {} \
         ORDER BY vec_distance ASC, id ASC LIMIT $limit) \
         ORDER BY vec_distance ASC, id ASC",
        predicates()
    )
}

fn simple_query(limit: usize) -> String {
    old_query(limit).replace(
        "($since = NONE OR created_at >= <datetime>$since) \
         AND ($source_uri = NONE OR source_id.uri = $source_uri) AND ",
        "",
    )
}

#[derive(Deserialize, SurrealValue)]
struct PayloadRow {
    payload: SearchResult,
    vec_distance: Option<f32>,
    source_uri: Option<String>,
}

fn explicit_query() -> String {
    format!(
        "SELECT {}, (1 - vector::similarity::cosine(embedding, $embedding)) AS vec_distance FROM note \
         WHERE array::len(embedding ?? []) = array::len($embedding) AND {} \
         ORDER BY vec_distance ASC, id ASC LIMIT $limit",
        projection(), predicates()
    )
}

fn literal_knn_query(limit: usize, embedding: &[f32]) -> String {
    let vector = format!(
        "[{}]",
        embedding
            .iter()
            .map(|value| format!("{}f", f64::from(*value)))
            .collect::<Vec<_>>()
            .join(",")
    );
    old_query(limit).replace("$embedding", &vector)
}

fn prepared_sources_query(limit: usize) -> String {
    let sql = old_query(limit)
        .replace("source_id.uri", "$sources[<string>source_id].uri")
        .replace(
            "source_id.successful_generation",
            "$sources[<string>source_id].generation",
        );
    format!(
        "RETURN {{ LET $sources = object::from_entries((SELECT VALUE \
         [<string>id, {{generation: successful_generation, uri: uri}}] FROM source)); \
         RETURN ({sql}); }};"
    )
}

async fn query(
    db: &DbConnection,
    sql: String,
    embedding: &[f32],
    limit: usize,
    since: Option<String>,
    source: Option<String>,
) -> Vec<SearchResult> {
    let payload = sql.contains(" AS payload");
    let mut response = db
        .query(sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", limit))
        .bind(("since", since))
        .bind(("source_uri", source))
        .await
        .unwrap();
    if payload {
        let rows: Vec<PayloadRow> = response.take(0).unwrap();
        rows.into_iter()
            .map(|mut row| {
                row.payload.vec_distance = row.vec_distance;
                row.payload.source_uri = row.source_uri;
                row.payload
            })
            .collect()
    } else {
        response.take(0).unwrap()
    }
}

#[tokio::test]
#[ignore = "opt-in pinned-engine plans and paired timing, not a hardware CI budget"]
async fn exact_vector_plan_and_projection_diagnostic() {
    let db = init_memory().await.unwrap();
    let mut query_embedding = vec![0.0_f32; 1024];
    query_embedding[0] = 1.0;
    for n in 0..160 {
        let mut embedding = query_embedding.clone();
        embedding[1] = n as f32 / 160.0;
        let note = Note::new(format!(
            "Fictional vector row {n}: {}",
            "payload ".repeat(1024)
        ))
        .with_title(format!("Fictional row {n}"))
        .with_embedding(embedding);
        let _: Option<Note> = db
            .create(("note", format!("fixture-{n:04}")))
            .content(note)
            .await
            .unwrap();
    }
    let mut indexed_plans = Vec::new();
    for limit in [5, 50] {
        for (name, sql) in [
            ("baseline", old_query(limit)),
            ("narrow", narrow_query(limit)),
            ("payload", payload_query(limit)),
            ("simple", simple_query(limit)),
            ("explicit", explicit_query()),
            ("literal_knn", literal_knn_query(limit, &query_embedding)),
        ] {
            let plan: Vec<serde_json::Value> = db
                .query(format!("{sql} EXPLAIN FULL"))
                .bind(("embedding", query_embedding.clone()))
                .bind(("limit", limit))
                .bind(("since", Option::<String>::None))
                .bind(("source_uri", Option::<String>::None))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            indexed_plans.push(serde_json::json!({"variant":name,"limit":limit,"plan":plan}));
        }
    }
    let mut response = db
        .query("REMOVE INDEX idx_note_embedding ON note;")
        .await
        .unwrap();
    assert!(response.take_errors().is_empty());
    let mut runs = Vec::new();
    for limit in [5, 50] {
        let baseline = query(&db, old_query(limit), &query_embedding, limit, None, None).await;
        for (name, sql) in [
            ("baseline", old_query(limit)),
            ("narrow", narrow_query(limit)),
            ("payload", payload_query(limit)),
            ("simple", simple_query(limit)),
            ("explicit", explicit_query()),
            ("literal_knn", literal_knn_query(limit, &query_embedding)),
        ] {
            let mut samples = Vec::new();
            let mut exact = true;
            for _ in 0..21 {
                let started = std::time::Instant::now();
                let result = query(&db, sql.clone(), &query_embedding, limit, None, None).await;
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
                exact &= serde_json::to_value(&result).unwrap()
                    == serde_json::to_value(&baseline).unwrap();
            }
            let plan: Vec<serde_json::Value> = db
                .query(format!("{sql} EXPLAIN FULL"))
                .bind(("embedding", query_embedding.clone()))
                .bind(("limit", limit))
                .bind(("since", Option::<String>::None))
                .bind(("source_uri", Option::<String>::None))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            runs.push(
                serde_json::json!({"variant":name,"limit":limit,"rows":baseline.len(),
                "full_rows_exact":exact,"first_ms":samples[0],"warm_ms":samples[1..],"plan":plan}),
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":160,"dimension":1024,
        "provider_calls":0,"engine":"surrealdb-core 3.2.4", "retained_hnsw_plans":indexed_plans,"runs":runs})
    );
}

#[tokio::test]
#[ignore = "opt-in source-heavy exact scan experiment; no hardware CI budget"]
async fn exact_vector_source_preparation_diagnostic() {
    let db = init_memory().await.unwrap();
    let mut sources = Vec::new();
    for n in 0..32 {
        let mut source = Source::manual().with_content("fictional source material ".repeat(2048));
        source.uri = Some(format!("file:///fictional/vector-source-{n:02}"));
        source.successful_generation = 2;
        source.generation = 3;
        let saved: Option<Source> = db
            .create(("source", format!("source-{n:02}")))
            .content(source)
            .await
            .unwrap();
        sources.push(saved.unwrap().id.unwrap());
    }
    let mut query_embedding = vec![0.0_f32; 1024];
    query_embedding[0] = 1.0;
    for n in 0..160 {
        let mut embedding = query_embedding.clone();
        embedding[1] = n as f32 / 160.0;
        let mut note = Note::new(format!("Fictional source note {n}"))
            .with_title(format!("Fictional source row {n}"))
            .with_embedding(embedding);
        note.source_id = Some(sources[n % sources.len()].clone());
        note.source_generation = Some(2);
        let _: Option<Note> = db
            .create(("note", format!("fixture-{n:04}")))
            .content(note)
            .await
            .unwrap();
    }
    let limit = 50;
    let baseline = query(&db, old_query(limit), &query_embedding, limit, None, None).await;
    let mut runs = Vec::new();
    for (name, sql) in [
        ("baseline", old_query(limit)),
        ("prepared_sources", prepared_sources_query(limit)),
    ] {
        let mut samples = Vec::new();
        let mut exact = true;
        for _ in 0..21 {
            let started = std::time::Instant::now();
            let result = query(&db, sql.clone(), &query_embedding, limit, None, None).await;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            exact &=
                serde_json::to_value(&result).unwrap() == serde_json::to_value(&baseline).unwrap();
        }
        let explain = if sql.starts_with("RETURN") {
            sql.replace("LIMIT $limit);", "LIMIT $limit EXPLAIN FULL);")
        } else {
            format!("{sql} EXPLAIN FULL")
        };
        let plan: Vec<serde_json::Value> = db
            .query(explain)
            .bind(("embedding", query_embedding.clone()))
            .bind(("limit", limit))
            .bind(("since", Option::<String>::None))
            .bind(("source_uri", Option::<String>::None))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        runs.push(serde_json::json!({"variant":name,"full_rows_exact":exact,
            "first_ms":samples[0],"warm_ms":samples[1..],"plan":plan}));
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":160,"fictional_sources":32,
        "source_body_bytes":"fictional source material ".len()*2048,"dimension":1024,"provider_calls":0,
        "engine":"surrealdb-core 3.2.4","runs":runs})
    );
}

#[derive(Deserialize, SurrealValue)]
struct VectorOnlyRow {
    id: RecordId,
    embedding: Option<Vec<f64>>,
    note_id: Option<RecordId>,
    embedding_cbor: Option<surrealdb::types::Bytes>,
}

#[derive(Clone)]
struct ExactVectorCandidate {
    id: RecordId,
    distance: f64,
    seq: usize,
}

impl PartialEq for ExactVectorCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for ExactVectorCandidate {}
impl PartialOrd for ExactVectorCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ExactVectorCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let distance = if self.distance == 0.0 && other.distance == 0.0 {
            std::cmp::Ordering::Equal
        } else {
            self.distance.total_cmp(&other.distance)
        };
        distance.then(self.seq.cmp(&other.seq))
    }
}

fn page_table(cbor: bool) -> &'static str {
    if cbor {
        "exact_vector_cbor_probe"
    } else {
        "note"
    }
}

fn page_range(after: Option<&RecordId>, cbor: bool) -> Option<RecordId> {
    use std::ops::Bound;
    use surrealdb::types::{RecordIdKey, RecordIdKeyRange};
    after.map(|id| {
        RecordId::new(
            page_table(cbor),
            RecordIdKey::Range(Box::new(RecordIdKeyRange {
                start: Bound::Excluded(id.key.clone()),
                end: Bound::Unbounded,
            })),
        )
    })
}

fn page_query(after: bool, cbor: bool) -> String {
    format!(
        "SELECT {} FROM {} WHERE {} ORDER BY id ASC LIMIT $page",
        if cbor {
            "id, note_id, embedding_cbor"
        } else {
            "id, embedding"
        },
        if after { "$range" } else { page_table(cbor) },
        predicates()
    )
}

async fn paged_exact_vectors(
    db: &DbConnection,
    embedding: &[f32],
    limit: usize,
    page: usize,
    cbor: bool,
    since: Option<String>,
    source: Option<String>,
) -> Vec<SearchResult> {
    assert!(page > 0, "private diagnostic page must be positive");
    let tx = (**db).clone().begin().await.unwrap();
    let query_vector: Vec<f64> = embedding.iter().map(|n| f64::from(*n)).collect();
    let magnitude = query_vector.iter().map(|n| n.powi(2)).sum::<f64>().sqrt();
    let mut after: Option<RecordId> = None;
    let mut heap = std::collections::BinaryHeap::with_capacity(limit);
    let mut seq = 0;
    loop {
        let rows: Vec<VectorOnlyRow> = tx
            .query(page_query(after.is_some(), cbor))
            .bind(("range", page_range(after.as_ref(), cbor)))
            .bind(("page", page))
            .bind(("since", since.clone()))
            .bind(("source_uri", source.clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let count = rows.len();
        for row in rows {
            after = Some(row.id.clone());
            let vector = if cbor {
                let Some(bytes) = row.embedding_cbor else {
                    continue;
                };
                ciborium::from_reader::<Vec<f64>, _>(bytes.as_ref()).unwrap()
            } else {
                let Some(vector) = row.embedding else {
                    continue;
                };
                vector
            };
            if vector.len() != query_vector.len() || vector.is_empty() {
                continue;
            }
            let dot = vector
                .iter()
                .zip(&query_vector)
                .fold(0.0, |a, (x, y)| a + x * y);
            let norm = vector.iter().map(|n| n.powi(2)).sum::<f64>().sqrt();
            let distance = 1.0 - dot / (norm * magnitude);
            let candidate = ExactVectorCandidate {
                id: row.note_id.unwrap_or(row.id),
                distance,
                seq,
            };
            seq += 1;
            if heap.len() < limit {
                heap.push(candidate);
            } else if heap
                .peek()
                .is_some_and(|worst| candidate.cmp(worst).is_lt())
            {
                heap.pop();
                heap.push(candidate);
            }
        }
        if count < page {
            break;
        }
    }
    let selected = heap.into_sorted_vec();
    let ids: Vec<_> = selected
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    let rows: Vec<SearchResult> = tx
        .query(format!("SELECT {} FROM $ids", projection()))
        .bind(("ids", ids))
        .await
        .unwrap()
        .take(0)
        .unwrap();
    let output = selected
        .into_iter()
        .map(|candidate| {
            let mut row = rows
                .iter()
                .find(|row| row.id == candidate.id)
                .unwrap()
                .clone();
            row.vec_distance = Some(candidate.distance as f32);
            row
        })
        .collect();
    tx.cancel().await.unwrap();
    output
}

#[tokio::test]
#[ignore = "opt-in bounded paginated vector experiment; no hardware CI budget"]
async fn exact_vector_paged_float_diagnostic() {
    let db = init_memory().await.unwrap();
    let mut query_embedding = vec![0.0_f32; 1024];
    query_embedding[0] = 1.0;
    for n in 0..160 {
        let mut embedding = query_embedding.clone();
        embedding[1] = n as f32 / 160.0;
        let note = Note::new(format!(
            "Fictional vector row {n}: {}",
            "payload ".repeat(1024)
        ))
        .with_title(format!("Fictional row {n}"))
        .with_embedding(embedding);
        let _: Option<Note> = db
            .create(("note", format!("fixture-{n:04}")))
            .content(note)
            .await
            .unwrap();
    }
    let limit = 50;
    let baseline = query(&db, old_query(limit), &query_embedding, limit, None, None).await;
    let mut runs = Vec::new();
    for page in [None, Some(32), Some(256)] {
        let mut samples = Vec::new();
        let mut exact = true;
        let mut bits_exact = true;
        for _ in 0..21 {
            let started = std::time::Instant::now();
            let rows = if let Some(page) = page {
                paged_exact_vectors(&db, &query_embedding, limit, page, false, None, None).await
            } else {
                query(&db, old_query(limit), &query_embedding, limit, None, None).await
            };
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            exact &=
                serde_json::to_value(&rows).unwrap() == serde_json::to_value(&baseline).unwrap();
            bits_exact &= rows
                .iter()
                .map(|row| row.vec_distance.map(f32::to_bits))
                .collect::<Vec<_>>()
                == baseline
                    .iter()
                    .map(|row| row.vec_distance.map(f32::to_bits))
                    .collect::<Vec<_>>();
        }
        runs.push(
            serde_json::json!({"variant":if page.is_none() {"baseline"} else {"paged_f64"},
          "page":page,"full_rows_exact":exact,"distance_bits_exact":bits_exact,
          "first_ms":samples[0],"warm_ms":samples[1..]}),
        );
    }
    let mut plans = Vec::new();
    for after in [false, true] {
        let plan: Vec<serde_json::Value> = db
            .query(format!("{} EXPLAIN FULL", page_query(after, false)))
            .bind((
                "range",
                page_range(
                    after
                        .then(|| RecordId::new("note", "fixture-0032"))
                        .as_ref(),
                    false,
                ),
            ))
            .bind(("page", 32))
            .bind(("since", Option::<String>::None))
            .bind(("source_uri", Option::<String>::None))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        plans.push(serde_json::json!({"after_bound":after,"plan":plan}));
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":160,"dimension":1024,
      "provider_calls":0,"engine":"surrealdb-core 3.2.4","runs":runs,"plans":plans})
    );
}

#[tokio::test]
#[ignore = "opt-in paged prototype edge-case equivalence; not adopted runtime code"]
async fn exact_vector_paged_semantics_diagnostic() {
    use chrono::TimeZone;
    use surrealdb::types::{Object, RecordIdKey};
    let db = init_memory().await.unwrap();
    db.query(
        "REMOVE INDEX idx_note_embedding ON note; DEFINE TABLE exact_vector_cbor_probe SCHEMALESS",
    )
    .await
    .unwrap()
    .check()
    .unwrap();
    let mut source = Source::manual();
    source.uri = Some("fixture://vector-current".into());
    source.successful_generation = 2;
    source.generation = 3;
    let saved: Option<Source> = db
        .create(("source", "current"))
        .content(source)
        .await
        .unwrap();
    let source_id = saved.unwrap().id.unwrap();
    let mut compound = Object::new();
    compound.insert("label", "compound");
    compound.insert("source_generation", 99i64);
    let ids = vec![
        RecordId::new("note", "z-equal"),
        RecordId::new("note", "a-equal"),
        RecordId::new("note", 10i64),
        RecordId::new("note", RecordIdKey::Object(compound)),
        RecordId::new("note", "current"),
        RecordId::new("note", "stale"),
        RecordId::new("note", "legacy"),
        RecordId::new("note", "zero"),
        RecordId::new("note", "negative"),
        RecordId::new("note", "missing"),
        RecordId::new("note", "wrong-dimension"),
        RecordId::new(
            "note",
            RecordIdKey::Uuid(
                uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000001")
                    .unwrap()
                    .into(),
            ),
        ),
        RecordId::new(
            "note",
            RecordIdKey::Uuid(
                uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000002")
                    .unwrap()
                    .into(),
            ),
        ),
        RecordId::new("note", "signed-zero"),
    ];
    let positive = {
        let mut v = vec![0.0; 1024];
        v[0] = 1.0;
        v
    };
    for (index, id) in ids.iter().enumerate() {
        let mut note = Note::new(format!("Fictional contract row {index}"))
            .with_title(format!("Fictional contract title {index}"));
        note.created_at = Utc
            .with_ymd_and_hms(2030, 1, if index == 1 { 1 } else { 3 }, 0, 0, 0)
            .unwrap();
        note.embedding = match index {
            7 => vec![0.0; 1024],
            8 => {
                let mut v = positive.clone();
                v[0] = -1.0;
                v
            }
            9 => Vec::new(),
            10 => vec![1.0, 0.25],
            13 => {
                let mut v = vec![-0.0; 1024];
                v[0] = 1.0;
                v
            }
            _ => positive.clone(),
        };
        if [4, 5, 6].contains(&index) {
            note.source_id = Some(source_id.clone());
            note.source_generation = match index {
                4 => Some(2),
                5 => Some(3),
                _ => None,
            };
        }
        let _: Vec<Note> = db
            .query("CREATE $id CONTENT $note; CREATE $cbor_id SET note_id=$id,embedding_cbor=encoding::cbor::encode($note.embedding),created_at=$note.created_at,source_id=$note.source_id,source_generation=$note.source_generation")
            .bind(("cbor_id",RecordId::new("exact_vector_cbor_probe",id.key.clone())))
            .bind(("id", id.clone()))
            .bind(("note", note))
            .await
            .unwrap()
            .take(0)
            .unwrap();
    }
    for embedding in [&positive, &vec![0.0; 1024], &vec![1.0, 0.0], &vec![]] {
        for (since, source) in [
            (None, None),
            (Some("2030-01-02T00:00:00Z".to_string()), None),
            (None, Some("fixture://vector-current".to_string())),
            (None, Some("fixture://absent".to_string())),
        ] {
            for limit in [1, 3, 50] {
                let old = query(
                    &db,
                    old_query(limit),
                    embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await;
                for (page, cbor) in [
                    (1, false),
                    (3, false),
                    (256, false),
                    (1, true),
                    (3, true),
                    (256, true),
                ] {
                    let new = paged_exact_vectors(
                        &db,
                        embedding,
                        limit,
                        page,
                        cbor,
                        since.clone(),
                        source.clone(),
                    )
                    .await;
                    assert_eq!(
                        serde_json::to_value(&old).unwrap(),
                        serde_json::to_value(&new).unwrap(),
                        "dimension={} limit={limit} page={page} since={since:?} source={source:?}",
                        embedding.len()
                    );
                    assert_eq!(
                        old.iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        new.iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        "distance bits dimension={} limit={limit} page={page}",
                        embedding.len()
                    );
                }
            }
        }
    }
}

fn skinny_query(limit: usize) -> String {
    format!(
        "SELECT note.id AS id, note.title AS title, note.content AS content, \
         note.note_type AS note_type, note.tags AS tags, note.created_at AS created_at, \
         note.source_id.uri AS source_uri, vec_distance \
         FROM (SELECT note_id.* AS note, vec_distance FROM \
         (SELECT id, note_id, vector::distance::knn() AS vec_distance \
          FROM exact_vector_probe WHERE embedding <|{limit},COSINE|> $embedding AND {} \
          ORDER BY vec_distance ASC, note_id ASC LIMIT $limit)) \
         ORDER BY vec_distance ASC, id ASC",
        predicates()
    )
}

#[tokio::test]
#[ignore = "opt-in exact skinny storage experiment; no state duplication adopted"]
async fn exact_vector_skinny_storage_diagnostic() {
    use chrono::TimeZone;
    let count = std::env::var("GRAPHRAG_VECTOR_PROBE_RECORDS")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(160);
    assert!(
        (50..=4096).contains(&count),
        "fictional diagnostic population bound"
    );
    let dense = std::env::var_os("GRAPHRAG_VECTOR_PROBE_DENSE").is_some();
    let bodies = if count > 160 {
        vec![1024]
    } else {
        vec![0, 1024, 4096]
    };
    let mut runs = Vec::new();
    for repeats in bodies.iter().copied() {
        let db = init_memory().await.unwrap();
        db.query("DEFINE TABLE exact_vector_probe SCHEMALESS; DEFINE TABLE exact_vector_cbor_probe SCHEMALESS")
            .await
            .unwrap()
            .check()
            .unwrap();
        let mut sources = Vec::new();
        for index in 0..4 {
            let mut source =
                Source::manual().with_content("fictional source payload ".repeat(2048));
            source.uri = Some(format!("fixture://skinny-source-{index}"));
            source.successful_generation = 2;
            source.generation = 3;
            let saved: Option<Source> = db
                .create(("source", format!("source-{index}")))
                .content(source)
                .await
                .unwrap();
            sources.push(saved.unwrap().id.unwrap());
        }
        let mut query_embedding = vec![0.0_f32; 1024];
        query_embedding[0] = 1.0;
        if dense {
            let mut state = 0x4d595df4d0f33173_u64;
            for value in &mut query_embedding {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *value = ((state >> 40) as u32) as f32 / 16777216.0 * 2.0 - 1.0;
            }
        }
        for n in 0..count {
            let mut embedding = query_embedding.clone();
            if dense {
                let mut state = (n as u64 + 1) * 0x9e3779b9;
                for value in &mut embedding {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *value = ((state >> 40) as u32) as f32 / 16777216.0 * 2.0 - 1.0;
                }
            } else {
                embedding[1] = n as f32 / count as f32;
            }
            let row_repeats = if count > 160 {
                match n % 3 {
                    0 => 0,
                    1 => 1024,
                    _ => 4096,
                }
            } else {
                repeats
            };
            let mut note = Note::new(format!(
                "Fictional row {n}: {}",
                "payload ".repeat(row_repeats)
            ))
            .with_title(format!("Fictional row {n}"))
            .with_embedding(embedding.clone());
            note.created_at = Utc
                .with_ymd_and_hms(2030, 1, if n % 7 == 0 { 1 } else { 3 }, 0, 0, 0)
                .unwrap();
            if n % 5 != 0 {
                note.source_id = Some(sources[n % 4].clone());
                note.source_generation = Some(if n % 17 == 0 { 3 } else { 2 });
            }
            let saved: Option<Note> = db
                .create(("note", format!("fixture-{n:04}")))
                .content(note)
                .await
                .unwrap();
            let note = saved.unwrap();
            db.query("CREATE $id SET note_id=$note, embedding=$embedding, created_at=$created, source_id=$source, source_generation=$generation; CREATE $cbor_id SET note_id=$note, embedding_cbor=encoding::cbor::encode($embedding), created_at=$created, source_id=$source, source_generation=$generation")
                .bind((
                    "id",
                    RecordId::new("exact_vector_probe", note.id.as_ref().unwrap().key.clone()),
                ))
                .bind(("cbor_id",RecordId::new("exact_vector_cbor_probe",note.id.as_ref().unwrap().key.clone())))
                .bind(("note", note.id.unwrap()))
                .bind(("embedding", embedding))
                .bind(("created", note.created_at))
                .bind(("source", note.source_id))
                .bind(("generation", note.source_generation))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for (filter, since, source_uri) in [
            ("all", None, None),
            ("recent", Some("2030-01-02T00:00:00Z".to_string()), None),
            (
                "source",
                None,
                Some("fixture://skinny-source-0".to_string()),
            ),
        ] {
            for limit in [5, 50] {
                let baseline = query(
                    &db,
                    old_query(limit),
                    &query_embedding,
                    limit,
                    since.clone(),
                    source_uri.clone(),
                )
                .await;
                for (name, sql) in [
                    ("baseline", old_query(limit)),
                    ("skinny", skinny_query(limit)),
                    ("cbor_paged", skinny_query(limit)),
                ] {
                    let mut samples = Vec::new();
                    let mut exact = true;
                    let mut bits_exact = true;
                    for _ in 0..21 {
                        let start = std::time::Instant::now();
                        let rows = if name == "cbor_paged" {
                            paged_exact_vectors(
                                &db,
                                &query_embedding,
                                limit,
                                256,
                                true,
                                since.clone(),
                                source_uri.clone(),
                            )
                            .await
                        } else {
                            query(
                                &db,
                                sql.clone(),
                                &query_embedding,
                                limit,
                                since.clone(),
                                source_uri.clone(),
                            )
                            .await
                        };
                        samples.push(start.elapsed().as_secs_f64() * 1000.0);
                        exact &= serde_json::to_value(&rows).unwrap()
                            == serde_json::to_value(&baseline).unwrap();
                        bits_exact &= rows
                            .iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>()
                            == baseline
                                .iter()
                                .map(|row| row.vec_distance.map(f32::to_bits))
                                .collect::<Vec<_>>();
                    }
                    let explain = if name == "cbor_paged" {
                        page_query(false, true)
                    } else {
                        sql
                    };
                    let plan: Vec<serde_json::Value> = db
                        .query(format!("{explain} EXPLAIN FULL"))
                        .bind(("page", 256))
                        .bind(("embedding", query_embedding.clone()))
                        .bind(("limit", limit))
                        .bind(("since", since.clone()))
                        .bind(("source_uri", source_uri.clone()))
                        .await
                        .unwrap()
                        .take(0)
                        .unwrap();
                    runs.push(
                        serde_json::json!({"variant":name,"filter":filter,"rows":baseline.len(),
                    "body_payload_bytes":if count>160 {None} else {Some(repeats*"payload ".len())},
                    "mixed_body_sizes":count>160,
                    "limit":limit,"full_rows_exact":exact,"distance_bits_exact":bits_exact,
                    "first_ms":samples[0],"warm_ms":samples[1..],"plan":plan}),
                    );
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records_per_population":count,"populations":bodies.len(),"dense_vectors":dense,
        "dimension":1024,"provider_calls":0,"engine":"surrealdb-core 3.2.4","runs":runs})
    );
}
