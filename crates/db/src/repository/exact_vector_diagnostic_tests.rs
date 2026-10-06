//! Provider-free exactness regressions and opt-in execution experiments.
//!
//! Helpers and schemas in this module are test-only prototypes. They do not
//! change runtime retrieval, migration registration or portable archive data.

use super::*;
use crate::init_memory;

async fn diagnostic_database(label: &str) -> (DbConnection, &'static str) {
    if let Some(root) = std::env::var_os("GRAPHRAG_VECTOR_PROBE_ROCKS_ROOT") {
        #[cfg(feature = "rocksdb")]
        {
            let path = std::path::PathBuf::from(root).join(label);
            assert!(!path.exists(), "diagnostic never reuses a database");
            return (crate::init_persistent(path).await.unwrap(), "rocksdb");
        }
        #[cfg(not(feature = "rocksdb"))]
        panic!("persistent diagnostic requires the rocksdb feature: {root:?}");
    }
    (init_memory().await.unwrap(), "memory")
}

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

#[derive(Serialize)]
struct NativeSqlPhases {
    sdk_await_ms: f64,
    dto_take_ms: f64,
    engine_execution_ms: Option<f64>,
}

async fn query_with_native_stats(
    db: &DbConnection,
    sql: String,
    embedding: &[f32],
    limit: usize,
    since: Option<String>,
    source: Option<String>,
) -> (Vec<SearchResult>, NativeSqlPhases) {
    let start = std::time::Instant::now();
    let mut response = db
        .query(sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", limit))
        .bind(("since", since))
        .bind(("source_uri", source))
        .with_stats()
        .await
        .unwrap();
    let sdk_await_ms = start.elapsed().as_secs_f64() * 1000.0;
    let take = std::time::Instant::now();
    let (stats, rows) = response.take::<Vec<SearchResult>>(0).unwrap();
    let rows = rows.unwrap();
    (
        rows,
        NativeSqlPhases {
            sdk_await_ms,
            dto_take_ms: take.elapsed().as_secs_f64() * 1000.0,
            engine_execution_ms: stats.execution_time.map(|v| v.as_secs_f64() * 1000.0),
        },
    )
}

#[tokio::test]
#[ignore = "opt-in pinned-engine plans and paired timing, not a hardware CI budget"]
async fn exact_vector_plan_and_projection_diagnostic() {
    let (db, _) = diagnostic_database("vector-sql-plans").await;
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
    embedding_fallback: Option<Vec<f64>>,
}

#[derive(Clone)]
struct ExactVectorCandidate {
    id: RecordId,
    distance: f64,
    seq: usize,
}

// The SDK opens optimistic WRITE transactions even for reads and does not
// cancel on Drop. This experimental owner keeps cancellation in an owned task
// when its caller is aborted or panics, including while normal cancellation
// is awaiting the embedded engine. It never commits.
struct VectorSnapshot {
    transaction: Option<surrealdb::method::Transaction<surrealdb::engine::local::Db>>,
    runtime: tokio::runtime::Handle,
    cancelled: Option<tokio::sync::oneshot::Sender<()>>,
}

impl VectorSnapshot {
    async fn begin(db: &DbConnection) -> Self {
        Self::begin_with_receipt(db, None).await
    }

    async fn begin_with_receipt(
        db: &DbConnection,
        cancelled: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Self {
        let client = (**db).clone();
        tokio::spawn(async move {
            Self {
                transaction: Some(client.begin().await.unwrap()),
                runtime: tokio::runtime::Handle::current(),
                cancelled,
            }
        })
        .await
        .unwrap()
    }

    fn transaction(&self) -> &surrealdb::method::Transaction<surrealdb::engine::local::Db> {
        self.transaction.as_ref().unwrap()
    }

    async fn cancel(mut self) {
        let transaction = self.transaction.take().unwrap();
        let cancelled = self.cancelled.take();
        self.runtime
            .spawn(async move {
                transaction.cancel().await.unwrap();
                if let Some(cancelled) = cancelled {
                    let _ = cancelled.send(());
                }
            })
            .await
            .unwrap();
    }
}

impl Drop for VectorSnapshot {
    fn drop(&mut self) {
        if let Some(transaction) = self.transaction.take() {
            let cancelled = self.cancelled.take();
            self.runtime.spawn(async move {
                transaction.cancel().await.unwrap();
                if let Some(cancelled) = cancelled {
                    let _ = cancelled.send(());
                }
            });
        }
    }
}

#[derive(Default, serde::Serialize)]
struct VectorPhases {
    begin_ms: f64,
    scan_ms: f64,
    heap_ms: f64,
    hydrate_ms: f64,
    cancel_ms: f64,
    pages: usize,
    scanned: usize,
    peak_page_rows: usize,
    peak_projected_embedding_bytes: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    page_trace: Vec<VectorPageTrace>,
}

/// Same-call page observations. Engine execution nests inside SDK await;
/// these durations are not added together to infer a causal total.
#[derive(serde::Serialize)]
struct VectorPageTrace {
    page_index: usize,
    query_start_offset_ms: f64,
    sdk_completed_offset_ms: f64,
    dto_completed_offset_ms: f64,
    sdk_await_ms: f64,
    dto_take_ms: f64,
    engine_execution_ms: Option<f64>,
    rows: usize,
    projected_embedding_bytes: usize,
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
            "id, note_id, embedding_cbor, embedding_fallback"
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
    paged_exact_vectors_traced(db, embedding, limit, page, cbor, since, source)
        .await
        .0
}

#[allow(clippy::too_many_arguments)]
async fn paged_exact_vectors_traced(
    db: &DbConnection,
    embedding: &[f32],
    limit: usize,
    page: usize,
    cbor: bool,
    since: Option<String>,
    source: Option<String>,
) -> (Vec<SearchResult>, VectorPhases) {
    paged_exact_vectors_checkpoint(db, embedding, limit, page, cbor, since, source, None).await
}

type FirstPageCheckpoint = (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
);

#[allow(clippy::too_many_arguments)]
async fn paged_exact_vectors_checkpoint(
    db: &DbConnection,
    embedding: &[f32],
    limit: usize,
    page: usize,
    cbor: bool,
    since: Option<String>,
    source: Option<String>,
    checkpoint: Option<FirstPageCheckpoint>,
) -> (Vec<SearchResult>, VectorPhases) {
    paged_exact_vectors_checkpoint_with_trace(
        db,
        embedding,
        limit,
        page,
        cbor,
        since,
        source,
        checkpoint,
        std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_TRACE").is_some(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn paged_exact_vectors_checkpoint_with_trace(
    db: &DbConnection,
    embedding: &[f32],
    limit: usize,
    page: usize,
    cbor: bool,
    since: Option<String>,
    source: Option<String>,
    mut checkpoint: Option<FirstPageCheckpoint>,
    trace_enabled: bool,
) -> (Vec<SearchResult>, VectorPhases) {
    assert!(
        (1..=512).contains(&page),
        "private diagnostic page must be bounded"
    );
    let enforce_fictional_payload_cap =
        std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some();
    let call_start = trace_enabled.then(std::time::Instant::now);
    let mut phases = VectorPhases::default();
    let start = std::time::Instant::now();
    let snapshot = VectorSnapshot::begin(db).await;
    phases.begin_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tx = snapshot.transaction();
    let query_vector: Vec<f64> = embedding.iter().map(|n| f64::from(*n)).collect();
    let magnitude = query_vector.iter().map(|n| n.powi(2)).sum::<f64>().sqrt();
    let mut after: Option<RecordId> = None;
    let mut heap = std::collections::BinaryHeap::with_capacity(limit);
    let mut seq = 0;
    loop {
        let start = std::time::Instant::now();
        let query_start_offset_ms = call_start.map(|clock| clock.elapsed().as_secs_f64() * 1000.0);
        let request = tx
            .query(page_query(after.is_some(), cbor))
            .bind(("range", page_range(after.as_ref(), cbor)))
            .bind(("page", page))
            .bind(("since", since.clone()))
            .bind(("source_uri", source.clone()));
        let (rows, mut trace): (Vec<VectorOnlyRow>, Option<VectorPageTrace>) =
            if let Some(clock) = call_start {
                let mut response = request.with_stats().await.unwrap();
                let sdk_completed_offset_ms = clock.elapsed().as_secs_f64() * 1000.0;
                let (stats, rows) = response.take::<Vec<VectorOnlyRow>>(0).unwrap();
                let rows = rows.unwrap();
                let dto_completed_offset_ms = clock.elapsed().as_secs_f64() * 1000.0;
                let query_start_offset_ms = query_start_offset_ms.unwrap();
                (
                    rows,
                    Some(VectorPageTrace {
                        page_index: phases.pages,
                        query_start_offset_ms,
                        sdk_completed_offset_ms,
                        dto_completed_offset_ms,
                        sdk_await_ms: sdk_completed_offset_ms - query_start_offset_ms,
                        dto_take_ms: dto_completed_offset_ms - sdk_completed_offset_ms,
                        engine_execution_ms: stats
                            .execution_time
                            .map(|duration| duration.as_secs_f64() * 1000.0),
                        rows: 0,
                        projected_embedding_bytes: 0,
                    }),
                )
            } else {
                (request.await.unwrap().take(0).unwrap(), None)
            };
        phases.scan_ms += start.elapsed().as_secs_f64() * 1000.0;
        phases.pages += 1;
        let count = rows.len();
        phases.scanned += count;
        assert!(count <= page, "SQL page row cap");
        phases.peak_page_rows = phases.peak_page_rows.max(count);
        let projected_embedding_bytes: usize = rows
            .iter()
            .map(|row| {
                row.embedding_cbor
                    .as_ref()
                    .map_or(0, |bytes| bytes.as_ref().len())
                    + row.embedding.as_ref().map_or(0, |vector| vector.len() * 8)
                    + row
                        .embedding_fallback
                        .as_ref()
                        .map_or(0, |vector| vector.len() * 8)
            })
            .sum();
        phases.peak_projected_embedding_bytes = phases
            .peak_projected_embedding_bytes
            .max(projected_embedding_bytes);
        if let Some(trace) = trace.as_mut() {
            trace.rows = count;
            trace.projected_embedding_bytes = projected_embedding_bytes;
        }
        if let Some(trace) = trace {
            phases.page_trace.push(trace);
        }
        if enforce_fictional_payload_cap {
            assert!(
                projected_embedding_bytes <= page * 8192,
                "fictional 1024-d vector payload cap"
            );
        }
        if let Some((ready, resume)) = checkpoint.take() {
            ready.send(()).unwrap();
            resume.await.unwrap();
        }
        let start = std::time::Instant::now();
        for row in rows {
            after = Some(row.id.clone());
            let vector = if cbor {
                if let Some(fallback) = row.embedding_fallback {
                    fallback
                } else {
                    let Some(bytes) = row.embedding_cbor else {
                        continue;
                    };
                    ciborium::from_reader::<Vec<f64>, _>(bytes.as_ref()).unwrap()
                }
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
        phases.heap_ms += start.elapsed().as_secs_f64() * 1000.0;
        if count < page {
            break;
        }
    }
    let selected = heap.into_sorted_vec();
    let ids: Vec<_> = selected
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    let start = std::time::Instant::now();
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
    phases.hydrate_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = std::time::Instant::now();
    snapshot.cancel().await;
    phases.cancel_ms = start.elapsed().as_secs_f64() * 1000.0;
    (output, phases)
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
async fn exact_vector_paged_semantics_diagnostic() {
    use chrono::TimeZone;
    use surrealdb::types::{Object, RecordIdKey};
    let (db, _) = diagnostic_database("vector-semantic-contract").await;
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
    db.query("CREATE note:absent_vector SET content='actual absent embedding'; CREATE exact_vector_cbor_probe:absent_vector SET note_id=note:absent_vector,embedding_cbor=NONE,created_at=note:absent_vector.created_at")
        .await.unwrap().check().unwrap();
    let mut precise = positive
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    precise[0] = 1.0 + 0.000_000_001;
    precise[1] = 0.000_000_003;
    db.query("CREATE note:precise_f64 SET content='persisted f64 precision fixture',embedding=$embedding; CREATE exact_vector_cbor_probe:precise_f64 SET note_id=note:precise_f64,embedding_cbor=encoding::cbor::encode(note:precise_f64.embedding),created_at=note:precise_f64.created_at")
        .bind(("embedding",precise.clone())).await.unwrap().check().unwrap();
    let encoded: Vec<surrealdb::types::Bytes> = db
        .query("SELECT VALUE embedding_cbor FROM exact_vector_cbor_probe:precise_f64")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    let decoded: Vec<f64> = ciborium::from_reader(encoded[0].as_ref()).unwrap();
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        precise.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for (name, exceptional) in [
        ("positive_nan", f64::NAN),
        ("negative_nan", -f64::NAN),
        ("payload_nan", f64::from_bits(0x7ff8_0000_0000_0123)),
        ("positive_inf", f64::INFINITY),
        ("negative_inf", f64::NEG_INFINITY),
    ] {
        let mut vector = precise.clone();
        vector[0] = exceptional;
        db.query("CREATE $id SET content='fictional nonfinite legacy vector',embedding=$embedding; LET $stored=$id.embedding; LET $finite=math::min($stored)>math::NEG_INFINITY AND math::max($stored)<math::INFINITY; CREATE $shadow SET note_id=$id,created_at=$id.created_at,embedding_cbor=IF $finite THEN encoding::cbor::encode($stored) ELSE NONE END,embedding_uses_primary=$finite=false,embedding_fallback=IF $finite THEN NONE ELSE $stored END;")
            .bind(("id",RecordId::new("note",name)))
            .bind(("shadow",RecordId::new("exact_vector_cbor_probe",name)))
            .bind(("embedding",vector.clone())).await.unwrap().check().unwrap();
        let fallback: Vec<Vec<f64>> = db
            .query("SELECT VALUE embedding_fallback FROM $shadow")
            .bind(("shadow", RecordId::new("exact_vector_cbor_probe", name)))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(
            fallback[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let mut positive_nan_query = positive.clone();
    positive_nan_query[0] = f32::NAN;
    let mut negative_nan_query = positive.clone();
    negative_nan_query[0] = -f32::NAN;
    let mut infinite_query = positive.clone();
    infinite_query[0] = f32::INFINITY;
    install_single_sql_cbor_probe(&db).await;
    for embedding in [
        &positive,
        &vec![0.0; 1024],
        &vec![1.0, 0.0],
        &vec![],
        &positive_nan_query,
        &negative_nan_query,
        &infinite_query,
    ] {
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
                let computed = query(
                    &db,
                    cbor_single_sql_query(limit),
                    embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await;
                assert_eq!(
                    serde_json::to_value(&old).unwrap(),
                    serde_json::to_value(&computed).unwrap(),
                    "computed exact engine KNN payloads"
                );
                assert_eq!(
                    old.iter()
                        .map(|r| r.vec_distance.map(f32::to_bits))
                        .collect::<Vec<_>>(),
                    computed
                        .iter()
                        .map(|r| r.vec_distance.map(f32::to_bits))
                        .collect::<Vec<_>>(),
                    "computed exact engine KNN distance bits"
                );
                for (page, cbor) in [
                    (1, false),
                    (3, false),
                    (128, false),
                    (256, false),
                    (512, false),
                    (1, true),
                    (3, true),
                    (128, true),
                    (256, true),
                    (512, true),
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

const CBOR_COMPUTED_EMBEDDING: &str = "DEFINE FIELD embedding ON exact_vector_sql_cbor_probe \
    TYPE any COMPUTED \
    IF embedding_uses_primary = true THEN note_id.embedding \
    ELSE IF embedding_cbor IS NONE THEN NONE \
    ELSE encoding::cbor::decode(embedding_cbor) END";

fn cbor_single_sql_query(limit: usize) -> String {
    skinny_query(limit).replace(
        "FROM exact_vector_probe",
        "FROM exact_vector_sql_cbor_probe",
    )
}

async fn install_single_sql_cbor_probe(db: &DbConnection) {
    // Keep the raw paged comparison free of this virtual field: a computed
    // field can be evaluated by the pinned scan pipeline even when omitted
    // from a selective projection. It must not contaminate that comparator.
    db.query("DEFINE TABLE exact_vector_sql_cbor_probe SCHEMALESS; FOR $id IN ((SELECT VALUE id FROM exact_vector_cbor_probe ORDER BY id ASC) ?? []) { CREATE type::record('exact_vector_sql_cbor_probe',record::id($id)) CONTENT object::remove($id.*,'id'); }")
        .await.unwrap().check().unwrap();
    db.query(CBOR_COMPUTED_EMBEDDING)
        .await
        .unwrap()
        .check()
        .unwrap();
}

#[tokio::test]
#[ignore = "opt-in exact skinny storage experiment; no state duplication adopted"]
async fn exact_vector_skinny_storage_diagnostic() {
    exact_vector_skinny_storage_diagnostic_inner("current_thread").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opt-in host-default runtime control; no concurrency default changes"]
async fn exact_vector_skinny_storage_host_default_diagnostic() {
    exact_vector_skinny_storage_diagnostic_inner("host_default").await;
}

async fn exact_vector_skinny_storage_diagnostic_inner(runtime: &str) {
    use chrono::TimeZone;
    use sha2::{Digest, Sha256};
    let worker_threads = tokio::runtime::Handle::current().metrics().num_workers();
    let page_trace_enabled = std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_TRACE").is_some();
    let diagnostic_clock = page_trace_enabled.then(std::time::Instant::now);
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
    let engine_export = std::env::var_os("GRAPHRAG_VECTOR_PROBE_ENGINE_EXPORT").is_some();
    let mut fixture_inputs = Vec::new();
    let mut cross_engine_references = Vec::new();
    for repeats in bodies.iter().copied() {
        let (db, backend) = diagnostic_database(&format!("fictional-{count}-{repeats}")).await;
        db.query("DEFINE TABLE exact_vector_probe SCHEMALESS; DEFINE TABLE exact_vector_cbor_probe SCHEMALESS")
            .await
            .unwrap()
            .check()
            .unwrap();
        let single_sql = std::env::var_os("GRAPHRAG_VECTOR_PROBE_SINGLE_SQL").is_some();
        let page_matrix = std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some();
        assert!(
            !(single_sql && page_matrix),
            "diagnostic modes are mutually exclusive"
        );
        let native_sql_stats_requested = single_sql || page_matrix;
        assert!(
            !engine_export || (page_matrix && dense),
            "cross-engine export requires native keys, dense ties and page-matrix mode"
        );
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
            if page_matrix && n < 72 {
                // A dense equal-distance pool crosses both result limits.
                embedding.clone_from(&query_embedding);
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
            let id = if page_matrix {
                use surrealdb::types::{Object, RecordIdKey};
                match n % 4 {
                    0 => RecordId::new("note", n as i64),
                    1 => RecordId::new("note", format!("fixture-{n:04}")),
                    2 => RecordId::new(
                        "note",
                        RecordIdKey::Uuid(uuid::Uuid::from_u128(n as u128 + 1).into()),
                    ),
                    _ => {
                        let mut key = Object::new();
                        key.insert("label", "fictional");
                        key.insert("ordinal", n as i64);
                        RecordId::new("note", RecordIdKey::Object(key))
                    }
                }
            } else {
                RecordId::new("note", format!("fixture-{n:04}"))
            };
            let saved: Option<Note> = db.create(id).content(note).await.unwrap();
            let note = saved.unwrap();
            if engine_export {
                let saved_embedding = &note.embedding;
                assert_eq!(
                    saved_embedding
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    embedding.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "fixture round-trip changes native float32 vector bits"
                );
                let vector_bits: Vec<u8> = saved_embedding
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect();
                fixture_inputs.push(serde_json::json!({
                    "population_body_repeats": repeats,
                    "ordinal": n,
                    "id": note.id.as_ref().unwrap(),
                    "native_key": &note.id.as_ref().unwrap().key,
                    "native_key_kind": match n % 4 {
                        0 => "integer", 1 => "string", 2 => "uuid", _ => "object"
                    },
                    "title": note.title,
                    "content_sha256": format!("{:x}", Sha256::digest(note.content.as_bytes())),
                    "content_bytes": note.content.len(),
                    "note_type": note.note_type,
                    "tags": note.tags,
                    "created_at": note.created_at,
                    "source_id": note.source_id,
                    "source_generation": note.source_generation,
                    "source_successful_generation": 2,
                    "source_latest_generation": 3,
                    "embedding_dimension": embedding.len(),
                    "embedding_f32_bits_sha256": format!("{:x}", Sha256::digest(vector_bits)),
                    "dense_equal_query_vector": n < 72
                }));
            }
            let materialize = if page_matrix {
                "CREATE $cbor_id SET note_id=$note, embedding_cbor=encoding::cbor::encode($embedding), created_at=$created, source_id=$source, source_generation=$generation"
            } else {
                "CREATE $id SET note_id=$note, embedding=$embedding, created_at=$created, source_id=$source, source_generation=$generation; CREATE $cbor_id SET note_id=$note, embedding_cbor=encoding::cbor::encode($embedding), created_at=$created, source_id=$source, source_generation=$generation"
            };
            db.query(materialize)
                .bind((
                    "id",
                    RecordId::new("exact_vector_probe", note.id.as_ref().unwrap().key.clone()),
                ))
                .bind((
                    "cbor_id",
                    RecordId::new(
                        "exact_vector_cbor_probe",
                        note.id.as_ref().unwrap().key.clone(),
                    ),
                ))
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
        if single_sql {
            install_single_sql_cbor_probe(&db).await;
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
                if engine_export {
                    cross_engine_references.push(serde_json::json!({
                        "population_body_repeats": repeats,
                        "filter": filter,
                        "limit": limit,
                        "ordered_full_rows": serde_json::to_value(&baseline).unwrap(),
                        "ordered_native_distance_f32_bits": baseline
                            .iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>()
                    }));
                }
                let mut variants = if page_matrix {
                    vec![
                        ("baseline", old_query(limit)),
                        ("cbor_paged_128", skinny_query(limit)),
                        ("cbor_paged_256", skinny_query(limit)),
                        ("cbor_paged_512", skinny_query(limit)),
                    ]
                } else {
                    vec![
                        ("baseline", old_query(limit)),
                        ("skinny", skinny_query(limit)),
                        ("cbor_paged", skinny_query(limit)),
                    ]
                };
                if single_sql {
                    variants.push(("cbor_single_sql", cbor_single_sql_query(limit)));
                }
                let variant_count = variants.len();
                let observations_per_variant =
                    if std::env::var_os("GRAPHRAG_VECTOR_PROBE_CORRECTNESS_ONLY").is_some() {
                        1
                    } else {
                        21
                    };
                let order = std::env::var("GRAPHRAG_VECTOR_PROBE_ORDER")
                    .unwrap_or_else(|_| "legacy_first".into());
                let schedule: Vec<usize> = match order.as_str() {
                    "legacy_first" => (0..variant_count)
                        .flat_map(|v| std::iter::repeat_n(v, observations_per_variant))
                        .collect(),
                    "cbor_first" => std::iter::once(variant_count - 1)
                        .chain(0..variant_count - 1)
                        .flat_map(|v| std::iter::repeat_n(v, observations_per_variant))
                        .collect(),
                    "alternating" => (0..observations_per_variant)
                        .flat_map(|round| {
                            (0..variant_count).map(move |v| (round + v) % variant_count)
                        })
                        .collect(),
                    _ => panic!("unsupported private diagnostic schedule"),
                };
                let mut samples: Vec<Vec<f64>> = (0..variant_count).map(|_| Vec::new()).collect();
                let mut sequences: Vec<Vec<usize>> =
                    (0..variant_count).map(|_| Vec::new()).collect();
                let mut phase_samples: Vec<Vec<VectorPhases>> =
                    (0..variant_count).map(|_| Vec::new()).collect();
                let mut sql_samples: Vec<Vec<NativeSqlPhases>> =
                    (0..variant_count).map(|_| Vec::new()).collect();
                let mut call_offsets: Vec<Vec<f64>> =
                    (0..variant_count).map(|_| Vec::new()).collect();
                let category_start = std::time::Instant::now();
                for (sequence, variant) in schedule.into_iter().enumerate() {
                    let (name, sql) = &variants[variant];
                    if let Some(clock) = diagnostic_clock {
                        call_offsets[variant].push(clock.elapsed().as_secs_f64() * 1000.0);
                    }
                    let start = std::time::Instant::now();
                    let page = diagnostic_variant_page(name);
                    let rows = if let Some(page) = page {
                        let (rows, phases) = paged_exact_vectors_traced(
                            &db,
                            &query_embedding,
                            limit,
                            page,
                            true,
                            since.clone(),
                            source_uri.clone(),
                        )
                        .await;
                        phase_samples[variant].push(phases);
                        rows
                    } else if native_sql_stats_requested {
                        let (rows, phases) = query_with_native_stats(
                            &db,
                            sql.clone(),
                            &query_embedding,
                            limit,
                            since.clone(),
                            source_uri.clone(),
                        )
                        .await;
                        sql_samples[variant].push(phases);
                        rows
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
                    samples[variant].push(start.elapsed().as_secs_f64() * 1000.0);
                    sequences[variant].push(sequence);
                    assert_eq!(
                        serde_json::to_value(&rows).unwrap(),
                        serde_json::to_value(&baseline).unwrap()
                    );
                    assert_eq!(
                        rows.iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>(),
                        baseline
                            .iter()
                            .map(|row| row.vec_distance.map(f32::to_bits))
                            .collect::<Vec<_>>()
                    );
                }
                let category_elapsed_ms = category_start.elapsed().as_secs_f64() * 1000.0;
                for (variant, (name, sql)) in variants.into_iter().enumerate() {
                    let page = diagnostic_variant_page(name);
                    let explain = if page.is_some() {
                        page_query(false, true)
                    } else {
                        sql
                    };
                    let plan: Vec<serde_json::Value> = db
                        .query(format!("{explain} EXPLAIN FULL"))
                        .bind(("page", page.unwrap_or(256)))
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
                    "backend":backend,"order":order,"variant_sequence":sequences[variant],
                    "category_elapsed_ms":category_elapsed_ms,"phase_samples":phase_samples[variant],
                    "native_sql_phase_samples":sql_samples[variant],
                    "call_offsets_ms":if page_trace_enabled {Some(&call_offsets[variant])} else {None},
                    "native_sql_stats_requested":native_sql_stats_requested && page.is_none(),
                    "paged_rows_per_query":page,
                    "body_payload_bytes":if count>160 {None} else {Some(repeats*"payload ".len())},
                    "mixed_body_sizes":count>160,
                    "limit":limit,"full_rows_exact":true,"distance_bits_exact":true,
                    "first_ms":samples[variant][0],"warm_ms":samples[variant][1..],"plan":plan}),
                    );
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records_per_population":count,"populations":bodies.len(),"dense_vectors":dense,
        "dimension":1024,"provider_calls":0,"engine":"surrealdb-core 3.2.4",
        "tokio_runtime":runtime,"tokio_worker_threads":worker_threads,
        "query_concurrency":1,"page_trace_enabled":page_trace_enabled,
        "page_trace_offset_kind":"monotonic offsets within each call; call_offsets_ms uses one diagnostic clock",
        "first_sample_kind":"first measured after unmeasured legacy reference query",
        "native_sql_stats_requested":std::env::var_os("GRAPHRAG_VECTOR_PROBE_SINGLE_SQL").is_some()
            || std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some(),
        "page_matrix":std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some(),
        "observations_per_variant":if std::env::var_os("GRAPHRAG_VECTOR_PROBE_CORRECTNESS_ONLY").is_some() {1} else {21},
        "native_key_mix":std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some(),
        "dense_tie_rows":if std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some() {count.min(72)} else {0},
        "projected_embedding_payload_cap_bytes_per_row":if std::env::var_os("GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX").is_some() {Some(8192)} else {None},
        "projected_embedding_payload_cap_qualification":"Fictional finite float32-origin 1024-d vectors; excludes metadata and SDK/engine allocations",
        "engine_comparison_export_version": if engine_export {Some(1)} else {None},
        "fixture_inputs": fixture_inputs,
        "cross_engine_references": cross_engine_references,
        "runs":runs})
    );
}

fn diagnostic_variant_page(name: &str) -> Option<usize> {
    match name {
        "cbor_paged_128" => Some(128),
        "cbor_paged" | "cbor_paged_256" => Some(256),
        "cbor_paged_512" => Some(512),
        _ => None,
    }
}

#[tokio::test]
async fn exact_vector_page_trace_preserves_results_and_offset_arithmetic() {
    // Pass tracing explicitly instead of changing process-wide environment
    // variables while unrelated CI tests run.
    let db = init_memory().await.unwrap();
    db.query("DEFINE TABLE exact_vector_cbor_probe SCHEMALESS")
        .await
        .unwrap()
        .check()
        .unwrap();
    let query_vector = vec![1.0_f32, 0.0, 0.0, 0.0];
    for ordinal in 0..9 {
        let note = Note::new(format!("fictional page trace {ordinal}")).with_embedding(vec![
            1.0,
            ordinal as f32 / 8.0,
            0.0,
            0.0,
        ]);
        db.query("CREATE $id CONTENT $note; CREATE $shadow SET note_id=$id,embedding_cbor=encoding::cbor::encode($note.embedding),created_at=$note.created_at")
            .bind(("id", RecordId::new("note", format!("page-trace-{ordinal}"))))
            .bind(("shadow", RecordId::new("exact_vector_cbor_probe", format!("page-trace-{ordinal}"))))
            .bind(("note", note)).await.unwrap().check().unwrap();
    }
    let reference = query(&db, old_query(5), &query_vector, 5, None, None).await;
    for cbor in [false, true] {
        let (ordinary, ordinary_phases) = paged_exact_vectors_checkpoint_with_trace(
            &db,
            &query_vector,
            5,
            3,
            cbor,
            None,
            None,
            None,
            false,
        )
        .await;
        let (traced, phases) = paged_exact_vectors_checkpoint_with_trace(
            &db,
            &query_vector,
            5,
            3,
            cbor,
            None,
            None,
            None,
            true,
        )
        .await;
        assert_ordered_vector_rows(&reference, &ordinary);
        assert_ordered_vector_rows(&reference, &traced);
        assert!(ordinary_phases.page_trace.is_empty());
        assert!(serde_json::to_value(ordinary_phases)
            .unwrap()
            .get("page_trace")
            .is_none());
        assert_eq!(phases.pages, 4, "includes the final empty page");
        assert_eq!(phases.page_trace.len(), phases.pages);
        assert_eq!(
            phases
                .page_trace
                .iter()
                .map(|page| page.rows)
                .sum::<usize>(),
            phases.scanned
        );
        assert_eq!(phases.scanned, 9);
        assert_eq!(
            phases.page_trace.iter().map(|page| page.rows).max(),
            Some(phases.peak_page_rows)
        );
        assert_eq!(
            phases
                .page_trace
                .iter()
                .map(|page| page.projected_embedding_bytes)
                .max(),
            Some(phases.peak_projected_embedding_bytes)
        );
        let mut previous_end = 0.0;
        for (index, page) in phases.page_trace.iter().enumerate() {
            assert_eq!(page.page_index, index);
            assert!(page.query_start_offset_ms >= previous_end);
            assert!(page.sdk_completed_offset_ms >= page.query_start_offset_ms);
            assert!(page.dto_completed_offset_ms >= page.sdk_completed_offset_ms);
            assert_eq!(
                page.sdk_await_ms,
                page.sdk_completed_offset_ms - page.query_start_offset_ms
            );
            assert_eq!(
                page.dto_take_ms,
                page.dto_completed_offset_ms - page.sdk_completed_offset_ms
            );
            assert!(page
                .engine_execution_ms
                .is_some_and(|elapsed| elapsed.is_finite() && elapsed >= 0.0));
            assert!(page.rows <= 3);
            assert_eq!(page.projected_embedding_bytes == 0, page.rows == 0);
            previous_end = page.dto_completed_offset_ms;
        }
    }
}

#[tokio::test]
async fn exact_vector_snapshot_cancels_on_panic_abort_and_normal_exit() {
    let (db, _) = diagnostic_database("snapshot-cancellation").await;
    db.query("DEFINE TABLE vector_guard_probe SCHEMALESS; CREATE vector_guard_probe:one SET value='persisted'")
        .await.unwrap().check().unwrap();
    for fault in ["normal", "panic", "abort"] {
        let (cancelled, receipt) = tokio::sync::oneshot::channel();
        let (ready, entered) = tokio::sync::oneshot::channel();
        let client = db.clone();
        let task = tokio::spawn(async move {
            let snapshot = VectorSnapshot::begin_with_receipt(&client, Some(cancelled)).await;
            snapshot
                .transaction()
                .query("UPDATE vector_guard_probe:one SET value='uncommitted'")
                .await
                .unwrap()
                .check()
                .unwrap();
            ready.send(()).unwrap();
            match fault {
                "normal" => snapshot.cancel().await,
                "panic" => panic!("intentional private transaction fault"),
                "abort" => std::future::pending::<()>().await,
                _ => unreachable!(),
            }
        });
        entered.await.unwrap();
        if fault == "abort" {
            task.abort();
        }
        let result = task.await;
        assert_eq!(result.is_ok(), fault == "normal");
        tokio::time::timeout(std::time::Duration::from_secs(5), receipt)
            .await
            .unwrap()
            .unwrap();
        let values: Vec<String> = db
            .query("SELECT VALUE value FROM vector_guard_probe:one")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(values, ["persisted"], "SDK-acknowledged rollback: {fault}");
    }
    // A separate transaction still writes/commits normally after all faults.
    db.query("BEGIN TRANSACTION; UPDATE vector_guard_probe:one SET value='after-faults'; COMMIT TRANSACTION")
        .await.unwrap().check().unwrap();
    let values: Vec<String> = db
        .query("SELECT VALUE value FROM vector_guard_probe:one")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    assert_eq!(values, ["after-faults"]);
}

#[tokio::test]
async fn exact_vector_snapshot_preserves_generation_and_payload_during_promotion() {
    let (db, _) = diagnostic_database("snapshot-promotion").await;
    db.query("DEFINE TABLE exact_vector_cbor_probe SCHEMALESS")
        .await
        .unwrap()
        .check()
        .unwrap();
    let mut source = Source::manual();
    source.uri = Some("fixture://snapshot-promotion".into());
    source.successful_generation = 1;
    source.generation = 2;
    let source: Option<Source> = db
        .create(("source", "fixture-promotion"))
        .content(source)
        .await
        .unwrap();
    let source_id = source.unwrap().id.unwrap();
    let mut embedding = vec![0.0; 1024];
    embedding[0] = 1.0;
    for n in 0..8 {
        let mut note =
            Note::new(format!("original snapshot row {n}")).with_embedding(embedding.clone());
        note.source_id = Some(source_id.clone());
        note.source_generation = Some(if n < 4 { 1 } else { 2 });
        db.query("CREATE $id CONTENT $note; CREATE $shadow SET note_id=$id,embedding_cbor=encoding::cbor::encode($note.embedding),created_at=$note.created_at,source_id=$note.source_id,source_generation=$note.source_generation")
            .bind(("id", RecordId::new("note", format!("promotion-{n}"))))
            .bind(("shadow", RecordId::new("exact_vector_cbor_probe", format!("promotion-{n}"))))
            .bind(("note", note)).await.unwrap().check().unwrap();
    }
    let baseline = query(&db, old_query(50), &embedding, 50, None, None).await;
    assert_eq!(baseline.len(), 4);
    let (ready, entered) = tokio::sync::oneshot::channel();
    let (resume, resumed) = tokio::sync::oneshot::channel();
    let reader = db.clone();
    let query_vector = embedding.clone();
    let task = tokio::spawn(async move {
        paged_exact_vectors_checkpoint(
            &reader,
            &query_vector,
            50,
            1,
            true,
            None,
            None,
            Some((ready, resumed)),
        )
        .await
        .0
    });
    entered.await.unwrap();
    db.query("BEGIN TRANSACTION; UPDATE $source SET successful_generation=2; UPDATE note SET content='newly committed payload' WHERE source_id=$source; COMMIT TRANSACTION")
        .bind(("source",source_id)).await.unwrap().check().unwrap();
    resume.send(()).unwrap();
    let snapshot = task.await.unwrap();
    assert_eq!(
        serde_json::to_value(&snapshot).unwrap(),
        serde_json::to_value(&baseline).unwrap()
    );
    assert_eq!(
        snapshot
            .iter()
            .map(|r| r.vec_distance.map(f32::to_bits))
            .collect::<Vec<_>>(),
        baseline
            .iter()
            .map(|r| r.vec_distance.map(f32::to_bits))
            .collect::<Vec<_>>()
    );
    let committed = query(&db, old_query(50), &embedding, 50, None, None).await;
    assert_eq!(committed.len(), 4);
    assert!(committed
        .iter()
        .all(|row| row.content == "newly committed payload"));
    assert!(committed
        .iter()
        .all(|row| !snapshot.iter().any(|old| old.id == row.id)));
}

#[tokio::test]
#[ignore = "opt-in SDK begin/cancel sham; no retrieval latency claim"]
async fn exact_vector_snapshot_route_lifecycle_diagnostic() {
    let (db, backend) = diagnostic_database("snapshot-route-sham").await;
    let mut runs = Vec::new();
    for query in [false, true] {
        let mut samples = Vec::new();
        for _ in 0..128 {
            let start = std::time::Instant::now();
            let snapshot = VectorSnapshot::begin(&db).await;
            let begin_ms = start.elapsed().as_secs_f64() * 1000.0;
            let query_start = std::time::Instant::now();
            if query {
                snapshot
                    .transaction()
                    .query("RETURN NONE")
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;
            let cancel_start = std::time::Instant::now();
            snapshot.cancel().await;
            samples.push(
                serde_json::json!({"begin_ms":begin_ms,"noop_query_ms":query_ms,
                "cancel_ms":cancel_start.elapsed().as_secs_f64()*1000.0,
                "total_ms":start.elapsed().as_secs_f64()*1000.0}),
            );
        }
        runs.push(serde_json::json!({"empty_query":query,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::json!({"fictional":true,"provider_calls":0,"backend":backend,"sham":true,"runs":runs})
    );
}

const PROJECTION_PROBE_SCHEMA: &str = include_str!("exact_vector_projection_poc.surql");

#[derive(SurrealValue)]
struct ProjectionPrimary {
    id: RecordId,
    embedding: Option<Vec<f64>>,
}

#[derive(SurrealValue)]
struct ProjectionProbe {
    id: RecordId,
    record_id: RecordId,
    embedding_cbor: Option<surrealdb::types::Bytes>,
    embedding_uses_primary: bool,
    embedding: Option<Vec<f64>>,
}

async fn assert_projection_probe(db: &DbConnection) {
    for (primary, derived, embedding) in [
        ("note", "exact_note_vector_probe", "embedding"),
        ("message", "exact_message_vector_probe", "embedding"),
        (
            "conversation",
            "exact_conversation_vector_probe",
            "summary_embedding",
        ),
    ] {
        let rows: Vec<ProjectionPrimary> = db
            .query(format!(
                "SELECT id,{embedding} AS embedding FROM {primary} ORDER BY id ASC"
            ))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let projection:Vec<ProjectionProbe> = db.query(format!("SELECT id,record_id,embedding_cbor,embedding_uses_primary,embedding FROM {derived} ORDER BY id ASC"))
            .await.unwrap().take(0).unwrap();
        assert_eq!(
            rows.len(),
            projection.len(),
            "allrows including missing vectors: {primary}"
        );
        for (row, indexed) in rows.iter().zip(projection) {
            assert_eq!(row.id, indexed.record_id);
            assert_eq!(row.id.key, indexed.id.key, "native typed key mirror");
            let decoded = indexed.embedding;
            if let Some(bytes) = indexed.embedding_cbor {
                let decoded_bytes = ciborium::from_reader::<Vec<f64>, _>(bytes.as_ref()).unwrap();
                assert!(!indexed.embedding_uses_primary);
                assert_eq!(
                    decoded_bytes
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    decoded
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                row.embedding
                    .as_ref()
                    .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
                decoded
                    .as_ref()
                    .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            );
        }
    }
}

async fn projection_ranked_query<T: SurrealValue>(
    db: &DbConnection,
    sql: String,
    embedding: &[f32],
    limit: usize,
    since: Option<String>,
    source: Option<String>,
) -> Vec<T> {
    db.query(sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", limit))
        .bind(("since", since))
        .bind(("source_uri", source))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

fn assert_ordered_vector_rows<T: Serialize + FusionRecord>(old: &[T], new: &[T]) {
    assert_eq!(
        serde_json::to_value(old).unwrap(),
        serde_json::to_value(new).unwrap()
    );
    assert_eq!(
        old.iter()
            .map(|r| r.vector_distance().map(f32::to_bits))
            .collect::<Vec<_>>(),
        new.iter()
            .map(|r| r.vector_distance().map(f32::to_bits))
            .collect::<Vec<_>>()
    );
}

async fn assert_scope_projection_ranking(db: &DbConnection) {
    let repository = Repository::new(db.clone());
    let mut positive = vec![0.0_f32; 1024];
    positive[0] = 1.0;
    let mut nan = positive.clone();
    nan[0] = -f32::NAN;
    for embedding in [positive, vec![0.0; 1024], vec![1.0, 0.0], Vec::new(), nan] {
        for (since, source) in [
            (None, None),
            (Some("2030-01-02T00:00:00Z".to_string()), None),
            (None, Some("fixture://edited".to_string())),
            (None, Some("fixture://absent".to_string())),
        ] {
            let timestamp = since.as_ref().map(|v| {
                chrono::DateTime::parse_from_rfc3339(v)
                    .unwrap()
                    .with_timezone(&Utc)
            });
            for limit in [1, 3, 50] {
                let note_sql = skinny_query(limit)
                    .replace("note_id", "record_id")
                    .replace("FROM exact_vector_probe", "FROM exact_note_vector_probe");
                let old = repository
                    .vector_search_notes(embedding.clone(), limit, timestamp, source.clone())
                    .await
                    .unwrap();
                if limit == 3 && since.is_none() && source.is_none() {
                    let (traced, phases) = query_with_native_stats(
                        db,
                        note_sql.clone(),
                        &embedding,
                        limit,
                        None,
                        None,
                    )
                    .await;
                    assert_ordered_vector_rows(&old, &traced);
                    assert!(phases.engine_execution_ms.is_some());
                }
                let new: Vec<SearchResult> = projection_ranked_query(
                    db,
                    note_sql,
                    &embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await;
                assert_ordered_vector_rows(&old, &new);
                let message_sql = format!(
                    "SELECT row.id AS id,row.conversation_id AS conversation_id,row.conversation_uuid AS conversation_uuid,\
                     row.message_index AS message_index,row.role AS role,row.content AS content,row.created_at AS created_at,\
                     row.conversation_id.source_uri AS source_uri,vec_distance FROM \
                     (SELECT record_id.* AS row,vec_distance FROM \
                      (SELECT id,record_id,vector::distance::knn() AS vec_distance FROM exact_message_vector_probe \
                       WHERE embedding <|{limit},COSINE|> $embedding \
                         AND ($since = NONE OR (created_at != NONE AND created_at >= <datetime>$since)) \
                         AND ($source_uri = NONE OR conversation_id.source_uri = $source_uri) \
                       ORDER BY vec_distance ASC,id ASC LIMIT $limit)) ORDER BY vec_distance ASC,id ASC"
                );
                let old = repository
                    .vector_search_messages(embedding.clone(), limit, timestamp, source.clone())
                    .await
                    .unwrap();
                let new: Vec<MessageSearchResult> = projection_ranked_query(
                    db,
                    message_sql,
                    &embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await;
                assert_ordered_vector_rows(&old, &new);
                let conversation_sql = format!(
                    "SELECT row.id AS id,row.uuid AS uuid,row.title AS title,row.summary AS summary,\
                     row.source_uri AS source_uri,row.updated_at AS updated_at,vec_distance FROM \
                     (SELECT record_id.* AS row,vec_distance FROM \
                      (SELECT id,record_id,vector::distance::knn() AS vec_distance FROM exact_conversation_vector_probe \
                       WHERE embedding <|{limit},COSINE|> $embedding \
                         AND ($since = NONE OR updated_at >= <datetime>$since) \
                         AND ($source_uri = NONE OR source_uri = $source_uri) \
                       ORDER BY vec_distance ASC,id ASC LIMIT $limit)) ORDER BY vec_distance ASC,id ASC"
                );
                let old = repository
                    .vector_search_conversation_summaries(
                        embedding.clone(),
                        limit,
                        timestamp,
                        source.clone(),
                    )
                    .await
                    .unwrap();
                let new: Vec<ConversationSearchResult> = projection_ranked_query(
                    db,
                    conversation_sql,
                    &embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await;
                assert_ordered_vector_rows(&old, &new);
            }
        }
    }
}

#[tokio::test]
async fn exact_vector_projection_events_backfill_restore_and_fault_atomicity() {
    let (db, _) = diagnostic_database("projection-events").await;
    let mut vector = vec![0.0_f64; 1024];
    vector[0] = 1.0;
    vector[1] = 0.000_000_003;
    for id in [
        RecordId::new("note", 7i64),
        RecordId::new("note", "string-fixture"),
        RecordId::new(
            "note",
            surrealdb::types::RecordIdKey::Uuid(
                uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000007")
                    .unwrap()
                    .into(),
            ),
        ),
    ] {
        db.query("CREATE $id SET content='fictional primary',embedding=$vector")
            .bind(("id", id))
            .bind(("vector", vector.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    db.query("CREATE note:missing_vector SET content='fixture absent vector'; CREATE conversation:fixture SET uuid='fixture',title='Fictional conversation',summary='Fictional summary',created_at=d'2030-01-01T00:00:00Z',updated_at=d'2030-01-03T00:00:00Z',source_uri='fixture://chat',summary_embedding=$vector; CREATE message:fixture SET message_key='fixture',conversation_id=conversation:fixture,conversation_uuid='fixture',message_index=0,role='user',content='Fictional message',embedding=$vector; CREATE conversation:missing_vector SET uuid='missing',created_at=d'2030-01-01T00:00:00Z',updated_at=d'2030-01-03T00:00:00Z'; CREATE message:missing_vector SET message_key='missing',conversation_id=conversation:missing_vector,conversation_uuid='missing',message_index=0,role='user',content='Fictional missing-vector message'")
        .bind(("vector",vector.clone())).await.unwrap().check().unwrap();
    for (index, key) in [
        surrealdb::types::RecordIdKey::Number(7),
        surrealdb::types::RecordIdKey::Uuid(
            uuid::Uuid::parse_str("00000000-0000-4000-8000-000000000007")
                .unwrap()
                .into(),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        db.query("CREATE $conversation SET uuid=$uuid,title='Native tie conversation',summary='Fictional native tie summary',summary_embedding=$vector,source_uri='fixture://edited',created_at=d'2030-01-01T00:00:00Z',updated_at=d'2030-01-03T00:00:00Z'; CREATE $message SET message_key=$uuid,conversation_id=$conversation,conversation_uuid=$uuid,message_index=0,role='user',content='Fictional native tie message',created_at=d'2030-01-03T00:00:00Z',embedding=$vector")
            .bind(("conversation",RecordId::new("conversation",key.clone())))
            .bind(("message",RecordId::new("message",key)))
            .bind(("uuid",format!("native-tie-{index}")))
            .bind(("vector",vector.clone())).await.unwrap().check().unwrap();
    }
    let repository = Repository::new(db.clone());
    let mut original = Vec::new();
    for table in PORTABLE_TABLES {
        original.push((
            *table,
            repository
                .portable_records_page(table, 0, 100)
                .await
                .unwrap(),
        ));
    }
    db.query(PROJECTION_PROBE_SCHEMA)
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_projection_probe(&db).await;
    assert_scope_projection_ranking(&db).await;
    for (table, records) in &original {
        assert_eq!(
            repository
                .portable_records_page(table, 0, 100)
                .await
                .unwrap(),
            *records,
            "backfill leaves every primary logical row unchanged"
        );
    }
    // Ordinary SDK/SQL mutation, bulk mutation, and deletion all update the same
    // synchronous projection; no caller-specific dual writes are required.
    let created = repository
        .create_note(Note::new("new post-projection note").with_embedding(vec![1.0; 1024]))
        .await
        .unwrap();
    db.query("UPDATE note SET embedding=$vector WHERE content='fictional primary'; UPDATE message:fixture SET embedding=$vector,created_at=d'2030-01-04T00:00:00Z'; UPDATE conversation:fixture SET summary_embedding=$vector,updated_at=d'2030-01-05T00:00:00Z',source_uri='fixture://edited'; DELETE message:missing_vector")
        .bind(("vector",vec![0.5_f64;1024])).await.unwrap().check().unwrap();
    assert_projection_probe(&db).await;
    assert_scope_projection_ranking(&db).await;
    let created_id = created.id.unwrap();
    db.query("DELETE $id")
        .bind(("id", created_id))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_projection_probe(&db).await;
    // A failed derived write fails the primary mutation in the same actual
    // storage transaction. Verify full originals, including all vector bits.
    db.query("DEFINE EVENT fail_projection ON exact_note_vector_probe WHEN true THEN {THROW 'fictional projection failure';}")
        .await.unwrap().check().unwrap();
    let before = repository
        .portable_records_page("note", 0, 100)
        .await
        .unwrap();
    let failed = repository
        .create_note(
            Note::new("must not survive failed projection").with_embedding(vec![1.0; 1024]),
        )
        .await;
    assert!(failed.is_err());
    assert!(db
        .query("UPDATE note SET content='must roll back'")
        .await
        .unwrap()
        .check()
        .is_err());
    assert!(db.query("DELETE note").await.unwrap().check().is_err());
    assert_eq!(
        repository
            .portable_records_page("note", 0, 100)
            .await
            .unwrap(),
        before
    );
    db.query("REMOVE EVENT fail_projection ON exact_note_vector_probe")
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_projection_probe(&db).await;
    for (primary, derived, update, create) in [
        (
            "message",
            "exact_message_vector_probe",
            "content='must roll back'",
            "CREATE message:failed_projection SET message_key='failed-projection',conversation_id=conversation:fixture,conversation_uuid='fixture',message_index=2,role='user',content='must roll back'",
        ),
        (
            "conversation",
            "exact_conversation_vector_probe",
            "summary='must roll back'",
            "CREATE conversation:failed_projection SET uuid='failed-projection',created_at=d'2030-01-01T00:00:00Z',updated_at=d'2030-01-01T00:00:00Z',summary='must roll back'",
        ),
    ] {
        let before = repository.portable_records_page(primary, 0, 100).await.unwrap();
        db.query(format!("DEFINE EVENT fail_projection ON {derived} WHEN true THEN {{THROW 'fictional projection failure';}}"))
            .await.unwrap().check().unwrap();
        for sql in [create.to_string(),format!("UPDATE {primary} SET {update}"),format!("DELETE {primary}")] {
            assert!(db.query(sql).await.unwrap().check().is_err(), "atomic {primary} projection failure");
            assert_eq!(repository.portable_records_page(primary,0,100).await.unwrap(), before);
        }
        db.query(format!("REMOVE EVENT fail_projection ON {derived}"))
            .await.unwrap().check().unwrap();
    }
    assert_projection_probe(&db).await;
    // Projection tables are excluded by the existing logical archive allowlist;
    // restore replays canonical rows through the same events without duplicate
    // primaries or indexed rows. Compare full exported rows after bulk restore.
    let (restored_db, _) = diagnostic_database("projection-restore").await;
    let mut schema = restored_db.query(PROJECTION_PROBE_SCHEMA).await.unwrap();
    let errors = schema.take_errors();
    assert!(
        errors.is_empty(),
        "empty projection schema errors: {errors:?}"
    );
    let restored = Repository::new(restored_db.clone());
    assert!(!PORTABLE_TABLES.iter().any(|table| [
        "exact_note_vector_probe",
        "exact_message_vector_probe",
        "exact_conversation_vector_probe"
    ]
    .contains(table)));
    for table in PORTABLE_TABLES {
        let records = repository
            .portable_records_page(table, 0, 100)
            .await
            .unwrap();
        for record in &records {
            restored
                .restore_portable_record(table, record.clone())
                .await
                .unwrap();
        }
        assert_eq!(
            restored.portable_records_page(table, 0, 100).await.unwrap(),
            records
        );
    }
    assert_projection_probe(&restored_db).await;
    assert_scope_projection_ranking(&restored_db).await;
    // Deterministic explicit rebuild repairs removed derived rows without
    // changing a primary row or adding duplicate materialized identities.
    let before = restored
        .portable_records_page("note", 0, 100)
        .await
        .unwrap();
    restored_db.query("DELETE exact_note_vector_probe; FOR $id IN (SELECT VALUE id FROM note ORDER BY id ASC) {fn::materialize_exact_note_vector_probe($id.*);}")
        .await.unwrap().check().unwrap();
    assert_projection_probe(&restored_db).await;
    assert_eq!(
        restored
            .portable_records_page("note", 0, 100)
            .await
            .unwrap(),
        before
    );
    // Rare historical nonfinite vectors remain on the canonical row; the
    // projection stores only a marker and reads their exact bits in its SQL
    // snapshot rather than letting CBOR canonicalize NaN payloads/signs.
    for (name, value) in [
        ("positive_nan", f64::NAN),
        ("negative_nan", -f64::NAN),
        ("payload_nan", f64::from_bits(0x7ff8_0000_0000_0123)),
        ("positive_infinity", f64::INFINITY),
        ("negative_infinity", f64::NEG_INFINITY),
    ] {
        let mut vector = vec![1.0_f64; 1024];
        vector[0] = value;
        restored_db
            .query("CREATE $id SET content='fictional historical nonfinite row',embedding=$vector")
            .bind(("id", RecordId::new("note", name)))
            .bind(("vector", vector))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    assert_projection_probe(&restored_db).await;
    assert_scope_projection_ranking(&restored_db).await;
}

#[tokio::test]
async fn exact_vector_single_sql_projection_preserves_snapshot_during_promotion() {
    let (db, _) = diagnostic_database("single-sql-promotion").await;
    let mut source = Source::manual();
    source.uri = Some("fixture://single-sql-promotion".into());
    source.generation = 2;
    source.successful_generation = 1;
    let source: Option<Source> = db
        .create(("source", "single-sql-promotion"))
        .content(source)
        .await
        .unwrap();
    let source_id = source.unwrap().id.unwrap();
    let embedding = vec![1.0_f32; 1024];
    for generation in [1, 2] {
        for index in 0..4 {
            let mut note = Note::new("epoch:1").with_embedding(embedding.clone());
            note.source_id = Some(source_id.clone());
            note.source_generation = Some(generation);
            let _: Option<Note> = db
                .create(("note", format!("g{generation}-{index}")))
                .content(note)
                .await
                .unwrap();
        }
    }
    db.query(PROJECTION_PROBE_SCHEMA)
        .await
        .unwrap()
        .check()
        .unwrap();
    let commits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let committed = commits.clone();
    let writer = db.clone();
    let task = tokio::spawn(async move {
        for round in 0..32 {
            let generation = if round % 2 == 0 { 2 } else { 1 };
            writer.query("BEGIN TRANSACTION; UPDATE $source SET successful_generation=$generation; UPDATE note SET content=$content WHERE source_id=$source; COMMIT TRANSACTION")
                .bind(("source",source_id.clone())).bind(("generation",generation))
                .bind(("content",format!("epoch:{generation}"))).await.unwrap().check().unwrap();
            committed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    });
    let sql = skinny_query(50)
        .replace("note_id", "record_id")
        .replace("FROM exact_vector_probe", "FROM exact_note_vector_probe");
    let mut concurrent = 0;
    for _ in 0..64 {
        let before = commits.load(std::sync::atomic::Ordering::SeqCst);
        let rows: Vec<SearchResult> =
            projection_ranked_query(&db, sql.clone(), &embedding, 50, None, None).await;
        let after = commits.load(std::sync::atomic::Ordering::SeqCst);
        concurrent += usize::from(after > before);
        assert_eq!(rows.len(), 4);
        let content = &rows[0].content;
        let generation = if content == "epoch:1" {
            "g1-"
        } else {
            assert_eq!(content, "epoch:2");
            "g2-"
        };
        assert!(rows
            .iter()
            .all(|r| r.content == *content && record_id_to_string(&r.id).contains(generation)));
    }
    task.await.unwrap();
    assert_eq!(commits.load(std::sync::atomic::Ordering::SeqCst), 32);
    assert!(
        concurrent > 0,
        "fixture must actually overlap a committed source/payload promotion with a read"
    );
}
