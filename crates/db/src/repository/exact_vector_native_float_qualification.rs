//! Test-only qualification of the historical thin native-Float mechanism.
//! No production query, schema, event, migration, restore or default changes.
//! The mirrors are populated once in bounded pages while this fixture is idle.

use super::*;
use chrono::TimeZone;
use sha2::{Digest, Sha256};
use std::io::Write;
use surrealdb::types::{Array, Datetime, Number, Object, RecordIdKey, RecordIdKeyRange, Value};

#[derive(Clone, Copy, Debug)]
enum Scope {
    Note,
    Message,
    Conversation,
}

const SCOPES: [Scope; 3] = [Scope::Note, Scope::Message, Scope::Conversation];

impl Scope {
    fn table(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Message => "message",
            Self::Conversation => "conversation",
        }
    }

    fn mirror(self) -> &'static str {
        match self {
            Self::Note => "native_float_note_probe",
            Self::Message => "native_float_message_probe",
            Self::Conversation => "native_float_conversation_probe",
        }
    }

    fn vector(self) -> &'static str {
        match self {
            Self::Conversation => "summary_embedding",
            _ => "embedding",
        }
    }

    fn fields(self) -> &'static str {
        match self {
            Self::Note => "id,title,content,note_type,tags,created_at,source_id.uri AS source_uri",
            Self::Message => "id,conversation_id,conversation_uuid,message_index,role,content,created_at,conversation_id.source_uri AS source_uri",
            Self::Conversation => "id,uuid,title,summary,source_uri,updated_at",
        }
    }

    fn hydrated_fields(self) -> &'static str {
        match self {
            Self::Note => "row.id AS id,row.title AS title,row.content AS content,row.note_type AS note_type,row.tags AS tags,row.created_at AS created_at,row.source_id.uri AS source_uri",
            Self::Message => "row.id AS id,row.conversation_id AS conversation_id,row.conversation_uuid AS conversation_uuid,row.message_index AS message_index,row.role AS role,row.content AS content,row.created_at AS created_at,row.conversation_id.source_uri AS source_uri",
            Self::Conversation => "row.id AS id,row.uuid AS uuid,row.title AS title,row.summary AS summary,row.source_uri AS source_uri,row.updated_at AS updated_at",
        }
    }

    fn predicate(self) -> &'static str {
        match self {
            Self::Note => "($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)",
            Self::Message => "($since = NONE OR (created_at != NONE AND created_at >= <datetime>$since)) AND ($source_uri = NONE OR conversation_id.source_uri = $source_uri)",
            Self::Conversation => "($since = NONE OR updated_at >= <datetime>$since) AND ($source_uri = NONE OR source_uri = $source_uri)",
        }
    }

    fn baseline_sql(self, limit: usize) -> String {
        format!("SELECT {},vector::distance::knn() AS vec_distance FROM {} WHERE {} <|{limit},COSINE|> $embedding AND {} ORDER BY vec_distance ASC,id ASC LIMIT $limit", self.fields(), self.table(), self.vector(), self.predicate())
    }

    fn mirror_sql(self, limit: usize) -> String {
        // One SQL statement: unchanged native KNN admits the limited pool,
        // then one record dereference hydrates each selected canonical row.
        // The mirror's native key equals the primary key; no string map or
        // new B-tree access path changes the scan sequence at a tied boundary.
        format!("SELECT {},vec_distance FROM (SELECT record_id.* AS row,vec_distance FROM (SELECT id,record_id,vector::distance::knn() AS vec_distance FROM {} WHERE embedding <|{limit},COSINE|> $embedding AND {} ORDER BY vec_distance ASC,id ASC LIMIT $limit)) ORDER BY vec_distance ASC,id ASC", self.hydrated_fields(), self.mirror(), self.predicate())
    }
}

// No COMPUTED field, norm cache, CBOR or events. Original HNSW indexes are
// retained in measured roles; explicit COSINE remains the exact KNN operator.
const MIRROR_SCHEMA: &str = "
DEFINE TABLE native_float_note_probe SCHEMAFULL;
DEFINE FIELD record_id ON native_float_note_probe TYPE record<note>;
DEFINE FIELD embedding ON native_float_note_probe TYPE option<array<float>>;
DEFINE FIELD created_at ON native_float_note_probe TYPE datetime;
DEFINE FIELD source_id ON native_float_note_probe TYPE option<record<source>>;
DEFINE FIELD source_generation ON native_float_note_probe TYPE option<int>;
DEFINE TABLE native_float_message_probe SCHEMAFULL;
DEFINE FIELD record_id ON native_float_message_probe TYPE record<message>;
DEFINE FIELD embedding ON native_float_message_probe TYPE option<array<float>>;
DEFINE FIELD created_at ON native_float_message_probe TYPE option<datetime>;
DEFINE FIELD conversation_id ON native_float_message_probe TYPE record<conversation>;
DEFINE TABLE native_float_conversation_probe SCHEMAFULL;
DEFINE FIELD record_id ON native_float_conversation_probe TYPE record<conversation>;
DEFINE FIELD embedding ON native_float_conversation_probe TYPE option<array<float>>;
DEFINE FIELD updated_at ON native_float_conversation_probe TYPE datetime;
DEFINE FIELD source_uri ON native_float_conversation_probe TYPE option<string>;
";

fn key(n: usize) -> RecordIdKey {
    match n % 6 {
        0 => RecordIdKey::Number(n as i64),
        1 => format!("fictional-{n:06}").into(),
        2 => RecordIdKey::Uuid(uuid::Uuid::from_u128(n as u128 + 1).into()),
        3 => {
            let mut value = Object::new();
            value.insert("label", "fictional");
            value.insert("ordinal", n as i64);
            RecordIdKey::Object(value)
        }
        _ => RecordIdKey::Array(Array::from(vec![
            if n % 6 == 4 {
                Value::Number(Number::Float(1.0))
            } else {
                Value::Number(Number::Int(1))
            },
            Value::Number(Number::Int(n as i64)),
        ])),
    }
}

fn dense(seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as u32) as f32 / 16777216.0 * 2.0 - 1.0
        })
        .collect()
}

async fn database(label: &str) -> (DbConnection, &'static str) {
    if let Some(root) = std::env::var_os("GRAPHRAG_NATIVE_FLOAT_ROCKS_ROOT") {
        #[cfg(feature = "rocksdb")]
        {
            let path = std::path::PathBuf::from(root).join(label);
            assert!(!path.exists(), "qualification never reuses a database");
            return (crate::init_persistent(path).await.unwrap(), "rocksdb");
        }
        #[cfg(not(feature = "rocksdb"))]
        panic!("qualification requires rocksdb feature: {root:?}");
    }
    (crate::init_memory().await.unwrap(), "memory")
}

async fn populate(db: &DbConnection, count: usize, historical: bool) -> Vec<f32> {
    assert!(count >= 160);
    if historical {
        // Historical malformed dimensions/nonfinite values must not be
        // rejected by HNSW ingestion before the exact-query comparison.
        // Measured roles never remove these indexes.
        db.query("REMOVE INDEX idx_note_embedding ON note; REMOVE INDEX idx_message_embedding ON message; REMOVE INDEX idx_conversation_summary_embedding ON conversation;")
            .await.unwrap().check().unwrap();
    }
    for n in 0..4 {
        let mut source = Source::manual().with_content("fictional source body ".repeat(512));
        source.uri = Some(format!("fixture://native-float-{n}"));
        source.successful_generation = 2;
        source.generation = 3;
        source.created_at = Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap();
        let _: Option<Source> = db
            .create(("source", format!("native-float-{n}")))
            .content(source)
            .await
            .unwrap();
    }
    let query = dense(0x4d595df4d0f33173);
    for start in (0..count).step_by(32) {
        let mut records = Vec::new();
        for n in start..(start + 32).min(count) {
            let mut embedding: Vec<f64> = if n < 96 {
                query.clone()
            } else {
                dense((n as u64 + 1) * 0x9e3779b9)
            }
            .into_iter()
            .map(f64::from)
            .collect();
            if historical {
                match n {
                    100 => embedding.fill(0.0),
                    101 => embedding.fill(-0.0),
                    102 => embedding[0] = f64::NAN,
                    103 => embedding[0] = f64::from_bits(0xfff8_0000_0000_0123),
                    104 => embedding[0] = f64::INFINITY,
                    105 => embedding[0] = f64::NEG_INFINITY,
                    106 => embedding = vec![1.0, 0.25],
                    107 | 108 => embedding.clear(),
                    109 => embedding[0] = f64::from_bits(0x3fd0_1234_5678_9abc),
                    110 => embedding[0] = f64::MAX,
                    _ => {}
                }
            }
            let repeats = [0, 1024, 4096][n % 3];
            let body = format!("Fictional row {n}: {}", "payload ".repeat(repeats));
            let timestamp = Utc
                .with_ymd_and_hms(2030, 1, if n % 7 == 0 { 1 } else { 3 }, 0, 0, 0)
                .unwrap();
            for scope in SCOPES {
                let mut row = Object::new();
                row.insert("id", RecordId::new(scope.table(), key(n)));
                if !(historical && n == 108) {
                    row.insert(scope.vector(), embedding.clone());
                }
                match scope {
                    Scope::Note => {
                        row.insert("title", format!("Fictional row {n}"));
                        row.insert("content", body.clone());
                        row.insert("created_at", Datetime::from(timestamp));
                        row.insert("updated_at", Datetime::from(timestamp));
                        row.insert("tags", vec!["fictional".to_string()]);
                        if n % 5 != 0 {
                            row.insert(
                                "source_id",
                                RecordId::new("source", format!("native-float-{}", n % 4)),
                            );
                            if n % 19 != 0 {
                                row.insert(
                                    "source_generation",
                                    if n % 17 == 0 { 3_i64 } else { 2_i64 },
                                );
                            }
                        }
                    }
                    Scope::Message => {
                        row.insert("message_key", format!("native-float-{n}"));
                        row.insert("conversation_id", RecordId::new("conversation", key(n)));
                        row.insert("conversation_uuid", format!("native-float-{n}"));
                        row.insert("message_index", n as i64);
                        row.insert("role", "user");
                        row.insert("content", body.clone());
                        row.insert("ingested_at", Datetime::from(timestamp));
                        if n % 11 != 0 {
                            row.insert("created_at", Datetime::from(timestamp));
                        }
                    }
                    Scope::Conversation => {
                        row.insert("uuid", format!("native-float-{n}"));
                        row.insert("title", format!("Fictional row {n}"));
                        row.insert("summary", body.clone());
                        row.insert("source_uri", format!("fixture://native-float-{}", n % 4));
                        row.insert("created_at", Datetime::from(timestamp));
                        row.insert("updated_at", Datetime::from(timestamp));
                        row.insert("ingested_at", Datetime::from(timestamp));
                        let mut metadata = Object::new();
                        metadata.insert("opaque", "fictional metadata ".repeat(n % 17));
                        metadata.insert(
                            "fraction",
                            Number::Float(f64::from_bits(0x3fd0_1234_5678_9abc)),
                        );
                        row.insert("metadata", metadata);
                    }
                }
                records.push(Value::Object(row));
            }
        }
        db.query("FOR $row IN $rows { LET $id=$row.id; CREATE $id CONTENT object::remove($row,'id') RETURN NONE; }; RETURN NONE;")
            .bind(("rows", records)).await.unwrap().check().unwrap();
    }
    query
}

async fn mirror_idle_fixture(db: &DbConnection) {
    // DEFINE is performed once, outside every timed query/transaction. A
    // second invocation refuses instead of repeating a DEFINE in a live tx.
    db.query(MIRROR_SCHEMA).await.unwrap().check().unwrap();
    for scope in SCOPES {
        let mut after: Option<RecordId> = None;
        loop {
            let range = RecordId::new(
                scope.table(),
                RecordIdKey::Range(Box::new(RecordIdKeyRange {
                    start: after.as_ref().map_or(std::ops::Bound::Unbounded, |id| {
                        std::ops::Bound::Excluded(id.key.clone())
                    }),
                    end: std::ops::Bound::Unbounded,
                })),
            );
            let ids: Vec<RecordId> = db
                .query("RETURN ((SELECT VALUE id FROM $range ORDER BY id ASC LIMIT 128) ?? []);")
                .bind(("range", range))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            let Some(last) = ids.last().cloned() else {
                break;
            };
            let fields = match scope {
                Scope::Note => "created_at:$id.created_at,source_id:$id.source_id,source_generation:$id.source_generation",
                Scope::Message => "created_at:$id.created_at,conversation_id:$id.conversation_id",
                Scope::Conversation => "updated_at:$id.updated_at,source_uri:$id.source_uri",
            };
            let sql = format!("FOR $id IN $ids {{ CREATE type::record('{}',record::id($id)) CONTENT {{record_id:$id,embedding:$id.{},{fields}}} RETURN NONE; }}; RETURN NONE;", scope.mirror(), scope.vector());
            db.query(sql)
                .bind(("ids", ids))
                .await
                .unwrap()
                .check()
                .unwrap();
            after = Some(last);
        }
    }
}

fn exact_key(key: &RecordIdKey) -> serde_json::Value {
    match key {
        RecordIdKey::Array(value) => {
            serde_json::json!({"Array":value.iter().map(exact_value).collect::<Vec<_>>()})
        }
        RecordIdKey::Object(value) => {
            serde_json::json!({"Object":value.iter().map(|(k,v)|(k.clone(),exact_value(v))).collect::<std::collections::BTreeMap<_,_>>()})
        }
        _ => serde_json::to_value(key).unwrap(),
    }
}

fn exact_value(value: &Value) -> serde_json::Value {
    match value {
        Value::Number(Number::Float(value)) => {
            serde_json::json!({"FloatBits":format!("{:016x}",value.to_bits())})
        }
        Value::RecordId(id) => {
            serde_json::json!({"RecordId":{"table":id.table.as_str(),"key":exact_key(&id.key)}})
        }
        Value::Array(value) => {
            serde_json::json!({"Array":value.iter().map(exact_value).collect::<Vec<_>>()})
        }
        Value::Object(value) => {
            serde_json::json!({"Object":value.iter().map(|(k,v)|(k.clone(),exact_value(v))).collect::<std::collections::BTreeMap<_,_>>()})
        }
        _ => serde_json::to_value(value).unwrap(),
    }
}

fn fingerprint(rows: &[Value]) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&rows.iter().map(exact_value).collect::<Vec<_>>()).unwrap()
        )
    )
}

fn distance_bits(rows: &[Value], serving_f32: bool) -> Vec<Option<u64>> {
    rows.iter()
        .map(|row| match row {
            Value::Object(row) => match row.get("vec_distance") {
                Some(Value::Number(number)) => number.to_f64().map(|n| {
                    if serving_f32 {
                        u64::from((n as f32).to_bits())
                    } else {
                        n.to_bits()
                    }
                }),
                None | Some(Value::None | Value::Null) => None,
                other => panic!("unexpected native distance {other:?}"),
            },
            other => panic!("unexpected result {other:?}"),
        })
        .collect()
}

#[derive(Serialize)]
struct Phases {
    sdk_await_ms: f64,
    dto_take_ms: f64,
    engine_execution_ms: Option<f64>,
}

async fn run_query(
    db: &DbConnection,
    sql: String,
    embedding: &[f32],
    limit: usize,
    since: Option<String>,
    source_uri: Option<String>,
) -> Result<(Vec<Value>, Phases)> {
    let started = std::time::Instant::now();
    let mut response = db
        .query(sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", limit))
        .bind(("since", since))
        .bind(("source_uri", source_uri))
        .with_stats()
        .await?;
    let sdk_await_ms = started.elapsed().as_secs_f64() * 1000.0;
    let take = std::time::Instant::now();
    let (stats, rows) = response
        .take::<Vec<Value>>(0)
        .ok_or_else(|| DbError::QueryFailed("missing native Float statement result".into()))?;
    let rows = rows?;
    Ok((
        rows,
        Phases {
            sdk_await_ms,
            dto_take_ms: take.elapsed().as_secs_f64() * 1000.0,
            engine_execution_ms: stats.execution_time.map(|v| v.as_secs_f64() * 1000.0),
        },
    ))
}

fn emit(value: serde_json::Value) {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{value}").unwrap();
    stdout.flush().unwrap();
}

async fn assert_public_contract(
    db: &DbConnection,
    scope: Scope,
    embedding: &[f32],
    limit: usize,
    since: Option<String>,
    source: Option<String>,
    rows: &[Value],
) {
    let repository = Repository::new(db.clone());
    let timestamp = since.as_ref().map(|value| {
        DateTime::parse_from_rfc3339(value)
            .unwrap()
            .with_timezone(&Utc)
    });
    let raw = Value::Array(Array::from(rows.to_vec()));
    let (expected, actual, bits) = match scope {
        Scope::Note => {
            let expected = Vec::<SearchResult>::from_value(raw).unwrap();
            let actual = repository
                .vector_search_notes(embedding.to_vec(), limit, timestamp, source)
                .await
                .unwrap();
            let bits = actual
                .iter()
                .map(|row| row.vec_distance.map(|v| u64::from(v.to_bits())))
                .collect::<Vec<_>>();
            (
                serde_json::to_value(expected).unwrap(),
                serde_json::to_value(actual).unwrap(),
                bits,
            )
        }
        Scope::Message => {
            let expected = Vec::<MessageSearchResult>::from_value(raw).unwrap();
            let actual = repository
                .vector_search_messages(embedding.to_vec(), limit, timestamp, source)
                .await
                .unwrap();
            let bits = actual
                .iter()
                .map(|row| row.vec_distance.map(|v| u64::from(v.to_bits())))
                .collect::<Vec<_>>();
            (
                serde_json::to_value(expected).unwrap(),
                serde_json::to_value(actual).unwrap(),
                bits,
            )
        }
        Scope::Conversation => {
            let expected = Vec::<ConversationSearchResult>::from_value(raw).unwrap();
            let actual = repository
                .vector_search_conversation_summaries(embedding.to_vec(), limit, timestamp, source)
                .await
                .unwrap();
            let bits = actual
                .iter()
                .map(|row| row.vec_distance.map(|v| u64::from(v.to_bits())))
                .collect::<Vec<_>>();
            (
                serde_json::to_value(expected).unwrap(),
                serde_json::to_value(actual).unwrap(),
                bits,
            )
        }
    };
    assert_eq!(
        expected, actual,
        "baseline SQL equals current public scope DTO"
    );
    assert_eq!(
        bits,
        distance_bits(rows, true),
        "public serving distance bits"
    );
}

async fn assert_inventory(db: &DbConnection, count: usize) -> Vec<String> {
    let mut primary = Vec::new();
    for scope in SCOPES {
        let mut after: Option<RecordId> = None;
        let mut seen = 0;
        let mut digest = Sha256::new();
        loop {
            let range = RecordId::new(
                scope.table(),
                RecordIdKey::Range(Box::new(RecordIdKeyRange {
                    start: after.as_ref().map_or(std::ops::Bound::Unbounded, |id| {
                        std::ops::Bound::Excluded(id.key.clone())
                    }),
                    end: std::ops::Bound::Unbounded,
                })),
            );
            let rows: Vec<Value> = db
                .query("SELECT * FROM $range ORDER BY id ASC LIMIT 128")
                .bind(("range", range))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert!(rows.len() <= 128);
            if rows.is_empty() {
                break;
            }
            let mut mirror_ids = Vec::new();
            let mut vectors = std::collections::BTreeMap::new();
            for value in &rows {
                let Value::Object(row) = value else {
                    panic!("expected canonical object")
                };
                let Some(Value::RecordId(id)) = row.get("id") else {
                    panic!("expected typed canonical ID")
                };
                mirror_ids.push(RecordId::new(scope.mirror(), id.key.clone()));
                vectors.insert(
                    serde_json::to_string(&exact_value(&Value::RecordId(id.clone()))).unwrap(),
                    exact_value(row.get(scope.vector()).unwrap_or(&Value::None)),
                );
                after = Some(id.clone());
                digest.update(serde_json::to_vec(&exact_value(value)).unwrap());
                digest.update(b"\n");
            }
            let mirrors: Vec<Value> = db
                .query("SELECT record_id AS id,embedding FROM $ids")
                .bind(("ids", mirror_ids))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert_eq!(mirrors.len(), rows.len());
            let actual: std::collections::BTreeMap<_, _> = mirrors
                .iter()
                .map(|value| {
                    let Value::Object(row) = value else {
                        panic!("expected mirror object")
                    };
                    let id = row.get("id").expect("mirror canonical ID");
                    (
                        serde_json::to_string(&exact_value(id)).unwrap(),
                        exact_value(row.get("embedding").unwrap_or(&Value::None)),
                    )
                })
                .collect();
            assert_eq!(
                actual, vectors,
                "full native vector bits and typed key inventory {scope:?}"
            );
            seen += rows.len();
        }
        assert_eq!(seen, count);
        let mirror_count: Vec<Value> = db
            .query(format!(
                "SELECT count() AS count FROM {} GROUP ALL",
                scope.mirror()
            ))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(mirror_count.len(), 1, "exactly one mirror count row");
        let Value::Object(row) = &mirror_count[0] else {
            panic!("expected native mirror count object")
        };
        assert_eq!(
            row.get("count"),
            Some(&Value::Number(Number::Int(count as i64))),
            "no extra mirror rows"
        );
        primary.push(format!("{:x}", digest.finalize()));
    }
    primary
}

async fn native_float_all_scopes_exact_correctness_body(database_label: &str) {
    let (db, _) = database(database_label).await;
    let positive = populate(&db, 160, true).await;
    mirror_idle_fixture(&db).await;
    let before = assert_inventory(&db, 160).await;
    let mut nan = positive.clone();
    nan[0] = -f32::NAN;
    let mut infinity = positive.clone();
    infinity[0] = f32::INFINITY;
    for embedding in [
        positive,
        vec![0.0; 1024],
        vec![-0.0; 1024],
        vec![1.0, 0.0],
        vec![],
        nan,
        infinity,
    ] {
        for scope in SCOPES {
            for (since, source) in [
                (None, None),
                (Some("2030-01-02T00:00:00Z".into()), None),
                (None, Some("fixture://native-float-1".into())),
                (
                    Some("2030-01-02T00:00:00Z".into()),
                    Some("fixture://native-float-1".into()),
                ),
                (None, Some("fixture://absent".into())),
            ] {
                for limit in [1, 5, 50] {
                    let (old, _) = run_query(
                        &db,
                        scope.baseline_sql(limit),
                        &embedding,
                        limit,
                        since.clone(),
                        source.clone(),
                    )
                    .await
                    .unwrap();
                    let (new, _) = run_query(
                        &db,
                        scope.mirror_sql(limit),
                        &embedding,
                        limit,
                        since.clone(),
                        source.clone(),
                    )
                    .await
                    .unwrap();
                    assert_public_contract(
                        &db,
                        scope,
                        &embedding,
                        limit,
                        since.clone(),
                        source.clone(),
                        &old,
                    )
                    .await;
                    assert_eq!(
                        fingerprint(&old),
                        fingerprint(&new),
                        "ordered full typed payload {scope:?}/{limit}"
                    );
                    assert_eq!(
                        distance_bits(&old, false),
                        distance_bits(&new, false),
                        "native F64 distance bits"
                    );
                    assert_eq!(
                        distance_bits(&old, true),
                        distance_bits(&new, true),
                        "serving F32 distance bits"
                    );
                }
            }
        }
    }
    // successful_generation is deliberately NOT copied into the mirror. A
    // source promotion changes the visible population in the same query
    // snapshot without a mirror rewrite or cached source-generation map.
    db.query("UPDATE source:⟨native-float-1⟩ SET successful_generation=3 RETURN NONE")
        .await
        .unwrap()
        .check()
        .unwrap();
    let promoted_embedding = dense(0x4d595df4d0f33173);
    for limit in [5, 50] {
        let (old, _) = run_query(
            &db,
            Scope::Note.baseline_sql(limit),
            &promoted_embedding,
            limit,
            None,
            Some("fixture://native-float-1".into()),
        )
        .await
        .unwrap();
        let (new, _) = run_query(
            &db,
            Scope::Note.mirror_sql(limit),
            &promoted_embedding,
            limit,
            None,
            Some("fixture://native-float-1".into()),
        )
        .await
        .unwrap();
        assert_eq!(
            fingerprint(&old),
            fingerprint(&new),
            "current generation after source promotion"
        );
        assert_eq!(distance_bits(&old, false), distance_bits(&new, false));
        assert_public_contract(
            &db,
            Scope::Note,
            &promoted_embedding,
            limit,
            None,
            Some("fixture://native-float-1".into()),
            &old,
        )
        .await;
    }
    assert_eq!(
        before,
        assert_inventory(&db, 160).await,
        "canonical records unchanged"
    );
}

async fn native_float_all_scopes_release_qualification_body(database_suffix: &str) {
    let count: usize = std::env::var("GRAPHRAG_NATIVE_FLOAT_RECORDS")
        .expect("explicit population")
        .parse()
        .unwrap();
    assert!([1024, 4096].contains(&count));
    let order =
        std::env::var("GRAPHRAG_NATIVE_FLOAT_ORDER").expect("explicit counterbalanced start order");
    assert!(["baseline_first", "mirror_first"].contains(&order.as_str()));
    let correctness_only = std::env::var("GRAPHRAG_NATIVE_FLOAT_CORRECTNESS_ONLY")
        .ok()
        .is_some_and(|v| {
            assert_eq!(v, "1");
            true
        });
    let observations = if correctness_only { 1 } else { 21 };
    let (db, backend) = database(&format!("native-float-{count}-{order}{database_suffix}")).await;
    let embedding = populate(&db, count, false).await;
    mirror_idle_fixture(&db).await;
    let inventory = assert_inventory(&db, count).await;
    emit(
        serde_json::json!({"native_float_fixture":"prepared","schema_version":1,"backend":backend,"population_per_scope":count,"body_payload_repeats_per_row":[0,1024,4096],"body_pattern":"n%3 mixed payload repetitions","engine":"surrealdb-core 3.2.4","dimension":1024,"tie_rows_per_scope":96,"primary_full_row_sha256":inventory,"qualification_only":true,"events_or_runtime_migration":false,"provider_calls":0,"correctness_only":correctness_only,"observations_per_variant":observations,"planned_attempts":18*2*observations,"order":order}),
    );
    let mut attempts = 0;
    for scope in SCOPES {
        for (filter, since, source) in [
            ("all", None, None),
            ("recent", Some("2030-01-02T00:00:00Z".into()), None),
            ("source", None, Some("fixture://native-float-1".into())),
        ] {
            for limit in [5, 50] {
                let sqls = [scope.baseline_sql(limit), scope.mirror_sql(limit)];
                let (reference, _) = run_query(
                    &db,
                    sqls[0].clone(),
                    &embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await
                .unwrap();
                emit(
                    serde_json::json!({"native_float_reference":true,"scope":scope.table(),"filter":filter,"limit":limit,"full_typed_payload":reference.iter().map(exact_value).collect::<Vec<_>>(),"full_typed_payload_sha256":fingerprint(&reference),"native_f64_distance_bits":distance_bits(&reference,false),"serving_f32_distance_bits":distance_bits(&reference,true),"unmeasured":true}),
                );
                let mut samples = [Vec::new(), Vec::new()];
                for sample_index in 0..observations {
                    let first = (sample_index + usize::from(order == "mirror_first")) % 2;
                    for variant in [first, 1 - first] {
                        let sequence = attempts;
                        attempts += 1;
                        let name = ["baseline", "native_float"][variant];
                        emit(
                            serde_json::json!({"native_float_attempt_event":"started","schema_version":1,"scope":scope.table(),"filter":filter,"limit":limit,"variant":name,"sequence":sequence,"sample_index":sample_index}),
                        );
                        let start = std::time::Instant::now();
                        let outcome = run_query(
                            &db,
                            sqls[variant].clone(),
                            &embedding,
                            limit,
                            since.clone(),
                            source.clone(),
                        )
                        .await;
                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let Ok((rows, phases)) = outcome else {
                            emit(
                                serde_json::json!({"native_float_attempt_event":"completed","status":"failed","scope":scope.table(),"filter":filter,"limit":limit,"variant":name,"sequence":sequence,"sample_index":sample_index,"elapsed_ms":elapsed_ms,"error":"native_query_or_result_conversion_failed"}),
                            );
                            panic!("failed qualification attempt retained; no retry");
                        };
                        let full_rows_exact = fingerprint(&rows) == fingerprint(&reference);
                        let f64_exact =
                            distance_bits(&rows, false) == distance_bits(&reference, false);
                        let f32_exact =
                            distance_bits(&rows, true) == distance_bits(&reference, true);
                        emit(
                            serde_json::json!({"native_float_attempt_event":"completed","status":"observed","scope":scope.table(),"filter":filter,"limit":limit,"variant":name,"sequence":sequence,"sample_index":sample_index,"elapsed_ms":elapsed_ms,"rows":rows.len(),"full_typed_payload_sha256":fingerprint(&rows),"full_rows_exact":full_rows_exact,"native_f64_distance_bits_exact":f64_exact,"serving_f32_distance_bits_exact":f32_exact,"phases":phases}),
                        );
                        assert!(
                            full_rows_exact && f64_exact && f32_exact,
                            "completed mismatched attempt retained"
                        );
                        samples[variant].push(elapsed_ms);
                    }
                }
                emit(
                    serde_json::json!({"native_float_category":"completed","scope":scope.table(),"filter":filter,"limit":limit,"first_and_warm_baseline_ms":samples[0],"first_and_warm_native_float_ms":samples[1],"correctness_only":correctness_only}),
                );
            }
        }
    }
    assert_eq!(attempts, 18 * 2 * observations);
    assert_eq!(
        inventory,
        assert_inventory(&db, count).await,
        "canonical records unchanged after all measured calls"
    );
    emit(
        serde_json::json!({"native_float_qualification":"observations_complete","status":"passed_exactness","completed_attempts":attempts,"timing_acceptance":"must_be_recomputed_from_all_first_and_20_warm_samples","performance_claim":false,"correctness_only":correctness_only,"provider_calls":0}),
    );
}

// The two scheduler settings share the same fixture and query bodies, with
// distinct startup database labels so both wrappers can use one fresh root.
// These observations are outside query timing. Worker metrics count Tokio
// scheduler workers, not blocking, RocksDB, Rayon or all operating-system
// threads, and do not prove that both workers execute a particular query.
#[derive(Clone, Copy)]
enum QualificationRuntime {
    CurrentThread,
    TwoWorker,
}

impl QualificationRuntime {
    fn observe(self, fixture: &str, stage: &str) {
        let handle = tokio::runtime::Handle::current();
        let flavor = handle.runtime_flavor();
        let workers = handle.metrics().num_workers();
        let (name, expected_flavor, expected_workers) = match self {
            Self::CurrentThread => (
                "current_thread",
                tokio::runtime::RuntimeFlavor::CurrentThread,
                1,
            ),
            Self::TwoWorker => (
                "multi_thread",
                tokio::runtime::RuntimeFlavor::MultiThread,
                2,
            ),
        };
        emit(serde_json::json!({
            "native_float_runtime": "observed",
            "schema_version": 1,
            "runtime": name,
            "observed_flavor": format!("{flavor:?}"),
            "observed_workers": workers,
            "expected_workers": expected_workers,
            "fixture": fixture,
            "stage": stage,
            "query_timing_included": false,
        }));
        assert_eq!(flavor, expected_flavor, "observed Tokio runtime flavor");
        assert_eq!(
            workers, expected_workers,
            "observed Tokio scheduler workers"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits() {
    QualificationRuntime::CurrentThread.observe("correctness", "before_fixture");
    native_float_all_scopes_exact_correctness_body("native-float-correctness").await;
    QualificationRuntime::CurrentThread.observe("correctness", "after_fixture");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits_two_worker_runtime()
{
    QualificationRuntime::TwoWorker.observe("correctness", "before_fixture");
    native_float_all_scopes_exact_correctness_body("native-float-correctness-two-worker").await;
    QualificationRuntime::TwoWorker.observe("correctness", "after_fixture");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "root-owned release qualification; no performance CI assertion"]
async fn native_float_all_scopes_release_qualification() {
    QualificationRuntime::CurrentThread.observe("measurement", "before_fixture");
    native_float_all_scopes_release_qualification_body("").await;
    QualificationRuntime::CurrentThread.observe("measurement", "after_fixture");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "root-owned release qualification; no performance CI assertion"]
async fn native_float_all_scopes_release_qualification_two_worker_runtime() {
    QualificationRuntime::TwoWorker.observe("measurement", "before_fixture");
    native_float_all_scopes_release_qualification_body("-two-worker").await;
    QualificationRuntime::TwoWorker.observe("measurement", "after_fixture");
}

#[path = "message_recent_stage_diagnostic.rs"]
mod message_recent_stage_diagnostic;
