//! Ignored, fictional message/recent/K50 correctness and plan screen.
//! This is a child of exact_vector_native_float_qualification so the frozen
//! fixture helpers remain unchanged. No elapsed sampling or adoption gate.

use super::*;

const POPULATION: usize = 1024;
const LIMIT: usize = 50;
const SINCE: &str = "2030-01-02T00:00:00Z";

fn admission_sql() -> String {
    format!("SELECT id,record_id,vector::distance::knn() AS vec_distance FROM {} WHERE embedding <|50,COSINE|> $embedding AND {} ORDER BY vec_distance ASC,id ASC LIMIT $limit", Scope::Message.mirror(), Scope::Message.predicate())
}

fn hydration_sql(mirror: &str, admission: &str) -> String {
    let inner = format!("({admission})");
    assert_eq!(mirror.matches(&inner).count(), 1);
    let hydration = mirror.replacen(&inner, "$admitted", 1);
    assert_eq!(
        hydration.replacen("FROM $admitted", &format!("FROM {inner}"), 1),
        mirror,
        "only the innermost admission is replaced by native bound rows"
    );
    hydration
}

// No JSON/string-ID/DTO round trip is used for the H input. Every returned
// Value, record key kind and distance Number from P is bound directly.
async fn rows(
    db: &DbConnection,
    sequence: &mut usize,
    stage: &str,
    variant: &str,
    sql: &str,
    embedding: &[f32],
    admitted: Option<&[Value]>,
) -> Vec<Value> {
    let ordinal = *sequence;
    *sequence += 1;
    emit(
        serde_json::json!({"message_recent_stage":"started","sequence":ordinal,"stage":stage,"variant":variant,"sql":sql,"timing_sample":false}),
    );
    let query = db
        .query(sql.to_string())
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", LIMIT))
        .bind(("since", Some(SINCE.to_string())))
        .bind(("source_uri", Option::<String>::None));
    let query = match admitted {
        Some(values) => query.bind(("admitted", values.to_vec())),
        None => query,
    };
    let outcome = query
        .await
        .map_err(|error| error.to_string())
        .and_then(|mut response| {
            response
                .take::<Vec<Value>>(0)
                .map_err(|error| error.to_string())
        });
    match outcome {
        Ok(values) => {
            // Persist complete rows before any correctness assertion.
            emit(
                serde_json::json!({"message_recent_stage":"completed","sequence":ordinal,"stage":stage,"variant":variant,"status":"returned_rows","full_typed_payload":values.iter().map(exact_value).collect::<Vec<_>>(),"full_typed_payload_sha256":fingerprint(&values),"native_f64_distance_bits":distance_bits(&values,false),"serving_f32_distance_bits":distance_bits(&values,true),"timing_sample":false}),
            );
            values
        }
        Err(error) => {
            emit(
                serde_json::json!({"message_recent_stage":"completed","sequence":ordinal,"stage":stage,"variant":variant,"status":"failed","error":error,"timing_sample":false}),
            );
            panic!("stage query failed; retained without retry");
        }
    }
}

fn object(value: &Value) -> &Object {
    let Value::Object(value) = value else {
        panic!("expected native object: {value:?}");
    };
    value
}

fn record(value: &Value) -> &RecordId {
    let Value::RecordId(value) = value else {
        panic!("expected native record ID: {value:?}");
    };
    value
}

fn identity_distances(values: &[Value], pool: bool) -> Vec<Value> {
    values
        .iter()
        .map(|value| {
            let value = object(value);
            let id = record(value.get(if pool { "record_id" } else { "id" }).unwrap());
            assert_eq!(id.table.as_str(), "message");
            if pool {
                let mirror = record(value.get("id").unwrap());
                assert_eq!(mirror.table.as_str(), Scope::Message.mirror());
                assert_eq!(exact_key(&mirror.key), exact_key(&id.key));
            }
            let mut result = Object::new();
            result.insert("id", id.clone());
            result.insert("vec_distance", value.get("vec_distance").unwrap().clone());
            Value::Object(result)
        })
        .collect()
}

fn assert_full(reference: &[Value], actual: &[Value], variant: &str) {
    assert_eq!(actual.len(), LIMIT, "complete selected payload {variant}");
    assert_eq!(
        fingerprint(reference),
        fingerprint(actual),
        "full native payload {variant}"
    );
    assert_eq!(
        distance_bits(reference, false),
        distance_bits(actual, false),
        "F64 {variant}"
    );
    assert_eq!(
        distance_bits(reference, true),
        distance_bits(actual, true),
        "F32 {variant}"
    );
    let dto = |values: &[Value]| {
        let values =
            Vec::<MessageSearchResult>::from_value(Value::Array(Array::from(values.to_vec())))
                .unwrap();
        serde_json::to_value(values).unwrap()
    };
    assert_eq!(dto(reference), dto(actual), "full serving DTO {variant}");
}

fn assert_pool(reference: &[Value], admitted: &[Value]) {
    assert_eq!(admitted.len(), LIMIT);
    assert_eq!(
        fingerprint(&identity_distances(reference, false)),
        fingerprint(&identity_distances(admitted, true)),
        "exact ordered native pool members, key kinds and distance Numbers"
    );
    assert_eq!(
        distance_bits(reference, false),
        distance_bits(admitted, false)
    );
    assert_eq!(
        distance_bits(reference, true),
        distance_bits(admitted, true)
    );
    let first = distance_bits(admitted, false)[0];
    assert!(first.is_some());
    assert!(distance_bits(admitted, false)
        .into_iter()
        .all(|value| value == first));
    for value in reference {
        let Some(Value::Number(Number::Int(index))) = object(value).get("message_index") else {
            panic!("native message index kind");
        };
        assert!(
            (0..96).contains(index),
            "all K50 members come from 75 eligible ties"
        );
    }
}

// Preserve the complete mirror as well as the existing canonical/vector
// inventory. This includes its date/link/filter columns, not only vectors.
async fn mirror_inventory(db: &DbConnection) -> Vec<String> {
    let mut inventories = Vec::new();
    for scope in SCOPES {
        let mut after: Option<RecordId> = None;
        let mut seen = 0;
        let mut digest = Sha256::new();
        loop {
            let range = RecordId::new(
                scope.mirror(),
                RecordIdKey::Range(Box::new(RecordIdKeyRange {
                    start: after.as_ref().map_or(std::ops::Bound::Unbounded, |id| {
                        std::ops::Bound::Excluded(id.key.clone())
                    }),
                    end: std::ops::Bound::Unbounded,
                })),
            );
            let values: Vec<Value> = db
                .query("SELECT * FROM $range ORDER BY id ASC LIMIT 128")
                .bind(("range", range))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert!(values.len() <= 128);
            if values.is_empty() {
                break;
            }
            for value in &values {
                after = Some(record(object(value).get("id").unwrap()).clone());
                digest.update(serde_json::to_vec(&exact_value(value)).unwrap());
                digest.update(b"\n");
            }
            seen += values.len();
        }
        assert_eq!(seen, POPULATION);
        inventories.push(format!("{:x}", digest.finalize()));
    }
    inventories
}

fn plan_root(raw: &serde_json::Value) -> &serde_json::Value {
    let root = if raw.is_object() {
        raw
    } else {
        let values = raw
            .as_array()
            .expect("plan must be an object or singleton wrapper");
        assert_eq!(values.len(), 1, "single actual streaming plan");
        &values[0]
    };
    assert!(root.is_object());
    assert!(
        root["operator"].is_string(),
        "unsupported plan shape retained: {raw}"
    );
    root
}

fn selected_nodes(root: &serde_json::Value) -> Vec<serde_json::Value> {
    fn visit(node: &serde_json::Value, path: String, values: &mut Vec<serde_json::Value>) {
        assert!(node.is_object() && node["operator"].is_string());
        values.push(serde_json::json!({"path":path,"operator":node["operator"],"attributes":node.get("attributes"),"metrics":node.get("metrics"),"metrics_scope":"children_tree_only_inclusive_poll_elapsed"}));
        if let Some(children) = node.get("children") {
            for (index, child) in children
                .as_array()
                .expect("actual children array")
                .iter()
                .enumerate()
            {
                visit(child, format!("{path}/children/{index}"), values);
            }
        }
        // Never count SQL text or expression-embedded plans as selected child
        // operators. 3.2.4 enables metrics through children(), not expressions.
    }
    let mut values = Vec::new();
    visit(root, "root".to_string(), &mut values);
    values
}

async fn plan(
    db: &DbConnection,
    sequence: &mut usize,
    variant: &str,
    sql: &str,
    full: bool,
    embedding: &[f32],
    admitted: Option<&[Value]>,
) {
    let ordinal = *sequence;
    *sequence += 1;
    let mode = if full { "full" } else { "plain" };
    let sql = format!("{sql} EXPLAIN{}", if full { " FULL" } else { "" });
    emit(
        serde_json::json!({"message_recent_stage":"started","sequence":ordinal,"stage":"plan","variant":variant,"mode":mode,"sql":sql,"executes_underlying_query":full,"timing_sample":false}),
    );
    let query = db
        .query(sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("limit", LIMIT))
        .bind(("since", Some(SINCE.to_string())))
        .bind(("source_uri", Option::<String>::None));
    let query = match admitted {
        Some(values) => query.bind(("admitted", values.to_vec())),
        None => query,
    };
    let outcome = query
        .await
        .map_err(|error| error.to_string())
        .and_then(|mut response| {
            response
                .take::<serde_json::Value>(0)
                .map_err(|error| error.to_string())
        });
    match outcome {
        Ok(raw) => {
            // Retain the raw plan, including embedded plans, before shape or
            // row-count assertions. FULL returns a plan, not verified payload.
            emit(
                serde_json::json!({"message_recent_stage":"completed","sequence":ordinal,"stage":"plan","variant":variant,"mode":mode,"status":"returned_plan","raw_plan":raw,"timing_sample":false}),
            );
            let root = plan_root(&raw);
            if full {
                assert_eq!(root["total_rows"].as_u64(), Some(LIMIT as u64));
            }
            emit(
                serde_json::json!({"message_recent_plan":"coverage","sequence":ordinal,"variant":variant,"mode":mode,"selected_children":selected_nodes(root),"embedded_plan_metrics_may_be_disabled":true,"cpu_or_full_await_attribution":false,"normal_payload_verified_by_separate_queries":true}),
            );
        }
        Err(error) => {
            emit(
                serde_json::json!({"message_recent_stage":"completed","sequence":ordinal,"stage":"plan","variant":variant,"mode":mode,"status":"failed","error":error,"timing_sample":false}),
            );
            panic!("plan query failed; retained without retry");
        }
    }
}

async fn stage_body(runtime: QualificationRuntime, label: &str) {
    runtime.observe("message_recent_stage", "before_fixture");
    let (db, backend) = database(label).await;
    let embedding = populate(&db, POPULATION, false).await;
    mirror_idle_fixture(&db).await;
    let canonical = assert_inventory(&db, POPULATION).await;
    let mirrors = mirror_inventory(&db).await;
    let b = Scope::Message.baseline_sql(LIMIT);
    let m = Scope::Message.mirror_sql(LIMIT);
    let p = admission_sql();
    let h = hydration_sql(&m, &p);
    let mut sequence = 0;

    let eligible = rows(
        &db,
        &mut sequence,
        "fixture",
        "eligible_primary",
        &format!(
            "SELECT id,message_index FROM message WHERE {} ORDER BY id ASC",
            Scope::Message.predicate()
        ),
        &embedding,
        None,
    )
    .await;
    assert_eq!(eligible.len(), 797);
    let eligible_ties = eligible.iter().filter(|value| {
        matches!(object(value).get("message_index"), Some(Value::Number(Number::Int(index))) if (0..96).contains(index))
    }).count();
    assert_eq!(eligible_ties, 75);
    let mirror_eligible = rows(
        &db,
        &mut sequence,
        "fixture",
        "eligible_mirror",
        &format!(
            "SELECT id,record_id FROM {} WHERE {} ORDER BY id ASC",
            Scope::Message.mirror(),
            Scope::Message.predicate()
        ),
        &embedding,
        None,
    )
    .await;
    assert_eq!(mirror_eligible.len(), eligible.len());
    for (primary, mirror) in eligible.iter().zip(&mirror_eligible) {
        let id = record(object(primary).get("id").unwrap());
        let row = object(mirror);
        let mirror_id = record(row.get("id").unwrap());
        assert_eq!(
            exact_value(&Value::RecordId(id.clone())),
            exact_value(row.get("record_id").unwrap())
        );
        assert_eq!(exact_key(&id.key), exact_key(&mirror_id.key));
    }
    emit(
        serde_json::json!({"message_recent_fixture":"prepared","backend":backend,"population_per_scope":POPULATION,"eligible_messages":797,"eligible_identical_vector_rows":eligible_ties,"limit":LIMIT,"dimension":1024,"canonical_inventory":canonical,"full_mirror_inventory":mirrors,"native_bound_pool":true,"provider_calls":0,"timing_samples":0,"events_or_production_coherence":false}),
    );

    let reference = rows(
        &db,
        &mut sequence,
        "before_plans",
        "B",
        &b,
        &embedding,
        None,
    )
    .await;
    let mirror = rows(
        &db,
        &mut sequence,
        "before_plans",
        "M",
        &m,
        &embedding,
        None,
    )
    .await;
    let admitted = rows(
        &db,
        &mut sequence,
        "before_plans",
        "P",
        &p,
        &embedding,
        None,
    )
    .await;
    let hydrated = rows(
        &db,
        &mut sequence,
        "before_plans",
        "H",
        &h,
        &embedding,
        Some(&admitted),
    )
    .await;
    assert_full(&reference, &mirror, "M");
    assert_pool(&reference, &admitted);
    assert_full(&reference, &hydrated, "H");
    assert_public_contract(
        &db,
        Scope::Message,
        &embedding,
        LIMIT,
        Some(SINCE.to_string()),
        None,
        &reference,
    )
    .await;

    for full in [false, true] {
        for (variant, sql, pool) in [
            ("B", b.as_str(), None),
            ("M", m.as_str(), None),
            ("P", p.as_str(), None),
            ("H", h.as_str(), Some(admitted.as_slice())),
        ] {
            plan(&db, &mut sequence, variant, sql, full, &embedding, pool).await;
        }
    }

    let after_b = rows(&db, &mut sequence, "after_plans", "B", &b, &embedding, None).await;
    let after_m = rows(&db, &mut sequence, "after_plans", "M", &m, &embedding, None).await;
    let after_p = rows(&db, &mut sequence, "after_plans", "P", &p, &embedding, None).await;
    let after_h = rows(
        &db,
        &mut sequence,
        "after_plans",
        "H",
        &h,
        &embedding,
        Some(&after_p),
    )
    .await;
    assert_full(&reference, &after_b, "B after plans");
    assert_full(&reference, &after_m, "M after plans");
    assert_pool(&reference, &after_p);
    assert_eq!(
        fingerprint(&admitted),
        fingerprint(&after_p),
        "pool remains exact after FULL executions"
    );
    assert_full(&reference, &after_h, "H after plans");
    assert_public_contract(
        &db,
        Scope::Message,
        &embedding,
        LIMIT,
        Some(SINCE.to_string()),
        None,
        &after_b,
    )
    .await;
    assert_eq!(canonical, assert_inventory(&db, POPULATION).await);
    assert_eq!(mirrors, mirror_inventory(&db).await);
    assert_eq!(
        sequence, 18,
        "2 fixture reads, 8 normal stages, 8 plan requests"
    );
    runtime.observe("message_recent_stage", "after_fixture");
    emit(
        serde_json::json!({"message_recent_stage_screen":"complete","status":"passed_correctness_and_plan_capture","declared_stage_requests":18,"normal_stage_result_queries":8,"fixture_read_queries":2,"plain_explain_queries":4,"full_explain_executions":4,"additional_public_contract_queries":2,"full_inventories_before_after":true,"timing_samples":0,"performance_acceptance":false,"provider_calls":0,"production_changes":false}),
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "root-owned fictional correctness/plan screen; no latency acceptance"]
async fn message_recent_k50_plans_current_thread() {
    stage_body(
        QualificationRuntime::CurrentThread,
        "message-recent-k50-stage-current-thread",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "root-owned fictional correctness/plan screen; no latency acceptance"]
async fn message_recent_k50_plans_two_worker() {
    stage_body(
        QualificationRuntime::TwoWorker,
        "message-recent-k50-stage-two-worker",
    )
    .await;
}
