//! Ignored, provider-free qualification of named-field canonical hydration.
//! Parent helpers and baseline/mirror SQL remain unchanged.
//! Fresh root-owned stores and an immutable release build are required.

use super::*;

type NativeRows = std::result::Result<Vec<Value>, String>;
const SINCE: &str = "2030-01-02T00:00:00Z";
const VARIANTS: [&str; 3] = ["B", "M", "D"];

struct Request<'a> {
    sql: String,
    embedding: &'a [f32],
    limit: usize,
    since: Option<String>,
    source: Option<String>,
}

fn needed(scope: Scope) -> &'static str {
    match scope {
        Scope::Note => "id,title,content,note_type,tags,created_at,source_id",
        Scope::Message => {
            "id,conversation_id,conversation_uuid,message_index,role,content,created_at"
        }
        Scope::Conversation => "id,uuid,title,summary,source_uri,updated_at",
    }
}

pub(super) fn named_sql(scope: Scope, limit: usize) -> String {
    let original = scope.mirror_sql(limit);
    let marker = "record_id.* AS row";
    assert_eq!(original.matches(marker).count(), 1);
    let replacement = format!("record_id.{{{}}} AS row", needed(scope));
    let named = original.replacen(marker, &replacement, 1);
    assert_eq!(named.replacen(&replacement, marker, 1), original);
    named
}

pub(super) fn dto(scope: Scope, rows: &[Value]) -> std::result::Result<serde_json::Value, String> {
    let value = Value::Array(Array::from(rows.to_vec()));
    match scope {
        Scope::Note => Vec::<SearchResult>::from_value(value)
            .map(|rows| serde_json::to_value(rows).unwrap())
            .map_err(|error| error.to_string()),
        Scope::Message => Vec::<MessageSearchResult>::from_value(value)
            .map(|rows| serde_json::to_value(rows).unwrap())
            .map_err(|error| error.to_string()),
        Scope::Conversation => Vec::<ConversationSearchResult>::from_value(value)
            .map(|rows| serde_json::to_value(rows).unwrap())
            .map_err(|error| error.to_string()),
    }
}

fn distance_shape(rows: &[Value]) -> bool {
    rows.iter().all(|row| match row {
        Value::Object(row) => matches!(
            row.get("vec_distance"),
            None | Some(Value::Number(_) | Value::None | Value::Null)
        ),
        _ => false,
    })
}

fn raw(outcome: &NativeRows) -> serde_json::Value {
    match outcome {
        Ok(rows) => {
            serde_json::json!({"status":"rows","rows":rows.len(),"full_typed_payload":rows.iter().map(exact_value).collect::<Vec<_>>(),"full_typed_payload_sha256":fingerprint(rows),"native_distance_shape_valid":distance_shape(rows),"native_f64_distance_bits":distance_shape(rows).then(||distance_bits(rows,false)),"serving_f32_distance_bits":distance_shape(rows).then(||distance_bits(rows,true))})
        }
        Err(error) => serde_json::json!({"status":"query_error","error":error}),
    }
}

fn same(scope: Scope, expected: &NativeRows, actual: &NativeRows) -> bool {
    match (expected, actual) {
        (Ok(a), Ok(b)) => {
            fingerprint(a) == fingerprint(b)
                && distance_shape(a)
                && distance_shape(b)
                && distance_bits(a, false) == distance_bits(b, false)
                && distance_bits(a, true) == distance_bits(b, true)
                && dto(scope, a) == dto(scope, b)
        }
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

fn verify_case(
    scope: Scope,
    case: &str,
    outcomes: &[NativeRows; 3],
    require_baseline: bool,
    require_successful_dto: bool,
) {
    let copy_exact = same(scope, &outcomes[1], &outcomes[2]);
    let baseline_exact = same(scope, &outcomes[0], &outcomes[1]);
    // Coherent historical cases require successful native rows; matching
    // errors are retained as observations, never counted as passed cases.
    let successful_coherent_rows =
        !require_baseline || outcomes.iter().all(std::result::Result::is_ok);
    let successful_dto = !require_successful_dto
        || outcomes
            .iter()
            .all(|outcome| matches!(outcome,Ok(rows) if dto(scope,rows).is_ok()));
    emit(
        serde_json::json!({"named_hydration_case":"completed","scope":scope.table(),"case":case,"M_D_full_payload_key_Number_bits_DTO_or_error_exact":copy_exact,"B_M_exact":baseline_exact,"require_baseline":require_baseline,"coherent_successful_rows_required":require_baseline,"successful_coherent_rows":successful_coherent_rows,"successful_DTO_required":require_successful_dto,"successful_DTO":successful_dto,"timing_sample":false}),
    );
    if !copy_exact
        || !successful_coherent_rows
        || !successful_dto
        || (require_baseline && !baseline_exact)
    {
        // Preserve all actual complete returns/errors before failure. No retry.
        emit(
            serde_json::json!({"named_hydration_failure":"retained","scope":scope.table(),"case":case,"B":raw(&outcomes[0]),"M":raw(&outcomes[1]),"D":raw(&outcomes[2]),"automatic_retry":false}),
        );
    }
    assert!(
        copy_exact
            && successful_coherent_rows
            && successful_dto
            && (!require_baseline || baseline_exact),
        "retained exactness failure"
    );
}

async fn observe(
    db: &DbConnection,
    sequence: &mut usize,
    stage: &str,
    variant: &str,
    request: Request<'_>,
) -> NativeRows {
    let ordinal = *sequence;
    *sequence += 1;
    emit(
        serde_json::json!({"named_hydration_attempt":"started","sequence":ordinal,"stage":stage,"variant":variant,"sql":request.sql,"timing_sample":false}),
    );
    let result = run_query(
        db,
        request.sql,
        request.embedding,
        request.limit,
        request.since,
        request.source,
    )
    .await
    .map(|(rows, _)| rows)
    .map_err(|error| error.to_string());
    match &result {
        Ok(rows) => {
            if !distance_shape(rows) {
                emit(
                    serde_json::json!({"named_hydration_failure":"retained","sequence":ordinal,"reason":"unexpected_native_distance_shape","full_actual":raw(&result),"automatic_retry":false}),
                );
            }
            emit(
                serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":stage,"variant":variant,"status":"rows","rows":rows.len(),"full_typed_payload_sha256":fingerprint(rows),"native_distance_shape_valid":distance_shape(rows),"native_f64_distance_bits":distance_shape(rows).then(||distance_bits(rows,false)),"serving_f32_distance_bits":distance_shape(rows).then(||distance_bits(rows,true)),"timing_sample":false}),
            );
        }
        Err(error) => emit(
            serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":stage,"variant":variant,"status":"query_error","error":error,"timing_sample":false}),
        ),
    }
    result
}

pub(super) async fn mirror_inventory(db: &DbConnection, count: usize) -> Vec<String> {
    // Same bounded native inventory pattern; original sibling stays private.
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
            for row in &rows {
                let Value::Object(object) = row else {
                    panic!("native mirror object")
                };
                let Some(Value::RecordId(id)) = object.get("id") else {
                    panic!("native mirror ID")
                };
                after = Some(id.clone());
                digest.update(serde_json::to_vec(&exact_value(row)).unwrap());
                digest.update(b"\n");
            }
            seen += rows.len();
        }
        assert_eq!(seen, count);
        inventories.push(format!("{:x}", digest.finalize()));
    }
    inventories
}

async fn plans(db: &DbConnection, embedding: &[f32], sequence: &mut usize) {
    for full in [false, true] {
        for (variant, sql) in VARIANTS.into_iter().zip([
            Scope::Message.baseline_sql(50),
            Scope::Message.mirror_sql(50),
            named_sql(Scope::Message, 50),
        ]) {
            let ordinal = *sequence;
            *sequence += 1;
            let sql = format!("{sql} EXPLAIN{}", if full { " FULL" } else { "" });
            emit(
                serde_json::json!({"named_hydration_attempt":"started","sequence":ordinal,"stage":"plan","variant":variant,"sql":sql,"full":full,"timing_sample":false}),
            );
            let outcome = db
                .query(sql)
                .bind(("embedding", embedding.to_vec()))
                .bind(("limit", 50usize))
                .bind(("since", Some(SINCE.to_string())))
                .bind(("source_uri", Option::<String>::None))
                .await
                .map_err(|error| error.to_string())
                .and_then(|mut response| {
                    response.take::<Value>(0).map_err(|error| error.to_string())
                });
            match outcome {
                Ok(native) => {
                    let json = native.clone().into_json_value();
                    emit(
                        serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"plan","variant":variant,"full":full,"status":"plan","raw_native_plan":exact_value(&native),"raw_plan":json,"timing_sample":false,"normal_payload_is_separate":true,"inclusive_metrics_not_CPU_or_full_await":true}),
                    );
                    let root = if json.is_object() {
                        &json
                    } else {
                        let values = json.as_array().expect("actual plan object or singleton");
                        assert_eq!(values.len(), 1);
                        &values[0]
                    };
                    assert!(root["operator"].is_string());
                    if full {
                        assert_eq!(root["total_rows"].as_u64(), Some(50));
                    }
                }
                Err(error) => {
                    emit(
                        serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"plan","variant":variant,"status":"query_error","error":error,"timing_sample":false}),
                    );
                    panic!("retained native plan failure; no retry");
                }
            }
        }
    }
}

async fn edges(db: &DbConnection, sequence: &mut usize) {
    // Separate schemaless payload controls avoid removing canonical constraints.
    // Native scope/key/vector coverage is supplied by the canonical fixture.
    for scope in SCOPES {
        // This bounded hydration-only control explicitly returns all six
        // native key forms, regardless of which keys win a tied KNN cap.
        let targets = (0..6)
            .map(|n| RecordId::new(scope.table(), key(n)))
            .collect::<Vec<_>>();
        let pool = targets
            .iter()
            .map(|target| {
                let mut row = Object::new();
                row.insert("record_id", target.clone());
                row.insert("vec_distance", Number::Float(0.125));
                Value::Object(row)
            })
            .collect::<Vec<_>>();
        let outer = scope.hydrated_fields();
        let sqls=[
            format!("SELECT {},$distance AS vec_distance FROM $targets ORDER BY vec_distance ASC,id ASC",scope.fields()),
            format!("SELECT {outer},vec_distance FROM (SELECT record_id.* AS row,vec_distance FROM $pool) ORDER BY vec_distance ASC,id ASC"),
            format!("SELECT {outer},vec_distance FROM (SELECT record_id.{{{}}} AS row,vec_distance FROM $pool) ORDER BY vec_distance ASC,id ASC",needed(scope)),
        ];
        let mut native_keys: [NativeRows; 3] = [Ok(Vec::new()), Ok(Vec::new()), Ok(Vec::new())];
        for (variant, outcome) in native_keys.iter_mut().enumerate() {
            let ordinal = *sequence;
            *sequence += 1;
            emit(
                serde_json::json!({"named_hydration_attempt":"started","sequence":ordinal,"stage":"payload_native_keys","scope":scope.table(),"variant":VARIANTS[variant],"sql":sqls[variant],"timing_sample":false}),
            );
            *outcome = db
                .query(sqls[variant].clone())
                .bind(("targets", targets.clone()))
                .bind(("pool", pool.clone()))
                .bind(("distance", Number::Float(0.125)))
                .await
                .map_err(|error| error.to_string())
                .and_then(|mut response| {
                    response
                        .take::<Vec<Value>>(0)
                        .map_err(|error| error.to_string())
                });
            emit(
                serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"payload_native_keys","scope":scope.table(),"variant":VARIANTS[variant],"outcome":raw(outcome),"timing_sample":false}),
            );
        }
        verify_case(scope, "all_six_native_key_forms", &native_keys, true, true);
        assert!(
            native_keys
                .iter()
                .all(|outcome| outcome.as_ref().unwrap().len() == 6),
            "all six native key forms are returned"
        );
        let table = format!("named_hydration_{}_payload_edge", scope.table());
        db.query(format!("DEFINE TABLE {table} SCHEMALESS"))
            .await
            .unwrap()
            .check()
            .unwrap();
        for edge in 0..9 {
            let case = [
                "absent_optional",
                "explicit_null",
                "missing_required",
                "missing_link",
                "null_link",
                "malformed_link",
                "object_key_shadow",
                "missing_canonical",
                "unused_computed_error",
            ][edge];
            let mut body = Object::new();
            body.insert("title", "Fictional hydration edge");
            body.insert("content", "Fictional payload");
            body.insert("note_type", "text");
            body.insert("tags", vec!["fictional".to_string()]);
            body.insert("uuid", "fictional edge");
            body.insert("summary", "Fictional summary");
            body.insert("source_uri", "fixture://hydration-edge");
            body.insert("conversation_uuid", "fictional edge");
            body.insert("message_index", 1_i64);
            body.insert("role", "user");
            let timestamp = Datetime::from(Utc.with_ymd_and_hms(2030, 1, 3, 0, 0, 0).unwrap());
            body.insert("created_at", timestamp);
            body.insert("updated_at", timestamp);
            body.insert("conversation_id", RecordId::new("conversation", key(1)));
            if edge == 0 || edge == 1 {
                for field in match scope {
                    Scope::Note => vec!["title"],
                    Scope::Message => vec!["created_at"],
                    Scope::Conversation => vec!["title", "summary", "source_uri"],
                } {
                    if edge == 0 {
                        body.remove(field);
                    } else {
                        body.insert(field, Value::Null);
                    }
                }
            }
            if edge == 2 {
                body.remove(match scope {
                    Scope::Note => "content",
                    Scope::Message => "role",
                    Scope::Conversation => "uuid",
                });
            }
            if edge == 3 {
                let missing = RecordId::new("named_hydration_absent_link", "fictional");
                body.insert("source_id", missing.clone());
                body.insert("conversation_id", missing);
            }
            if edge == 4 || edge == 5 {
                let link = if edge == 4 {
                    Value::Null
                } else {
                    Value::String("fictional-malformed-link".into())
                };
                body.insert("source_id", link.clone());
                body.insert("conversation_id", link);
            }
            let target = if edge == 7 {
                RecordId::new(scope.table(), key(1))
            } else if edge == 6 {
                let mut key = Object::new();
                key.insert("id", "fictional key id");
                key.insert("content", "fictional key content");
                let mut linked = Object::new();
                linked.insert("uri", "fixture://key-uri");
                linked.insert("source_uri", "fixture://key-source-uri");
                let link =
                    RecordId::new("named_hydration_absent_link", RecordIdKey::Object(linked));
                body.insert("source_id", link.clone());
                body.insert("conversation_id", link);
                RecordId::new(table.as_str(), RecordIdKey::Object(key))
            } else {
                RecordId::new(table.as_str(), edge as i64)
            };
            if edge == 7 {
                // Orphan an actual canonical target while retaining its mirror.
                // Production must fence/readthrough this case; no baseline waiver.
                db.query("DELETE $target RETURN NONE")
                    .bind(("target", target.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            } else {
                db.query("CREATE $target CONTENT $body RETURN NONE")
                    .bind(("target", target.clone()))
                    .bind(("body", Value::Object(body)))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            if edge == 8 {
                db.query(format!(
                    "DEFINE FIELD ignored_failure ON {table} COMPUTED <int>'fictional-not-an-int'"
                ))
                .await
                .unwrap()
                .check()
                .unwrap();
            }
            let mut pool_row = Object::new();
            pool_row.insert("record_id", target.clone());
            pool_row.insert("vec_distance", Number::Float(0.125));
            let pool = vec![Value::Object(pool_row)];
            let outer = scope.hydrated_fields();
            let sqls = [
                format!("SELECT {},$distance AS vec_distance FROM $target ORDER BY vec_distance ASC,id ASC", scope.fields()),
                format!("SELECT {outer},vec_distance FROM (SELECT record_id.* AS row,vec_distance FROM $pool) ORDER BY vec_distance ASC,id ASC"),
                format!("SELECT {outer},vec_distance FROM (SELECT record_id.{{{}}} AS row,vec_distance FROM $pool) ORDER BY vec_distance ASC,id ASC", needed(scope)),
            ];
            let mut outcomes: [NativeRows; 3] = [Ok(Vec::new()), Ok(Vec::new()), Ok(Vec::new())];
            for (variant, outcome) in outcomes.iter_mut().enumerate() {
                let ordinal = *sequence;
                *sequence += 1;
                emit(
                    serde_json::json!({"named_hydration_attempt":"started","sequence":ordinal,"stage":"payload_edge","scope":scope.table(),"case":case,"variant":VARIANTS[variant],"sql":sqls[variant],"timing_sample":false}),
                );
                *outcome = db
                    .query(sqls[variant].clone())
                    .bind(("target", target.clone()))
                    .bind(("pool", pool.clone()))
                    .bind(("distance", Number::Float(0.125)))
                    .await
                    .map_err(|error| error.to_string())
                    .and_then(|mut response| {
                        response
                            .take::<Vec<Value>>(0)
                            .map_err(|error| error.to_string())
                    });
                emit(
                    serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"payload_edge","scope":scope.table(),"case":case,"variant":VARIANTS[variant],"outcome":raw(outcome),"timing_sample":false}),
                );
            }
            // These two cases expressly compare original hydration behavior;
            // a missing canonical row or unused computed error is not a proof
            // that the mirror mechanism is production-equivalent to baseline.
            let required_dto_error = edge == 1
                || edge == 2
                || (matches!(scope, Scope::Message) && (edge == 4 || edge == 5));
            verify_case(
                scope,
                case,
                &outcomes,
                edge != 7 && edge != 8,
                !required_dto_error && edge != 7 && edge != 8,
            );
            if required_dto_error {
                assert!(
                    outcomes
                        .iter()
                        .all(|outcome| dto(scope, outcome.as_ref().unwrap()).is_err()),
                    "required-field DTO errors remain visible"
                );
            }
            if edge == 7 {
                assert!(outcomes[0].as_ref().unwrap().is_empty());
            }
            if edge == 8 {
                assert!(
                    outcomes[1].is_err() && outcomes[2].is_err(),
                    "unselected computed error must remain visible"
                );
            }
        }
    }
}

async fn correctness(runtime: QualificationRuntime, canonical_label: &str, edge_label: &str) {
    runtime.observe("named_hydration_correctness", "before_fixture");
    let (db, backend) = database(canonical_label).await;
    let positive = populate(&db, 160, true).await;
    mirror_idle_fixture(&db).await;
    let primary = assert_inventory(&db, 160).await;
    let mirrors = mirror_inventory(&db, 160).await;
    let mut nan = positive.clone();
    nan[0] = -f32::NAN;
    let mut infinity = positive.clone();
    infinity[0] = f32::INFINITY;
    let mut sequence = 0;
    let mut cases = 0;
    let mut public_contracts = 0;
    for (vector, embedding) in [
        positive.clone(),
        vec![0.0; 1024],
        vec![-0.0; 1024],
        vec![1.0, 0.0],
        vec![],
        nan,
        infinity,
    ]
    .into_iter()
    .enumerate()
    {
        for scope in SCOPES {
            for (filter, since, source) in [
                ("all", None, None),
                ("recent", Some(SINCE.to_string()), None),
                ("source", None, Some("fixture://native-float-1".to_string())),
                (
                    "both",
                    Some(SINCE.to_string()),
                    Some("fixture://native-float-1".to_string()),
                ),
                ("empty", None, Some("fixture://absent".to_string())),
            ] {
                for limit in [1, 5, 50] {
                    let case = format!("vector-{vector}/{filter}/K{limit}");
                    let sqls = [
                        scope.baseline_sql(limit),
                        scope.mirror_sql(limit),
                        named_sql(scope, limit),
                    ];
                    let mut outcomes: [NativeRows; 3] =
                        [Ok(Vec::new()), Ok(Vec::new()), Ok(Vec::new())];
                    for (variant, outcome) in outcomes.iter_mut().enumerate() {
                        *outcome = observe(
                            &db,
                            &mut sequence,
                            "all_scopes",
                            VARIANTS[variant],
                            Request {
                                sql: sqls[variant].clone(),
                                embedding: &embedding,
                                limit,
                                since: since.clone(),
                                source: source.clone(),
                            },
                        )
                        .await;
                    }
                    verify_case(scope, &case, &outcomes, true, true);
                    if let Ok(baseline) = &outcomes[0] {
                        assert_public_contract(
                            &db,
                            scope,
                            &embedding,
                            limit,
                            since.clone(),
                            source.clone(),
                            baseline,
                        )
                        .await;
                        public_contracts += 1;
                    } else {
                        // verify_case retained all errors and rejected this
                        // outcome before any public-contract call.
                        panic!("retained coherent query error cannot count as passed");
                    }
                    if vector == 0 && filter == "recent" && limit == 50 {
                        emit(
                            serde_json::json!({"named_hydration_reference":"full","scope":scope.table(),"case":case,"B":raw(&outcomes[0]),"M":raw(&outcomes[1]),"D":raw(&outcomes[2]),"timing_sample":false}),
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    emit(
        serde_json::json!({"named_hydration_fixture_mutation":"source_promotion","table":"source","field":"successful_generation","from":2,"to":3,"timing_sample":false,"inventory_does_not_claim_entire_database_unchanged":true}),
    );
    db.query("UPDATE source:⟨native-float-1⟩ SET successful_generation=3 RETURN NONE")
        .await
        .unwrap()
        .check()
        .unwrap();
    let promoted = dense(0x4d595df4d0f33173);
    for limit in [5, 50] {
        let mut outcomes: [NativeRows; 3] = [Ok(Vec::new()), Ok(Vec::new()), Ok(Vec::new())];
        for (variant, sql) in [
            Scope::Note.baseline_sql(limit),
            Scope::Note.mirror_sql(limit),
            named_sql(Scope::Note, limit),
        ]
        .into_iter()
        .enumerate()
        {
            outcomes[variant] = observe(
                &db,
                &mut sequence,
                "source_promoted",
                VARIANTS[variant],
                Request {
                    sql,
                    embedding: &promoted,
                    limit,
                    since: None,
                    source: Some("fixture://native-float-1".to_string()),
                },
            )
            .await;
        }
        verify_case(
            Scope::Note,
            &format!("promoted_source/K{limit}"),
            &outcomes,
            true,
            true,
        );
        assert_public_contract(
            &db,
            Scope::Note,
            &promoted,
            limit,
            None,
            Some("fixture://native-float-1".to_string()),
            outcomes[0].as_ref().unwrap(),
        )
        .await;
        public_contracts += 1;
        cases += 1;
    }
    plans(&db, &positive, &mut sequence).await;
    // Inventories cover canonical note/message/conversation rows and mirrors,
    // not the deliberately promoted Source.successful_generation metadata.
    assert_eq!(primary, assert_inventory(&db, 160).await);
    assert_eq!(mirrors, mirror_inventory(&db, 160).await);
    // Corruption controls are a separate fresh fixture, not a mutation of the
    // untouched historical correctness snapshot or a production coherence test.
    let (edge_db, _) = database(edge_label).await;
    populate(&edge_db, 160, true).await;
    mirror_idle_fixture(&edge_db).await;
    edges(&edge_db, &mut sequence).await;
    assert_eq!(cases, 317);
    assert_eq!(public_contracts, 317);
    assert_eq!(sequence, 1047);
    runtime.observe("named_hydration_correctness", "after_fixture");
    emit(
        serde_json::json!({"named_hydration_correctness":"complete","backend":backend,"historical_query_vectors_successful":["finite_positive","zero","negative_zero","dimension_two","empty","negative_NaN","positive_infinity"],"cases_per_historical_vector":45,"coherent_KNN_cases_B_M_D_successful_rows_and_DTO_exact":317,"native_key_hydration_cases_B_M_D_exact":3,"six_native_key_forms_per_scope":6,"payload_edge_cases_M_D_payload_DTO_or_error_exact":27,"declared_attempt_pairs":1047,"plain_plans":3,"full_plan_executions":3,"outside_ledger_public_contract_queries":317,"inventory_tables_unchanged":["note","message","conversation","native_float_note_probe","native_float_message_probe","native_float_conversation_probe"],"intentional_Source_successful_generation_promotion":true,"entire_database_unchanged_claim":false,"timing_samples":0,"production_coherence_or_adoption":false,"provider_calls":0}),
    );
}

fn identities(rows: &[Value]) -> Vec<serde_json::Value> {
    rows.iter().map(|row| {
        // Preserve unexpected values as well; exactness failure is reported
        // with the complete native payload before any assertion can panic.
        match row {
            Value::Object(row) => serde_json::json!({"id":exact_value(row.get("id").unwrap_or(&Value::None)),"raw_distance_Number":exact_value(row.get("vec_distance").unwrap_or(&Value::None))}),
            other => serde_json::json!({"unexpected_row":exact_value(other)}),
        }
    }).collect()
}

async fn pilot() {
    QualificationRuntime::CurrentThread.observe("named_hydration_pilot", "before_fixture");
    let (db, backend) = database("named-hydration-message-recent-k50-pilot-current-thread").await;
    let embedding = populate(&db, 1024, false).await;
    mirror_idle_fixture(&db).await;
    let primary = assert_inventory(&db, 1024).await;
    let mirrors = mirror_inventory(&db, 1024).await;
    let sqls = [
        Scope::Message.baseline_sql(50),
        Scope::Message.mirror_sql(50),
        named_sql(Scope::Message, 50),
    ];
    let mut sequence = 0;
    let mut references: [NativeRows; 3] = [Ok(Vec::new()), Ok(Vec::new()), Ok(Vec::new())];
    for (variant, reference) in references.iter_mut().enumerate() {
        *reference = observe(
            &db,
            &mut sequence,
            "declared_untimed_reference",
            VARIANTS[variant],
            Request {
                sql: sqls[variant].clone(),
                embedding: &embedding,
                limit: 50,
                since: Some(SINCE.to_string()),
                source: None,
            },
        )
        .await;
    }
    verify_case(
        Scope::Message,
        "pilot_declared_reference",
        &references,
        true,
        true,
    );
    for (variant, reference) in references.iter().enumerate() {
        emit(
            serde_json::json!({"named_hydration_reference":"full","variant":VARIANTS[variant],"outcome":raw(reference),"timing_sample":false}),
        );
        let rows = reference.as_ref().unwrap();
        assert_eq!(rows.len(), 50);
        assert!(rows.iter().all(|row|matches!(row,Value::Object(row) if matches!(row.get("id"),Some(Value::RecordId(_))))),"declared reference has complete native canonical IDs");
    }
    let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for sample_index in 0..21 {
        // Fixed rotation: B/M/D, M/D/B, D/B/M. Seven rounds per lead.
        for offset in 0..3 {
            let variant = (sample_index + offset) % 3;
            let ordinal = sequence;
            sequence += 1;
            emit(
                serde_json::json!({"named_hydration_attempt":"started","sequence":ordinal,"stage":"pilot","variant":VARIANTS[variant],"sample_index":sample_index,"timing_sample":true,"first_measured_after_untimed_references":sample_index==0}),
            );
            let started = std::time::Instant::now();
            let outcome = run_query(
                &db,
                sqls[variant].clone(),
                &embedding,
                50,
                Some(SINCE.to_string()),
                None,
            )
            .await;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (rows, phases) = match outcome {
                Ok(value) => value,
                Err(error) => {
                    emit(
                        serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"pilot","variant":VARIANTS[variant],"sample_index":sample_index,"status":"query_error","elapsed_ms":elapsed_ms,"error":error.to_string(),"timing_sample":true}),
                    );
                    panic!("retained failed pilot call; no retry");
                }
            };
            let actual = Ok(rows);
            let exact = same(Scope::Message, &references[variant], &actual);
            let values = actual.as_ref().unwrap();
            emit(
                serde_json::json!({"named_hydration_attempt":"completed","sequence":ordinal,"stage":"pilot","variant":VARIANTS[variant],"sample_index":sample_index,"status":"observed","elapsed_ms":elapsed_ms,"rows":values.len(),"full_typed_payload_sha256":fingerprint(values),"ordered_native_ids_and_raw_distance_Numbers":identities(values),"native_distance_shape_valid":distance_shape(values),"native_f64_distance_bits":distance_shape(values).then(||distance_bits(values,false)),"serving_f32_distance_bits":distance_shape(values).then(||distance_bits(values,true)),"full_payload_DTO_key_Number_F64_F32_exact":exact,"phases":{"sdk_await_ms":phases.sdk_await_ms,"native_Value_take_ms":phases.dto_take_ms,"engine_execution_ms":phases.engine_execution_ms},"timing_sample":true,"public_DTO_and_fingerprint_checks_outside_elapsed":true}),
            );
            if !exact {
                emit(
                    serde_json::json!({"named_hydration_failure":"retained","sequence":ordinal,"full_actual":raw(&actual),"full_reference":raw(&references[variant]),"automatic_retry":false}),
                );
            }
            assert!(exact, "retained pilot mismatch");
            assert!(elapsed_ms.is_finite() && elapsed_ms >= 0.0);
            samples[variant].push(elapsed_ms);
        }
    }
    assert_eq!(sequence, 66);
    assert!(samples.iter().all(|values| values.len() == 21));
    assert_public_contract(
        &db,
        Scope::Message,
        &embedding,
        50,
        Some(SINCE.to_string()),
        None,
        references[0].as_ref().unwrap(),
    )
    .await;
    assert_eq!(primary, assert_inventory(&db, 1024).await);
    assert_eq!(mirrors, mirror_inventory(&db, 1024).await);
    QualificationRuntime::CurrentThread.observe("named_hydration_pilot", "after_fixture");
    emit(
        serde_json::json!({"named_hydration_pilot":"observations_complete","backend":backend,"scope":"message","filter":"recent","limit":50,"population_per_scope":1024,"dimension":1024,"declared_untimed_reference_queries":3,"timed_queries":63,"outside_ledger_public_contract_queries":1,"first_and_20_warm_B_ms":samples[0],"first_and_20_warm_M_ms":samples[1],"first_and_20_warm_D_ms":samples[2],"first_is_measured_after_declared_references_not_cold_or_first_use":true,"between_call_full_fingerprint_DTO_key_bit_work_declared":true,"performance_acceptance":false,"production_coherence_or_adoption":false,"provider_calls":0}),
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "root-owned fictional named-field exactness; no production adoption"]
async fn named_field_all_scopes_exact_current_thread() {
    correctness(
        QualificationRuntime::CurrentThread,
        "named-hydration-correctness-current-thread",
        "named-hydration-payload-edges-current-thread",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "root-owned fictional named-field exactness; no production adoption"]
async fn named_field_all_scopes_exact_two_worker() {
    correctness(
        QualificationRuntime::TwoWorker,
        "named-hydration-correctness-two-worker",
        "named-hydration-payload-edges-two-worker",
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "root-owned focused first-measured plus20warm pilot; no blanket timing gate"]
async fn named_field_message_recent_k50_pilot_current_thread() {
    pilot().await;
}
