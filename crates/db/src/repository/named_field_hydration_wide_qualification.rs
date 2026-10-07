//! Ignored three-arm qualification of the named-field hydration diagnostic.
//! B/M/D reuse the parent baseline/mirror and the reviewed named-field SQL.
//! One idle fictional mirror snapshot; no production adoption or migration.

use super::named_field_hydration_diagnostic::{dto, mirror_inventory, named_sql};
use super::*;

const VARIANTS: [&str; 3] = ["B", "M", "D"];

fn identities(rows: &[Value]) -> Vec<serde_json::Value> {
    rows.iter()
        .map(|row| match row {
            Value::Object(row) => serde_json::json!({
                "id":exact_value(row.get("id").unwrap_or(&Value::None)),
                "raw_distance_Number":exact_value(row.get("vec_distance").unwrap_or(&Value::None)),
            }),
            other => serde_json::json!({"unexpected_row":exact_value(other)}),
        })
        .collect()
}

fn canonical_distance_shape(scope: Scope, rows: &[Value], limit: usize) -> bool {
    rows.len() == limit && rows.iter().all(|row| match row {
        Value::Object(row) => {
            matches!(row.get("id"), Some(Value::RecordId(id)) if id.table.as_str() == scope.table())
                && matches!(row.get("vec_distance"), Some(Value::Number(number))
                    if number.to_f64().is_some_and(|value| value.is_finite() && (value as f32).is_finite()))
        }
        _ => false,
    })
}

fn public_dto_hash(value: &serde_json::Value) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

async fn wide(runtime: QualificationRuntime, database_label: &str) {
    runtime.observe("named_hydration_wide", "before_fixture");
    let count: usize = std::env::var("GRAPHRAG_NATIVE_FLOAT_RECORDS")
        .expect("explicit population")
        .parse()
        .unwrap();
    assert!([1024, 4096].contains(&count));
    let order =
        std::env::var("GRAPHRAG_NATIVE_FLOAT_ORDER").expect("explicit three-arm rotation offset");
    assert!(["baseline_first", "mirror_first"].contains(&order.as_str()));
    // This selector always retains first +20 warm observations; the old
    // paired fixture's optional shorter correctness mode does not apply.
    assert!(std::env::var_os("GRAPHRAG_NATIVE_FLOAT_CORRECTNESS_ONLY").is_none());
    let order_offset = usize::from(order == "mirror_first");
    let (db, backend) = database(database_label).await;
    let embedding = populate(&db, count, false).await;
    mirror_idle_fixture(&db).await;
    let primary = assert_inventory(&db, count).await;
    let mirrors = mirror_inventory(&db, count).await;
    emit(serde_json::json!({
        "named_hydration_wide_fixture":"prepared",
        "schema_version":1,
        "backend":backend,
        "population_per_scope":count,
        "dimension":1024,
        "tie_rows_per_scope":96,
        "body_payload_repeats_per_row":[0,1024,4096],
        "body_pattern":"n%3 mixed payload repetitions",
        "engine":"surrealdb-core 3.2.4",
        "primary_full_row_sha256":primary,
        "mirror_full_row_sha256":mirrors,
        "order":order,
        "order_offset":order_offset,
        "variants":VARIANTS,
        "categories":18,
        "observations_per_variant_per_category":21,
        "declared_unmeasured_B_references":18,
        "planned_measured_attempt_pairs":1134,
        "outside_ledger_public_contract_queries":0,
        "new_binary_correctness_prerequisites_required":true,
        "original_HNSW_indexes_retained":true,
        "qualification_only":true,
        "events_or_runtime_migration":false,
        "provider_calls":0,
    }));
    let mut attempts = 0;
    let mut categories = 0;
    let mut references = 0;
    for scope in SCOPES {
        for (filter, since, source) in [
            ("all", None, None),
            ("recent", Some("2030-01-02T00:00:00Z".into()), None),
            ("source", None, Some("fixture://native-float-1".into())),
        ] {
            for limit in [5, 50] {
                let category_index = categories;
                let reference_index = references;
                let sqls = [
                    scope.baseline_sql(limit),
                    scope.mirror_sql(limit),
                    named_sql(scope, limit),
                ];
                emit(serde_json::json!({
                    "named_hydration_wide_reference":"started",
                    "schema_version":1,
                    "category_index":category_index,
                    "reference_index":reference_index,
                    "scope":scope.table(),"filter":filter,"limit":limit,
                    "variant":"B","sql":sqls[0],"timing_sample":false,
                }));
                let reference = match run_query(
                    &db,
                    sqls[0].clone(),
                    &embedding,
                    limit,
                    since.clone(),
                    source.clone(),
                )
                .await
                {
                    Ok((rows, _)) => rows,
                    Err(error) => {
                        emit(serde_json::json!({
                            "named_hydration_wide_reference":"completed",
                            "schema_version":1,"category_index":category_index,
                            "reference_index":reference_index,
                            "scope":scope.table(),"filter":filter,"limit":limit,
                            "variant":"B","status":"query_error",
                            "error":error.to_string(),"timing_sample":false,
                            "automatic_retry":false,
                        }));
                        panic!("retained failed baseline reference; no retry");
                    }
                };
                let shape = canonical_distance_shape(scope, &reference, limit);
                let reference_dto = dto(scope, &reference);
                let dto_success = reference_dto.is_ok();
                let reference_hash = fingerprint(&reference);
                let reference_identities = identities(&reference);
                let reference_f64 = shape.then(|| distance_bits(&reference, false));
                let reference_f32 = shape.then(|| distance_bits(&reference, true));
                emit(serde_json::json!({
                    "named_hydration_wide_reference":"completed",
                    "schema_version":1,"category_index":category_index,
                    "reference_index":reference_index,
                    "scope":scope.table(),"filter":filter,"limit":limit,
                    "variant":"B","status":"rows","rows":reference.len(),
                    "rows_equal_limit":reference.len() == limit,
                    "full_typed_payload":reference.iter().map(exact_value).collect::<Vec<_>>(),
                    "full_typed_payload_sha256":reference_hash,
                    "ordered_native_ids_and_raw_distance_Numbers":reference_identities,
                    "canonical_native_keys_and_distance_shape_valid":shape,
                    "native_f64_distance_bits":reference_f64,
                    "serving_f32_distance_bits":reference_f32,
                    "successful_public_DTO":dto_success,
                    "public_DTO_sha256":reference_dto.as_ref().ok().map(public_dto_hash),
                    "public_DTO_error":reference_dto.as_ref().err(),
                    "timing_sample":false,
                    "one_full_native_payload_only_no_duplicate_DTO_body":true,
                }));
                assert!(
                    shape && dto_success,
                    "retained malformed baseline reference"
                );
                let reference_dto = reference_dto.unwrap();
                let reference_f64 = reference_f64.unwrap();
                let reference_f32 = reference_f32.unwrap();
                references += 1;
                let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                for sample_index in 0..21 {
                    // Every variant appears seven times in every call slot.
                    for call_offset in 0..3 {
                        let variant = (sample_index + order_offset + call_offset) % 3;
                        let sequence = attempts;
                        attempts += 1;
                        emit(serde_json::json!({
                            "named_hydration_wide_attempt":"started",
                            "schema_version":1,"category_index":category_index,
                            "scope":scope.table(),"filter":filter,"limit":limit,
                            "variant":VARIANTS[variant],"sequence":sequence,
                            "sample_index":sample_index,"call_offset":call_offset,
                            "timing_sample":true,
                        }));
                        let started = std::time::Instant::now();
                        let outcome = run_query(
                            &db,
                            sqls[variant].clone(),
                            &embedding,
                            limit,
                            since.clone(),
                            source.clone(),
                        )
                        .await;
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let (rows, phases) = match outcome {
                            Ok(value) => value,
                            Err(error) => {
                                emit(serde_json::json!({
                                    "named_hydration_wide_attempt":"completed",
                                    "schema_version":1,"category_index":category_index,
                                    "scope":scope.table(),"filter":filter,"limit":limit,
                                    "variant":VARIANTS[variant],"sequence":sequence,
                                    "sample_index":sample_index,"call_offset":call_offset,
                                    "status":"query_error","elapsed_ms":elapsed_ms,
                                    "error":error.to_string(),"timing_sample":true,
                                    "automatic_retry":false,
                                }));
                                panic!("retained failed wide call; no retry");
                            }
                        };
                        // Native payload/key/Number/bits and public DTO work
                        // occurs after elapsed, exactly once per return.
                        let shape = canonical_distance_shape(scope, &rows, limit);
                        let actual_hash = fingerprint(&rows);
                        let actual_identities = identities(&rows);
                        let actual_f64 = shape.then(|| distance_bits(&rows, false));
                        let actual_f32 = shape.then(|| distance_bits(&rows, true));
                        let actual_dto = dto(scope, &rows);
                        let full_exact = actual_hash == reference_hash;
                        let key_number_exact = actual_identities == reference_identities;
                        let f64_exact = actual_f64.as_ref() == Some(&reference_f64);
                        let f32_exact = actual_f32.as_ref() == Some(&reference_f32);
                        let dto_exact = actual_dto.as_ref() == Ok(&reference_dto);
                        let exact = shape
                            && full_exact
                            && key_number_exact
                            && f64_exact
                            && f32_exact
                            && dto_exact;
                        emit(serde_json::json!({
                            "named_hydration_wide_attempt":"completed",
                            "schema_version":1,"category_index":category_index,
                            "scope":scope.table(),"filter":filter,"limit":limit,
                            "variant":VARIANTS[variant],"sequence":sequence,
                            "sample_index":sample_index,"call_offset":call_offset,
                            "status":"observed","elapsed_ms":elapsed_ms,"rows":rows.len(),
                            "rows_equal_limit":rows.len() == limit,
                            "full_typed_payload_sha256":actual_hash,
                            "ordered_native_ids_and_raw_distance_Numbers":actual_identities,
                            "canonical_native_keys_and_distance_shape_valid":shape,
                            "native_f64_distance_bits":actual_f64,
                            "serving_f32_distance_bits":actual_f32,
                            "full_rows_exact":full_exact,
                            "ordered_native_keys_and_raw_distance_Numbers_exact":key_number_exact,
                            "native_f64_distance_bits_exact":f64_exact,
                            "serving_f32_distance_bits_exact":f32_exact,
                            "successful_public_DTO":actual_dto.is_ok(),
                            "public_DTO_exact":dto_exact,
                            "full_payload_DTO_key_Number_F64_F32_exact":exact,
                            "phases":{"sdk_await_ms":phases.sdk_await_ms,
                                "native_Value_take_ms":phases.dto_take_ms,
                                "engine_execution_ms":phases.engine_execution_ms},
                            "timing_sample":true,
                            "public_DTO_and_fingerprint_checks_outside_elapsed":true,
                        }));
                        if !exact {
                            emit(serde_json::json!({
                                "named_hydration_wide_failure":"retained",
                                "sequence":sequence,"category_index":category_index,
                                "full_actual":rows.iter().map(exact_value).collect::<Vec<_>>(),
                                "full_reference":reference.iter().map(exact_value).collect::<Vec<_>>(),
                                "public_DTO_error":actual_dto.as_ref().err(),
                                "automatic_retry":false,
                            }));
                        }
                        assert!(exact, "retained wide payload/DTO/key/Number/bit mismatch");
                        assert!(elapsed_ms.is_finite() && elapsed_ms >= 0.0);
                        samples[variant].push(elapsed_ms);
                    }
                }
                assert!(samples.iter().all(|values| values.len() == 21));
                emit(serde_json::json!({
                    "named_hydration_wide_category":"completed",
                    "schema_version":1,"category_index":category_index,
                    "scope":scope.table(),"filter":filter,"limit":limit,
                    "first_and_20_warm_B_ms":samples[0],
                    "first_and_20_warm_M_ms":samples[1],
                    "first_and_20_warm_D_ms":samples[2],
                    "M_comparisons_descriptive_only":true,
                }));
                categories += 1;
            }
        }
    }
    assert_eq!(attempts, 1134);
    assert_eq!(references, 18);
    assert_eq!(categories, 18);
    let primary_after = assert_inventory(&db, count).await;
    let mirrors_after = mirror_inventory(&db, count).await;
    assert_eq!(
        primary, primary_after,
        "canonical full rows/vectors/typed keys unchanged"
    );
    assert_eq!(mirrors, mirrors_after, "all mirror rows unchanged");
    runtime.observe("named_hydration_wide", "after_fixture");
    emit(serde_json::json!({
        "named_hydration_wide":"observations_complete",
        "schema_version":1,"status":"passed_exactness",
        "backend":backend,"population_per_scope":count,
        "order":order,"order_offset":order_offset,
        "categories":categories,"completed_measured_attempt_pairs":attempts,
        "completed_unmeasured_B_references":references,
        "observations_per_variant_per_category":21,
        "primary_full_row_sha256_after":primary_after,
        "mirror_full_row_sha256_after":mirrors_after,
        "full_native_payload_DTO_key_Number_F64_F32_exact_every_measured_call":true,
        "successful_public_DTO_every_reference_and_measured_call":true,
        "exact_K_rows_every_reference_and_measured_call":true,
        "finite_numeric_native_and_serving_distances_every_reference_and_measured_call":true,
        "first_is_measured_after_one_B_reference_per_category_not_cold_or_first_use":true,
        "between_call_full_fingerprint_DTO_key_bit_work_declared":true,
        "outside_ledger_public_contract_queries":0,
        "M_comparisons_descriptive_only":true,
        "performance_acceptance":false,
        "production_coherence_or_adoption":false,
        "provider_calls":0,
    }));
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "root-owned B/M/D first-measured plus20warm matrix; no adoption"]
async fn named_field_wide_current_thread() {
    wide(
        QualificationRuntime::CurrentThread,
        "named-hydration-wide-current-thread",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "root-owned B/M/D first-measured plus20warm matrix; no adoption"]
async fn named_field_wide_two_worker() {
    wide(
        QualificationRuntime::TwoWorker,
        "named-hydration-wide-two-worker",
    )
    .await;
}
