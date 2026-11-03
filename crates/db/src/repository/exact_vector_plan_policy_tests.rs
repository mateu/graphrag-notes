//! Deliberate exact-vector scan exception (#191), not an indexed-plan guard.
//! Capture actual Repository SQL and fail closed on a changed access path.
use super::graph_retrieval_batch_tests::mention_plan_operators;
use super::*;
use chrono::TimeZone;
use surrealdb_types::Variables;

fn exact_scan(plan: &serde_json::Value, table: &str, field: &str, k: usize) -> bool {
    let ops = mention_plan_operators(plan);
    let scans: Vec<_> = ops
        .iter()
        .filter(|op| op["operator"] == "TableScan")
        .collect();
    let knn: Vec<_> = ops
        .iter()
        .filter(|op| op["operator"] == "KnnTopK")
        .collect();
    scans.len() == 1
        && scans[0]["attributes"]["table"] == table
        && knn.len() == 1
        && knn[0]["attributes"]["field"] == field
        && knn[0]["attributes"]["distance"] == "Cosine"
        && knn[0]["attributes"]["dimension"] == "1024"
        && knn[0]["attributes"]["k"]
            .as_str()
            .and_then(|s| s.parse::<usize>().ok())
            == Some(k)
        && knn[0]["children"].as_array().is_some_and(|children| {
            children.len() == 1
                && children[0]["operator"] == "TableScan"
                && children[0]["attributes"]["table"] == table
        })
        && !ops.iter().any(|op| {
            matches!(
                op["operator"].as_str(),
                Some("IndexScan" | "FullTextScan" | "AnnTopK")
            )
        })
}

#[tokio::test]
async fn exact_vector_scan_exception_preserves_plans_and_filtered_top_k() {
    let mut db = crate::init_memory().await.unwrap();
    let mut vector = vec![0.0f32; 1024];
    vector[0] = 1.0;
    // Unique distances: row i has cosine distance 1 - 1/sqrt(1+(i/256)^2).
    // Closer rows deliberately fail filters; an ANN/global-top-k-then-filter
    // substitution cannot satisfy these independently calculated eligible IDs.
    db.query(r#"
        FOR $s IN 0..3 {
            CREATE type::record('source', <string>$s) SET uri=string::concat('fixture://', <string>$s), successful_generation=2;
        };
        FOR $i IN 0..256 {
            LET $key = <string>$i;
            LET $time = IF $i % 7 = 0 THEN d'2030-01-01T00:00:00Z' ELSE d'2030-01-03T00:00:00Z' END;
            LET $source = IF $i % 17 = 0 THEN NONE ELSE type::record('source', <string>($i % 3)) END;
            LET $generation = IF $i % 13 = 0 THEN NONE ELSE (IF $i % 11 = 0 THEN 1 ELSE 2 END) END;
            LET $v = array::concat([1.0f, <float>$i / 256.0f], array::repeat(0.0f, 1022));
            LET $conversation = type::record('conversation', $key);
            CREATE type::record('note', $key) SET title=$key, content=$key, embedding=$v, created_at=$time, source_id=$source, source_generation=$generation;
            CREATE $conversation SET uuid=$key, title=$key, summary=$key, summary_embedding=$v, source_uri=string::concat('fixture://', <string>($i % 3)), created_at=$time, updated_at=$time;
            CREATE type::record('message', $key) SET message_key=$key, conversation_id=$conversation, conversation_uuid=$key, message_index=0, role='user', content=$key, embedding=$v, created_at=IF $i % 19 = 0 THEN NONE ELSE $time END;
        };
    "#).await.unwrap().check().unwrap();
    let capture = Arc::new(std::sync::Mutex::new(Vec::<(String, Variables)>::new()));
    db.plan_capture = Some(capture.clone());
    let repo = Repository::new(db);
    let mut checked = 0;
    for k in [1, 5, 17] {
        for (recent, source) in [(false, false), (true, false), (false, true), (true, true)] {
            let since = recent.then(|| Utc.with_ymd_and_hms(2030, 1, 2, 0, 0, 0).unwrap());
            let uri = source.then(|| "fixture://1".to_string());
            let notes = repo
                .vector_search_notes(vector.clone(), k, since, uri.clone())
                .await
                .unwrap();
            let messages = repo
                .vector_search_messages(vector.clone(), k, since, uri.clone())
                .await
                .unwrap();
            let conversations = repo
                .vector_search_conversation_summaries(vector.clone(), k, since, uri)
                .await
                .unwrap();
            let results = [
                notes
                    .into_iter()
                    .map(|r| (r.id, r.vec_distance.unwrap()))
                    .collect::<Vec<_>>(),
                messages
                    .into_iter()
                    .map(|r| (r.id, r.vec_distance.unwrap()))
                    .collect(),
                conversations
                    .into_iter()
                    .map(|r| (r.id, r.vec_distance.unwrap()))
                    .collect(),
            ];
            let calls = std::mem::take(&mut *capture.lock().unwrap());
            assert_eq!(calls.len(), 3, "exactly the three production SELECTs");
            for ((table, field), (rows, (sql, vars))) in [
                ("note", "embedding"),
                ("message", "embedding"),
                ("conversation", "summary_embedding"),
            ]
            .into_iter()
            .zip(results.into_iter().zip(calls))
            {
                let expected: Vec<_> = (0..256)
                    .filter(|i| {
                        (!recent || i % 7 != 0)
                            && (!source || (i % 3 == 1 && (table != "note" || i % 17 != 0)))
                            && (table != "message" || !recent || i % 19 != 0)
                            && (table != "note" || i % 17 == 0 || i % 13 == 0 || i % 11 != 0)
                    })
                    .take(k)
                    .collect();
                assert_eq!(
                    rows.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|i| RecordId::new(table, i.to_string()))
                        .collect::<Vec<_>>(),
                    "{table} k={k} recent={recent} source={source}"
                );
                for ((_, distance), i) in rows.iter().zip(expected) {
                    let x = f64::from(i) / 256.0;
                    let expected_distance = (1.0 - 1.0 / (1.0 + x * x).sqrt()) as f32;
                    assert!(
                        (distance - expected_distance).abs() < 1e-7,
                        "independent cosine distance: {distance} vs {expected_distance}"
                    );
                }
                assert!(
                    sql.contains(&format!("{field} <|{k},COSINE|> $embedding")),
                    "no silent approximate operator: {sql}"
                );
                let plans: Vec<serde_json::Value> = repo
                    .db
                    .query(format!("{} EXPLAIN FULL", sql.trim()))
                    .bind(vars)
                    .await
                    .unwrap()
                    .take(0)
                    .unwrap();
                assert_eq!(plans.len(), 1);
                let plan = &plans[0];
                assert!(
                    exact_scan(plan, table, field, k),
                    "review exact-vector exception before accepting changed plan: {plan}"
                );
                // The same predicate rejects an indexed substitution or metric drift.
                let mut wrong_distance = plan.clone();
                fn replace(value: &mut serde_json::Value, from: &str, to: &str) {
                    match value {
                        serde_json::Value::String(s) if s == from => *s = to.to_string(),
                        serde_json::Value::Object(o) => {
                            o.values_mut().for_each(|v| replace(v, from, to))
                        }
                        serde_json::Value::Array(a) => {
                            a.iter_mut().for_each(|v| replace(v, from, to))
                        }
                        _ => {}
                    }
                }
                replace(&mut wrong_distance, "Cosine", "Euclidean");
                assert!(!exact_scan(&wrong_distance, table, field, k));
                let mut wrong_scan = plan.clone();
                replace(&mut wrong_scan, "TableScan", "IndexScan");
                assert!(!exact_scan(&wrong_scan, table, field, k));
                println!(
                    "exact-vector-policy={}",
                    serde_json::json!({"table":table,"k":k,"recent":recent,"source":source,"sql":sql,"plan":plan,"returned_ids":rows.iter().map(|(id,_)|format!("{id:?}")).collect::<Vec<_>>() })
                );
                checked += 1;
            }
            // Ignore replay observations; the next iteration captures only calls.
            capture.lock().unwrap().clear();
        }
    }
    assert_eq!(checked, 36);
}
