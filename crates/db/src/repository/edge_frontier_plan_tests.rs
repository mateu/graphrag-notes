//! Accepted-edge access paths, independent of endpoint hydration qualification.
use super::graph_retrieval_batch_tests::mention_plan_operators;
use super::*;

#[tokio::test]
async fn accepted_edge_frontiers_use_indexes_in_every_direction() {
    // Populate the pre-index schema first: migration DDL must backfill existing edges.
    let db = crate::migrations::test_support::init_memory_at_version(25)
        .await
        .unwrap();
    db.query(r#"
        FOR $i IN 0..256 {
            LET $note = type::record('note', <string>$i);
            CREATE $note SET content='Fictional frontier';
            FOR $table IN ['supports','contradicts','related_to','derived_from'] {
                CREATE type::record($table, <string>$i) SET in=IF $i % 2 = 0 THEN $note ELSE note:other END, out=IF $i % 2 = 0 THEN note:other ELSE $note END, confidence=0.8f,is_manual=false;
            };
        };
        CREATE note:other SET content='Fictional neighbor';
    "#).await.unwrap().check().unwrap();
    crate::migrations::apply_all(&db).await.unwrap();
    crate::migrations::apply_all(&db).await.unwrap();
    assert_eq!(
        crate::migrations::current_version(&db).await.unwrap(),
        crate::migrations::latest_version()
    );
    let mut evidence = Vec::new();
    for table in ["supports", "contradicts", "related_to", "derived_from"] {
        for (outbound, inbound) in [(true, false), (false, true), (true, true)] {
            let sql = graph::graph_edge_batch_sql(1, &[table], outbound, inbound);
            let plans: Vec<serde_json::Value> = db
                .query(format!("{} EXPLAIN FULL", sql.trim_end_matches(';')))
                .bind(("note_0", RecordId::new("note", "other")))
                .bind(("visited_0", Vec::<RecordId>::new()))
                .bind(("min_confidence", 0.0f32))
                .bind(("since", Option::<String>::None))
                .bind(("source_uri", Option::<String>::None))
                .bind(("limit", 5i64))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert_eq!(
                plans[0]["total_rows"], 5,
                "backfilled edge indexes must find a full page"
            );
            let operators = mention_plan_operators(&plans[0]);
            assert!(
                !operators.iter().any(|op| op["operator"] == "TableScan"),
                "{table}/{outbound}/{inbound}: {plans:?}"
            );
            for endpoint in [outbound.then_some("in"), inbound.then_some("out")]
                .into_iter()
                .flatten()
            {
                let expected = format!("idx_{table}_{endpoint}");
                assert!(
                    operators
                        .iter()
                        .any(|op| op["operator"] == "IndexScan"
                            && op["attributes"]["index"] == expected),
                    "{expected}: {plans:?}"
                );
            }
            evidence.push(serde_json::json!({"table":table,"outbound":outbound,"inbound":inbound,"sql":sql,"plan":plans}));
        }
    }
    // EXPLAIN does not expose the internal record fetches in endpoint.*.
    // Qualify a bound endpoint independently, never treating the outer edge
    // index as proof of endpoint hydration's access path.
    let endpoint_sql = format!(
        "SELECT id FROM $endpoint WHERE {} EXPLAIN FULL",
        graph::graph_endpoint_eligible_sql("id")
    );
    let endpoint_plan: Vec<serde_json::Value> = db
        .query(&endpoint_sql)
        .bind(("endpoint", RecordId::new("note", "other")))
        .bind(("since", Option::<String>::None))
        .bind(("source_uri", Option::<String>::None))
        .await
        .unwrap()
        .take(0)
        .unwrap();
    let operators = mention_plan_operators(&endpoint_plan[0]);
    assert!(!operators.iter().any(|op| op["operator"] == "TableScan"));
    assert!(
        operators.iter().any(|op| op["operator"] == "RecordIdScan"),
        "{endpoint_plan:?}"
    );
    assert_eq!(endpoint_plan[0]["total_rows"], 1);
    evidence.push(serde_json::json!({"qualification":"bound_endpoint_hydration", "sql":endpoint_sql,"plan":endpoint_plan}));
    println!(
        "edge-frontier-plans={}",
        serde_json::to_string(&evidence).unwrap()
    );
}

/// Independent scan access path catches index unions duplicating self loops,
/// dropping a direction, or coercing native endpoint/relation IDs. The legacy
/// note-membership differential fixture separately guards eligibility semantics.
#[tokio::test]
async fn indexed_edge_frontiers_match_scans_for_native_ids_and_self_loops() {
    let db = crate::init_memory().await.unwrap();
    db.query(r#"
        CREATE note:42 SET content='Integer';
        CREATE note:`42` SET content='String';
        CREATE note:['compound',42] SET content='Array';
        FOR $table IN ['supports','contradicts','related_to','derived_from'] {
            CREATE type::record($table, 42) SET in=note:42,out=note:`42`,confidence=0.8f,is_manual=false;
            CREATE type::record($table, '42') SET in=note:`42`,out=note:42,confidence=0.8f,is_manual=false;
            CREATE type::record($table, ['compound',42]) SET in=note:42,out=note:42,confidence=NONE,is_manual=false;
            CREATE type::record($table, {id:'compound'}) SET in=note:['compound',42],out=note:42,confidence=0.8f,is_manual=false;
        };
    "#).await.unwrap().check().unwrap();
    for table in ["supports", "contradicts", "related_to", "derived_from"] {
        for (outbound, inbound) in [(true, false), (false, true), (true, true)] {
            for limit in [1i64, 5] {
                for seed in [RecordId::new("note", 42i64), RecordId::new("note", "42")] {
                    for visited in [Vec::new(), vec![RecordId::new("note", 42i64)]] {
                        let sql = graph::graph_edge_batch_sql(1, &[table], outbound, inbound);
                        let scan = sql.replace(
                            &format!("FROM {table} WHERE"),
                            &format!("FROM {table} WITH NOINDEX WHERE"),
                        );
                        let mut results = Vec::new();
                        for statement in [&sql, &scan] {
                            let rows: Vec<NoteEdgeRow> = db
                                .query(statement)
                                .bind(("note_0", seed.clone()))
                                .bind(("visited_0", visited.clone()))
                                .bind(("min_confidence", 0.5f32))
                                .bind(("since", Option::<String>::None))
                                .bind(("source_uri", Option::<String>::None))
                                .bind(("limit", limit))
                                .await
                                .unwrap()
                                .check()
                                .unwrap()
                                .take(0)
                                .unwrap();
                            results.push(serde_json::to_value(rows).unwrap());
                        }
                        assert_eq!(
                            results[0], results[1],
                            "{table}/{outbound}/{inbound}/{limit}/{seed:?}/{visited:?}"
                        );
                    }
                }
            }
        }
    }
}
