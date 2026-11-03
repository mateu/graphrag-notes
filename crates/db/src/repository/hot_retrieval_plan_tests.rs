//! Provider-free access-path guards replay the SQL and native bindings actually
//! issued by Repository. The capture seam exists only in unit-test builds.
use super::graph_retrieval_batch_tests::mention_plan_operators;
use super::*;
use surrealdb_types::Variables;

type Capture = Arc<std::sync::Mutex<Vec<(String, Variables)>>>;

async fn fixture() -> (Repository, Capture, Vec<f32>) {
    let mut db = crate::init_memory().await.unwrap();
    let mut vector = vec![0.0f32; 1024];
    vector[0] = 1.0;
    db.query(r#"
        FOR $i IN 0..256 {
            LET $key = <string>$i;
            LET $note = type::record('note', $key);
            LET $entity = type::record('entity', $key);
            LET $conversation = type::record('conversation', $key);
            LET $message = type::record('message', $key);
            LET $text = IF $i = 0 THEN 'needle token' ELSE string::concat('fictional unrelated ', $key) END;
            CREATE $note SET content=$text, title=$text, embedding=$vector;
            CREATE $entity SET name=$text, canonical_name=$text, identity_key=string::concat('legacy:', $key), entity_type='technology', metadata={};
            CREATE type::record('mentions', $key) SET in=$note, out=$entity;
            CREATE $conversation SET uuid=$key,title=$text,summary=$text,summary_embedding=$vector,created_at=time::now(),updated_at=time::now();
            CREATE $message SET message_key=$key,conversation_id=$conversation,conversation_uuid=$key,message_index=0,role='user',content=$text,embedding=$vector;
            CREATE type::record('note_from_conversation', $key) SET in=$note,out=$conversation;
            CREATE type::record('note_from_message', $key) SET in=$note,out=$message;
            FOR $table IN ['supports','contradicts','related_to','derived_from'] {
                CREATE type::record($table, $key) SET in=$note,out=note:other,confidence=0.8f,is_manual=false;
            };
        };
        CREATE note:other SET content='Fictional destination';
    "#).bind(("vector", vector.clone())).await.unwrap().check().unwrap();
    let capture = Arc::new(std::sync::Mutex::new(Vec::new()));
    db.plan_capture = Some(capture.clone());
    (Repository::new(db), capture, vector)
}

fn take(capture: &Capture) -> Vec<(String, Variables)> {
    std::mem::take(&mut *capture.lock().unwrap())
}

async fn explain(repo: &Repository, sql: &str, variables: Variables) -> serde_json::Value {
    let plans: Vec<serde_json::Value> = repo
        .db
        .query(format!("{} EXPLAIN FULL", sql.trim().trim_end_matches(';')))
        .bind(variables)
        .await
        .unwrap()
        .take(0)
        .unwrap();
    assert_eq!(plans.len(), 1, "one execution plan: {sql}");
    println!(
        "hot-plan={}",
        serde_json::json!({"sql": sql, "plan": plans[0]})
    );
    plans.into_iter().next().unwrap()
}

fn indexed(plan: &serde_json::Value, table: &str, expected: &[&str]) -> bool {
    let operators = mention_plan_operators(plan);
    !operators
        .iter()
        .any(|op| op["operator"] == "TableScan" && op["attributes"]["table"] == table)
        && expected.iter().all(|name| {
            operators.iter().any(|op| {
                (op["operator"] == "IndexScan" || op["operator"] == "FullTextScan")
                    && op["attributes"]["index"] == *name
            })
        })
}

fn no_table_scan(plan: &serde_json::Value) -> bool {
    !mention_plan_operators(plan)
        .iter()
        .any(|op| op["operator"] == "TableScan")
}

#[tokio::test]
async fn indexed_hot_retrieval_statements_keep_their_access_paths() {
    let (repo, capture, _) = fixture().await;
    repo.fulltext_search_notes("needle token absent", 5, None, None)
        .await
        .unwrap();
    repo.fulltext_search_messages("needle", 5, None, None)
        .await
        .unwrap();
    repo.fulltext_search_conversation_summaries("needle", 5, None, None)
        .await
        .unwrap();
    repo.graph_notes_by_ids(&[RecordId::new("note", "0")], None, None)
        .await
        .unwrap();
    let calls = take(&capture);
    let mut checked = 0;
    for (sql, variables) in calls {
        if !sql.trim_start().starts_with("SELECT") {
            continue;
        }
        let plan = explain(&repo, &sql, variables.clone()).await;
        if sql.contains("FROM note_search") {
            assert!(
                indexed(
                    &plan,
                    "note_search",
                    &[
                        "idx_note_search_context",
                        "idx_note_search_content",
                        "idx_note_search_title"
                    ]
                ),
                "{plan}"
            );
            // Explain the actual lexical candidate SELECT independently, not
            // merely the enclosing hydration SELECT.
            let start = sql.find("SELECT id, record_id,").unwrap();
            let end = sql.rfind(')').unwrap();
            let inner = explain(&repo, &sql[start..end], variables).await;
            assert!(
                indexed(
                    &inner,
                    "note_search",
                    &[
                        "idx_note_search_context",
                        "idx_note_search_content",
                        "idx_note_search_title"
                    ]
                ),
                "{inner}"
            );
        } else if sql.contains("FROM message") {
            assert!(
                indexed(&plan, "message", &["idx_message_content"]),
                "{plan}"
            );
        } else if sql.contains("FROM conversation") {
            assert!(
                indexed(
                    &plan,
                    "conversation",
                    &["idx_conversation_summary", "idx_conversation_title"]
                ),
                "{plan}"
            );
        } else {
            assert!(
                sql.contains("FROM $note_ids"),
                "unclassified hot statement: {sql}"
            );
            assert!(
                no_table_scan(&plan),
                "direct record hydration must not scan: {plan}"
            );
            assert!(mention_plan_operators(&plan)
                .iter()
                .any(|op| op["operator"] == "SourceExpr"));
            assert_eq!(plan["total_rows"], 1);
        }
        assert!(no_table_scan(&plan), "{plan}");
        checked += 1;
    }
    assert_eq!(
        checked, 5,
        "both note tiers, messages, conversations, hydration"
    );
}

#[tokio::test]
async fn entity_lexeme_and_bound_mention_statements_reject_historical_scans() {
    let (repo, capture, _) = fixture().await;
    let note = RecordId::new("note", "0");
    for text in ["needle token", "explain needle token deployment", "need"] {
        let matches = repo
            .find_graph_entities_for_search(text, 5, std::slice::from_ref(&note), None, None)
            .await
            .unwrap();
        assert!(
            !matches.is_empty(),
            "representative tier must execute: {text}"
        );
    }
    repo.graph_notes_for_entities_ranked_for_query(
        &[RecordId::new("entity", "0")],
        std::slice::from_ref(&note),
        5,
        None,
        None,
        "needle token",
    )
    .await
    .unwrap();
    let mut checked_lexeme = 0;
    let mut checked_mentions = 0;
    let mut checked_hydration = 0;
    for (sql, variables) in take(&capture) {
        if sql.starts_with("RETURN array::concat") {
            // RETURN does not expose either child plan: explain each actual
            // nested SELECT independently with its original native bindings.
            for table in ["mention_lexeme", "entity_lexeme"] {
                let needle = format!("SELECT VALUE entity FROM {table}");
                let start = sql.find(&needle).unwrap();
                let end = start + sql[start..].find(')').unwrap();
                let plan = explain(&repo, &sql[start..end], variables.clone()).await;
                let index = format!("idx_{table}_lookup");
                assert!(indexed(&plan, table, &[&index]), "{plan}");
                checked_lexeme += 1;
            }
        } else if sql.starts_with("SELECT VALUE entity FROM entity_lexeme") {
            let plan = explain(&repo, &sql, variables).await;
            assert!(
                indexed(&plan, "entity_lexeme", &["idx_entity_lexeme_lookup"]),
                "{plan}"
            );
            checked_lexeme += 1;
        } else if sql.trim_start().starts_with("SELECT") && sql.contains("FROM mentions") {
            for statement in sql.split(';').filter(|s| !s.trim().is_empty()) {
                let plan = explain(&repo, statement, variables.clone()).await;
                assert!(
                    indexed(&plan, "mentions", &["idx_mentions_entity_note"]),
                    "{plan}"
                );
                checked_mentions += 1;
            }
        } else if sql.trim_start().starts_with("SELECT") && sql.contains("FROM $ids") {
            let plan = explain(&repo, &sql, variables.clone()).await;
            assert!(no_table_scan(&plan), "candidate entity hydration: {plan}");
            // The source projection is independently explained too.
            let start = sql
                .find("SELECT id, name, canonical_name, metadata,")
                .unwrap();
            let end = sql.find("FROM $ids)").unwrap() + "FROM $ids".len();
            let inner = explain(&repo, &sql[start..end], variables).await;
            assert!(no_table_scan(&inner), "{inner}");
            checked_hydration += 1;
        }
    }
    assert!(checked_lexeme >= 5, "exact+phrase children and prefix");
    assert!(
        checked_mentions >= 6,
        "candidate lookups plus alias/direct/fallback seeds"
    );
    assert_eq!(checked_hydration, 3);

    // Same guard, positive indexed entity/mentions paths, then de-indexed
    // entity corpus and genuinely correlated $parent.id association plans.
    let entity = RecordId::new("entity", "0");
    let mut vars = Variables::new();
    vars.insert("name", "needle token");
    vars.insert("entity", entity);
    let positive = explain(
        &repo,
        "SELECT id FROM entity WHERE canonical_name = $name",
        vars.clone(),
    )
    .await;
    assert!(indexed(&positive, "entity", &["idx_entity_canonical"]));
    let deindexed = explain(
        &repo,
        "SELECT id FROM entity WITH NOINDEX WHERE canonical_name = $name",
        vars.clone(),
    )
    .await;
    assert!(
        !indexed(&deindexed, "entity", &["idx_entity_canonical"]),
        "guard must reject #163 shape"
    );
    let positive = explain(
        &repo,
        "SELECT in FROM mentions WHERE out = $entity LIMIT 1",
        vars,
    )
    .await;
    assert!(indexed(
        &positive,
        "mentions",
        &["idx_mentions_entity_note"]
    ));
    #[derive(Deserialize, SurrealValue)]
    struct Correlated {
        plan: serde_json::Value,
    }
    let rows: Vec<Correlated> = repo.db.query("SELECT (SELECT in FROM mentions WHERE out = $parent.id LIMIT 1 EXPLAIN FULL) AS plan FROM entity:`0`").await.unwrap().take(0).unwrap();
    assert_eq!(rows.len(), 1);
    println!("historical-correlated-plan={}", rows[0].plan);
    assert!(
        !indexed(&rows[0].plan, "mentions", &["idx_mentions_entity_note"]),
        "guard must reject #164 shape"
    );
}

// Not a passing no-scan gate: these existing scans conflict with AC-1 while
// #179 explicitly forbids changing production queries/plans. Preserve the
// provider-free reproduction without normalizing them into green exceptions.
#[tokio::test]
#[ignore = "existing scan gap evidence; not an AC-1 no-scan guard"]
async fn existing_scan_gap_diagnostic() {
    let (repo, capture, vector) = fixture().await;
    repo.vector_search_notes(vector.clone(), 5, None, None)
        .await
        .unwrap();
    repo.vector_search_messages(vector.clone(), 5, None, None)
        .await
        .unwrap();
    repo.vector_search_conversation_summaries(vector, 5, None, None)
        .await
        .unwrap();
    let note = RecordId::new("note", "0");
    repo.graph_note_provenance_ids(std::slice::from_ref(&note))
        .await
        .unwrap();
    let tables = ["supports", "contradicts", "related_to", "derived_from"].map(String::from);
    for (outbound, inbound) in [(true, false), (false, true), (true, true)] {
        repo.graph_note_edges_excluding_visited(
            std::slice::from_ref(&note),
            &tables,
            5,
            outbound,
            inbound,
            0.0,
            None,
            None,
            &HashMap::new(),
        )
        .await
        .unwrap();
    }
    for (sql, variables) in take(&capture) {
        for statement in sql.split(';').filter(|s| !s.trim().is_empty()) {
            let _ = explain(&repo, statement, variables.clone()).await;
        }
    }
}
