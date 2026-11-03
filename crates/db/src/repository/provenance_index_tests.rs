//! Replay actual Repository SQL, not a duplicated membership query.
use super::graph_retrieval_batch_tests::mention_plan_operators;
use super::*;
use surrealdb_types::{Object, Variables};

type Capture = Arc<std::sync::Mutex<Vec<(String, Variables)>>>;

async fn populate(db: &DbConnection) {
    db.query(r#"
        FOR $i IN 0..512 {
            LET $key = <string>$i;
            LET $note = type::record('note', $key);
            LET $conversation = type::record('conversation', $key);
            LET $message = type::record('message', $key);
            CREATE $note SET content=string::concat('Fictional note ', $key);
            CREATE $conversation SET uuid=$key,title=$key,created_at=time::now(),updated_at=time::now();
            CREATE $message SET message_key=$key,conversation_id=$conversation,conversation_uuid=$key,message_index=0,role='user',content=$key;
            CREATE type::record('note_from_conversation', $key) SET in=$note,out=$conversation;
            CREATE type::record('note_from_message', $key) SET in=$note,out=$message;
        };
        -- Numeric IDs must not collapse into same-looking string IDs. The
        -- endpoints need not exist: provenance returns IDs, not hydrated rows.
        CREATE note_from_conversation:numeric SET in=note:7,out=conversation:7;
        CREATE note_from_message:numeric SET in=note:7,out=message:7;
        CREATE note_from_conversation:missing SET in=note:missing,out=conversation:missing;
        CREATE note_from_message:missing SET in=note:missing,out=message:missing;
        -- Non-unique indexes must retain repeated and one-to-many provenance.
        CREATE note_from_conversation:duplicate SET in=note:⟨7⟩,out=conversation:⟨7⟩;
        CREATE note_from_message:extra SET in=note:⟨7⟩,out=message:absent;
    "#).await.unwrap().check().unwrap();
}

async fn records(db: &DbConnection) -> Vec<Vec<Object>> {
    let mut rows = Vec::new();
    for table in [
        "note",
        "conversation",
        "message",
        "note_from_conversation",
        "note_from_message",
    ] {
        rows.push(
            db.query(format!("SELECT * FROM {table} ORDER BY id"))
                .await
                .unwrap()
                .take(0)
                .unwrap(),
        );
    }
    rows
}

fn expected() -> HashMap<String, Vec<String>> {
    let mut result = HashMap::<String, Vec<String>>::new();
    for (note, mut endpoints) in [
        (
            RecordId::new("note", "0"),
            vec![
                RecordId::new("conversation", "0"),
                RecordId::new("message", "0"),
            ],
        ),
        (
            RecordId::new("note", "7"),
            vec![
                RecordId::new("conversation", "7"),
                RecordId::new("message", "7"),
                RecordId::new("message", "absent"),
            ],
        ),
        (
            RecordId::new("note", 7),
            vec![
                RecordId::new("conversation", 7),
                RecordId::new("message", 7),
            ],
        ),
        (
            RecordId::new("note", "missing"),
            vec![
                RecordId::new("conversation", "missing"),
                RecordId::new("message", "missing"),
            ],
        ),
    ] {
        endpoints.sort_by_key(record_id_to_string);
        result
            .entry(record_id_to_string(&note))
            .or_default()
            .extend(endpoints.iter().map(record_id_to_string));
    }
    // Preserve the API's existing textual serialization and deduplication;
    // native key distinctions are checked independently below.
    for endpoints in result.values_mut() {
        endpoints.sort();
        endpoints.dedup();
    }
    result
}

fn indexed(plan: &serde_json::Value, index: &str) -> bool {
    let operators = mention_plan_operators(plan);
    !operators.iter().any(|op| op["operator"] == "TableScan")
        && operators
            .iter()
            .any(|op| op["operator"] == "IndexScan" && op["attributes"]["index"] == index)
}

async fn check(repo: &Repository, capture: &Capture, indexed_expected: bool, phase: &str) {
    // Empty requests must remain query-free.
    assert!(repo
        .graph_note_provenance_ids(&[])
        .await
        .unwrap()
        .is_empty());
    assert!(capture.lock().unwrap().is_empty());
    let ids = vec![
        RecordId::new("note", "0"),
        RecordId::new("note", "7"),
        RecordId::new("note", 7),
        RecordId::new("note", "missing"),
        RecordId::new("note", "no_provenance"),
        RecordId::new("note", "7"),
    ];
    assert_eq!(
        repo.graph_note_provenance_ids(&ids).await.unwrap(),
        expected()
    );
    let calls = std::mem::take(&mut *capture.lock().unwrap());
    assert_eq!(calls.len(), 1, "one batched Repository request");
    let (sql, variables) = &calls[0];
    let statements: Vec<_> = sql
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(
        statements.len(),
        2,
        "each actual statement must be explained"
    );
    for (statement, table) in statements
        .iter()
        .zip(["note_from_conversation", "note_from_message"])
    {
        assert!(statement.contains(&format!("FROM {table} ")));
        let plans: Vec<serde_json::Value> = repo
            .db
            .query(format!("{statement} EXPLAIN FULL"))
            .bind(variables.clone())
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        let index = format!("idx_{table}_in");
        println!(
            "provenance-plan={}",
            serde_json::json!({"phase":phase,"sql":statement,"plan":plan})
        );
        assert_eq!(indexed(plan, &index), indexed_expected, "{phase}: {plan}");
        if !indexed_expected {
            assert!(mention_plan_operators(plan)
                .iter()
                .any(|op| op["operator"] == "TableScan" && op["attributes"]["table"] == table));
        }
    }
    // A numeric-only request must not pick up the string note's extra edge.
    for (key, extra) in [
        (RecordId::new("note", 7), false),
        (RecordId::new("note", "7"), true),
    ] {
        let actual = repo.graph_note_provenance_ids(&[key]).await.unwrap();
        let mut endpoints = vec!["conversation:7".to_string(), "message:7".to_string()];
        if extra {
            endpoints.push("message:absent".to_string());
        }
        assert_eq!(actual, HashMap::from([("note:7".to_string(), endpoints)]));
    }
    assert!(repo
        .graph_note_provenance_ids(&[RecordId::new("note", "no_provenance")])
        .await
        .unwrap()
        .is_empty());
    capture.lock().unwrap().clear();
}

#[tokio::test]
async fn provenance_membership_indexes_preserve_upgrade_and_fresh_results() {
    // A real v25 fixture ensures existing rows are indexed by the migration,
    // rather than by inserting them after the index already exists.
    let mut db = crate::migrations::test_support::init_memory_at_version(25)
        .await
        .unwrap();
    populate(&db).await;
    let before = records(&db).await;
    let capture: Capture = Arc::new(std::sync::Mutex::new(Vec::new()));
    db.plan_capture = Some(capture.clone());
    let repo = Repository::new(db);
    check(&repo, &capture, false, "v25-negative-control").await;
    crate::migrations::apply_all(&repo.db).await.unwrap();
    crate::migrations::apply_all(&repo.db).await.unwrap();
    assert_eq!(
        records(&repo.db).await,
        before,
        "migration/reinitialization must not rewrite records"
    );
    capture.lock().unwrap().clear();
    check(&repo, &capture, true, "upgraded").await;
    // Exercise ordinary writes after upgrade as well as migration backfill.
    repo.db
        .query("CREATE note_from_message:after_upgrade SET in=note:⟨0⟩,out=message:⟨0⟩;")
        .await
        .unwrap()
        .check()
        .unwrap();
    capture.lock().unwrap().clear();
    check(&repo, &capture, true, "post-upgrade-write").await;

    let mut db = crate::init_memory().await.unwrap();
    populate(&db).await;
    let capture: Capture = Arc::new(std::sync::Mutex::new(Vec::new()));
    db.plan_capture = Some(capture.clone());
    check(&Repository::new(db), &capture, true, "fresh").await;
}
