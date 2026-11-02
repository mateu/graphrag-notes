//! Sequential, warm matcher benchmark. Only accepts an issue166 backup copy.
use graphrag_db::{init_persistent, Repository};
use std::time::Instant;
#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::fs::canonicalize(&args[1]).unwrap();
    assert!(path
        .to_string_lossy()
        .contains("/.graphrag/shared/tmp/issue166-"));
    let start = Instant::now();
    let db = init_persistent(&path).await.unwrap();
    eprintln!("migration_open_ms={}", start.elapsed().as_millis());
    if args.get(2).is_some_and(|s| s == "--sql") {
        let sql = std::fs::read_to_string(&args[3]).unwrap();
        let result = db.query(sql).await.unwrap().check().unwrap();
        println!("{result:?}");
        return;
    }
    let repo = Repository::new(db.clone());
    let queries = [
        "OpenClaw",
        "GraphRAG",
        "Proxmox",
        "how does openclaw work",
        "how does memory work",
        "main",
        "index",
        "openc",
    ];
    for query in queries {
        let normalized = query.to_lowercase();
        let mut grams = Vec::new();
        let words: Vec<_> = normalized.split_whitespace().collect();
        for i in 0..words.len() {
            for j in i + 1..=words.len() {
                grams.push(words[i..j].join(" "));
            }
        }
        let mut counts = db.query("RETURN array::len(array::distinct(SELECT VALUE entity FROM entity_lexeme WHERE lexeme = $q)); RETURN array::len(array::distinct(SELECT VALUE entity FROM entity_lexeme WHERE lexeme IN $grams)); RETURN array::len(array::distinct(SELECT VALUE entity FROM entity_lexeme WHERE lexeme >= $q AND lexeme < $upper));")
            .bind(("q", normalized.clone())).bind(("grams", grams)).bind(("upper", format!("{normalized}\u{10ffff}"))).await.unwrap().check().unwrap();
        let counts: Vec<i64> = (0..3)
            .map(|i| counts.take::<Option<i64>>(i).unwrap().unwrap())
            .collect();
        let mut samples = Vec::new();
        let mut expected = None;
        for iteration in 0..23 {
            let start = Instant::now();
            let rows = repo
                .find_graph_entities_for_search(query, 20, &[], None, None)
                .await
                .unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let value = serde_json::to_value(rows).unwrap();
            if let Some(previous) = &expected {
                assert_eq!(previous, &value);
            }
            expected = Some(value);
            if iteration >= 3 {
                samples.push(ms);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"query":query,"candidate_counts_exact_contained_prefix":counts,"p50_ms":samples[9],"p95_ms":samples[18],"samples_ms":samples,"results":expected})
        );
    }
}
