//! Fictional pinned-engine FTS projection experiments; rankings remain exact.
use super::*;
use crate::init_memory;

fn projection() -> &'static str {
    "id, title, content, note_type, tags, created_at, source_id.uri AS source_uri"
}

fn predicates() -> &'static str {
    "(search_content @0@ $query OR content @1@ $query OR title @2@ $query) \
     AND ($since = NONE OR created_at >= <datetime>$since) \
     AND ($source_uri = NONE OR source_id.uri = $source_uri) \
     AND (source_id IS NONE OR source_generation IS NONE \
          OR source_generation = source_id.successful_generation)"
}

fn exact_title() -> &'static str {
    "($normalized_title != '' AND string::lowercase(string::trim(title ?? '')) = $normalized_title)"
}

fn score() -> &'static str {
    "(search::score(0) * 0.7 + search::score(1) * 0.2 + search::score(2) * 0.1)"
}

fn old_query() -> String {
    format!(
        "SELECT {}, {} AS exact_title_match, {} AS fts_score FROM note WHERE {} \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC LIMIT $limit",
        projection(),
        exact_title(),
        score(),
        predicates()
    )
}

fn candidate_query() -> String {
    format!(
        "SELECT id, {} AS exact_title_match, {} AS fts_score FROM note WHERE {} \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC LIMIT $limit",
        exact_title(),
        score(),
        predicates()
    )
}

fn narrow_query() -> String {
    format!(
        "SELECT id, id.title AS title, id.content AS content, id.note_type AS note_type, \
         id.tags AS tags, id.created_at AS created_at, id.source_id.uri AS source_uri, \
         exact_title_match, fts_score FROM ({}) \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
        candidate_query()
    )
}

fn once_hydrated_query() -> String {
    format!(
        "SELECT note.id AS id, note.title AS title, note.content AS content, \
         note.note_type AS note_type, note.tags AS tags, note.created_at AS created_at, \
         note.source_id.uri AS source_uri, exact_title_match, fts_score FROM \
         (SELECT id.* AS note, exact_title_match, fts_score FROM ({})) \
         ORDER BY exact_title_match DESC, fts_score DESC, id ASC",
        candidate_query()
    )
}

async fn query(db: &DbConnection, sql: &str, text: &str, limit: usize) -> Vec<SearchResult> {
    db.query(sql.to_owned())
        .bind(("query", text.to_owned()))
        .bind(("normalized_title", text.trim().to_lowercase()))
        .bind(("limit", limit))
        .bind(("since", Option::<String>::None))
        .bind(("source_uri", Option::<String>::None))
        .await
        .unwrap()
        .take(0)
        .unwrap()
}

#[tokio::test]
#[ignore = "opt-in broad/narrow FTS plan and projection experiment; no hardware CI budget"]
async fn fulltext_plan_and_projection_diagnostic() {
    let db = init_memory().await.unwrap();
    for n in 0..256 {
        let mut vector = vec![0.0; 1024];
        vector[0] = 1.0;
        let term = if n % 5 == 0 { "quasar" } else { "unrelated" };
        let note = Note::new(format!(
            "quasar unique{n} {}",
            "fictional body ".repeat(512)
        ))
        .with_title(if n == 255 {
            "quasar".to_owned()
        } else {
            format!("{term} fixture{n}")
        })
        .with_embedding(vector);
        let mut note = note;
        note.search_content = Some(format!(
            "quasar unique{n} {}",
            "fictional searchable text ".repeat(256)
        ));
        let _: Option<Note> = db
            .create(("note", format!("fixture-{n:04}")))
            .content(note)
            .await
            .unwrap();
    }
    let mut runs = Vec::new();
    for text in ["quasar", "unique123", "absent fictional constellation"] {
        for limit in [5, 50] {
            let baseline = query(&db, &old_query(), text, limit).await;
            for (name, sql) in [
                ("baseline", old_query()),
                ("narrow", narrow_query()),
                ("once_hydrated", once_hydrated_query()),
            ] {
                let mut samples = Vec::new();
                let mut exact = true;
                let mut score_bits_exact = true;
                for _ in 0..21 {
                    let started = std::time::Instant::now();
                    let rows = query(&db, &sql, text, limit).await;
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    exact &= serde_json::to_value(&rows).unwrap()
                        == serde_json::to_value(&baseline).unwrap();
                    score_bits_exact &= rows
                        .iter()
                        .map(|row| row.fts_score.map(f32::to_bits))
                        .collect::<Vec<_>>()
                        == baseline
                            .iter()
                            .map(|row| row.fts_score.map(f32::to_bits))
                            .collect::<Vec<_>>();
                }
                let plan: Vec<serde_json::Value> = db
                    .query(format!("{sql} EXPLAIN FULL"))
                    .bind(("query", text.to_owned()))
                    .bind(("normalized_title", text.to_lowercase()))
                    .bind(("limit", limit))
                    .bind(("since", Option::<String>::None))
                    .bind(("source_uri", Option::<String>::None))
                    .await
                    .unwrap()
                    .take(0)
                    .unwrap();
                runs.push(serde_json::json!({"variant":name,"query":text,"limit":limit,"rows":baseline.len(),
                    "full_rows_exact":exact,"score_bits_exact":score_bits_exact,"first_ms":samples[0],
                    "warm_ms":samples[1..],"plan":plan}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"fictional_records":256,"provider_calls":0,
        "engine":"surrealdb-core 3.2.4","runs":runs})
    );
}
