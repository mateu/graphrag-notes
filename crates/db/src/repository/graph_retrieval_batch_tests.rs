//! Differential regressions for bounded graph queries. All records are
//! fictional; legacy SQL remains an independent visibility/ordering oracle.

use super::graph::graph_endpoint_eligible_sql;
use super::*;
use crate::init_memory;
use chrono::TimeZone;
use std::ops::Bound;
use surrealdb::types::{Object, RecordIdKey, RecordIdKeyRange};

#[derive(Deserialize, SurrealValue)]
struct EndpointComparison {
    legacy_visible: bool,
    fetched_visible: bool,
    legacy_eligible: bool,
    fetched_eligible: bool,
}

fn timestamp() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 20, 12, 0, 0).unwrap()
}

fn compound_note(label: &str) -> RecordId {
    let mut key = Object::new();
    key.insert("id", label.to_string());
    key.insert("source_generation", 99i64);
    // This key component must never replace the stored source reference.
    key.insert("source_id", RecordId::new("source", "key-only-source"));
    RecordId::new("note", RecordIdKey::Object(key))
}

fn note_range() -> RecordId {
    RecordId::new(
        "note",
        RecordIdKeyRange {
            start: Bound::Included(RecordIdKey::String("a".into())),
            end: Bound::Excluded(RecordIdKey::String("zzzz".into())),
        },
    )
}

async fn source(repo: &Repository, key: &str, uri: &str) -> RecordId {
    let id = RecordId::new("source", key);
    repo.db
        .query("CREATE $id SET uri = $uri, generation = 2, successful_generation = 1, status = 'failed'")
        .bind(("id", id.clone()))
        .bind(("uri", uri.to_string()))
        .await
        .unwrap()
        .check()
        .unwrap();
    id
}

async fn note(
    repo: &Repository,
    id: RecordId,
    source: Option<RecordId>,
    generation: Option<i64>,
    created_at: DateTime<Utc>,
) -> RecordId {
    repo.db
        .query("CREATE $id SET content = 'Fictional Orion fixture', source_id = $source, source_generation = $generation, created_at = <datetime>$created_at")
        .bind(("id", id.clone()))
        .bind(("source", source))
        .bind(("generation", generation))
        .bind(("created_at", created_at.to_rfc3339()))
        .await
        .unwrap()
        .check()
        .unwrap();
    id
}

async fn edge(
    repo: &Repository,
    table: &str,
    key: &str,
    from: &RecordId,
    to: &RecordId,
    confidence: Option<f32>,
) -> RecordId {
    let id = RecordId::new(table, key);
    repo.db
        .query("CREATE $id SET in = $from, out = $to, confidence = $confidence")
        .bind(("id", id.clone()))
        .bind(("from", from.clone()))
        .bind(("to", to.clone()))
        .bind(("confidence", confidence))
        .await
        .unwrap()
        .check()
        .unwrap();
    id
}

async fn mention(repo: &Repository, note_id: &RecordId, entity_id: &RecordId) {
    repo.db
        .query("CREATE mentions SET in = $note, out = $entity")
        .bind(("note", note_id.clone()))
        .bind(("entity", entity_id.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
}

#[tokio::test]
async fn fetched_endpoint_visibility_matches_note_membership_for_typed_and_dangling_ids() {
    let repo = Repository::new(init_memory().await.unwrap());
    let current_source = source(&repo, "orion", "fixture://orion").await;
    let missing_source = RecordId::new("source", "missing");
    let compound = compound_note("existing");
    let deleted_compound = compound_note("deleted");
    let deleted_string = RecordId::new("note", "deleted-string");
    let missing_compound = compound_note("never-created");
    let integer = RecordId::new("note", 42i64);
    let uuid = RecordId::new("note", surrealdb_types::Uuid::new_v4());
    let mut cases = vec![
        (RecordId::new("note", "manual"), None, None, true),
        (
            RecordId::new("note", "manual-generation"),
            None,
            Some(7),
            true,
        ),
        (
            RecordId::new("note", "current"),
            Some(current_source.clone()),
            Some(1),
            true,
        ),
        (
            RecordId::new("note", "pending"),
            Some(current_source.clone()),
            Some(2),
            false,
        ),
        (
            RecordId::new("note", "superseded"),
            Some(current_source.clone()),
            Some(0),
            false,
        ),
        (
            RecordId::new("note", "legacy-source"),
            Some(current_source.clone()),
            None,
            true,
        ),
        (
            RecordId::new("note", "legacy-missing-source"),
            Some(missing_source.clone()),
            None,
            true,
        ),
        (
            RecordId::new("note", "owned-missing-source"),
            Some(missing_source),
            Some(1),
            false,
        ),
        (
            compound.clone(),
            Some(current_source.clone()),
            Some(1),
            true,
        ),
        (
            deleted_compound.clone(),
            Some(current_source.clone()),
            Some(1),
            false,
        ),
        (deleted_string.clone(), None, None, false),
        (integer, Some(current_source.clone()), Some(1), true),
        (uuid, Some(current_source.clone()), Some(1), true),
    ];
    for (id, owner, generation, _) in &cases {
        note(&repo, id.clone(), owner.clone(), *generation, timestamp()).await;
    }
    repo.db
        .query("DELETE $first; DELETE $second;")
        .bind(("first", deleted_compound))
        .bind(("second", deleted_string))
        .await
        .unwrap()
        .check()
        .unwrap();
    cases.push((missing_compound, None, None, false));
    cases.push((RecordId::new("note", "missing"), None, None, false));
    cases.push((note_range(), None, None, false));
    // Non-note record IDs must not enter a note-table visibility set.
    cases.push((current_source, None, None, false));

    let visible = graph::graph_endpoint_visible_sql("$endpoint");
    let eligible = graph::graph_endpoint_eligible_sql("$endpoint");
    for (id, _, _, expected) in cases {
        for (since, uri) in [
            (None, None),
            (Some(timestamp()), None),
            (Some(timestamp() + chrono::Duration::seconds(1)), None),
            (None, Some("fixture://orion".to_string())),
            (None, Some("fixture://other".to_string())),
        ] {
            let result: Option<EndpointComparison> = repo.db
                .query(format!(
                    "RETURN {{ legacy_visible: $endpoint IN (SELECT VALUE id FROM note WHERE {VISIBLE_NOTE_CONDITION}), \
                     fetched_visible: {visible}, \
                     legacy_eligible: $endpoint IN (SELECT VALUE id FROM note WHERE {VISIBLE_NOTE_CONDITION} AND ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri)), \
                     fetched_eligible: {eligible} }};"
                ))
                .bind(("endpoint", id.clone()))
                .bind(("since", since.map(|value| value.to_rfc3339())))
                .bind(("source_uri", uri))
                .await.unwrap().check().unwrap().take(0).unwrap();
            let result = result.unwrap();
            assert_eq!(result.legacy_visible, expected, "fixture {id:?}");
            assert_eq!(
                result.fetched_visible, result.legacy_visible,
                "visibility {id:?}"
            );
            assert_eq!(
                result.fetched_eligible, result.legacy_eligible,
                "eligibility {id:?}"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn legacy_edges(
    repo: &Repository,
    ids: &[RecordId],
    tables: &[String],
    limit: usize,
    outbound: bool,
    inbound: bool,
    confidence: f32,
    since: Option<DateTime<Utc>>,
    uri: Option<String>,
    visited: &HashMap<String, Vec<RecordId>>,
) -> Vec<NoteEdgeRow> {
    let eligible = format!("(SELECT VALUE id FROM note WHERE ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION})");
    let mut result = HashMap::new();
    for table in ["supports", "contradicts", "related_to", "derived_from"] {
        if !tables.iter().any(|candidate| candidate == table) {
            continue;
        }
        for id in ids {
            let direction = match (outbound, inbound) {
                (true, true) => format!("((in = $note AND out IN {eligible} AND out NOT IN $visited) OR (out = $note AND in IN {eligible} AND in NOT IN $visited))"),
                (true, false) => format!("in = $note AND out IN {eligible} AND out NOT IN $visited"),
                (false, true) => format!("out = $note AND in IN {eligible} AND in NOT IN $visited"),
                (false, false) => "false".into(),
            };
            let rows: Vec<NoteEdgeRow> = repo.db
                .query(format!("SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at, IF confidence = NONE THEN 1.0 ELSE confidence END AS graph_confidence FROM {table} WHERE {direction} AND (confidence = NONE OR confidence >= $min_confidence) AND {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION} ORDER BY graph_confidence DESC, id ASC LIMIT $limit"))
                .bind(("note", id.clone()))
                .bind(("visited", visited.get(&record_id_to_string(id)).cloned().unwrap_or_default()))
                .bind(("min_confidence", confidence))
                .bind(("limit", limit as i64))
                .bind(("since", since.map(|value| value.to_rfc3339())))
                .bind(("source_uri", uri.clone()))
                .await.unwrap().take(0).unwrap();
            for row in rows {
                result.entry(record_id_to_string(&row.id)).or_insert(row);
            }
        }
    }
    let mut rows = result.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        a.edge_type
            .cmp(&b.edge_type)
            .then_with(|| record_id_to_string(&a.id).cmp(&record_id_to_string(&b.id)))
    });
    rows
}

#[tokio::test]
async fn edge_batches_preserve_pre_limit_filters_direction_confidence_cycles_and_sources() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "eligible", "fixture://eligible").await;
    let foreign = source(&repo, "foreign", "fixture://foreign").await;
    let old_time = timestamp() - chrono::Duration::days(30);
    // The source/age filter belongs to the neighbor, not the frontier itself.
    let seed = note(
        &repo,
        RecordId::new("note", "seed"),
        Some(foreign.clone()),
        Some(1),
        old_time,
    )
    .await;
    let second_seed = note(
        &repo,
        RecordId::new("note", "second-seed"),
        None,
        None,
        timestamp(),
    )
    .await;
    let current = note(
        &repo,
        RecordId::new("note", "current"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let inbound_note = note(
        &repo,
        RecordId::new("note", "inbound"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let hidden = note(
        &repo,
        RecordId::new("note", "hidden"),
        Some(owner.clone()),
        Some(2),
        timestamp(),
    )
    .await;
    let old = note(
        &repo,
        RecordId::new("note", "old"),
        Some(owner.clone()),
        Some(1),
        old_time,
    )
    .await;
    let foreign_note = note(
        &repo,
        RecordId::new("note", "foreign"),
        Some(foreign),
        Some(1),
        timestamp(),
    )
    .await;
    let compound = note(
        &repo,
        compound_note("current-edge"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let dangling = compound_note("missing-edge");
    let range = note_range();
    for (key, target) in [
        ("hidden", &hidden),
        ("old", &old),
        ("foreign", &foreign_note),
        ("dangling", &dangling),
        ("range", &range),
    ] {
        edge(&repo, "supports", key, &seed, target, Some(0.99)).await;
    }
    let outgoing = edge(&repo, "supports", "outgoing", &seed, &current, Some(0.8)).await;
    let incoming = edge(
        &repo,
        "supports",
        "incoming",
        &inbound_note,
        &seed,
        Some(0.7),
    )
    .await;
    edge(&repo, "supports", "weak", &seed, &compound, Some(0.2)).await;
    edge(
        &repo,
        "contradicts",
        "null-confidence",
        &seed,
        &compound,
        None,
    )
    .await;
    edge(
        &repo,
        "related_to",
        "second-frontier",
        &second_seed,
        &current,
        Some(0.9),
    )
    .await;
    edge(
        &repo,
        "supports",
        "hidden-frontier",
        &hidden,
        &current,
        Some(1.0),
    )
    .await;
    let ids = vec![seed.clone(), second_seed, hidden, dangling, range];
    let tables = vec![
        "supports".into(),
        "contradicts".into(),
        "related_to".into(),
        "derived_from".into(),
    ];
    let mut visited = HashMap::new();
    visited.insert(record_id_to_string(&seed), vec![seed.clone(), inbound_note]);
    for (outbound, inbound) in [(true, false), (false, true), (true, true)] {
        for limit in [1, 3] {
            for (since, uri) in [
                (None, None),
                (Some(timestamp()), Some("fixture://eligible".to_string())),
            ] {
                let scoped = since.is_some();
                let expected = legacy_edges(
                    &repo,
                    &ids,
                    &tables,
                    limit,
                    outbound,
                    inbound,
                    0.5,
                    since,
                    uri.clone(),
                    &visited,
                )
                .await;
                let actual = repo
                    .graph_note_edges_excluding_visited(
                        &ids, &tables, limit, outbound, inbound, 0.5, since, uri, &visited,
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&actual).unwrap(),
                    serde_json::to_value(&expected).unwrap()
                );
                assert!(actual.iter().all(|row| row.id != incoming));
                if outbound && (scoped || limit == 3) {
                    assert!(actual.iter().any(|row| row.id == outgoing));
                }
            }
        }
    }
    assert!(repo
        .graph_note_edges(&ids, &tables, 1, false, false, 0.0, None, None)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn mention_batches_keep_per_entity_pages_and_filter_before_the_limit() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "orion", "fixture://orion").await;
    let mut alpha = Entity::new("Orion Alpha", graphrag_core::EntityType::Project);
    alpha.metadata = serde_json::json!({});
    let alpha = repo.upsert_entity(alpha).await.unwrap().id.unwrap();
    let mut beta = Entity::new("Orion Beta", graphrag_core::EntityType::Project);
    beta.metadata = serde_json::json!({});
    let beta = repo.upsert_entity(beta).await.unwrap().id.unwrap();
    let hidden = note(
        &repo,
        RecordId::new("note", "00-hidden"),
        Some(owner.clone()),
        Some(2),
        timestamp(),
    )
    .await;
    let old = note(
        &repo,
        RecordId::new("note", "01-old"),
        Some(owner.clone()),
        Some(1),
        timestamp() - chrono::Duration::days(1),
    )
    .await;
    let compound = note(
        &repo,
        compound_note("real-mention"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    for id in [
        &hidden,
        &old,
        &compound_note("missing-mention"),
        &note_range(),
    ] {
        mention(&repo, id, &alpha).await;
    }
    for index in 0..5 {
        let id = note(
            &repo,
            RecordId::new("note", format!("a-{index}")),
            Some(owner.clone()),
            Some(1),
            timestamp(),
        )
        .await;
        mention(&repo, &id, &alpha).await;
    }
    let later = note(
        &repo,
        RecordId::new("note", "z-beta"),
        Some(owner),
        Some(1),
        timestamp(),
    )
    .await;
    mention(&repo, &later, &beta).await;
    mention(&repo, &compound, &beta).await;
    for (since, uri) in [
        (None, None),
        (Some(timestamp()), Some("fixture://orion".to_string())),
    ] {
        let mut expected = Vec::new();
        for entity in [&alpha, &beta] {
            let mut rows: Vec<GraphEntityNoteSeed> = repo.db
                .query(format!("SELECT in AS note_id, out AS entity_id FROM mentions WHERE out = $entity AND in IN (SELECT VALUE id FROM note WHERE {VISIBLE_NOTE_CONDITION} AND ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri)) ORDER BY in ASC LIMIT 2"))
                .bind(("entity", entity.clone()))
                .bind(("since", since.map(|value| value.to_rfc3339())))
                .bind(("source_uri", uri.clone()))
                .await.unwrap().take(0).unwrap();
            expected.append(&mut rows);
        }
        let actual = repo
            .graph_notes_for_entities(&[alpha.clone(), beta.clone()], 2, since, uri)
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(
            actual.iter().filter(|row| row.entity_id == alpha).count(),
            2
        );
        assert_eq!(actual.iter().filter(|row| row.entity_id == beta).count(), 2);
        assert!(actual.iter().any(|row| row.note_id == later));
    }
}

#[tokio::test]
async fn graph_hydration_keeps_note_membership_and_provenance_order_without_duplicate_ids() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "orion", "fixture://orion").await;
    let current = note(
        &repo,
        compound_note("hydration"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let hidden = note(
        &repo,
        RecordId::new("note", "hidden"),
        Some(owner.clone()),
        Some(2),
        timestamp(),
    )
    .await;
    let integer = note(
        &repo,
        RecordId::new("note", 42i64),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let numeric_string = note(
        &repo,
        RecordId::new("note", "42"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let missing = compound_note("missing-hydration");
    let ids = vec![
        current.clone(),
        hidden,
        missing,
        current.clone(),
        owner,
        integer,
        numeric_string,
        note_range(),
    ];
    let actual = repo
        .graph_notes_by_ids(&ids, Some(timestamp()), Some("fixture://orion".into()))
        .await
        .unwrap();
    let expected: Vec<SearchResult> = repo.db
        .query(format!("SELECT id, title, content, note_type, tags, created_at, source_id.uri AS source_uri FROM note WHERE id IN $ids AND created_at >= <datetime>$since AND source_id.uri = $source_uri AND {VISIBLE_NOTE_CONDITION} ORDER BY id ASC"))
        .bind(("ids", ids))
        .bind(("since", timestamp().to_rfc3339()))
        .bind(("source_uri", "fixture://orion"))
        .await.unwrap().take(0).unwrap();
    assert_eq!(actual.len(), 3);
    assert_eq!(
        serde_json::to_value(&actual).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    // Relations may reference chat records no longer present; provenance
    // retains their canonical IDs rather than requiring endpoint hydration.
    for table in ["note_from_message", "note_from_conversation"] {
        let destination = if table == "note_from_message" {
            "message"
        } else {
            "conversation"
        };
        for _ in 0..2 {
            repo.db
                .query(format!("CREATE {table} SET in = $note, out = $chat"))
                .bind(("note", current.clone()))
                .bind(("chat", RecordId::new(destination, "orion")))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
    }
    let provenance = repo
        .graph_note_provenance_ids(std::slice::from_ref(&current))
        .await
        .unwrap();
    assert_eq!(
        provenance[&record_id_to_string(&current)],
        vec!["conversation:orion", "message:orion"]
    );
}

#[test]
fn graph_batch_work_does_not_add_full_corpus_eligibility_queries_as_frontier_grows() {
    let tables = ["supports", "contradicts", "related_to", "derived_from"];
    for frontier_size in [1, 12, 32, 200] {
        let sql = graph::graph_edge_batch_sql(frontier_size, &tables, true, true);
        assert!(
            !sql.contains("FROM note"),
            "edge eligibility must use endpoint document fetches"
        );
        assert!(
            !sql.contains("LET "),
            "empty edge tables must not force full corpus materialization"
        );
        assert_eq!(
            sql.matches("LIMIT $limit;").count(),
            frontier_size * tables.len()
        );
        let mentions = graph::graph_mention_batch_sql(frontier_size);
        assert!(!mentions.contains("FROM note"));
        assert_eq!(mentions.matches("LIMIT $limit;").count(), frontier_size);
    }
}

/// Negative optimization result: inspect whether adding a single-column
/// access path changes the actual plan for scoped out-only mention lookups.
/// The pinned engine already uses the existing (out, in) association index.
#[tokio::test]
#[ignore = "opt-in fictional query-plan experiment, not a hardware CI gate"]
async fn out_only_mention_plan_diagnostic() {
    let repo = Repository::new(init_memory().await.unwrap());
    let current = source(&repo, "index-current", "fixture://index-current").await;
    let entity_id = RecordId::new("entity", "index-atlas");
    repo.db.query("CREATE $id CONTENT {name:'Atlas',canonical_name:'atlas',entity_type:'project',metadata:{}}")
        .bind(("id", entity_id.clone())).await.unwrap().check().unwrap();
    for index in 0..260 {
        let id = RecordId::new("note", format!("index-{index:03}"));
        note(
            &repo,
            id.clone(),
            Some(current.clone()),
            Some(if index % 9 == 0 { 2 } else { 1 }),
            timestamp(),
        )
        .await;
        mention(&repo, &id, &entity_id).await;
    }
    let eligible = graph_endpoint_eligible_sql("in");
    let sql = format!("SELECT in AS note_id, out AS entity_id, ({}) AS aliases FROM mentions WHERE out = $entity AND {eligible} ORDER BY in ASC LIMIT 200", "array::slice(IF string::starts_with(out.identity_key ?? '', 'extracted-v1:') THEN metadata.aliases ?? [] ELSE out.metadata.aliases ?? [] END, 0, 8)");
    async fn rows(repo: &Repository, sql: &str, entity_id: &RecordId) -> Vec<GraphEntityNoteSeed> {
        repo.db
            .query(sql)
            .bind(("entity", entity_id.clone()))
            .bind(("since", Option::<String>::None))
            .bind(("source_uri", Option::<String>::None))
            .await
            .unwrap()
            .take(0)
            .unwrap()
    }
    async fn plan(repo: &Repository, sql: &str, entity_id: &RecordId) -> String {
        let values: Vec<serde_json::Value> = repo
            .db
            .query(format!("{sql} EXPLAIN FULL"))
            .bind(("entity", entity_id.clone()))
            .bind(("since", Option::<String>::None))
            .bind(("source_uri", Option::<String>::None))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        serde_json::to_string(&values).unwrap()
    }
    let before = rows(&repo, &sql, &entity_id).await;
    let before_plan = plan(&repo, &sql, &entity_id).await;
    repo.db
        .query("DEFINE INDEX idx_mentions_entity ON mentions FIELDS out")
        .await
        .unwrap()
        .check()
        .unwrap();
    let after = rows(&repo, &sql, &entity_id).await;
    let after_plan = plan(&repo, &sql, &entity_id).await;
    assert_eq!(
        serde_json::to_value(&before).unwrap(),
        serde_json::to_value(&after).unwrap()
    );
    assert_eq!(after.len(), 200);
    assert!(
        before_plan.contains("\"index\":\"idx_mentions_entity_note\""),
        "baseline must use the existing compound index: {before_plan}"
    );
    assert!(
        after_plan.contains("\"index\":\"idx_mentions_entity_note\""),
        "redundant index must not be assumed to improve the plan: {after_plan}"
    );
    if std::env::var_os("GRAPHRAG_GRAPH_PLAN_REPORT").is_some() {
        println!("before-plan={before_plan}\nafter-plan={after_plan}");
    }
}
