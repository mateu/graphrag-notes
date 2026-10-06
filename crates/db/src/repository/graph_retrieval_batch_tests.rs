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
    let eligible = format!(
        "(SELECT VALUE id FROM note WHERE ($since = NONE OR created_at >= <datetime>$since) AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION})"
    );
    let mut result = HashMap::new();
    for table in ["supports", "contradicts", "related_to", "derived_from"] {
        if !tables.iter().any(|candidate| candidate == table) {
            continue;
        }
        for id in ids {
            let direction = match (outbound, inbound) {
                (true, true) => format!(
                    "((in = $note AND out IN {eligible} AND out NOT IN $visited) OR (out = $note AND in IN {eligible} AND in NOT IN $visited))"
                ),
                (true, false) => {
                    format!("in = $note AND out IN {eligible} AND out NOT IN $visited")
                }
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
    let sql = format!(
        "SELECT in AS note_id, out AS entity_id, ({}) AS aliases FROM mentions WHERE out = $entity AND {eligible} ORDER BY in ASC LIMIT 200",
        "array::slice(IF string::starts_with(out.identity_key ?? '', 'extracted-v1:') THEN metadata.aliases ?? [] ELSE out.metadata.aliases ?? [] END, 0, 8)"
    );
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

async fn alias_entity(repo: &Repository, key: &str, name: &str, extracted: bool) -> RecordId {
    let mut entity = Entity::new(name, graphrag_core::EntityType::Technology);
    entity.identity_key = Some(format!(
        "{}:{key}",
        if extracted { "extracted-v1" } else { "legacy" }
    ));
    entity.metadata = serde_json::json!({"aliases": ["GPT-4", "東京", "Needle Token"]});
    repo.upsert_entity(entity).await.unwrap().id.unwrap()
}

async fn alias_mention(
    repo: &Repository,
    note_id: &RecordId,
    entity_id: &RecordId,
    metadata: serde_json::Value,
) {
    repo.db
        .query("CREATE mentions SET in = $note, out = $entity, metadata = $metadata")
        .bind(("note", note_id.clone()))
        .bind(("entity", entity_id.clone()))
        .bind(("metadata", metadata))
        .await
        .unwrap()
        .check()
        .unwrap();
}

// The references retain both the pre-optimization predicate and the alias-
// first variant with its original $parent correlation. Compare all three
// complete ordered graph payloads, including hydrated notes and provenance.
async fn assert_alias_query_matches_original(
    repo: &Repository,
    query: &str,
    limit: usize,
    ranked: Option<&[RecordId]>,
    since: Option<DateTime<Utc>>,
    uri: Option<String>,
) -> Vec<GraphEntityMatch> {
    let original = repo
        .find_graph_entities_original_alias_order(query, limit, ranked, since, uri.clone())
        .await
        .unwrap();
    let parent = repo
        .find_graph_entities_parent_alias_order(query, limit, ranked, since, uri.clone())
        .await
        .unwrap();
    let candidate = match ranked {
        Some(ids) => {
            repo.find_graph_entities_for_search(query, limit, ids, since, uri.clone())
                .await
        }
        None => repo.find_graph_entities(query, limit).await,
    }
    .unwrap();
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&original).unwrap(),
        "entity payload query={query:?}, limit={limit}, since={since:?}, uri={uri:?}"
    );
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&parent).unwrap(),
        "local binding must preserve the alias-first parent query {query:?}"
    );
    let mut payloads = Vec::new();
    for entities in [&original, &parent, &candidate] {
        let ids = entities
            .iter()
            .map(|entity| entity.id.clone())
            .collect::<Vec<_>>();
        let seeds = repo
            .graph_notes_for_entities_ranked_for_query(
                &ids,
                ranked.unwrap_or(&[]),
                3,
                since,
                uri.clone(),
                query,
            )
            .await
            .unwrap();
        let note_ids = seeds
            .iter()
            .map(|seed| seed.note_id.clone())
            .collect::<Vec<_>>();
        let notes = repo
            .graph_notes_by_ids(&note_ids, since, uri.clone())
            .await
            .unwrap();
        let provenance = repo.graph_note_provenance_ids(&note_ids).await.unwrap();
        payloads.push(serde_json::json!({
            "entities": entities, "seeds": seeds, "notes": notes, "provenance": provenance,
        }));
    }
    assert_eq!(payloads[0], payloads[1], "complete graph payload {query:?}");
    assert_eq!(payloads[0], payloads[2], "local graph payload {query:?}");
    candidate
}

#[tokio::test]
async fn graph_alias_fast_path_preserves_high_degree_pages_tiers_and_alias_caps() {
    let repo = Repository::new(init_memory().await.unwrap());
    // An empty table still parses the complete exact/contained/prefix SQL.
    // Cover both scoped and unscoped alias blocks at the engine's unchanged
    // default depth before any matching rows can hide an unvisited tier.
    for ranked in [None, Some(&[][..])] {
        let rows =
            assert_alias_query_matches_original(&repo, "syntax absent", 2, ranked, None, None)
                .await;
        assert!(rows.is_empty());
    }
    let owner = source(&repo, "alias-pages", "fixture://alias-pages").await;
    let extracted = alias_entity(&repo, "alias-pages", "Compiler", true).await;
    let legacy = alias_entity(&repo, "legacy-pages", "Legacy Compiler", false).await;
    let mut ids = Vec::new();
    for index in 0..260 {
        let id = note(
            &repo,
            RecordId::new("note", format!("alias-page-{index:03}")),
            Some(owner.clone()),
            Some(1),
            timestamp(),
        )
        .await;
        alias_mention(
            &repo,
            &id,
            &extracted,
            serde_json::json!({"aliases": if index == 259 {
                vec!["Needle Token".to_string(), "GPT-4".to_string(), "東京".to_string()]
            } else { vec![format!("Unrelated {index}")] }}),
        )
        .await;
        ids.push(id);
    }
    alias_mention(&repo, &ids[0], &legacy, serde_json::json!({})).await;
    let homonym = alias_entity(&repo, "alias-homonym", "Compiler", true).await;
    alias_mention(
        &repo,
        &ids[0],
        &homonym,
        serde_json::json!({"aliases": ["Unrelated"]}),
    )
    .await;
    // The only matching extracted alias lies beyond the first 200 note IDs:
    // both predicates must match before the retained page, then cap aliases.
    for query in [
        "Needle Token",
        "status Needle Token",
        "GPT-4?",
        "東京",
        "compi",
        "absent stellar",
    ] {
        for ranked in [None, Some(&ids[259..])] {
            assert_alias_query_matches_original(&repo, query, 2, ranked, None, None).await;
        }
    }
    let matches = assert_alias_query_matches_original(
        &repo,
        "Needle Token",
        10,
        Some(&[]),
        None,
        Some("fixture://alias-pages".into()),
    )
    .await;
    assert!(matches.iter().any(|entity| entity.id == extracted));
    assert_eq!(
        matches
            .iter()
            .find(|entity| entity.id == extracted)
            .unwrap()
            .metadata["aliases"],
        serde_json::json!(["Needle Token"])
    );

    // Dense matches cross the 200-mention limit and the eight-alias union cap.
    // Keep per-mention arrays bounded, including repeated normalized aliases.
    for (index, id) in ids.iter().enumerate() {
        repo.db
            .query("UPDATE mentions SET metadata = $metadata WHERE in = $note AND out = $entity")
            .bind(("note", id.clone()))
            .bind(("entity", extracted.clone()))
            .bind((
                "metadata",
                serde_json::json!({"aliases": ["Needle Token", format!("Variant {}", index % 9)]}),
            ))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let query = "Needle Token Variant 0 Variant 1 Variant 2 Variant 3 Variant 4 Variant 5 Variant 6 Variant 7 Variant 8";
    let matches = assert_alias_query_matches_original(
        &repo,
        query,
        10,
        Some(&ids[259..]),
        Some(timestamp()),
        Some("fixture://alias-pages".into()),
    )
    .await;
    let aliases = matches
        .iter()
        .find(|entity| entity.id == extracted)
        .unwrap()
        .metadata["aliases"]
        .as_array()
        .unwrap();
    assert_eq!(aliases.len(), 8);
    assert_eq!(aliases[0], "Needle Token");
    assert_eq!(aliases[7], "Variant 6");
    for limit in [0, 1, 2] {
        assert_alias_query_matches_original(
            &repo,
            "Compiler",
            limit,
            Some(&ids[259..]),
            None,
            None,
        )
        .await;
    }
    let preferred =
        assert_alias_query_matches_original(&repo, "Compiler", 1, Some(&ids[259..]), None, None)
            .await;
    assert_eq!(
        preferred[0].id, extracted,
        "direct-note preference must precede ID ties"
    );
}

#[tokio::test]
async fn graph_alias_fast_path_preserves_native_endpoints_scope_and_nonfinite_vectors() {
    let repo = Repository::new(init_memory().await.unwrap());
    // Legacy nonfinite primaries cannot enter HNSW. Alias eligibility does no
    // vector arithmetic; preserve their stored F64 and serving F32 bits.
    repo.db
        .query("REMOVE INDEX idx_note_embedding ON note")
        .await
        .unwrap()
        .check()
        .unwrap();
    let current = source(&repo, "alias-current", "fixture://alias-current").await;
    let foreign = source(&repo, "alias-foreign", "fixture://alias-foreign").await;
    let missing = compound_note("alias-never-created");
    let deleted = compound_note("alias-deleted");
    let mut cases = vec![
        (
            RecordId::new("note", "alias-manual"),
            None,
            None,
            timestamp(),
        ),
        (
            RecordId::new("note", "alias-current"),
            Some(current.clone()),
            Some(1),
            timestamp(),
        ),
        (
            RecordId::new("note", "alias-pending"),
            Some(current.clone()),
            Some(2),
            timestamp(),
        ),
        (
            RecordId::new("note", "alias-superseded"),
            Some(current.clone()),
            Some(0),
            timestamp(),
        ),
        (
            RecordId::new("note", "alias-old"),
            Some(current.clone()),
            Some(1),
            timestamp() - chrono::Duration::seconds(1),
        ),
        (
            RecordId::new("note", "alias-foreign"),
            Some(foreign),
            Some(1),
            timestamp(),
        ),
        (
            RecordId::new("note", 42i64),
            Some(current.clone()),
            Some(1),
            timestamp(),
        ),
        (
            RecordId::new("note", "42"),
            Some(current.clone()),
            Some(1),
            timestamp(),
        ),
        (
            RecordId::new(
                "note",
                surrealdb_types::Uuid::from(uuid::Uuid::from_u128(126)),
            ),
            Some(current.clone()),
            Some(1),
            timestamp(),
        ),
        (
            compound_note("alias-current"),
            Some(current.clone()),
            Some(1),
            timestamp(),
        ),
        (deleted.clone(), Some(current.clone()), Some(1), timestamp()),
    ];
    let mut vector_bits_before = Vec::new();
    let mut ranked = Vec::new();
    for (index, (id, owner, generation, created)) in cases.iter().enumerate() {
        note(&repo, id.clone(), owner.clone(), *generation, *created).await;
        let mut vector = vec![1.0f64; 1024];
        vector[0] = [
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0 + 1e-9,
        ][index % 5];
        repo.db
            .query("UPDATE $note SET embedding = $embedding")
            .bind(("note", id.clone()))
            .bind(("embedding", vector))
            .await
            .unwrap()
            .check()
            .unwrap();
        let stored: Vec<Vec<f64>> = repo
            .db
            .query("SELECT VALUE embedding FROM $note")
            .bind(("note", id.clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        vector_bits_before.push((
            id.clone(),
            stored[0]
                .iter()
                .map(|value| (value.to_bits(), (*value as f32).to_bits()))
                .collect::<Vec<_>>(),
        ));
        ranked.push(id.clone());
    }
    repo.db
        .query("DELETE $note")
        .bind(("note", deleted.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
    cases.push((missing, None, None, timestamp()));
    cases.push((note_range(), None, None, timestamp()));
    let mut entity_ids = Vec::new();
    for (index, (id, _, _, _)) in cases.iter().enumerate() {
        let entity = alias_entity(&repo, &format!("native-{index:02}"), "Compiler", true).await;
        alias_mention(
            &repo,
            id,
            &entity,
            serde_json::json!({"aliases": ["Needle Token", "GPT-4", "東京"]}),
        )
        .await;
        entity_ids.push(entity);
    }
    let legacy = alias_entity(&repo, "native-legacy", "Legacy Compiler", false).await;
    alias_mention(&repo, &ranked[0], &legacy, serde_json::json!({})).await;
    for (query, since, uri) in [
        ("Needle Token", None, None),
        (
            "Needle Token",
            Some(timestamp()),
            Some("fixture://alias-current".into()),
        ),
        (
            "status Needle Token",
            Some(timestamp() + chrono::Duration::seconds(1)),
            None,
        ),
        ("GPT-4", None, Some("fixture://alias-foreign".into())),
        ("東京", None, None),
        ("compi", None, None),
    ] {
        assert_alias_query_matches_original(&repo, query, 20, Some(&ranked), since, uri).await;
    }
    let matches = assert_alias_query_matches_original(
        &repo,
        "Needle Token",
        20,
        Some(&ranked),
        Some(timestamp()),
        Some("fixture://alias-current".into()),
    )
    .await;
    // Exact timestamp is inclusive, failed origins retain successful gen 1,
    // and key-only/deleted/staged evidence cannot supply an extracted alias.
    let matched_ids = matches.iter().map(|entity| &entity.id).collect::<Vec<_>>();
    assert!(matched_ids.contains(&&entity_ids[1]));
    for excluded in [2, 3, 4, 5, 10, 11, 12] {
        assert!(!matched_ids.contains(&&entity_ids[excluded]));
    }
    for (id, expected) in vector_bits_before {
        if id == deleted {
            continue;
        }
        let stored: Vec<Vec<f64>> = repo
            .db
            .query("SELECT VALUE embedding FROM $note")
            .bind(("note", id))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(
            stored[0]
                .iter()
                .map(|value| (value.to_bits(), (*value as f32).to_bits()))
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[tokio::test]
async fn graph_alias_fast_path_keeps_malformed_and_oversized_error_order() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "alias-errors", "fixture://alias-errors").await;
    let visible = note(
        &repo,
        RecordId::new("note", "alias-visible"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let hidden = note(
        &repo,
        RecordId::new("note", "alias-hidden"),
        Some(owner),
        Some(2),
        timestamp(),
    )
    .await;
    let dangling = compound_note("alias-errors-dangling");
    let entity = alias_entity(&repo, "alias-errors", "Compiler", true).await;
    let mut oversized = (0..8)
        .map(|index| format!("Unrelated {index}"))
        .collect::<Vec<_>>();
    oversized.push("Needle Token".into());
    for metadata in [
        serde_json::json!({}),
        serde_json::json!({"aliases": null}),
        serde_json::json!({"aliases": []}),
        serde_json::json!({"aliases": ["Needle Token", "GPT-4", "東京", "four", "five", "six", "seven", "eight"]}),
        serde_json::json!({"aliases": oversized}),
        serde_json::json!({"aliases": "Needle Token"}),
        serde_json::json!({"aliases": {"value": "Needle Token"}}),
        serde_json::json!({"aliases": [42, "Needle Token"]}),
        serde_json::json!({"aliases": ["Needle Token", null]}),
        serde_json::json!({"aliases": ["Unrelated", {"value": "Needle Token"}]}),
    ] {
        for (endpoint, since, uri) in [
            (&visible, None, None),
            (&hidden, None, None),
            (&dangling, None, None),
            (
                &visible,
                Some(timestamp() + chrono::Duration::seconds(1)),
                None,
            ),
            (&visible, None, Some("fixture://elsewhere".to_string())),
        ] {
            repo.db
                .query("DELETE mentions")
                .await
                .unwrap()
                .check()
                .unwrap();
            alias_mention(&repo, endpoint, &entity, metadata.clone()).await;
            let original = repo
                .find_graph_entities_original_alias_order(
                    "Needle Token",
                    10,
                    Some(&[]),
                    since,
                    uri.clone(),
                )
                .await;
            if endpoint != &visible || since.is_some() || uri.is_some() {
                assert!(
                    original.as_ref().unwrap().is_empty(),
                    "ineligible metadata must remain shielded: {metadata}"
                );
            } else {
                let aliases = metadata.get("aliases").filter(|value| !value.is_null());
                match aliases {
                    Some(serde_json::Value::Array(values))
                        if values.iter().all(serde_json::Value::is_string) =>
                    {
                        assert_eq!(
                            original.as_ref().unwrap().len(),
                            usize::from(values.iter().any(|value| value == "Needle Token"))
                        );
                    }
                    None => assert!(original.as_ref().unwrap().is_empty()),
                    _ => assert!(
                        original.is_err(),
                        "eligible malformed aliases must keep their error: {metadata}"
                    ),
                }
            }
            if endpoint == &hidden && metadata["aliases"].is_string() {
                // A bare reorder is observably incompatible: this relation's
                // invalid array used to be shielded by its hidden generation.
                let eligible = graph_endpoint_eligible_sql("in");
                assert!(repo.db.query(format!("SELECT VALUE in FROM mentions WHERE out = $entity AND array::any(metadata.aliases ?? [], |$alias| string::lowercase($alias) = $query) AND {eligible}"))
                    .bind(("entity", entity.clone())).bind(("query", "needle token"))
                    .bind(("since", Option::<String>::None)).bind(("source_uri", Option::<String>::None))
                    .await.unwrap().check().is_err());
            }
            let parent = repo
                .find_graph_entities_parent_alias_order(
                    "Needle Token",
                    10,
                    Some(&[]),
                    since,
                    uri.clone(),
                )
                .await;
            let candidate = repo
                .find_graph_entities_for_search("Needle Token", 10, &[], since, uri.clone())
                .await;
            for result in [parent, candidate] {
                match (&original, result) {
                    (Ok(original), Ok(candidate)) => assert_eq!(
                        serde_json::to_value(candidate).unwrap(),
                        serde_json::to_value(original).unwrap()
                    ),
                    (Err(original), Err(candidate)) => assert_eq!(
                        original.to_string(),
                        candidate.to_string(),
                        "metadata={metadata}, endpoint={endpoint:?}"
                    ),
                    (original, candidate) => panic!(
                        "changed error behavior metadata={metadata}, endpoint={endpoint:?}: original={original:?}, candidate={candidate:?}"
                    ),
                }
            }
        }
    }
}

#[derive(Debug, Deserialize, SurrealValue)]
struct CapturedGraphEntityId {
    id: RecordId,
    captured_id: RecordId,
}

#[derive(Debug, serde::Serialize, Deserialize, SurrealValue)]
struct NestedGraphEntityPlan {
    id: RecordId,
    plan: Vec<serde_json::Value>,
}

#[tokio::test]
async fn graph_alias_local_binding_preserves_malformed_errors_beyond_the_alias_page() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "late-alias-errors", "fixture://late-alias-errors").await;
    let entity = alias_entity(&repo, "late-alias-errors", "Compiler", true).await;
    for index in 0..350 {
        let endpoint = note(
            &repo,
            RecordId::new("note", format!("late-alias-valid-{index:03}")),
            Some(owner.clone()),
            Some(1),
            timestamp(),
        )
        .await;
        repo.db
            .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
            .bind((
                "id",
                RecordId::new("mentions", format!("100-valid-{index:03}")),
            ))
            .bind(("note", endpoint))
            .bind(("entity", entity.clone()))
            .bind(("metadata", serde_json::json!({"aliases": ["Needle Token"]})))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let late_endpoint = note(
        &repo,
        RecordId::new("note", "zz-late-alias-malformed"),
        Some(owner),
        Some(1),
        timestamp(),
    )
    .await;
    // The bad endpoint always sorts beyond 200 matching aliases. Put its
    // relation ID both first and last in the table scan: the first case
    // cannot be hidden behind an already-populated table TopK threshold.
    // The second also checks that forcing an indexed sort does not introduce
    // a new error if the pinned table scan can skip a late noncompetitive row.
    for bad_key in ["000-bad", "999-bad"] {
        let bad_id = RecordId::new("mentions", bad_key);
        for metadata in [
            serde_json::json!({"aliases": "Needle Token"}),
            serde_json::json!({"aliases": {"value": "Needle Token"}}),
            serde_json::json!({"aliases": [42, "Needle Token"]}),
        ] {
            for generation in [1i64, 2] {
                repo.db
                    .query("UPDATE $note SET source_generation = $generation; CREATE $id SET in = $note, out = $entity, metadata = $metadata")
                    .bind(("id", bad_id.clone()))
                    .bind(("note", late_endpoint.clone()))
                    .bind(("generation", generation))
                    .bind(("entity", entity.clone()))
                    .bind(("metadata", metadata.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
                let original = repo
                    .find_graph_entities_original_alias_order(
                        "Needle Token",
                        10,
                        Some(&[]),
                        None,
                        None,
                    )
                    .await;
                if bad_key == "000-bad" && generation == 1 {
                    assert!(original.is_err(), "early eligible malformed row must error");
                }
                if generation == 2 {
                    assert!(
                        original.is_ok(),
                        "hidden generation must shield malformed row"
                    );
                }
                let parent = repo
                    .find_graph_entities_parent_alias_order(
                        "Needle Token",
                        10,
                        Some(&[]),
                        None,
                        None,
                    )
                    .await;
                let candidate = repo
                    .find_graph_entities_for_search("Needle Token", 10, &[], None, None)
                    .await;
                for actual in [parent, candidate] {
                    match (&original, actual) {
                        (Ok(expected), Ok(actual)) => assert_eq!(
                            serde_json::to_value(expected).unwrap(),
                            serde_json::to_value(actual).unwrap()
                        ),
                        (Err(expected), Err(actual)) => assert_eq!(
                            expected.to_string(),
                            actual.to_string(),
                            "late alias metadata={metadata}, relation={bad_key}, generation={generation}"
                        ),
                        (expected, actual) => panic!(
                            "changed late error/page behavior metadata={metadata}, relation={bad_key}, generation={generation}: expected={expected:?}, actual={actual:?}"
                        ),
                    }
                }
                repo.db
                    .query("DELETE $id")
                    .bind(("id", bad_id.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
        }
    }
}

#[tokio::test]
async fn graph_alias_local_binding_preserves_duplicate_mentions_before_direct_rank_cap() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "local-duplicates", "fixture://local-duplicates").await;
    let first = alias_entity(&repo, "local-duplicates-a", "A Compiler", true).await;
    let second = alias_entity(&repo, "local-duplicates-b", "B Compiler", true).await;
    let lower_rank = note(
        &repo,
        RecordId::new("note", "local-duplicates-a"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let best_rank = note(
        &repo,
        RecordId::new("note", "local-duplicates-z"),
        Some(owner),
        Some(1),
        timestamp(),
    )
    .await;
    // Relation-ID order encounters the best direct hit before two duplicate
    // lower hits. Association-index order visits both lower IDs first. The
    // non-unique association index must not alter the existing direct-rank
    // query's limited subset and let the second homonym displace the first.
    for (key, endpoint, entity) in [
        ("local-duplicates-00", &best_rank, &first),
        ("local-duplicates-10", &lower_rank, &first),
        ("local-duplicates-20", &lower_rank, &first),
        ("local-duplicates-30", &best_rank, &second),
    ] {
        repo.db
            .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
            .bind(("id", RecordId::new("mentions", key)))
            .bind(("note", endpoint.clone()))
            .bind(("entity", entity.clone()))
            .bind(("metadata", serde_json::json!({"aliases": ["Needle Token"]})))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let ranked = [best_rank, lower_rank];
    for limit in [1, 2] {
        let rows = assert_alias_query_matches_original(
            &repo,
            "Needle Token",
            limit,
            Some(&ranked),
            None,
            Some("fixture://local-duplicates".into()),
        )
        .await;
        assert_eq!(rows[0].id, first);
    }
    // Alias paging is explicitly ordered, but many relation rows can share
    // the same endpoint. Place different matching spellings across its 200-
    // mention boundary and retain exact union/cap behavior under those ties.
    let dense = alias_entity(&repo, "local-duplicates-dense", "Dense Compiler", true).await;
    for index in 0..220 {
        repo.db
            .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
            .bind((
                "id",
                RecordId::new("mentions", format!("local-dense-{index:03}")),
            ))
            .bind(("note", ranked[0].clone()))
            .bind(("entity", dense.clone()))
            .bind((
                "metadata",
                serde_json::json!({"aliases": if index < 199 {
                    vec!["Needle Token".to_string()]
                } else {
                    vec!["Needle Token".to_string(), format!("Boundary {index}")]
                }}),
            ))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
    let query = format!(
        "Needle Token {}",
        (199..220)
            .map(|index| format!("Boundary {index}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for scope in [None, Some(ranked.as_slice())] {
        assert_alias_query_matches_original(&repo, &query, 10, scope, None, None).await;
    }
}

#[tokio::test]
async fn graph_alias_local_binding_preserves_complex_relation_id_ties_at_the_page_cap() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "relation-key-ties", "fixture://relation-key-ties").await;
    let entity = alias_entity(&repo, "relation-key-ties", "Compiler", true).await;
    let endpoint = note(
        &repo,
        RecordId::new("note", "relation-key-ties"),
        Some(owner),
        Some(1),
        timestamp(),
    )
    .await;
    let mut integer_object = Object::new();
    integer_object.insert("kind", 1i64);
    integer_object.insert("ordinal", 1i64);
    let mut float_object = Object::new();
    float_object.insert("kind", 1.0f64);
    float_object.insert("ordinal", 0i64);
    let integer_array = vec![
        surrealdb::types::Value::Number(surrealdb::types::Number::Int(1)),
        surrealdb::types::Value::Number(surrealdb::types::Number::Int(1)),
    ];
    let float_array = vec![
        surrealdb::types::Value::Number(surrealdb::types::Number::Float(1.0)),
        surrealdb::types::Value::Number(surrealdb::types::Number::Int(0)),
    ];
    for (integer_key, float_key) in [
        (
            RecordIdKey::Array(integer_array.into()),
            RecordIdKey::Array(float_array.into()),
        ),
        (
            RecordIdKey::Object(integer_object),
            RecordIdKey::Object(float_object),
        ),
    ] {
        repo.db
            .query("DELETE mentions")
            .await
            .unwrap()
            .check()
            .unwrap();
        // Simple relation IDs fill 199 equal-in positions. The primary key
        // encodes nested NumberKind while the association key omits it; the
        // two complex IDs can therefore enter the 200th tie in a different
        // order. Their distinct matching alias evidence exposes any change.
        for index in 0..199 {
            repo.db
                .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
                .bind((
                    "id",
                    RecordId::new("mentions", format!("00-simple-{index:03}")),
                ))
                .bind(("note", endpoint.clone()))
                .bind(("entity", entity.clone()))
                .bind(("metadata", serde_json::json!({"aliases": ["Needle Token"]})))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for (key, alias) in [
            (integer_key, "Boundary Integer"),
            (float_key, "Boundary Float"),
        ] {
            repo.db
                .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
                .bind(("id", RecordId::new("mentions", key)))
                .bind(("note", endpoint.clone()))
                .bind(("entity", entity.clone()))
                .bind((
                    "metadata",
                    serde_json::json!({"aliases": ["Needle Token", alias]}),
                ))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for scope in [None, Some(std::slice::from_ref(&endpoint))] {
            assert_alias_query_matches_original(
                &repo,
                "Needle Token Boundary Integer Boundary Float",
                10,
                scope,
                None,
                None,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn graph_alias_local_binding_plan_falls_back_for_unsafe_mention_shapes() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "local-fallback", "fixture://local-fallback").await;
    let entity = alias_entity(&repo, "local-fallback", "Compiler", true).await;
    let visible = note(
        &repo,
        RecordId::new("note", "local-fallback-visible"),
        Some(owner.clone()),
        Some(1),
        timestamp(),
    )
    .await;
    let hidden = note(
        &repo,
        RecordId::new("note", "local-fallback-hidden"),
        Some(owner),
        Some(2),
        timestamp(),
    )
    .await;
    alias_mention(
        &repo,
        &visible,
        &entity,
        serde_json::json!({"aliases": ["Needle Token"]}),
    )
    .await;
    let eligible = graph_endpoint_eligible_sql("in");
    let predicate = graph::graph_alias_eligible_sql(&eligible, "$alias = $query");
    let select = format!("SELECT VALUE array::filter(metadata.aliases ?? [], |$alias| $alias = $query) FROM mentions WHERE out = $parent.id AND {predicate} ORDER BY in ASC LIMIT 200");
    let explained = graph::graph_entity_mentions_local_sql(&format!("{select} EXPLAIN FULL"));
    let sql = format!("SELECT id, ({explained}) AS plan FROM entity WHERE id = $entity");
    let mut response = repo
        .db
        .query(sql.clone())
        .bind(("entity", entity.clone()))
        .bind(("query", "Needle Token"))
        .bind(("since", Option::<String>::None))
        .bind(("source_uri", Option::<String>::None))
        .await
        .unwrap();
    let plan: Vec<NestedGraphEntityPlan> = response.take(0).unwrap();
    assert_eq!(plan.len(), 1);
    assert!(serde_json::to_string(&plan[0].plan)
        .unwrap()
        .contains("\"index\":\"idx_mentions_entity_note\""));
    let mut complex = Object::new();
    complex.insert("kind", 1.0f64);
    complex.insert("ordinal", 0i64);
    for (key, endpoint, aliases) in [
        (
            RecordIdKey::String("local-fallback-nine".into()),
            &visible,
            serde_json::json!(vec!["Needle Token"; 9]),
        ),
        (
            RecordIdKey::String("local-fallback-scalar".into()),
            &hidden,
            serde_json::json!("Needle Token"),
        ),
        (
            RecordIdKey::String("local-fallback-mixed".into()),
            &hidden,
            serde_json::json!([42, "Needle Token"]),
        ),
        (
            RecordIdKey::Object(complex),
            &visible,
            serde_json::json!(["Needle Token"]),
        ),
        (
            RecordIdKey::Array(vec![1i64, 2i64].into()),
            &visible,
            serde_json::json!(["Needle Token"]),
        ),
    ] {
        let id = RecordId::new("mentions", key);
        repo.db
            .query("CREATE $id SET in = $note, out = $entity, metadata = $metadata")
            .bind(("id", id.clone()))
            .bind(("note", endpoint.clone()))
            .bind(("entity", entity.clone()))
            .bind(("metadata", serde_json::json!({"aliases": aliases})))
            .await
            .unwrap()
            .check()
            .unwrap();
        // This EXPLAIN runs inside the actual entity projection and selects
        // the legacy branch at runtime; its result must not be the index plan
        // of the separate safe-shape probe or of the unselected fast branch.
        let mut response = repo
            .db
            .query(sql.clone())
            .bind(("entity", entity.clone()))
            .bind(("query", "Needle Token"))
            .bind(("since", Option::<String>::None))
            .bind(("source_uri", Option::<String>::None))
            .await
            .unwrap();
        let plan: Vec<NestedGraphEntityPlan> = response.take(0).unwrap();
        assert_eq!(plan.len(), 1);
        assert!(!serde_json::to_string(&plan[0].plan)
            .unwrap()
            .contains("\"index\":\"idx_mentions_entity_note\""));
        assert_alias_query_matches_original(
            &repo,
            "Needle Token",
            10,
            Some(std::slice::from_ref(&visible)),
            None,
            None,
        )
        .await;
        repo.db
            .query("DELETE $id")
            .bind(("id", id))
            .await
            .unwrap()
            .check()
            .unwrap();
    }
}

#[tokio::test]
async fn graph_alias_local_binding_indexes_actual_nested_queries_and_keeps_native_ids() {
    let repo = Repository::new(init_memory().await.unwrap());
    let owner = source(&repo, "local-plan", "fixture://local-plan").await;
    let mut object_key = Object::new();
    object_key.insert("kind", "fictional");
    object_key.insert("ordinal", 7i64);
    let entity_ids = [
        RecordId::new("entity", 42i64),
        RecordId::new("entity", "42"),
        RecordId::new(
            "entity",
            surrealdb_types::Uuid::from(uuid::Uuid::from_u128(126)),
        ),
        RecordId::new("entity", RecordIdKey::Object(object_key)),
        RecordId::new(
            "entity",
            RecordIdKey::Array(
                vec![
                    surrealdb::types::Value::String("fictional".to_string()),
                    surrealdb::types::Value::Number(surrealdb::types::Number::Int(7)),
                ]
                .into(),
            ),
        ),
    ];
    let mut note_ids = Vec::new();
    for (index, entity_id) in entity_ids.iter().enumerate() {
        // upsert_entity intentionally generates its own ID; direct fictional
        // creation is required to exercise the native entity-key variants.
        repo.db
            .query("CREATE $id SET name = 'Compiler', canonical_name = 'compiler', entity_type = 'technology', identity_key = $identity, metadata = {}")
            .bind(("id", entity_id.clone()))
            .bind(("identity", format!("extracted-v1:local-plan-{index}")))
            .await
            .unwrap()
            .check()
            .unwrap();
        let id = note(
            &repo,
            RecordId::new("note", format!("local-plan-{index}")),
            Some(owner.clone()),
            Some(1),
            timestamp(),
        )
        .await;
        alias_mention(
            &repo,
            &id,
            entity_id,
            serde_json::json!({"aliases": ["Needle Token"]}),
        )
        .await;
        note_ids.push(id);
    }
    // These blocks execute inside the entity SELECT itself. A direct client
    // $entity binding or top-level EXPLAIN cannot prove correlated planning.
    // Bind an intentionally wrong outer $parent and local parameter as well:
    // only the current row's native ID may reach the inner association query.
    let wrong_entity = RecordId::new("entity", "fictional-not-the-current-entity");
    let captured: Vec<CapturedGraphEntityId> = repo
        .db
        .query("SELECT id, ({ LET $graph_entity_id = id; $graph_entity_id; }) AS captured_id FROM entity ORDER BY id ASC")
        .bind(("parent", serde_json::json!({"id": wrong_entity.clone()})))
        .bind(("graph_entity_id", wrong_entity.clone()))
        .await
        .unwrap()
        .take(0)
        .unwrap();
    assert_eq!(captured.len(), entity_ids.len());
    for row in &captured {
        assert_eq!(row.id, row.captured_id);
        assert!(entity_ids.contains(&row.id));
    }
    let eligible = graph_endpoint_eligible_sql("in");
    let alias_matches = "$alias = $query";
    let alias_eligible = graph::graph_alias_eligible_sql(&eligible, alias_matches);
    let queries = [
        format!(
            "SELECT VALUE array::filter(metadata.aliases ?? [], |$alias| {alias_matches}) FROM mentions WHERE out = $parent.id AND {alias_eligible} ORDER BY in ASC LIMIT 200"
        ),
    ];
    let mut reported = Vec::new();
    for select in queries {
        let mut outputs = Vec::new();
        let mut plans = Vec::new();
        for local in [false, true] {
            let nested = if local {
                graph::graph_entity_mentions_local_sql(&select)
            } else {
                select.clone()
            };
            let explained = format!("{select} EXPLAIN FULL");
            let explained = if local {
                graph::graph_entity_mentions_local_sql(&explained)
            } else {
                explained
            };
            let sql = format!(
                "SELECT id, ({nested}) AS result FROM entity ORDER BY id ASC; \
                 SELECT id, ({explained}) AS plan FROM entity ORDER BY id ASC;"
            );
            let mut response = repo
                .db
                .query(sql)
                .bind(("parent", serde_json::json!({"id": wrong_entity.clone()})))
                .bind(("graph_entity_id", wrong_entity.clone()))
                .bind(("query", "Needle Token"))
                .bind(("ranked_notes", note_ids.clone()))
                .bind(("ranked_limit", note_ids.len() as i64))
                .bind(("since", Option::<String>::None))
                .bind(("source_uri", Option::<String>::None))
                .await
                .unwrap()
                .check()
                .unwrap();
            let rows: Vec<serde_json::Value> = response.take(0).unwrap();
            let nested_plans: Vec<NestedGraphEntityPlan> = response.take(1).unwrap();
            assert_eq!(rows.len(), entity_ids.len());
            assert_eq!(nested_plans.len(), entity_ids.len());
            for row in &nested_plans {
                assert!(entity_ids.contains(&row.id));
                let plan = serde_json::to_string(&row.plan).unwrap();
                // The pinned streaming analyzer accepts literal simple keys
                // but cannot plan object/array record-ID equality. Those
                // native keys must retain exact results through its fallback.
                let simple_key = matches!(
                    &row.id.key,
                    RecordIdKey::Number(_) | RecordIdKey::String(_) | RecordIdKey::Uuid(_)
                );
                assert_eq!(
                    plan.contains("\"index\":\"idx_mentions_entity_note\""),
                    local && simple_key,
                    "local simple IDs expose the out-prefix; complex IDs keep fallback: {plan}"
                );
            }
            outputs.push(rows);
            plans.push(serde_json::to_value(&nested_plans).unwrap());
        }
        assert_eq!(outputs[0], outputs[1], "nested SELECT changed: {select}");
        reported.push(serde_json::json!({
            "select": select, "parent": plans[0], "local": plans[1],
        }));
    }
    if std::env::var_os("GRAPHRAG_GRAPH_ALIAS_PLAN_REPORT").is_some() {
        println!("graph-local-alias-plan={}", serde_json::json!(reported));
    }
    // A reversed direct-hit order must retain its per-entity rank before the
    // cap, independently of native entity-ID ordering or block-local state.
    note_ids.reverse();
    for query in ["Needle Token", "Which Needle Token?", "Compiler", "Comp"] {
        assert_alias_query_matches_original(
            &repo,
            query,
            3,
            Some(&note_ids),
            None,
            Some("fixture://local-plan".into()),
        )
        .await;
    }
}

#[tokio::test]
async fn graph_alias_fast_path_short_circuits_only_bounded_string_arrays_and_keeps_index_plan() {
    let repo = Repository::new(init_memory().await.unwrap());
    // An invalid integer cast stands in for an expensive/erroring endpoint:
    // no match skips it only on the trusted bounded shape. This proves actual
    // pinned-engine expression behavior without a hardware timing assertion.
    let eligible = "(<int>'fictional-invalid-int') > 0";
    let matches = "$alias = 'Needle Token'";
    let candidate = graph::graph_alias_eligible_sql(eligible, matches)
        .replace("metadata.aliases", "$metadata.aliases");
    let original =
        format!("{eligible} AND array::any($metadata.aliases ?? [], |$alias| {matches})");
    for (metadata, skips) in [
        (serde_json::json!({}), true),
        (serde_json::json!({"aliases": null}), true),
        (serde_json::json!({"aliases": []}), true),
        (serde_json::json!({"aliases": vec!["Unrelated"; 8]}), true),
        (serde_json::json!({"aliases": ["Needle Token"]}), false),
        (serde_json::json!({"aliases": vec!["Unrelated"; 9]}), false),
        (serde_json::json!({"aliases": ["Unrelated", 1]}), false),
        (serde_json::json!({"aliases": "Unrelated"}), false),
    ] {
        let original = repo
            .db
            .query(format!("RETURN ({original})"))
            .bind(("metadata", metadata.clone()))
            .await
            .unwrap()
            .check();
        assert!(original.is_err());
        let candidate = repo
            .db
            .query(format!("RETURN ({candidate})"))
            .bind(("metadata", metadata))
            .await
            .unwrap()
            .check();
        if skips {
            let result: Option<bool> = candidate.unwrap().take(0).unwrap();
            assert_eq!(result, Some(false));
        } else {
            assert!(candidate.is_err());
        }
    }
    let owner = source(&repo, "alias-plan", "fixture://alias-plan").await;
    let entity = alias_entity(&repo, "alias-plan", "Compiler", true).await;
    let id = note(
        &repo,
        RecordId::new("note", "alias-plan"),
        Some(owner),
        Some(1),
        timestamp(),
    )
    .await;
    alias_mention(
        &repo,
        &id,
        &entity,
        serde_json::json!({"aliases": ["Needle Token"]}),
    )
    .await;
    let eligible = graph_endpoint_eligible_sql("in");
    let predicates = [
        format!("{eligible} AND array::any(metadata.aliases ?? [], |$alias| $alias = $query)"),
        graph::graph_alias_eligible_sql(&eligible, "$alias = $query"),
    ];
    let mut plans = Vec::new();
    for predicate in predicates {
        let rows: Vec<serde_json::Value> = repo.db.query(format!("SELECT in, out FROM mentions WHERE out = $entity AND {predicate} ORDER BY in ASC LIMIT 200 EXPLAIN FULL"))
            .bind(("entity", entity.clone())).bind(("query", "Needle Token"))
            .bind(("since", Option::<String>::None)).bind(("source_uri", Option::<String>::None))
            .await.unwrap().take(0).unwrap();
        let plan = serde_json::to_string(&rows).unwrap();
        assert!(
            plan.contains("\"index\":\"idx_mentions_entity_note\""),
            "existing association index must remain usable: {plan}"
        );
        plans.push(rows);
    }
    if std::env::var_os("GRAPHRAG_GRAPH_ALIAS_PLAN_REPORT").is_some() {
        println!(
            "graph-alias-plan={}",
            serde_json::json!({"original": plans[0], "candidate": plans[1]})
        );
    }
}
