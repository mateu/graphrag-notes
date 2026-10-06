//! Complete ranked/evidence comparison against independent pre-reuse lexical
//! preparation. Fictional populations cross both the displayed top-k and the
//! bounded candidate page; timing is never an assertion.
use super::*;
use crate::DeterministicEmbedder;
use graphrag_core::{EdgeType, EntityType, Note};
use graphrag_db::init_memory;
use serde_json::{json, Value};
use std::sync::Arc;

fn snapshot(result: &GraphSearchResults) -> Value {
    json!({"summary": result.summary, "hits": result.hits.iter().map(|hit| json!({
        "id":hit.id,"title":hit.title,"content":hit.content,"created_at":hit.created_at,
        "source_uri":hit.source_uri,"explanation":hit.explanation(),"score":hit.score,
        "conversation_uuid":hit.conversation_uuid,"message_index":hit.message_index,"role":hit.role,
    })).collect::<Vec<_>>()})
}

async fn legacy_graph(
    search: &SearchAgent,
    query: &str,
    limit: usize,
    since_days: Option<u32>,
    source: Option<String>,
    mode: GraphMode,
) -> GraphSearchResults {
    let mut result = search
        .search_with_scope_graph(
            query,
            limit,
            SearchScope::Notes,
            since_days,
            source.clone(),
            GraphMode::Off,
        )
        .await
        .unwrap();
    if mode == GraphMode::Off {
        return result;
    }
    let graph = search
        .graph_candidates(
            query,
            &result.hits,
            since_days.map(|days| Utc::now() - Duration::days(days as i64)),
            source,
            mode == GraphMode::Auto,
        )
        .await
        .unwrap();
    result.summary = graph.summary;
    if !graph.hits.is_empty() || mode == GraphMode::On {
        merge_graph_hits(&mut result.hits, graph.hits);
    }
    rank_scoped_results(&mut result.hits);
    result.hits.truncate(limit);
    result.summary.candidates_selected =
        result.hits.iter().filter(|hit| hit.graph.is_some()).count();
    result.summary.candidates_dropped = result
        .summary
        .candidates_considered
        .saturating_sub(result.summary.candidates_selected);
    result
}

#[tokio::test]
async fn retained_lexical_prefix_matches_complete_legacy_graph_at_sparse_and_dense_scale() {
    for count in [32, 256] {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        repo.record_embedding_metadata(
            &EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
            None,
        )
        .await
        .unwrap();
        let source = RecordId::new("source", "scope");
        db.query("CREATE $id SET uri = 'fixture://scope', generation = 2, successful_generation = 1, status = 'failed'").bind(("id",source.clone())).await.unwrap().check().unwrap();
        let mut entity = Entity::new("Atlas", EntityType::Project);
        entity.identity_key = Some("extracted-v1:fictional-scope".into());
        entity.metadata = json!({"aliases":["Launch Atlas"]});
        let entity = repo.upsert_entity(entity).await.unwrap();
        let mut note_ids = Vec::new();
        for index in 0..count {
            let mut note = Note::new(format!(
                "Atlas retries deployment policy {index} {}",
                "background ".repeat(64)
            ));
            note.title = Some(if index == count - 1 {
                "Atlas".into()
            } else {
                format!("Deployment {index}")
            });
            note.embedding = vec![1.0; 1024];
            if index % 3 != 0 {
                note.source_id = Some(source.clone());
                note.source_generation = Some(if index % 11 == 0 { 2 } else { 1 });
            }
            if index % 7 == 0 {
                note.created_at = Utc::now() - Duration::days(10);
            }
            let note = repo.create_note(note).await.unwrap();
            note_ids.push(note.id.unwrap());
        }
        let target = repo
            .create_note(Note::new("Verified independent accepted path evidence"))
            .await
            .unwrap();
        // Sparse then dense eligible mentions exercise scoped alias evidence and
        // high-degree retention past the first 200 IDs.
        for dense in [false, true] {
            for (index, id) in note_ids.iter().enumerate() {
                if !dense && index != count - 1 {
                    continue;
                }
                // Staged generations are purposely linked through raw fixture
                // rows, to prove retrieval filters rather than write admission.
                db.query("CREATE mentions SET in = $note, out = $entity, metadata = {aliases:['Launch Atlas']}")
                    .bind(("note",id.clone())).bind(("entity",entity.id.clone().unwrap())).await.unwrap().check().unwrap();
            }
            repo.create_edge(
                &note_ids[count - 1],
                target.id.as_ref().unwrap(),
                EdgeType::Supports,
                Some(0.9),
            )
            .await
            .unwrap();
            for min_pool in [2, 50] {
                let search =
                    SearchAgent::new(repo.clone(), Arc::new(DeterministicEmbedder::default()))
                        .with_fusion_config(
                            FusionConfig {
                                candidate_pool_min: min_pool,
                                ..Default::default()
                            },
                            1.0,
                            1.0,
                            1.0,
                        );
                for (query, since, uri) in [
                    ("Atlas", None, None),
                    ("Launch Atlas retries", None, None),
                    (
                        "Atlas retries",
                        Some(1),
                        Some("fixture://scope".to_string()),
                    ),
                    ("absent fictional constellation", None, None),
                ] {
                    for mode in [GraphMode::Off, GraphMode::Auto, GraphMode::On] {
                        let old = legacy_graph(&search, query, 2, since, uri.clone(), mode).await;
                        let new = search
                            .search_with_scope_graph(
                                query,
                                2,
                                SearchScope::Notes,
                                since,
                                uri.clone(),
                                mode,
                            )
                            .await
                            .unwrap();
                        assert_eq!(
                            snapshot(&old),
                            snapshot(&new),
                            "count={count}, dense={dense}, min_pool={min_pool}, mode={mode:?}"
                        );
                        assert!(new.hits.len() <= 2);
                        assert!(new.summary.entities_matched <= 4);
                        assert!(new.summary.candidates_considered <= 32);
                    }
                }
            }
            db.query("DELETE mentions; DELETE supports")
                .await
                .unwrap()
                .check()
                .unwrap();
        }
    }
}
