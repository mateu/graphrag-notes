//! Full-text retrieval is independent of provider availability and vector state.
use async_trait::async_trait;
use chrono::{Duration, Utc};
use graphrag_agents::{Embedder, InferenceCapabilities, SearchAgent, SearchHitType, SearchScope};
use graphrag_core::{ChatExport, Note, SourceType};
use graphrag_db::{compatibility::embedding_metadata, init_memory, Repository};
use std::sync::Arc;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("keyword retrieval must not embed")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("keyword retrieval must not batch embed")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("keyword retrieval must not check provider health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("keyword retrieval must not inspect provider/cache identity")
    }
}

async fn seed(repo: &Repository) {
    for (label, age) in [("recent", 0), ("old", 90)] {
        let timestamp = Utc::now() - Duration::days(age);
        let uri = format!("file:///keyword-{label}.md");
        let mut source = repo
            .begin_file_import(
                SourceType::Markdown,
                label.into(),
                uri.clone(),
                "cafelexeme launch notes".into(),
                graphrag_core::normalized_content_hash("cafelexeme launch notes"),
                false,
            )
            .await
            .unwrap()
            .source;
        let mut note = Note::new("cafelexeme launch notes")
            .with_source(source.id.clone().unwrap())
            .with_source_generation(source.generation);
        note.created_at = timestamp;
        // Deliberately legacy vectors without deployment metadata. Keyword
        // retrieval must neither reject them nor initialize vector metadata.
        note.embedding = vec![1.0; 1024];
        repo.create_note(note).await.unwrap();
        repo.complete_file_import(&mut source).await.unwrap();
        let conversation = ChatExport::from_json(&serde_json::json!([{
            "uuid": format!("keyword-{label}"),
            "name": format!("{label} discussion"),
            "summary": "cafelexeme launch summary",
            "created_at": timestamp.to_rfc3339(),
            "updated_at": timestamp.to_rfc3339(),
            "chat_messages": [{"uuid": format!("keyword-message-{label}"), "sender": "human", "text": "cafelexeme launch message", "created_at": timestamp.to_rfc3339()}]
        }]).to_string()).unwrap().conversations.remove(0);
        let id = repo
            .upsert_conversation(&conversation, Some(uri), serde_json::json!({"conversation_id": conversation.uuid, "created_at": conversation.created_at, "summary": conversation.summary}), None)
            .await
            .unwrap();
        repo.upsert_message(&id, &conversation.uuid, 0, &conversation.messages[0], None)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn keyword_all_scopes_filters_and_empty_results_never_touch_inference() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    seed(&repo).await;
    let search = SearchAgent::new(repo.clone(), Arc::new(NoInference));
    let all = search
        .keyword_search_with_scope("cafelexeme", 20, SearchScope::All, None, None)
        .await
        .unwrap();
    assert_eq!(all.hits.len(), 6);
    for kind in [
        SearchHitType::Note,
        SearchHitType::Message,
        SearchHitType::ConversationSummary,
    ] {
        assert_eq!(
            all.hits.iter().filter(|hit| hit.hit_type == kind).count(),
            2
        );
    }
    assert_eq!(all.summary.candidates_considered, 0);
    for (index, hit) in all.hits.iter().enumerate() {
        let evidence = hit.explanation();
        assert_eq!(evidence.rank, index + 1);
        assert!(evidence.vector.is_none());
        assert!(evidence.graph.is_none());
        assert!(evidence.full_text.is_some());
        assert_eq!(evidence.final_score.kind, graphrag_agents::ScoreKind::Bm25);
        assert_eq!(
            evidence.fused.value,
            evidence.full_text.unwrap().raw_value.unwrap()
        );
        assert!(evidence.embedding_provider.is_none());
    }
    for (scope, expected) in [(SearchScope::Notes, 2), (SearchScope::Messages, 2)] {
        assert_eq!(
            search
                .keyword_search_with_scope("cafelexeme", 20, scope, None, None)
                .await
                .unwrap()
                .hits
                .len(),
            expected
        );
    }
    let recent = search
        .keyword_search_with_scope(
            "cafelexeme",
            20,
            SearchScope::All,
            Some(1),
            Some("file:///keyword-recent.md".into()),
        )
        .await
        .unwrap();
    assert_eq!(recent.hits.len(), 3);
    assert!(recent
        .hits
        .iter()
        .all(|hit| hit.source_uri.as_deref() == Some("file:///keyword-recent.md")));
    let old = search
        .keyword_search_with_scope(
            "cafelexeme",
            20,
            SearchScope::All,
            None,
            Some("file:///keyword-old.md".into()),
        )
        .await
        .unwrap();
    assert_eq!(old.hits.len(), 3);
    assert!(search
        .keyword_search_with_scope(
            "cafelexeme",
            20,
            SearchScope::All,
            Some(1),
            Some("file:///keyword-old.md".into())
        )
        .await
        .unwrap()
        .hits
        .is_empty());
    assert!(search
        .keyword_search_with_scope("unmatchedlexeme", 20, SearchScope::All, None, None)
        .await
        .unwrap()
        .hits
        .is_empty());
    assert!(search
        .keyword_search_with_scope("cafelexeme", 0, SearchScope::All, None, None)
        .await
        .unwrap()
        .hits
        .is_empty());
    assert!(embedding_metadata(&db).await.unwrap().is_none());
    let cache_keys: Vec<String> = db
        .query("SELECT VALUE cache_key FROM inference_cache")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    assert!(cache_keys.is_empty());
}

#[tokio::test]
async fn keyword_weights_and_limits_use_the_reported_bm25_ranker() {
    let repo = Repository::new(init_memory().await.unwrap());
    seed(&repo).await;
    let search = SearchAgent::new(repo, Arc::new(NoInference)).with_fusion_config(
        graphrag_db::fusion::FusionConfig::default(),
        0.0,
        2.0,
        1.0,
    );
    let results = search
        .keyword_search_with_scope("cafelexeme", 20, SearchScope::All, None, None)
        .await
        .unwrap();
    for hit in &results.hits {
        assert_eq!(
            hit.score,
            hit.fusion.fulltext_score.unwrap() * hit.effective_weight
        );
    }
    assert!(results
        .hits
        .windows(2)
        .all(|hits| hits[0].score >= hits[1].score));
    let limited = search
        .keyword_search_with_scope("cafelexeme", 1, SearchScope::All, None, None)
        .await
        .unwrap();
    assert_eq!(limited.hits.len(), 1);
    assert_eq!(limited.hits[0].id, results.hits[0].id);
    let repeated = search
        .keyword_search_with_scope("cafelexeme", 20, SearchScope::All, None, None)
        .await
        .unwrap();
    assert_eq!(
        results.hits.iter().map(|hit| &hit.id).collect::<Vec<_>>(),
        repeated.hits.iter().map(|hit| &hit.id).collect::<Vec<_>>()
    );
}
