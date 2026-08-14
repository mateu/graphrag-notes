//! Search Agent - Handles user queries with hybrid search

use crate::{inference::validate_embedding_dim, Result, SharedEmbedder};
use chrono::{Duration, Utc};
use graphrag_core::record_id_to_string;
use graphrag_db::compatibility::EmbeddingIdentity;
use graphrag_db::repository::{
    ConversationSearchResult, MessageSearchResult, RelatedNotes, SearchResult, SimilarNote,
};
use graphrag_db::{
    fusion::{self, FusionConfig, FusionEvidence},
    Repository,
};

use std::collections::HashSet;

use tracing::{debug, info, instrument};

/// Search result with optional graph context
#[derive(Debug)]
pub struct EnrichedSearchResult {
    pub result: SearchResult,
    pub related: Option<RelatedNotes>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScope {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchHitType {
    Note,
    Message,
    ConversationSummary,
}

#[derive(Debug, Clone)]
pub struct ScopedSearchResult {
    pub hit_type: SearchHitType,
    pub id: String,
    pub title: Option<String>,
    pub content: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source_uri: Option<String>,
    pub score: f32,
    /// Retrieval evidence is intentionally retained for a later `--explain`
    /// surface while current CLI output remains unchanged.
    pub fusion: FusionEvidence,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AugmentOptions {
    pub max_chunks: usize,
    pub max_total_tokens: usize,
    pub max_chunk_tokens: usize,
}

impl Default for AugmentOptions {
    fn default() -> Self {
        Self {
            max_chunks: 8,
            max_total_tokens: 1200,
            max_chunk_tokens: 180,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AugmentChunk {
    pub citation: usize,
    pub hit_type: SearchHitType,
    pub id: String,
    pub title: Option<String>,
    pub snippet: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source_uri: Option<String>,
    pub score: f32,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    pub approx_tokens: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct AugmentContext {
    pub query: String,
    pub scope: SearchScope,
    pub entity_filter: Option<String>,
    pub chunks: Vec<AugmentChunk>,
    pub total_tokens: usize,
    pub dropped_duplicates: usize,
    pub dropped_for_budget: usize,
    pub dropped_for_entity_filter: usize,
}

impl AugmentContext {
    pub fn render_prompt_block(&self) -> String {
        let mut out = String::new();
        out.push_str("<context>\n");
        for chunk in &self.chunks {
            let title = chunk.title.as_deref().unwrap_or("(untitled)");
            out.push_str(&format!(
                "[C{}] [{}] {}\n{}\n\n",
                chunk.citation,
                hit_type_label(chunk.hit_type),
                title,
                chunk.snippet
            ));
        }
        out.push_str("</context>");
        out
    }
}

/// The Search agent handles user queries
pub struct SearchAgent {
    repo: Repository,
    embedder: SharedEmbedder,
    fusion: FusionConfig,
    note_weight: f32,
    message_weight: f32,
    conversation_summary_weight: f32,
}

impl SearchAgent {
    /// Create a new Search agent
    pub fn new(repo: Repository, embedder: SharedEmbedder) -> Self {
        Self {
            repo,
            embedder,
            fusion: FusionConfig::default(),
            note_weight: 1.0,
            message_weight: 1.0,
            conversation_summary_weight: 1.0,
        }
    }

    /// Set vector and full-text weights while retaining the RRF default.
    /// Prefer [`Self::with_fusion_config`] for all runtime controls.
    pub fn with_hybrid_weights(mut self, vector_weight: f32, fulltext_weight: f32) -> Self {
        self.fusion.vector_weight = vector_weight;
        self.fusion.fulltext_weight = fulltext_weight;
        self
    }

    /// Configure the one fusion component used for repository and all-scope
    /// search. Callers obtain validated values from `graphrag-config`.
    pub fn with_fusion_config(
        mut self,
        fusion: FusionConfig,
        note_weight: f32,
        message_weight: f32,
        conversation_summary_weight: f32,
    ) -> Self {
        self.fusion = fusion;
        self.note_weight = note_weight;
        self.message_weight = message_weight;
        self.conversation_summary_weight = conversation_summary_weight;
        self
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed(query, true).await?;
        validate_embedding_dim(embedding.len())?;
        let capability = self.embedder.capabilities();
        self.repo
            .record_embedding_metadata(
                &EmbeddingIdentity::new(capability.provider, capability.model, embedding.len()),
                None,
            )
            .await?;
        Ok(embedding)
    }

    /// Perform hybrid search (vector + full-text)
    #[instrument(skip(self))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_notes(query, limit, None, None).await
    }

    #[instrument(skip(self))]
    pub async fn search_notes(
        &self,
        query: &str,
        limit: usize,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        info!("Searching for: {}", query);

        // Generate query embedding
        debug!("Generating query embedding...");
        let embedding = self.embed_query(query).await?;
        let since = since_days.map(|days| Utc::now() - Duration::days(days as i64));

        // Perform hybrid search
        let results = self
            .repo
            .hybrid_search_notes_with_fusion(
                query,
                embedding,
                limit,
                since,
                source_uri,
                &self.fusion,
            )
            .await?;

        info!("Found {} results", results.len());

        Ok(results)
    }

    /// Search with graph context (includes related notes)
    #[instrument(skip(self))]
    pub async fn search_with_context(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<EnrichedSearchResult>> {
        self.search_with_context_filtered(query, limit, None, None)
            .await
    }

    #[instrument(skip(self))]
    pub async fn search_with_context_filtered(
        &self,
        query: &str,
        limit: usize,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<EnrichedSearchResult>> {
        let results = self
            .search_notes(query, limit, since_days, source_uri)
            .await?;

        let mut enriched = Vec::new();

        for result in results {
            // Try to get related notes (best effort) using the full RecordId
            let related = self.repo.get_related_notes(&result.id).await.ok();

            enriched.push(EnrichedSearchResult { result, related });
        }

        Ok(enriched)
    }

    /// Vector-only search (semantic similarity)
    #[instrument(skip(self))]
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing semantic search for: {}", query);

        let embedding = self.embed_query(query).await?;
        let results = self.repo.vector_search(embedding, limit).await?;

        Ok(results)
    }

    /// Full-text only search (keyword matching)
    #[instrument(skip(self))]
    pub async fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing keyword search for: {}", query);

        let results = self.repo.fulltext_search(query, limit).await?;

        Ok(results)
    }

    #[instrument(skip(self))]
    pub async fn search_with_scope(
        &self,
        query: &str,
        limit: usize,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<ScopedSearchResult>> {
        let since = since_days.map(|days| Utc::now() - Duration::days(days as i64));
        let embedding = self.embed_query(query).await?;
        let mut scoped_results = Vec::new();

        if matches!(scope, SearchScope::Notes | SearchScope::All) {
            let notes = self
                .repo
                .hybrid_search_notes_with_fusion(
                    query,
                    embedding.clone(),
                    limit,
                    since,
                    source_uri.clone(),
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                notes
                    .into_iter()
                    .map(|result| self.from_note_result(result)),
            );
        }

        if matches!(scope, SearchScope::Messages | SearchScope::All) {
            let messages = self
                .repo
                .hybrid_search_messages_with_fusion(
                    query,
                    embedding.clone(),
                    limit,
                    since,
                    source_uri.clone(),
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                messages
                    .into_iter()
                    .map(|result| self.from_message_result(result)),
            );
        }

        if matches!(scope, SearchScope::All) {
            let conversations = self
                .repo
                .hybrid_search_conversation_summaries_with_fusion(
                    query,
                    embedding,
                    limit,
                    since,
                    source_uri,
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                conversations
                    .into_iter()
                    .map(|result| self.from_conversation_result(result)),
            );
        }

        rank_scoped_results(&mut scoped_results);
        if scoped_results.len() > limit {
            scoped_results.truncate(limit);
        }

        Ok(scoped_results)
    }

    /// Retrieve ranked context snippets and package them for prompt augmentation.
    #[instrument(skip(self))]
    pub async fn build_augmented_context(
        &self,
        query: &str,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
        entity_filter: Option<String>,
        options: AugmentOptions,
    ) -> Result<AugmentContext> {
        if options.max_chunks == 0 || options.max_total_tokens == 0 || options.max_chunk_tokens == 0
        {
            return Ok(AugmentContext {
                query: query.to_string(),
                scope,
                entity_filter,
                chunks: Vec::new(),
                total_tokens: 0,
                dropped_duplicates: 0,
                dropped_for_budget: 0,
                dropped_for_entity_filter: 0,
            });
        }

        let fetch_limit = (options.max_chunks * 4).clamp(options.max_chunks, 200);
        let mut hits = self
            .search_with_scope(query, fetch_limit, scope, since_days, source_uri)
            .await?;

        let mut dropped_for_entity_filter = 0usize;
        if let Some(filter) = entity_filter.as_ref() {
            let mut filtered = Vec::with_capacity(hits.len());
            for hit in hits {
                if hit.hit_type == SearchHitType::Note
                    && !self.repo.note_has_entity_name(&hit.id, filter).await?
                {
                    dropped_for_entity_filter += 1;
                    continue;
                }
                filtered.push(hit);
            }
            hits = filtered;
        }

        Ok(build_augment_context_from_hits(
            query.to_string(),
            scope,
            entity_filter,
            hits,
            options,
            dropped_for_entity_filter,
        ))
    }

    fn from_note_result(&self, result: SearchResult) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: record_id_to_string(&result.id),
            title: result.title,
            content: result.content,
            created_at: Some(result.created_at),
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.note_weight),
            fusion: result.fusion,
            conversation_uuid: None,
            message_index: None,
            role: None,
        }
    }

    fn from_message_result(&self, result: MessageSearchResult) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Message,
            id: record_id_to_string(&result.id),
            title: Some(format!(
                "{} message #{}",
                result.role,
                result.message_index + 1
            )),
            content: result.content,
            created_at: result.created_at,
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.message_weight),
            fusion: result.fusion,
            conversation_uuid: Some(result.conversation_uuid),
            message_index: Some(result.message_index),
            role: Some(result.role),
        }
    }

    fn from_conversation_result(&self, result: ConversationSearchResult) -> ScopedSearchResult {
        let title = result
            .title
            .clone()
            .or_else(|| Some(format!("Conversation {}", result.uuid)));
        let content = result.summary.unwrap_or_default();
        ScopedSearchResult {
            hit_type: SearchHitType::ConversationSummary,
            id: record_id_to_string(&result.id),
            title,
            content,
            created_at: Some(result.updated_at),
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.conversation_summary_weight),
            fusion: result.fusion,
            conversation_uuid: Some(result.uuid),
            message_index: None,
            role: None,
        }
    }

    /// Find notes similar to a given note
    #[instrument(skip(self))]
    pub async fn find_similar(&self, note_id: &str, limit: usize) -> Result<Vec<SimilarNote>> {
        // First get the note to get its embedding
        let note = self
            .repo
            .get_note(note_id)
            .await?
            .ok_or_else(|| crate::AgentError::NotFound(format!("Note {}", note_id)))?;

        if note.embedding.is_empty() {
            return Err(crate::AgentError::NotFound("Note has no embedding".into()));
        }

        let similar = self
            .repo
            .find_similar_notes(note_id, note.embedding, 0.5, limit)
            .await?;

        Ok(similar)
    }
}

fn hit_type_order(hit_type: SearchHitType) -> usize {
    match hit_type {
        SearchHitType::Note => 0,
        SearchHitType::Message => 1,
        SearchHitType::ConversationSummary => 2,
    }
}

fn rank_scoped_results(results: &mut [ScopedSearchResult]) {
    results.sort_by(|left, right| {
        fusion::compare_scoped(
            left.score,
            &left.fusion,
            hit_type_order(left.hit_type),
            &left.id,
            right.score,
            &right.fusion,
            hit_type_order(right.hit_type),
            &right.id,
        )
    });
    for (index, result) in results.iter_mut().enumerate() {
        result.fusion.final_rank = index + 1;
    }
}

fn build_augment_context_from_hits(
    query: String,
    scope: SearchScope,
    entity_filter: Option<String>,
    mut hits: Vec<ScopedSearchResult>,
    options: AugmentOptions,
    dropped_for_entity_filter: usize,
) -> AugmentContext {
    rank_scoped_results(&mut hits);

    let mut chunks = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut seen_content = HashSet::new();
    let mut total_tokens = 0usize;
    let mut dropped_duplicates = 0usize;
    let mut dropped_for_budget = 0usize;

    for hit in hits {
        if chunks.len() >= options.max_chunks {
            break;
        }

        if !seen_ids.insert(hit.id.clone()) {
            dropped_duplicates += 1;
            continue;
        }

        let raw_content = hit.content.trim();
        if raw_content.is_empty() {
            dropped_duplicates += 1;
            continue;
        }

        let dedupe_key = normalize_text_for_dedupe(raw_content);
        if !dedupe_key.is_empty() && !seen_content.insert(dedupe_key) {
            dropped_duplicates += 1;
            continue;
        }

        let (snippet, approx_tokens, truncated) =
            truncate_to_token_limit(raw_content, options.max_chunk_tokens);
        if approx_tokens == 0 {
            dropped_duplicates += 1;
            continue;
        }

        if total_tokens + approx_tokens > options.max_total_tokens {
            dropped_for_budget += 1;
            continue;
        }

        total_tokens += approx_tokens;
        let citation = chunks.len() + 1;
        chunks.push(AugmentChunk {
            citation,
            hit_type: hit.hit_type,
            id: hit.id,
            title: hit.title,
            snippet,
            created_at: hit.created_at,
            source_uri: hit.source_uri,
            score: hit.score,
            conversation_uuid: hit.conversation_uuid,
            message_index: hit.message_index,
            role: hit.role,
            approx_tokens,
            truncated,
        });
    }

    AugmentContext {
        query,
        scope,
        entity_filter,
        chunks,
        total_tokens,
        dropped_duplicates,
        dropped_for_budget,
        dropped_for_entity_filter,
    }
}

fn normalize_text_for_dedupe(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;

    for ch in text.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }

    out.trim().to_string()
}

fn truncate_to_token_limit(text: &str, max_tokens: usize) -> (String, usize, bool) {
    if max_tokens == 0 {
        return (String::new(), 0, false);
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return (String::new(), 0, false);
    }

    if words.len() <= max_tokens {
        return (words.join(" "), words.len(), false);
    }

    let clipped = words[..max_tokens].join(" ");
    (format!("{clipped} ..."), max_tokens, true)
}

fn hit_type_label(hit_type: SearchHitType) -> &'static str {
    match hit_type {
        SearchHitType::Note => "note",
        SearchHitType::Message => "message",
        SearchHitType::ConversationSummary => "conversation-summary",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeterministicEmbedder;
    use graphrag_db::{compatibility::embedding_metadata, init_memory, Repository};
    use std::sync::Arc;

    #[tokio::test]
    async fn vector_queries_initialize_metadata_and_reject_a_model_change() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        let original = SearchAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("fixture", "model-a")),
        );
        original.embed_query("first query").await.unwrap();
        assert_eq!(
            embedding_metadata(&db)
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "model-a"
        );

        let changed = SearchAgent::new(
            repo,
            Arc::new(DeterministicEmbedder::default().with_identity("fixture", "model-b")),
        );
        let error = changed.embed_query("second query").await.unwrap_err();
        assert!(error.to_string().contains("graphrag reindex --all"));
    }

    #[tokio::test]
    async fn unavailable_embedder_does_not_initialize_metadata() {
        let db = init_memory().await.unwrap();
        let search = SearchAgent::new(
            Repository::new(db.clone()),
            Arc::new(DeterministicEmbedder::default().fail_next_requests(1, "offline")),
        );
        assert!(search.embed_query("offline query").await.is_err());
        assert!(embedding_metadata(&db).await.unwrap().is_none());
    }

    fn make_hit(id: &str, score: f32, content: &str) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: id.to_string(),
            title: Some("title".to_string()),
            content: content.to_string(),
            created_at: None,
            source_uri: None,
            score,
            fusion: FusionEvidence {
                fused_score: score,
                ..FusionEvidence::default()
            },
            conversation_uuid: None,
            message_index: None,
            role: None,
        }
    }

    #[test]
    fn deduplicates_similar_content() {
        let hits = vec![
            make_hit("note:a", 0.9, "Alpha beta gamma"),
            make_hit("note:b", 0.8, "alpha beta gamma"),
            make_hit("note:c", 0.7, "delta epsilon"),
        ];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 5,
                max_total_tokens: 200,
                max_chunk_tokens: 30,
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 2);
        assert_eq!(ctx.dropped_duplicates, 1);
        assert_eq!(ctx.chunks[0].id, "note:a");
        assert_eq!(ctx.chunks[1].id, "note:c");
    }

    #[test]
    fn retains_source_provenance_in_augment_chunks() {
        let mut hit = make_hit("note:a", 0.9, "Alpha beta gamma");
        hit.source_uri = Some("file:///notes/alpha.md".to_string());

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            vec![hit],
            AugmentOptions::default(),
            0,
        );

        assert_eq!(
            ctx.chunks[0].source_uri.as_deref(),
            Some("file:///notes/alpha.md")
        );
    }

    #[test]
    fn scope_ties_are_stable_by_hit_type_then_record_id() {
        let mut message = make_hit("message:z", 0.5, "message");
        message.hit_type = SearchHitType::Message;
        let note_b = make_hit("note:b", 0.5, "note b");
        let note_a = make_hit("note:a", 0.5, "note a");
        let mut hits = vec![message, note_b, note_a];

        rank_scoped_results(&mut hits);

        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            vec!["note:a", "note:b", "message:z"]
        );
        assert_eq!(hits[0].fusion.final_rank, 1);
    }

    #[test]
    fn enforces_total_token_budget() {
        let hits = vec![
            make_hit("note:a", 0.9, "one two three four five six"),
            make_hit("note:b", 0.8, "seven eight nine ten eleven"),
        ];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 5,
                max_total_tokens: 8,
                max_chunk_tokens: 30,
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 1);
        assert_eq!(ctx.total_tokens, 6);
        assert_eq!(ctx.dropped_for_budget, 1);
    }

    #[test]
    fn truncates_each_chunk_to_token_limit() {
        let hits = vec![make_hit("note:a", 0.9, "one two three four five six")];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 2,
                max_total_tokens: 100,
                max_chunk_tokens: 4,
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 1);
        assert_eq!(ctx.chunks[0].approx_tokens, 4);
        assert!(ctx.chunks[0].truncated);
        assert_eq!(ctx.chunks[0].snippet, "one two three four ...");
    }

    #[test]
    fn library_defaults_use_rrf_with_balanced_scope_weights() {
        let config = FusionConfig::default();
        assert_eq!(config.strategy, fusion::FusionStrategy::ReciprocalRank);
        assert_eq!(config.rrf_k, 60);
        assert_eq!(config.vector_weight, 0.7);
        assert_eq!(config.fulltext_weight, 0.3);
    }
}
