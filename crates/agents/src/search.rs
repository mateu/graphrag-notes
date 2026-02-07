//! Search Agent - Handles user queries with hybrid search

use crate::{Result, TeiClient};
use chrono::{Duration, Utc};
use graphrag_db::repository::{
    ConversationSearchResult, MessageSearchResult, RelatedNotes, SearchResult, SimilarNote,
};
use graphrag_db::Repository;
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
    pub score: f32,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
}

/// The Search agent handles user queries
pub struct SearchAgent {
    repo: Repository,
    tei: TeiClient,
}

impl SearchAgent {
    /// Create a new Search agent
    pub fn new(repo: Repository, tei: TeiClient) -> Self {
        Self { repo, tei }
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
        let embedding = self.tei.embed(query, true).await?;
        let since = since_days.map(|days| Utc::now() - Duration::days(days as i64));

        // Perform hybrid search
        let results = self
            .repo
            .hybrid_search_notes(query, embedding, limit, since, source_uri)
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

        let embedding = self.tei.embed(query, true).await?;
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
        let embedding = self.tei.embed(query, true).await?;
        let mut scoped_results = Vec::new();

        if matches!(scope, SearchScope::Notes | SearchScope::All) {
            let notes = self
                .repo
                .hybrid_search_notes(query, embedding.clone(), limit, since, source_uri.clone())
                .await?;
            scoped_results.extend(notes.into_iter().map(Self::from_note_result));
        }

        if matches!(scope, SearchScope::Messages | SearchScope::All) {
            let messages = self
                .repo
                .hybrid_search_messages(query, embedding.clone(), limit, since, source_uri.clone())
                .await?;
            scoped_results.extend(messages.into_iter().map(Self::from_message_result));
        }

        if matches!(scope, SearchScope::All) {
            let conversations = self
                .repo
                .hybrid_search_conversation_summaries(query, embedding, limit, since, source_uri)
                .await?;
            scoped_results.extend(
                conversations
                    .into_iter()
                    .map(Self::from_conversation_result),
            );
        }

        scoped_results.sort_by(|a, b| b.score.total_cmp(&a.score));
        if scoped_results.len() > limit {
            scoped_results.truncate(limit);
        }

        Ok(scoped_results)
    }

    fn from_note_result(result: SearchResult) -> ScopedSearchResult {
        let score = Self::rank_score(result.vec_distance, result.fts_score);
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: result.id.to_string(),
            title: result.title,
            content: result.content,
            created_at: Some(result.created_at),
            score,
            conversation_uuid: None,
            message_index: None,
            role: None,
        }
    }

    fn from_message_result(result: MessageSearchResult) -> ScopedSearchResult {
        let score = Self::rank_score(result.vec_distance, result.fts_score);
        ScopedSearchResult {
            hit_type: SearchHitType::Message,
            id: result.id.to_string(),
            title: Some(format!(
                "{} message #{}",
                result.role,
                result.message_index + 1
            )),
            content: result.content,
            created_at: result.created_at,
            score,
            conversation_uuid: Some(result.conversation_uuid),
            message_index: Some(result.message_index),
            role: Some(result.role),
        }
    }

    fn from_conversation_result(result: ConversationSearchResult) -> ScopedSearchResult {
        let score = Self::rank_score(result.vec_distance, result.fts_score);
        let title = result
            .title
            .clone()
            .or_else(|| Some(format!("Conversation {}", result.uuid)));
        let content = result.summary.unwrap_or_default();
        ScopedSearchResult {
            hit_type: SearchHitType::ConversationSummary,
            id: result.id.to_string(),
            title,
            content,
            created_at: Some(result.updated_at),
            score,
            conversation_uuid: Some(result.uuid),
            message_index: None,
            role: None,
        }
    }

    fn rank_score(vec_distance: Option<f32>, fts_score: Option<f32>) -> f32 {
        let vec_component = vec_distance
            .map(|distance| 1.0 / (1.0 + distance.max(0.0)))
            .unwrap_or(0.0);
        let fts_component = fts_score
            .map(|score| (score / 10.0).min(1.0))
            .unwrap_or(0.0);
        (vec_component * 0.7) + (fts_component * 0.3)
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

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
}
