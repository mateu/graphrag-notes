//! Search Agent - Handles user queries with hybrid search

use crate::{MlClient, Result};
use graphrag_db::Repository;
use graphrag_db::repository::{SearchResult, RelatedNotes, SimilarNote};
use tracing::{info, debug, instrument};

/// Search result with optional graph context
#[derive(Debug)]
pub struct EnrichedSearchResult {
    pub result: SearchResult,
    pub related: Option<RelatedNotes>,
}

/// The Search agent handles user queries
pub struct SearchAgent {
    repo: Repository,
    ml: MlClient,
}

impl SearchAgent {
    /// Create a new Search agent
    pub fn new(repo: Repository, ml: MlClient) -> Self {
        Self { repo, ml }
    }
    
    /// Perform hybrid search (vector + full-text)
    #[instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        info!("Searching for: {}", query);
        
        // Generate query embedding
        debug!("Generating query embedding...");
        let embedding = self.ml.embed_one(query).await?;
        
        // Perform hybrid search
        let results = self.repo.hybrid_search(query, embedding, limit).await?;
        
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
        let results = self.search(query, limit).await?;
        
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
    pub async fn semantic_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        debug!("Performing semantic search for: {}", query);
        
        let embedding = self.ml.embed_one(query).await?;
        let results = self.repo.vector_search(embedding, limit).await?;
        
        Ok(results)
    }
    
    /// Full-text only search (keyword matching)
    #[instrument(skip(self))]
    pub async fn keyword_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        debug!("Performing keyword search for: {}", query);
        
        let results = self.repo.fulltext_search(query, limit).await?;
        
        Ok(results)
    }
    
    /// Find notes similar to a given note
    #[instrument(skip(self))]
    pub async fn find_similar(
        &self,
        note_id: &str,
        limit: usize,
    ) -> Result<Vec<SimilarNote>> {
        // First get the note to get its embedding
        let note = self.repo.get_note(note_id).await?
            .ok_or_else(|| crate::AgentError::NotFound(format!("Note {}", note_id)))?;
        
        if note.embedding.is_empty() {
            return Err(crate::AgentError::NotFound(
                "Note has no embedding".into()
            ));
        }
        
        let similar = self.repo
            .find_similar_notes(note_id, note.embedding, 0.5, limit)
            .await?;
        
        Ok(similar)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
}
