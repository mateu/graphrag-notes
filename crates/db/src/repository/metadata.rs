//! Embedding/extraction metadata ownership.
//!
//! These methods delegate to the compatibility module and deliberately do not
//! write migrations or mutate an established model identity.

use super::*;

impl Repository {
    /// Check the active embedding identity before a vector read or write.
    /// This method is read-only and therefore safe to use for search paths.
    pub async fn check_embedding_compatibility(
        &self,
        embedding: &EmbeddingIdentity,
    ) -> Result<CompatibilityState> {
        check_embedding_compatibility(&self.db, embedding).await
    }

    /// Initialize empty-corpus metadata after a successful embedding probe.
    /// Existing metadata is never overwritten by a different model.
    pub async fn record_embedding_metadata(
        &self,
        embedding: &EmbeddingIdentity,
        extraction: Option<&ExtractionIdentity>,
    ) -> Result<CompatibilityState> {
        record_embedding_metadata(&self.db, embedding, extraction).await
    }
}
