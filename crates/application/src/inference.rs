//! Cancel provider preparation without abandoning an atomic note transaction.

use crate::ActionCancellation;
use async_trait::async_trait;
use graphrag_agents::{
    AgentError, Embedder, EntityExtraction, EntityExtractor, InferenceCapabilities, SharedEmbedder,
    SharedEntityExtractor,
};

pub(crate) struct CancellableEmbedder(pub SharedEmbedder, pub ActionCancellation);
pub(crate) struct CancellableExtractor(pub SharedEntityExtractor, pub ActionCancellation);

#[async_trait]
impl Embedder for CancellableEmbedder {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        tokio::select! {
            biased;
            _ = self.1.cancelled() => Err(AgentError::Cancelled),
            result = self.0.embed(text, query) => result,
        }
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        query: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        tokio::select! {
            biased;
            _ = self.1.cancelled() => Err(AgentError::Cancelled),
            result = self.0.embed_batch(texts, query) => result,
        }
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        self.0.health().await
    }
    fn capabilities(&self) -> InferenceCapabilities {
        self.0.capabilities()
    }
    fn max_batch_size(&self) -> Option<usize> {
        self.0.max_batch_size()
    }
}

#[async_trait]
impl EntityExtractor for CancellableExtractor {
    async fn extract(&self, text: &str) -> graphrag_agents::Result<EntityExtraction> {
        tokio::select! {
            biased;
            _ = self.1.cancelled() => Err(AgentError::Cancelled),
            result = self.0.extract(text) => result,
        }
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        self.0.health().await
    }
    fn capabilities(&self) -> InferenceCapabilities {
        self.0.capabilities()
    }
}
