//! AI Agents for GraphRAG Notes
//!
//! This crate contains the agent implementations:
//! - Librarian: Ingests content and creates notes
//! - Search: Handles user queries with hybrid search
//! - Gardener: Maintains graph connections

pub mod chunking;
pub mod context_packing;
pub mod error;
pub mod gardener;
pub mod inference;
pub mod librarian;
pub mod processing;
pub mod reindex;
pub mod search;

pub use chunking::{Chunk, Chunker, ChunkingConfig, MarkdownChunker};
pub use error::{AgentError, Result};
pub use gardener::GardenerAgent;
pub use inference::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, ExtractedEntity,
    ExtractedRelationship, FixtureEntityExtractor, InferenceCapabilities, InferenceProviderConfig,
    InferenceProviders, SharedEmbedder, SharedEntityExtractor, TeiClient, TgiClient,
};
pub use librarian::{
    ChatImportMode, ChatImportPreview, ChatImportResult, ChatIngestOptions, LibrarianAgent,
    LibrarianRuntimeConfig, MarkdownImportResult, ProcessingRunResult,
};
pub use processing::{
    classify_retry, retry_delay, ProcessingConfig, ProcessingStatsSnapshot, ResilientEmbedder,
    ResilientEntityExtractor, RetryClassification,
};
pub use reindex::{ReindexAgent, ReindexPreview, ReindexResult, ReindexScope};
pub use search::{
    AugmentContext, AugmentDiagnostics, AugmentOptions, ConservativeTokenCounter, GraphEvidence,
    GraphMode, GraphPathStep, GraphRetrievalConfig, GraphRetrievalSummary, GraphSearchResults,
    SearchAgent, SearchHitType, SearchScope, TokenCountMode, TokenCounter,
};
