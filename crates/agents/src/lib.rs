//! AI Agents for GraphRAG Notes
//!
//! This crate contains the agent implementations:
//! - Librarian: Ingests content and creates notes
//! - Search: Handles user queries with hybrid search
//! - Gardener: Maintains graph connections

pub mod context_packing;
pub mod error;
pub mod gardener;
pub mod inference;
pub mod librarian;
pub mod search;

pub use error::{AgentError, Result};
pub use gardener::GardenerAgent;
pub use inference::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, ExtractedEntity,
    ExtractedRelationship, FixtureEntityExtractor, InferenceCapabilities, InferenceProviderConfig,
    InferenceProviders, SharedEmbedder, SharedEntityExtractor, TeiClient, TgiClient,
};
pub use librarian::{
    ChatImportMode, ChatImportPreview, ChatImportResult, ChatIngestOptions, LibrarianAgent,
    LibrarianRuntimeConfig, MarkdownImportResult,
};
pub use search::{
    AugmentContext, AugmentDiagnostics, AugmentOptions, ConservativeTokenCounter, SearchAgent,
    SearchHitType, SearchScope, TokenCountMode, TokenCounter,
};
