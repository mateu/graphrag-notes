//! AI Agents for GraphRAG Notes
//!
//! ## Module ownership and call flow
//!
//! - [`inference`] owns provider contracts, configuration, and HTTP
//!   compatibility; it does not depend on repository orchestration.
//! - [`ingestion`] owns librarian orchestration from chunking through durable
//!   processing and reconciliation.
//! - [`search`] owns repository-backed retrieval, then delegates pure context
//!   selection to [`context_packing`] and explanation serialization to
//!   [`evidence`].
//! - [`gardener`] owns graph-maintenance orchestration and proposal policy.
//!
//! The end-to-end ingest path is: caller → `ingestion::librarian` → chunking
//! → inference traits → repository. The search path is: caller →
//! `search::service` → repository/fusion/graph → `context_packing` →
//! `evidence`. Public root re-exports below preserve the v0.2 API while these
//! private implementations are mechanically decomposed.

pub mod chunking;
pub mod context_packing;
pub mod error;
pub mod evidence;
#[path = "gardener/mod.rs"]
pub mod gardener;
#[path = "inference/mod.rs"]
pub mod inference;
pub mod ingestion;
pub use ingestion::librarian;
pub mod processing;
pub mod reindex;
#[path = "search/mod.rs"]
pub mod search;

pub use chunking::{Chunk, Chunker, ChunkingConfig, MarkdownChunker};
pub use error::{AgentError, Result};
pub use evidence::{
    final_rank_score, fusion_scores, InclusionReason, NearDuplicateEvidence, ProvenanceEvidence,
    RelevanceEvidence, RetrievalExplanation, ScoreEvidence, ScoreKind, SearchHitTypeEvidence,
    SelectionEvidence, EXPLANATION_SCHEMA_VERSION,
};
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
