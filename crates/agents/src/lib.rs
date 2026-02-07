//! AI Agents for GraphRAG Notes
//!
//! This crate contains the agent implementations:
//! - Librarian: Ingests content and creates notes
//! - Search: Handles user queries with hybrid search
//! - Gardener: Maintains graph connections

pub mod error;
pub mod gardener;
pub mod inference;
pub mod librarian;
pub mod search;

pub use error::{AgentError, Result};
pub use gardener::GardenerAgent;
pub use inference::{
    EntityExtraction, ExtractedEntity, ExtractedRelationship, TeiClient, TgiClient,
};
pub use librarian::{ChatImportMode, ChatImportResult, LibrarianAgent};
pub use search::SearchAgent;
