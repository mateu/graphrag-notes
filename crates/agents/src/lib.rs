//! AI Agents for GraphRAG Notes
//! 
//! This crate contains the agent implementations:
//! - Librarian: Ingests content and creates notes
//! - Search: Handles user queries with hybrid search
//! - Gardener: Maintains graph connections

pub mod librarian;
pub mod search;
pub mod gardener;
pub mod ml_client;
pub mod error;

pub use librarian::LibrarianAgent;
pub use search::SearchAgent;
pub use gardener::GardenerAgent;
pub use ml_client::MlClient;
pub use error::{AgentError, Result};
