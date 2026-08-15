//! Ingestion pipeline ownership.
//!
//! The [`librarian`] service owns repository orchestration. The sibling
//! modules document the stable lanes within that service so future mechanical
//! moves do not change ingestion behavior or public constructors.

pub mod librarian;

/// Markdown chunking, note construction, and source reconciliation lane.
pub mod markdown {
    pub use super::librarian::MarkdownImportResult;
}

/// Chat preview and import lane.
pub mod chats {
    pub use super::librarian::{
        ChatImportMode, ChatImportPreview, ChatImportResult, ChatIngestOptions,
    };
}

/// Entity extraction and mention replacement lane.
pub mod entities {}

/// Source-generation and markdown reconciliation lane.
pub mod reconcile {}

/// Durable processing-job progress lane.
pub mod progress {
    pub use super::librarian::ProcessingRunResult;
}

pub use librarian::*;
