//! Retrieval and context-packing ownership.
//!
//! `service` performs repository-backed orchestration. Candidate ranking,
//! graph traversal, packing, and evidence are deliberately surfaced as
//! separate lanes; the latter two remain canonical modules during this
//! mechanical move so existing imports stay source-compatible.

pub mod service;

/// Retrieval result identities and ranking candidates.
pub mod candidates {
    pub use super::service::{
        EnrichedSearchResult, ScopedSearchResult, SearchHitType, SearchScope,
    };
}

/// Hybrid fusion score evidence.
pub mod fusion {
    pub use crate::{final_rank_score, fusion_scores, ScoreEvidence, ScoreKind};
}

/// Graph traversal configuration and evidence.
pub mod graph {
    pub use super::service::{
        GraphEvidence, GraphMode, GraphPathStep, GraphRetrievalConfig, GraphRetrievalSummary,
        GraphSearchResults,
    };
}

/// Pure token budgeting and MMR context packing.
pub mod packing {
    pub use crate::context_packing::*;
}

/// Versioned retrieval and packing explanation schema.
pub mod evidence {
    pub use crate::evidence::*;
}

pub use service::*;
