//! Graph-maintenance ownership.
//!
//! `service` owns repository-backed maintenance orchestration. Proposal and
//! policy surfaces are kept explicit for future private implementation moves.
//! `candidates` owns the bounded, high-precision candidate scan.

mod candidate_quality;
#[cfg(test)]
mod candidate_quality_scan_tests;
#[cfg(test)]
mod candidate_scan_tests;
pub mod candidates;
pub mod service;

/// Proposal and scan report compatibility surface.
pub mod proposals {
    pub use super::service::{ScanReport, SuggestedConnection};
}

/// Acceptance and maintenance policy compatibility surface.
pub mod policy {
    pub use super::service::MaintenanceReport;
}

pub use candidates::{
    CandidatePolicy, CandidateRejections, CandidateScanReport, CandidateScanRequest,
    CandidateScope, CandidateSignal, CandidateStopReason, CrossReferenceDirection,
    CrossReferenceEvidence, CrossReferenceKind, RelationshipCandidate, SharedEntityEvidence,
    TextualCueEvidence, TextualCueKind, CANDIDATE_CONTRACT, CANDIDATE_GENERATOR,
};
pub use service::*;
