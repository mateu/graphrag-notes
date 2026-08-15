//! Graph-maintenance ownership.
//!
//! `service` owns repository-backed maintenance orchestration. Proposal and
//! policy surfaces are kept explicit for future private implementation moves.

pub mod service;

/// Proposal and scan report compatibility surface.
pub mod proposals {
    pub use super::service::{ScanReport, SuggestedConnection};
}

/// Acceptance and maintenance policy compatibility surface.
pub mod policy {
    pub use super::service::MaintenanceReport;
}

pub use service::*;
