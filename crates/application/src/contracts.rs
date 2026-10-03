//! Transport-neutral local application requests and results.

use graphrag_db::InspectionProvenance;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub const APPLICATION_CONTRACT_VERSION: u32 = 1;
pub const MAX_RESULT_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct RecordRef {
    pub id: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Keyword,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphPolicy {
    Off,
    Auto,
    On,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub scope: Scope,
    pub limit: usize,
    pub graph: GraphPolicy,
    pub since_days: Option<u32>,
    pub source_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordSummary {
    pub id: String,
    /// Missing only when a record changed or disappeared during enrichment.
    pub revision: Option<String>,
    pub hit_type: String,
    pub title: Option<String>,
    pub content: String,
    pub provenance: Option<InspectionProvenance>,
    pub score: Option<f32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRequest {
    pub content: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStatus {
    pub id: String,
    pub title: Option<String>,
    /// Identifies a corpus-owner source; it is not automatically a client path.
    pub uri: Option<String>,
    pub status: String,
    pub generation: u64,
    pub successful_generation: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalEndpoint {
    pub id: String,
    pub available: bool,
    pub title: Option<String>,
    pub excerpt: Option<String>,
    pub revision: Option<String>,
    pub provenance: Option<InspectionProvenance>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalCard {
    pub id: String,
    pub status: graphrag_core::ProposedEdgeStatus,
    pub edge_type: String,
    pub confidence: f32,
    pub reason: String,
    pub generator: String,
    pub generator_version: Option<String>,
    pub model: Option<String>,
    pub updated_at: String,
    pub reviewed_at: Option<String>,
    pub reviewer: Option<String>,
    pub action_reason: Option<String>,
    pub acceptance_is_manual: Option<bool>,
    pub supersession_reason: Option<String>,
    pub resulting_edge_id: Option<String>,
    pub from: ProposalEndpoint,
    pub to: ProposalEndpoint,
    pub accept_allowed: bool,
    pub accept_blocked_reason: Option<String>,
    pub revision: String,
}

/// Client interaction chooses an explicit decision before invoking the policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalAction {
    Accept,
    Reject,
    Undo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalDecisionRequest {
    pub id: String,
    pub revision: String,
    pub action: ProposalAction,
    pub reason: Option<String>,
    pub confirmed: bool,
    /// Trusted adapter-supplied audit identity; future remote adapters derive it
    /// from authentication rather than accepting it from a tool request.
    pub reviewer: String,
}

/// One action owns one fresh flag; providers and database connections remain shared.
#[derive(Debug, Clone, Default)]
pub struct ActionCancellation(Arc<AtomicBool>);

impl ActionCancellation {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
    pub async fn cancelled(&self) {
        while !self.is_cancelled() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
}
