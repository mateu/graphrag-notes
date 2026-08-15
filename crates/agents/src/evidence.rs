//! Versioned, presentation-neutral retrieval evidence.
//!
//! This module deliberately describes observed retrieval and packing decisions;
//! it never participates in ranking or context selection.

use crate::{GraphEvidence, SearchHitType};
use graphrag_db::fusion::FusionEvidence;
use serde::Serialize;

pub const EXPLANATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InclusionReason {
    Selected,
    Filtered,
    Duplicate,
    NearDuplicate,
    RelevanceThreshold,
    TokenBudget,
    DiagnosticLimit,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScoreEvidence {
    pub value: f32,
    /// Stable description of the score scale, never an unexplained float.
    pub meaning: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProvenanceEvidence {
    pub source_uri: Option<String>,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    pub selected_span_start: Option<usize>,
    pub selected_span_end: Option<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RetrievalExplanation {
    pub schema_version: u32,
    pub rank: usize,
    pub hit_type: SearchHitTypeEvidence,
    pub fused: ScoreEvidence,
    pub vector: Option<ScoreEvidence>,
    pub full_text: Option<ScoreEvidence>,
    pub graph: Option<GraphEvidence>,
    pub inclusion: InclusionReason,
    pub token_count: Option<usize>,
    pub provenance: ProvenanceEvidence,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchHitTypeEvidence {
    Note,
    Message,
    ConversationSummary,
}

impl From<SearchHitType> for SearchHitTypeEvidence {
    fn from(value: SearchHitType) -> Self {
        match value {
            SearchHitType::Note => Self::Note,
            SearchHitType::Message => Self::Message,
            SearchHitType::ConversationSummary => Self::ConversationSummary,
        }
    }
}

pub fn fusion_scores(
    fusion: &FusionEvidence,
) -> (ScoreEvidence, Option<ScoreEvidence>, Option<ScoreEvidence>) {
    (
        ScoreEvidence {
            value: fusion.fused_score,
            meaning: "weighted reciprocal-rank fusion score",
        },
        fusion.vector_rank.map(|rank| ScoreEvidence {
            value: rank as f32,
            meaning: "vector retrieval rank; lower is better",
        }),
        fusion.fulltext_rank.map(|rank| ScoreEvidence {
            value: rank as f32,
            meaning: "full-text retrieval rank; lower is better",
        }),
    )
}
