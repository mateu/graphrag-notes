//! Versioned, presentation-neutral retrieval evidence.
//!
//! This module deliberately describes observed retrieval and packing decisions;
//! it never participates in ranking or context selection.

use crate::{GraphEvidence, SearchHitType};
use graphrag_db::fusion::{FusionEvidence, FusionStrategy};
use serde::Serialize;

pub const EXPLANATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InclusionReason {
    Selected,
    Filtered,
    Duplicate,
    EmptyContent,
    NearDuplicate,
    RelevanceThreshold,
    TokenBudget,
    DiagnosticLimit,
}

/// The concrete algorithm behind a score. Consumers must not infer this from
/// score magnitude because RRF, weighted fusion, graph traversal, distance,
/// and BM25 live on different scales.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoreKind {
    ReciprocalRankFusion,
    WeightedFusion,
    GraphTraversal,
    VectorDistance,
    Bm25,
}

impl From<FusionStrategy> for ScoreKind {
    fn from(value: FusionStrategy) -> Self {
        match value {
            FusionStrategy::ReciprocalRank => Self::ReciprocalRankFusion,
            FusionStrategy::Weighted => Self::WeightedFusion,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScoreEvidence {
    pub value: f32,
    pub kind: ScoreKind,
    /// Stable description of the score scale, never an unexplained float.
    pub meaning: &'static str,
    /// Position in the corresponding retrieval channel, when that channel
    /// contributed the result. The final fused position lives on the parent
    /// explanation as `rank`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Vector distance or BM25 score, depending on `meaning`. Keeping the
    /// channel-native value alongside its rank lets a consumer explain a
    /// result without treating reciprocal rank as a similarity score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_value: Option<f32>,
}

/// Packing relevance after normalizing the eligible retrieval candidate pool.
/// Present only when a candidate is excluded by the relevance threshold.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RelevanceEvidence {
    pub normalized: f32,
    pub threshold: f32,
}

/// Observed MMR inputs and outcome for a chunk admitted to prompt context.
/// `novelty` is the candidate's minimum novelty against the chunks already
/// selected at the moment this decision was made.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SelectionEvidence {
    pub normalized_relevance: f32,
    pub novelty: f32,
    pub score: f32,
}

/// The selected chunk that caused a candidate to be omitted as too similar.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NearDuplicateEvidence {
    pub matching_result_id: String,
    pub jaccard_similarity: f32,
    pub threshold: f32,
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
    /// Stable record/chunk identity that correlates evidence with result data
    /// without requiring consumers to match rank or score floats.
    pub result_id: String,
    pub title: Option<String>,
    /// Final position after hybrid/graph retrieval fusion.
    pub rank: usize,
    /// Position after augmentation packing, when this result was selected for
    /// prompt context. This is distinct from the final retrieval rank because
    /// diversity and token-budget decisions can reorder or omit candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_rank: Option<usize>,
    pub hit_type: SearchHitTypeEvidence,
    /// The final hit-type-weighted score used by the shared result sorter.
    /// This, rather than the unweighted fused channel score, determines rank.
    pub final_score: ScoreEvidence,
    /// Configured effective hit-type weight applied to `fused` to produce
    /// `final_score` (or to graph traversal score for graph-only hits).
    pub effective_weight: f32,
    pub fused: ScoreEvidence,
    pub vector: Option<ScoreEvidence>,
    pub full_text: Option<ScoreEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevance: Option<RelevanceEvidence>,
    /// Present for selected context chunks so consumers can inspect the
    /// observed MMR decision rather than inferring it from retrieval rank.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<SelectionEvidence>,
    /// Present when the candidate was excluded because it matched an already
    /// selected context chunk closely enough to cross the configured limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub near_duplicate: Option<NearDuplicateEvidence>,
    pub graph: Option<GraphEvidence>,
    pub inclusion: InclusionReason,
    pub token_count: Option<usize>,
    /// Query embedding deployment identity, if the caller has it. This is
    /// metadata only and is deliberately not inferred from cache contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    pub provenance: ProvenanceEvidence,
}

impl RetrievalExplanation {
    pub fn with_embedding_identity(mut self, provider: &str, model: &str) -> Self {
        self.embedding_provider = Some(provider.to_owned());
        self.embedding_model = Some(model.to_owned());
        self
    }
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
    fusion_kind: ScoreKind,
) -> (ScoreEvidence, Option<ScoreEvidence>, Option<ScoreEvidence>) {
    (
        ScoreEvidence {
            value: fusion.fused_score,
            kind: fusion_kind,
            meaning: match fusion_kind {
                ScoreKind::ReciprocalRankFusion => "reciprocal-rank fusion score",
                ScoreKind::WeightedFusion => "weighted vector-distance and BM25 fusion score",
                ScoreKind::GraphTraversal => "accepted-edge graph traversal score",
                ScoreKind::VectorDistance | ScoreKind::Bm25 => {
                    unreachable!("only fusion score kinds are valid for fused evidence")
                }
            },
            rank: None,
            raw_value: None,
        },
        fusion.vector_rank.map(|rank| ScoreEvidence {
            value: rank as f32,
            kind: ScoreKind::VectorDistance,
            meaning: "vector retrieval rank; lower is better; raw_value is distance",
            rank: Some(rank),
            raw_value: fusion.vector_distance,
        }),
        fusion.fulltext_rank.map(|rank| ScoreEvidence {
            value: rank as f32,
            kind: ScoreKind::Bm25,
            meaning: "full-text retrieval rank; lower is better; raw_value is BM25 score",
            rank: Some(rank),
            raw_value: fusion.fulltext_score,
        }),
    )
}

pub fn final_rank_score(value: f32, kind: ScoreKind) -> ScoreEvidence {
    ScoreEvidence {
        value,
        kind,
        meaning: "hit-type-weighted final rank score",
        rank: None,
        raw_value: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_evidence_uses_stable_score_meanings_and_schema_version() {
        let fusion = FusionEvidence {
            vector_rank: Some(2),
            vector_distance: Some(0.12),
            fulltext_rank: Some(3),
            fulltext_score: Some(8.5),
            fused_score: 0.42,
            ..Default::default()
        };
        let (fused, vector, full_text) = fusion_scores(&fusion, ScoreKind::WeightedFusion);
        assert_eq!(fused.value, 0.42);
        assert_eq!(fused.kind, ScoreKind::WeightedFusion);
        assert_eq!(
            fused.meaning,
            "weighted vector-distance and BM25 fusion score"
        );
        let vector = vector.unwrap();
        assert_eq!(
            vector.meaning,
            "vector retrieval rank; lower is better; raw_value is distance"
        );
        assert_eq!(vector.rank, Some(2));
        assert_eq!(vector.raw_value, Some(0.12));
        let full_text = full_text.unwrap();
        assert_eq!(
            full_text.meaning,
            "full-text retrieval rank; lower is better; raw_value is BM25 score"
        );
        assert_eq!(full_text.rank, Some(3));
        assert_eq!(full_text.raw_value, Some(8.5));
        assert_eq!(EXPLANATION_SCHEMA_VERSION, 1);
    }

    #[test]
    fn score_kind_distinguishes_rrf_weighted_and_graph_scores() {
        assert_eq!(
            ScoreKind::from(FusionStrategy::ReciprocalRank),
            ScoreKind::ReciprocalRankFusion
        );
        assert_eq!(
            ScoreKind::from(FusionStrategy::Weighted),
            ScoreKind::WeightedFusion
        );
        let (graph, _, _) = fusion_scores(
            &FusionEvidence {
                fused_score: 0.7,
                ..Default::default()
            },
            ScoreKind::GraphTraversal,
        );
        assert_eq!(graph.kind, ScoreKind::GraphTraversal);
        assert_eq!(graph.meaning, "accepted-edge graph traversal score");
    }
}
