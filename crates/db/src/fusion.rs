//! Deterministic rank fusion for heterogeneous retrieval results.
//!
//! Reciprocal-rank fusion (RRF) intentionally consumes ranks rather than raw
//! vector distances and BM25 values: those values do not share a stable scale
//! across indexes or hit types.  The legacy weighted strategy is retained only
//! for controlled comparison during migration.

use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap};
use surrealdb_types::SurrealValue;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FusionStrategy {
    #[default]
    ReciprocalRank,
    Weighted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionConfig {
    pub strategy: FusionStrategy,
    pub rrf_k: usize,
    pub vector_weight: f32,
    pub fulltext_weight: f32,
    pub candidate_pool_multiplier: usize,
    pub candidate_pool_min: usize,
    pub candidate_pool_max: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::ReciprocalRank,
            // 60 is the original RRF paper's conservative default. It makes
            // one high rank from either independent retriever meaningful
            // without letting a single list dominate the other.
            rrf_k: 60,
            vector_weight: 0.7,
            fulltext_weight: 0.3,
            candidate_pool_multiplier: 4,
            candidate_pool_min: 50,
            candidate_pool_max: 200,
        }
    }
}

impl FusionConfig {
    pub fn candidate_limit(&self, requested_limit: usize) -> usize {
        requested_limit
            .saturating_mul(self.candidate_pool_multiplier)
            .clamp(self.candidate_pool_min, self.candidate_pool_max)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, SurrealValue)]
pub struct FusionEvidence {
    pub vector_rank: Option<usize>,
    pub vector_distance: Option<f32>,
    pub fulltext_rank: Option<usize>,
    pub fulltext_score: Option<f32>,
    pub fused_score: f32,
    pub final_rank: usize,
}

/// The minimum surface a database result needs to participate in fusion.
/// Implementations keep the source record's payload while this module owns all
/// scoring and deterministic ordering behavior.
pub trait FusionRecord: Clone {
    fn fusion_id(&self) -> String;
    fn vector_distance(&self) -> Option<f32>;
    fn fulltext_score(&self) -> Option<f32>;
    fn set_fusion_evidence(&mut self, evidence: FusionEvidence);
}

#[derive(Debug, Clone)]
struct Candidate<T> {
    result: T,
    vector_rank: Option<usize>,
    fulltext_rank: Option<usize>,
}

/// Merge vector and full-text candidate lists and return a deterministic,
/// ranked result set. `merge` fills payload fields that are absent from the
/// result initially returned by the vector query.
pub fn fuse<T, F>(vector: Vec<T>, fulltext: Vec<T>, config: &FusionConfig, mut merge: F) -> Vec<T>
where
    T: FusionRecord,
    F: FnMut(&mut T, T),
{
    let mut candidates: BTreeMap<String, Candidate<T>> = BTreeMap::new();

    for (index, result) in vector.into_iter().enumerate() {
        let id = result.fusion_id();
        candidates
            .entry(id)
            .and_modify(|candidate| {
                if candidate.vector_rank.is_none() {
                    candidate.vector_rank = Some(index + 1);
                }
            })
            .or_insert(Candidate {
                result,
                vector_rank: Some(index + 1),
                fulltext_rank: None,
            });
    }

    for (index, result) in fulltext.into_iter().enumerate() {
        let id = result.fusion_id();
        candidates
            .entry(id)
            .and_modify(|candidate| {
                candidate.fulltext_rank.get_or_insert(index + 1);
                merge(&mut candidate.result, result.clone());
            })
            .or_insert(Candidate {
                result,
                vector_rank: None,
                fulltext_rank: Some(index + 1),
            });
    }

    let mut fused: Vec<(String, T, FusionEvidence)> = candidates
        .into_iter()
        .map(|(id, candidate)| {
            let evidence = FusionEvidence {
                vector_rank: candidate.vector_rank,
                vector_distance: candidate.result.vector_distance(),
                fulltext_rank: candidate.fulltext_rank,
                fulltext_score: candidate.result.fulltext_score(),
                fused_score: fused_score(
                    candidate.vector_rank,
                    candidate.result.vector_distance(),
                    candidate.fulltext_rank,
                    candidate.result.fulltext_score(),
                    config,
                ),
                final_rank: 0,
            };
            (id, candidate.result, evidence)
        })
        .collect();

    // Contract: score descending, then the strongest component rank, then
    // canonical record id. The BTreeMap also eliminates HashMap iteration
    // order before this comparison is applied.
    fused.sort_by(|(left_id, _, left), (right_id, _, right)| {
        right
            .fused_score
            .total_cmp(&left.fused_score)
            .then_with(|| best_rank(left).cmp(&best_rank(right)))
            .then_with(|| left_id.cmp(right_id))
    });

    fused
        .into_iter()
        .enumerate()
        .map(|(index, (_, mut result, mut evidence))| {
            evidence.final_rank = index + 1;
            result.set_fusion_evidence(evidence);
            result
        })
        .collect()
}

pub fn fused_score(
    vector_rank: Option<usize>,
    vector_distance: Option<f32>,
    fulltext_rank: Option<usize>,
    fulltext_score: Option<f32>,
    config: &FusionConfig,
) -> f32 {
    match config.strategy {
        FusionStrategy::ReciprocalRank => {
            let k = config.rrf_k as f32;
            vector_rank
                .map(|rank| config.vector_weight / (k + rank as f32))
                .unwrap_or_default()
                + fulltext_rank
                    .map(|rank| config.fulltext_weight / (k + rank as f32))
                    .unwrap_or_default()
        }
        FusionStrategy::Weighted => {
            let vector = vector_distance
                .map(|distance| 1.0 / (1.0 + distance.max(0.0)))
                .unwrap_or_default();
            let fulltext = fulltext_score
                .map(|score| (score / 10.0).clamp(0.0, 1.0))
                .unwrap_or_default();
            vector * config.vector_weight + fulltext * config.fulltext_weight
        }
    }
}

pub fn best_rank(evidence: &FusionEvidence) -> usize {
    evidence
        .vector_rank
        .into_iter()
        .chain(evidence.fulltext_rank)
        .min()
        .unwrap_or(usize::MAX)
}

/// Score and order already-fused results from distinct hit types. The caller
/// supplies a fixed hit-type ordinal (note, message, summary) and canonical
/// record id; this keeps `scope=all` deterministic without comparing raw
/// retriever scores from different tables.
pub fn compare_scoped(
    left_score: f32,
    left: &FusionEvidence,
    left_hit_type: usize,
    left_id: &str,
    right_score: f32,
    right: &FusionEvidence,
    right_hit_type: usize,
    right_id: &str,
) -> Ordering {
    right_score
        .total_cmp(&left_score)
        .then_with(|| best_rank(left).cmp(&best_rank(right)))
        .then_with(|| left_hit_type.cmp(&right_hit_type))
        .then_with(|| left_id.cmp(right_id))
}

pub fn apply_hit_type_weight(evidence: &FusionEvidence, weight: f32) -> f32 {
    evidence.fused_score * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Result {
        id: &'static str,
        vector: Option<f32>,
        fulltext: Option<f32>,
        evidence: FusionEvidence,
    }

    impl FusionRecord for Result {
        fn fusion_id(&self) -> String {
            self.id.into()
        }
        fn vector_distance(&self) -> Option<f32> {
            self.vector
        }
        fn fulltext_score(&self) -> Option<f32> {
            self.fulltext
        }
        fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
            self.evidence = evidence;
        }
    }

    fn result(id: &'static str, vector: Option<f32>, fulltext: Option<f32>) -> Result {
        Result {
            id,
            vector,
            fulltext,
            evidence: FusionEvidence::default(),
        }
    }

    #[test]
    fn rrf_uses_hand_calculated_weighted_ranks() {
        let config = FusionConfig {
            rrf_k: 10,
            vector_weight: 0.7,
            fulltext_weight: 0.3,
            ..FusionConfig::default()
        };
        let score = fused_score(Some(1), Some(0.2), Some(2), Some(3.0), &config);
        assert!((score - (0.7 / 11.0 + 0.3 / 12.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn retains_one_component_candidates_and_evidence() {
        let results = fuse(
            vec![result("vector", Some(0.1), None)],
            vec![result("text", None, Some(2.0))],
            &FusionConfig::default(),
            |_, _| {},
        );
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|result| result.id == "vector"
            && result.evidence.vector_rank == Some(1)
            && result.evidence.fulltext_rank.is_none()));
        assert!(results.iter().any(|result| result.id == "text"
            && result.evidence.fulltext_rank == Some(1)
            && result.evidence.vector_rank.is_none()));
    }

    #[test]
    fn merges_duplicates_and_breaks_ties_by_canonical_id() {
        let results = fuse(
            vec![result("b", Some(0.1), None)],
            vec![result("a", None, Some(1.0))],
            &FusionConfig {
                vector_weight: 0.5,
                fulltext_weight: 0.5,
                ..FusionConfig::default()
            },
            |existing, incoming| existing.fulltext = incoming.fulltext,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[0].evidence.final_rank, 1);
        assert_eq!(results[1].evidence.final_rank, 2);
    }

    #[test]
    fn candidate_pool_is_bounded_and_uses_requested_limit() {
        let config = FusionConfig {
            candidate_pool_multiplier: 4,
            candidate_pool_min: 3,
            candidate_pool_max: 20,
            ..FusionConfig::default()
        };
        assert_eq!(config.candidate_limit(1), 4);
        assert_eq!(config.candidate_limit(5), 20);
        assert_eq!(config.candidate_limit(100), 20);
    }

    #[test]
    fn cross_hit_type_order_uses_weighted_fusion_then_type() {
        let evidence = FusionEvidence {
            fused_score: 0.5,
            vector_rank: Some(1),
            ..FusionEvidence::default()
        };
        let note_score = apply_hit_type_weight(&evidence, 1.0);
        let message_score = apply_hit_type_weight(&evidence, 0.8);
        assert_eq!(
            compare_scoped(
                note_score,
                &evidence,
                0,
                "note:a",
                message_score,
                &evidence,
                1,
                "message:a"
            ),
            Ordering::Less
        );

        // Equal weighted scores prefer the documented hit-type ordinal before
        // their canonical record id.
        assert_eq!(
            compare_scoped(
                note_score,
                &evidence,
                0,
                "note:z",
                note_score,
                &evidence,
                1,
                "message:a"
            ),
            Ordering::Less
        );
    }
}
