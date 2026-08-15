//! Narrow explainability rendering shared by future CLI output surfaces.

use graphrag_agents::{
    AugmentDiagnostics, AugmentOptions, GraphRetrievalSummary, RetrievalExplanation,
};
use graphrag_db::repository::RelatedNotes;
use std::collections::HashMap;

/// Stable JSON payload: callers own envelopes/format negotiation, while this
/// renderer guarantees evidence comes from the shared schema.
pub fn json(explanations: &[RetrievalExplanation]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": graphrag_agents::EXPLANATION_SCHEMA_VERSION,
        "results": explanations,
    })
}

/// Explain output retains aggregate pipeline decisions in addition to each
/// selected result. The packer intentionally does not retain discarded text,
/// so these counts reveal drop reasons without exposing prompt content.
pub fn augmentation_json(
    explanations: &[RetrievalExplanation],
    diagnostics: &AugmentDiagnostics,
    total_tokens: usize,
    filters: serde_json::Value,
    options: &AugmentOptions,
) -> serde_json::Value {
    let mut output = json(explanations);
    output["pipeline"] = augmentation_pipeline(diagnostics, total_tokens, filters, options);
    output
}

pub fn augmentation_pipeline(
    diagnostics: &AugmentDiagnostics,
    total_tokens: usize,
    filters: serde_json::Value,
    options: &AugmentOptions,
) -> serde_json::Value {
    serde_json::json!({
        "rendered_tokens": total_tokens,
        "diagnostics": diagnostics,
        "filters": filters,
        "controls": {
            "max_chunks": options.max_chunks,
            "max_total_tokens": options.max_total_tokens,
            "max_chunk_tokens": options.max_chunk_tokens,
            "novelty_weight": options.novelty_weight,
            "min_relevance": options.min_relevance,
            "near_duplicate_threshold": options.near_duplicate_threshold,
        },
    })
}

pub fn search_json(
    explanations: &[RetrievalExplanation],
    summary: &GraphRetrievalSummary,
    filters: serde_json::Value,
    related_by_note: &HashMap<String, RelatedNotes>,
) -> serde_json::Value {
    let mut output = json(explanations);
    output["pipeline"] = search_pipeline(summary, filters, related_by_note);
    output
}

pub fn search_pipeline(
    summary: &GraphRetrievalSummary,
    filters: serde_json::Value,
    related_by_note: &HashMap<String, RelatedNotes>,
) -> serde_json::Value {
    serde_json::json!({
        "graph": summary,
        "filters": filters,
        "related": related_by_note,
    })
}

/// Compact human evidence line suitable for indentation beneath a result.
pub fn human(explanation: &RetrievalExplanation) -> String {
    let channels = [
        explanation.vector.as_ref().map(|_| "vector"),
        explanation.full_text.as_ref().map(|_| "full_text"),
        explanation.graph.as_ref().map(|_| "graph"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(",");
    format!(
        "id={} rank={} final={:.4} weight={:.3} fused={:.4} channels={} decision={:?}",
        explanation.result_id,
        explanation.rank,
        explanation.final_score.value,
        explanation.effective_weight,
        explanation.fused.value,
        channels,
        explanation.inclusion
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_agents::{
        InclusionReason, ProvenanceEvidence, RetrievalExplanation, ScoreEvidence,
        SearchHitTypeEvidence,
    };

    fn sample() -> RetrievalExplanation {
        RetrievalExplanation {
            schema_version: 1,
            result_id: "note:fixture".into(),
            title: Some("Fixture".into()),
            rank: 1,
            context_rank: Some(1),
            hit_type: SearchHitTypeEvidence::Note,
            final_score: ScoreEvidence {
                value: 0.5,
                kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                meaning: "final",
                rank: None,
                raw_value: None,
            },
            effective_weight: 0.7,
            fused: ScoreEvidence {
                value: 0.5,
                kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                meaning: "fusion",
                rank: None,
                raw_value: None,
            },
            vector: Some(ScoreEvidence {
                value: 1.0,
                kind: graphrag_agents::ScoreKind::VectorDistance,
                meaning: "rank",
                rank: Some(1),
                raw_value: Some(0.1),
            }),
            full_text: None,
            relevance: None,
            graph: None,
            inclusion: InclusionReason::Selected,
            token_count: Some(3),
            embedding_provider: Some("fixture".into()),
            embedding_model: Some("fixture-model".into()),
            provenance: ProvenanceEvidence {
                source_uri: None,
                conversation_uuid: None,
                message_index: None,
                role: None,
                selected_span_start: Some(0),
                selected_span_end: Some(3),
            },
        }
    }

    #[test]
    fn human_and_json_share_the_versioned_evidence_contract() {
        let evidence = sample();
        let human = human(&evidence);
        assert!(human.contains("id=note:fixture"));
        assert!(human.contains("final=0.5000"));
        assert!(human.contains("channels=vector"));
        let json = json(&[evidence]);
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["results"][0]["result_id"], "note:fixture");
        assert_eq!(
            json["results"][0]["fused"]["kind"],
            "reciprocal_rank_fusion"
        );
        assert_eq!(json["results"][0]["final_score"]["value"], 0.5);
        assert!((json["results"][0]["effective_weight"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(json["results"][0]["vector"]["rank"], 1);
        assert!((json["results"][0]["vector"]["raw_value"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn jsonl_pipelines_retain_the_same_aggregate_metadata_as_json() {
        let diagnostics = AugmentDiagnostics {
            token_count_mode: graphrag_agents::TokenCountMode::Estimated,
            header_tokens: 2,
            dropped_duplicates: 1,
            dropped_near_duplicates: 0,
            dropped_for_relevance: 0,
            dropped_for_budget: 3,
            dropped_for_entity_filter: 4,
            graph_candidates_considered: 5,
            graph_candidates_selected: 1,
            graph_candidates_dropped: 4,
        };
        let filters = serde_json::json!({"scope": "Notes", "entity": "Atlas"});
        let options = AugmentOptions {
            max_chunks: 3,
            max_total_tokens: 42,
            max_chunk_tokens: 12,
            novelty_weight: 0.25,
            min_relevance: 0.4,
            near_duplicate_threshold: 0.8,
            ..Default::default()
        };
        let augment = augmentation_pipeline(&diagnostics, 42, filters.clone(), &options);
        assert_eq!(augment["rendered_tokens"], 42);
        assert_eq!(augment["diagnostics"]["dropped_for_budget"], 3);
        assert_eq!(augment["filters"], filters);
        assert_eq!(augment["controls"]["max_chunks"], 3);
        assert!((augment["controls"]["min_relevance"].as_f64().unwrap() - 0.4).abs() < 1e-6);

        let graph = GraphRetrievalSummary {
            entities_matched: 2,
            candidates_considered: 5,
            candidates_selected: 3,
            candidates_dropped: 2,
        };
        let related = HashMap::from([("note:atlas".to_owned(), RelatedNotes::default())]);
        let search = search_pipeline(&graph, filters.clone(), &related);
        assert_eq!(search["graph"]["candidates_considered"], 5);
        assert_eq!(search["filters"], filters);
        assert!(search["related"]["note:atlas"].is_object());
    }
}
