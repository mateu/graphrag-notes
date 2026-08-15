//! Narrow explainability rendering shared by future CLI output surfaces.

use graphrag_agents::{AugmentDiagnostics, GraphRetrievalSummary, RetrievalExplanation};

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
) -> serde_json::Value {
    let mut output = json(explanations);
    output["pipeline"] = serde_json::json!({
        "rendered_tokens": total_tokens,
        "diagnostics": diagnostics,
        "filters": filters,
    });
    output
}

pub fn search_json(
    explanations: &[RetrievalExplanation],
    summary: &GraphRetrievalSummary,
    filters: serde_json::Value,
) -> serde_json::Value {
    let mut output = json(explanations);
    output["pipeline"] = serde_json::json!({ "graph": summary, "filters": filters });
    output
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
        "rank={} fused={:.4} channels={} decision={:?}",
        explanation.rank, explanation.fused.value, channels, explanation.inclusion
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
            rank: 1,
            context_rank: Some(1),
            hit_type: SearchHitTypeEvidence::Note,
            fused: ScoreEvidence {
                value: 0.5,
                meaning: "fusion",
                rank: None,
                raw_value: None,
            },
            vector: Some(ScoreEvidence {
                value: 1.0,
                meaning: "rank",
                rank: Some(1),
                raw_value: Some(0.1),
            }),
            full_text: None,
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
        assert!(human(&evidence).contains("channels=vector"));
        let json = json(&[evidence]);
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["results"][0]["vector"]["rank"], 1);
        assert!((json["results"][0]["vector"]["raw_value"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    }
}
