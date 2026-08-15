//! Narrow explainability rendering shared by future CLI output surfaces.

use graphrag_agents::RetrievalExplanation;

/// Stable JSON payload: callers own envelopes/format negotiation, while this
/// renderer guarantees evidence comes from the shared schema.
pub fn json(explanations: &[RetrievalExplanation]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": graphrag_agents::EXPLANATION_SCHEMA_VERSION,
        "results": explanations,
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
            hit_type: SearchHitTypeEvidence::Note,
            fused: ScoreEvidence {
                value: 0.5,
                meaning: "fusion",
            },
            vector: Some(ScoreEvidence {
                value: 1.0,
                meaning: "rank",
            }),
            full_text: None,
            graph: None,
            inclusion: InclusionReason::Selected,
            token_count: Some(3),
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
        assert_eq!(json(&[evidence])["schema_version"], 1);
    }
}
