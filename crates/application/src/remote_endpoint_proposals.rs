//! Validation and persistence hand-off for quote-grounded endpoint proposals.
use crate::*;
use graphrag_db::{RemoteEndpointProposalInput, RemoteMutationInput};
use sha2::{Digest, Sha256};

fn valid_revision(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn validate_endpoint_proposal(
    caller: &CallerIdentity,
    request: &RemoteEndpointProposalRequest,
) -> ApplicationResult<()> {
    for value in [&caller.instance_id, &request.request_id] {
        if value.is_empty()
            || value.len() > 256
            || value.chars().count() > 128
            || value.trim() != *value
            || value.chars().any(char::is_control)
        {
            return Err(ApplicationError::Validation("Request/instance identities must be nonempty, at most 128 characters/256 bytes, without controls or surrounding whitespace.".into()));
        }
    }
    if !request.confirmed
        || request.rationale.len() > 2048
        || request.rationale.trim().is_empty()
        || request.rationale.contains('\0')
    {
        return Err(ApplicationError::Validation("Review both exact note revisions and quotes, provide a bounded rationale, then set confirmed=true.".into()));
    }
    if request.from.id == request.to.id {
        return Err(ApplicationError::Validation(
            "A proposal requires exactly two distinct canonical note endpoints.".into(),
        ));
    }
    for evidence in [&request.from, &request.to] {
        if !evidence.id.starts_with("note:")
            || evidence.id.len() > 512
            || evidence.id.trim() != evidence.id
            || evidence.id.chars().any(char::is_control)
        {
            return Err(ApplicationError::Validation("Each endpoint must be a canonical note: record ID without controls or surrounding whitespace.".into()));
        }
        let parsed = graphrag_db::parse_record_id(&evidence.id, Some("note"))?;
        if graphrag_core::record_id_to_string(&parsed) != evidence.id {
            return Err(ApplicationError::Validation(
                "Canonical note identity required".into(),
            ));
        }
        if !valid_revision(&evidence.revision)
            || evidence.quote.trim().is_empty()
            || evidence.quote.trim() != evidence.quote
            || evidence.quote.contains('\0')
            || evidence.quote.len() > MAX_ENDPOINT_QUOTE_BYTES
            || evidence.quote.chars().count() > MAX_ENDPOINT_QUOTE_CHARS
        {
            return Err(ApplicationError::Validation("Each endpoint needs a current 64-character revision and a trimmed nonempty quote of at most 1024 characters/2048 UTF-8 bytes.".into()));
        }
    }
    if let Some(assessment) = &request.assessment {
        graphrag_core::relation_evidence::validate_request(
            &request.relationship.into(),
            assessment,
        )
        .map_err(|error| ApplicationError::Validation(error.to_string()))?;
    }
    Ok(())
}

pub fn remote_endpoint_proposal_fingerprint(
    request: &RemoteEndpointProposalRequest,
) -> ApplicationResult<String> {
    let bytes = serde_json::to_vec(&(REMOTE_ENDPOINT_PROPOSAL_PAYLOAD_VERSION, request))
        .map_err(|error| ApplicationError::Internal(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

impl EmbeddedApplication {
    pub(crate) async fn remote_endpoint_proposal_impl(
        &self,
        caller: CallerIdentity,
        request: RemoteEndpointProposalRequest,
    ) -> ApplicationResult<RemoteEndpointProposalResponse> {
        validate_endpoint_proposal(&caller, &request)?;
        let payload = serde_json::to_value(&request)
            .map_err(|error| ApplicationError::Internal(error.to_string()))?;
        let input = RemoteMutationInput {
            instance_id: caller.instance_id.clone(),
            request_id: request.request_id.clone(),
            operation: "propose_endpoint".into(),
            payload_fingerprint: remote_endpoint_proposal_fingerprint(&request)?,
            payload,
            result: serde_json::json!({}),
        };
        let receipt = self
            .repo
            .propose_remote_endpoint(RemoteEndpointProposalInput {
                mutation: input,
                from_id: request.from.id,
                from_revision: request.from.revision,
                from_quote: request.from.quote,
                to_id: request.to.id,
                to_revision: request.to.revision,
                to_quote: request.to.quote,
                edge_type: request.relationship.into(),
                rationale: request.rationale,
                assessment: request.assessment,
            })
            .await?;
        let outcome = serde_json::from_value(receipt.result).map_err(|error| {
            ApplicationError::Internal(format!("Unsupported endpoint proposal receipt: {error}"))
        })?;
        Ok(RemoteEndpointProposalResponse {
            request_id: request.request_id,
            replayed: receipt.replayed,
            outcome,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RemoteEndpointProposalRequest {
        RemoteEndpointProposalRequest {
            request_id: "endpoint-1".into(),
            from: EndpointProposalEvidence {
                id: "note:a".into(),
                revision: "a".repeat(64),
                quote: "first exact quote".into(),
            },
            to: EndpointProposalEvidence {
                id: "note:b".into(),
                revision: "b".repeat(64),
                quote: "second exact quote".into(),
            },
            relationship: EndpointRelationship::RelatedTo,
            rationale: "The quoted passages identify the same project constraint.".into(),
            assessment: None,
            confirmed: true,
        }
    }

    #[test]
    fn endpoint_quotes_are_bounded() {
        let caller = CallerIdentity {
            instance_id: "reviewer".into(),
        };
        let input = request();
        validate_endpoint_proposal(&caller, &input).unwrap();
        let mut invalid = input;
        invalid.to.quote = "x".repeat(MAX_ENDPOINT_QUOTE_BYTES + 1);
        assert!(validate_endpoint_proposal(&caller, &invalid).is_err());
    }

    #[test]
    fn requires_two_distinct_revisions_and_explicit_confirmation() {
        let caller = CallerIdentity {
            instance_id: "reviewer".into(),
        };
        let mut input = request();
        input.confirmed = false;
        assert!(validate_endpoint_proposal(&caller, &input).is_err());
        input.confirmed = true;
        input.to.id = input.from.id.clone();
        assert!(validate_endpoint_proposal(&caller, &input).is_err());
    }

    fn assessment(value: serde_json::Value) -> Option<EvidenceAssessmentRequest> {
        Some(serde_json::from_value(value).unwrap())
    }

    #[test]
    fn legacy_requests_keep_their_payload_and_fingerprint() {
        // Pinned from the pre-#205 layout: omitting the optional assessment
        // must not change any stored payload or replay identity.
        assert_eq!(
            remote_endpoint_proposal_fingerprint(&request()).unwrap(),
            "1afff2f501cf2317e7ab3d754af1d31648ef81a7be51b0ff0bee954f5cbb5aa5"
        );
        assert!(serde_json::to_value(request())
            .unwrap()
            .get("assessment")
            .is_none());
        let mut typed = request();
        typed.assessment =
            assessment(serde_json::json!({"evidence_mode": "analogy", "direction": "symmetric"}));
        assert_ne!(
            remote_endpoint_proposal_fingerprint(&typed).unwrap(),
            remote_endpoint_proposal_fingerprint(&request()).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&typed).unwrap()["assessment"]["lineage"],
            "unknown"
        );
    }

    #[test]
    fn typed_assessments_are_checked_before_storage() {
        let caller = CallerIdentity {
            instance_id: "reviewer".into(),
        };
        let mut input = request();
        for (value, expected) in [
            (
                serde_json::json!({"evidence_mode": "explicit_statement", "direction": "symmetric"}),
                Some("related_to requires explicit link evidence"),
            ),
            (
                serde_json::json!({"evidence_mode": "derivation", "direction": "symmetric"}),
                Some("related_to cannot be accepted on derivation evidence"),
            ),
            (
                serde_json::json!({"evidence_mode": "analogy", "direction": "from_to"}),
                Some("direction from_to does not match related_to"),
            ),
            (
                serde_json::json!({"evidence_mode": "hypothesis", "direction": "symmetric", "candidate": {"generator": "gardener-similarity", "similarity": 1.5}}),
                Some("candidate similarity"),
            ),
            (
                serde_json::json!({"evidence_mode": "hypothesis", "direction": "symmetric", "scope": " padded "}),
                Some("scope must be trimmed"),
            ),
            // Speculative claims are proposable; they simply stay pending.
            (
                serde_json::json!({"evidence_mode": "hypothesis", "direction": "symmetric"}),
                None,
            ),
        ] {
            input.assessment = assessment(value);
            match (validate_endpoint_proposal(&caller, &input), expected) {
                (Ok(()), None) => {}
                (Err(ApplicationError::Validation(message)), Some(expected)) => {
                    assert!(message.contains(expected), "{message}")
                }
                (result, expected) => panic!("{result:?} for {expected:?}"),
            }
        }
        // A confidence-style field is not part of the contract.
        let mut payload = serde_json::to_value(request()).unwrap();
        payload["assessment"] = serde_json::json!({"evidence_mode": "derivation", "direction": "symmetric", "confidence": 0.9});
        assert!(serde_json::from_value::<RemoteEndpointProposalRequest>(payload).is_err());
    }

    #[test]
    fn rationale_enforces_utf8_byte_boundary() {
        let caller = CallerIdentity {
            instance_id: "reviewer".into(),
        };
        let mut input = request();
        input.rationale = "\u{10400}".repeat(512);
        validate_endpoint_proposal(&caller, &input).unwrap();
        input.rationale.push('x');
        assert!(matches!(
            validate_endpoint_proposal(&caller, &input),
            Err(ApplicationError::Validation(_))
        ));
    }
}
