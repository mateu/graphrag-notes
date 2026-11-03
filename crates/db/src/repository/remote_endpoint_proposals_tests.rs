use super::*;
use crate::init_memory;
use graphrag_core::relation_evidence::{
    CandidateScore, DeclaredLineage, EvidenceAssessment, EvidenceAssessmentRequest, EvidenceMode,
    LinkEvidenceRequest, ProposerDeclaration, RelationDirection,
};
use graphrag_core::MessageRole;

async fn propose(repo: &Repository, a: &Note, b: &Note, request_id: &str) -> ProposedEdge {
    let from = record_id_to_string(a.id.as_ref().unwrap());
    let to = record_id_to_string(b.id.as_ref().unwrap());
    let ra = repo.inspect_record(&from, 0).await.unwrap().revision;
    let rb = repo.inspect_record(&to, 0).await.unwrap().revision;
    let payload = serde_json::json!({"request_id":request_id,"from":{"id":from,"revision":ra,"quote":a.content},"to":{"id":to,"revision":rb,"quote":b.content},"relationship":"supports","rationale":"Fictional directly reviewed evidence","confirmed":true});
    let receipt = repo
        .propose_remote_endpoint(RemoteEndpointProposalInput {
            mutation: RemoteMutationInput {
                instance_id: "fixture".into(),
                request_id: request_id.into(),
                operation: "propose_endpoint".into(),
                payload_fingerprint: "a".repeat(64),
                payload,
                result: serde_json::json!({}),
            },
            from_id: from,
            from_revision: ra,
            from_quote: a.content.clone(),
            to_id: to,
            to_revision: rb,
            to_quote: b.content.clone(),
            edge_type: EdgeType::Supports,
            rationale: "Fictional directly reviewed evidence".into(),
            assessment: None,
        })
        .await
        .unwrap();
    let id = parse_record_id(
        receipt.result["id"].as_str().unwrap(),
        Some("proposed_edge"),
    )
    .unwrap();
    repo.get_edge_proposal(&id).await.unwrap().unwrap()
}
async fn fixture(repo: &Repository) -> (ProposedEdge, Vec<Note>) {
    let a = repo
        .create_note(Note::new("Fictional claim quote"))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Fictional evidence quote"))
        .await
        .unwrap();
    let from = record_id_to_string(a.id.as_ref().unwrap());
    let to = record_id_to_string(b.id.as_ref().unwrap());
    let ra = repo.inspect_record(&from, 0).await.unwrap().revision;
    let rb = repo.inspect_record(&to, 0).await.unwrap().revision;
    let payload = serde_json::json!({"request_id":"review","from":{"id":from,"revision":ra,"quote":a.content},"to":{"id":to,"revision":rb,"quote":b.content},"relationship":"supports","rationale":"Fictional directly reviewed evidence","confirmed":true});
    let receipt = repo
        .propose_remote_endpoint(RemoteEndpointProposalInput {
            mutation: RemoteMutationInput {
                instance_id: "fixture".into(),
                request_id: "review".into(),
                operation: "propose_endpoint".into(),
                payload_fingerprint: "a".repeat(64),
                payload,
                result: serde_json::json!({}),
            },
            from_id: from,
            from_revision: ra,
            from_quote: a.content.clone(),
            to_id: to,
            to_revision: rb,
            to_quote: b.content.clone(),
            edge_type: EdgeType::Supports,
            rationale: "Fictional directly reviewed evidence".into(),
            assessment: None,
        })
        .await
        .unwrap();
    let id = parse_record_id(
        receipt.result["id"].as_str().unwrap(),
        Some("proposed_edge"),
    )
    .unwrap();
    (
        repo.get_edge_proposal(&id).await.unwrap().unwrap(),
        vec![a, b],
    )
}
async fn count(repo: &Repository, table: &str) -> usize {
    repo.portable_records_page(table, 0, 100)
        .await
        .unwrap()
        .len()
}

#[tokio::test]
async fn oversized_utf8_rationale_cannot_persist_proposal_or_receipt() {
    let repo = Repository::new(init_memory().await.unwrap());
    let notes = [
        repo.create_note(Note::new("Fictional claim quote"))
            .await
            .unwrap(),
        repo.create_note(Note::new("Fictional evidence quote"))
            .await
            .unwrap(),
    ];
    let from = record_id_to_string(notes[0].id.as_ref().unwrap());
    let to = record_id_to_string(notes[1].id.as_ref().unwrap());
    let from_revision = repo.inspect_record(&from, 0).await.unwrap().revision;
    let to_revision = repo.inspect_record(&to, 0).await.unwrap().revision;
    let rationale = format!("{}x", "\u{10400}".repeat(512));
    let payload = serde_json::json!({"request_id":"oversized","from":{"id":from,"revision":from_revision,"quote":notes[0].content},"to":{"id":to,"revision":to_revision,"quote":notes[1].content},"relationship":"supports","rationale":rationale,"confirmed":true});
    let result = repo
        .propose_remote_endpoint(RemoteEndpointProposalInput {
            mutation: RemoteMutationInput {
                instance_id: "fixture".into(),
                request_id: "oversized".into(),
                operation: "propose_endpoint".into(),
                payload_fingerprint: "b".repeat(64),
                payload,
                result: serde_json::json!({}),
            },
            from_id: from,
            from_revision,
            from_quote: notes[0].content.clone(),
            to_id: to,
            to_revision,
            to_quote: notes[1].content.clone(),
            edge_type: EdgeType::Supports,
            rationale,
            assessment: None,
        })
        .await;
    assert!(
        matches!(&result, Err(DbError::InvalidMutationRequest(_))),
        "{result:?}"
    );
    assert_eq!(count(&repo, "proposed_edge").await, 0);
    assert_eq!(count(&repo, "remote_mutation_receipt").await, 0);
    assert_eq!(count(&repo, "supports").await, 0);
}
#[tokio::test]
async fn generic_acceptance_transaction_fences_changed_endpoint_after_preflight() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (proposal, notes) = fixture(&repo).await;
    assert!(repo
        .reviewed_endpoint_proposal_current(&proposal)
        .await
        .unwrap());
    // Deterministic race schedule: the captured valid snapshot precedes a raw database writer.
    repo.db
        .query("UPDATE $note SET content='Changed after acceptance preflight'")
        .bind(("note", notes[1].id.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert!(repo
        .commit_reviewed_endpoint_acceptance(
            proposal.clone(),
            notes,
            vec![],
            Some("reviewer".into()),
            None
        )
        .await
        .is_err());
    assert_eq!(count(&repo, "supports").await, 0);
    assert_eq!(
        repo.get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ProposedEdgeStatus::Pending
    );
    assert!(repo
        .accept_edge_proposal(
            proposal.id.as_ref().unwrap(),
            Some("reviewer".into()),
            None,
            true
        )
        .await
        .is_err());
}
#[tokio::test]
async fn generic_reviewed_acceptance_is_atomic_manual_only_and_replayed() {
    let repo = Repository::new(init_memory().await.unwrap());
    let (proposal, _) = fixture(&repo).await;
    let id = proposal.id.as_ref().unwrap();
    assert!(repo
        .accept_edge_proposal(id, Some("automatic".into()), None, false)
        .await
        .is_err());
    assert_eq!(count(&repo, "supports").await, 0);
    let accepted = repo
        .accept_edge_proposal(id, Some("reviewer".into()), None, true)
        .await
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    let replay = repo
        .accept_edge_proposal(id, Some("reviewer".into()), None, true)
        .await
        .unwrap();
    assert_eq!(accepted.resulting_edge_id, replay.resulting_edge_id);
    assert_eq!(count(&repo, "supports").await, 1);
}

async fn assert_revision_mutation_retires_reviewed_edge(
    repo: &Repository,
    proposal: ProposedEdge,
    notes: Vec<Note>,
    accepted: bool,
) {
    let manual_target = repo
        .create_note(Note::new("Unrelated manual edge target"))
        .await
        .unwrap();
    repo.create_edge(
        notes[0].id.as_ref().unwrap(),
        manual_target.id.as_ref().unwrap(),
        EdgeType::Supports,
        None,
    )
    .await
    .unwrap();
    if accepted {
        repo.accept_edge_proposal(
            proposal.id.as_ref().unwrap(),
            Some("reviewer".into()),
            None,
            true,
        )
        .await
        .unwrap();
    }

    repo.update_note_embedding(notes[0].id.as_ref().unwrap(), vec![0.25; 1024])
        .await
        .unwrap();

    assert_eq!(
        repo.get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ProposedEdgeStatus::Superseded
    );
    assert_eq!(count(repo, "supports").await, 1);
    let edges = repo
        .graph_note_edges(
            std::slice::from_ref(notes[0].id.as_ref().unwrap()),
            &["supports".into()],
            10,
            true,
            false,
            0.0,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        edges
            .iter()
            .map(|edge| &edge.out_id)
            .eq(std::iter::once(manual_target.id.as_ref().unwrap())),
        "{edges:?}"
    );
}

#[tokio::test]
async fn embedding_updates_supersede_pending_and_accepted_reviewed_edges_without_touching_manual_edges(
) {
    for accepted in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let (proposal, notes) = fixture(&repo).await;
        assert_revision_mutation_retires_reviewed_edge(&repo, proposal, notes, accepted).await;
    }
}

#[tokio::test]
async fn committed_reindex_supersedes_pending_and_accepted_reviewed_edges_but_staging_does_not() {
    for accepted in [false, true] {
        let repo = Repository::new(init_memory().await.unwrap());
        let (proposal, notes) = fixture(&repo).await;
        let manual_target = repo
            .create_note(Note::new("Unrelated manual edge target"))
            .await
            .unwrap();
        repo.create_edge(
            notes[0].id.as_ref().unwrap(),
            manual_target.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();
        if accepted {
            repo.accept_edge_proposal(
                proposal.id.as_ref().unwrap(),
                Some("reviewer".into()),
                None,
                true,
            )
            .await
            .unwrap();
        }
        let item = repo
            .get_reindex_item(&record_id_to_string(notes[0].id.as_ref().unwrap()))
            .await
            .unwrap()
            .unwrap();
        let embedding = EmbeddingIdentity {
            provider: "fixture".into(),
            model: "fixture".into(),
            dimension: 1024,
        };
        let job = repo
            .create_reindex_processing_job(
                1,
                "selected".into(),
                vec![item.id.clone()],
                &embedding,
                std::collections::BTreeMap::new(),
            )
            .await
            .unwrap();
        let owner = "fixture-worker";
        repo.claim_reindex_processing_job(
            job.id.as_ref().unwrap(),
            owner,
            Utc::now() + chrono::Duration::minutes(1),
        )
        .await
        .unwrap();
        assert!(repo
            .commit_reindex(
                job.id.as_ref().unwrap(),
                owner,
                std::slice::from_ref(&item.id),
                &embedding,
                false,
                false,
                1,
            )
            .await
            .is_err());
        assert_eq!(
            repo.get_edge_proposal(proposal.id.as_ref().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            if accepted {
                ProposedEdgeStatus::Accepted
            } else {
                ProposedEdgeStatus::Pending
            }
        );
        repo.stage_reindex_embedding(&item, vec![0.5; 1024], owner)
            .await
            .unwrap();
        assert_eq!(
            repo.get_edge_proposal(proposal.id.as_ref().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            if accepted {
                ProposedEdgeStatus::Accepted
            } else {
                ProposedEdgeStatus::Pending
            }
        );
        repo.commit_reindex(
            job.id.as_ref().unwrap(),
            owner,
            &[item.id],
            &embedding,
            false,
            false,
            1,
        )
        .await
        .unwrap();
        assert_eq!(
            repo.get_edge_proposal(proposal.id.as_ref().unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
        assert_eq!(count(&repo, "supports").await, 1);
    }
}

#[tokio::test]
async fn linked_chat_context_updates_retire_only_changed_reviewed_snapshots() {
    for accepted in [false, true] {
        for linked_message in [false, true] {
            for update_conversation in [false, true] {
                let repo = Repository::new(init_memory().await.unwrap());
                let now = Utc::now();
                let mut conversation = ChatConversation {
                    uuid: "context".into(),
                    name: "Context".into(),
                    summary: "before".into(),
                    created_at: now,
                    updated_at: now,
                    account: None,
                    messages: vec![],
                };
                let conversation_id = repo
                    .upsert_conversation(&conversation, None, serde_json::json!({}), None)
                    .await
                    .unwrap();
                let mut message = ChatMessage {
                    uuid: Some("message".into()),
                    role: MessageRole::Human,
                    content: "before".into(),
                    content_blocks: serde_json::json!([]),
                    created_at: None,
                    updated_at: None,
                    attachments: vec![],
                    files: vec![],
                };
                let message_id = repo
                    .upsert_message(&conversation_id, &conversation.uuid, 0, &message, None)
                    .await
                    .unwrap();
                let a = repo
                    .create_note(Note::new("Context endpoint"))
                    .await
                    .unwrap();
                let b = repo
                    .create_note(Note::new("Context evidence"))
                    .await
                    .unwrap();
                let manual = repo.create_note(Note::new("Manual target")).await.unwrap();
                let a_id = a.id.as_ref().unwrap();
                if linked_message {
                    repo.link_note_to_message(a_id, &message_id).await.unwrap();
                } else {
                    repo.link_note_to_conversation(a_id, &conversation_id)
                        .await
                        .unwrap();
                }
                repo.create_edge(a_id, manual.id.as_ref().unwrap(), EdgeType::Supports, None)
                    .await
                    .unwrap();
                let proposal = propose(&repo, &a, &b, "context-review").await;
                if accepted {
                    repo.accept_edge_proposal(
                        proposal.id.as_ref().unwrap(),
                        Some("reviewer".into()),
                        None,
                        true,
                    )
                    .await
                    .unwrap();
                }
                // Replaying the same link must preserve a fresh review.
                let before = repo
                    .get_edge_proposal(proposal.id.as_ref().unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                let replay = if linked_message {
                    repo.link_note_to_message(a_id, &message_id).await.unwrap()
                } else {
                    repo.link_note_to_conversation(a_id, &conversation_id)
                        .await
                        .unwrap()
                };
                assert!(!replay);
                let after_replay = repo
                    .get_edge_proposal(proposal.id.as_ref().unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(after_replay.status, before.status);
                assert_eq!(after_replay.resulting_edge_id, before.resulting_edge_id);
                let opening = repo
                    .inspect_record(&record_id_to_string(a_id), 0)
                    .await
                    .unwrap()
                    .revision;
                if update_conversation {
                    conversation.summary = "after".into();
                    repo.upsert_conversation(&conversation, None, serde_json::json!({}), None)
                        .await
                        .unwrap();
                } else {
                    message.content = "after".into();
                    repo.upsert_message(&conversation_id, &conversation.uuid, 0, &message, None)
                        .await
                        .unwrap();
                }
                let changed = update_conversation || linked_message;
                assert_eq!(
                    repo.inspect_record(&record_id_to_string(a_id), 0)
                        .await
                        .unwrap()
                        .revision
                        != opening,
                    changed
                );
                assert_eq!(
                    repo.get_edge_proposal(proposal.id.as_ref().unwrap())
                        .await
                        .unwrap()
                        .unwrap()
                        .status,
                    if changed {
                        ProposedEdgeStatus::Superseded
                    } else if accepted {
                        ProposedEdgeStatus::Accepted
                    } else {
                        ProposedEdgeStatus::Pending
                    }
                );
                let mut actual: Vec<_> = repo
                    .graph_note_edges(
                        std::slice::from_ref(a_id),
                        &["supports".into()],
                        10,
                        true,
                        false,
                        0.0,
                        None,
                        None,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|edge| record_id_to_string(&edge.out_id))
                    .collect();
                let mut expected = vec![record_id_to_string(manual.id.as_ref().unwrap())];
                if accepted && !changed {
                    expected.push(record_id_to_string(b.id.as_ref().unwrap()));
                }
                actual.sort();
                expected.sort();
                assert_eq!(actual, expected);
            }
        }
    }
}

#[tokio::test]
async fn adding_chat_provenance_retires_pending_and_accepted_reviewed_edges() {
    for accepted in [false, true] {
        for link_message in [false, true] {
            let repo = Repository::new(init_memory().await.unwrap());
            let now = Utc::now();
            let conversation = ChatConversation {
                uuid: "new-context".into(),
                name: "Context".into(),
                summary: "Context".into(),
                created_at: now,
                updated_at: now,
                account: None,
                messages: vec![],
            };
            let conversation_id = repo
                .upsert_conversation(&conversation, None, serde_json::json!({}), None)
                .await
                .unwrap();
            let message: ChatMessage = serde_json::from_value(serde_json::json!({
                "uuid": "new-message", "role": "human", "text": "Context"
            }))
            .unwrap();
            let message_id = repo
                .upsert_message(&conversation_id, &conversation.uuid, 0, &message, None)
                .await
                .unwrap();
            let (proposal, notes) = fixture(&repo).await;
            if accepted {
                repo.accept_edge_proposal(
                    proposal.id.as_ref().unwrap(),
                    Some("reviewer".into()),
                    None,
                    true,
                )
                .await
                .unwrap();
            }
            let a_id = notes[0].id.as_ref().unwrap();
            let before = repo
                .inspect_record(&record_id_to_string(a_id), 0)
                .await
                .unwrap()
                .revision;
            let linked = if link_message {
                repo.link_note_to_message(a_id, &message_id).await.unwrap()
            } else {
                repo.link_note_to_conversation(a_id, &conversation_id)
                    .await
                    .unwrap()
            };
            assert!(linked);
            assert_ne!(
                repo.inspect_record(&record_id_to_string(a_id), 0)
                    .await
                    .unwrap()
                    .revision,
                before
            );
            assert_eq!(
                repo.get_edge_proposal(proposal.id.as_ref().unwrap())
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                ProposedEdgeStatus::Superseded
            );
            assert_eq!(count(&repo, "supports").await, 0);
        }
    }
}

// Typed evidence assessments (#205). Fictional notes only.

fn typed(mode: EvidenceMode, direction: RelationDirection) -> EvidenceAssessmentRequest {
    EvidenceAssessmentRequest {
        evidence_mode: mode,
        direction,
        lineage: DeclaredLineage::Unknown,
        from_support: vec![],
        to_support: vec![],
        link: None,
        candidate: None,
        proposer: None,
        alternative_explanation: None,
        scope: None,
    }
}

async fn propose_typed(
    repo: &Repository,
    a: &Note,
    b: &Note,
    request_id: &str,
    edge_type: EdgeType,
    assessment: EvidenceAssessmentRequest,
) -> Result<ProposedEdge> {
    propose_assessed(repo, a, b, request_id, edge_type, Some(assessment)).await
}

async fn propose_assessed(
    repo: &Repository,
    a: &Note,
    b: &Note,
    request_id: &str,
    edge_type: EdgeType,
    assessment: Option<EvidenceAssessmentRequest>,
) -> Result<ProposedEdge> {
    let from = record_id_to_string(a.id.as_ref().unwrap());
    let to = record_id_to_string(b.id.as_ref().unwrap());
    let ra = repo.inspect_record(&from, 0).await.unwrap().revision;
    let rb = repo.inspect_record(&to, 0).await.unwrap().revision;
    let rationale = "Fictional typed review".to_string();
    let payload = serde_json::json!({"request_id":request_id,"from":{"id":from,"revision":ra,"quote":a.content},"to":{"id":to,"revision":rb,"quote":b.content},"relationship":edge_type.to_string(),"rationale":rationale,"assessment":assessment,"confirmed":true});
    let receipt = repo
        .propose_remote_endpoint(RemoteEndpointProposalInput {
            mutation: RemoteMutationInput {
                instance_id: "fixture".into(),
                request_id: request_id.into(),
                operation: "propose_endpoint".into(),
                payload_fingerprint: "c".repeat(64),
                payload,
                result: serde_json::json!({}),
            },
            from_id: from,
            from_revision: ra,
            from_quote: a.content.clone(),
            to_id: to,
            to_revision: rb,
            to_quote: b.content.clone(),
            edge_type,
            rationale,
            assessment,
        })
        .await?;
    let id = parse_record_id(
        receipt.result["id"].as_str().unwrap(),
        Some("proposed_edge"),
    )
    .unwrap();
    Ok(repo.get_edge_proposal(&id).await.unwrap().unwrap())
}

async fn edge_assessments(repo: &Repository, table: &str) -> Vec<serde_json::Value> {
    repo.portable_records_page(table, 0, 100)
        .await
        .unwrap()
        .into_iter()
        .map(|row| row["evidence_assessment"].clone())
        .collect()
}

#[tokio::test]
async fn typed_assessment_is_validated_persisted_and_copied_onto_the_accepted_edge() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new(
            "Canary results passed on the pinned build. Logs attached.",
        ))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Release requires a passing canary."))
        .await
        .unwrap();
    let mut assessment = typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo);
    assessment.from_support = vec!["Logs attached.".into()];
    assessment.candidate = Some(CandidateScore {
        generator: "gardener-similarity".into(),
        generator_version: Some("1".into()),
        similarity: Some(0.72),
    });
    assessment.scope = Some("Pinned build only".into());
    assessment.alternative_explanation = Some("The canary could be unrelated to release".into());
    let proposal = propose_typed(&repo, &a, &b, "typed", EdgeType::Supports, assessment)
        .await
        .unwrap();
    // The legacy weight keeps its reviewed convention; similarity stays separate.
    assert_eq!(proposal.confidence, 1.0);
    let stored = proposal.parsed_evidence_assessment().unwrap().unwrap();
    assert_eq!(stored.evidence_mode, EvidenceMode::ExplicitStatement);
    assert_eq!(stored.proposer.actor, "mcp:fixture");
    assert_eq!(stored.candidate.as_ref().unwrap().similarity, Some(0.72));
    assert_eq!(stored.endpoints[0].support.len(), 2);
    assert_eq!(stored.endpoints[0].support[1].byte_start, 43);
    assert!(!stored.lineage.same_origin);
    assert_eq!(
        proposal.evidence_summary().status,
        graphrag_core::EvidenceStatus::Assessed
    );
    assert!(repo
        .reviewed_endpoint_proposal_current(&proposal)
        .await
        .unwrap());
    let accepted = repo
        .accept_edge_proposal(
            proposal.id.as_ref().unwrap(),
            Some("reviewer".into()),
            None,
            true,
        )
        .await
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    assert_eq!(
        edge_assessments(&repo, "supports").await,
        vec![proposal.evidence_assessment.clone().unwrap()]
    );
    let edges = repo
        .graph_note_edges(
            std::slice::from_ref(a.id.as_ref().unwrap()),
            &["supports".into()],
            10,
            true,
            false,
            0.0,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].confidence, Some(1.0));
    assert_eq!(edges[0].evidence_assessment, proposal.evidence_assessment);
    // Undo removes the assessed edge and retains the immutable proposal record.
    assert!(repo
        .undo_edge(accepted.resulting_edge_id.as_ref().unwrap(), None)
        .await
        .unwrap());
    assert_eq!(count(&repo, "supports").await, 0);
    let undone = repo
        .get_edge_proposal(proposal.id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(undone.status, ProposedEdgeStatus::Superseded);
    assert_eq!(undone.evidence_assessment, proposal.evidence_assessment);
}

#[tokio::test]
async fn analogy_and_hypothesis_proposals_stay_pending_and_out_of_the_accepted_graph() {
    for mode in [EvidenceMode::Analogy, EvidenceMode::Hypothesis] {
        let repo = Repository::new(init_memory().await.unwrap());
        let a = repo
            .create_note(Note::new("Queue latency rose in March."))
            .await
            .unwrap();
        let b = repo
            .create_note(Note::new("Cache churn rose in March."))
            .await
            .unwrap();
        let proposal = propose_typed(
            &repo,
            &a,
            &b,
            "speculative",
            EdgeType::RelatedTo,
            typed(mode, RelationDirection::Symmetric),
        )
        .await
        .unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Pending);
        let refused = repo
            .accept_edge_proposal(
                proposal.id.as_ref().unwrap(),
                Some("reviewer".into()),
                None,
                true,
            )
            .await;
        assert!(
            matches!(&refused, Err(DbError::InvalidMutationRequest(message)) if message == PENDING_ONLY_EVIDENCE),
            "{refused:?}"
        );
        // Even a raw acceptance transaction cannot place it in the graph.
        let notes = vec![a.clone(), b.clone()];
        assert!(repo
            .commit_reviewed_endpoint_acceptance(
                proposal.clone(),
                notes,
                vec![],
                Some("reviewer".into()),
                None
            )
            .await
            .is_err());
        assert_eq!(count(&repo, "related_to").await, 0);
        let rejected = repo
            .reject_edge_proposal(proposal.id.as_ref().unwrap(), Some("reviewer".into()), None)
            .await
            .unwrap();
        assert_eq!(rejected.status, ProposedEdgeStatus::Rejected);
    }
}

fn is_conflict(result: &Result<ProposedEdge>) -> bool {
    matches!(result, Err(DbError::MutationRevisionConflict(_)))
}

#[tokio::test]
async fn typed_evidence_is_part_of_proposal_identity_while_legacy_identity_is_unchanged() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new(
            "Canary results passed on the pinned build. Logs attached.",
        ))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Release requires a passing canary."))
        .await
        .unwrap();
    let hypothesis = typed(EvidenceMode::Hypothesis, RelationDirection::FromTo);
    let speculative = propose_typed(
        &repo,
        &a,
        &b,
        "speculative",
        EdgeType::Supports,
        hypothesis.clone(),
    )
    .await
    .unwrap();
    let refused = repo
        .accept_edge_proposal(
            speculative.id.as_ref().unwrap(),
            Some("reviewer".into()),
            None,
            true,
        )
        .await;
    assert!(
        matches!(&refused, Err(DbError::InvalidMutationRequest(message)) if message == PENDING_ONLY_EVIDENCE),
        "{refused:?}"
    );
    repo.reject_edge_proposal(
        speculative.id.as_ref().unwrap(),
        Some("reviewer".into()),
        None,
    )
    .await
    .unwrap();
    // The rejected hypothesis still deduplicates identical evidence...
    let repeated = propose_typed(
        &repo,
        &a,
        &b,
        "speculative-again",
        EdgeType::Supports,
        hypothesis,
    )
    .await;
    assert!(is_conflict(&repeated), "{repeated:?}");
    // ...but no longer blocks the fresh, evidenced proposal the refusal asks
    // for, over the same endpoints, revisions and primary quotes.
    let explicit = propose_typed(
        &repo,
        &a,
        &b,
        "explicit",
        EdgeType::Supports,
        typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo),
    )
    .await
    .unwrap();
    let mut more_support = typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo);
    more_support.from_support = vec!["Logs attached.".into()];
    let supported = propose_typed(&repo, &a, &b, "supported", EdgeType::Supports, more_support)
        .await
        .unwrap();
    assert_eq!(explicit.status, ProposedEdgeStatus::Pending);
    assert_ne!(explicit.id, speculative.id);
    assert_ne!(supported.id, explicit.id);
    // Who typed the pair is provenance, not evidence.
    let mut retyped = typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo);
    retyped.proposer = Some(ProposerDeclaration {
        generator: "fictional-typer".into(),
        generator_version: None,
        model: None,
    });
    let repeated = propose_typed(&repo, &a, &b, "retyped", EdgeType::Supports, retyped).await;
    assert!(is_conflict(&repeated), "{repeated:?}");
    // Without typed evidence the legacy identity is unchanged: the same
    // endpoints, revisions and quotes conflict under a new request ID.
    propose_assessed(&repo, &a, &b, "legacy", EdgeType::Supports, None)
        .await
        .unwrap();
    let repeated = propose_assessed(&repo, &a, &b, "legacy-again", EdgeType::Supports, None).await;
    assert!(is_conflict(&repeated), "{repeated:?}");
    // The accepted graph still holds one edge per pair and relation.
    repo.accept_edge_proposal(
        explicit.id.as_ref().unwrap(),
        Some("reviewer".into()),
        None,
        true,
    )
    .await
    .unwrap();
    assert!(repo
        .accept_edge_proposal(
            supported.id.as_ref().unwrap(),
            Some("reviewer".into()),
            None,
            true
        )
        .await
        .is_err());
    assert_eq!(count(&repo, "supports").await, 1);
}

#[tokio::test]
async fn symmetric_typed_evidence_deduplicates_in_either_endpoint_order() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new("Queue latency rose in March. Alerts fired."))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Cache churn rose in March. Evictions doubled."))
        .await
        .unwrap();
    let mut forward = typed(EvidenceMode::Hypothesis, RelationDirection::Symmetric);
    forward.from_support = vec!["Alerts fired.".into()];
    forward.to_support = vec!["Evictions doubled.".into()];
    let mut mirrored = forward.clone();
    std::mem::swap(&mut mirrored.from_support, &mut mirrored.to_support);
    propose_typed(&repo, &a, &b, "forward", EdgeType::RelatedTo, forward)
        .await
        .unwrap();
    let repeated = propose_typed(&repo, &b, &a, "mirrored", EdgeType::RelatedTo, mirrored).await;
    assert!(is_conflict(&repeated), "{repeated:?}");
    // Different support in the reversed order is still different evidence.
    let mut fewer_spans = typed(EvidenceMode::Hypothesis, RelationDirection::Symmetric);
    fewer_spans.from_support = vec!["Evictions doubled.".into()];
    let fresh = propose_typed(
        &repo,
        &b,
        &a,
        "fewer-spans",
        EdgeType::RelatedTo,
        fewer_spans,
    )
    .await;
    assert!(fresh.is_ok(), "{fresh:?}");
}

#[tokio::test]
async fn same_origin_pairs_cannot_be_supports_but_may_be_derived_from() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = repo.create_source(Source::manual()).await.unwrap();
    let source_id = source.id.unwrap();
    let detail = repo
        .create_note(
            Note::new("Detail: the rollback took 4 minutes.").with_source(source_id.clone()),
        )
        .await
        .unwrap();
    let summary = repo
        .create_note(Note::new("Summary: rollback was quick.").with_source(source_id.clone()))
        .await
        .unwrap();
    let refused = propose_typed(
        &repo,
        &detail,
        &summary,
        "same-origin-supports",
        EdgeType::Supports,
        typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo),
    )
    .await;
    assert!(
        matches!(&refused, Err(DbError::InvalidMutationRequest(message)) if message.contains("derived_from")),
        "{refused:?}"
    );
    assert_eq!(count(&repo, "proposed_edge").await, 0);
    assert_eq!(count(&repo, "remote_mutation_receipt").await, 0);
    let derived = propose_typed(
        &repo,
        &summary,
        &detail,
        "summary-derived",
        EdgeType::DerivedFrom,
        typed(EvidenceMode::Derivation, RelationDirection::FromTo),
    )
    .await
    .unwrap();
    let assessment = derived.parsed_evidence_assessment().unwrap().unwrap();
    assert!(assessment.lineage.same_origin);
    assert_eq!(
        assessment.endpoints[0].origin,
        record_id_to_string(&source_id)
    );
    repo.accept_edge_proposal(
        derived.id.as_ref().unwrap(),
        Some("reviewer".into()),
        None,
        true,
    )
    .await
    .unwrap();
    assert_eq!(count(&repo, "derived_from").await, 1);
}

#[tokio::test]
async fn related_to_needs_a_verbatim_link_that_names_the_other_note() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new("For rollback steps see Beacon runbook."))
        .await
        .unwrap();
    let mut target = Note::new("Rollback: drain, then restore the last image.");
    target.title = Some("Beacon runbook".into());
    let b = repo.create_note(target).await.unwrap();
    let missing = propose_typed(
        &repo,
        &a,
        &b,
        "unlinked",
        EdgeType::RelatedTo,
        typed(
            EvidenceMode::ExplicitStatement,
            RelationDirection::Symmetric,
        ),
    )
    .await;
    assert!(matches!(missing, Err(DbError::InvalidMutationRequest(_))));
    let mut topical = typed(
        EvidenceMode::ExplicitStatement,
        RelationDirection::Symmetric,
    );
    topical.link = Some(LinkEvidenceRequest {
        endpoint: graphrag_core::relation_evidence::EndpointRole::From,
        quote: "For rollback steps".into(),
        reference: "rollback".into(),
    });
    assert!(
        propose_typed(&repo, &a, &b, "topical", EdgeType::RelatedTo, topical)
            .await
            .is_err()
    );
    let mut linked = typed(
        EvidenceMode::ExplicitStatement,
        RelationDirection::Symmetric,
    );
    linked.link = Some(LinkEvidenceRequest {
        endpoint: graphrag_core::relation_evidence::EndpointRole::From,
        quote: "see Beacon runbook".into(),
        reference: "Beacon runbook".into(),
    });
    let proposal = propose_typed(&repo, &a, &b, "linked", EdgeType::RelatedTo, linked)
        .await
        .unwrap();
    let assessment = proposal.parsed_evidence_assessment().unwrap().unwrap();
    let link = assessment.link.unwrap();
    assert_eq!(link.note, record_id_to_string(a.id.as_ref().unwrap()));
    assert_eq!(link.target, record_id_to_string(b.id.as_ref().unwrap()));
    assert_eq!(
        link.matched,
        graphrag_core::relation_evidence::LinkReferenceKind::NoteTitle
    );
    assert_eq!(count(&repo, "proposed_edge").await, 1);
    assert_eq!(count(&repo, "remote_mutation_receipt").await, 1);
    repo.accept_edge_proposal(
        proposal.id.as_ref().unwrap(),
        Some("reviewer".into()),
        None,
        true,
    )
    .await
    .unwrap();
    assert_eq!(edge_assessments(&repo, "related_to").await.len(), 1);
}

#[tokio::test]
async fn tampered_assessments_fail_closed_and_edits_retire_assessed_edges() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new("Fictional claim quote"))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Fictional evidence quote"))
        .await
        .unwrap();
    let proposal = propose_typed(
        &repo,
        &a,
        &b,
        "typed",
        EdgeType::Supports,
        typed(EvidenceMode::Derivation, RelationDirection::FromTo),
    )
    .await
    .unwrap();
    let id = proposal.id.clone().unwrap();
    let original = proposal.evidence_assessment.clone().unwrap();
    // A shifted span offset passes the storage ASSERT but not the evidence check.
    let mut shifted = original.clone();
    shifted["endpoints"][0]["support"][0]["byte_start"] = 1.into();
    for (tampered, expected) in [
        (shifted, None),
        (
            serde_json::json!({"version": 1, "calibration": "none", "evidence_mode": "derivation"}),
            Some(INVALID_EVIDENCE),
        ),
    ] {
        repo.db
            .query("UPDATE $id SET evidence_assessment = $value")
            .bind(("id", id.clone()))
            .bind(("value", tampered))
            .await
            .unwrap()
            .check()
            .unwrap();
        let current = repo.get_edge_proposal(&id).await.unwrap().unwrap();
        assert!(!repo
            .reviewed_endpoint_proposal_current(&current)
            .await
            .unwrap());
        let refused = repo
            .accept_edge_proposal(&id, Some("reviewer".into()), None, true)
            .await;
        match expected {
            Some(message) => assert!(
                matches!(&refused, Err(DbError::InvalidMutationRequest(m)) if m == message),
                "{refused:?}"
            ),
            None => assert!(refused.is_err()),
        }
        assert_eq!(count(&repo, "supports").await, 0);
    }
    repo.db
        .query("UPDATE $id SET evidence_assessment = $value")
        .bind(("id", id.clone()))
        .bind(("value", original))
        .await
        .unwrap()
        .check()
        .unwrap();
    let restored = repo.get_edge_proposal(&id).await.unwrap().unwrap();
    assert!(
        EvidenceAssessment::from_stored(restored.evidence_assessment.as_ref().unwrap()).is_ok()
    );
    // Endpoint edits retire assessed edges exactly like legacy reviewed ones.
    assert_revision_mutation_retires_reviewed_edge(&repo, restored, vec![a, b], true).await;
}

async fn restore_proposal(
    repo: &Repository,
    id: &RecordId,
    update: &str,
    value: serde_json::Value,
) {
    repo.db
        .query(format!("UPDATE $id SET {update}"))
        .bind(("id", id.clone()))
        .bind(("value", value))
        .await
        .unwrap()
        .check()
        .unwrap();
}

/// A restored or tampered proposal must not reach the graph: `current` stays
/// false, acceptance is refused and the edge table stays empty.
async fn assert_refused(repo: &Repository, id: &RecordId, table: &str, case: &str) {
    let current = repo.get_edge_proposal(id).await.unwrap().unwrap();
    assert!(
        !repo
            .reviewed_endpoint_proposal_current(&current)
            .await
            .unwrap(),
        "{case}: reported current"
    );
    let refused = repo
        .accept_edge_proposal(id, Some("reviewer".into()), None, true)
        .await;
    assert!(refused.is_err(), "{case}: accepted {refused:?}");
    assert_eq!(count(repo, table).await, 0, "{case}");
}

#[tokio::test]
async fn restored_link_evidence_must_be_quoted_from_one_endpoint_and_name_the_other() {
    let repo = Repository::new(init_memory().await.unwrap());
    let holder = "For rollback steps see Beacon runbook.";
    let a = repo.create_note(Note::new(holder)).await.unwrap();
    let mut target = Note::new("Rollback: drain, then restore the last image.");
    target.title = Some("Beacon runbook".into());
    let b = repo.create_note(target).await.unwrap();
    let third = "Cedar log: the Beacon runbook was reviewed.";
    let c = repo.create_note(Note::new(third)).await.unwrap();
    let [a_id, b_id, c_id] =
        [&a, &b, &c].map(|note| record_id_to_string(note.id.as_ref().unwrap()));
    let mut linked = typed(
        EvidenceMode::ExplicitStatement,
        RelationDirection::Symmetric,
    );
    linked.link = Some(LinkEvidenceRequest {
        endpoint: graphrag_core::relation_evidence::EndpointRole::From,
        quote: "see Beacon runbook".into(),
        reference: "Beacon runbook".into(),
    });
    let proposal = propose_typed(&repo, &a, &b, "linked", EdgeType::RelatedTo, linked)
        .await
        .unwrap();
    let id = proposal.id.clone().unwrap();
    let original = proposal.evidence_assessment.clone().unwrap();
    // Offsets are self-consistent, so only a check against the endpoints helps.
    let link = |note: &str, target: &str, quote: &str, reference: &str| {
        let mut tampered = original.clone();
        tampered["link"] = serde_json::json!({
            "note": note, "target": target, "quote": quote, "reference": reference,
            "matched": "note_title", "byte_start": 0, "byte_end": quote.len()
        });
        tampered
    };
    for (case, tampered) in [
        // Held by a third note: the quote is in neither endpoint.
        (
            "third-note holder",
            link(&c_id, &b_id, third, "Beacon runbook"),
        ),
        (
            "third-note holder",
            link(&c_id, &a_id, third, "Beacon runbook"),
        ),
        // It must name the opposite endpoint, not its holder or a third note.
        ("self target", link(&a_id, &a_id, holder, "Beacon runbook")),
        (
            "third-note target",
            link(&a_id, &c_id, holder, "Beacon runbook"),
        ),
        // Held by an endpoint whose pinned content lacks the quote.
        (
            "missing quote",
            link(&a_id, &b_id, "see Beacon runbook twice", "Beacon runbook"),
        ),
        (
            "quote in the other note",
            link(&b_id, &a_id, "see Beacon runbook", "Beacon runbook"),
        ),
        // Verbatim in its holder, but the reference does not name the target.
        (
            "unnamed target",
            link(&a_id, &b_id, holder, "rollback steps"),
        ),
    ] {
        restore_proposal(&repo, &id, "evidence_assessment = $value", tampered).await;
        assert_refused(&repo, &id, "related_to", case).await;
    }
    restore_proposal(&repo, &id, "evidence_assessment = $value", original.clone()).await;
    let accepted = repo
        .accept_edge_proposal(&id, Some("reviewer".into()), None, true)
        .await
        .unwrap();
    assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
    assert_eq!(edge_assessments(&repo, "related_to").await, vec![original]);
}

#[tokio::test]
async fn acceptance_rederives_restored_assessments_from_the_pinned_endpoints() {
    let repo = Repository::new(init_memory().await.unwrap());
    let source = repo.create_source(Source::manual()).await.unwrap();
    let source_id = source.id.unwrap();
    let detail = repo
        .create_note(
            Note::new("Detail: the rollback took 4 minutes.").with_source(source_id.clone()),
        )
        .await
        .unwrap();
    let summary = repo
        .create_note(Note::new("Summary: rollback was quick.").with_source(source_id.clone()))
        .await
        .unwrap();
    // A same-origin derived_from restored as an independent `supports`. The
    // stored lineage is self-consistent and passes the edge-table ASSERT.
    let derived = propose_typed(
        &repo,
        &summary,
        &detail,
        "summary-derived",
        EdgeType::DerivedFrom,
        typed(EvidenceMode::Derivation, RelationDirection::FromTo),
    )
    .await
    .unwrap();
    let id = derived.id.clone().unwrap();
    let mut independent = derived.evidence_assessment.clone().unwrap();
    independent["relation"] = "supports".into();
    independent["lineage"] = serde_json::json!({
        "declared": "unknown", "same_origin": false, "basis": "distinct_origins"
    });
    independent["endpoints"][1]["origin"] = "source:elsewhere".into();
    restore_proposal(
        &repo,
        &id,
        "edge_type = 'supports', evidence_assessment = $value",
        independent,
    )
    .await;
    assert!(EvidenceAssessment::from_stored(
        repo.get_edge_proposal(&id)
            .await
            .unwrap()
            .unwrap()
            .evidence_assessment
            .as_ref()
            .unwrap()
    )
    .is_ok());
    assert_refused(&repo, &id, "supports", "same-origin supports").await;

    let a = repo
        .create_note(Note::new("Canary results passed on the pinned build."))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Release requires a passing canary."))
        .await
        .unwrap();
    let proposal = propose_typed(
        &repo,
        &a,
        &b,
        "directed",
        EdgeType::Supports,
        typed(EvidenceMode::ExplicitStatement, RelationDirection::FromTo),
    )
    .await
    .unwrap();
    let id = proposal.id.clone().unwrap();
    let original = proposal.evidence_assessment.clone().unwrap();
    // A plausible derived fact that the pinned content does not reproduce.
    let mut occurrences = original.clone();
    occurrences["endpoints"][0]["support"][0]["occurrences"] = 2.into();
    restore_proposal(&repo, &id, "evidence_assessment = $value", occurrences).await;
    assert_refused(&repo, &id, "supports", "occurrence count").await;
    restore_proposal(&repo, &id, "evidence_assessment = $value", original.clone()).await;
    // Reversing a directional edge reverses the assessed claim.
    for (from, to) in [(&b, &a), (&a, &b)] {
        repo.db
            .query("UPDATE $id SET in = $from, out = $to")
            .bind(("id", id.clone()))
            .bind(("from", from.id.clone().unwrap()))
            .bind(("to", to.id.clone().unwrap()))
            .await
            .unwrap()
            .check()
            .unwrap();
        if from.id == b.id {
            assert_refused(&repo, &id, "supports", "reversed direction").await;
        }
    }
    // Only reviewed-endpoint proposals pin the revisions an assessment needs.
    restore_proposal(
        &repo,
        &id,
        "generator = $value",
        "gardener-similarity".into(),
    )
    .await;
    let refused = repo
        .accept_edge_proposal(&id, Some("reviewer".into()), None, true)
        .await;
    assert!(
        matches!(&refused, Err(DbError::InvalidMutationRequest(m)) if m == INVALID_EVIDENCE),
        "{refused:?}"
    );
    assert_eq!(count(&repo, "supports").await, 0);
    restore_proposal(
        &repo,
        &id,
        "generator = $value",
        "remote-reviewed-endpoint".into(),
    )
    .await;
    repo.accept_edge_proposal(&id, Some("reviewer".into()), None, true)
        .await
        .unwrap();
    assert_eq!(edge_assessments(&repo, "supports").await, vec![original]);
}
