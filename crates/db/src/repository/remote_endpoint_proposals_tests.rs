use super::*;
use crate::init_memory;
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
