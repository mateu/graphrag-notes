use super::*;
use crate::init_memory;
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
