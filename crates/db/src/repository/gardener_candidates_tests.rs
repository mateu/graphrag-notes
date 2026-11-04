//! Fictional fixtures for Gardener candidate evidence reads.
use super::*;
use crate::init_memory;
use graphrag_core::EntityType;

async fn note(repo: &Repository, content: &str) -> RecordId {
    repo.create_note(Note::new(content))
        .await
        .unwrap()
        .id
        .unwrap()
}

/// Unit-length 1024-dimensional vector whose cosine with axis 0 is `cosine`;
/// the remainder lies on a private `axis`, so distinct axes are orthogonal.
fn embedding(cosine: f32, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = cosine;
    vector[axis] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

fn entity(name: &str, kind: EntityType) -> Entity {
    let mut entity = Entity::new(name, kind);
    entity.metadata = serde_json::json!({});
    entity
}

async fn source(repo: &Repository, path: &str) -> RecordId {
    repo.create_source(Source::from_file(path, SourceType::Markdown).unwrap())
        .await
        .unwrap()
        .id
        .unwrap()
}

async fn sourced(
    repo: &Repository,
    source: &RecordId,
    content: &str,
    vector: Vec<f32>,
) -> RecordId {
    repo.create_note(
        Note::new(content)
            .with_source(source.clone())
            .with_embedding(vector),
    )
    .await
    .unwrap()
    .id
    .unwrap()
}

#[tokio::test]
async fn orphans_exclude_notes_linked_by_every_accepted_note_edge_type_in_either_direction() {
    let repo = Repository::new(init_memory().await.unwrap());
    let isolated = note(&repo, "Fictional isolated observation").await;
    let mut linked = Vec::new();
    for edge_type in [
        EdgeType::Supports,
        EdgeType::Contradicts,
        EdgeType::DerivedFrom,
        EdgeType::RelatedTo,
    ] {
        let from = note(&repo, &format!("Fictional {edge_type} origin")).await;
        let to = note(&repo, &format!("Fictional {edge_type} target")).await;
        repo.create_edge(&from, &to, edge_type, None).await.unwrap();
        linked.push(from);
        linked.push(to);
    }

    let orphans = repo
        .find_orphan_notes()
        .await
        .unwrap()
        .into_iter()
        .map(|note| note.id.unwrap())
        .collect::<Vec<_>>();
    // Both endpoints of every accepted relation, including derived_from in
    // each direction, are connected; only the unlinked note remains.
    assert_eq!(orphans, vec![isolated.clone()]);
    for id in linked {
        assert!(
            !orphans.contains(&id),
            "{} counted as orphan",
            record_id_to_string(&id)
        );
    }

    // The candidate anchor page applies the identical orphan definition.
    let isolated_with_vector = repo
        .create_note(Note::new("Fictional embedded orphan").with_embedding(embedding(1.0, 1)))
        .await
        .unwrap()
        .id
        .unwrap();
    let anchors = repo
        .candidate_anchor_notes(CandidateAnchorScope::Orphans, None, 10)
        .await
        .unwrap();
    assert_eq!(
        anchors.into_iter().map(|note| note.id).collect::<Vec<_>>(),
        vec![isolated_with_vector]
    );
}

#[tokio::test]
async fn similar_notes_exclude_the_anchor_source_before_the_neighbour_limit() {
    let repo = Repository::new(init_memory().await.unwrap());
    let alpha = source(&repo, "/fixtures/alpha.md").await;
    let beta = source(&repo, "/fixtures/beta.md").await;
    let gamma = source(&repo, "/fixtures/gamma.md").await;
    let anchor_id = sourced(&repo, &alpha, "Alpha anchor chunk", embedding(1.0, 1)).await;
    // Adjacent chunks of the same file are the nearest neighbours.
    sourced(&repo, &alpha, "Alpha adjacent chunk", embedding(1.0, 2)).await;
    sourced(&repo, &alpha, "Alpha second chunk", embedding(0.99, 3)).await;
    let beta_near = sourced(&repo, &beta, "Beta related chunk", embedding(0.9, 4)).await;
    let gamma_near = sourced(&repo, &gamma, "Gamma related chunk", embedding(0.85, 5)).await;
    sourced(&repo, &gamma, "Gamma unrelated chunk", embedding(0.3, 6)).await;

    let anchor = repo
        .candidate_anchor_notes(
            CandidateAnchorScope::Notes(std::slice::from_ref(&anchor_id)),
            None,
            5,
        )
        .await
        .unwrap()
        .remove(0);
    assert_eq!(
        anchor.source_uri.as_deref(),
        Some("file:///fixtures/alpha.md")
    );
    let one = repo
        .candidate_similar_notes(&anchor, 0.8, 1, None)
        .await
        .unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].id, beta_near);
    let all = repo
        .candidate_similar_notes(&anchor, 0.8, 10, None)
        .await
        .unwrap();
    assert_eq!(
        all.iter().map(|note| note.id.clone()).collect::<Vec<_>>(),
        vec![beta_near.clone(), gamma_near.clone()]
    );
    assert!((all[0].similarity.unwrap() - 0.9).abs() < 1e-4);
    let gamma_only = repo
        .candidate_similar_notes(&anchor, 0.8, 10, Some(std::slice::from_ref(&gamma)))
        .await
        .unwrap();
    assert_eq!(
        gamma_only
            .into_iter()
            .map(|note| note.id)
            .collect::<Vec<_>>(),
        vec![gamma_near]
    );
}

#[tokio::test]
async fn similarity_ties_and_anchor_pages_are_ordered_by_record_id() {
    let repo = Repository::new(init_memory().await.unwrap());
    let alpha = source(&repo, "/fixtures/alpha.md").await;
    let beta = source(&repo, "/fixtures/beta.md").await;
    let anchor_id = sourced(&repo, &alpha, "Alpha anchor", embedding(1.0, 1)).await;
    let mut tied = Vec::new();
    for index in 0..4 {
        tied.push(
            sourced(
                &repo,
                &beta,
                &format!("Beta twin {index}"),
                embedding(0.9, 2),
            )
            .await,
        );
    }
    tied.sort_by_key(record_id_to_string);
    let anchor = repo
        .candidate_anchor_notes(CandidateAnchorScope::Notes(&[anchor_id]), None, 1)
        .await
        .unwrap()
        .remove(0);
    let first = repo
        .candidate_similar_notes(&anchor, 0.8, 2, None)
        .await
        .unwrap();
    assert_eq!(
        first.into_iter().map(|note| note.id).collect::<Vec<_>>(),
        tied[..2].to_vec()
    );

    let page = repo
        .candidate_anchor_notes(
            CandidateAnchorScope::Sources(std::slice::from_ref(&beta)),
            None,
            3,
        )
        .await
        .unwrap();
    assert_eq!(
        page.iter().map(|note| note.id.clone()).collect::<Vec<_>>(),
        tied[..3].to_vec()
    );
    let next = repo
        .candidate_anchor_notes(CandidateAnchorScope::Sources(&[beta]), Some(&tied[2]), 3)
        .await
        .unwrap();
    assert_eq!(
        next.into_iter().map(|note| note.id).collect::<Vec<_>>(),
        tied[3..].to_vec()
    );
}

#[tokio::test]
async fn name_frequencies_count_distinct_notes_and_flag_truncated_probes() {
    let repo = Repository::new(init_memory().await.unwrap());
    let mut notes = Vec::new();
    for index in 0..3 {
        let id = note(&repo, &format!("Fictional Aurora Ledger note {index}")).await;
        repo.upsert_entities_and_link_note(
            &id,
            vec![
                entity("Aurora Ledger", EntityType::Project),
                entity(&format!("Rare marker {index}"), EntityType::Concept),
            ],
        )
        .await
        .unwrap();
        notes.push(id);
    }
    let names = vec![
        "aurora ledger".to_owned(),
        "rare marker 1".to_owned(),
        "absent name".to_owned(),
    ];
    let wide = repo.candidate_name_frequencies(&names, 16).await.unwrap();
    assert_eq!(
        wide,
        vec![
            CandidateNameFrequency {
                notes: 3,
                truncated: false
            },
            CandidateNameFrequency {
                notes: 1,
                truncated: false
            },
            CandidateNameFrequency {
                notes: 0,
                truncated: false
            },
        ]
    );
    let narrow = repo
        .candidate_name_frequencies(&names[..1], 2)
        .await
        .unwrap();
    assert!(narrow[0].truncated && narrow[0].notes <= 2);

    let mentions = repo.candidate_note_mentions(&notes[..1]).await.unwrap();
    let mut names = mentions
        .iter()
        .map(|mention| (mention.name.clone().unwrap(), mention.kind.clone().unwrap()))
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        vec![
            ("aurora ledger".to_owned(), "project".to_owned()),
            ("rare marker 0".to_owned(), "concept".to_owned()),
        ]
    );
}

#[tokio::test]
async fn existing_links_report_accepted_relations_and_proposals_between_candidates() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "Fictional A").await;
    let b = note(&repo, "Fictional B").await;
    let c = note(&repo, "Fictional C").await;
    let outside = note(&repo, "Fictional outside").await;
    repo.create_edge(&b, &a, EdgeType::DerivedFrom, None)
        .await
        .unwrap();
    repo.create_edge(&a, &outside, EdgeType::Supports, None)
        .await
        .unwrap();
    let pending = repo
        .upsert_gardener_proposal(&b, &c, 0.9, "fixture".into(), None, None)
        .await
        .unwrap();
    let links = repo
        .candidate_existing_links(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    assert_eq!(links.accepted, vec![(b.clone(), a.clone())]);
    assert_eq!(links.proposals.len(), 1);
    assert_eq!(links.proposals[0].status, "pending");
    assert_eq!(links.proposals[0].generator, "gardener-similarity");
    assert_eq!(
        (&links.proposals[0].from_id, &links.proposals[0].to_id),
        (&pending.from_id, &pending.to_id)
    );
}

#[tokio::test]
async fn deadline_bounded_candidate_recording_writes_nothing_while_the_lifecycle_lock_is_busy() {
    let repo = Repository::new(init_memory().await.unwrap());
    let from = note(&repo, "Fictional heron survey").await;
    let to = note(&repo, "Fictional heron follow-up").await;
    let draft = || EdgeProposalDraft {
        from_id: from.clone(),
        to_id: to.clone(),
        edge_type: EdgeType::RelatedTo,
        confidence: 0.9,
        reason: "fixture".into(),
        generator: "gardener-candidate".into(),
        generator_version: Some("1".into()),
        model: None,
    };
    let held = Arc::clone(&repo.proposal_acceptance_lock)
        .lock_owned()
        .await;
    let soon = tokio::time::Instant::now() + std::time::Duration::from_millis(50);
    assert!(repo
        .record_candidate_proposal_before(draft(), soon)
        .await
        .unwrap()
        .is_none());
    assert!(repo.list_edge_proposals(None, 10).await.unwrap().is_empty());
    drop(held);
    let later = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    let outcome = repo
        .record_candidate_proposal_before(draft(), later)
        .await
        .unwrap()
        .expect("an available lock admits the write");
    let CandidateProposalOutcome::Created(created) = outcome else {
        panic!("nothing joins the pair yet");
    };
    assert_eq!(created.status, ProposedEdgeStatus::Pending);
    assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn candidate_recording_rechecks_the_pair_under_the_lifecycle_lock_and_never_refreshes() {
    let repo = Repository::new(init_memory().await.unwrap());
    let from = note(&repo, "Fictional plover survey").await;
    let to = note(&repo, "Fictional plover follow-up").await;
    let draft = |generator: &str, reason: &str| EdgeProposalDraft {
        from_id: from.clone(),
        to_id: to.clone(),
        edge_type: EdgeType::RelatedTo,
        confidence: 0.9,
        reason: reason.into(),
        generator: generator.into(),
        generator_version: Some("1".into()),
        model: None,
    };
    let later = || tokio::time::Instant::now() + std::time::Duration::from_secs(10);

    // The scan waits for the lifecycle lock while another lifecycle writer
    // records a reviewed proposal for the same pair inside its own hold.
    let held = Arc::clone(&repo.proposal_acceptance_lock)
        .lock_owned()
        .await;
    let scan = tokio::spawn({
        let repo = repo.clone();
        let draft = draft("gardener-candidate", "candidate fixture");
        let deadline = later();
        async move { repo.record_candidate_proposal_before(draft, deadline).await }
    });
    let mut reviewed = draft("remote-reviewed-endpoint", "reviewed fixture");
    let key = super::graph::prepare_edge_proposal(&mut reviewed).unwrap();
    let recorded = repo
        .insert_pending_edge_proposal(reviewed, &key)
        .await
        .unwrap()
        .expect("the first writer records the pair");
    drop(held);
    let outcome = scan
        .await
        .unwrap()
        .unwrap()
        .expect("the lock frees before the deadline");
    let CandidateProposalOutcome::Existing(links) = outcome else {
        panic!("the scan must skip the pair, not write it");
    };
    assert_eq!(links.proposals.len(), 1);
    assert_eq!(links.proposals[0].generator, "remote-reviewed-endpoint");
    let stored = repo.list_edge_proposals(None, 10).await.unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].generator, "remote-reviewed-endpoint");
    assert_eq!(stored[0].reason, "reviewed fixture");
    assert_eq!(stored[0].updated_at, recorded.updated_at);

    // The insert is create-only: the same key never refreshes a proposal.
    let mut again = draft("gardener-candidate", "refresh attempt");
    let key = super::graph::prepare_edge_proposal(&mut again).unwrap();
    assert!(repo
        .insert_pending_edge_proposal(again, &key)
        .await
        .unwrap()
        .is_none());
    let stored = repo.list_edge_proposals(None, 10).await.unwrap();
    assert_eq!(stored[0].reason, "reviewed fixture");

    // An accepted relation of any type excludes the pair as well.
    let left = note(&repo, "Fictional tern survey").await;
    let right = note(&repo, "Fictional tern follow-up").await;
    repo.create_edge(&left, &right, EdgeType::Supports, None)
        .await
        .unwrap();
    let outcome = repo
        .record_candidate_proposal_before(
            EdgeProposalDraft {
                from_id: left.clone(),
                to_id: right.clone(),
                ..draft("gardener-candidate", "candidate fixture")
            },
            later(),
        )
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        outcome,
        CandidateProposalOutcome::Existing(ref links)
            if links.accepted.len() == 1 && links.proposals.is_empty()
    ));
    assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn candidate_origins_resolve_messages_to_their_conversation() {
    let repo = Repository::new(init_memory().await.unwrap());
    repo.db
        .query(
            r#"
            CREATE conversation:x SET uuid='x', title='x', created_at=time::now(), updated_at=time::now();
            CREATE conversation:y SET uuid='y', title='y', created_at=time::now(), updated_at=time::now();
            CREATE message:m1 SET message_key='m1', conversation_id=conversation:x, conversation_uuid='x', message_index=0, role='user', content='one';
            CREATE message:m2 SET message_key='m2', conversation_id=conversation:x, conversation_uuid='x', message_index=1, role='user', content='two';
            CREATE message:m3 SET message_key='m3', conversation_id=conversation:y, conversation_uuid='y', message_index=0, role='user', content='three';
            FOR $key IN ['a', 'b', 'c', 'd', 'e', 'f'] {
                CREATE type::record('note', $key) SET content = string::concat('Fictional ', $key);
            };
            CREATE note_from_message:a SET in=note:a, out=message:m1;
            CREATE note_from_message:b SET in=note:b, out=message:m2;
            CREATE note_from_conversation:c SET in=note:c, out=conversation:x;
            CREATE note_from_message:d SET in=note:d, out=message:m3;
            CREATE note_from_message:f SET in=note:f, out=message:gone;
            "#,
        )
        .await
        .unwrap()
        .check()
        .unwrap();
    let ids = ["a", "b", "c", "d", "e", "f"].map(|key| RecordId::new("note", key));
    let origins = repo.candidate_note_origins(&ids).await.unwrap();
    let id = |table: &str, key: &str| record_id_to_string(&RecordId::new(table, key));
    let origin = |key: &str| {
        origins
            .get(&id("note", key))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>()
    };
    // Different messages of one conversation, and the conversation itself,
    // are one origin; another conversation's message is not.
    assert_eq!(origin("a"), vec![id("conversation", "x")]);
    assert_eq!(origin("b"), vec![id("conversation", "x")]);
    assert_eq!(origin("c"), vec![id("conversation", "x")]);
    assert_eq!(origin("d"), vec![id("conversation", "y")]);
    assert!(origin("e").is_empty());
    // A message whose record is gone stands for itself.
    assert_eq!(origin("f"), vec![id("message", "gone")]);
}

#[tokio::test]
async fn upserts_refresh_only_their_own_generators_pending_proposals() {
    let repo = Repository::new(init_memory().await.unwrap());
    let from = note(&repo, "Fictional crane survey").await;
    let to = note(&repo, "Fictional crane follow-up").await;
    let later = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    let outcome = repo
        .record_candidate_proposal_before(
            EdgeProposalDraft {
                from_id: from.clone(),
                to_id: to.clone(),
                edge_type: EdgeType::RelatedTo,
                confidence: 0.9,
                reason: "candidate evidence".into(),
                generator: "gardener-candidate".into(),
                generator_version: Some("1".into()),
                model: None,
            },
            later,
        )
        .await
        .unwrap()
        .unwrap();
    let CandidateProposalOutcome::Created(candidate) = outcome else {
        panic!("nothing joins the pair yet");
    };
    // A legacy scan nominating the same pair gets the candidate back
    // unchanged: no relabel, no lost evidence, no batch-acceptance eligibility.
    let legacy = repo
        .upsert_gardener_proposal(&from, &to, 0.99, "legacy similarity".into(), None, None)
        .await
        .unwrap();
    assert_eq!(legacy.id, candidate.id);
    assert_eq!(legacy.generator, "gardener-candidate");
    let stored = repo
        .get_edge_proposal(candidate.id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.generator, "gardener-candidate");
    assert_eq!(stored.reason, "candidate evidence");
    assert_eq!(stored.confidence, candidate.confidence);
    assert_eq!(stored.updated_at, candidate.updated_at);
    assert_eq!(
        repo.accept_gardener_proposals_above(0.0, None)
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        repo.get_edge_proposal(candidate.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ProposedEdgeStatus::Pending
    );

    // A generator still refreshes its own pending proposal.
    let left = note(&repo, "Fictional rail survey").await;
    let right = note(&repo, "Fictional rail follow-up").await;
    repo.upsert_gardener_proposal(&left, &right, 0.8, "first".into(), None, None)
        .await
        .unwrap();
    let refreshed = repo
        .upsert_gardener_proposal(&left, &right, 0.9, "second".into(), None, None)
        .await
        .unwrap();
    assert_eq!(refreshed.reason, "second");
    assert_eq!(refreshed.generator, "gardener-similarity");
}

#[tokio::test]
async fn candidate_recording_skips_endpoints_that_vanished() {
    let repo = Repository::new(init_memory().await.unwrap());
    let from = note(&repo, "Fictional egret survey").await;
    let to = note(&repo, "Fictional egret follow-up").await;
    // The note is deleted after nomination, before persistence takes the
    // lifecycle lock: nothing is recorded for the invisible endpoint.
    repo.delete_note(&record_id_to_string(&to)).await.unwrap();
    let later = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    let outcome = repo
        .record_candidate_proposal_before(
            EdgeProposalDraft {
                from_id: from.clone(),
                to_id: to.clone(),
                edge_type: EdgeType::RelatedTo,
                confidence: 0.9,
                reason: "candidate evidence".into(),
                generator: "gardener-candidate".into(),
                generator_version: Some("1".into()),
                model: None,
            },
            later,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(outcome, CandidateProposalOutcome::Vanished));
    assert!(repo.list_edge_proposals(None, 10).await.unwrap().is_empty());
}

#[tokio::test]
async fn batched_similarity_reads_equal_per_anchor_reads() {
    let repo = Repository::new(init_memory().await.unwrap());
    let alpha = source(&repo, "/fixtures/alpha.md").await;
    let beta = source(&repo, "/fixtures/beta.md").await;
    let gamma = source(&repo, "/fixtures/gamma.md").await;
    // An uploaded part that declares the document it was split from.
    let mut part =
        Source::from_file("/fixtures/uploads/delta.part-1", SourceType::Markdown).unwrap();
    part.metadata =
        serde_json::json!({"remote_upload": {"source": {"uri": "vault://ops/delta.md"}}});
    let delta = repo.create_source(part).await.unwrap().id.unwrap();
    let mut anchors = Vec::new();
    let mut all = Vec::new();
    for index in 0..12 {
        let owner = [&alpha, &beta, &gamma, &delta][index % 4];
        // Ties at 0.95 and distinct strengths, all above the floor but two.
        let cosine = [1.0, 0.95, 0.95, 0.9, 0.85, 0.5][index % 6];
        let id = sourced(
            &repo,
            owner,
            &format!("Fictional chunk {index}"),
            embedding(cosine, index + 1),
        )
        .await;
        if index % 4 == 0 || index == 5 {
            anchors.push(id.clone());
        }
        all.push(id);
    }
    // A legacy note without a source, a joined pair, and a shared conversation.
    let manual = repo
        .create_note(Note::new("Fictional manual note").with_embedding(embedding(0.92, 40)))
        .await
        .unwrap()
        .id
        .unwrap();
    repo.upsert_gardener_proposal(&all[0], &all[1], 0.9, "fixture".into(), None, None)
        .await
        .unwrap();
    repo.create_edge(&all[2], &all[4], EdgeType::Supports, None)
        .await
        .unwrap();
    repo.db
        .query(format!(
            "CREATE conversation:x SET uuid='x', title='x', created_at=time::now(), updated_at=time::now(); \
             CREATE note_from_conversation:a SET in={}, out=conversation:x; \
             CREATE note_from_conversation:b SET in={}, out=conversation:x;",
            record_id_to_string(&all[4]),
            record_id_to_string(&all[7]),
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    // A note of a superseded generation is invisible to both reads.
    repo.db
        .query(format!(
            "UPDATE {} SET source_generation = 99",
            record_id_to_string(&all[9])
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    let anchors = repo
        .candidate_anchor_notes(CandidateAnchorScope::Notes(&anchors), None, 20)
        .await
        .unwrap();
    assert!(anchors.len() >= 4);
    for counterparts in [None, Some(vec![beta.clone(), delta.clone()])] {
        for limit in [1, 2, 8] {
            let batched = repo
                .candidate_similar_notes_batch(&anchors, 0.8, limit, counterparts.as_deref())
                .await
                .unwrap();
            assert_eq!(batched.len(), anchors.len());
            for (anchor, batch) in anchors.iter().zip(&batched) {
                let single = repo
                    .candidate_similar_notes(anchor, 0.8, limit, counterparts.as_deref())
                    .await
                    .unwrap();
                let shape = |rows: &[CandidateNote]| {
                    rows.iter()
                        .map(|row| {
                            (
                                record_id_to_string(&row.id),
                                row.similarity.map(f32::to_bits),
                                row.source_uri.clone(),
                                row.origin_uri.clone(),
                                row.content_hash.clone(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(shape(batch), shape(&single), "limit {limit}");
            }
        }
    }
    let manual_anchor = repo
        .candidate_anchor_notes(CandidateAnchorScope::Notes(&[manual]), None, 1)
        .await
        .unwrap();
    let batched = repo
        .candidate_similar_notes_batch(&manual_anchor, 0.8, 8, None)
        .await
        .unwrap();
    let single = repo
        .candidate_similar_notes(&manual_anchor[0], 0.8, 8, None)
        .await
        .unwrap();
    assert_eq!(
        batched[0]
            .iter()
            .map(|row| row.id.clone())
            .collect::<Vec<_>>(),
        single.iter().map(|row| row.id.clone()).collect::<Vec<_>>()
    );
    // Origin documents: a declared upload origin wins over the source URI.
    let origins = repo.candidate_note_origin_uris(&all).await.unwrap();
    assert_eq!(
        origins.get(&all[3]).map(String::as_str),
        Some("vault://ops/delta.md")
    );
    assert_eq!(
        origins.get(&all[0]).map(String::as_str),
        Some("file:///fixtures/alpha.md")
    );
    let names = repo
        .candidate_name_notes(&["absent name".to_owned()], 4)
        .await
        .unwrap();
    assert_eq!(
        names,
        vec![CandidateNameNotes {
            notes: Vec::new(),
            truncated: false
        }]
    );
}

#[tokio::test]
async fn batched_similarity_keeps_native_record_ids_with_one_display_form_distinct() {
    let repo = Repository::new(init_memory().await.unwrap());
    let alpha = source(&repo, "/fixtures/alpha.md").await;
    let beta = source(&repo, "/fixtures/beta.md").await;
    let anchor_id = sourced(&repo, &alpha, "Fictional anchor", embedding(1.0, 1)).await;
    // `note:42` and `note:⟨42⟩` print alike but are different records; the
    // string key is the closer neighbour.
    repo.db
        .query(
            "CREATE note:42 SET content = 'Fictional numeric', source_id = $beta, embedding = $far; \
             CREATE type::record('note', '42') SET content = 'Fictional string', source_id = $beta, embedding = $near;",
        )
        .bind(("beta", beta.clone()))
        .bind(("far", embedding(0.85, 2)))
        .bind(("near", embedding(0.95, 3)))
        .await
        .unwrap()
        .check()
        .unwrap();
    let anchor = repo
        .candidate_anchor_notes(CandidateAnchorScope::Notes(&[anchor_id]), None, 1)
        .await
        .unwrap()
        .remove(0);
    for limit in [1, 2] {
        let single = repo
            .candidate_similar_notes(&anchor, 0.8, limit, None)
            .await
            .unwrap();
        let batched = repo
            .candidate_similar_notes_batch(std::slice::from_ref(&anchor), 0.8, limit, None)
            .await
            .unwrap()
            .remove(0);
        let shape = |rows: &[CandidateNote]| {
            rows.iter()
                .map(|row| {
                    (
                        row.id.clone(),
                        row.content.clone(),
                        row.similarity.map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(single.len(), limit);
        assert_eq!(shape(&batched), shape(&single), "limit {limit}");
    }
}
