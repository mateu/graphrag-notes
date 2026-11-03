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
async fn deadline_bounded_upsert_writes_nothing_while_the_lifecycle_lock_is_busy() {
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
        .upsert_edge_proposal_before(draft(), soon)
        .await
        .unwrap()
        .is_none());
    assert!(repo.list_edge_proposals(None, 10).await.unwrap().is_empty());
    drop(held);
    let later = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    let created = repo
        .upsert_edge_proposal_before(draft(), later)
        .await
        .unwrap()
        .expect("an available lock admits the write");
    assert_eq!(created.status, ProposedEdgeStatus::Pending);
    assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
}
