//! Disposable fictional corpus only. No HTTP/provider/corpus dependencies.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, FixtureEntityExtractor,
    InferenceCapabilities, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, Entity, EntityType, Note};
use graphrag_db::{init_memory, parse_record_id, Repository, PORTABLE_TABLES};
use std::sync::Arc;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("proposal must not embed")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("proposal must not scan/embed")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("proposal must not probe providers")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("proposal must not read provider identities")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("proposal must not infer")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("proposal must not probe extraction")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("proposal must not inspect providers")
    }
}
fn app(repo: &Repository) -> EmbeddedApplication {
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), Arc::new(NoInference)),
        Arc::new(NoInference),
        Arc::new(NoInference),
        LibrarianRuntimeConfig::default(),
    )
}
fn caller() -> CallerIdentity {
    CallerIdentity {
        instance_id: "fixture-reviewer".into(),
    }
}
async fn note(repo: &Repository, id: &str, body: &str) -> String {
    let mut note = Note::new(body);
    note.id = Some(parse_record_id(id, Some("note")).unwrap());
    record_id_to_string(repo.create_note(note).await.unwrap().id.as_ref().unwrap())
}
async fn request(
    repo: &Repository,
    from: &str,
    to: &str,
    id: &str,
) -> RemoteEndpointProposalRequest {
    let a = repo.inspect_record(from, 0).await.unwrap();
    let b = repo.inspect_record(to, 0).await.unwrap();
    RemoteEndpointProposalRequest {request_id:id.into(),from:EndpointProposalEvidence{id:a.id,revision:a.revision,quote:a.content},to:EndpointProposalEvidence{id:b.id,revision:b.revision,quote:b.content},relationship:EndpointRelationship::Supports,rationale:"The second record supplies the verified evidence needed by the first claim; review is explicit, not similarity-derived.".into(),confirmed:true}
}
async fn rows(repo: &Repository, table: &str) -> Vec<serde_json::Value> {
    repo.portable_records_page(table, 0, 1000).await.unwrap()
}
async fn inventory(
    repo: &Repository,
) -> std::collections::BTreeMap<String, Vec<serde_json::Value>> {
    let mut all = std::collections::BTreeMap::new();
    for table in PORTABLE_TABLES {
        all.insert(table.to_string(), rows(repo, table).await);
    }
    all
}
#[tokio::test]
async fn one_pending_proposal_only_atomic_replay_and_pair_conflicts() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "note:a", "Marker is not execution proof").await;
    let b = note(&repo, "note:b", "Verified execution supplies proof").await;
    note(&repo, "note:u", "Unrelated similar marker").await;
    let app = app(&repo);
    let before = inventory(&repo).await;
    let request = request(&repo, &a, &b, "review-1").await;
    let result = app
        .propose_endpoint_remote(caller(), request.clone())
        .await
        .unwrap();
    assert!(!result.replayed);
    assert_eq!(result.outcome.status, "pending");
    let after = inventory(&repo).await;
    for (table, values) in &before {
        if table != "proposed_edge" && table != "remote_mutation_receipt" {
            assert_eq!(
                after.get(table).unwrap(),
                values,
                "unexpected {table} write"
            );
        }
    }
    assert_eq!(rows(&repo, "proposed_edge").await.len(), 1);
    assert_eq!(rows(&repo, "remote_mutation_receipt").await.len(), 1);
    assert!(rows(&repo, "supports").await.is_empty());
    let replay = app
        .propose_endpoint_remote(caller(), request.clone())
        .await
        .unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.outcome, result.outcome);
    assert_eq!(inventory(&repo).await, after);
    let mut changed = request.clone();
    changed.rationale = "Different claim".into();
    assert!(app
        .propose_endpoint_remote(caller(), changed)
        .await
        .is_err());
    let mut pair = request.clone();
    pair.request_id = "another-review".into();
    assert!(app.propose_endpoint_remote(caller(), pair).await.is_err());
    assert_eq!(inventory(&repo).await, after);
    let card = app.proposal(&result.outcome.id).await.unwrap();
    assert!(card.accept_allowed);
    let evidence: serde_json::Value = serde_json::from_str(&card.reason).unwrap();
    assert_eq!(evidence["from"]["quote"], request.from.quote);
}
#[tokio::test]
async fn cardinality_quotes_revisions_confirmation_and_unknown_types_fail_without_effects() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "note:a", "First quote").await;
    let b = note(&repo, "note:b", "Second quote").await;
    let app = app(&repo);
    let original = request(&repo, &a, &b, "review").await;
    let before = inventory(&repo).await;
    for bad in 0..9 {
        let mut r = original.clone();
        match bad {
            0 => r.confirmed = false,
            1 => r.to.id = r.from.id.clone(),
            2 => r.from.id = "source:a".into(),
            3 => r.to.revision = "".into(),
            4 => r.to.revision = "a".repeat(64),
            5 => r.to.quote = "quote not present".into(),
            6 => r.to.quote = "".into(),
            7 => r.to.quote = "x".repeat(2049),
            _ => r.from.id = "note:*".into(),
        }
        assert!(app.propose_endpoint_remote(caller(), r).await.is_err());
        assert_eq!(inventory(&repo).await, before);
    }
    let mut payload = serde_json::to_value(original).unwrap();
    payload["relationship"] = "depends_on".into();
    assert!(serde_json::from_value::<RemoteEndpointProposalRequest>(payload.clone()).is_err());
    payload["relationship"] = "related_to".into();
    payload["accept"] = true.into();
    assert!(serde_json::from_value::<RemoteEndpointProposalRequest>(payload).is_err());
}
#[tokio::test]
async fn symmetric_reverse_pair_and_simultaneous_replay_do_not_duplicate_effects() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "note:a", "First quote").await;
    let b = note(&repo, "note:b", "Second quote").await;
    let app = app(&repo);
    let mut r = request(&repo, &b, &a, "review").await;
    r.relationship = EndpointRelationship::RelatedTo;
    let (left, right) = tokio::join!(
        app.propose_endpoint_remote(caller(), r.clone()),
        app.propose_endpoint_remote(caller(), r.clone())
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_ne!(left.replayed, right.replayed);
    assert_eq!(left.outcome, right.outcome);
    let card = app.proposal(&left.outcome.id).await.unwrap();
    let mut ordered = [a, b];
    ordered.sort();
    assert_eq!(card.from.id, ordered[0]);
    assert_eq!(card.to.id, ordered[1]);
    assert!(card.accept_allowed);
    assert_eq!(rows(&repo, "proposed_edge").await.len(), 1);
}
#[tokio::test]
async fn stale_evidence_is_blocked_and_edits_retire_only_reviewed_edges() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "note:a", "First quote").await;
    let b = note(&repo, "note:b", "Second quote").await;
    let app = app(&repo);
    let r = request(&repo, &a, &b, "review").await;
    let result = app
        .propose_endpoint_remote(caller(), r.clone())
        .await
        .unwrap();
    let card = app.proposal(&result.outcome.id).await.unwrap();
    app.decide_remote(
        caller(),
        RemoteDecisionRequest {
            request_id: "accept".into(),
            id: card.id.clone(),
            revision: card.revision,
            action: ProposalAction::Accept,
            reason: None,
            confirmed: true,
        },
    )
    .await
    .unwrap();
    assert_eq!(rows(&repo, "supports").await.len(), 1);
    let c = note(&repo, "note:c", "Manual independent edge").await;
    repo.create_edge(
        &parse_record_id(&a, Some("note")).unwrap(),
        &parse_record_id(&c, Some("note")).unwrap(),
        graphrag_core::EdgeType::RelatedTo,
        None,
    )
    .await
    .unwrap();
    let mut replacement = repo.get_note(&a).await.unwrap().unwrap();
    replacement.content = "Changed claim".into();
    repo.update_note(&a, replacement).await.unwrap();
    assert!(rows(&repo, "supports").await.is_empty());
    assert_eq!(rows(&repo, "related_to").await.len(), 1);
    let card = app.proposal(&result.outcome.id).await.unwrap();
    assert!(!card.accept_allowed);
    assert_eq!(card.status, graphrag_core::ProposedEdgeStatus::Superseded);
    assert!(
        app.propose_endpoint_remote(caller(), r)
            .await
            .unwrap()
            .replayed
    );
    let fresh_request = request(&repo, &a, &b, "fresh-review-after-edit").await;
    let fresh = app
        .propose_endpoint_remote(caller(), fresh_request)
        .await
        .unwrap();
    assert_ne!(fresh.outcome.id, result.outcome.id);
    assert!(
        app.proposal(&fresh.outcome.id)
            .await
            .unwrap()
            .accept_allowed
    );
    assert_eq!(rows(&repo, "proposed_edge").await.len(), 2);
}
#[tokio::test]
async fn propose_versus_edit_linearizes_without_stale_pending_evidence() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = note(&repo, "note:a", "First quote").await;
    let b = note(&repo, "note:b", "Second quote").await;
    let app = app(&repo);
    let r = request(&repo, &a, &b, "review").await;
    let mut replacement = repo.get_note(&a).await.unwrap().unwrap();
    replacement.content = "Changed claim".into();
    let (proposal, edited) = tokio::join!(
        app.propose_endpoint_remote(caller(), r),
        repo.update_note(&a, replacement)
    );
    edited.unwrap();
    if let Ok(p) = proposal {
        let card = app.proposal(&p.outcome.id).await.unwrap();
        assert!(!card.accept_allowed);
        assert_eq!(card.status, graphrag_core::ProposedEdgeStatus::Superseded);
    }
    assert!(rows(&repo, "supports").await.is_empty());
}
fn vector(a: f32, b: f32) -> Vec<f32> {
    let mut v = vec![0.0; 1024];
    v[0] = a;
    v[1] = b;
    v
}
#[tokio::test]
async fn accepted_evidence_edge_reaches_missed_fact_off_on_on_off_and_undo_removes_it() {
    let repo = Repository::new(init_memory().await.unwrap());
    let query = "Nacre scheduled marker protection";
    repo.record_embedding_metadata(
        &graphrag_db::compatibility::EmbeddingIdentity::new("deterministic-test", "fixture", 1024),
        None,
    )
    .await
    .unwrap();
    let mut a=Note::new("Nacre scheduled marker protection is not execution proof; a separate verified execution record supplies the missing protection evidence.");
    a.id = Some(parse_record_id("note:a", Some("note")).unwrap());
    a.embedding = vector(1.0, 0.0);
    let a = repo.create_note(a).await.unwrap();
    let mut b=Note::new("Verified execution removed the blocking runtime residue and recovered 37 fictional storage units. Held-out verification code: amber-heron-37.");
    b.id = Some(parse_record_id("note:b", Some("note")).unwrap());
    b.embedding = vector(0.0, 1.0);
    let b = repo.create_note(b).await.unwrap();
    let mut d=Note::new("An unrelated calendar records fictional reminder settings, not the preregistered execution evidence.");
    d.id = Some(parse_record_id("note:d", Some("note")).unwrap());
    // Vector-only distractor outranks B off; graph evidence outranks this single channel on.
    d.embedding = vector(0.95, 0.2);
    repo.create_note(d).await.unwrap();
    let mut entity = Entity::new("Nacre", EntityType::Project);
    entity.metadata = serde_json::json!({});
    let entity = repo.upsert_entity(entity).await.unwrap();
    repo.link_note_to_entity(a.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
        .await
        .unwrap();
    let embedding =
        Arc::new(DeterministicEmbedder::default().with_embedding(query, true, vector(1.0, 0.0)));
    let app = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedding.clone()).with_graph_config(
            graphrag_agents::GraphRetrievalConfig {
                per_hop_decay: 1.0,
                ..Default::default()
            },
        ),
        embedding,
        Arc::new(FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig::default(),
    );
    let a = record_id_to_string(a.id.as_ref().unwrap());
    let b = record_id_to_string(b.id.as_ref().unwrap());
    let r = request(&repo, &a, &b, "review").await;
    let proposal = app.propose_endpoint_remote(caller(), r).await.unwrap();
    let context = |graph| BuildContextRequest {
        query: query.into(),
        scope: Scope::Notes,
        graph,
        since_days: None,
        source_uri: None,
        entity_filter: None,
        max_chunks: Some(2),
        max_total_tokens: Some(1000),
        max_chunk_tokens: Some(400),
    };
    let pending = app
        .build_context(context(GraphPolicy::On), ActionCancellation::new())
        .await
        .unwrap();
    assert!(!pending.chunks.iter().any(|c| c.id == b));
    let card = app.proposal(&proposal.outcome.id).await.unwrap();
    let accepted = app
        .decide_remote(
            caller(),
            RemoteDecisionRequest {
                request_id: "accept".into(),
                id: card.id,
                revision: card.revision,
                action: ProposalAction::Accept,
                reason: Some(
                    "Direct quote-backed narrower support claim, not a causal relation".into(),
                ),
                confirmed: true,
            },
        )
        .await
        .unwrap();
    let edge = accepted.outcome.resulting_edge_id.unwrap();
    let mut rounds = Vec::new();
    for mode in [
        GraphPolicy::Off,
        GraphPolicy::On,
        GraphPolicy::On,
        GraphPolicy::Off,
    ] {
        rounds.push(
            app.build_context(context(mode), ActionCancellation::new())
                .await
                .unwrap(),
        );
    }
    eprintln!(
        "USEFULNESS_DISPOSABLE_RECEIPT {}",
        serde_json::to_string(&rounds).unwrap()
    );
    for (i, round) in rounds.iter().enumerate() {
        assert!(round.chunks.len() <= 2);
        assert!(round.total_tokens <= 1000);
        assert!(round.chunks.iter().all(|c| c.rendered_tokens <= 400));
        let target = round.chunks.iter().find(|c| c.id == b);
        if i == 1 || i == 2 {
            let target = target
                .expect("accepted edge must reach preregistered missed fact under fixed budgets");
            assert!(target.snippet.contains("amber-heron-37"));
            let graph = target.graph.as_ref().expect("hop evidence");
            assert_eq!(graph["path"][0]["edge_type"], "supports");
            assert_eq!(graph["path"][0]["edge_id"], edge);
            assert!(round.rendered_context.contains("amber-heron-37"));
        } else {
            assert!(
                target.is_none(),
                "graph-off must miss the preregistered supporting record"
            );
        }
    }
    assert_eq!(
        rounds[0].chunks.iter().map(|c| &c.id).collect::<Vec<_>>(),
        rounds[3].chunks.iter().map(|c| &c.id).collect::<Vec<_>>()
    );
    assert_eq!(
        rounds[1].chunks.iter().map(|c| &c.id).collect::<Vec<_>>(),
        rounds[2].chunks.iter().map(|c| &c.id).collect::<Vec<_>>()
    );
    let card = app.proposal(&proposal.outcome.id).await.unwrap();
    app.decide_remote(
        caller(),
        RemoteDecisionRequest {
            request_id: "undo".into(),
            id: card.id,
            revision: card.revision,
            action: ProposalAction::Undo,
            reason: None,
            confirmed: true,
        },
    )
    .await
    .unwrap();
    let undone = app
        .build_context(context(GraphPolicy::On), ActionCancellation::new())
        .await
        .unwrap();
    assert!(!undone.chunks.iter().any(|c| c.id == b));
}
