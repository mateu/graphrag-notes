//! Fictional fixtures for the capability-gated relationship candidate scan.
use async_trait::async_trait;
use graphrag_agents::gardener::CandidatePolicy;
use graphrag_agents::{
    Embedder, EntityExtraction, EntityExtractor, InferenceCapabilities, LibrarianRuntimeConfig,
    SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{Entity, EntityType, Note, Source, SourceType};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const TOOL: &str = "scan_relationship_candidates";

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("candidate scans never embed")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("candidate scans never embed")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("candidate scans never probe providers")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("candidate scans never read provider identity")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("candidate scans never extract")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("candidate scans never probe providers")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        graphrag_agents::FixtureEntityExtractor::default().capabilities()
    }
}

fn token(name: &str) -> String {
    format!("{name}-synthetic-token-at-least-thirty-two-bytes")
}

struct Fixture {
    _temp: tempfile::TempDir,
    url: String,
    client: reqwest::Client,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}

impl Fixture {
    async fn new(repo: &Repository, policy: CandidatePolicy) -> Self {
        let embedder: Arc<dyn Embedder> = Arc::new(NoInference);
        let application = Arc::new(
            EmbeddedApplication::new(
                repo.clone(),
                SearchAgent::new(repo.clone(), embedder.clone()),
                embedder,
                Arc::new(NoInference),
                LibrarianRuntimeConfig {
                    skip_entity_extraction: true,
                    ..Default::default()
                },
            )
            .with_candidate_policy(policy),
        );
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("credentials.json");
        let credentials = CredentialFile {
            schema_version: 1,
            credentials: [
                ("reader", vec![Capability::Read]),
                ("proposer", vec![Capability::Propose]),
                ("acceptor", vec![Capability::Accept]),
                ("capturer", vec![Capability::Capture]),
                ("jobs", vec![Capability::Jobs]),
                (
                    "everything-but-garden",
                    vec![
                        Capability::Read,
                        Capability::Capture,
                        Capability::Propose,
                        Capability::Enrich,
                        Capability::Edit,
                        Capability::Delete,
                        Capability::Accept,
                        Capability::Reject,
                        Capability::Undo,
                        Capability::Upload,
                        Capability::Jobs,
                    ],
                ),
                ("gardener", vec![Capability::Garden]),
            ]
            .into_iter()
            .map(|(name, capabilities)| Credential {
                instance_id: name.into(),
                token_sha256: format!("{:x}", Sha256::digest(token(name).as_bytes())),
                capabilities,
            })
            .collect(),
        };
        std::fs::write(&file, serde_json::to_vec(&credentials).unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen = listener.local_addr().unwrap();
        let options = ServiceOptions {
            listen,
            credentials_file: file,
            ..Default::default()
        };
        let shutdown = CancellationToken::new();
        let trigger = shutdown.clone();
        let task =
            tokio::spawn(async move { serve(listener, application, options, trigger).await });
        Self {
            _temp: temp,
            url: format!("http://{listen}/mcp"),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            shutdown,
            task,
        }
    }

    async fn rpc(&self, name: &str, method: &str, params: Value) -> Value {
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(token(name))
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2025-11-25")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        response.json().await.unwrap()
    }

    async fn tool(&self, name: &str, tool: &str, args: Value) -> Value {
        self.rpc(name, "tools/call", json!({"name":tool,"arguments":args}))
            .await
    }

    async fn names(&self, name: &str) -> Vec<String> {
        let list = self.rpc(name, "tools/list", json!({})).await;
        list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap().to_owned())
            .collect()
    }

    async fn stop(self) {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

fn data(value: &Value) -> &Value {
    &value["result"]["structuredContent"]["data"]
}
fn error(value: &Value) -> &Value {
    &value["result"]["structuredContent"]["error"]["code"]
}

fn vector(cosine: f32, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = cosine;
    vector[axis] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

async fn note(
    repo: &Repository,
    file: &str,
    content: &str,
    embedding: Vec<f32>,
    names: &[&str],
) -> String {
    let source = repo
        .create_source(
            Source::from_file(format!("/fixtures/vault/{file}"), SourceType::Markdown).unwrap(),
        )
        .await
        .unwrap()
        .id
        .unwrap();
    let id = repo
        .create_note(
            Note::new(content)
                .with_embedding(embedding)
                .with_source(source),
        )
        .await
        .unwrap()
        .id
        .unwrap();
    let entities = names
        .iter()
        .map(|name| {
            let mut entity = Entity::new(*name, EntityType::Concept);
            entity.metadata = json!({});
            entity
        })
        .collect::<Vec<_>>();
    repo.upsert_entities_and_link_note(&id, entities)
        .await
        .unwrap();
    graphrag_core::record_id_to_string(&id)
}

/// One hub with three cross-source spokes sharing distinct rare markers, and
/// a same-source sibling of the hub that shares every marker.
async fn corpus(repo: &Repository) -> String {
    let hub = note(
        repo,
        "hub.md",
        "Fictional hub index for the Osprey survey",
        vector(1.0, 1),
        &[
            "Osprey marker one",
            "Osprey marker two",
            "Osprey marker three",
        ],
    )
    .await;
    for (index, (marker, cosine)) in [
        ("Osprey marker one", 0.95),
        ("Osprey marker two", 0.93),
        ("Osprey marker three", 0.91),
    ]
    .into_iter()
    .enumerate()
    {
        note(
            repo,
            &format!("spoke-{index}.md"),
            &format!("Fictional spoke {index} of the Osprey survey"),
            vector(cosine, index + 2),
            &[marker],
        )
        .await;
    }
    hub
}

async fn accepted_edges(repo: &Repository) -> usize {
    let mut total = 0;
    for table in ["supports", "contradicts", "derived_from", "related_to"] {
        total += repo
            .portable_records_page(table, 0, 100)
            .await
            .unwrap()
            .len();
    }
    total
}

async fn proposals(repo: &Repository) -> Vec<Value> {
    repo.portable_records_page("proposed_edge", 0, 100)
        .await
        .unwrap()
}

fn scan(hub: &str, dry_run: bool) -> Value {
    json!({
        "scope": {"kind": "notes", "ids": [hub]},
        "max_candidates": 5,
        "max_endpoints": 10,
        "deadline_ms": 20000,
        "dry_run": dry_run
    })
}

#[tokio::test]
async fn only_the_garden_capability_sees_or_runs_the_scan_and_it_grants_nothing_else() {
    let repo = Repository::new(init_memory().await.unwrap());
    let hub = corpus(&repo).await;
    let fixture = Fixture::new(&repo, CandidatePolicy::default()).await;
    for name in [
        "reader",
        "proposer",
        "acceptor",
        "capturer",
        "jobs",
        "everything-but-garden",
    ] {
        assert!(
            !fixture.names(name).await.iter().any(|tool| tool == TOOL),
            "{name} must not discover the scan"
        );
        assert_eq!(
            error(&fixture.tool(name, TOOL, scan(&hub, true)).await),
            "forbidden",
            "{name}"
        );
    }
    assert_eq!(fixture.names("gardener").await, vec![TOOL.to_owned()]);
    for (tool, args) in [
        ("list_proposals", json!({"status":"pending","limit":10})),
        (
            "get_record",
            json!({"id":hub,"revision":null,"neighbors":0}),
        ),
        (
            "decide_proposal",
            json!({"request_id":"garden-accept","id":"proposed_edge:fixture","revision":"a".repeat(64),"action":"accept","confirmed":true}),
        ),
    ] {
        assert_eq!(
            error(&fixture.tool("gardener", tool, args).await),
            "forbidden",
            "garden must not grant {tool}"
        );
    }
    assert!(proposals(&repo).await.is_empty());
    fixture.stop().await;
}

#[tokio::test]
async fn receipts_echo_resolved_limits_and_policy_and_dry_runs_change_nothing() {
    let repo = Repository::new(init_memory().await.unwrap());
    let hub = corpus(&repo).await;
    let policy = CandidatePolicy {
        similarity_floor: 0.9,
        max_per_endpoint: 3,
        ..CandidatePolicy::default()
    };
    let fixture = Fixture::new(&repo, policy).await;

    for (mutate, field) in [
        (
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("max_candidates");
            }) as Box<dyn Fn(&mut Value)>,
            "missing max_candidates",
        ),
        (
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("dry_run");
            }),
            "missing dry_run",
        ),
        (
            Box::new(|v: &mut Value| {
                v.as_object_mut().unwrap().remove("deadline_ms");
            }),
            "missing deadline_ms",
        ),
        (
            Box::new(|v: &mut Value| v["max_candidates"] = json!(51)),
            "max_candidates over budget",
        ),
        (
            Box::new(|v: &mut Value| v["deadline_ms"] = json!(60001)),
            "deadline over budget",
        ),
        (
            Box::new(|v: &mut Value| v["accept"] = json!(true)),
            "unknown field",
        ),
        (
            Box::new(|v: &mut Value| v["scope"]["ids"] = json!(["hub"])),
            "malformed id",
        ),
        (
            Box::new(|v: &mut Value| v["scope"] = json!({"kind":"everything"})),
            "unknown scope",
        ),
    ] {
        let mut args = scan(&hub, true);
        mutate(&mut args);
        assert_eq!(
            error(&fixture.tool("gardener", TOOL, args).await),
            "invalid_input",
            "{field}"
        );
    }

    let mut request = scan(&hub, true);
    request["min_similarity"] = json!(0.5);
    request["max_per_endpoint"] = json!(2);
    let preview = fixture.tool("gardener", TOOL, request).await;
    let receipt = data(&preview);
    assert_eq!(receipt["contract"], 1);
    assert_eq!(receipt["actor"], "mcp:gardener");
    assert_eq!(receipt["dry_run"], true);
    assert_eq!(
        receipt["limits"],
        json!({"max_candidates":5,"max_endpoints":10,"deadline_ms":20000})
    );
    // A request can tighten (max_per_endpoint 3 → 2) but never loosen
    // (min_similarity 0.5 < floor 0.9) the operator's policy.
    assert_eq!(
        receipt["policy"]["similarity_floor"].as_f64().unwrap() as f32,
        0.9
    );
    assert_eq!(receipt["policy"]["max_per_endpoint"], 2);
    assert_eq!(receipt["policy"]["generator"], "gardener-candidate");
    assert_eq!(receipt["policy"]["relationship"], "related_to");
    assert_eq!(receipt["scope"], json!({"kind":"notes","ids":[hub]}));
    assert_eq!(receipt["stop_reason"], "completed");
    assert_eq!(receipt["endpoints_examined"], 1);
    assert_eq!(receipt["scope_exhausted"], true);
    assert_eq!(receipt["next_after_note_id"], Value::Null);
    assert_eq!(receipt["nominations"], 3);
    assert_eq!(receipt["rejections"]["endpoint_cap"], 1);
    assert_eq!(receipt["proposals_created"], 0);
    let candidates = receipt["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2);
    for candidate in candidates {
        assert_eq!(candidate["relationship"], "related_to");
        assert_eq!(candidate["signals"], json!(["shared_specific_entity"]));
        assert_eq!(candidate["proposal"], Value::Null);
        assert_ne!(candidate["from"]["source_id"], candidate["to"]["source_id"]);
    }
    assert!(proposals(&repo).await.is_empty());
    assert_eq!(accepted_edges(&repo).await, 0);

    let persisted = fixture.tool("gardener", TOOL, scan(&hub, false)).await;
    let receipt = data(&persisted);
    assert_eq!(receipt["dry_run"], false);
    assert_eq!(receipt["proposals_created"], 3);
    for candidate in receipt["candidates"].as_array().unwrap() {
        assert_eq!(candidate["proposal"]["status"], "pending");
        assert!(candidate["proposal"]["id"]
            .as_str()
            .unwrap()
            .starts_with("proposed_edge:"));
    }
    let stored = proposals(&repo).await;
    assert_eq!(stored.len(), 3);
    assert!(stored.iter().all(|proposal| proposal["status"] == "pending"
        && proposal["generator"] == "gardener-candidate"
        && proposal["edge_type"] == "related_to"));
    assert_eq!(accepted_edges(&repo).await, 0);

    // Re-running never refreshes the pending proposals; it abstains.
    let rerun = fixture.tool("gardener", TOOL, scan(&hub, false)).await;
    let receipt = data(&rerun);
    assert_eq!(receipt["proposals_created"], 0);
    assert_eq!(receipt["rejections"]["already_proposed"], 3);
    assert_eq!(receipt["candidates"], json!([]));
    assert_eq!(proposals(&repo).await, stored);
    assert_eq!(accepted_edges(&repo).await, 0);
    fixture.stop().await;
}
