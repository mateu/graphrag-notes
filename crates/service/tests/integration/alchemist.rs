//! Opt-in Alchemist synthesis over MCP: capability gating, receipts and the
//! no-write guarantee, using fictional notes and a fixture provider only.
use graphrag_agents::alchemist::{
    CitationPolicy, FixtureSynthesisGenerator, SynthesisBudgets, SynthesisGenerator,
};
use graphrag_agents::{DeterministicEmbedder, Embedder, LibrarianRuntimeConfig, SearchAgent};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, EdgeType, Note};
use graphrag_db::{init_memory, DbConnection, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const PRINCIPALS: [(&str, &[Capability]); 6] = [
    ("synthesist", &[Capability::Read, Capability::Alchemist]),
    ("alchemist-only", &[Capability::Alchemist]),
    ("reader", &[Capability::Read]),
    (
        "reviewer",
        &[
            Capability::Read,
            Capability::Propose,
            Capability::Accept,
            Capability::Reject,
            Capability::Undo,
        ],
    ),
    (
        "writer",
        &[
            Capability::Read,
            Capability::Capture,
            Capability::Edit,
            Capability::Delete,
            Capability::Upload,
            Capability::Jobs,
        ],
    ),
    ("enricher", &[Capability::Read, Capability::Enrich]),
];

fn token(name: &str) -> String {
    format!("{name}-synthetic-alchemist-token-at-least-thirty-two-bytes")
}

struct Fixture {
    _temp: tempfile::TempDir,
    url: String,
    client: reqwest::Client,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}

impl Fixture {
    async fn new(application: Arc<dyn RemoteApplicationOperations>) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("credentials.json");
        let credentials = CredentialFile {
            schema_version: 1,
            credentials: PRINCIPALS
                .iter()
                .map(|(name, capabilities)| Credential {
                    instance_id: (*name).into(),
                    token_sha256: format!("{:x}", Sha256::digest(token(name).as_bytes())),
                    capabilities: capabilities.to_vec(),
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
                .timeout(Duration::from_secs(10))
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

    async fn tools(&self, name: &str) -> Vec<String> {
        self.rpc(name, "tools/list", json!({})).await["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap().to_string())
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

fn data(v: &Value) -> &Value {
    &v["result"]["structuredContent"]["data"]
}

fn error(v: &Value) -> &Value {
    &v["result"]["structuredContent"]["error"]["code"]
}

struct Corpus {
    db: DbConnection,
    repo: Repository,
    seed: String,
    support: String,
}

async fn corpus() -> Corpus {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let seed = repo
        .create_note(Note::new(
            "Fictional log: the greenhouse vents opened at 24 degrees on Tuesday.",
        ))
        .await
        .unwrap();
    let support = repo
        .create_note(Note::new(
            "Fictional sensor export: vent actuator logged an opening at 24 degrees.",
        ))
        .await
        .unwrap();
    repo.create_edge(
        support.id.as_ref().unwrap(),
        seed.id.as_ref().unwrap(),
        EdgeType::Supports,
        None,
    )
    .await
    .unwrap();
    Corpus {
        db,
        repo,
        seed: record_id_to_string(seed.id.as_ref().unwrap()),
        support: record_id_to_string(support.id.as_ref().unwrap()),
    }
}

fn application(repo: &Repository) -> EmbeddedApplication {
    let embedder: Arc<dyn Embedder> = Arc::new(DeterministicEmbedder::default());
    EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        Arc::new(graphrag_agents::FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
}

fn runtime(generator: impl SynthesisGenerator + 'static) -> AlchemistRuntime {
    AlchemistRuntime {
        ceilings: SynthesisBudgets {
            max_chunks: 4,
            deadline_ms: 5_000,
            ..SynthesisBudgets::default()
        },
        citation_policy: CitationPolicy::Drop,
        generator: Arc::new(generator),
    }
}

fn draft(corpus: &Corpus) -> String {
    json!({
        "claims": [
            {"text": "The vents opened at 24 degrees.", "citations": [
                {"chunk": corpus.seed, "quote": "the greenhouse vents opened at 24 degrees"},
                {"chunk": corpus.support, "quote": "vent actuator logged an opening at 24 degrees"}
            ]},
            {"text": "A fabricated claim.", "citations": [{"chunk": "C1", "quote": "the vents never opened at all"}]}
        ],
        "conflicts": [],
        "open_questions": [],
        "proposals": [{"from": corpus.seed, "to": corpus.support, "type": "supports"}]
    })
    .to_string()
}

async fn snapshot(db: &DbConnection) -> BTreeMap<String, String> {
    let info = db
        .query("INFO FOR DB")
        .await
        .unwrap()
        .take::<Option<Value>>(0)
        .unwrap()
        .unwrap();
    let mut tables = BTreeMap::new();
    for table in info["tables"].as_object().unwrap().keys() {
        let rows: Vec<Value> = db
            .query(format!("SELECT * FROM {table} ORDER BY id"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        tables.insert(table.clone(), serde_json::to_string(&rows).unwrap());
    }
    tables
}

#[tokio::test]
async fn only_read_plus_alchemist_may_list_or_call_and_alchemist_grants_nothing_else() {
    let corpus = corpus().await;
    let generator = FixtureSynthesisGenerator::fixed(draft(&corpus));
    let fixture = Fixture::new(Arc::new(
        application(&corpus.repo).with_alchemist(runtime(generator.clone())),
    ))
    .await;
    let args = json!({"seed": corpus.seed});
    for (name, _) in PRINCIPALS.iter().skip(1) {
        assert!(
            !fixture
                .tools(name)
                .await
                .contains(&"synthesize_notes".into()),
            "{name}"
        );
        assert_eq!(
            error(&fixture.tool(name, "synthesize_notes", args.clone()).await),
            "forbidden",
            "{name}"
        );
    }
    // The alchemist capability alone grants no tool at all.
    assert!(fixture.tools("alchemist-only").await.is_empty());
    for (tool, arguments) in [
        (
            "search_notes",
            json!({"query": "vents", "mode": "keyword", "graph": "off"}),
        ),
        ("get_record", json!({"id": corpus.seed})),
        ("build_context", json!({"query": "vents"})),
        ("list_proposals", json!({})),
    ] {
        assert_eq!(
            error(&fixture.tool("alchemist-only", tool, arguments).await),
            "forbidden",
            "{tool}"
        );
    }
    assert!(
        generator.requests().is_empty(),
        "denied calls reach no provider"
    );
    let tools = fixture.tools("synthesist").await;
    assert!(tools.contains(&"synthesize_notes".into()));
    assert!(tools.contains(&"search_notes".into()));
    fixture.stop().await;
}

#[tokio::test]
async fn synthesis_over_mcp_is_verified_receipted_and_writes_nothing() {
    let corpus = corpus().await;
    let before = snapshot(&corpus.db).await;
    let fixture = Fixture::new(Arc::new(
        application(&corpus.repo)
            .with_alchemist(runtime(FixtureSynthesisGenerator::fixed(draft(&corpus)))),
    ))
    .await;
    let result = fixture
        .tool(
            "synthesist",
            "synthesize_notes",
            json!({"seed": corpus.seed, "max_chunks": 3, "max_hops": 1, "deadline_ms": 4000}),
        )
        .await;
    let response: SynthesisResponse = serde_json::from_value(data(&result).clone()).unwrap();
    let claims = &response.synthesis.claims;
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].citations.len(), 2);
    assert_eq!(claims[0].strength.independent_origins, 2);
    let receipt = &response.receipt;
    assert_eq!(
        (receipt.mode.as_str(), receipt.seed.as_deref()),
        ("seed", Some(corpus.seed.as_str()))
    );
    assert_eq!(
        (
            receipt.budgets.max_chunks,
            receipt.budgets.max_hops,
            receipt.budgets.deadline_ms
        ),
        (3, 1, 4000)
    );
    assert_eq!(
        (receipt.ceilings.max_chunks, receipt.ceilings.deadline_ms),
        (4, 5000)
    );
    assert_eq!(receipt.policy.citation_policy, CitationPolicy::Drop);
    assert_eq!(
        (receipt.policy.writes.as_str(), receipt.policy.persisted),
        ("none", false)
    );
    assert!(!receipt.policy.feeds_retrieval);
    assert_eq!(receipt.verification.claims_dropped, 1);
    assert_eq!(receipt.verification.rejections["quote_not_found"], 1);
    assert!(response
        .synthesis
        .relations
        .iter()
        .all(|relation| relation.evidence == "unassessed"));

    for invalid in [
        json!({}),
        json!({"query": "vents", "seed": corpus.seed}),
        json!({"seed": corpus.seed, "max_chunks": 5}),
        json!({"seed": corpus.seed, "max_hops": 3}),
        json!({"seed": corpus.seed, "write": true}),
    ] {
        assert_eq!(
            error(
                &fixture
                    .tool("synthesist", "synthesize_notes", invalid)
                    .await
            ),
            "invalid_input"
        );
    }
    assert_eq!(
        error(
            &fixture
                .tool(
                    "synthesist",
                    "synthesize_notes",
                    json!({"seed": "note:missing"})
                )
                .await
        ),
        "not_found"
    );
    assert_eq!(
        error(
            &fixture
                .tool(
                    "synthesist",
                    "synthesize_notes",
                    json!({"seed": corpus.seed, "citation_policy": "strict"})
                )
                .await
        ),
        "unverifiable_synthesis"
    );
    fixture.stop().await;
    assert_eq!(snapshot(&corpus.db).await, before);
}

#[tokio::test]
async fn disabled_by_default_and_deadline_bounded() {
    let corpus = corpus().await;
    let disabled = Fixture::new(Arc::new(application(&corpus.repo))).await;
    assert_eq!(
        error(
            &disabled
                .tool(
                    "synthesist",
                    "synthesize_notes",
                    json!({"seed": corpus.seed})
                )
                .await
        ),
        "alchemist_disabled"
    );
    disabled.stop().await;

    let slow = FixtureSynthesisGenerator::fixed(draft(&corpus)).with_delay(Duration::from_secs(5));
    let fixture = Fixture::new(Arc::new(
        application(&corpus.repo).with_alchemist(runtime(slow.clone())),
    ))
    .await;
    let started = std::time::Instant::now();
    // Seed retrieval runs inside the same deadline before the provider call.
    // Leave it room on a slow runner, so the deadline expires during the 5 s
    // generation that the request count below proves.
    let result = fixture
        .tool(
            "synthesist",
            "synthesize_notes",
            json!({"seed": corpus.seed, "deadline_ms": 1_000}),
        )
        .await;
    assert_eq!(error(&result), "deadline_exceeded");
    assert!(data(&result).is_null());
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(slow.requests().len(), 1);
    fixture.stop().await;
}
