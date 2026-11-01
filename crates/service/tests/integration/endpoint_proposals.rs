use async_trait::async_trait;
use graphrag_agents::{
    Embedder, EntityExtraction, EntityExtractor, InferenceCapabilities, LibrarianRuntimeConfig,
    SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

struct NoInference;
#[async_trait]
impl Embedder for NoInference {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("unexpected embedding")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("unexpected embedding batch")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        panic!("unexpected provider identity")
    }
}
#[async_trait]
impl EntityExtractor for NoInference {
    async fn extract(&self, _: &str) -> graphrag_agents::Result<EntityExtraction> {
        panic!("unexpected extraction")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        panic!("unexpected extractor health")
    }
    fn capabilities(&self) -> InferenceCapabilities {
        graphrag_agents::FixtureEntityExtractor::default().capabilities()
    }
}
fn application(repo: &Repository, embedder: Arc<dyn Embedder>) -> Arc<EmbeddedApplication> {
    Arc::new(EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        Arc::new(NoInference),
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    ))
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
    async fn new(application: Arc<dyn RemoteApplicationOperations>) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("credentials.json");
        let credentials = CredentialFile {
            schema_version: 1,
            credentials: [
                ("reader", Capability::Read),
                ("editor", Capability::Edit),
                ("deleter", Capability::Delete),
                ("acceptor", Capability::Accept),
                ("rejector", Capability::Reject),
                ("undoer", Capability::Undo),
                ("capturer", Capability::Capture),
                ("proposer", Capability::Propose),
                ("uploader", Capability::Upload),
                ("jobs", Capability::Jobs),
            ]
            .into_iter()
            .map(|(name, cap)| Credential {
                instance_id: name.into(),
                token_sha256: format!("{:x}", Sha256::digest(token(name).as_bytes())),
                capabilities: vec![cap],
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
    fn request(&self, name: &str, method: &str, params: Value) -> reqwest::RequestBuilder {
        self.client
            .post(&self.url)
            .bearer_auth(token(name))
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2025-11-25")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}))
    }
    async fn rpc(&self, name: &str, method: &str, params: Value) -> Value {
        let response = self.request(name, method, params).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        response.json().await.unwrap()
    }
    async fn tool(&self, name: &str, tool: &str, args: Value) -> Value {
        self.rpc(name, "tools/call", json!({"name":tool,"arguments":args}))
            .await
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

#[tokio::test]
async fn exactly_scoped_pending_only_capability_is_not_automatic_acceptance() {
    let repo = Repository::new(init_memory().await.unwrap());
    let a = repo
        .create_note(Note::new("Explicit schedule claim"))
        .await
        .unwrap();
    let b = repo
        .create_note(Note::new("Verified execution evidence"))
        .await
        .unwrap();
    let a = record_id_to_string(a.id.as_ref().unwrap());
    let b = record_id_to_string(b.id.as_ref().unwrap());
    let ra = repo.inspect_record(&a, 0).await.unwrap();
    let rb = repo.inspect_record(&b, 0).await.unwrap();
    let fixture = Fixture::new(application(&repo, Arc::new(NoInference))).await;
    let payload = json!({"request_id":"reviewed-proposal","from":{"id":a,"revision":ra.revision,"quote":ra.content},"to":{"id":b,"revision":rb.revision,"quote":rb.content},"relationship":"supports","rationale":"Second quotation supplies directly reviewed execution evidence for the first claim.","confirmed":true});
    for name in [
        "reader", "editor", "deleter", "acceptor", "rejector", "undoer", "capturer", "uploader",
        "jobs",
    ] {
        let list = fixture.rpc(name, "tools/list", json!({})).await;
        assert!(!list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "propose_endpoint_relationship"));
        assert_eq!(
            error(
                &fixture
                    .tool(name, "propose_endpoint_relationship", payload.clone())
                    .await
            ),
            "forbidden"
        );
    }
    let list = fixture.rpc("proposer", "tools/list", json!({})).await;
    let tools = list["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "propose_endpoint_relationship");
    let mut injected = payload.clone();
    injected["accept"] = true.into();
    assert_eq!(
        error(
            &fixture
                .tool("proposer", "propose_endpoint_relationship", injected)
                .await
        ),
        "invalid_input"
    );
    let mut causal = payload.clone();
    causal["relationship"] = "depends_on".into();
    assert_eq!(
        error(
            &fixture
                .tool("proposer", "propose_endpoint_relationship", causal)
                .await
        ),
        "invalid_input"
    );
    let result = fixture
        .tool("proposer", "propose_endpoint_relationship", payload.clone())
        .await;
    assert_eq!(data(&result)["outcome"]["actor"], "mcp:proposer");
    assert_eq!(data(&result)["outcome"]["status"], "pending");
    let replay = fixture
        .tool("proposer", "propose_endpoint_relationship", payload)
        .await;
    assert_eq!(data(&replay)["replayed"], true);
    assert_eq!(
        repo.portable_records_page("proposed_edge", 0, 100)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(repo
        .portable_records_page("supports", 0, 100)
        .await
        .unwrap()
        .is_empty());
    let denied=fixture.tool("proposer","decide_proposal",json!({"request_id":"forbidden-accept","id":data(&result)["outcome"]["id"],"revision":"a".repeat(64),"action":"accept","confirmed":true})).await;
    assert_eq!(error(&denied), "forbidden");
    fixture.stop().await;
}
