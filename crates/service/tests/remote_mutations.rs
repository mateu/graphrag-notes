use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, EntityExtraction, EntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::*;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::Notify};
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
async fn seeded() -> (Repository, String) {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo
        .create_note(Note::new("Atlas manual note"))
        .await
        .unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    (repo, id)
}

#[tokio::test]
async fn independent_capabilities_confirmations_trusted_actor_and_durable_replay() {
    let (repo, id) = seeded().await;
    let fixture = Fixture::new(application(&repo, Arc::new(NoInference))).await;
    for (name, expected) in [
        ("reader", 7),
        ("editor", 1),
        ("deleter", 1),
        ("acceptor", 1),
        ("rejector", 1),
        ("undoer", 1),
        ("capturer", 1),
    ] {
        let list = fixture.rpc(name, "tools/list", json!({})).await;
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), expected);
    }
    let before = fixture.tool("reader", "get_note", json!({"id":id})).await;
    let revision = data(&before)["revision"].as_str().unwrap();
    let edit = json!({"request_id":"edit-1","id":id,"revision":revision,"patch":{"title":"Reviewed","tags":["shared"]}});
    for name in [
        "reader", "capturer", "deleter", "acceptor", "rejector", "undoer",
    ] {
        assert_eq!(
            error(&fixture.tool(name, "edit_note", edit.clone()).await),
            "forbidden"
        );
    }
    let edited = fixture.tool("editor", "edit_note", edit.clone()).await;
    assert_eq!(data(&edited)["outcome"]["actor"], "mcp:editor");
    assert_eq!(data(&edited)["replayed"], false);
    let replay = fixture.tool("editor", "edit_note", edit.clone()).await;
    assert_eq!(data(&replay)["replayed"], true);
    assert_eq!(data(&replay)["outcome"], data(&edited)["outcome"]);
    let changed =
        json!({"request_id":"edit-1","id":id,"revision":revision,"patch":{"title":"Different"}});
    assert_eq!(
        error(&fixture.tool("editor", "edit_note", changed).await),
        "revision_conflict"
    );
    let after = fixture.tool("reader", "get_note", json!({"id":id})).await;
    let deletion = json!({"request_id":"delete-1","id":id,"revision":data(&after)["revision"],"confirmed":false});
    assert_eq!(
        error(
            &fixture
                .tool("deleter", "delete_note", deletion.clone())
                .await
        ),
        "invalid_input"
    );
    let mut confirmed = deletion;
    confirmed["confirmed"] = json!(true);
    let removed = fixture
        .tool("deleter", "delete_note", confirmed.clone())
        .await;
    assert_eq!(data(&removed)["outcome"]["cascade"]["notes"], 1);
    assert_eq!(
        error(&fixture.tool("reader", "get_note", json!({"id":id})).await),
        "not_found"
    );
    assert_eq!(
        data(&fixture.tool("deleter", "delete_note", confirmed).await)["replayed"],
        true
    );
    assert_eq!(
        data(&fixture.tool("editor", "edit_note", edit).await)["replayed"],
        true
    );
    fixture.stop().await;
}

#[tokio::test]
async fn proposal_actions_are_separate_capabilities_and_cannot_spoof_reviewer() {
    let (repo, id) = seeded().await;
    let a = repo.get_note(&id).await.unwrap().unwrap();
    let b = repo.create_note(Note::new("second note")).await.unwrap();
    let proposal = repo
        .upsert_gardener_proposal(
            a.id.as_ref().unwrap(),
            b.id.as_ref().unwrap(),
            0.8,
            "evidence".into(),
            None,
            None,
        )
        .await
        .unwrap();
    let pid = record_id_to_string(proposal.id.as_ref().unwrap());
    let fixture = Fixture::new(application(&repo, Arc::new(NoInference))).await;
    let card = fixture
        .tool("reader", "get_proposal", json!({"id":pid}))
        .await;
    let decision = json!({"request_id":"accept-1","id":pid,"revision":data(&card)["revision"],"action":"accept","reason":"Human reviewed","confirmed":true});
    for name in ["rejector", "undoer", "editor", "reader", "capturer"] {
        assert_eq!(
            error(
                &fixture
                    .tool(name, "decide_proposal", decision.clone())
                    .await
            ),
            "forbidden"
        );
    }
    let mut spoof = decision.clone();
    spoof["reviewer"] = json!("corpus-owner");
    assert_eq!(
        error(&fixture.tool("acceptor", "decide_proposal", spoof).await),
        "invalid_input"
    );
    let mut unconfirmed = decision.clone();
    unconfirmed["confirmed"] = json!(false);
    assert_eq!(
        error(
            &fixture
                .tool("acceptor", "decide_proposal", unconfirmed)
                .await
        ),
        "invalid_input"
    );
    let accepted = fixture
        .tool("acceptor", "decide_proposal", decision.clone())
        .await;
    assert_eq!(data(&accepted)["outcome"]["status"], "accepted");
    let current = fixture
        .tool("reader", "get_proposal", json!({"id":pid}))
        .await;
    assert_eq!(data(&current)["reviewer"], "mcp:acceptor");
    let undo = json!({"request_id":"undo-1","id":pid,"revision":data(&current)["revision"],"action":"undo","reason":"New evidence changes the decision","confirmed":true});
    assert_eq!(
        error(
            &fixture
                .tool("acceptor", "decide_proposal", undo.clone())
                .await
        ),
        "forbidden"
    );
    assert_eq!(
        data(
            &fixture
                .tool("undoer", "decide_proposal", undo.clone())
                .await
        )["outcome"]["status"],
        "superseded"
    );
    let undone = fixture
        .tool("reader", "get_proposal", json!({"id":pid}))
        .await;
    assert_eq!(
        data(&undone)["supersession_reason"],
        "New evidence changes the decision"
    );
    assert_eq!(data(&undone)["action_reason"], "Human reviewed");
    assert_eq!(data(&undone)["reviewer"], "mcp:acceptor");
    assert_eq!(
        data(&fixture.tool("undoer", "decide_proposal", undo).await)["replayed"],
        true
    );
    assert_eq!(
        data(&fixture.tool("acceptor", "decide_proposal", decision).await)["replayed"],
        true
    );
    fixture.stop().await;
}

struct SlowEmbedder {
    started: Notify,
    release: Notify,
}
#[async_trait]
impl Embedder for SlowEmbedder {
    async fn embed(&self, text: &str, truncate: bool) -> graphrag_agents::Result<Vec<f32>> {
        self.started.notify_one();
        self.release.notified().await;
        DeterministicEmbedder::default().embed(text, truncate).await
    }
    async fn embed_batch(
        &self,
        text: &[String],
        truncate: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        self.started.notify_one();
        self.release.notified().await;
        DeterministicEmbedder::default()
            .embed_batch(text, truncate)
            .await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}
#[tokio::test]
async fn disconnected_edit_drains_before_shutdown_and_replays_without_another_provider_call() {
    let (repo, id) = seeded().await;
    let slow = Arc::new(SlowEmbedder {
        started: Notify::new(),
        release: Notify::new(),
    });
    let fixture = Fixture::new(application(&repo, slow.clone())).await;
    let before = fixture.tool("reader", "get_note", json!({"id":id})).await;
    let args = json!({"request_id":"disconnected-edit","id":id,"revision":data(&before)["revision"],"patch":{"content":"Atlas newly prepared body"}});
    let request = fixture.request(
        "editor",
        "tools/call",
        json!({"name":"edit_note","arguments":args.clone()}),
    );
    let waiter = tokio::spawn(async move { request.send().await });
    tokio::time::timeout(Duration::from_secs(5), slow.started.notified())
        .await
        .unwrap();
    waiter.abort();
    fixture.shutdown.cancel();
    tokio::task::yield_now().await;
    assert!(!fixture.task.is_finished());
    slow.release.notify_one();
    fixture.stop().await;
    assert_eq!(
        repo.get_note(&id).await.unwrap().unwrap().content,
        "Atlas newly prepared body"
    );
    let request: RemoteEditRequest = serde_json::from_value(args).unwrap();
    let replay = application(&repo, Arc::new(NoInference))
        .edit_remote(
            CallerIdentity {
                instance_id: "editor".into(),
            },
            request,
            ActionCancellation::new(),
        )
        .await
        .unwrap();
    assert!(replay.replayed);
}
